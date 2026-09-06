"""Census duplicate business keys in the YesLab factor sources, read-only.

C3.4 of the factor-governance plan asks for a regression fixture drawn from the real
GP_017 / GP_077 duplicate-key problem, and no such audit exists in this repository: the
thirteen files mentioning those two factors were each inspected and none of them counts
duplicate business keys (the one GP_077 diagnostic is about scalar degeneracy, which is
the C7 clustering domain). This script is the measurement that fixture is built from --
the duplicate-group shape histogram of one factor at a time, over its whole range.

Two properties are built in deliberately.

Aggregates only, never rows. Every query ends in a single-row ``select`` over a per-key
aggregate, so a 27M-row factor costs one row of traffic, and this script cannot be turned
into a bulk source extract by accident. It also issues one factor per query: batching five
factors of ``factor_data8`` into one query exhausted the server's memory, while a
per-factor query against the same table does not.

It does not trust ``nunique``. DolphinDB counts NULL as one distinct value --
``nunique([1.0, NULL, 2.0, 1.0]) == 3``, measured -- so the obvious predicate
``nunique(value) > 1`` reports a key whose duplicate rows are all NULL, and a key holding
one value plus one NULL, as a genuine value conflict. On GP_017 that naive predicate
overstates conflicts by more than a factor of two. The corrected count subtracts the NULL
bucket whenever the group holds any unusable row.

"Unusable" is exactly ``isValid(value)``, and that equivalence is measured rather than
assumed. A DolphinDB DOUBLE column cannot hold an infinity at all: uploading ``inf``,
``-inf`` or ``1e309`` reads back as NULL, and every arithmetic route to one (``1/0``,
``log(0)``, ``exp(1000)``, ``pow(10.0, 400)``, ``1e308*10``) also yields NULL. On top of
that ``abs(NULL) <= 1e308`` evaluates to **True**, so a ``abs(value) <= 1e308`` guard can
never exclude a row -- it would only look like an infinity filter while filtering nothing.
The live ``value`` column is DOUBLE on all of factor_data1..11, so NULL and infinity are
the same absence here, and C2.3's "invalid rather than finite" treatment of +/-inf applies
to whatever a producer *intended*, which arrives as NULL.

``--scope self-test`` checks the correction against hand-computed expectations on a
synthetic in-memory table before any source is read. ``--scope crossval`` re-measures
GP_017 and GP_077 and requires the recorded numbers to be reproduced exactly, which is
what makes the frozen-set census comparable with them.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import dolphindb as ddb
import pandas as pd

from research_v2.db.session import YesLabReadOnlySession

_SAFE_TABLE = re.compile(r"^[A-Za-z0-9_]+$")
_UNSAFE_NAME = re.compile(r"[;\n\r]")
_SELF_TEST_TABLE = "census_self_test_shape"
_METADATA_QUERY = (
    "select factor_id,factor_name,store_factor from loadTable('dfs://YesLab','factor_metadata')"
)

# Distinct usable values in one (ts_code, trade_date) group. ``nunique`` still counts the
# NULL bucket that the substitution introduces, so subtract one exactly when the group
# holds at least one row that was not usable.
_CORRECTED_DISTINCT = "dv_usable - iif(n - usable_n > 0, 1, 0)"

_METRIC_EXPRESSIONS: dict[str, str] = {
    "key_count": "count(*)",
    "row_count": "sum(n)",
    "duplicated_key_count": "sum(iif(n > 1, 1, 0))",
    "max_multiplicity": "max(n)",
    "conflict_key_count": f"sum(iif({_CORRECTED_DISTINCT} > 1, 1, 0))",
    "conflict_key_count_naive": "sum(iif(dv_raw > 1, 1, 0))",
    # Equality inside a function call must be ``==``: DolphinDB parses a bare ``=`` there
    # as a keyword-argument separator, so ``iif(x = 0, 1, 0)`` is a syntax error.
    "no_usable_value_key_count": f"sum(iif({_CORRECTED_DISTINCT} == 0, 1, 0))",
    "duplicated_no_usable_value_key_count": f"sum(iif(n > 1 && {_CORRECTED_DISTINCT} == 0, 1, 0))",
    "unique_key_count": "sum(iif(n == 1, 1, 0))",
    "usable_row_count": "sum(usable_n)",
    # The six cells below partition the key space, and each one is exactly what the
    # deduplicator does with a business key: publish it as UNIQUE, quarantine it as an
    # invalid unique, collapse identical rows, keep the one non-null value, quarantine a
    # conflict, or quarantine a group with no usable value at all. Counting them is what
    # lets a fixture be built to the measured shape instead of to a guess.
    "unique_usable_key_count": "sum(iif(n == 1 && usable_n == 1, 1, 0))",
    "unique_unusable_key_count": "sum(iif(n == 1 && usable_n == 0, 1, 0))",
    "duplicated_identical_key_count": (
        f"sum(iif(n > 1 && {_CORRECTED_DISTINCT} == 1 && usable_n == n, 1, 0))"
    ),
    "duplicated_single_nonnull_key_count": (
        f"sum(iif(n > 1 && {_CORRECTED_DISTINCT} == 1 && usable_n < n, 1, 0))"
    ),
    "duplicated_conflict_key_count": f"sum(iif({_CORRECTED_DISTINCT} > 1, 1, 0))",
    # Within-group shape, so a fixture can be built to the measured pattern instead of to
    # a guess about it. The class counts above say how many keys fall in each class; these
    # say what a group in that class actually looks like -- how many of its rows carry a
    # usable value, and for a conflict, how many distinct values it holds. Both are sums
    # over keys, so the average is the sum divided by the class count, and both stay
    # additive when a census falls back to one query per year.
    "single_nonnull_usable_row_count": (
        f"sum(iif(n > 1 && {_CORRECTED_DISTINCT} == 1 && usable_n < n, usable_n, 0))"
    ),
    "conflict_usable_row_count": f"sum(iif({_CORRECTED_DISTINCT} > 1, usable_n, 0))",
    "conflict_distinct_value_total": (
        f"sum(iif({_CORRECTED_DISTINCT} > 1, {_CORRECTED_DISTINCT}, 0))"
    ),
}
METRICS: tuple[str, ...] = tuple(_METRIC_EXPRESSIONS)

PARTITION_CELLS: tuple[str, ...] = (
    "unique_usable_key_count",
    "unique_unusable_key_count",
    "duplicated_identical_key_count",
    "duplicated_single_nonnull_key_count",
    "duplicated_conflict_key_count",
    "duplicated_no_usable_value_key_count",
)


def check_partition(metrics: Mapping[str, int]) -> tuple[str, ...]:
    """Internal consistency of one census, checked against itself rather than trusted.

    A census whose six cells do not add up to its own key count is measuring something
    other than what it claims, and any fixture built from it would inherit the error
    silently -- the failure mode this project keeps finding in its own gates.
    """
    violations: list[str] = []
    cells = sum(metrics[name] for name in PARTITION_CELLS)
    if cells != metrics["key_count"]:
        violations.append(f"the six cells sum to {cells}, not to key_count {metrics['key_count']}")
    uniques = metrics["unique_usable_key_count"] + metrics["unique_unusable_key_count"]
    if uniques != metrics["unique_key_count"]:
        violations.append(
            f"the two unique cells sum to {uniques}, "
            f"not to unique_key_count {metrics['unique_key_count']}"
        )
    duplicated = sum(metrics[name] for name in PARTITION_CELLS[2:])
    if duplicated != metrics["duplicated_key_count"]:
        violations.append(
            f"the four duplicated cells sum to {duplicated}, "
            f"not to duplicated_key_count {metrics['duplicated_key_count']}"
        )
    if metrics["duplicated_conflict_key_count"] != metrics["conflict_key_count"]:
        violations.append(
            "a conflict needs two distinct values, so every conflicting key is a duplicated "
            "key and the two conflict counts must agree"
        )
    # The naive predicate over-counts by exactly one class. ``nunique`` adds the NULL bucket
    # once, so it reports a second distinct value for a group holding one value plus a NULL,
    # and only for such a group: an all-NULL group still has nunique 1, as does a group of
    # identical values and any single-row key. Verified on all 18 censused factors, where it
    # holds with zero violations -- which is also the large-scale confirmation that the NULL
    # model the correction rests on is the one DolphinDB actually implements.
    expected_naive = (
        metrics["duplicated_conflict_key_count"] + metrics["duplicated_single_nonnull_key_count"]
    )
    if metrics["conflict_key_count_naive"] != expected_naive:
        violations.append(
            f"conflict_key_count_naive {metrics['conflict_key_count_naive']} is not the "
            f"{expected_naive} keys that hold a conflict or a value plus a NULL"
        )
    return tuple(violations)


# Hand-computed from _self_test_frame(): A identical x3, B genuine conflict, C duplicate
# NULLs, D one finite row, E two rows a producer meant as +inf/-inf (a DOUBLE column
# stores neither, so both arrive as NULL: unusable, and NOT two distinct values), F one
# finite value plus one NULL (the trap that inflates the naive count), G identical on a
# second day, H a single NULL row (unusable but not a duplicated key).
#
# The naive count is 2, not 3, precisely because E cannot contribute: the infinities do
# not survive the upload, so ``nunique`` sees one distinct value there. The naive error is
# therefore a NULL-only error, which is what the correction has to undo and nothing more.
SELF_TEST_EXPECTED: dict[str, int] = {
    "key_count": 8,
    "row_count": 15,
    "duplicated_key_count": 6,
    "max_multiplicity": 3,
    "conflict_key_count": 1,
    "conflict_key_count_naive": 2,
    "no_usable_value_key_count": 3,
    "duplicated_no_usable_value_key_count": 2,
    "unique_key_count": 2,
    "usable_row_count": 9,
    # D one usable row; H one NULL row; A and G identical rows; F one value plus one NULL;
    # B the only genuine conflict; C and E duplicated with nothing usable in them.
    "unique_usable_key_count": 1,
    "unique_unusable_key_count": 1,
    "duplicated_identical_key_count": 2,
    "duplicated_single_nonnull_key_count": 1,
    "duplicated_conflict_key_count": 1,
    # F is the only single-non-null group and one of its two rows is usable; B is the only
    # conflict, both of its rows are usable, and it holds two distinct values.
    "single_nonnull_usable_row_count": 1,
    "conflict_usable_row_count": 2,
    "conflict_distinct_value_total": 2,
}

# Recorded 2026-09-06 in qrv2_research_artifacts/gp017_gp077_duplicate_census.json
# (sha256 b99a1aceca9e0db0c48984ad0a5389cee375fa148aa84bd238f3c1562aa5cf67). The
# "no usable value" figure was recorded under one name only, so the cross-validation
# reports which of the two denominators reproduces it instead of assuming one.
PRIOR_MEASUREMENT: dict[str, dict[str, int]] = {
    "GP_017": {
        "key_count": 9457875,
        "row_count": 27361491,
        "duplicated_key_count": 8951808,
        "max_multiplicity": 3,
        "conflict_key_count": 3184961,
        "no_usable_value_key_count": 2072174,
    },
    "GP_077": {
        "key_count": 9665247,
        "row_count": 18627522,
        "duplicated_key_count": 8962275,
        "max_multiplicity": 2,
        "conflict_key_count": 0,
        "no_usable_value_key_count": 2701899,
    },
}
PRIOR_FACTORS: dict[str, str] = {"GP_017": "yeslab_386", "GP_077": "yeslab_446"}


def build_script(source_clause: str) -> str:
    """Return the full census query for one factor, ending in a single aggregate row."""
    metrics = ", ".join(f"{expr} as {name}" for name, expr in _METRIC_EXPRESSIONS.items())
    return (
        f"raw = select ts_code, trade_date, value from {source_clause}\n"
        "tagged = select ts_code as tc, trade_date as td, value as rv, "
        "iif(isValid(value), value, double(NULL)) as fv from raw\n"
        "agg = select count(*) as n, nunique(fv) as dv_usable, nunique(rv) as dv_raw, "
        "count(fv) as usable_n from tagged group by tc, td\n"
        f"select {metrics} from agg"
    )


def _scalar(value: object) -> int:
    if hasattr(value, "item"):
        return int(value.item())  # type: ignore[union-attr]
    if isinstance(value, (list, tuple)):
        return int(value[0])
    return int(value)  # type: ignore[arg-type]


def read_metrics(result: object) -> dict[str, int]:
    """Read one aggregate row, however the DolphinDB client chose to return it."""
    if isinstance(result, pd.DataFrame):
        if result.empty:
            raise RuntimeError("census query returned no row")
        row = result.iloc[0]
        values = [row.iloc[position] for position in range(len(METRICS))]
    elif isinstance(result, dict):
        values = [result[name] for name in METRICS]
    elif isinstance(result, (list, tuple)):
        if len(result) != len(METRICS):
            raise RuntimeError(f"census query returned {len(result)} values, want {len(METRICS)}")
        values = list(result)
    else:
        raise RuntimeError(f"unexpected census result type: {type(result).__name__}")
    return dict(zip(METRICS, (_scalar(value) for value in values), strict=True))


def _combine(parts: list[dict[str, int]]) -> dict[str, int]:
    """Sum per-year censuses; only the maximum multiplicity is not additive.

    A year range partitions the (ts_code, trade_date) key space, so every count adds and
    only ``max_multiplicity`` must be maximised instead.
    """
    combined = {name: sum(part[name] for part in parts) for name in METRICS}
    combined["max_multiplicity"] = max(part["max_multiplicity"] for part in parts)
    return combined


def census_factor(
    source: YesLabReadOnlySession,
    *,
    table: str,
    factor_name: str,
    fallback_years: tuple[int, ...] = tuple(range(2010, 2027)),
) -> tuple[dict[str, int], dict[str, Any]]:
    """Census one factor, falling back to one query per year if the server refuses."""
    if not _SAFE_TABLE.fullmatch(table):
        raise ValueError(f"unsafe source table: {table}")
    if _UNSAFE_NAME.search(factor_name):
        raise ValueError(f"unsafe factor name: {factor_name!r}")
    escaped = factor_name.replace("'", "''")
    base = f"loadTable('dfs://YesLab','{table}') where factor_name = '{escaped}'"
    evidence: dict[str, Any] = {"queries": [], "chunked_by_year": False}
    try:
        script = build_script(base)
        evidence["queries"].append(script)
        measured = read_metrics(source.query(script))
        evidence["partition_violations"] = list(check_partition(measured))
        return measured, evidence
    except RuntimeError as exc:
        evidence["whole_range_error"] = f"{type(exc).__name__}: {exc}"
    parts: list[dict[str, int]] = []
    for year in fallback_years:
        script = build_script(
            f"{base} and trade_date between date('{year}.01.01') and date('{year}.12.31')"
        )
        evidence["queries"].append(script)
        part = read_metrics(source.query(script))
        if part["key_count"]:
            parts.append(part)
    if not parts:
        raise RuntimeError(f"census found no rows for {table}/{factor_name}")
    evidence["chunked_by_year"] = True
    combined = _combine(parts)
    evidence["partition_violations"] = list(check_partition(combined))
    return combined, evidence


def _self_test_frame() -> pd.DataFrame:
    day_one = date(2020, 1, 2)
    day_two = date(2020, 1, 3)
    rows: list[tuple[str, date, float]] = [
        ("A", day_one, 1.0),
        ("A", day_one, 1.0),
        ("A", day_one, 1.0),
        ("B", day_one, 1.0),
        ("B", day_one, 2.0),
        ("C", day_one, float("nan")),
        ("C", day_one, float("nan")),
        ("D", day_one, 3.0),
        # A producer's +inf and -inf. Neither survives a DOUBLE column: both read back
        # NULL, so this case pins "intended infinity is absence", not "infinity is a
        # value that must be filtered out" -- there is nothing to filter.
        ("E", day_one, float("inf")),
        ("E", day_one, float("-inf")),
        ("F", day_one, 4.0),
        ("F", day_one, float("nan")),
        ("G", day_two, 5.0),
        ("G", day_two, 5.0),
        ("H", day_two, float("nan")),
    ]
    return pd.DataFrame(
        {
            "ts_code": [row[0] for row in rows],
            "trade_date": [row[1] for row in rows],
            "value": [row[2] for row in rows],
        }
    )


def run_self_test(client: ddb.session, source: YesLabReadOnlySession) -> dict[str, Any]:
    """Prove the corrected predicate on synthetic shapes, with no source read at all.

    Uploading a session-local variable creates no database object: nothing is written to
    any DFS database, and the query still passes through the read-only session. pandas
    cannot hold NULL and NaN apart in a float column and the client maps NaN to a
    DolphinDB NULL double, which is what the source stores anyway.
    """
    frame = _self_test_frame()
    client.upload({_SELF_TEST_TABLE: frame})
    script = build_script(_SELF_TEST_TABLE)
    measured = read_metrics(source.query(script))
    differences = {
        name: {"measured": measured[name], "expected": SELF_TEST_EXPECTED[name]}
        for name in METRICS
        if measured[name] != SELF_TEST_EXPECTED[name]
    }
    return {
        "status": "PASS" if not differences else "FAIL",
        "measured": measured,
        "expected": SELF_TEST_EXPECTED,
        "differences": differences,
        "partition_violations": list(check_partition(measured)),
        "script": script,
        "frame_rows": len(frame),
    }


def _source_map(source: YesLabReadOnlySession, factor_ids: list[str]) -> dict[str, dict[str, str]]:
    metadata = source.query(_METADATA_QUERY)
    if not isinstance(metadata, pd.DataFrame):
        raise RuntimeError("factor_metadata did not return a DataFrame")
    keyed = metadata.assign(
        factor_key=metadata["factor_id"].map(lambda value: f"yeslab_{int(value)}")
    ).set_index("factor_key")
    return {
        factor_id: {
            "factor_name": str(keyed.loc[factor_id, "factor_name"]),
            "source_table": str(keyed.loc[factor_id, "store_factor"]),
        }
        for factor_id in factor_ids
        if factor_id in keyed.index
    }


def run_crossval(source: YesLabReadOnlySession) -> dict[str, Any]:
    """Re-measure GP_017 and GP_077 and require the recorded figures to be reproduced."""
    mapping = _source_map(source, sorted(PRIOR_FACTORS.values()))
    results: dict[str, Any] = {}
    for label, factor_id in sorted(PRIOR_FACTORS.items()):
        entry = mapping[factor_id]
        measured, evidence = census_factor(
            source, table=entry["source_table"], factor_name=entry["factor_name"]
        )
        prior = PRIOR_MEASUREMENT[label]
        reproduced = {
            name: measured[name] == prior[name] for name in prior if name in measured
        }
        recorded_no_usable = prior["no_usable_value_key_count"]
        if measured["duplicated_no_usable_value_key_count"] == recorded_no_usable:
            denominator = "duplicated_keys_only"
        elif measured["no_usable_value_key_count"] == recorded_no_usable:
            denominator = "all_keys"
        else:
            denominator = "neither"
        reproduced["no_usable_value_key_count"] = denominator != "neither"
        results[label] = {
            "factor_id": factor_id,
            "source_table": entry["source_table"],
            "factor_name": entry["factor_name"],
            "measured": measured,
            "prior": prior,
            "reproduced": reproduced,
            "no_usable_value_denominator": denominator,
            "queries_issued": len(evidence["queries"]),
            "chunked_by_year": evidence["chunked_by_year"],
        }
    reproduced_everything = all(
        all(entry["reproduced"].values()) for entry in results.values()
    )
    return {
        "status": "PASS" if reproduced_everything else "FAIL",
        "factors": results,
    }


def run_frozen(source: YesLabReadOnlySession, contract_path: Path) -> dict[str, Any]:
    """Census every frozen-set factor that the source catalog can resolve."""
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    factor_ids = [str(item) for item in contract["factors"]]
    mapping = _source_map(source, factor_ids)
    factors: dict[str, Any] = {}
    for factor_id in factor_ids:
        if factor_id not in mapping:
            factors[factor_id] = {
                "status": "NOT_CENSUSED",
                "reason": "no factor_metadata row: repaired variant, not a raw source factor",
            }
            continue
        entry = mapping[factor_id]
        try:
            measured, evidence = census_factor(
                source, table=entry["source_table"], factor_name=entry["factor_name"]
            )
        except RuntimeError as exc:
            factors[factor_id] = {
                "status": "FAILED",
                "reason": f"{type(exc).__name__}: {exc}",
                "factor_name": entry["factor_name"],
                "source_table": entry["source_table"],
            }
            continue
        factors[factor_id] = {
            "status": "CENSUSED",
            "factor_name": entry["factor_name"],
            "source_table": entry["source_table"],
            "metrics": measured,
            "queries_issued": len(evidence["queries"]),
            "chunked_by_year": evidence["chunked_by_year"],
        }
        print(
            f"{factor_id} {entry['factor_name']} keys={measured['key_count']} "
            f"dup={measured['duplicated_key_count']} conflict={measured['conflict_key_count']} "
            f"naive={measured['conflict_key_count_naive']}",
            flush=True,
        )
    return {"contract": str(contract_path), "factors": factors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--scope", choices=("self-test", "crossval", "frozen", "all"), default="self-test"
    )
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--frozen-contract",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "artifacts"
        / "diagnostics"
        / "factor_set_frozen_v1.json",
    )
    parser.add_argument("--as-of", type=date.fromisoformat, default=date(2026, 9, 6))
    args = parser.parse_args()

    client = ddb.session()
    client.connect(
        os.getenv("QRV2_DDB_HOST", "10.176.55.211"),
        int(os.getenv("QRV2_DDB_PORT", "8848")),
        os.getenv("QRV2_DDB_USERNAME", "xiahongjie"),
        os.getenv("QRV2_DDB_PASSWORD", "123456"),
    )
    source = YesLabReadOnlySession(client)
    report: dict[str, Any] = {
        "as_of_date": args.as_of.isoformat(),
        "scope": args.scope,
        "metric_definitions": {
            "key": "(ts_code, trade_date) within one factor",
            "usable": "isValid(value); a DOUBLE column cannot hold an infinity (measured)",
            "conflict_key_count": "keys holding two or more distinct usable values",
            "conflict_key_count_naive": "keys where nunique(value) > 1, NULL and inf counted",
            "duplicated_no_usable_value_key_count": "duplicated keys with no usable value",
            "no_usable_value_key_count": "all keys with no usable value, duplicated or not",
        },
    }
    try:
        if args.scope in {"self-test", "all"}:
            report["self_test"] = run_self_test(client, source)
            print(f"self-test: {report['self_test']['status']}", flush=True)
        if args.scope in {"crossval", "all"}:
            report["crossval"] = run_crossval(source)
            print(f"crossval: {report['crossval']['status']}", flush=True)
        if args.scope in {"frozen", "all"}:
            report["frozen"] = run_frozen(source, args.frozen_contract)
    finally:
        client.close()

    document = json.dumps(report, ensure_ascii=False, indent=1, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(document + "\n", encoding="utf-8")
        print(f"written: {args.output}", flush=True)
    else:
        print(document)


if __name__ == "__main__":
    main()
