"""Measure how a factor's unusable values are spread across days, read-only.

裁决 4 leans on ``yeslab_484`` being the dense control. The project's coverage gate gives it
``median_cov`` 1.0018 over 2020-2024, while a census of the same source over the same window
finds **21.0% of its keys hold no usable value**. The two numbers use different denominators --
the gate divides by universe members per day and takes the median over days, the census divides
by source keys and pools the whole window -- but 21 points is too wide to be explained by that
alone, and the two candidate explanations have different consequences:

* if the NULLs sit on **whole days**, the median daily share stays near 1 while the pooled share
  falls. A median-of-daily-ratios gate cannot see that, and the baseline's blanket ``dropna``
  would delete those days entire. This is the same shape as the crash-day collapse Task 3.1b
  already recorded, where 11 lost days inside a five-year train segment is a 99.1% valid share.
* if instead **every day is uniformly degraded**, median and pooled agree and the gap is a
  denominator artifact, which leaves the dense control standing.

So this measures the one quantity that separates them: the distribution across days of each
factor's usable share. ``median_daily_share`` against ``pooled_share`` is the discriminating
pair, and the two day counts say whether whole days are empty.

Aggregates only, one factor per query, bounded to an explicit window -- the same three
properties the census it follows is built on. ``--scope self-test`` checks the query against
hand-computed expectations on the census's synthetic frame before any source is read.
"""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping, Sequence
from datetime import date
from pathlib import Path
from typing import Any

import dolphindb as ddb
import pandas as pd
from census_duplicate_key_shapes import _SAFE_TABLE, _UNSAFE_NAME, _self_test_frame

from research_v2.db.session import YesLabReadOnlySession

_SELF_TEST_TABLE = "daily_share_self_test_shape"
_METADATA_QUERY = (
    "select factor_id,factor_name,store_factor from loadTable('dfs://YesLab','factor_metadata')"
)

# Equality inside a function call has to be ``==``: DolphinDB reads a bare ``=`` there as a
# keyword-argument separator. Same trap the census hit.
_METRIC_EXPRESSIONS: dict[str, str] = {
    "day_count": "count(*)",
    "days_zero_usable": "sum(iif(usable == 0, 1, 0))",
    "days_below_half": "sum(iif(usable * 2 < n, 1, 0))",
    "min_daily_share": "min(double(usable) / n)",
    "median_daily_share": "median(double(usable) / n)",
    "usable_row_count": "sum(usable)",
    "row_count": "sum(n)",
}
METRICS: tuple[str, ...] = tuple(_METRIC_EXPRESSIONS)

# Hand-computed from the census's synthetic frame. Day 2020-01-02 holds twelve rows of which
# seven are usable (three identical, two conflicting, one lone finite, one finite plus a NULL,
# and two groups with nothing usable), so its share is 7/12; day 2020-01-03 holds three rows of
# which two are usable, so 2/3. Neither day is empty and neither is below half.
SELF_TEST_EXPECTED: dict[str, float] = {
    "day_count": 2,
    "days_zero_usable": 0,
    "days_below_half": 0,
    "min_daily_share": 7 / 12,
    "median_daily_share": (7 / 12 + 2 / 3) / 2,
    "usable_row_count": 9,
    "row_count": 15,
}


def build_script(source_clause: str) -> str:
    """Return the whole measurement for one factor, ending in a single aggregate row."""
    metrics = ", ".join(f"{expr} as {name}" for name, expr in _METRIC_EXPRESSIONS.items())
    return (
        f"raw = select trade_date, value from {source_clause}\n"
        # Only aggregates in the projection: selecting the group key beside them makes this
        # engine complain that the columns have different lengths.
        "daily = select count(*) as n, "
        "count(iif(isValid(value), 1, NULL)) as usable from raw group by trade_date\n"
        f"select {metrics} from daily"
    )


def read_metrics(result: object) -> dict[str, float]:
    """Read one aggregate row, however the DolphinDB client chose to return it."""
    if isinstance(result, pd.DataFrame):
        if result.empty:
            raise RuntimeError("daily-share query returned no row")
        row = result.iloc[0]
        values = [row.iloc[position] for position in range(len(METRICS))]
    elif isinstance(result, dict):
        values = [result[name] for name in METRICS]
    elif isinstance(result, (list, tuple)):
        if len(result) != len(METRICS):
            raise RuntimeError(f"daily-share query returned {len(result)} values")
        values = list(result)
    else:
        raise RuntimeError(f"unexpected result type: {type(result).__name__}")
    out: dict[str, float] = {}
    for name, value in zip(METRICS, values, strict=True):
        scalar = value.item() if hasattr(value, "item") else value
        out[name] = float(scalar)
    out["pooled_share"] = (
        out["usable_row_count"] / out["row_count"] if out["row_count"] else 0.0
    )
    return out


def measure_factor(
    source: YesLabReadOnlySession,
    *,
    table: str,
    factor_name: str,
    start: date,
    end: date,
) -> tuple[dict[str, float], list[str]]:
    """Measure one factor over one window. The window is mandatory: an unbounded run would
    silently mix the screening window with years nobody is arguing about."""
    if not _SAFE_TABLE.fullmatch(table):
        raise ValueError(f"unsafe source table: {table}")
    if _UNSAFE_NAME.search(factor_name):
        raise ValueError(f"unsafe factor name: {factor_name!r}")
    escaped = factor_name.replace("'", "''")
    clause = (
        f"loadTable('dfs://YesLab','{table}') where factor_name = '{escaped}' "
        f"and trade_date between date('{start:%Y.%m.%d}') and date('{end:%Y.%m.%d}')"
    )
    script = build_script(clause)
    return read_metrics(source.query(script)), [script]


def run_self_test(client: ddb.session, source: YesLabReadOnlySession) -> dict[str, Any]:
    """Check the query against hand-computed expectations, with no source read at all.

    Uploading a session-local variable creates no database object: nothing is written to any
    DFS database, and the query still passes through the read-only session.
    """
    client.upload({_SELF_TEST_TABLE: _self_test_frame()})
    measured = read_metrics(source.query(build_script(_SELF_TEST_TABLE)))
    differences = {
        name: {"measured": measured[name], "expected": expected}
        for name, expected in SELF_TEST_EXPECTED.items()
        if abs(measured[name] - expected) > 1e-9
    }
    return {
        "status": "PASS" if not differences else "FAIL",
        "measured": measured,
        "expected": SELF_TEST_EXPECTED,
        "differences": differences,
        "script": build_script(_SELF_TEST_TABLE),
    }


def _source_map(
    source: YesLabReadOnlySession, factor_ids: Sequence[str]
) -> dict[str, dict[str, str]]:
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


def classify(metrics: Mapping[str, float], *, gate_median_cov: float | None = None) -> str:
    """Read the numbers into one of two verdicts, with the counts that qualify it.

    The discriminator is whether the per-day median and the pooled share agree -- not whether
    any day is empty. An earlier version of this rule keyed on the day counts instead and
    mislabelled the first real data it saw: GP_003 has 879 of 1216 days below half usable, but
    its median (0.361) and pooled share (0.380) agree, so its sparsity is chronic and even, and
    saying "a median over days reads better than the window" was simply false for it. Empty-day
    counts matter, but as detail that qualifies a verdict, never as the verdict.

    If median and pooled agree, the unusable values are spread evenly across every day, so
    nothing about the day distribution can explain a gap against a gate that takes a median of
    daily ratios -- the gap has to be in the denominator. If they disagree, the median day reads
    better than the window, which is what a minority of bad days does, and a
    median-of-daily-ratios gate cannot see it while a blanket ``dropna`` deletes those days.
    """
    median = metrics["median_daily_share"]
    pooled = metrics["pooled_share"]
    gap = median - pooled
    detail = (
        f"median {median:.3f} vs pooled {pooled:.3f}, worst day "
        f"{metrics['min_daily_share']:.3f}, {int(metrics['days_zero_usable'])} empty days, "
        f"{int(metrics['days_below_half'])} of {int(metrics['day_count'])} days below half"
    )
    if abs(gap) <= 0.02:
        verdict = (
            "UNIFORM_ACROSS_DAYS: the median day and the whole window agree, so the unusable "
            f"values are spread evenly rather than concentrated in a few days ({detail}). A gap "
            "against a median-of-daily-ratios gate is therefore a denominator gap, not a "
            "day-level hole"
        )
    else:
        verdict = (
            "DAY_LEVEL_SKEW: the median day reads better than the window, so the unusable "
            f"values are concentrated in a minority of days ({detail}). A gate that takes a "
            "median of daily ratios reads the median day and cannot see this, and a blanket "
            "dropna would delete the bad days entire"
        )
    if gate_median_cov:
        verdict += (
            f"; gate median_cov {gate_median_cov:.4f}, this median / gate = "
            f"{median / gate_median_cov:.3f}"
        )
    return verdict


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scope", choices=("self-test", "measure", "all"), default="self-test")
    parser.add_argument("--factors", default="")
    parser.add_argument("--start", type=date.fromisoformat, default=date(2020, 1, 1))
    parser.add_argument("--end", type=date.fromisoformat, default=date(2024, 12, 31))
    parser.add_argument("--output", type=Path, default=None)
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
        "window": {"start": args.start.isoformat(), "end": args.end.isoformat()},
        "metric_definitions": {
            "median_daily_share": "median over days of (usable rows / rows present that day)",
            "pooled_share": "usable rows over the whole window / rows over the whole window",
            "days_zero_usable": "days on which no key at all holds a usable value",
            "days_below_half": "days on which fewer than half the present keys are usable",
        },
    }
    try:
        if args.scope in {"self-test", "all"}:
            report["self_test"] = run_self_test(client, source)
            print(f"self-test: {report['self_test']['status']}", flush=True)
        if args.scope in {"measure", "all"}:
            selected = [item.strip() for item in args.factors.split(",") if item.strip()]
            if not selected:
                raise SystemExit("--scope measure needs --factors")
            mapping = _source_map(source, selected)
            factors: dict[str, Any] = {}
            for factor_id in selected:
                if factor_id not in mapping:
                    factors[factor_id] = {
                        "status": "NOT_MEASURED",
                        "reason": "no factor_metadata row",
                    }
                    continue
                entry = mapping[factor_id]
                metrics, _ = measure_factor(
                    source,
                    table=entry["source_table"],
                    factor_name=entry["factor_name"],
                    start=args.start,
                    end=args.end,
                )
                factors[factor_id] = {
                    "status": "MEASURED",
                    "factor_name": entry["factor_name"],
                    "source_table": entry["source_table"],
                    "metrics": metrics,
                    "verdict": classify(metrics),
                }
                print(
                    f"{factor_id} {entry['factor_name']} "
                    f"median={metrics['median_daily_share']:.3f} "
                    f"pooled={metrics['pooled_share']:.3f} "
                    f"zero_days={int(metrics['days_zero_usable'])} "
                    f"below_half={int(metrics['days_below_half'])}",
                    flush=True,
                )
                print(f"   {factors[factor_id]['verdict']}", flush=True)
            report["factors"] = factors
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
