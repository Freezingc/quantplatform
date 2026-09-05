"""Fail-closed verification of materialized ``factor_daily`` year regions.

Used to gate the 2014-2019 backfill that the approved F01-F06 folds require
(their train windows start in 2014 while the original materialization covered
2020-2025 only). For each requested year this asserts:

* exactly ``FROZEN_FACTOR_COUNT`` distinct ``factor_id`` values are present;
* a single ``data_version``, equal to the value predicted from the materializer's
  window convention ``sha256(sha256(scope,start,end) + start + end)`` -- verified
  to reproduce all six registered 2020-2025 hashes bit-for-bit;
* zero duplicate business keys ``(factor_id, signal_date, ts_code)``;
* per-factor and total row counts within a tight tolerance of a read-only dry-run
  prediction, when one is supplied.

Row counts are tolerant rather than exact on purpose. The materializer keeps a row
only when ``np.isfinite(value)`` holds, and for GP-style factors (e.g.
``yeslab_406`` / GP_037, ``replay_lookback_days=180``) a near-zero denominator can
flip a marginal cell between finite and infinite depending on BLAS reduction
order, i.e. on host load. Observed once: 2014 ``yeslab_406`` stored 195,378 rows
against 195,377 in an earlier dry-run, while the other 16 factors matched exactly
and the stored values were all finite with no nulls. Everything that must be exact
(version, factor set, duplicate keys) stays exact; any within-tolerance delta is
printed, never swallowed.

Usage:
  verify_factor_daily_backfill.py --years 2014..2019
  verify_factor_daily_backfill.py --years 2014 --expectations 2014=/tmp/mat2014_dryrun.log
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

FROZEN_FACTOR_COUNT = 17
SCOPE = "FROZEN_FACTOR_DAILY_V1"
OBJECT_NAME = "factor_daily"
ROW_LINE = re.compile(r"^(yeslab_\S+): (\d+) universe-aligned rows$")
YEAR_LINE = re.compile(r"^(\d{4})$")


def predicted_version(year: int) -> str:
    """Recompute the data_version the materializer derives from a one-year window.

    ``materialize_frozen_factor_daily.py`` hashes the window itself, so the version
    a year *will* have is knowable before it is written. That makes post-hoc
    verification an equality check instead of an inspection.
    """
    start, end = f"{year}-01-01", f"{year}-12-31"
    watermark = hashlib.sha256(
        json.dumps({"scope": SCOPE, "start": start, "end": end}, sort_keys=True).encode()
    ).hexdigest()
    return hashlib.sha256((watermark + start + end).encode()).hexdigest()


def parse_dryrun_expectations(text: str) -> dict[str, int]:
    """Extract ``{factor_id: universe-aligned rows}`` from a materializer dry-run log."""
    expectations: dict[str, int] = {}
    for line in text.splitlines():
        match = ROW_LINE.match(line.strip())
        if match:
            expectations[match.group(1)] = int(match.group(2))
    return expectations


def row_tolerance(expected: int) -> int:
    """Absolute row-count slack: 4 rows, or 0.002% for large factors."""
    return max(4, int(expected * 0.00002))


def parse_year_range(spec: str) -> list[int]:
    """Expand ``2014..2019`` (inclusive) or a single year into a year list."""
    start, separator, end = spec.partition("..")
    bounds = (start, end) if separator else (start, start)
    for bound in bounds:
        if not YEAR_LINE.fullmatch(bound):
            raise ValueError(f"invalid year bound: {bound}")
    first, last = int(bounds[0]), int(bounds[1])
    if last < first:
        raise ValueError(f"year range end {last} precedes start {first}")
    return list(range(first, last + 1))


@dataclass(frozen=True)
class YearAssessment:
    year: int
    passed: bool
    failures: tuple[str, ...]
    notes: tuple[str, ...]
    total_rows: int
    factor_count: int
    data_version: str | None
    duplicates: int


def assess_year(
    year: int,
    *,
    versions: Mapping[str, int],
    factor_counts: Mapping[str, int],
    duplicates: int,
    expectations: Mapping[str, int] | None = None,
) -> YearAssessment:
    """Pure verdict for one year: exact version/factor/duplicate gates, tolerant counts.

    ``versions`` maps each observed ``data_version`` to its row count, so one mapping
    carries both the version-set check and the per-version volume; ``factor_counts``
    maps ``factor_id`` to rows and drives the factor-count and tolerance checks.
    """
    failures: list[str] = []
    notes: list[str] = []

    observed_versions = sorted(versions)
    expected_version = predicted_version(year)
    if observed_versions != [expected_version]:
        failures.append(
            f"data_version set {[v[:16] for v in observed_versions]} != predicted "
            f"{expected_version[:16]}"
        )

    factor_count = len(factor_counts)
    if factor_count != FROZEN_FACTOR_COUNT:
        failures.append(f"factor count {factor_count} != {FROZEN_FACTOR_COUNT}")

    if duplicates != 0:
        failures.append(f"duplicate business keys {duplicates} != 0")

    total = sum(factor_counts.values())
    if expectations:
        expected_total = sum(expectations.values())
        if abs(total - expected_total) > row_tolerance(expected_total):
            failures.append(
                f"total rows {total} outside +/-{row_tolerance(expected_total)} of "
                f"predicted {expected_total}"
            )
        for factor, expected in sorted(expectations.items()):
            observed = factor_counts.get(factor)
            if observed is None or abs(observed - expected) > row_tolerance(expected):
                failures.append(
                    f"{factor}: {observed} rows outside +/-{row_tolerance(expected)} "
                    f"of predicted {expected}"
                )
            elif observed != expected:
                notes.append(f"{factor}: {observed} vs predicted {expected} (within tolerance)")
    elif total <= 0:
        failures.append("no dry-run expectation supplied and total rows is 0")

    return YearAssessment(
        year=year,
        passed=not failures,
        failures=tuple(failures),
        notes=tuple(notes),
        total_rows=total,
        factor_count=factor_count,
        data_version=observed_versions[0] if len(observed_versions) == 1 else None,
        duplicates=duplicates,
    )


def collect_year_state(
    holder: dict[str, Any], year: int
) -> tuple[dict[str, int], dict[str, int], int]:
    """Read one year's observed state: versions->rows, factor->rows, duplicate groups."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from walk_forward_common import TARGET_DATABASE, query

    table = f"loadTable('{TARGET_DATABASE}','{OBJECT_NAME}')"
    bounds = f"signal_date >= {year}.01.01 and signal_date < {year + 1}.01.01"

    ok, frame = query(
        holder, f"select factor_id, data_version, count(*) as n from {table} "
                f"where {bounds} group by factor_id, data_version"
    )
    versions: dict[str, int] = {}
    counts: dict[str, int] = {}
    if ok and isinstance(frame, pd.DataFrame) and not frame.empty:
        # Column access (not itertuples) keeps the element type as Any, which is what
        # the untyped dolphindb driver actually returns.
        for factor_id, data_version, count in zip(
            frame["factor_id"], frame["data_version"], frame["n"], strict=True
        ):
            version_key = str(data_version)
            versions[version_key] = versions.get(version_key, 0) + int(count)
            factor_key = str(factor_id)
            counts[factor_key] = counts.get(factor_key, 0) + int(count)

    ok_d, dup = query(
        holder,
        f"select count(*) as n from (select factor_id, signal_date, ts_code from {table} "
        f"where {bounds} group by factor_id, signal_date, ts_code having count(*) > 1)",
    )
    duplicates = (
        int(dup.iloc[0]["n"]) if ok_d and isinstance(dup, pd.DataFrame) and len(dup) else -1
    )
    return versions, counts, duplicates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--years", required=True, help="e.g. 2014..2019 or 2014")
    parser.add_argument(
        "--expectations",
        action="append",
        default=[],
        help="YEAR=PATH to a materializer dry-run log (repeatable)",
    )
    args = parser.parse_args(argv)
    years = parse_year_range(args.years)

    expectations: dict[int, dict[str, int]] = {}
    for item in args.expectations:
        year_text, separator, path = item.partition("=")
        if not separator:
            raise SystemExit(f"--expectations must look like 2014=/path/to/log, got {item}")
        expectations[int(year_text)] = parse_dryrun_expectations(Path(path).read_text())

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from walk_forward_common import new_holder

    holder = new_holder()
    failed = False
    for year in years:
        versions, counts, duplicates = collect_year_state(holder, year)
        assessment = assess_year(
            year,
            versions=versions,
            factor_counts=counts,
            duplicates=duplicates,
            expectations=expectations.get(year),
        )
        version_text = assessment.data_version[:16] if assessment.data_version else "NONE/MIXED"
        print(
            f"YEAR {year}: factors={assessment.factor_count} "
            f"total_rows={assessment.total_rows:,} version={version_text} "
            f"duplicates={assessment.duplicates} "
            f"expectations={'yes' if year in expectations else 'none'}"
        )
        for note in assessment.notes:
            print(f"  NOTE {note}")
        for failure in assessment.failures:
            print(f"  FAIL {failure}")
        print(f"YEAR {year}: {'PASS' if assessment.passed else 'FAIL'}")
        failed |= not assessment.passed
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
