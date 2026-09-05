"""Per-factor structural coverage screen on factor_daily (C plan C6.2, correct source).

The screen that admitted the frozen set ran against the YesLab source tables, while the
frozen chain consumes the replayed ``factor_daily``. That mismatch produced both a false
negative and two false positives: ``yeslab_375`` was never structurally screened at all
yet measures 43.9-62.8% coverage, below the C6.2 gate of 0.70; and ``yeslab_398`` /
``yeslab_464`` were failed for MISSING_MONTH_OR_LOW_MONTHLY_COVERAGE on the source
(1159 and 1185 valid days of 1212) although ``factor_daily`` carries 1212/1212 days and
60/60 months for both.

This script re-runs the C6.2 metric set on the source that is actually consumed, so the
numbers behind any factor-set decision come from the right table. Per C6.3 the coverage
denominator is same-day ``universe_daily`` members with ``is_member=1``, never "stocks
that happen to appear in the factor table".

Metrics per factor: expected/valid day counts and day_ratio, minimum monthly day ratio,
median / p05 / minimum per-day coverage ratio, duplicate business keys, and a verdict
using the original reason-code vocabulary so results are comparable with the historical
screen.

Usage:
  screen_factor_daily_coverage.py --start 2014-01-01 --end 2019-12-31
  screen_factor_daily_coverage.py --start 2020-01-01 --end 2024-12-31 --json <path>
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

COVERAGE_GATE = 0.70          # C6.2: train scope coverage < 70% -> ELIGIBILITY_FAIL
MEDIAN_GATE = 0.70            # C6.2 LOW_MEDIAN_SAMPLE_COVERAGE
P05_GATE = 0.30               # C6.2 LOW_P05_SAMPLE_COVERAGE
MONTHLY_DAY_RATIO_GATE = 0.90  # C6.2 MISSING_MONTH_OR_LOW_MONTHLY_COVERAGE


@dataclass(frozen=True)
class FactorCoverage:
    factor_id: str
    expected_day_count: int
    valid_day_count: int
    day_ratio: float
    minimum_monthly_day_ratio: float
    median_coverage_ratio: float
    p05_coverage_ratio: float
    minimum_coverage_ratio: float
    duplicate_row_count: int
    coverage_gate_pass: bool
    eligibility_status: str
    reason_codes: tuple[str, ...]


def _quantile(sorted_values: list[float], q: float) -> float:
    """Nearest-rank quantile on an already-sorted list; empty input gives 0.0."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    index = int(round(q * (len(sorted_values) - 1)))
    return sorted_values[max(0, min(len(sorted_values) - 1, index))]


def screen_factor(
    factor_id: str,
    per_day_rows: dict[date, int],
    universe_per_day: dict[date, int],
    *,
    duplicate_row_count: int = 0,
) -> FactorCoverage:
    """Pure C6.2 evaluation for one factor from per-day row counts.

    ``universe_per_day`` is the denominator and also defines the expected trading days,
    so a day the universe itself does not cover is not counted against the factor.
    """
    expected_days = sorted(universe_per_day)
    expected_day_count = len(expected_days)
    ratios: list[float] = []
    valid_days = 0
    for day in expected_days:
        universe_rows = universe_per_day[day]
        rows = per_day_rows.get(day, 0)
        if rows > 0:
            valid_days += 1
        if universe_rows > 0:
            ratios.append(rows / universe_rows)
    ratios_sorted = sorted(ratios)

    per_month_present: dict[tuple[int, int], int] = defaultdict(int)
    per_month_expected: dict[tuple[int, int], int] = defaultdict(int)
    for day in expected_days:
        key = (day.year, day.month)
        per_month_expected[key] += 1
        if per_day_rows.get(day, 0) > 0:
            per_month_present[key] += 1
    monthly_ratios = [
        per_month_present[key] / per_month_expected[key] for key in per_month_expected
    ]

    median = _quantile(ratios_sorted, 0.50)
    p05 = _quantile(ratios_sorted, 0.05)
    minimum = ratios_sorted[0] if ratios_sorted else 0.0
    day_ratio = valid_days / expected_day_count if expected_day_count else 0.0
    min_monthly = min(monthly_ratios) if monthly_ratios else 0.0

    reasons: list[str] = []
    if duplicate_row_count:
        reasons.append("DUPLICATE_BUSINESS_KEYS")
    if min_monthly < MONTHLY_DAY_RATIO_GATE:
        reasons.append("MISSING_MONTH_OR_LOW_MONTHLY_COVERAGE")
    if median < MEDIAN_GATE:
        reasons.append("LOW_MEDIAN_SAMPLE_COVERAGE")
    if p05 < P05_GATE:
        reasons.append("LOW_P05_SAMPLE_COVERAGE")
    gate_pass = median >= COVERAGE_GATE and not reasons
    if median < COVERAGE_GATE and "LOW_MEDIAN_SAMPLE_COVERAGE" not in reasons:
        reasons.append("COVERAGE_GATE_FAIL")
    if gate_pass:
        reasons = ["COVERAGE_GATE_PASS"]

    return FactorCoverage(
        factor_id=factor_id,
        expected_day_count=expected_day_count,
        valid_day_count=valid_days,
        day_ratio=day_ratio,
        minimum_monthly_day_ratio=min_monthly,
        median_coverage_ratio=median,
        p05_coverage_ratio=p05,
        minimum_coverage_ratio=minimum,
        duplicate_row_count=duplicate_row_count,
        coverage_gate_pass=gate_pass,
        eligibility_status="PASS" if gate_pass else "ELIGIBILITY_FAIL",
        reason_codes=tuple(reasons),
    )


def collect(
    holder: dict[str, Any], start: date, end: date, factor_ids: tuple[str, ...]
) -> tuple[dict[date, int], dict[str, dict[date, int]], dict[str, int]]:
    """Three queries for the whole window: universe, per-factor per-day, duplicates."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import pandas as pd
    from walk_forward_common import TARGET_DATABASE, query

    bounds = f"signal_date >= {start:%Y.%m.%d} and signal_date <= {end:%Y.%m.%d}"
    fd = f"loadTable('{TARGET_DATABASE}','factor_daily')"

    ok_u, uni = query(
        holder,
        f"select signal_date, count(*) as n from loadTable('{TARGET_DATABASE}','universe_daily') "
        f"where {bounds} and is_member=1 group by signal_date",
    )
    universe_per_day: dict[date, int] = {}
    if ok_u and isinstance(uni, pd.DataFrame):
        for day, count in zip(uni["signal_date"], uni["n"], strict=True):
            universe_per_day[pd.Timestamp(day).date()] = int(count)

    ok_f, per = query(
        holder,
        f"select signal_date, factor_id, count(*) as n from {fd} where {bounds} "
        f"group by signal_date, factor_id",
    )
    per_factor: dict[str, dict[date, int]] = {fid: {} for fid in factor_ids}
    if ok_f and isinstance(per, pd.DataFrame):
        for day, fid, count in zip(per["signal_date"], per["factor_id"], per["n"], strict=True):
            key = str(fid)
            if key in per_factor:
                per_factor[key][pd.Timestamp(day).date()] = int(count)

    ok_d, dup = query(
        holder,
        f"select factor_id, count(*) as n from (select factor_id, signal_date, ts_code "
        f"from {fd} where {bounds} group by factor_id, signal_date, ts_code "
        f"having count(*) > 1) group by factor_id",
    )
    duplicates: dict[str, int] = {}
    if ok_d and isinstance(dup, pd.DataFrame):
        for fid, count in zip(dup["factor_id"], dup["n"], strict=True):
            duplicates[str(fid)] = int(count)

    return universe_per_day, per_factor, duplicates


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--json", type=Path, default=None, help="also write metrics as JSON")
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from walk_forward_common import new_holder
    from walk_forward_factor_daily import FROZEN_FACTOR_SET_V1

    universe_per_day, per_factor, duplicates = collect(
        new_holder(), args.start, args.end, FROZEN_FACTOR_SET_V1
    )
    print(f"window {args.start} .. {args.end}; universe trading days: {len(universe_per_day)}")
    print("denominator: same-day universe_daily members with is_member=1 (C6.3)")
    print(f"gates: median>={COVERAGE_GATE}, p05>={P05_GATE}, "
          f"min_monthly_day_ratio>={MONTHLY_DAY_RATIO_GATE}, duplicates==0")
    print()
    header = (f"{'factor':30} {'days':>10} {'day_ratio':>10} {'min_month':>10} "
              f"{'median':>8} {'p05':>8} {'min':>8} {'dups':>6}  status")
    print(header)

    results: list[FactorCoverage] = []
    for factor_id in FROZEN_FACTOR_SET_V1:
        result = screen_factor(
            factor_id,
            per_factor.get(factor_id, {}),
            universe_per_day,
            duplicate_row_count=duplicates.get(factor_id, 0),
        )
        results.append(result)
        print(f"{factor_id:30} {result.valid_day_count:>5}/{result.expected_day_count:<4} "
              f"{result.day_ratio:>10.4f} {result.minimum_monthly_day_ratio:>10.4f} "
              f"{result.median_coverage_ratio:>8.4f} {result.p05_coverage_ratio:>8.4f} "
              f"{result.minimum_coverage_ratio:>8.4f} {result.duplicate_row_count:>6}  "
              f"{result.eligibility_status} {list(result.reason_codes)}")

    failed = [r.factor_id for r in results if not r.coverage_gate_pass]
    print()
    print(f"PASS {len(results) - len(failed)}/{len(results)}; "
          f"ELIGIBILITY_FAIL: {failed or 'none'}")
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(
                {
                    "window": [args.start.isoformat(), args.end.isoformat()],
                    "universe_day_count": len(universe_per_day),
                    "gates": {
                        "coverage": COVERAGE_GATE, "p05": P05_GATE,
                        "monthly_day_ratio": MONTHLY_DAY_RATIO_GATE,
                    },
                    "factors": [
                        r.__dict__ | {"reason_codes": list(r.reason_codes)} for r in results
                    ],
                },
                ensure_ascii=False, indent=2,
            ),
            encoding="utf-8",
        )
        print(f"json written: {args.json}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
