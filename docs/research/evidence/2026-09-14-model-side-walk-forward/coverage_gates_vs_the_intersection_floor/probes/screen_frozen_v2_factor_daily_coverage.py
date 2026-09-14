"""Would ANY coverage gate in use have caught the 24 dropped OOS days? Screen frozen-v2 on factor_daily.

The root-cause work (docs/research/.../star_industry_coverage/root_cause.md) established that
`yeslab_357` and `yeslab_471` have no STAR-Market rows before 2023-09-25 in the artifact the frozen
chain actually consumes, because the replay needs a PIT industry label the SW2021 cache does not have
for those securities. That explains the mechanism. It leaves a governance question nobody has
measured, and it is the one that matters for whether this can happen again:

    The 24 days died on a 0.90 floor applied to the INTERSECTION of nine dense factors. Every
    coverage instrument in use screens factors ONE AT A TIME. So does a factor that passes all of
    them still have enough room to kill evaluation days?

Two instruments exist and they disagree by design (docs/handoffs/2026-09-06-c-plan-audit-decision-list.md
§3f-续三十二 (3)): the persisted snapshot reads daily RankIC `sample_count` against reference factor
`yeslab_0` with median>=0.80 / p05>=0.70 / monthly>=0.70 / valid_days>=120 / day_ratio>=0.80 /
duplicates==0, while the live screen `screen_factor_daily_coverage.py` reads `factor_daily` row counts
against same-day `universe_daily` members with median>=0.70 / p05>=0.30 / monthly>=0.90 /
duplicates==0. Both were run for the frozen **v1** (17 factors) over the **train** window. Neither has
ever been run for the frozen **v2** set (the 15 factors M0R/M1/M2/M3 consume) over the **OOS** window
2022-2025, which is where the 24 days are.

This runs the live screen's own pure functions on frozen-v2 for a chosen window, and then applies the
snapshot instrument's thresholds to the SAME factor_daily counts. That second verdict is labelled as
what it is -- snapshot thresholds on factor_daily counts, a third combination that exists nowhere in
production -- because the question is not "which instrument is right" but "is there any threshold pair
in use under which these two factors fail". If both verdicts are PASS while 24 days still died, the
gap is structural: per-factor screening cannot see an intersection floor.

Reads `collect()` from the existing screen, so the queries are the three server-side aggregates it
already uses (universe per day, per-factor per-day counts, duplicate business keys), with one
deliberate difference: the factor_id filter is pushed into the SQL. The original scans every factor
in the window -- all 476 of them -- because it was written to audit a set that was not yet fixed.
Here the set is the frozen 15, and a 30x smaller scan matters because M2 is mid-run and will need
this database to assemble its next fold. Thresholds and the gate logic are still the imported pure
functions, so nothing about the verdict is re-implemented. No writes.

SAFETY: SELECT-only against dfs://QuantResearchV2Test. No weights, no artifact mutation, and it does
not edit or import anything the M2 driver executes at runtime. Run from research_v2/.
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import date
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from screen_factor_daily_coverage import (  # noqa: E402
    COVERAGE_GATE,
    MONTHLY_DAY_RATIO_GATE,
    P05_GATE,
    FactorCoverage,
    screen_factor,
)
from walk_forward_common import TARGET_DATABASE, new_holder, query  # noqa: E402
from walk_forward_common import current_versions  # noqa: E402
from walk_forward_factor_daily import (  # noqa: E402
    FROZEN_FACTOR_SET_V2,
    FROZEN_FACTOR_SET_V2_DENSE,
)


def collect_restricted(
    holder: dict, start: date, end: date, factor_ids: tuple[str, ...]
) -> tuple[dict[date, int], dict[str, dict[date, int]], dict[str, int]]:
    """The same three aggregates as screen_factor_daily_coverage.collect, filtered to our factors."""
    import pandas as pd

    bounds = f"signal_date >= {start:%Y.%m.%d} and signal_date <= {end:%Y.%m.%d}"
    fd = f"loadTable('{TARGET_DATABASE}','factor_daily')"
    ids = ", ".join(f"'{fid}'" for fid in factor_ids)

    ok_u, uni = query(
        holder,
        f"select signal_date, count(*) as n from loadTable('{TARGET_DATABASE}','universe_daily') "
        f"where {bounds} and is_member=1 group by signal_date",
    )
    if not ok_u or not isinstance(uni, pd.DataFrame):
        raise RuntimeError("universe_daily aggregate query failed")
    universe_per_day = {
        pd.Timestamp(day).date(): int(count)
        for day, count in zip(uni["signal_date"], uni["n"], strict=True)
    }

    ok_f, per = query(
        holder,
        f"select signal_date, factor_id, count(*) as n from {fd} where {bounds} "
        f"and factor_id in [{ids}] group by signal_date, factor_id",
    )
    if not ok_f or not isinstance(per, pd.DataFrame):
        raise RuntimeError("factor_daily per-day aggregate query failed")
    per_factor: dict[str, dict[date, int]] = {fid: {} for fid in factor_ids}
    for day, fid, count in zip(per["signal_date"], per["factor_id"], per["n"], strict=True):
        key = str(fid)
        if key in per_factor:
            per_factor[key][pd.Timestamp(day).date()] = int(count)

    ok_d, dup = query(
        holder,
        f"select factor_id, count(*) as n from (select factor_id, signal_date, ts_code "
        f"from {fd} where {bounds} and factor_id in [{ids}] group by factor_id, signal_date, ts_code "
        f"having count(*) > 1) group by factor_id",
    )
    if not ok_d:
        raise RuntimeError("factor_daily duplicate-key aggregate query failed")
    duplicates: dict[str, int] = {}
    if isinstance(dup, pd.DataFrame):
        for fid, count in zip(dup["factor_id"], dup["n"], strict=True):
            duplicates[str(fid)] = int(count)

    return universe_per_day, per_factor, duplicates


def collect_universe_pinned(
    holder: dict, start: date, end: date
) -> tuple[dict[date, int], dict]:
    """Universe denominator the frozen chain itself consumes: pinned data_version only.

    `universe_daily` keeps every published version, and more than one of them can carry
    is_member=1 for the same day. Counting rows without resolving the pin therefore counts the same
    security several times. Measured on 2025-01-09: four versions are is_member=1 (3,923 + 3,923 +
    7,846 + 3,923 = 19,615 rows), the pinned one holds 3,923, and the published panel for that day
    holds 3,923 rows -- so the pin, not the raw count, is the denominator the runs actually used.
    `filter_current` in the assembly path does this filtering; a count(*) denominator does not.
    """
    import pandas as pd

    pins = current_versions(holder, "universe_daily")
    bounds = f"signal_date >= {start:%Y.%m.%d} and signal_date <= {end:%Y.%m.%d}"
    ok, frame = query(
        holder,
        f"select signal_date, data_version, count(*) as n "
        f"from loadTable('{TARGET_DATABASE}','universe_daily') "
        f"where {bounds} and is_member=1 group by signal_date, data_version",
    )
    if not ok or not isinstance(frame, pd.DataFrame):
        raise RuntimeError("universe_daily per-version aggregate query failed")
    per_day: dict[date, int] = {}
    versions_per_day: dict[date, int] = {}
    raw_total: dict[date, int] = {}
    for row in frame.itertuples(index=False):
        day = pd.Timestamp(row.signal_date).date()
        count = int(row.n)
        versions_per_day[day] = versions_per_day.get(day, 0) + 1
        raw_total[day] = raw_total.get(day, 0) + count
        if pins.get(f"{day.year}-{day.month:02d}") == str(row.data_version):
            per_day[day] = count
    diagnostics = {
        "days_with_a_pinned_version": len(per_day),
        "days_with_is_member_rows": len(raw_total),
        "days_missing_a_pinned_version": sorted(
            str(day) for day in raw_total if day not in per_day
        ),
        "days_with_more_than_one_version": int(sum(1 for n in versions_per_day.values() if n > 1)),
        "worst_inflated_days": sorted(
            (
                {
                    "day": str(day),
                    "raw_rows": raw_total[day],
                    "pinned_rows": per_day.get(day),
                    "inflation": round(raw_total[day] / per_day[day], 3) if per_day.get(day) else None,
                    "versions": versions_per_day[day],
                }
                for day in raw_total
                if per_day.get(day) and raw_total[day] != per_day[day]
            ),
            key=lambda item: -(item["inflation"] or 0),
        )[:12],
    }
    return per_day, diagnostics
# The persisted snapshot instrument's thresholds (record_structural_quality_snapshot.py:107-131).
SNAPSHOT_MEDIAN_GATE = 0.80
SNAPSHOT_P05_GATE = 0.70
SNAPSHOT_MONTHLY_GATE = 0.70
SNAPSHOT_MIN_VALID_DAYS = 120
SNAPSHOT_DAY_RATIO_GATE = 0.80


def snapshot_style_verdict(result: FactorCoverage) -> dict:
    """Apply the snapshot instrument's six conditions to one live-screen result.

    Same numbers, different thresholds: this answers "would the stricter instrument have failed this
    factor", not "what does the snapshot table say" (that one reads RankIC sample_count, not rows).
    """
    failures = []
    if result.valid_day_count < SNAPSHOT_MIN_VALID_DAYS:
        failures.append(f"valid_days {result.valid_day_count} < {SNAPSHOT_MIN_VALID_DAYS}")
    if result.day_ratio < SNAPSHOT_DAY_RATIO_GATE:
        failures.append(f"day_ratio {result.day_ratio:.4f} < {SNAPSHOT_DAY_RATIO_GATE}")
    if result.median_coverage_ratio < SNAPSHOT_MEDIAN_GATE:
        failures.append(f"median {result.median_coverage_ratio:.4f} < {SNAPSHOT_MEDIAN_GATE}")
    if result.p05_coverage_ratio < SNAPSHOT_P05_GATE:
        failures.append(f"p05 {result.p05_coverage_ratio:.4f} < {SNAPSHOT_P05_GATE}")
    if result.minimum_monthly_day_ratio < SNAPSHOT_MONTHLY_GATE:
        failures.append(f"min_monthly {result.minimum_monthly_day_ratio:.4f} < {SNAPSHOT_MONTHLY_GATE}")
    if result.duplicate_row_count:
        failures.append(f"duplicates {result.duplicate_row_count} != 0")
    return {"pass": not failures, "failures": failures}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--label", required=True, help="window name recorded in the JSON")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--denominator",
        choices=("raw", "pinned"),
        default="pinned",
        help="raw reproduces screen_factor_daily_coverage.py's count(*); pinned resolves the "
        "published data_version first, which is what the frozen chain consumes",
    )
    args = parser.parse_args()

    factor_ids = tuple(FROZEN_FACTOR_SET_V2)
    dense = set(FROZEN_FACTOR_SET_V2_DENSE)
    print(
        f"window {args.start} .. {args.end} ({args.label}); frozen-v2 factors: {len(factor_ids)}; "
        f"dense subset: {len(dense)}",
        flush=True,
    )
    holder = new_holder()
    universe_raw, per_factor, duplicates = collect_restricted(
        holder, args.start, args.end, factor_ids
    )
    universe_pinned, universe_diagnostics = collect_universe_pinned(holder, args.start, args.end)
    universe_per_day = universe_pinned if args.denominator == "pinned" else universe_raw
    print(
        f"universe trading days in window: raw={len(universe_raw)} pinned={len(universe_pinned)}; "
        f"denominator in use: {args.denominator}",
        flush=True,
    )
    print(
        f"  days with >1 is_member version: {universe_diagnostics['days_with_more_than_one_version']}; "
        f"days missing a pinned version: {universe_diagnostics['days_missing_a_pinned_version'] or 'none'}",
        flush=True,
    )
    for entry in universe_diagnostics["worst_inflated_days"][:5]:
        print(
            f"  inflated: {entry['day']} raw={entry['raw_rows']} pinned={entry['pinned_rows']} "
            f"x{entry['inflation']} ({entry['versions']} versions)",
            flush=True,
        )
    print(
        f"live-screen gates: median>={COVERAGE_GATE}, p05>={P05_GATE}, "
        f"min_monthly_day_ratio>={MONTHLY_DAY_RATIO_GATE}, duplicates==0",
        flush=True,
    )
    print(
        f"snapshot-style gates applied to the same counts: median>={SNAPSHOT_MEDIAN_GATE}, "
        f"p05>={SNAPSHOT_P05_GATE}, monthly>={SNAPSHOT_MONTHLY_GATE}, "
        f"valid_days>={SNAPSHOT_MIN_VALID_DAYS}, day_ratio>={SNAPSHOT_DAY_RATIO_GATE}, duplicates==0",
        flush=True,
    )
    print(
        f"\n{'factor':30} {'dense':>5} {'days':>10} {'day_ratio':>9} {'min_mon':>8} "
        f"{'median':>7} {'p05':>7} {'min':>7} {'dups':>5}  live / snapshot-style",
        flush=True,
    )

    entries = []
    for factor_id in factor_ids:
        result = screen_factor(
            factor_id,
            per_factor.get(factor_id, {}),
            universe_per_day,
            duplicate_row_count=duplicates.get(factor_id, 0),
        )
        snapshot = snapshot_style_verdict(result)
        record = asdict(result) | {
            "reason_codes": list(result.reason_codes),
            "in_dense_subset": factor_id in dense,
            "snapshot_style": snapshot,
        }
        entries.append(record)
        print(
            f"{factor_id:30} {'DENSE' if factor_id in dense else '':>5} "
            f"{result.valid_day_count:>5}/{result.expected_day_count:<4} "
            f"{result.day_ratio:>9.4f} {result.minimum_monthly_day_ratio:>8.4f} "
            f"{result.median_coverage_ratio:>7.4f} {result.p05_coverage_ratio:>7.4f} "
            f"{result.minimum_coverage_ratio:>7.4f} {result.duplicate_row_count:>5}  "
            f"{'PASS' if result.coverage_gate_pass else 'FAIL'} / "
            f"{'PASS' if snapshot['pass'] else 'FAIL ' + '; '.join(snapshot['failures'])}",
            flush=True,
        )

    live_fail = [e["factor_id"] for e in entries if not e["coverage_gate_pass"]]
    snap_fail = [e["factor_id"] for e in entries if not e["snapshot_style"]["pass"]]
    print(
        f"\nlive-screen ELIGIBILITY_FAIL: {live_fail or 'none'}\n"
        f"snapshot-style FAIL:          {snap_fail or 'none'}",
        flush=True,
    )
    payload = {
        "label": args.label,
        "window": [args.start.isoformat(), args.end.isoformat()],
        "factor_set": "FROZEN_FACTOR_SET_V2",
        "factor_ids": list(factor_ids),
        "dense_subset": sorted(dense),
        "universe_day_count": len(universe_per_day),
        "source": "factor_daily row counts / same-day universe_daily is_member=1 (C6.3)",
        "denominator": args.denominator,
        "universe_denominator_diagnostics": universe_diagnostics,
        "universe_days_raw": len(universe_raw),
        "universe_days_pinned": len(universe_pinned),
        "live_screen_gates": {
            "median": COVERAGE_GATE,
            "p05": P05_GATE,
            "monthly_day_ratio": MONTHLY_DAY_RATIO_GATE,
            "duplicates": 0,
        },
        "snapshot_style_gates": {
            "median": SNAPSHOT_MEDIAN_GATE,
            "p05": SNAPSHOT_P05_GATE,
            "monthly_day_ratio": SNAPSHOT_MONTHLY_GATE,
            "min_valid_days": SNAPSHOT_MIN_VALID_DAYS,
            "day_ratio": SNAPSHOT_DAY_RATIO_GATE,
            "duplicates": 0,
            "note": (
                "the persisted snapshot reads daily RankIC sample_count against reference yeslab_0; "
                "this applies its thresholds to factor_daily row counts, a combination that exists "
                "nowhere in production, purely to ask whether a stricter gate would have caught it"
            ),
        },
        "factors": entries,
        "summary": {"live_screen_fail": live_fail, "snapshot_style_fail": snap_fail},
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nwrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
