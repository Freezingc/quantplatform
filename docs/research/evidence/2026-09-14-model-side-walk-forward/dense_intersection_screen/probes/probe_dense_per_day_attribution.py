"""Which dense factor drives the intersection down on the worst days of a window?

The dense-intersection screen says how many days fall below the score-health floor; the per-factor
screen says each factor's median and p05 over the whole window. Neither can attribute a particular
bad day, and the train window has days that need it: 2015 is the worst year (intersection median
0.7631, min 0.4453, and every one of the ten worst days falls in 2015-07..2015-10), while the two
weakest dense factors over the whole 2014-2021 window are the two repair variants (yeslab_206 at
median 0.8935 and yeslab_443_reg30_max20_v1 at 0.9131). Whether those two are also what sinks
July-October 2015 is a different question and needs per-day numbers.

This reuses the pinned-denominator collector from screen_frozen_v2_factor_daily_coverage.py, so the
counts are the same ones the per-factor screen reports, and adds the one thing that screen
aggregates away: coverage per factor per day, and the ranking of factors on each of the worst days.

SAFETY: SELECT-only against dfs://QuantResearchV2Test. No writes, no weights. Run from research_v2/.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, "/data/zhangyudong/qrv2_research_artifacts")
sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

from screen_frozen_v2_factor_daily_coverage import collect_restricted  # noqa: E402
from walk_forward_common import new_holder  # noqa: E402
from walk_forward_factor_daily import FROZEN_FACTOR_SET_V2_DENSE  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument("--worst", type=int, default=10)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    dense = tuple(FROZEN_FACTOR_SET_V2_DENSE)
    universe, per_factor, _ = collect_restricted(new_holder(), args.start, args.end, dense)
    print(f"window {args.start} .. {args.end}; universe days: {len(universe)}; factors: {len(dense)}")

    days: list[dict] = []
    for day in sorted(universe):
        denominator = universe[day]
        if denominator <= 0:
            continue
        coverages = {fid: per_factor[fid].get(day, 0) / denominator for fid in dense}
        days.append(
            {
                "day": day.isoformat(),
                "universe_rows": denominator,
                "min_factor_coverage": min(coverages.values()),
                "worst_factor": min(coverages, key=lambda fid: coverages[fid]),
                "coverages": {fid: round(value, 6) for fid, value in coverages.items()},
                "product_of_coverages": round(
                    __import__("math").prod(coverages.values()), 6
                ),
            }
        )

    worst = sorted(days, key=lambda entry: entry["min_factor_coverage"])[: args.worst]
    print(f"\nthe {len(worst)} days whose weakest dense factor is lowest:")
    for entry in worst:
        print(
            f"  {entry['day']} weakest={entry['worst_factor']} "
            f"{entry['min_factor_coverage']:.4f} (universe {entry['universe_rows']})"
        )
        ordered = sorted(entry["coverages"].items(), key=lambda item: item[1])
        print("      " + ", ".join(f"{fid.replace('yeslab_', '')}={value:.3f}" for fid, value in ordered))

    counts: dict[str, int] = {}
    for entry in days:
        counts[entry["worst_factor"]] = counts.get(entry["worst_factor"], 0) + 1
    print("\nhow often each factor is the weakest over the whole window:")
    for fid, count in sorted(counts.items(), key=lambda item: -item[1]):
        print(f"  {fid:<28} {count:>4} days")

    per_factor_summary = {
        fid: {
            "median": sorted(entry["coverages"][fid] for entry in days)[len(days) // 2],
            "min": min(entry["coverages"][fid] for entry in days),
        }
        for fid in dense
    }
    print("\nper-factor median/min of daily coverage in this window:")
    for fid, stats in per_factor_summary.items():
        print(f"  {fid:<28} median={stats['median']:.4f} min={stats['min']:.4f}")

    args.out.write_text(
        json.dumps(
            {
                "window": [args.start.isoformat(), args.end.isoformat()],
                "factor_ids": list(dense),
                "days": days,
                "worst_days": worst,
                "weakest_factor_day_counts": counts,
                "per_factor_summary": per_factor_summary,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
