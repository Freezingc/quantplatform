"""Price Option 1 exactly: what the 0.90 floor would see after a full STAR industry-label backfill.

Two bounds, because the per-security join of "STAR row" x "which of the other seven factors it has" is
not in this probe's output:

  optimistic  every STAR valid-label row that is not already complete becomes complete
  conservative a STAR row only becomes complete if it already carries all seven non-industry factors,
               so the gain per day is the smallest per-factor STAR presence among those seven

The binding day is the same under both, which is what makes the number usable.

Reads the two measurement files already written. No database, no weights.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

INDUSTRY_DEPENDENT = {"yeslab_357", "yeslab_471"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--floor", type=float, default=0.90)
    ap.add_argument("--proposed-bar", type=float, default=0.95)
    args = ap.parse_args()

    d = json.loads(args.rows.read_text())
    if d["reproduces_the_instrument"]["verdict"] != "EXACT":
        raise SystemExit("the probe does not reproduce the instrument; refusing to project from it")

    per = []
    for r in d["per_day"]:
        V, N = r["valid_label_rows"], r["intersection_rows"]
        others = {f: n for f, n in r["star_rows_present_per_factor"].items()
                  if f not in INDUSTRY_DEPENDENT}
        opt = N + r["gap_fixable_by_star_labels"]
        cons = N + min(min(others.values()), r["gap_fixable_by_star_labels"])
        per.append({
            "day": r["day"], "valid_label_rows": V, "intersection_rows_now": N,
            "coverage_now": N / V,
            "rows_needed_at_floor": r["rows_needed_to_reach_floor"],
            "gain_optimistic": r["gap_fixable_by_star_labels"],
            "gain_conservative": cons - N,
            "coverage_after_optimistic": opt / V,
            "coverage_after_conservative": cons / V,
            "clears_floor_optimistic": (opt / V) >= args.floor,
            "clears_floor_conservative": (cons / V) >= args.floor,
            "clears_proposed_bar_optimistic": (opt / V) >= args.proposed_bar,
            "clears_proposed_bar_conservative": (cons / V) >= args.proposed_bar,
            "rows_short_of_proposed_bar_conservative":
                max(0, -(-(args.proposed_bar * V // 1)) + 0) and None,  # replaced below
            "residual_non_star_rows": r["residual_gap_not_star"],
        })
        import math
        p = per[-1]
        p["rows_short_of_proposed_bar_conservative"] = max(
            0, math.ceil(args.proposed_bar * V) - (N + p["gain_conservative"]))
        p["rows_short_of_proposed_bar_optimistic"] = max(
            0, math.ceil(args.proposed_bar * V) - (N + p["gain_optimistic"]))

    worst_c = min(per, key=lambda x: x["coverage_after_conservative"])
    worst_o = min(per, key=lambda x: x["coverage_after_optimistic"])
    out = {
        "what_this_prices": "Option 1 -- backfill STAR industry history, keep the nine-factor dense set",
        "floor": args.floor, "proposed_robustness_bar": args.proposed_bar,
        "days": len(per),
        "days_below_floor_after": {
            "optimistic": sum(1 for p in per if not p["clears_floor_optimistic"]),
            "conservative": sum(1 for p in per if not p["clears_floor_conservative"]),
            "before": len(per),
        },
        "g6_after": {
            "optimistic": 1.0 if all(p["clears_floor_optimistic"] for p in per) else None,
            "conservative": 1.0 if all(p["clears_floor_conservative"] for p in per) else None,
            "note": "computed over the 24 problem days only; the other 945 days can only improve, "
                    "because a backfill adds rows and never removes them",
        },
        "worst_day_after": {
            "conservative": {"day": worst_c["day"], "coverage": worst_c["coverage_after_conservative"],
                             "margin_above_floor": worst_c["coverage_after_conservative"] - args.floor,
                             "rows_short_of_proposed_bar": worst_c["rows_short_of_proposed_bar_conservative"]},
            "optimistic": {"day": worst_o["day"], "coverage": worst_o["coverage_after_optimistic"],
                           "margin_above_floor": worst_o["coverage_after_optimistic"] - args.floor,
                           "rows_short_of_proposed_bar": worst_o["rows_short_of_proposed_bar_optimistic"]},
        },
        "meets_the_proposed_0_95_bar": {
            "optimistic": all(p["clears_proposed_bar_optimistic"] for p in per),
            "conservative": all(p["clears_proposed_bar_conservative"] for p in per),
        },
        "for_reference_option_2_coverage_min": 0.9713701431492843,
        "per_day": per,
        "database_writes": False, "weights_written": False,
    }
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_day"}, ensure_ascii=False, indent=1))
    print(f"{'day':12}{'now':>9}{'after_cons':>12}{'after_opt':>11}{'short@0.95':>11}")
    for p in sorted(per, key=lambda x: x["coverage_after_conservative"])[:6]:
        print(f"{p['day']:12}{p['coverage_now']:9.6f}{p['coverage_after_conservative']:12.6f}"
              f"{p['coverage_after_optimistic']:11.6f}{p['rows_short_of_proposed_bar_conservative']:11d}")


if __name__ == "__main__":
    main()
