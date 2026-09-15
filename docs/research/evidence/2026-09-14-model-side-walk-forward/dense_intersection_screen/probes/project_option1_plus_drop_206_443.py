#!/usr/bin/env python3
"""Fill the missing cell of the §AF matrix: option 1 AND removing {206, 443} from the dense subset.

§AF's matrix has six rows: the nine-factor baseline, option 2 (nine minus the two industry factors), nine
minus 206, nine minus {206, 443}, the five-factor set, and option 1 projected. The cell nobody can measure
directly is option 1 combined with removing {206, 443}, because option 1 is not implemented -- there is no
published artifact whose seven-factor intersection has backfilled STAR labels. It is nevertheless the cell
that answers the question §AC/§AE raised: if the owner wants the floor above 0.974 without halving the dense
subset, does backfilling the labels AND dropping the two suspension-limited factors get there?

It can be projected from three existing measurements without touching the database. The seven-factor
intersection per day comes from the instrument run leave_out_206_443_2022_2025.json. The STAR rows a backfill
would lift come from star_rows_in_denominator_420.json, which records per day both the number of STAR
valid-label rows and how many rows each of the nine factors has for those securities, so the gain can be
bounded conservatively by the weakest of the five non-industry factors this subset still needs (381, 425,
464, 484, 96) rather than assumed complete. Post-cutover days need no adjustment because STAR labels already
exist there and the SH/SZ union perturbs published values by a median relative difference of 0.0.

Pure post-processing over existing measurement files: no database, no cache, no repository import.

Usage: python project_option1_plus_drop_206_443.py \
           --seven  measurements/leave_out_206_443_2022_2025.json \
           --stars  ../star_industry_coverage/measurements/star_rows_in_denominator_420.json \
           --cutover 2023-09-25 --out measurements/option1_plus_drop_206_443.json
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import pandas as pd

NEEDED_NON_INDUSTRY = ["yeslab_381", "yeslab_425", "yeslab_464", "yeslab_484", "yeslab_96"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seven", type=Path, required=True)
    ap.add_argument("--stars", type=Path, required=True)
    ap.add_argument("--union-extra", type=Path, default=None,
                    help="residual_gap_attribution.json: measured extra rows the SH/SZ union restores beyond "
                         "the STAR-only bound, on the eight largest-residual days")
    ap.add_argument("--cutover", default="2023-09-25")
    ap.add_argument("--floor", type=float, default=0.90)
    ap.add_argument("--bar", type=float, default=0.95)
    ap.add_argument("--ceiling", type=float, default=0.974)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    inst = pd.DataFrame(json.loads(args.seven.read_text())["per_day"])
    inst["day"] = inst.day.astype(str)
    stars = pd.DataFrame(json.loads(args.stars.read_text())["per_day"])
    stars["day"] = stars.day.astype(str)
    star_by_day = stars.set_index("day").to_dict("index")
    # §AB measured, for the eight largest-residual days, how many NON-STAR rows the SH/SZ union restores
    # because the cache lost their label in a bulk expiry. Those rows miss only the two industry factors and
    # carry all seven non-industry ones, so they are restorable in this subset too. Where that was measured it
    # is added; elsewhere the STAR-only bound stands, which understates coverage.
    extra_by_day = {}
    if args.union_extra is not None:
        for r in json.loads(args.union_extra.read_text())["per_day"]:
            extra_by_day[r["day"]] = int(r["residual_classification"].get("FIXABLE_BY_UNION_LABEL", 0))

    rows = []
    for _, r in inst.iterrows():
        day = r.day
        V = int(r.valid_label_rows); N = int(r.intersection_rows)
        pre = day < args.cutover
        if pre:
            s = star_by_day.get(day)
            if s is None:
                raise SystemExit(f"{day} is pre-cutover but has no STAR decomposition; cannot project it")
            counts = [int(s["star_rows_present_per_factor"][f]) for f in NEEDED_NON_INDUSTRY]
            gain_opt = int(s["star_valid_label_rows"])
            gain_cons = min(min(counts), gain_opt)
            extra = extra_by_day.get(day, 0)
            gain_opt += extra
            gain_cons += extra
        else:
            gain_opt = gain_cons = 0
        rows.append({"day": day, "pre_cutover": pre, "valid_label_rows": V, "intersection_rows_now": N,
                     "coverage_now": N / V, "gain_optimistic": gain_opt, "gain_conservative": gain_cons,
                     "coverage_after_optimistic": (N + gain_opt) / V,
                     "coverage_after_conservative": (N + gain_cons) / V})

    df = pd.DataFrame(rows)
    c = df.coverage_after_conservative
    co = df.coverage_after_optimistic
    pre = df.pre_cutover

    def block(series: pd.Series, label: str) -> dict:
        return {"bound": label, "min": float(series.min()),
                "min_day": str(df.loc[series.idxmin(), "day"]),
                "median": float(series.median()), "max": float(series.max()),
                f"days_below_{args.floor}": int((series < args.floor).sum()),
                f"days_below_{args.bar}": int((series < args.bar).sum()),
                f"days_below_{args.ceiling}": int((series < args.ceiling).sum()),
                "pre_cutover_min": float(series[pre].min()), "pre_cutover_median": float(series[pre].median()),
                "post_cutover_min": float(series[~pre].min())}

    out = {
        "question": "What does option 1 combined with removing {206, 443} from the dense subset achieve, "
                    "over all 969 OOS days?",
        "dense_subset": ["yeslab_357", "yeslab_381", "yeslab_425", "yeslab_464", "yeslab_471",
                         "yeslab_484", "yeslab_96"],
        "method": "seven-factor intersection measured by the instrument; STAR rows a label backfill would lift "
                  "taken from the 420-day decomposition, bounded conservatively by the weakest of the five "
                  "non-industry factors this subset still needs; post-cutover days unadjusted",
        "days": len(df), "cutover": args.cutover,
        "before": block(df.coverage_now, "seven factors, no backfill (measured)"),
        "after_conservative": block(c, "seven factors + option 1, conservative bound"),
        "after_optimistic": block(co, "seven factors + option 1, optimistic bound"),
        "bounds_agree_on_the_minimum": bool(abs(float(c.min()) - float(co.min())) < 1e-12),
        "days_where_the_bounds_differ": int((co - c > 1e-12).sum()),
        "union_extra_applied_on": sorted(extra_by_day) if extra_by_day else [],
        "bound_is_hybrid": bool(extra_by_day),
        "g6_at_floor": float((c >= args.floor).mean()),
        "answer": None, "per_day": rows,
        "database_writes": False, "weights_written": False, "cache_writes": False,
    }
    b = out["after_conservative"]
    out["answer"] = (
        f"Option 1 plus removing 206 and 443 leaves {b[f'days_below_{args.floor}']} days below {args.floor} and "
        f"{b[f'days_below_{args.ceiling}']} below {args.ceiling} out of {len(df)}, with a minimum coverage of "
        f"{b['min']:.6f} on {b['min_day']} and a median of {b['median']:.6f}; G6 at the floor is "
        f"{out['g6_at_floor']:.6f}. The optimistic and conservative bounds differ on "
        f"{out['days_where_the_bounds_differ']} days and "
        f"{'agree' if out['bounds_agree_on_the_minimum'] else 'disagree'} on the minimum.")
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(out["answer"])
    print(json.dumps({k: out[k] for k in ("before", "after_conservative", "after_optimistic",
                                          "g6_at_floor", "days_where_the_bounds_differ")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
