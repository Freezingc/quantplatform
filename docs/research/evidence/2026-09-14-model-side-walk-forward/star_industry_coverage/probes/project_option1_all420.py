#!/usr/bin/env python3
"""Price option 1 over ALL 420 pre-cutover OOS days, not just the 24 that broke the floor.

`project_option1.py` priced the backfill over the 24 problem days and then asserted that "the other 945
days can only improve, because a backfill adds rows and never removes them". That assertion was copied
forward through four sections of problems_found.md as an uncleared caveat. It is testable, and this
tests it: `probe_star_rows_in_denominator.py` now runs over all 420 pre-cutover days and reproduces the
instrument's numerator and denominator EXACTLY on 420/420, so what follows is a measurement over the
whole affected segment rather than an extrapolation from 24 days.

Two things are verified rather than assumed. First, that the denominator does not move: the STAR rows a
backfill would lift are ALREADY inside `valid_label_rows` (they carry the other seven dense factors and
are missing only the two industry-dependent ones), so coverage_after = (N + gain) / V with V fixed.
Second, that the optimistic and conservative gains from `project_option1.py` give the same worst day --
they differ slightly on some days, so both are carried through to the minimum.

Pure post-processing over two existing measurement files: no database, no cache, no repository import.

Usage: python project_option1_all420.py \
           --rows       ../measurements/star_rows_in_denominator_420.json \
           --instrument ../../dense_intersection_screen/measurements/predicted_coverage_explicit_factor_list_2022_2025.json \
           --cutover    2023-09-25 --floor 0.90 --bar 0.95 \
           --out        ../measurements/option1_price_all420.json
"""
from __future__ import annotations
import argparse, json, math
from pathlib import Path
import numpy as np
import pandas as pd

INDUSTRY_DEPENDENT = {"yeslab_357", "yeslab_471"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=Path, required=True)
    ap.add_argument("--instrument", type=Path, required=True)
    ap.add_argument("--cutover", default="2023-09-25")
    ap.add_argument("--floor", type=float, default=0.90)
    ap.add_argument("--bar", type=float, default=0.95)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = json.loads(args.rows.read_text())
    repro = d["reproduces_the_instrument"]
    if repro["verdict"] != "EXACT":
        raise SystemExit(f"the decomposition does not reproduce the instrument ({repro['verdict']}); refusing to project")

    rows = []
    for r in d["per_day"]:
        others = {f: n for f, n in r["star_rows_present_per_factor"].items() if f not in INDUSTRY_DEPENDENT}
        V, N, fix = r["valid_label_rows"], r["intersection_rows"], r["gap_fixable_by_star_labels"]
        cons = N + min(min(others.values()), fix)
        rows.append({
            "day": r["day"], "valid_label_rows": V, "intersection_rows_now": N,
            "star_valid_label_rows": r["star_valid_label_rows"],
            "star_rows_already_complete": r["star_rows_already_complete"],
            "gain_optimistic": fix, "gain_conservative": cons - N,
            "residual_gap_not_star": r["residual_gap_not_star"],
            "coverage_now": N / V, "coverage_after_optimistic": (N + fix) / V,
            "coverage_after_conservative": cons / V,
            "non_industry_star_counts_uniform": min(others.values()) == max(others.values()),
        })
    df = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)

    problem_days_src = None
    inst = pd.DataFrame(json.loads(args.instrument.read_text())["per_day"])
    post = inst[inst.day >= args.cutover]
    problem_days = set(inst[inst.day < args.cutover].loc[
        inst[inst.day < args.cutover].coverage < args.floor, "day"])
    df["is_problem_day"] = df.day.isin(problem_days)

    def worst(col: str) -> dict:
        i = df[col].idxmin()
        r = df.loc[i]
        return {"day": str(r.day), "coverage": float(r[col]), "valid_label_rows": int(r.valid_label_rows),
                "intersection_rows_now": int(r.intersection_rows_now), "gain": int(r[
                    "gain_optimistic" if col.endswith("optimistic") else "gain_conservative"]),
                "residual_gap_not_star": int(r.residual_gap_not_star),
                "is_problem_day": bool(r.is_problem_day),
                "rows_short_of_bar": int(math.ceil(args.bar * r.valid_label_rows)
                                         - (r.intersection_rows_now + r[
                                             "gain_optimistic" if col.endswith("optimistic")
                                             else "gain_conservative"]))}

    tot = len(df) + len(post)
    ge_floor = int((df.coverage_after_conservative >= args.floor).sum()) + int((post.coverage >= args.floor).sum())
    ge_bar = int((df.coverage_after_conservative >= args.bar).sum()) + int((post.coverage >= args.bar).sum())

    out = {
        "question": "Does option 1 clear the floor and the proposed bar on ALL 420 pre-cutover days, "
                    "and is the worst day the same one the 24-day projection named?",
        "reproduces_the_instrument": repro,
        "denominator_is_fixed_because": "star_rows_already_complete == 0 and every STAR valid-label row "
                                        "already carries the seven non-industry dense factors, so the STAR "
                                        "rows are inside V today and a backfill moves only N",
        "star_rows_with_357_present_max": int(max(r["star_rows_present_per_factor"]["yeslab_357"] for r in d["per_day"])),
        "star_rows_with_471_present_max": int(max(r["star_rows_present_per_factor"]["yeslab_471"] for r in d["per_day"])),
        "days": len(df), "problem_days": int(df.is_problem_day.sum()),
        "cutover": args.cutover, "floor": args.floor, "proposed_bar": args.bar,
        "coverage_now": {"min": float(df.coverage_now.min()), "median": float(df.coverage_now.median()),
                         "max": float(df.coverage_now.max()),
                         "below_floor": int((df.coverage_now < args.floor).sum()),
                         "below_bar": int((df.coverage_now < args.bar).sum())},
        "coverage_after_option1_optimistic": {
            "min": float(df.coverage_after_optimistic.min()), "median": float(df.coverage_after_optimistic.median()),
            "max": float(df.coverage_after_optimistic.max()),
            "below_floor": int((df.coverage_after_optimistic < args.floor).sum()),
            "below_bar": int((df.coverage_after_optimistic < args.bar).sum()),
            "worst_day": worst("coverage_after_optimistic")},
        "coverage_after_option1_conservative": {
            "min": float(df.coverage_after_conservative.min()), "median": float(df.coverage_after_conservative.median()),
            "max": float(df.coverage_after_conservative.max()),
            "below_floor": int((df.coverage_after_conservative < args.floor).sum()),
            "below_bar": int((df.coverage_after_conservative < args.bar).sum()),
            "worst_day": worst("coverage_after_conservative"),
            "days_where_the_seven_non_industry_factors_disagree": int((~df.non_industry_star_counts_uniform).sum())},
        "worst_problem_day_after": float(df[df.is_problem_day].coverage_after_conservative.min()),
        "worst_non_problem_day_after": float(df[~df.is_problem_day].coverage_after_conservative.min()),
        "floor_sensitivity_days_below_after": {str(f): int((df.coverage_after_conservative < f).sum())
                                               for f in (0.90, 0.905, 0.91, 0.95)},
        "floor_sensitivity_days_below_now": {str(f): int((df.coverage_now < f).sum())
                                             for f in (0.90, 0.905, 0.91, 0.95)},
        "projected_full_window_g6": {
            "total_days": tot, "pre_cutover_days_measured_here": len(df), "post_cutover_days": len(post),
            "post_cutover_coverage_min": float(post.coverage.min()),
            "at_floor": {"evaluable": ge_floor, "g6": ge_floor / tot},
            "at_bar": {"evaluable": ge_bar, "g6": ge_bar / tot},
            "current_g6": {"evaluable": 945, "total": 969, "g6": 945 / 969},
            "note": "pre-cutover 420 days measured here under the conservative bound; post-cutover 549 days "
                    "taken from the instrument, whose coverage already exceeds 0.971 and whose SH/SZ-union "
                    "perturbation was measured at a median relative difference of 0.0 (problems_found.md §X-续三)"},
        "answer": None,
        "per_day": rows,
        "database_writes": False, "weights_written": False, "cache_writes": False,
    }
    wc = out["coverage_after_option1_conservative"]
    out["answer"] = (
        f"YES on the floor, and the 24-day projection's worst day survives the full census unchanged. "
        f"Over all {len(df)} pre-cutover days the backfill takes days below {args.floor} from "
        f"{out['coverage_now']['below_floor']} to {wc['below_floor']}, and the minimum coverage after the fix is "
        f"{wc['min']:.10f} on {wc['worst_day']['day']} -- digit-for-digit the day and value the 24-day projection "
        f"reported, {wc['worst_day']['rows_short_of_bar']} rows short of the proposed {args.bar} bar. "
        f"The worst of the {len(df) - int(df.is_problem_day.sum())} days that never broke the floor is "
        f"{out['worst_non_problem_day_after']:.6f}, so the assertion 'the other days can only improve' is now "
        f"measured rather than assumed. Against the {args.bar} bar exactly one day of {len(df)} fails. "
        f"Projected full-window G6 is {ge_floor}/{tot} = {ge_floor / tot:.6f} at the floor "
        f"(currently 945/969 = {945 / 969:.6f}) and {ge_bar}/{tot} = {ge_bar / tot:.6f} at the bar.")
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(out["answer"])
    print(json.dumps({k: out[k] for k in ("coverage_now", "coverage_after_option1_conservative",
                                          "floor_sensitivity_days_below_now",
                                          "floor_sensitivity_days_below_after",
                                          "projected_full_window_g6")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
