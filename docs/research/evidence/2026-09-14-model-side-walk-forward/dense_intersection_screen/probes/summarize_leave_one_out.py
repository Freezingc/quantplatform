#!/usr/bin/env python3
"""What does removing each dense factor actually buy? Read the leave-one-out runs side by side, all 969 days.

The decision sheet's option 2 is "move yeslab_357 and yeslab_471 out of the dense subset". A cheaper
variant sounds plausible -- move only one of them out and keep more signal -- and problems_found.md
§AC/§AD implied that the ~0.974 coverage ceiling is set by yeslab_206's complete-window requirement,
which would suggest that dropping 206 lifts it. Both implications are testable against runs that already
exist in this directory and had never been read together: leave_out_357, leave_out_471, leave_out_both,
and leave_out_drop_<factor> for each of the other seven. Each covers all 969 OOS days.

This joins them against the nine-factor baseline and reports, per removal, whether the per-day coverage
and intersection row counts are IDENTICAL to the baseline (not merely similar), the gain distribution,
and the counts below 0.90 / 0.95 / 0.974. Identity is checked element-wise over all 969 days so that
"buys nothing" is a measurement rather than a reading of the summary block.

Pure post-processing over existing measurement files: no database, no cache, no repository import.

Usage:
  python summarize_leave_one_out.py \
      --baseline measurements/predicted_coverage_explicit_factor_list_2022_2025.json \
      --dir measurements --out measurements/leave_one_out_full_window.json
"""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import pandas as pd


def load(path: Path) -> pd.DataFrame:
    d = pd.DataFrame(json.loads(path.read_text())["per_day"])
    d["day"] = d["day"].astype(str)
    d["coverage"] = d["coverage"].astype(float)
    return d.sort_values("day").reset_index(drop=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=Path, required=True)
    ap.add_argument("--dir", type=Path, required=True)
    ap.add_argument("--cutover", default="2023-09-25")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    base = load(args.baseline)
    files = sorted(glob.glob(str(args.dir / "leave_out_*.json")))
    if not files:
        raise SystemExit(f"no leave_out_*.json under {args.dir}")

    res = []
    for f in files:
        d = load(Path(f))
        if len(d) != len(base) or not (d.day.values == base.day.values).all():
            raise SystemExit(f"{f} does not cover the same days as the baseline; refusing to compare")
        name = (Path(f).name.replace("leave_out_", "").replace("drop_", "")
                .replace("_2022_2025.json", "").replace(".json", ""))
        g = pd.Series(d.intersection_rows.values - base.intersection_rows.values)
        pre = (d.day < args.cutover).values
        res.append({
            "removed": name,
            "identical_coverage_to_baseline": bool((d.coverage.values == base.coverage.values).all()),
            "identical_intersection_rows_to_baseline": bool(
                (d.intersection_rows.values == base.intersection_rows.values).all()),
            "gain_rows_min": int(g.min()), "gain_rows_median": float(g.median()), "gain_rows_max": int(g.max()),
            "gain_rows_median_pre_cutover": float(g[pre].median()),
            "gain_rows_median_post_cutover": float(g[~pre].median()),
            "coverage_min": float(d.coverage.min()),
            "coverage_min_day": str(d.loc[d.coverage.idxmin(), "day"]),
            "coverage_median": float(d.coverage.median()),
            "days_below_0_90": int((d.coverage < 0.90).sum()),
            "days_below_0_95": int((d.coverage < 0.95).sum()),
            "days_below_0_974": int((d.coverage < 0.974).sum()),
        })

    df = pd.DataFrame(res)
    pre_b = (base.day < args.cutover).values
    out = {
        "question": "What does removing each dense factor actually buy, measured over all 969 OOS days?",
        "cutover": args.cutover, "days": len(base),
        "baseline_nine_factors": {
            "coverage_min": float(base.coverage.min()),
            "coverage_min_day": str(base.loc[base.coverage.idxmin(), "day"]),
            "coverage_median": float(base.coverage.median()),
            "days_below_0_90": int((base.coverage < 0.90).sum()),
            "days_below_0_95": int((base.coverage < 0.95).sum()),
            "days_below_0_974": int((base.coverage < 0.974).sum()),
            "days_below_0_974_pre_cutover": int(((base.coverage < 0.974).values & pre_b).sum()),
            "days_below_0_974_post_cutover": int(((base.coverage < 0.974).values & ~pre_b).sum()),
        },
        "per_removal": res,
        "removals_that_are_strictly_no_ops": sorted(
            df.loc[df.identical_coverage_to_baseline & df.identical_intersection_rows_to_baseline,
                   "removed"].tolist()),
        "answer": None,
        "database_writes": False, "weights_written": False, "cache_writes": False,
    }
    noop = out["removals_that_are_strictly_no_ops"]
    both = next((r for r in res if r["removed"] == "both"), None)
    d206 = next((r for r in res if r["removed"] == "yeslab_206"), None)
    parts = [f"{len(noop)} of the {len(res)} removals are strict no-ops -- identical per-day coverage AND "
             f"identical intersection row counts on all {len(base)} days: {', '.join(noop)}"]
    if d206:
        parts.append(f"removing yeslab_206 alone adds a median of {d206['gain_rows_median']:.0f} rows "
                     f"(max {d206['gain_rows_max']}) and still leaves {d206['days_below_0_90']} days "
                     f"below the floor")
    if both:
        parts.append(f"removing BOTH industry factors is the only removal that clears it "
                     f"({both['days_below_0_90']} days below 0.90, {both['days_below_0_95']} below 0.95, "
                     f"coverage min {both['coverage_min']:.6f} on {both['coverage_min_day']}), and its gain "
                     f"is bimodal -- median {both['gain_rows_median']:.0f} row but max "
                     f"{both['gain_rows_max']}, adding {both['gain_rows_median_pre_cutover']:.0f} rows "
                     f"pre-cutover and {both['gain_rows_median_post_cutover']:.0f} after")
    out["answer"] = ". ".join(parts) + "."
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(out["answer"])
    pd.set_option("display.width", 300)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
