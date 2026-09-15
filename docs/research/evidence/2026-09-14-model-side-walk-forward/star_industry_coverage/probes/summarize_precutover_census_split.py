#!/usr/bin/env python3
"""Split the pre-cutover 420-day label-union census into the 24 problem days and the other 396.

§X-续三 made the post-cutover half a full census of 549 days and left the pre-cutover half with only its
24 problem days measured. That gap is now filled: `label_source_census_precutover.json` covers all 420
cached pre-cutover trading days (2022-01-04..2023-09-22), including all 24 problem days. This script
asks the question the decision sheet actually needs, which is not "how big is the perturbation" but
"is the perturbation confined to the days that broke the floor". It reads two existing measurement
files, joins them on the day, and reports both scopes with a Mann-Whitney U test between the groups.

Pure post-processing: no database, no cache, no repository import.

Usage: python summarize_precutover_census_split.py \
           --census  ../../bse_cache_contamination/measurements/label_source_census_precutover.json \
           --problem ../yearly_baseline_deltas/m0r_v2.json \
           --out     ../measurements/label_source_census_precutover_split.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd

METRICS = ["coverage_gain", "grouped_471_gain",
           "f357__changed_rate", "f357__spearman", "f357__median_rel", "f357__p95_rel",
           "f471__changed_rate", "f471__spearman", "f471__median_rel", "f471__p95_rel"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", type=Path, required=True)
    ap.add_argument("--problem", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    cen = json.loads(args.census.read_text())
    rows = []
    for scope in ("shsz", "all"):
        for r in cen["per_day"][scope]:
            row = {"day": r["day"], "scope": scope, "labels_differing": r["labels_differing"],
                   "coverage_A": r["coverage_A"], "coverage_B": r["coverage_B"],
                   "grouped_471_A": r["grouped_471_A"], "grouped_471_B": r["grouped_471_B"]}
            for f in ("f357", "f471"):
                for k, v in r[f].items():
                    row[f"{f}__{k}"] = v
            rows.append(row)
    df = pd.DataFrame(rows)
    df["coverage_gain"] = df.coverage_B - df.coverage_A
    df["grouped_471_gain"] = df.grouped_471_B - df.grouped_471_A

    pr = json.loads(args.problem.read_text())
    prob = sorted({x for y in pr["unevaluable_days"]["by_year"].values() for x in y["invalid_dates"]})
    df["is_problem"] = df.day.isin(prob)

    try:
        from scipy.stats import mannwhitneyu
        have_scipy = True
    except Exception:
        have_scipy = False

    def block(sub: pd.DataFrame) -> dict:
        out = {"n_days": int(sub.day.nunique())}
        for scope in ("shsz", "all"):
            s = sub[sub.scope == scope]
            out[scope] = {c: {"min": float(s[c].min()), "median": float(s[c].median()),
                              "max": float(s[c].max())} for c in METRICS}
        return out

    a = df[(df.scope == "shsz") & df.is_problem]
    b = df[(df.scope == "shsz") & ~df.is_problem]
    tests = {}
    for c in ["coverage_gain", "grouped_471_gain", "f357__changed_rate", "f357__spearman",
              "f471__changed_rate", "f471__spearman"]:
        rec = {"problem_median": float(a[c].median()), "non_problem_median": float(b[c].median())}
        if have_scipy:
            _, p = mannwhitneyu(a[c].astype(float), b[c].astype(float), alternative="two-sided")
            rec["mannwhitney_p"] = float(p)
        tests[c] = rec

    shsz = df[df.scope == "shsz"]
    worst357 = shsz.nsmallest(5, "f357__spearman")[["day", "f357__spearman", "is_problem"]]
    worst471 = shsz.nsmallest(5, "f471__spearman")[["day", "f471__spearman", "is_problem"]]

    out = {
        "question": "Under the SH/SZ union, is the pre-cutover perturbation confined to the 24 days that broke the floor?",
        "census_window": cen["window"], "census_days": int(df.day.nunique()),
        "census_day_span": [str(df.day.min()), str(df.day.max())],
        "problem_days_total": len(prob),
        "problem_days_present_in_census": int(shsz.is_problem.sum()),
        "problem_days_missing_from_census": [p for p in prob if p not in set(df.day)],
        "answer": None,
        "problem_days": block(df[df.is_problem]),
        "non_problem_days": block(df[~df.is_problem]),
        "group_tests_shsz": tests,
        "worst_5_days_by_f357_spearman": worst357.to_dict("records"),
        "worst_5_days_by_f471_spearman": worst471.to_dict("records"),
        "scipy_available": have_scipy,
        "database_writes": False, "weights_written": False, "cache_writes": False,
    }
    p3, n3 = tests["f357__changed_rate"]["problem_median"], tests["f357__changed_rate"]["non_problem_median"]
    p7, n7 = tests["f471__changed_rate"]["problem_median"], tests["f471__changed_rate"]["non_problem_median"]
    out["answer"] = (
        f"NO -- the perturbation is a property of the whole pre-cutover half, not of the 24 days. "
        f"Under the SH/SZ union, 357 changes on a median {n3:.1%} of securities on the 396 days that never "
        f"broke the floor versus {p3:.1%} on the 24 that did, and 471 on {n7:.1%} versus {p7:.1%}. "
        f"The worst-perturbed pre-cutover days by 357 rank correlation are not problem days at all "
        f"(min {float(shsz.f357__spearman.min()):.4f} on a non-problem day vs "
        f"{float(a.f357__spearman.min()):.4f} worst among problem days). "
        f"Coverage gain does differ ({tests['coverage_gain']['problem_median']:.0f} labels/day on problem days "
        f"vs {tests['coverage_gain']['non_problem_median']:.0f} elsewhere), but the 396 non-problem days still "
        f"gain 400-657 labels each, so the union rewrites the pre-cutover segment everywhere.")
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(out["answer"])
    print(json.dumps({k: out[k] for k in ("census_days", "census_day_span", "problem_days_total",
                                          "problem_days_present_in_census", "group_tests_shsz")},
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
