#!/usr/bin/env python3
"""Read-only probe: is the recovered-row predictability difference between folds real, or a few outliers?

§BT.6 measured the rows that lowering minimum_valid_history_length from 15 to 8 admits, and found their
mean within-day standardized score-label product to be +0.247711 in F1 (4.09x the rows already there) and
-0.196191 in F2 (-3.71x), with the four folds disagreeing in direction and all four Mann-Whitney tests
non-significant at p = 0.148, 0.805, 0.468 and 0.841. §BT.8 left two things open: why the sign flips
between folds, and it guessed listing age.

The companion probe rules the guess out -- the newly admitted rows have mean observed_count 10.99 in F1
and 11.00 in F2, with near-uniform distributions over 8..14 in both. So this probe asks the remaining
question properly.

Two things made the §BT.6 numbers hard to read, and both are addressed here. First, Mann-Whitney tests
stochastic dominance, not means, so a group can differ 4-fold in mean and still show no distributional
difference -- which is what a heavy-tailed group does. The pooled median across folds was -0.000916
against a pooled mean of +0.037519, which is already the signature of a few extreme rows carrying the
mean. Second, no test compared the folds' recovered groups against EACH OTHER, which is the comparison
the sign flip actually claims. So this probe reports mean, median, standard deviation and quantiles per
group, adds Welch t-tests on the means to sit beside the rank tests, and tests F1's recovered rows
against F2's directly.

Safety: reads parquet artifacts in place. No database access, no weight loading, no training, no
repository module imported or executed, nothing written into any artifact directory, no process signalled.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
FOLD_DATASET = {
    "F1": "226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538",
    "F2": "eb9bba1b5a0bd66d0ad6c82536b295de1fc079aec125cb27bb68491bdb21a234",
    "F3": "9766e69f342511c5bda30188df15a7b9b17d936bdc26b9b60ba244e8e7e0c9e7",
    "F4": "6f9478d7645c8c745a73f00c03b6122e17d0c06c05cc159ece531fb00df9fb9d",
}
LABEL_WORKSPACE = "m1_full_v1"
KEYS = ["signal_date", "ts_code"]
HISTORY_LENGTH, K_FROZEN, K_LOWERED = 20, 15, 8


def load_panel(fold: str) -> pd.DataFrame:
    parts = []
    for split in ("train", "validation", "test"):
        paths = sorted(glob.glob(str(ART / LABEL_WORKSPACE / "datasets" / FOLD_DATASET[fold]
                                     / f"split={split}" / "*" / "*.parquet")))
        f = pd.concat([pd.read_parquet(p, columns=["signal_date", "ts_code", "raw_return_5d", "label_valid"])
                       for p in paths], ignore_index=True)
        f["split"] = split
        parts.append(f)
    p = pd.concat(parts, ignore_index=True)
    p["signal_date"] = pd.to_datetime(p["signal_date"], errors="raise").dt.date
    p["label_valid"] = p["label_valid"].fillna(False).astype(bool)
    p["raw_return_5d"] = pd.to_numeric(p["raw_return_5d"], errors="coerce")
    p = p.sort_values(["ts_code", "signal_date"], kind="mergesort").reset_index(drop=True)
    _, code_start = np.unique(p["ts_code"].to_numpy(), return_index=True)
    code_start = np.asarray(code_start, dtype=np.int64)
    runs = np.append(code_start[1:], len(p)) - code_start
    assert (runs > 0).all()
    ids = np.repeat(np.arange(len(code_start), dtype=np.int64), runs)
    p["observed_count"] = np.arange(len(p), dtype=np.int64) - code_start[ids] + 1
    p["valid_row"] = p["label_valid"] & np.isfinite(p["raw_return_5d"])
    return p


def describe(x: pd.Series) -> dict:
    x = x.dropna()
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()),
            "sd": float(x.std(ddof=1)), "sem": float(x.std(ddof=1) / math.sqrt(len(x))),
            "min": float(x.min()), "p01": float(x.quantile(0.01)), "p25": float(x.quantile(0.25)),
            "p75": float(x.quantile(0.75)), "p99": float(x.quantile(0.99)), "max": float(x.max()),
            "mean_to_median_gap": float(x.mean() - x.median()),
            "mean_over_sem": float(x.mean() / (x.std(ddof=1) / math.sqrt(len(x)))) if x.std(ddof=1) > 0 else None,
            "share_of_abs_sum_from_the_top_1pct": float(
                x.abs().nlargest(max(1, int(0.01 * len(x)))).sum() / x.abs().sum()) if len(x) else None}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "k8_model_effect")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    per_fold, recovered_by_fold = {}, {}
    for fold in ("F1", "F2", "F3", "F4"):
        p = load_panel(fold)
        test = p.loc[p["split"] == "test", ["signal_date", "ts_code", "raw_return_5d",
                                            "valid_row", "observed_count"]].reset_index(drop=True)
        base = pd.read_parquet(ART / "baseline_v2" / "baseline_oos" / f"{fold}.parquet",
                               columns=["signal_date", "ts_code", "score"])
        base["signal_date"] = pd.to_datetime(base["signal_date"], errors="raise").dt.date
        base["score"] = pd.to_numeric(base["score"], errors="coerce")
        j = test.merge(base, on=KEYS, how="left", validate="one_to_one")
        j["in_baseline"] = j["score"].notna()
        elig15 = j["valid_row"] & (np.minimum(HISTORY_LENGTH, j["observed_count"]) >= K_FROZEN)
        elig8 = j["valid_row"] & (np.minimum(HISTORY_LENGTH, j["observed_count"]) >= K_LOWERED)
        m15 = (elig15 & j["in_baseline"]).to_numpy()
        m8 = (elig8 & j["in_baseline"]).to_numpy()
        rec = m8 & ~m15

        pool = j.loc[m8, ["signal_date", "score", "raw_return_5d"]].reset_index(drop=True)
        pool["recovered"] = rec[np.flatnonzero(m8)]

        def _z(x):
            sd = float(x.std(ddof=0))
            return (x - x.mean()) / sd if sd > 0 else pd.Series(np.nan, index=x.index)
        g = pool.groupby("signal_date")
        pool["product"] = g["score"].transform(_z) * g["raw_return_5d"].transform(_z)

        rr = pool.loc[pool["recovered"], "product"].dropna()
        rt = pool.loc[~pool["recovered"], "product"].dropna()
        recovered_by_fold[fold] = rr
        mw = stats.mannwhitneyu(rr, rt, alternative="two-sided")
        tt = stats.ttest_ind(rr, rt, equal_var=False)
        per_fold[fold] = {
            "recovered": describe(rr), "retained": describe(rt),
            "recovered_vs_retained": {
                "mannwhitney_U": float(mw.statistic), "mannwhitney_p": float(mw.pvalue),
                "welch_t": float(tt.statistic), "welch_p": float(tt.pvalue),
                "mean_difference": float(rr.mean() - rt.mean()),
                "mean_difference_in_recovered_SEM": float((rr.mean() - rt.mean()) /
                                                          (rr.std(ddof=1) / math.sqrt(len(rr)))),
                "reading": ("Mann-Whitney tests stochastic dominance and Welch tests the means; when the "
                            "two disagree the group is heavy-tailed and the mean is being carried by a "
                            "few rows"),
            },
            "recovered_observed_count_mean": float(j.loc[rec, "observed_count"].mean()),
            "recovered_observed_count_sd": float(j.loc[rec, "observed_count"].std(ddof=1)),
        }
        d, c = per_fold[fold]["recovered"], per_fold[fold]["recovered_vs_retained"]
        print(f"[{fold}] recovered n={d['n']:>5} mean {d['mean']:+.6f} median {d['median']:+.6f} "
              f"sd {d['sd']:.4f} sem {d['sem']:.4f} mean/sem {d['mean_over_sem']:+.2f}", flush=True)
        print(f"       mean-median gap {d['mean_to_median_gap']:+.6f}; top 1% of |product| carries "
              f"{100*d['share_of_abs_sum_from_the_top_1pct']:.1f}% of the absolute sum", flush=True)
        print(f"       vs retained: mean diff {c['mean_difference']:+.6f} "
              f"({c['mean_difference_in_recovered_SEM']:+.2f} SEM) | Welch p={c['welch_p']:.3g} | "
              f"Mann-Whitney p={c['mannwhitney_p']:.3g}", flush=True)
        del p, j, pool

    # the comparison §BT.6 never made: F1's recovered rows against F2's
    pairs = {}
    for a, b in (("F1", "F2"), ("F1", "F3"), ("F1", "F4"), ("F2", "F3"), ("F2", "F4"), ("F3", "F4")):
        x, y = recovered_by_fold[a], recovered_by_fold[b]
        mw = stats.mannwhitneyu(x, y, alternative="two-sided")
        tt = stats.ttest_ind(x, y, equal_var=False)
        pairs[f"{a}_vs_{b}"] = {
            "mean_a": float(x.mean()), "mean_b": float(y.mean()),
            "mean_difference": float(x.mean() - y.mean()),
            "median_a": float(x.median()), "median_b": float(y.median()),
            "welch_t": float(tt.statistic), "welch_p": float(tt.pvalue),
            "mannwhitney_U": float(mw.statistic), "mannwhitney_p": float(mw.pvalue),
            "n_a": int(len(x)), "n_b": int(len(y)),
        }
    allrec = pd.concat(recovered_by_fold.values())
    result = {
        "question": ("§BT.6 found the rows that k=8 admits to be strongly predictive in F1 and "
                     "anti-predictive in F2 with no fold's rank test significant, and §BT.8 guessed "
                     "listing age. The companion probe rules listing age out. Is the sign flip a real "
                     "fold difference or a few extreme rows?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False, "training_performed": False,
        "repository_module_imported_or_executed": False,
        "listing_age_hypothesis": {
            "status": "REFUTED by probe_k8_window_masking.py",
            "evidence": {"test_side_mean_observed_count": {f: per_fold[f]["recovered_observed_count_mean"]
                                                           for f in per_fold},
                         "test_side_sd_observed_count": {f: per_fold[f]["recovered_observed_count_sd"]
                                                         for f in per_fold},
                         "note": ("observed_count IS listing age in panel rows; F1 and F2 are identical to "
                                  "two decimals, and the companion probe shows near-uniform distributions "
                                  "over 8..14 in every fold, so the sign flip is not an age effect")},
        },
        "per_fold": per_fold,
        "fold_to_fold_comparisons_of_the_recovered_groups": pairs,
        "pooled_recovered": describe(allrec),
        "verdict": {
            "is_the_mean_difference_within_fold_significant": {
                f: per_fold[f]["recovered_vs_retained"]["welch_p"] < 0.05 for f in per_fold},
            "is_the_distribution_shifted_within_fold": {
                f: per_fold[f]["recovered_vs_retained"]["mannwhitney_p"] < 0.05 for f in per_fold},
            "is_the_F1_vs_F2_sign_flip_significant": pairs["F1_vs_F2"]["welch_p"] < 0.05,
            "F1_vs_F2": pairs["F1_vs_F2"],
        },
    }
    out = args.out / "recovered_row_predictability.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out}")
    print("\n=== F1 vs F2 recovered groups, directly ===")
    v = pairs["F1_vs_F2"]
    print(f"  F1 mean {v['mean_a']:+.6f} (median {v['median_a']:+.6f}, n={v['n_a']})")
    print(f"  F2 mean {v['mean_b']:+.6f} (median {v['median_b']:+.6f}, n={v['n_b']})")
    print(f"  difference {v['mean_difference']:+.6f}  Welch p={v['welch_p']:.4g}  "
          f"Mann-Whitney p={v['mannwhitney_p']:.4g}")
    print("\n=== per-fold significance ===")
    print(f"  {'fold':<6}{'mean-diff sig (Welch)':>24}{'distribution shifted (MW)':>28}")
    for f in ("F1", "F2", "F3", "F4"):
        print(f"  {f:<6}{str(result['verdict']['is_the_mean_difference_within_fold_significant'][f]):>24}"
              f"{str(result['verdict']['is_the_distribution_shifted_within_fold'][f]):>28}")
    pr = result["pooled_recovered"]
    print(f"\npooled recovered: n={pr['n']} mean {pr['mean']:+.6f} median {pr['median']:+.6f} "
          f"sd {pr['sd']:.4f} p01 {pr['p01']:+.4f} p99 {pr['p99']:+.4f} "
          f"top1% carries {100*pr['share_of_abs_sum_from_the_top_1pct']:.1f}% of |sum|")


if __name__ == "__main__":
    main()
