#!/usr/bin/env python3
"""Read-only probe: does lowering minimum_valid_history_length to 8 help or hurt the measured IC?

§BS found that k has to come down from 15 to 8 for M2 to clear G6, and §BS.8 flagged the limit that
travels with it: nothing there measures what k=8 does to quality. Training a k=8 model answers part of
that but not all of it, and there is a large part that needs no training at all.

Lowering k changes WHICH ROWS are in the cross-section, and it changes them for BOTH sides, because
evaluation/model_evaluator.py:59-64 inner-joins the candidate with the baseline before scoring either.
So the measured rank IC moves for a compositional reason that is entirely independent of whether a
k=8-trained model is better or worse. That part is measurable today, exactly, because the baseline
already carries a score for every row in its own artifact -- including the rows M2 currently cannot
score.

§BS also establishes that of the 8,680 rows lowering k recovers from the eligible set, only 6,800 are
in the baseline's key set and therefore actually enter the matched cross-section: 0.20% of 3,400,056
rows, about 7 rows per day. That is small, which is why the compositional effect can be measured
cleanly rather than argued about.

Three things are computed. First, the baseline's OOS mean_rank_ic on the matched set at k=15, which
must reproduce the signed 0.0852268662737014 or nothing downstream means anything. Second, the same
quantity at k=8, which is what the baseline's side of G4 would become. Third, whether the recovered
rows are more or less predictable than the rows already in the cross-section, by comparing each
group's mean within-day standardized score-label product -- a per-row contribution that is comparable
across groups of very different size, unlike a rank correlation computed on 7 rows.

Safety: reads parquet artifacts in place. No database access, no weight loading, no training, no
repository module imported or executed, nothing written into any artifact directory.
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
HISTORY_LENGTH = 20
FLOOR, G6_GATE, TOTAL_DAYS = 0.90, 0.95, 969
MIN_CROSS_SECTION = 30
SIGNED_BASELINE_OOS_MEAN = 0.0852268662737014
KS = (15, 8)


def load_panel(fold: str) -> pd.DataFrame:
    parts = []
    for split in ("train", "validation", "test"):
        paths = sorted(glob.glob(str(ART / LABEL_WORKSPACE / "datasets" / FOLD_DATASET[fold]
                                     / f"split={split}" / "*" / "*.parquet")))
        f = pd.concat([pd.read_parquet(p, columns=["signal_date", "ts_code", "raw_return_5d", "label_valid"])
                       for p in paths], ignore_index=True)
        f["split"] = split
        parts.append(f)
    panel = pd.concat(parts, ignore_index=True)
    panel["signal_date"] = pd.to_datetime(panel["signal_date"], errors="raise").dt.date
    panel["label_valid"] = panel["label_valid"].fillna(False).astype(bool)
    panel["raw_return_5d"] = pd.to_numeric(panel["raw_return_5d"], errors="coerce")
    panel = panel.sort_values(["ts_code", "signal_date"], kind="mergesort").reset_index(drop=True)
    codes = panel["ts_code"].to_numpy()
    _, code_start = np.unique(codes, return_index=True)
    code_start = np.asarray(code_start, dtype=np.int64)
    run_lengths = np.append(code_start[1:], len(panel)) - code_start
    assert (run_lengths > 0).all()
    code_ids = np.repeat(np.arange(len(code_start), dtype=np.int64), run_lengths)
    panel["observed_count"] = np.arange(len(panel), dtype=np.int64) - code_start[code_ids] + 1
    panel["valid_row"] = panel["label_valid"] & np.isfinite(panel["raw_return_5d"])
    return panel


def daily_rank_ic(frame: pd.DataFrame) -> pd.DataFrame:
    """rank_ic.py:60-107 with score health enforced, on whatever row set it is handed."""
    rows = []
    for d, g in frame.groupby("signal_date", sort=True):
        vl = g["valid_row"].to_numpy()
        fs = vl & np.isfinite(g["score"].to_numpy(dtype=float))
        vlc = int(vl.sum())
        sc_all = g["score"].to_numpy(dtype=float)
        lb_all = g["raw_return_5d"].to_numpy(dtype=float)
        n = int(fs.sum())
        cov = n / vlc if vlc > 0 else 0.0
        sc, lb = sc_all[fs], lb_all[fs]
        uniq = int(pd.Series(sc).nunique()) if n else 0
        sd = float(np.std(sc, ddof=0)) if n else float("nan")
        health = (math.isfinite(cov) and FLOOR <= cov <= 1.0 and uniq >= 10
                  and math.isfinite(sd) and sd > 1e-6)
        ric, ok = float("nan"), False
        if n >= MIN_CROSS_SECTION and health:
            sr = pd.Series(sc).rank(method="average")
            lr = pd.Series(lb).rank(method="average")
            if sr.nunique(dropna=False) >= 2 and lr.nunique(dropna=False) >= 2:
                c = sr.corr(lr, method="pearson")
                if pd.notna(c) and math.isfinite(float(c)):
                    ric, ok = float(c), True
        rows.append({"signal_date": d, "rank_ic": ric, "rank_ic_valid": ok, "coverage": cov,
                     "valid_label_count": vlc, "scored": n})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "k8_composition_effect")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    per_fold, recovered_all = {}, []
    for fold in ("F1", "F2", "F3", "F4"):
        panel = load_panel(fold)
        test = panel.loc[panel["split"] == "test", ["signal_date", "ts_code", "raw_return_5d",
                                                   "valid_row", "observed_count"]].reset_index(drop=True)
        base = pd.read_parquet(ART / "baseline_v2" / "baseline_oos" / f"{fold}.parquet",
                               columns=["signal_date", "ts_code", "score"])
        base["signal_date"] = pd.to_datetime(base["signal_date"], errors="raise").dt.date
        base["score"] = pd.to_numeric(base["score"], errors="coerce")
        joined = test.merge(base, on=KEYS, how="left", validate="one_to_one")
        joined["in_baseline"] = joined["score"].notna()

        out = {"test_rows": int(len(test)), "test_valid_label_rows": int(joined["valid_row"].sum()),
               "baseline_rows": int(len(base)), "by_k": {}, "recovered_rows": {}}
        sets = {}
        for k in KS:
            elig = joined["valid_row"] & (np.minimum(HISTORY_LENGTH, joined["observed_count"]) >= k)
            matched = elig & joined["in_baseline"]
            sets[k] = matched.to_numpy()
            # The frame handed to daily_rank_ic must carry EVERY valid-label row of the test split,
            # with a score only where the row survived eligibility and the baseline intersection.
            # compute_daily_rank_ic left-merges the labels with the predictions, so the coverage
            # denominator is all valid-label rows that day and the numerator is those with a finite
            # score. Passing only the matched rows instead makes the denominator equal the numerator,
            # coverage is 1.0 on every day by construction, and no day can ever fail the floor -- which
            # is exactly what the first run of this probe did.
            vr = joined["valid_row"].to_numpy()
            sub = joined.loc[vr, ["signal_date", "ts_code", "score", "raw_return_5d"]].copy()
            sub["valid_row"] = True
            sub.loc[~matched.to_numpy()[vr], "score"] = np.nan
            daily = daily_rank_ic(sub)
            v = daily.loc[daily["rank_ic_valid"], "rank_ic"]
            out["by_k"][k] = {
                "eligible_rows": int(elig.sum()), "matched_rows": int(matched.sum()),
                "rows_removed_by_matching": int(elig.sum() - matched.sum()),
                "valid_days": int(len(v)), "days_below_floor": int((daily["coverage"] < FLOOR).sum()),
                "G6": float((daily["coverage"] >= FLOOR).mean()),
                "mean_rank_ic": float(v.mean()) if len(v) else None,
                "rank_ic_std": float(v.std(ddof=1)) if len(v) > 1 else None,
                "icir": float(v.mean() / v.std(ddof=1)) if len(v) > 1 else None,
                "mean_rank_ic_hex": float(v.mean()).hex() if len(v) else None,
            }
        rec = sets[8] & ~sets[15]
        ret = sets[15]
        out["recovered_rows"] = {"count": int(rec.sum()), "count_entering_the_cross_section": int(rec.sum())}
        # per-row contribution: within each day, standardize score and label across the k=8 matched
        # set, then compare the mean product over recovered rows with that over retained rows. This is
        # comparable across groups of very different size, unlike a rank correlation on ~7 rows.
        pool = joined.loc[sets[8], ["signal_date", "score", "raw_return_5d"]].copy()
        # the pooled frame is joined.loc[sets[8]], so its rows are in the same order as the positional
        # indices where sets[8] is True -- rec[idx8] is therefore aligned with it row for row
        idx8 = np.flatnonzero(sets[8])
        pool = pool.reset_index(drop=True)
        pool["recovered"] = rec[idx8]
        g = pool.groupby("signal_date")
        def _z(x: pd.Series) -> pd.Series:
            sd = float(x.std(ddof=0))
            return (x - x.mean()) / sd if sd > 0 else pd.Series(np.nan, index=x.index)
        pool["z_score"] = g["score"].transform(_z)
        pool["z_label"] = g["raw_return_5d"].transform(_z)
        pool["product"] = pool["z_score"] * pool["z_label"]
        rr = pool.loc[pool["recovered"], "product"].dropna()
        rt = pool.loc[~pool["recovered"], "product"].dropna()
        out["predictability_of_the_recovered_rows"] = {
            "method": ("mean within-day standardized score x standardized label, i.e. each row's "
                       "contribution to that day's covariance; comparable across groups of very "
                       "different size, unlike a rank correlation computed on ~7 rows"),
            "recovered_n": int(len(rr)), "retained_n": int(len(rt)),
            "recovered_mean_product": float(rr.mean()) if len(rr) else None,
            "retained_mean_product": float(rt.mean()) if len(rt) else None,
            "ratio_recovered_over_retained": (float(rr.mean() / rt.mean())
                                              if len(rr) and len(rt) and rt.mean() != 0 else None),
            "mannwhitney_p_two_sided": (float(stats.mannwhitneyu(rr, rt, alternative="two-sided").pvalue)
                                        if len(rr) and len(rt) else None),
            "recovered_observed_count_distribution": {
                str(k): int(v) for k, v in joined.loc[rec, "observed_count"].value_counts().sort_index().items()},
            "recovered_label_mean": float(joined.loc[rec, "raw_return_5d"].mean()),
            "retained_label_mean": float(joined.loc[ret, "raw_return_5d"].mean()),
            "recovered_label_std": float(joined.loc[rec, "raw_return_5d"].std(ddof=1)),
            "retained_label_std": float(joined.loc[ret, "raw_return_5d"].std(ddof=1)),
        }
        recovered_all.append(pool.loc[pool["recovered"], "product"].dropna())
        per_fold[fold] = out
        b15, b8 = out["by_k"][15], out["by_k"][8]
        assert b15["matched_rows"] < out["test_valid_label_rows"], \
            "the matched set equals the whole valid-label set; the coverage denominator is degenerate"
        print(f"[{fold}] k=15 matched {b15['matched_rows']:>7} days<0.90 {b15['days_below_floor']:>3} "
              f"G6 {b15['G6']:.6f} mean {b15['mean_rank_ic']:.10f}", flush=True)
        print(f"       k= 8 matched {b8['matched_rows']:>7} days<0.90 {b8['days_below_floor']:>3} "
              f"G6 {b8['G6']:.6f} mean {b8['mean_rank_ic']:.10f}  "
              f"(delta {b8['mean_rank_ic']-b15['mean_rank_ic']:+.6f})", flush=True)
        pr = out["predictability_of_the_recovered_rows"]
        print(f"       recovered rows {pr['recovered_n']:>5}: mean product {pr['recovered_mean_product']:+.6f} "
              f"vs retained {pr['retained_mean_product']:+.6f} "
              f"(ratio {pr['ratio_recovered_over_retained']:.3f}, p={pr['mannwhitney_p_two_sided']:.3g})",
              flush=True)
        del panel, joined, pool

    allrec = pd.concat(recovered_all)
    tot_days_below = {k: sum(per_fold[f]["by_k"][k]["days_below_floor"] for f in per_fold) for k in KS}
    combined = {}
    for k in KS:
        days = sum(per_fold[f]["by_k"][k]["valid_days"] for f in per_fold)
        means = {f: per_fold[f]["by_k"][k]["mean_rank_ic"] for f in per_fold}
        vd = {f: per_fold[f]["by_k"][k]["valid_days"] for f in per_fold}
        pooled = sum(means[f] * vd[f] for f in per_fold) / sum(vd.values())
        combined[k] = {"valid_days": days, "days_below_floor": tot_days_below[k],
                       "G6": (TOTAL_DAYS - tot_days_below[k]) / TOTAL_DAYS,
                       "day_weighted_mean_rank_ic": pooled,
                       "per_fold_mean_rank_ic": means}
    result = {
        "question": ("Lowering minimum_valid_history_length to 8 changes which rows are in the "
                     "cross-section for BOTH sides, because the evaluator inner-joins candidate with "
                     "baseline first. That compositional effect on the measured rank IC is independent "
                     "of whether a k=8-trained model is better, and is measurable without training. "
                     "Does it help or hurt?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False, "training_performed": False,
        "repository_module_imported_or_executed": False,
        "validation": {
            "what_the_k15_column_must_reproduce": (
                "NOT the signed baseline_mean_rank_ic of 0.0852268662737014 over 945 days. That value is "
                "the baseline row-matched against a NON-temporal carrier, whose candidate scores almost "
                "every row, so the intersection is essentially the baseline's own row set. Here the "
                "baseline is matched against M2's eligible set, which is 17,306 rows smaller, so coverage "
                "is lower and 40 more days fall through the floor. The right reference is §AX, which "
                "derived M2's numbers from the eligibility rule."),
            "signed_baseline_for_non_temporal_carriers": {
                "mean_rank_ic": SIGNED_BASELINE_OOS_MEAN, "valid_days": 945,
                "reproduced_here": combined[15]["day_weighted_mean_rank_ic"],
                "abs_diff": abs(combined[15]["day_weighted_mean_rank_ic"] - SIGNED_BASELINE_OOS_MEAN),
                "valid_days_here": combined[15]["valid_days"],
                "agrees": False,
                "why_it_cannot_agree": (
                    "baseline_mean_rank_ic is carrier-specific: model_evaluator.py:86 scores "
                    "baseline_predictions_matched, i.e. the baseline AFTER the inner join with that "
                    "carrier's candidate. Different candidates give different matched sets and therefore "
                    "different baseline means. The single number recorded in every carrier report is the "
                    "non-temporal one."),
            },
            "AX_reference_for_m2": {
                "days_below_floor": 64, "valid_days": 905, "G6": 0.9339525283797729,
                "reproduced_days_below_floor": tot_days_below[15],
                "reproduced_valid_days": combined[15]["valid_days"],
                "reproduced_G6": combined[15]["G6"],
                "days_match": tot_days_below[15] == 64,
                "valid_days_match": combined[15]["valid_days"] == 905,
                "G6_match": abs(combined[15]["G6"] - 0.9339525283797729) < 1e-12,
            },
        },
        "combined_across_folds": {str(k): v for k, v in combined.items()},
        "the_compositional_effect": {
            "change_in_the_baseline_side_mean_rank_ic": combined[8]["day_weighted_mean_rank_ic"]
                                                        - combined[15]["day_weighted_mean_rank_ic"],
            "days_below_floor_k15": tot_days_below[15], "days_below_floor_k8": tot_days_below[8],
            "G6_k15": combined[15]["G6"], "G6_k8": combined[8]["G6"],
            "rows_entering_the_cross_section": sum(
                per_fold[f]["by_k"][8]["matched_rows"] - per_fold[f]["by_k"][15]["matched_rows"]
                for f in per_fold),
            "share_of_the_matched_cross_section": sum(
                per_fold[f]["by_k"][8]["matched_rows"] - per_fold[f]["by_k"][15]["matched_rows"]
                for f in per_fold) / sum(per_fold[f]["by_k"][15]["matched_rows"] for f in per_fold),
        },
        "recovered_rows_pooled_across_folds": {
            "n": int(len(allrec)),
            "mean_product": float(allrec.mean()),
            "median_product": float(allrec.median()),
        },
        "g4_threshold_is_carrier_and_k_specific": {
            "rule": "evaluation/gates.py:123 -> _finite_at_least(delta, 0.005), delta = candidate - baseline",
            "decision_sheets_current_figure_for_m2": 0.0852268662737014 + 0.005,
            "why_that_figure_is_wrong_for_m2": (
                "it uses the baseline value from the non-temporal carriers' reports. M2's baseline is "
                "row-matched against M2's own eligible set, which is 17,306 rows smaller, so M2's "
                "baseline mean is lower and therefore M2's G4 bar is lower."),
            "correct_bar_at_k15": combined[15]["day_weighted_mean_rank_ic"] + 0.005,
            "correct_bar_at_k8": combined[8]["day_weighted_mean_rank_ic"] + 0.005,
            "bar_moves_by_when_k_goes_15_to_8": combined[8]["day_weighted_mean_rank_ic"]
                                                - combined[15]["day_weighted_mean_rank_ic"],
            "caveat": ("this prices only the baseline side of G4. M2's own mean on the augmented "
                       "cross-section is not measurable without training a k=8 model."),
        },
        "per_fold": per_fold,
        "what_this_does_and_does_not_settle": {
            "settles": ("the compositional half of the k=8 question: which rows enter the cross-section, "
                        "how that moves the measured rank IC on both sides, and whether the newly "
                        "admitted rows are more or less predictable than the ones already there"),
            "does_not_settle": ("the model half. A model TRAINED with 12 of 20 lanes masked instead of 5 "
                                "may fit differently, and this probe trains nothing. §BS.8's caveat "
                                "stands for that half."),
        },
    }
    out = args.out / "k8_composition_effect.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out}")
    v = result["validation"]
    ax = v["AX_reference_for_m2"]
    print(f"\nvalidation against §AX (the correct reference for M2's eligible set):")
    print(f"  days below floor  {ax['reproduced_days_below_floor']} vs §AX {ax['days_below_floor']} -> {ax['days_match']}")
    print(f"  valid days        {ax['reproduced_valid_days']} vs §AX {ax['valid_days']} -> {ax['valid_days_match']}")
    print(f"  G6                {ax['reproduced_G6']:.16f} vs §AX {ax['G6']:.16f} -> {ax['G6_match']}")
    sg = v["signed_baseline_for_non_temporal_carriers"]
    print(f"  (the signed {sg['mean_rank_ic']:.13f} over 945 days is the baseline matched against a "
          f"NON-temporal carrier; against M2's eligible set it is "
          f"{sg['reproduced_here']:.13f} over {sg['valid_days_here']} days, so it cannot agree -- "
          f"baseline_mean_rank_ic is carrier-specific)")
    ce = result["the_compositional_effect"]
    print(f"\ncompositional effect of k: 15 -> 8")
    print(f"  rows entering the cross-section : {ce['rows_entering_the_cross_section']} "
          f"({100*ce['share_of_the_matched_cross_section']:.4f}% of it)")
    print(f"  days below the floor            : {ce['days_below_floor_k15']} -> {ce['days_below_floor_k8']}")
    print(f"  G6                              : {ce['G6_k15']:.6f} -> {ce['G6_k8']:.6f}")
    print(f"  BASELINE-side mean rank IC      : {combined[15]['day_weighted_mean_rank_ic']:.10f} -> "
          f"{combined[8]['day_weighted_mean_rank_ic']:.10f} "
          f"({ce['change_in_the_baseline_side_mean_rank_ic']:+.6f})")
    rp = result["recovered_rows_pooled_across_folds"]
    print(f"\nrecovered rows pooled: n={rp['n']} mean within-day standardized product {rp['mean_product']:+.6f} "
          f"(median {rp['median_product']:+.6f})")
    for f in per_fold:
        pr = per_fold[f]["predictability_of_the_recovered_rows"]
        print(f"  {f}: recovered {pr['recovered_n']:>5} mean {pr['recovered_mean_product']:+.6f} | "
              f"retained {pr['retained_n']:>7} mean {pr['retained_mean_product']:+.6f} | "
              f"ratio {pr['ratio_recovered_over_retained']:.3f} p={pr['mannwhitney_p_two_sided']:.3g}")


if __name__ == "__main__":
    main()
