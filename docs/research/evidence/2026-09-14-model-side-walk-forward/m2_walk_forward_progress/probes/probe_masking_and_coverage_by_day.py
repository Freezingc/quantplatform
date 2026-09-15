#!/usr/bin/env python3
"""Read-only probe: do the masking effect and the coverage effect of k=8 land on the same days?

§BU.2 measured the window-masking cost of lowering minimum_valid_history_length from 15 to 8 in
aggregate (training rows +0.938%, mean window length 19.9800 -> 19.8965, rows more than half masked
0.0000% -> 0.3985%) and §BU.6 listed the obvious follow-up it did not do: the same statistics per day,
to see whether the masking concentrates on particular dates, and whether those dates are the ones that
matter.

They might be, and that is the whole reason to look. §BS.4 found 43 days in F2 sitting within 0.005 above
the coverage floor at k=8, and §BT.5 found that admitting the recovered rows moves the floor crossings
from 64 days to 45 while moving the baseline's mean rank IC by only -0.000779. Those two facts sit
uncomfortably together unless the recovered rows are concentrated exactly where the coverage is thin --
about 7 rows a day on average is enough to flip a day whose coverage is within 0.0021 of the floor, but
nowhere near enough to move a day whose coverage is 0.99. If the recovered rows do concentrate on the
thin days, then the masking cost and the coverage benefit are the same rows on the same days, which is a
much sharper statement about path 2 than either measurement alone: the rows that buy the 19 flipped days
are precisely the rows the model has least history for.

This probe builds the per-day table for all four folds and tests the concentration directly, by
comparing the recovered-row count on the days that flipped against the days that did not, and by
correlating it with the coverage gain.

Safety: reads parquet artifacts in place. No database access, no weight loading, no training, no
repository module imported or executed, nothing written into any artifact directory, no process signalled.
"""
from __future__ import annotations

import argparse
import glob
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

pd.set_option("future.no_silent_downcasting", True)

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
FOLD_DATASET = {
    "F1": "226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538",
    "F2": "eb9bba1b5a0bd66d0ad6c82536b295de1fc079aec125cb27bb68491bdb21a234",
    "F3": "9766e69f342511c5bda30188df15a7b9b17d936bdc26b9b60ba244e8e7e0c9e7",
    "F4": "6f9478d7645c8c745a73f00c03b6122e17d0c06c05cc159ece531fb00df9fb9d",
}
LABEL_WORKSPACE = "m1_full_v1"
KEYS = ["signal_date", "ts_code"]
HISTORY_LENGTH, K_HI, K_LO = 20, 15, 8
FLOOR, NEAR_BAND = 0.90, 0.005


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
    _, cs = np.unique(p["ts_code"].to_numpy(), return_index=True)
    cs = np.asarray(cs, dtype=np.int64)
    runs = np.append(cs[1:], len(p)) - cs
    assert (runs > 0).all(), "code runs not contiguous; frame is not sorted by ts_code"
    ids = np.repeat(np.arange(len(cs), dtype=np.int64), runs)
    p["observed_count"] = np.arange(len(p), dtype=np.int64) - cs[ids] + 1
    p["valid_row"] = p["label_valid"] & np.isfinite(p["raw_return_5d"])
    p["length"] = np.minimum(HISTORY_LENGTH, p["observed_count"])
    p["masked_lanes"] = HISTORY_LENGTH - p["length"]
    return p


def build_day_table(fold: str) -> pd.DataFrame:
    p = load_panel(fold)
    t = p.loc[p["split"] == "test"].reset_index(drop=True)
    base = pd.read_parquet(ART / "baseline_v2" / "baseline_oos" / f"{fold}.parquet",
                           columns=["signal_date", "ts_code", "score"])
    base["signal_date"] = pd.to_datetime(base["signal_date"], errors="raise").dt.date
    t = t.merge(base.loc[:, KEYS].assign(in_baseline=True), on=KEYS, how="left")
    t["in_baseline"] = t["in_baseline"].fillna(False).astype(bool)
    oc = t["observed_count"].to_numpy()
    vr = t["valid_row"].to_numpy()
    ib = t["in_baseline"].to_numpy(dtype=bool)
    t["m15"] = vr & (oc >= K_HI) & ib
    t["m8"] = vr & (oc >= K_LO) & ib
    t["recovered"] = t["m8"].to_numpy() & ~t["m15"].to_numpy()

    vl = t[t["valid_row"]].groupby("signal_date").size().rename("valid_label_count")
    g = t.groupby("signal_date")
    day = pd.DataFrame({
        "valid_label_count": vl,
        "matched_k15": g["m15"].sum(),
        "matched_k8": g["m8"].sum(),
        "recovered": g["recovered"].sum(),
        "test_rows": g.size(),
    })
    day["cov_k15"] = day["matched_k15"] / day["valid_label_count"]
    day["cov_k8"] = day["matched_k8"] / day["valid_label_count"]
    day["cov_gain"] = day["cov_k8"] - day["cov_k15"]
    day["below_k15"] = day["cov_k15"] < FLOOR
    day["below_k8"] = day["cov_k8"] < FLOOR
    day["near_floor_k8"] = (day["cov_k8"] >= FLOOR) & (day["cov_k8"] < FLOOR + NEAR_BAND)
    day["flipped"] = day["below_k15"] & ~day["below_k8"]
    day["stayed_below"] = day["below_k15"] & day["below_k8"]
    # masking on the test side, per day, over the rows that are actually in the cross-section at k=8
    m8 = t[t["m8"]]
    day["mean_length_k8"] = m8.groupby("signal_date")["length"].mean()
    day["mean_masked_share_k8"] = m8.groupby("signal_date")["masked_lanes"].mean() / HISTORY_LENGTH
    day["share_over_half_masked_k8"] = m8.assign(h=m8["length"] <= HISTORY_LENGTH // 2) \
                                         .groupby("signal_date")["h"].mean()
    m15 = t[t["m15"]]
    day["mean_length_k15"] = m15.groupby("signal_date")["length"].mean()
    day["share_over_half_masked_k15"] = m15.assign(h=m15["length"] <= HISTORY_LENGTH // 2) \
                                           .groupby("signal_date")["h"].mean()
    rec = t[t["recovered"]]
    if len(rec):
        day["recovered_mean_observed_count"] = rec.groupby("signal_date")["observed_count"].mean()
    else:
        day["recovered_mean_observed_count"] = np.nan
    day["fold"] = fold
    return day.reset_index()


def summarise(day: pd.DataFrame) -> dict:
    out = {
        "days": int(len(day)),
        "days_below_floor_k15": int(day["below_k15"].sum()),
        "days_below_floor_k8": int(day["below_k8"].sum()),
        "days_flipped": int(day["flipped"].sum()),
        "days_stayed_below": int(day["stayed_below"].sum()),
        "days_near_floor_at_k8": int(day["near_floor_k8"].sum()),
        "recovered_rows_total": int(day["recovered"].sum()),
        "recovered_rows_per_day_mean": float(day["recovered"].mean()),
        "recovered_rows_per_day_max": int(day["recovered"].max()),
        "coverage_gain_mean": float(day["cov_gain"].mean()),
        "coverage_gain_max": float(day["cov_gain"].max()),
        "masking_concentration": {
            "share_over_half_masked_k15_mean": float(day["share_over_half_masked_k15"].mean()),
            "share_over_half_masked_k8_mean": float(day["share_over_half_masked_k8"].mean()),
            "share_over_half_masked_k8_max": float(day["share_over_half_masked_k8"].max()),
            "mean_length_k15_min_over_days": float(day["mean_length_k15"].min()),
            "mean_length_k8_min_over_days": float(day["mean_length_k8"].min()),
            "does_masking_concentrate_on_particular_days": bool(
                day["share_over_half_masked_k8"].max() > 3 * day["share_over_half_masked_k8"].mean()),
        },
    }
    fl, nf, rest = day[day["flipped"]], day[day["near_floor_k8"]], day[~day["flipped"]]
    out["are_recovered_rows_concentrated_where_coverage_is_thin"] = {
        "recovered_mean_on_flipped_days": float(fl["recovered"].mean()) if len(fl) else None,
        "recovered_mean_on_near_floor_days": float(nf["recovered"].mean()) if len(nf) else None,
        "recovered_mean_on_all_other_days": float(rest["recovered"].mean()) if len(rest) else None,
        "pearson_r_recovered_vs_coverage_gain": (
            float(np.corrcoef(np.asarray(day["recovered"], dtype=float),
                              np.asarray(day["cov_gain"], dtype=float))[0, 1])
            if float(np.asarray(day["cov_gain"], dtype=float).std()) > 0 else None),
        "spearman_rho_recovered_vs_cov_k8": float(stats.spearmanr(
            np.asarray(day["recovered"], dtype=float),
            np.asarray(day["cov_k8"], dtype=float)).statistic),
        "spearman_p": float(stats.spearmanr(
            np.asarray(day["recovered"], dtype=float),
            np.asarray(day["cov_k8"], dtype=float)).pvalue),
        "coverage_k15_on_flipped_days_min_max": [float(fl["cov_k15"].min()), float(fl["cov_k15"].max())]
                                                 if len(fl) else None,
        "coverage_k8_on_flipped_days_min_max": [float(fl["cov_k8"].min()), float(fl["cov_k8"].max())]
                                               if len(fl) else None,
        "reading": ("if the recovered rows were spread evenly, the flipped days would look like any "
                    "other day; concentration means the rows that buy the flipped days are the same "
                    "rows the model has least history for"),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "masking_and_coverage_by_day")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    per_fold, frames = {}, []
    for fold in ("F1", "F2", "F3", "F4"):
        day = build_day_table(fold)
        frames.append(day)
        per_fold[fold] = summarise(day)
        s = per_fold[fold]
        c = s["are_recovered_rows_concentrated_where_coverage_is_thin"]
        print(f"[{fold}] days {s['days']}  below floor k15 {s['days_below_floor_k15']} -> k8 "
              f"{s['days_below_floor_k8']}  flipped {s['days_flipped']}  near-floor@k8 "
              f"{s['days_near_floor_at_k8']}", flush=True)
        print(f"       recovered rows {s['recovered_rows_total']} (mean/day "
              f"{s['recovered_rows_per_day_mean']:.2f}, max {s['recovered_rows_per_day_max']}); "
              f"coverage gain mean {s['coverage_gain_mean']:.6f} max {s['coverage_gain_max']:.6f}", flush=True)
        def fmt(x, spec=".2f"):
            # a fold with no flipped days (F1 has none) yields None here, and None has no format spec
            return "n/a" if x is None else format(x, spec)
        print(f"       recovered/day: flipped {fmt(c['recovered_mean_on_flipped_days'])}, "
              f"near-floor {fmt(c['recovered_mean_on_near_floor_days'])}, "
              f"other {fmt(c['recovered_mean_on_all_other_days'])}", flush=True)
        print(f"       spearman(recovered, cov_k8) = {fmt(c['spearman_rho_recovered_vs_cov_k8'], '+.4f')} "
              f"(p={fmt(c['spearman_p'], '.3g')}); pearson(recovered, cov_gain) = "
              f"{fmt(c['pearson_r_recovered_vs_coverage_gain'], '+.4f')}", flush=True)
        mk = s["masking_concentration"]
        print(f"       >half-masked share of the test cross-section: k15 {100*mk['share_over_half_masked_k15_mean']:.4f}% "
              f"-> k8 {100*mk['share_over_half_masked_k8_mean']:.4f}% (worst day "
              f"{100*mk['share_over_half_masked_k8_max']:.4f}%); concentrated={mk['does_masking_concentrate_on_particular_days']}",
              flush=True)
        del day

    all_days = pd.concat(frames, ignore_index=True)
    csv = args.out / "masking_and_coverage_by_day.csv"
    all_days.to_csv(csv, index=False)
    pooled = summarise(all_days)

    result = {
        "question": ("Does the window-masking cost of k=8 concentrate on the same days as its coverage "
                     "benefit, i.e. are the rows that buy the 19 flipped days the same rows the model "
                     "has least history for?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False, "training_performed": False,
        "repository_module_imported_or_executed": False,
        "thresholds": {"history_length": HISTORY_LENGTH, "k_frozen": K_HI, "k_lowered": K_LO,
                       "coverage_floor": FLOOR, "near_floor_band": NEAR_BAND},
        "per_fold": per_fold,
        "pooled_across_folds": pooled,
        "per_day_csv": str(csv),
        "per_day_rows": int(len(all_days)),
        "cross_reference": {
            "BS4_found_43_near_floor_days_in_F2_at_k8": per_fold["F2"]["days_near_floor_at_k8"],
            "BS3_found_64_below_at_k15_and_45_at_k8": [pooled["days_below_floor_k15"],
                                                       pooled["days_below_floor_k8"]],
            "BT5_found_6800_recovered_rows_entering_the_cross_section": pooled["recovered_rows_total"],
        },
    }
    out = args.out / "masking_and_coverage_by_day.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out}\nwrote {csv} ({len(all_days)} day-rows)")
    print("\n=== pooled across the four folds ===")
    print(json.dumps({k: v for k, v in pooled.items()
                      if k != "are_recovered_rows_concentrated_where_coverage_is_thin"},
                     ensure_ascii=False, indent=1))
    print(json.dumps(pooled["are_recovered_rows_concentrated_where_coverage_is_thin"],
                     ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
