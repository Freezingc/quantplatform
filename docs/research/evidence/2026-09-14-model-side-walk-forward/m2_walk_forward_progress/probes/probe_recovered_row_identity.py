#!/usr/bin/env python3
"""Read-only probe: who ARE the rows that lowering minimum_valid_history_length admits?

Two questions in the pack are about the same 6,800 rows and neither has been answered.

§BV.8 asks why F4 contributes only 3.42 recovered rows a day against about 8.5 in F1 and F2, and offers
an unverified guess: F4's test year is 2025, further from most securities' listing dates, so fewer
securities have under 15 of their own panel rows. §BU.3 refuted the listing-age explanation for a
different question -- the F1-versus-F2 sign flip in predictability -- by showing the recovered rows have
mean observed_count 10.99 in F1 and 11.00 in F2. But that comparison only looked at the age *within*
the 8..14 band, which is near-uniform by construction. It did not look at how many securities fall into
the band at all, which is what determines the row count, nor at when those securities first appear,
which is what would distinguish a genuine IPO from a late addition to the frozen universe.

So this probe characterises the group properly: how many distinct securities are in it, how many rows
each contributes, when they first appear in the panel, how far the test date is from that first
appearance, how the rows are distributed across the test window, and how their labels compare with the
retained rows'. The count of securities newly appearing in the panel near each fold's test window is
the direct test of the F4 guess.

Safety: reads parquet artifacts in place. No database access, no weight loading, no training, no
repository module imported or executed, nothing written into any artifact directory, no process
signalled.
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
    assert (runs > 0).all()
    ids = np.repeat(np.arange(len(cs), dtype=np.int64), runs)
    p["observed_count"] = np.arange(len(p), dtype=np.int64) - cs[ids] + 1
    # first panel appearance per security: the row at code_start, broadcast back over the run
    p["first_seen"] = p["signal_date"].to_numpy()[cs][ids]
    p["valid_row"] = p["label_valid"] & np.isfinite(p["raw_return_5d"])
    return p


def describe_labels(x: pd.Series) -> dict:
    x = x.dropna()
    if not len(x):
        return {"n": 0}
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(x.median()),
            "sd": float(x.std(ddof=1)), "min": float(x.min()), "p01": float(x.quantile(0.01)),
            "p99": float(x.quantile(0.99)), "max": float(x.max()),
            "mean_abs": float(x.abs().mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "recovered_row_identity")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    per_fold, rows_out = {}, []
    for fold in ("F1", "F2", "F3", "F4"):
        p = load_panel(fold)
        t = p.loc[p["split"] == "test"].reset_index(drop=True)
        base = pd.read_parquet(ART / "baseline_v2" / "baseline_oos" / f"{fold}.parquet", columns=KEYS)
        base["signal_date"] = pd.to_datetime(base["signal_date"], errors="raise").dt.date
        t = t.merge(base.assign(in_baseline=True), on=KEYS, how="left")
        t["in_baseline"] = t["in_baseline"].fillna(False).astype(bool)
        oc = t["observed_count"].to_numpy()
        vr = t["valid_row"].to_numpy()
        ib = t["in_baseline"].to_numpy()
        t["recovered"] = vr & (oc >= K_LO) & (oc < K_HI) & ib
        t["retained"] = vr & (oc >= K_HI) & ib
        # days from first panel appearance to the test date
        t["days_since_first_seen"] = [
            (a - b).days for a, b in zip(t["signal_date"], t["first_seen"])]

        test_start, test_end = min(t["signal_date"]), max(t["signal_date"])
        # the direct test of the F4 guess: how many securities first appear in the panel close to, or
        # inside, this fold's test window
        first = p.groupby("ts_code")["first_seen"].min()
        new_in_window = int(((first >= test_start) & (first <= test_end)).sum())
        new_within_60d = int(((first >= test_start - pd.Timedelta(days=60)) & (first <= test_end)).sum())
        all_codes = int(first.size)

        rec, ret = t[t["recovered"]], t[t["retained"]]
        per_sec = rec.groupby("ts_code").size()
        by_month = rec.assign(m=rec["signal_date"].astype(str).str[:7]).groupby("m").size()
        entry = {
            "test_window": [str(test_start), str(test_end)],
            "distinct_securities_in_panel": all_codes,
            "securities_first_appearing_inside_the_test_window": new_in_window,
            "securities_first_appearing_within_60d_before_or_inside": new_within_60d,
            "share_of_panel_securities_new_in_window": new_in_window / all_codes if all_codes else None,
            "recovered": {
                "rows": int(len(rec)), "distinct_securities": int(rec["ts_code"].nunique()),
                "rows_per_security_mean": float(per_sec.mean()) if len(per_sec) else None,
                "rows_per_security_max": int(per_sec.max()) if len(per_sec) else None,
                "rows_per_security_median": float(per_sec.median()) if len(per_sec) else None,
                "rows_per_test_day": float(len(rec) / t["signal_date"].nunique()),
                "observed_count_mean": float(rec["observed_count"].mean()) if len(rec) else None,
                "days_since_first_seen_mean": float(rec["days_since_first_seen"].mean()) if len(rec) else None,
                "days_since_first_seen_median": float(rec["days_since_first_seen"].median()) if len(rec) else None,
                "days_since_first_seen_min": int(rec["days_since_first_seen"].min()) if len(rec) else None,
                "days_since_first_seen_max": int(rec["days_since_first_seen"].max()) if len(rec) else None,
                "labels": describe_labels(rec["raw_return_5d"]),
                "rows_by_month": {k: int(v) for k, v in by_month.items()},
                "n_distinct_test_dates_covered": int(rec["signal_date"].nunique()),
            },
            "retained": {
                "rows": int(len(ret)), "distinct_securities": int(ret["ts_code"].nunique()),
                "labels": describe_labels(ret["raw_return_5d"]),
                "days_since_first_seen_mean": float(ret["days_since_first_seen"].mean()) if len(ret) else None,
            },
        }
        entry["label_comparison"] = {
            "recovered_mean": entry["recovered"]["labels"].get("mean"),
            "retained_mean": entry["retained"]["labels"].get("mean"),
            "recovered_sd": entry["recovered"]["labels"].get("sd"),
            "retained_sd": entry["retained"]["labels"].get("sd"),
            "recovered_mean_abs": entry["recovered"]["labels"].get("mean_abs"),
            "retained_mean_abs": entry["retained"]["labels"].get("mean_abs"),
            "mannwhitney_p_on_labels": float(stats.mannwhitneyu(
                rec["raw_return_5d"].dropna(), ret["raw_return_5d"].dropna(),
                alternative="two-sided").pvalue) if len(rec) else None,
            "levene_p_on_label_dispersion": float(stats.levene(
                rec["raw_return_5d"].dropna(), ret["raw_return_5d"].dropna()).pvalue) if len(rec) else None,
        }
        per_fold[fold] = entry
        r = t[t["recovered"] | t["retained"]][["signal_date", "ts_code", "observed_count",
                                               "first_seen", "days_since_first_seen",
                                               "raw_return_5d", "recovered"]].copy()
        r["fold"] = fold
        rows_out.append(r)
        rc, rt = entry["recovered"], entry["retained"]
        print(f"[{fold}] test window {test_start}..{test_end}", flush=True)
        print(f"       panel securities {all_codes}; first appearing INSIDE the test window "
              f"{new_in_window} ({100*new_in_window/all_codes:.3f}%); within 60d before or inside "
              f"{new_within_60d}", flush=True)
        print(f"       recovered: {rc['rows']} rows / {rc['distinct_securities']} securities "
              f"({rc['rows_per_security_mean']:.2f} rows each, max {rc['rows_per_security_max']}), "
              f"{rc['rows_per_test_day']:.2f} rows/day, on {rc['n_distinct_test_dates_covered']} distinct dates",
              flush=True)
        print(f"       recovered days_since_first_seen: mean {rc['days_since_first_seen_mean']:.1f} "
              f"median {rc['days_since_first_seen_median']:.1f} min {rc['days_since_first_seen_min']} "
              f"max {rc['days_since_first_seen_max']}", flush=True)
        print(f"       labels: recovered mean {rc['labels']['mean']:+.6f} sd {rc['labels']['sd']:.6f} "
              f"|ret| {rc['labels']['mean_abs']:.6f}   vs retained mean {rt['labels']['mean']:+.6f} "
              f"sd {rt['labels']['sd']:.6f} |ret| {rt['labels']['mean_abs']:.6f}", flush=True)
        lc = entry["label_comparison"]
        print(f"       Mann-Whitney on labels p={lc['mannwhitney_p_on_labels']:.3g}   "
              f"Levene on dispersion p={lc['levene_p_on_label_dispersion']:.3g}", flush=True)
        del p, t

    allrows = pd.concat(rows_out, ignore_index=True)
    csv = args.out / "recovered_and_retained_rows.csv"
    allrows.to_csv(csv, index=False)

    result = {
        "question": ("Who are the rows that lowering minimum_valid_history_length from 15 to 8 admits: "
                     "how many securities, when did they first appear in the panel, how do their labels "
                     "compare with the retained rows, and does the count of newly appearing securities "
                     "explain why F4 contributes only 3.42 recovered rows a day against about 8.5 in F1 "
                     "and F2?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False, "training_performed": False,
        "repository_module_imported_or_executed": False,
        "thresholds": {"history_length": HISTORY_LENGTH, "k_frozen": K_HI, "k_lowered": K_LO},
        "per_fold": per_fold,
        "f4_hypothesis_test": {
            "hypothesis": ("§BV.8's guess: F4's test year 2025 is further from securities' listing "
                           "dates, so fewer securities have under 15 of their own panel rows"),
            "securities_first_appearing_inside_the_test_window": {
                f: per_fold[f]["securities_first_appearing_inside_the_test_window"] for f in per_fold},
            "share_of_panel_securities": {
                f: per_fold[f]["share_of_panel_securities_new_in_window"] for f in per_fold},
            "recovered_rows_per_test_day": {
                f: per_fold[f]["recovered"]["rows_per_test_day"] for f in per_fold},
            "recovered_distinct_securities": {
                f: per_fold[f]["recovered"]["distinct_securities"] for f in per_fold},
        },
        "per_row_csv": str(csv),
        "per_row_count": int(len(allrows)),
    }
    out = args.out / "recovered_row_identity.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out}\nwrote {csv} ({len(allrows)} rows)")
    print("\n=== the F4 hypothesis, side by side ===")
    h = result["f4_hypothesis_test"]
    print(f"  {'fold':<6}{'new in test window':>20}{'share of panel':>16}{'recovered securities':>22}{'rows/day':>10}")
    for f in ("F1", "F2", "F3", "F4"):
        print(f"  {f:<6}{h['securities_first_appearing_inside_the_test_window'][f]:>20}"
              f"{100*h['share_of_panel_securities'][f]:>15.3f}%"
              f"{h['recovered_distinct_securities'][f]:>22}{h['recovered_rows_per_test_day'][f]:>10.2f}")


if __name__ == "__main__":
    main()
