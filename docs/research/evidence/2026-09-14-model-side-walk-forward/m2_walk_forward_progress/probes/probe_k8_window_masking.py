#!/usr/bin/env python3
"""Read-only probe: what would the model actually SEE at k=8, on the training splits as well as the test split?

§BT settled the compositional half of the k=8 question and left the model half open, and the resource
snapshot beside this probe records why the model half is deferred rather than answered: swap is
exhausted, M2's tree holds 369.3 GiB, and temporal_fold_batches peaks at 12.77 GiB per process, so
training six paired runs now would risk an OOM on the critical path for an experiment that costs
nothing to run five days later.

This probe measures the part of the model half that does not need a trained model: the input. A row's
window has `length = min(history_length, observed_count)` live lanes out of 20, and the rest are masked
(temporal_batches.py:194-201 builds `live = steps < length` and zeroes everything else). So the
threshold is not only a rule about which rows get SCORED -- it is a rule about which rows get TRAINED
ON, and how much of each training window is real data versus mask.

At k=15 an eligible row has at most 5 masked lanes, i.e. at most 25% of its window is padding. At k=8 a
row may have 12 masked lanes, 60% padding. Lowering the threshold therefore does two things at once: it
admits test rows that the model must score from a mostly-empty window, and it admits TRAINING rows
built the same way, which changes the fit for every row rather than only for the new ones. That second
effect is the mechanism by which k=8 could make the model worse, and it is measurable from the frozen
panel alone at a few hundred MB per fold instead of 12.77 GiB.

It also tests the hypothesis §BT.8 left open -- that F1's recovered rows are strongly predictive while
F2's are anti-predictive because the two groups differ in listing age. observed_count IS the listing
age measured in panel rows, so the distributions can be compared directly.

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

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
FOLD_DATASET = {
    "F1": "226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538",
    "F2": "eb9bba1b5a0bd66d0ad6c82536b295de1fc079aec125cb27bb68491bdb21a234",
    "F3": "9766e69f342511c5bda30188df15a7b9b17d936bdc26b9b60ba244e8e7e0c9e7",
    "F4": "6f9478d7645c8c745a73f00c03b6122e17d0c06c05cc159ece531fb00df9fb9d",
}
LABEL_WORKSPACE = "m1_full_v1"
HISTORY_LENGTH = 20
KS = (15, 14, 12, 10, 8)


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
    p["length"] = np.minimum(HISTORY_LENGTH, p["observed_count"])
    p["masked_lanes"] = HISTORY_LENGTH - p["length"]
    return p


def summarise(sub: pd.DataFrame) -> dict:
    if not len(sub):
        return {"rows": 0}
    length = sub["length"].to_numpy()
    return {
        "rows": int(len(sub)),
        "mean_length": float(length.mean()),
        "median_length": float(np.median(length)),
        "min_length": int(length.min()),
        "mean_masked_lanes_of_20": float((HISTORY_LENGTH - length).mean()),
        "mean_masked_share": float(((HISTORY_LENGTH - length) / HISTORY_LENGTH).mean()),
        "share_of_rows_with_any_masking": float((length < HISTORY_LENGTH).mean()),
        "share_of_rows_over_half_masked": float((length <= HISTORY_LENGTH // 2).mean()),
        "length_histogram": {str(v): int(c) for v, c in
                             zip(*np.unique(length, return_counts=True))},
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "k8_model_effect")
    ap.add_argument("--folds", default="F1,F2,F3,F4")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    per_fold, train_pooled, test_pooled = {}, {k: [] for k in KS}, {k: [] for k in KS}
    for fold in args.folds.split(","):
        p = load_panel(fold)
        tr = p["split"].isin(["train", "validation"]).to_numpy()
        te = (p["split"] == "test").to_numpy()
        vr = p["valid_row"].to_numpy()
        oc = p["observed_count"].to_numpy()
        entry = {"panel_rows": int(len(p)), "train_plus_validation_rows": int(tr.sum()),
                 "test_rows": int(te.sum()), "by_k": {}}
        for k in KS:
            elig = vr & (oc >= k)
            trk, tek = elig & tr, elig & te
            base = vr & (oc >= 15)
            entry["by_k"][k] = {
                "threshold": k,
                "max_masked_lanes_a_row_may_have": HISTORY_LENGTH - k,
                "max_masked_share": (HISTORY_LENGTH - k) / HISTORY_LENGTH,
                "training_side": summarise(p.loc[trk]),
                "test_side": summarise(p.loc[tek]),
                "training_rows_added_vs_k15": int(trk.sum() - (base & tr).sum()),
                "training_rows_added_pct": float(100 * (trk.sum() - (base & tr).sum()) / (base & tr).sum()),
                "test_rows_added_vs_k15": int(tek.sum() - (base & te).sum()),
            }
            train_pooled[k].append(p.loc[trk, ["length"]])
            test_pooled[k].append(p.loc[tek, ["length"]])
        # the rows k=8 admits that k=15 rejects, per split -- these are the newly TRAINED-ON rows
        rec = vr & (oc >= 8) & (oc < 15)
        entry["newly_admitted_rows"] = {
            "training_side": summarise(p.loc[rec & tr]),
            "test_side": summarise(p.loc[rec & te]),
            "observed_count_distribution_training_side": {
                str(v): int(c) for v, c in zip(*np.unique(oc[rec & tr], return_counts=True))},
            "observed_count_distribution_test_side": {
                str(v): int(c) for v, c in zip(*np.unique(oc[rec & te], return_counts=True))},
            "test_side_mean_observed_count": float(oc[rec & te].mean()) if (rec & te).any() else None,
        }
        per_fold[fold] = entry
        b15, b8 = entry["by_k"][15], entry["by_k"][8]
        print(f"[{fold}] TRAIN+VAL eligible rows: k=15 {b15['training_side']['rows']:>8} -> "
              f"k=8 {b8['training_side']['rows']:>8} (+{b8['training_side']['rows']-b15['training_side']['rows']}, "
              f"{b8['training_rows_added_pct']:+.3f}%)", flush=True)
        print(f"       mean window length: k=15 {b15['training_side']['mean_length']:.4f} -> "
              f"k=8 {b8['training_side']['mean_length']:.4f}   mean masked share "
              f"{b15['training_side']['mean_masked_share']:.4%} -> {b8['training_side']['mean_masked_share']:.4%}",
              flush=True)
        print(f"       share of training rows over half masked: "
              f"{b15['training_side']['share_of_rows_over_half_masked']:.4%} -> "
              f"{b8['training_side']['share_of_rows_over_half_masked']:.4%}", flush=True)
        na = entry["newly_admitted_rows"]
        print(f"       newly admitted rows: train {na['training_side']['rows']}, "
              f"test {na['test_side']['rows']}; test-side mean observed_count "
              f"{na['test_side_mean_observed_count']:.2f}", flush=True)
        del p

    pooled = {}
    for k in KS:
        tr = pd.concat(train_pooled[k]); te = pd.concat(test_pooled[k])
        pooled[k] = {"training_side": summarise(tr), "test_side": summarise(te)}

    result = {
        "question": ("Lowering minimum_valid_history_length admits rows whose 20-lane window is mostly "
                     "masked, and it admits them on the TRAINING splits too, not only the test split. "
                     "How much masking does each threshold let into training, and do the newly admitted "
                     "rows differ in listing age between F1 and F2?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False, "training_performed": False,
        "repository_module_imported_or_executed": False,
        "why_no_training_was_needed": (
            "a row's window has length = min(history_length, observed_count) live lanes and the rest are "
            "masked (temporal_batches.py:194-201), and observed_count comes from the frozen panel, so the "
            "masking distribution is a property of the data and the threshold, not of any fitted model"),
        "history_length": HISTORY_LENGTH,
        "thresholds_scanned": list(KS),
        "pooled_across_folds": {str(k): v for k, v in pooled.items()},
        "per_fold": per_fold,
        "the_two_effects_of_lowering_k": {
            "test_side": "admits rows the model must SCORE from a mostly-empty window",
            "training_side": ("admits rows the model is TRAINED on that are mostly mask, which changes the "
                              "fit for every row rather than only for the new ones -- this is the mechanism "
                              "by which k=8 could make the model worse, and it is the half of §BT.1(b) that "
                              "does not need a trained model to characterise"),
        },
        "BT8_hypothesis_check": {
            "hypothesis": ("§BT.8 asked why F1's recovered rows are strongly predictive (+0.247711) while "
                           "F2's are anti-predictive (-0.196191), and guessed a difference in listing age"),
            "test_side_mean_observed_count_by_fold": {
                f: per_fold[f]["newly_admitted_rows"]["test_side_mean_observed_count"] for f in per_fold},
            "test_side_observed_count_distributions_by_fold": {
                f: per_fold[f]["newly_admitted_rows"]["observed_count_distribution_test_side"] for f in per_fold},
        },
    }
    out = args.out / "k8_window_masking.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out}")
    print(f"\n=== pooled across the four folds: what the model would be TRAINED on ===")
    print(f"{'k':>3}{'train rows':>12}{'mean len':>10}{'masked share':>14}{'any mask':>10}{'>half mask':>12}"
          f"{'test rows':>11}{'test mean len':>15}")
    for k in KS:
        t, e = pooled[k]["training_side"], pooled[k]["test_side"]
        print(f"{k:>3}{t['rows']:>12}{t['mean_length']:>10.4f}{t['mean_masked_share']:>13.4%}"
              f"{t['share_of_rows_with_any_masking']:>10.4%}{t['share_of_rows_over_half_masked']:>12.4%}"
              f"{e['rows']:>11}{e['mean_length']:>15.4f}")
    print(f"\n=== newly admitted rows (8 <= observed_count < 15), test side, by fold ===")
    for f in sorted(per_fold):
        na = per_fold[f]["newly_admitted_rows"]
        print(f"  {f}: n={na['test_side']['rows']:>5} mean observed_count "
              f"{na['test_side_mean_observed_count']:.2f}  distribution {na['observed_count_distribution_test_side']}")


if __name__ == "__main__":
    main()
