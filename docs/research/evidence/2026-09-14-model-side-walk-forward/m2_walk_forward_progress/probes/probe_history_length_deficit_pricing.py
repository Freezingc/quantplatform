#!/usr/bin/env python3
"""Read-only probe: what does minimum_valid_history_length have to become to clear G6?

§BR.3 priced §AY's path 2 as a target -- recovering 55.4% of M2's 0.5387% row deficit moves G6 from
0.932921 to 0.951496 and passes -- but explicitly did not find the parameter that hits the target, and
§BR.6 lists that as the thing it did not do. This probe finds it.

The reason it can be found without waiting for M2, and without training anything, is that the
eligibility rule does not depend on the model at all. training/temporal_batches.py:186-192 keeps a
security on date T when three things hold: it has a row in the panel on T; that row is valid_row, i.e.
label_valid and a finite raw_return_5d; and min(history_length, observed_count) >= 
minimum_valid_history_length, where observed_count is position - code_start + 1, the number of that
security's OWN panel rows at or before T. With history_length = 20 and any threshold at or below 20
that third test reduces to observed_count >= threshold. So the scoreable row set is a pure function of
the frozen panel and the threshold, and can be computed for all four folds today.

That also means the whole of §BR.3's pricing can be redone exactly rather than through the
multiplicative approximation: for each threshold, intersect the eligible set with the baseline's
prediction keys the way evaluation/model_evaluator.py:_match_prediction_coverage does, and compute the
per-day coverage directly.

The rule re-implementation is validated before it is trusted, on the one fold where it can be: at the
frozen threshold of 15 the computed eligible set for F1 is compared with the keys M2 actually produced
predictions for. If those two sets do not agree, nothing downstream is reported.

Safety: reads parquet artifacts in place. No database access, no weight loading, no repository module
imported or executed, nothing written into any artifact directory, no process signalled.
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
KEYS = ["signal_date", "ts_code"]
HISTORY_LENGTH = 20
FROZEN_THRESHOLD = 15
THRESHOLDS = tuple(range(8, FROZEN_THRESHOLD + 1))
FLOOR = 0.90
G6_GATE = 0.95
TOTAL_OOS_DAYS = 969
TARGET_RATIO = 0.99760          # §BR.3: the first ratio that clears G6 at the unchanged floor


def load_panel(fold: str) -> pd.DataFrame:
    """train + validation + test concatenated, exactly as temporal_search.py:155-173 does."""
    ds = FOLD_DATASET[fold]
    parts = []
    for split in ("train", "validation", "test"):
        paths = sorted(glob.glob(str(ART / LABEL_WORKSPACE / "datasets" / ds / f"split={split}" / "*" / "*.parquet")))
        assert paths, f"{fold}/{split} missing"
        f = pd.concat([pd.read_parquet(p, columns=["signal_date", "ts_code", "raw_return_5d", "label_valid"])
                       for p in paths], ignore_index=True)
        f["split"] = split
        parts.append(f)
    panel = pd.concat(parts, ignore_index=True)
    panel["signal_date"] = pd.to_datetime(panel["signal_date"], errors="raise").dt.date
    panel["label_valid"] = panel["label_valid"].fillna(False).astype(bool)
    panel["raw_return_5d"] = pd.to_numeric(panel["raw_return_5d"], errors="coerce")
    # temporal_batches.py:79 -- stable sort on (ts_code, signal_date), which is what makes each
    # security's rows one contiguous run and code_start sufficient to recover it. NOTE the order:
    # code first, then date. Sorting on KEYS (signal_date, ts_code) instead leaves the runs
    # interleaved and run_lengths goes negative, which is how this bug announced itself.
    return panel.sort_values(["ts_code", "signal_date"], kind="mergesort").reset_index(drop=True)


def observed_count(panel: pd.DataFrame) -> np.ndarray:
    """position - code_start + 1, per temporal_batches.py:190."""
    codes = panel["ts_code"].to_numpy()
    unique_codes, code_start = np.unique(codes, return_index=True)
    code_start = np.asarray(code_start, dtype=np.int64)
    run_lengths = np.append(code_start[1:], len(panel)) - code_start
    assert (run_lengths > 0).all(), "code runs are not contiguous; the frame is not sorted by ts_code"
    assert (np.diff(code_start) > 0).all(), "code_start is not ascending"
    code_ids = np.repeat(np.arange(len(unique_codes), dtype=np.int64), run_lengths)
    return np.arange(len(panel), dtype=np.int64) - code_start[code_ids] + 1


def read_pred(path: Path) -> pd.DataFrame:
    f = pd.read_parquet(path, columns=["signal_date", "ts_code"])
    f["signal_date"] = pd.to_datetime(f["signal_date"], errors="raise").dt.date
    return f


def get_pred_keys(workspace: str, fold: str, kind: str) -> pd.DataFrame | None:
    for man in sorted(glob.glob(str(ART / workspace / "predictions" / "*" / "manifest.json"))):
        md = json.load(open(man, encoding="utf-8")).get("metadata", {})
        if md.get("kind") == kind and md.get("fold_id") == fold:
            return read_pred(Path(man).parent / "scores.parquet")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "history_length_deficit_pricing")
    ap.add_argument("--folds", default="F1,F2,F3,F4")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    folds = args.folds.split(",")

    per_fold, validation = {}, {}
    for fold in folds:
        panel = load_panel(fold)
        oc = observed_count(panel)
        valid_row = panel["label_valid"].to_numpy() & np.isfinite(panel["raw_return_5d"].to_numpy())
        test = (panel["split"] == "test").to_numpy()
        assert not (test & ~valid_row).sum() < 0
        dates = panel.loc[test, "signal_date"].to_numpy()
        oc_t, vr_t = oc[test], valid_row[test]

        base_keys = get_pred_keys("m2_v1", fold, "best_simple")
        if base_keys is None:
            base_keys = read_pred(ART / "baseline_v2" / "baseline_oos" / f"{fold}.parquet")
        base_flag = base_keys[KEYS].drop_duplicates().assign(_in_baseline=True)
        codes_t = panel.loc[test, "ts_code"].to_numpy()

        # ---- VALIDATION on whichever fold has M2 predictions: does the computed eligible set at the
        # ---- frozen threshold equal the keys M2 actually scored?
        m2_keys = get_pred_keys("m2_v1", fold, "candidate")
        if m2_keys is not None:
            elig = vr_t & (np.minimum(HISTORY_LENGTH, oc_t) >= FROZEN_THRESHOLD)
            computed_frame = pd.DataFrame({"signal_date": dates[elig], "ts_code": codes_t[elig]})
            actual_frame = m2_keys[KEYS].drop_duplicates()
            both = computed_frame.merge(actual_frame.assign(_a=True), on=KEYS, how="outer", indicator=True)
            computed = set(map(tuple, computed_frame.to_numpy()))
            actual = set(map(tuple, actual_frame.to_numpy()))
            validation[fold] = {
                "threshold": FROZEN_THRESHOLD,
                "computed_eligible_test_rows": len(computed),
                "m2_actual_prediction_rows": len(actual),
                "in_both": len(computed & actual),
                "computed_only": len(computed - actual),
                "actual_only": len(actual - computed),
                "jaccard": len(computed & actual) / len(computed | actual) if (computed | actual) else None,
                "exact_set_equality": computed == actual,
            }
            print(f"[{fold}] VALIDATION at threshold {FROZEN_THRESHOLD}: computed {len(computed)} vs "
                  f"M2 actual {len(actual)}, intersection {len(computed & actual)}, "
                  f"computed_only {len(computed - actual)}, actual_only {len(actual - computed)}, "
                  f"exact={computed == actual}", flush=True)

        # ---- per threshold, the exact matched coverage and the resulting G6
        rows = {}
        for k in THRESHOLDS:
            elig = vr_t & (np.minimum(HISTORY_LENGTH, oc_t) >= k)
            elig_frame = pd.DataFrame({"signal_date": dates[elig], "ts_code": codes_t[elig]})
            # _match_prediction_coverage is an inner join on the business keys; a left merge with an
            # indicator reproduces membership without building a python set of 800k tuples per threshold
            joined = elig_frame.merge(base_flag, on=KEYS, how="left")
            matched = joined["_in_baseline"].fillna(False).to_numpy(dtype=bool)
            frame = pd.DataFrame({"date": joined["signal_date"].to_numpy(), "matched": matched})
            vl = pd.DataFrame({"date": dates, "vr": vr_t}).groupby("date")["vr"].sum()
            sc = frame.groupby("date")["matched"].sum()
            sc = sc.reindex(vl.index, fill_value=0)
            cov = (sc / vl.replace(0, np.nan)).fillna(0.0)
            below = int((cov < FLOOR).sum())
            # headroom: how many days sit just above the floor. At the first passing threshold the
            # margin is only a few days, so "does it pass" is not enough -- "by how many days, and how
            # many more would a slightly larger deficit lose" is the question an owner can act on.
            near = int(((cov >= FLOOR) & (cov < FLOOR + 0.005)).sum())
            rows[k] = {
                "days_within_0.005_above_the_floor": near,
                "days_below_floor_plus_0.005": int((cov < FLOOR + 0.005).sum()),
                "G6_if_the_floor_rose_by_0.005": float((len(cov) - int((cov < FLOOR + 0.005).sum())) / len(cov)),
                "threshold": k,
                "eligible_test_rows": int(elig.sum()),
                "deficit_rows_vs_valid_label_rows": int(vr_t.sum() - elig.sum()),
                "deficit_share_of_valid_label_rows": float((vr_t.sum() - elig.sum()) / vr_t.sum()),
                "rows_removed_by_baseline_matching": int(elig.sum() - int(matched.sum())) if elig.any() else 0,
                "coverage_min": float(cov.min()), "coverage_median": float(cov.median()),
                "days_below_floor": below,
                "valid_days": int(len(cov) - below),
                "G6_this_fold": float((len(cov) - below) / len(cov)),
            }
            print(f"  [{fold}] k={k:>2} eligible={rows[k]['eligible_test_rows']:>7} "
                  f"deficit={rows[k]['deficit_rows_vs_valid_label_rows']:>6} "
                  f"({100*rows[k]['deficit_share_of_valid_label_rows']:.4f}%) "
                  f"cov_min={rows[k]['coverage_min']:.6f} days_below={below:>3} "
                  f"near_floor={rows[k]['days_within_0.005_above_the_floor']:>3} "
                  f"G6_fold={rows[k]['G6_this_fold']:.6f}", flush=True)
        per_fold[fold] = {"panel_rows": int(len(panel)), "test_rows": int(test.sum()),
                          "test_valid_label_rows": int(vr_t.sum()), "by_threshold": rows}
        del panel, oc, valid_row

    # ---- combine folds
    combined = []
    for k in THRESHOLDS:
        days_below = sum(per_fold[f]["by_threshold"][k]["days_below_floor"] for f in per_fold)
        elig = sum(per_fold[f]["by_threshold"][k]["eligible_test_rows"] for f in per_fold)
        deficit = sum(per_fold[f]["by_threshold"][k]["deficit_rows_vs_valid_label_rows"] for f in per_fold)
        vl = sum(per_fold[f]["test_valid_label_rows"] for f in per_fold)
        g6 = (TOTAL_OOS_DAYS - days_below) / TOTAL_OOS_DAYS
        combined.append({
            "threshold": k, "eligible_test_rows_all_folds": elig,
            "deficit_rows_all_folds": deficit,
            "deficit_share": deficit / vl,
            "days_below_floor": days_below, "valid_days": TOTAL_OOS_DAYS - days_below,
            "days_within_0.005_above_the_floor": sum(
                per_fold[f]["by_threshold"][k]["days_within_0.005_above_the_floor"] for f in per_fold),
            "headroom_in_days": (TOTAL_OOS_DAYS - days_below) - int(np.ceil(G6_GATE * TOTAL_OOS_DAYS)),
            "G6": g6, "passes_G6": g6 >= G6_GATE,
            "delta_vs_frozen_threshold": None,
        })
    frozen = next(c for c in combined if c["threshold"] == FROZEN_THRESHOLD)
    for c in combined:
        c["delta_vs_frozen_threshold"] = {"days_below_floor": c["days_below_floor"] - frozen["days_below_floor"],
                                          "G6": c["G6"] - frozen["G6"]}
    first_pass = next((c for c in combined if c["passes_G6"]), None)
    # what share of the frozen deficit has to be recovered, and does the passing threshold hit §BR.3's target ratio
    recovered = None
    if first_pass:
        recovered = 1 - first_pass["deficit_rows_all_folds"] / frozen["deficit_rows_all_folds"]

    result = {
        "question": ("What does minimum_valid_history_length have to become for M2 to clear G6 at the "
                     "unchanged 0.90 floor, i.e. what parameter achieves the 55.4% deficit recovery "
                     "§BR.3 priced but did not locate?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False,
        "repository_module_imported_or_executed": False,
        "why_this_needs_no_training": (
            "the eligibility rule at temporal_batches.py:186-192 depends only on the frozen panel and the "
            "threshold: a security is kept on date T iff it has a row on T, that row has a valid label and "
            "a finite target, and its own panel row count at or before T is at least the threshold "
            "(min(20, observed_count) >= k reduces to observed_count >= k for k <= 20). No model, no "
            "weights and no predictions enter it."),
        "rule_source": {"eligibility": "training/temporal_batches.py:186-192",
                        "valid_row": "training/temporal_batches.py:138",
                        "sort_and_code_start": "training/temporal_batches.py:79,141",
                        "panel_concatenation": "run_real_candidate_training.py:3028 -> temporal_search.py:155-173",
                        "row_matching": "evaluation/model_evaluator.py:59-64 -> _match_prediction_coverage:140-168",
                        "history_length": HISTORY_LENGTH, "frozen_threshold": FROZEN_THRESHOLD},
        "validation_against_m2s_actual_predictions": validation,
        "folds": sorted(per_fold),
        "per_fold": per_fold,
        "combined_across_folds": combined,
        "answer": {
            "frozen_threshold": FROZEN_THRESHOLD,
            "frozen_deficit_rows": frozen["deficit_rows_all_folds"],
            "frozen_deficit_share": frozen["deficit_share"],
            "frozen_days_below_floor": frozen["days_below_floor"],
            "frozen_G6": frozen["G6"],
            "first_threshold_that_passes": (first_pass or {}).get("threshold"),
            "its_deficit_rows": (first_pass or {}).get("deficit_rows_all_folds"),
            "its_deficit_share": (first_pass or {}).get("deficit_share"),
            "its_days_below_floor": (first_pass or {}).get("days_below_floor"),
            "its_G6": (first_pass or {}).get("G6"),
            "share_of_the_frozen_deficit_recovered_at_that_threshold": recovered,
            "BR3_target_ratio_for_cross_check": TARGET_RATIO,
            "BR3_target_deficit_recovery_for_cross_check": 0.554,
        },
    }
    out = args.out / "history_length_deficit_pricing.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out}")
    print("\n=== combined across folds (exact, not projected) ===")
    print(f"{'k':>3}{'eligible rows':>15}{'deficit rows':>14}{'deficit %':>11}{'days<0.90':>11}{'G6':>12}{'headroom':>10}{'verdict':>9}")
    for c in combined:
        print(f"{c['threshold']:>3}{c['eligible_test_rows_all_folds']:>15}{c['deficit_rows_all_folds']:>14}"
              f"{100*c['deficit_share']:>11.4f}{c['days_below_floor']:>11}{c['G6']:>12.6f}"
              f"{c['headroom_in_days']:>10}{'PASS' if c['passes_G6'] else 'FAIL':>9}")
    a = result["answer"]
    print(f"\nfrozen k=15: deficit {a['frozen_deficit_rows']} rows ({100*a['frozen_deficit_share']:.4f}%), "
          f"{a['frozen_days_below_floor']} days below, G6 {a['frozen_G6']:.6f}")
    print(f"first passing threshold: k={a['first_threshold_that_passes']} -> G6 {a['its_G6']:.6f}, "
          f"deficit {a['its_deficit_rows']} rows, recovering "
          f"{100*(a['share_of_the_frozen_deficit_recovered_at_that_threshold'] or 0):.1f}% of the frozen deficit")
    print(f"§BR.3 asked for a 55.4% recovery; this locates it at k={a['first_threshold_that_passes']}")


if __name__ == "__main__":
    main()
