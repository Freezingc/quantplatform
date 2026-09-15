#!/usr/bin/env python3
"""Read-only probe: reconstruct M2's scored row set EXACTLY from source, then apply it to the TEST splits.

§AW projected M2's row deficit with a rule fitted on the one fold that has ground truth (precision
0.9369 / recall 0.9723). Reading the source removes the need to fit anything. The eligibility rule is
fully determined by four columns that the immutable dataset artifacts already carry:

  temporal_batches.py:_batch_for_date
      position        = row index of (security, date) in the panel sorted by (ts_code, signal_date)
      observed_count  = position - code_start + 1        # that security's own rows at or before T
      length          = min(history_length, observed_count)
      kept            = length >= minimum_valid_history_length
      eligible_mask   = observed AND valid_row           # valid_row = label_valid & finite target
      a date is emitted at all only if >= minimum_valid_stocks securities are eligible

  run_real_candidate_training.py:3019-3033 (the real, non-smoke path)
      validation panel = prepared_train + prepared_validation, filtered to validation_dates
      test panel       = prepared_train + prepared_validation + prepared_test, filtered to test_dates
                         (temporal_search.temporal_test_panel_batches)
  configs/models/temporal.yaml
      history_length 20, minimum_valid_history_length 15; minimum_valid_stocks defaults to 30
  inference/torch_scoring.prepare_frame is row-preserving, so it cannot change any count.

Step 1 verifies this reconstruction against M2's real F1 validation predictions. Only if the two key
sets are IDENTICAL does step 2 mean anything.

Step 2 then computes, for each fold's TEST split, the set that
`evaluation/walk_forward.py:_require_candidate_covers_the_scoreable_labels` compares against:
`scoreable - candidate_keys`, where scoreable = label_valid & finite raw_return_5d. That guard RAISES
when the difference is non-empty; its docstring justifies itself by saying the torch carriers drop only
unlabelled rows, which is true of the static family and is NOT true of the temporal family.

Read-only: no database, no weights, no model, nothing written into any artifact directory.
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
FOLD_DATASETS = {
    "F1": "226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538",
    "F2": "eb9bba1b5a0bd66d0ad6c82536b295de1fc079aec125cb27bb68491bdb21a234",
    "F3": "9766e69f342511c5bda30188df15a7b9b17d936bdc26b9b60ba244e8e7e0c9e7",
    "F4": "6f9478d7645c8c745a73f00c03b6122e17d0c06c05cc159ece531fb00df9fb9d",
}
WORKSPACES = ("m2_v1", "m0r_v2", "m1_full_v1", "m0_v3", "m0_v2")
COLS = ["signal_date", "ts_code", "label_valid", "raw_return_5d"]
HISTORY_LENGTH = 20
MIN_VALID_HISTORY = 15
MIN_VALID_STOCKS = 30


def load_split(artifacts: Path, ds: str, split: str) -> pd.DataFrame:
    for ws in WORKSPACES:
        paths = sorted(glob.glob(str(artifacts / ws / "datasets" / ds / f"split={split}" / "*" / "*.parquet")))
        if paths:
            frame = pd.concat(
                [pd.read_parquet(p, columns=COLS) for p in paths], ignore_index=True
            )
            frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.date
            return frame
    raise FileNotFoundError(f"{ds} split={split}")


def eligible_keys(panel: pd.DataFrame, allowed_dates: set) -> tuple[set, dict]:
    """Reproduce temporal_batches' eligibility on one concatenated panel, restricted to allowed dates."""
    p = panel.sort_values(["ts_code", "signal_date"], kind="mergesort").reset_index(drop=True)
    codes, first_rows = np.unique(p["ts_code"].to_numpy(), return_index=True)
    code_start = np.asarray(first_rows, dtype=np.int64)
    run_lengths = np.append(code_start[1:], len(p)) - code_start
    code_ids = np.repeat(np.arange(len(codes), dtype=np.int64), run_lengths)
    target = pd.to_numeric(p["raw_return_5d"], errors="coerce").to_numpy(dtype=np.float64)
    valid_row = p["label_valid"].fillna(False).astype(bool).to_numpy() & np.isfinite(target)
    positions = np.arange(len(p), dtype=np.int64)
    observed_count = positions - code_start[code_ids] + 1
    length = np.minimum(HISTORY_LENGTH, observed_count)
    kept = (length >= MIN_VALID_HISTORY) & valid_row
    dates = p["signal_date"].to_numpy()
    # np.isin on object dtype does O(n*m) python comparisons -- 3M rows x 240 dates is minutes per
    # call and this runs five times. isin against a set hashes instead, which is O(n).
    in_scope = pd.Series(dates).isin(set(allowed_dates)).to_numpy()
    chosen = kept & in_scope
    diag = {
        "panel_rows": int(len(p)),
        "rows_in_scope": int(in_scope.sum()),
        "rows_valid_label_in_scope": int((valid_row & in_scope).sum()),
        "rows_eligible_in_scope": int(chosen.sum()),
        "rows_dropped_for_short_history_in_scope": int(
            (valid_row & in_scope & (length < MIN_VALID_HISTORY)).sum()
        ),
    }
    # the per-date minimum_valid_stocks gate
    d = pd.DataFrame({"d": dates[chosen]})
    per_date = d.groupby("d").size()
    all_dates = sorted(allowed_dates)
    per_date = per_date.reindex(all_dates).fillna(0).astype(int)
    diag["dates_in_scope"] = len(all_dates)
    diag["dates_emitted"] = int((per_date >= MIN_VALID_STOCKS).sum())
    diag["dates_suppressed_by_minimum_valid_stocks"] = int((per_date < MIN_VALID_STOCKS).sum())
    diag["min_eligible_securities_on_any_emitted_date"] = int(per_date[per_date >= MIN_VALID_STOCKS].min()) \
        if (per_date >= MIN_VALID_STOCKS).any() else 0
    sub = p.loc[chosen, ["signal_date", "ts_code"]]
    keys = set(map(tuple, sub.itertuples(index=False, name=None)))
    return keys, diag


def scoreable_keys(frame: pd.DataFrame) -> set:
    target = pd.to_numeric(frame["raw_return_5d"], errors="coerce")
    usable = frame["label_valid"].fillna(False).astype(bool) & np.isfinite(target)
    return set(map(tuple, frame.loc[usable, ["signal_date", "ts_code"]].itertuples(index=False, name=None)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=ART)
    ap.add_argument("--out", type=Path, default=ART / "m2_rowset")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    result = {
        "question": (
            "Can M2's scored row set be reconstructed exactly from source, and if so, how many scoreable "
            "TEST rows will it be missing -- the number walk_forward's coverage guard raises on?"
        ),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False, "model_loaded": False,
        "constants": {"history_length": HISTORY_LENGTH,
                      "minimum_valid_history_length": MIN_VALID_HISTORY,
                      "minimum_valid_stocks": MIN_VALID_STOCKS},
    }

    # ---- step 1: verify against the one fold with ground truth --------------------
    ds1 = FOLD_DATASETS["F1"]
    train = load_split(args.artifacts, ds1, "train")
    val = load_split(args.artifacts, ds1, "validation")
    val_dates = set(val["signal_date"])
    predicted, diag = eligible_keys(pd.concat([train, val], ignore_index=True), val_dates)
    actual_frame = pd.read_parquet(
        args.artifacts / "m2_v1" / "models" / "m2-f1" / "validation_predictions.parquet",
        columns=["signal_date", "ts_code"],
    )
    actual_frame["signal_date"] = pd.to_datetime(actual_frame["signal_date"]).dt.date
    actual = set(map(tuple, actual_frame.itertuples(index=False, name=None)))
    result["step1_validation_reconstruction"] = {
        "diagnostics": diag,
        "predicted_keys": len(predicted),
        "actual_keys": len(actual),
        "identical": predicted == actual,
        "symmetric_difference": len(predicted ^ actual),
        "predicted_minus_actual": len(predicted - actual),
        "actual_minus_predicted": len(actual - predicted),
        "examples_of_disagreement": sorted(map(str, list(predicted ^ actual)[:10])),
        "verdict": (
            "the reconstruction reproduces M2's real scored key set exactly, so step 2 is a computation "
            "rather than a projection"
            if predicted == actual
            else "the reconstruction does NOT reproduce the real key set; step 2 must be read as a "
                 "projection only, and the disagreement examples above are where to look"
        ),
    }

    print("STEP1 done:", result["step1_validation_reconstruction"]["verdict"], flush=True)
    print("       ", json.dumps(result["step1_validation_reconstruction"]["diagnostics"]), flush=True)

    # ---- step 2: the TEST side, which is what the coverage guard reads -------------
    folds = {}
    for fold, ds in FOLD_DATASETS.items():
        tr = train if ds == ds1 else load_split(args.artifacts, ds, "train")
        va = val if ds == ds1 else load_split(args.artifacts, ds, "validation")
        print(f"  {fold}: splits loaded, building the test panel", flush=True)
        te = load_split(args.artifacts, ds, "test")
        test_dates = set(te["signal_date"])
        panel = pd.concat([tr, va, te], ignore_index=True)
        pred_test, diag_t = eligible_keys(panel, test_dates)
        sc = scoreable_keys(te)
        missing = sc - pred_test
        invented = pred_test - sc
        by_date = pd.Series(sorted(d for d, _ in missing)).value_counts().sort_index()
        securities = sorted({c for _, c in missing})
        first_test_day = min(test_dates)
        sec_first_day = te.groupby("ts_code")["signal_date"].min()
        folds[fold] = {
            "dataset_id": ds,
            "test_rows": int(len(te)),
            "diagnostics": diag_t,
            "scoreable_test_rows": len(sc),
            "predicted_candidate_keys": len(pred_test),
            "rows_with_a_usable_label_not_scored_by_the_candidate": len(missing),
            "candidate_keys_outside_the_scoreable_set": len(invented),
            "the_guard_would_raise": len(missing) > 0,
            "guard_message_would_be": (
                f"{len(missing)} rows with a usable label are not scored by the candidate"
            ),
            "missing_by_signal_date": {
                "dates_affected": int(len(by_date)),
                "dates_total": len(test_dates),
                "min_per_date": int(by_date.min()) if len(by_date) else 0,
                "median_per_date": float(by_date.median()) if len(by_date) else 0.0,
                "max_per_date": int(by_date.max()) if len(by_date) else 0,
                "first_ten": {str(k): int(v) for k, v in by_date.head(10).items()},
            },
            "missing_securities": {
                "count": len(securities),
                "securities_in_the_test_frame": int(te["ts_code"].nunique()),
                "of_those_first_appearing_on_the_test_window_open": int(
                    sum(1 for s in securities if sec_first_day.get(s) == first_test_day)
                ),
                "median_first_appearance": str(
                    pd.Series([sec_first_day[s] for s in securities if s in sec_first_day.index])
                    .sort_values().iloc[len(securities) // 2]
                ) if securities else None,
                "test_window_start": str(first_test_day),
            },
        }
    result["step2_test_side"] = folds
    result["headline"] = {
        "folds_where_the_guard_would_raise": sorted(
            f for f, v in folds.items() if v["the_guard_would_raise"]
        ),
        "total_scoreable_test_rows_not_scored": sum(
            v["rows_with_a_usable_label_not_scored_by_the_candidate"] for v in folds.values()
        ),
        "what_that_means": (
            "walk_forward._require_candidate_covers_the_scoreable_labels raises ValueError when the "
            "scoreable label rows are not a subset of the candidate's keys. Its docstring reasons that "
            "torch carriers drop only unlabelled rows -- true of build_static_training_batches, not of "
            "build_temporal_training_batches, which also drops short-history rows. If the numbers here "
            "hold, M2's OOS stage ends in that ValueError rather than in a gate verdict."
        ),
    }

    out = args.out / "temporal_eligibility_exact.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    s1 = result["step1_validation_reconstruction"]
    print("STEP1 identical:", s1["identical"], "| predicted", s1["predicted_keys"],
          "| actual", s1["actual_keys"], "| symdiff", s1["symmetric_difference"])
    print("        ", json.dumps(s1["diagnostics"]))
    for fold, v in folds.items():
        print(f"{fold}: scoreable {v['scoreable_test_rows']:7} predicted {v['predicted_candidate_keys']:7} "
              f"missing {v['rows_with_a_usable_label_not_scored_by_the_candidate']:6} "
              f"guard_raises={v['the_guard_would_raise']} dates_affected "
              f"{v['missing_by_signal_date']['dates_affected']}/{v['missing_by_signal_date']['dates_total']} "
              f"securities {v['missing_securities']['count']}")
    print(json.dumps(result["headline"], ensure_ascii=False, indent=1)[:900])


if __name__ == "__main__":
    main()
