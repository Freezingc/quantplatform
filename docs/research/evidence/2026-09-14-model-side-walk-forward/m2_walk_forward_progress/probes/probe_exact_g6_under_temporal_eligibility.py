#!/usr/bin/env python3
"""Read-only: M2's G6 computed exactly, using the source-derived eligibility rule.

§AW projected G6 with a rule fitted on one fold (precision 0.9369 / recall 0.9723).
`probe_temporal_eligibility_exact.py` then showed the rule can be taken straight from
`temporal_batches._batch_for_date` and reproduces M2's real F1 validation key set EXACTLY
(symmetric difference 0). So the projection can be replaced by a computation.

Coverage here is the same quantity §AW step 1 validated bit-for-bit against the M2 pre-registration
on all four folds: per signal date, |rows the metric can score that both sides cover| / |scoreable
label rows|, where the baseline side's key set is "all nine dense columns non-null" and the floor is
the 0.90 in evaluation/gates.py. G6 = days at or above the floor / 969.

Read-only: no database, no weights, no model, nothing written into any artifact directory.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from statistics import median

import numpy as np
import pandas as pd

from probe_temporal_eligibility_exact import (
    ART, COLS, FOLD_DATASETS, load_split, eligible_keys, scoreable_keys,
)

DENSE = [f"f{i:02d}" for i in (0, 1, 4, 9, 10, 11, 12, 13, 14)]
FLOOR = 0.90
G6_DENOMINATOR = 969
G6_PREREGISTERED = 0.9752321981424149
ALL_COLS = COLS + [f"f{i:02d}" for i in range(15)]


def load_split_full(artifacts: Path, ds: str, split: str) -> pd.DataFrame:
    import glob
    for ws in ("m2_v1", "m0r_v2", "m1_full_v1", "m0_v3", "m0_v2"):
        paths = sorted(glob.glob(str(artifacts / ws / "datasets" / ds / f"split={split}" / "*" / "*.parquet")))
        if paths:
            frame = pd.concat([pd.read_parquet(p, columns=ALL_COLS) for p in paths], ignore_index=True)
            frame["signal_date"] = pd.to_datetime(frame["signal_date"]).dt.date
            return frame
    raise FileNotFoundError(f"{ds} split={split}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=ART)
    ap.add_argument("--out", type=Path, default=ART / "m2_rowset")
    args = ap.parse_args()

    result = {
        "question": "With the eligibility rule taken from source and verified exact, what is M2's G6?",
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False,
        "method": (
            "candidate keys = the exact temporal eligibility set over the train+validation+test panel; "
            "baseline keys = all nine dense columns non-null (the reconstruction §AW step 1 matched to "
            "0.00e+00 against the pre-registration on all four folds); the metric scores their "
            "intersection, as model_evaluator._match_prediction_coverage does; coverage per date = "
            "|intersection AND scoreable| / |scoreable|; a date below 0.90 is dropped"
        ),
        "folds": {},
    }
    tot_below_before = tot_below_after = tot_days = 0
    for fold, ds in FOLD_DATASETS.items():
        tr = load_split(args.artifacts, ds, "train")
        va = load_split(args.artifacts, ds, "validation")
        te_full = load_split_full(args.artifacts, ds, "test")
        test_dates = set(te_full["signal_date"])
        panel = pd.concat([tr, va, te_full.loc[:, COLS]], ignore_index=True)
        cand, diag = eligible_keys(panel, test_dates)

        feat = te_full[[f"f{i:02d}" for i in range(15)]].apply(pd.to_numeric, errors="coerce")
        dense_ok = feat[DENSE].notna().all(axis=1).to_numpy()
        sc = scoreable_keys(te_full)
        te_keys = list(map(tuple, te_full[["signal_date", "ts_code"]].itertuples(index=False, name=None)))
        base = {k for k, d in zip(te_keys, dense_ok) if d} & sc
        matched = base & cand

        den = pd.Series([1 for k in te_keys if k in sc]).groupby(
            pd.Series([k[0] for k in te_keys if k in sc]).values).size()
        def per_day(keyset):
            s = pd.Series([k[0] for k in keyset]).value_counts()
            return s.reindex(den.index).fillna(0) / den
        cb, ca = per_day(base), per_day(matched)
        delta = ca - cb
        assert float(delta.max()) <= 0.0, f"{fold}: intersecting raised coverage by {float(delta.max())}"

        result["folds"][fold] = {
            "dataset_id": ds,
            "eligibility_diagnostics": diag,
            "scoreable_rows": len(sc),
            "baseline_key_rows": len(base),
            "candidate_key_rows": len(cand),
            "matched_rows": len(matched),
            "rows_the_candidate_lacks_that_the_baseline_has": len(base - cand),
            "coverage_baseline_only": {"min": float(cb.min()), "median": float(cb.median())},
            "coverage_matched": {"min": float(ca.min()), "median": float(ca.median())},
            "per_day_delta": {"min": float(delta.min()), "median": float(median(delta.tolist())),
                              "max": float(delta.max()), "days_changed": int((delta.abs() > 0).sum()),
                              "days_total": int(len(den))},
            "days_below_floor_baseline_only": int((cb < FLOOR).sum()),
            "days_below_floor_matched": int((ca < FLOOR).sum()),
            "days_pushed_below_by_the_candidate": int(((cb >= FLOOR) & (ca < FLOOR)).sum()),
            "days_near_the_floor_baseline_only": {
                "in_[0.900,0.905)": int(((cb >= 0.900) & (cb < 0.905)).sum()),
                "in_[0.905,0.910)": int(((cb >= 0.905) & (cb < 0.910)).sum()),
            },
        }
        tot_below_before += int((cb < FLOOR).sum())
        tot_below_after += int((ca < FLOOR).sum())
        tot_days += int(len(den))
        print(f"  {fold}: below floor {int((cb<FLOOR).sum())} -> {int((ca<FLOOR).sum())} "
              f"cov min {cb.min():.6f} -> {ca.min():.6f} delta med {median(delta.tolist()):+.6f}", flush=True)

    result["pooled"] = {
        "oos_days": tot_days,
        "g6_denominator_used": G6_DENOMINATOR,
        "days_below_floor_baseline_only": tot_below_before,
        "days_below_floor_with_m2": tot_below_after,
        "g6_baseline_only": (G6_DENOMINATOR - tot_below_before) / G6_DENOMINATOR,
        "g6_with_m2": (G6_DENOMINATOR - tot_below_after) / G6_DENOMINATOR,
        "g6_preregistered": G6_PREREGISTERED,
        "g6_gate_threshold": 0.95,
        "g6_with_m2_clears_the_gate": (G6_DENOMINATOR - tot_below_after) / G6_DENOMINATOR >= 0.95,
        "aw_projection_was": {"days_below_floor": 64, "g6": 0.9339525283797729,
                              "band": [0.9318885448916409, 0.934984520123839]},
    }
    out = args.out / "g6_exact_under_temporal_eligibility.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps(result["pooled"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
