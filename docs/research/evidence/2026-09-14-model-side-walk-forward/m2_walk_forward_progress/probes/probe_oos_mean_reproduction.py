#!/usr/bin/env python3
"""Read-only probe: reproduce every signed OOS mean_rank_ic from the frozen artifacts, independently.

The companion probe (probe_oos_score_health_day_effect.py) re-implemented the per-day arithmetic and
reproduced the frozen baseline's OOS mean_rank_ic to -1.39e-17, but reproduced the *candidates*' means
only to within 2.2e-03 -- and restricting candidates to the baseline's 945 valid days closed that to
between -2.2e-05 and +4.2e-04 without closing it exactly. The residue has a named cause:
evaluation/model_evaluator.py:59-64 calls _match_prediction_coverage, whose docstring is "Restrict
candidate and baseline scores to the identical test date/security keys", and which inner-joins the two
frames on (signal_date, ts_code) before either is scored. So the candidate is not scored on the
baseline's day set, it is scored on the baseline's ROW set, which changes each surviving day's
cross-section and therefore each day's rank correlation.

This probe implements that restriction and asks the only question that matters: does an independent
re-implementation, reading nothing but the frozen parquet artifacts, reproduce the signed numbers in
each carrier's report.json -- mean_rank_ic, valid_day_count and icir -- exactly?

If it does, three things follow at once. The 945-day OOS denominator is not an artifact of one code
path. The candidates really are scored on the baseline's row set, which is §K's claim that "the
baseline artifact's row coverage determines the candidate's evaluable day set", now reproduced
numerically rather than argued from source. And the score-health criteria really do bite the baseline
rather than the candidates, since the candidates inherit the baseline's coverage once matched.

Safety: reads parquet and JSON artifacts in place. No database access, no weight loading, no repository
module imported or executed, nothing written into any artifact directory.
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

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
FOLD_DATASET = {
    "F1": "226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538",
    "F2": "eb9bba1b5a0bd66d0ad6c82536b295de1fc079aec125cb27bb68491bdb21a234",
    "F3": "9766e69f342511c5bda30188df15a7b9b17d936bdc26b9b60ba244e8e7e0c9e7",
    "F4": "6f9478d7645c8c745a73f00c03b6122e17d0c06c05cc159ece531fb00df9fb9d",
}
LABEL_WORKSPACE = "m1_full_v1"
MIN_CROSS_SECTION = 30
COV_LO, COV_HI, MIN_UNIQUE, MIN_STD = 0.90, 1.0, 10, 1e-6
CARRIERS = (("M0R", "m0r_v2"), ("M0R-replicate", "m0r_v3"), ("M0-base", "m0_v2"),
            ("M0-refined", "m0_v3"), ("M1-full", "m1_full_v1"))


def load_labels(fold: str) -> pd.DataFrame:
    paths = sorted(glob.glob(str(ART / LABEL_WORKSPACE / "datasets" / FOLD_DATASET[fold]
                                 / "split=test" / "*" / "*.parquet")))
    frame = pd.concat([pd.read_parquet(p, columns=["signal_date", "ts_code", "raw_return_5d", "label_valid"])
                       for p in paths], ignore_index=True)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
    frame["label_valid"] = frame["label_valid"].fillna(False).astype(bool)
    frame["raw_return_5d"] = pd.to_numeric(frame["raw_return_5d"], errors="coerce")
    return frame


def find_pred(workspace: str, fold: str, kind: str) -> pd.DataFrame | None:
    for man in sorted(glob.glob(str(ART / workspace / "predictions" / "*" / "manifest.json"))):
        md = json.load(open(man, encoding="utf-8")).get("metadata", {})
        if md.get("kind") == kind and md.get("fold_id") == fold:
            f = pd.read_parquet(Path(man).parent / "scores.parquet",
                                columns=["signal_date", "ts_code", "score"])
            f["signal_date"] = pd.to_datetime(f["signal_date"], errors="raise").dt.date
            f["score"] = pd.to_numeric(f["score"], errors="coerce")
            return f
    return None


def daily_rank_ic(pred: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """evaluation/rank_ic.py:60-107 with include_invalid=True and require_score_health=True."""
    joined = labels.merge(pred, on=["signal_date", "ts_code"], how="left", validate="one_to_one")
    rows = []
    for d, g in joined.groupby("signal_date", sort=True):
        vl = g["label_valid"].to_numpy() & np.isfinite(g["raw_return_5d"].to_numpy())
        score = g["score"].to_numpy(dtype=float)
        lab = g["raw_return_5d"].to_numpy(dtype=float)
        vlc = int(vl.sum())
        fs = vl & np.isfinite(score)
        n_scored = int(fs.sum())
        coverage = n_scored / vlc if vlc > 0 else 0.0
        sc, lb = score[fs], lab[fs]
        uniq = int(pd.Series(sc).nunique()) if n_scored else 0
        std = float(np.std(sc, ddof=0)) if n_scored else float("nan")
        health = (math.isfinite(coverage) and COV_LO <= coverage <= COV_HI
                  and isinstance(uniq, int) and uniq >= MIN_UNIQUE
                  and math.isfinite(std) and std > MIN_STD)
        ric, valid = float("nan"), False
        if n_scored >= MIN_CROSS_SECTION and health:
            sr = pd.Series(sc).rank(method="average")
            lr = pd.Series(lb).rank(method="average")
            if sr.nunique(dropna=False) >= 2 and lr.nunique(dropna=False) >= 2:
                c = sr.corr(lr, method="pearson")
                if pd.notna(c) and math.isfinite(float(c)):
                    ric, valid = float(c), True
        rows.append({"signal_date": d, "rank_ic": ric, "rank_ic_valid": valid,
                     "coverage": coverage, "sample_count": n_scored,
                     "valid_label_count": vlc})
    return pd.DataFrame(rows)


def summarise(daily: pd.DataFrame) -> dict:
    v = daily.loc[daily["rank_ic_valid"], "rank_ic"]
    mean = float(v.mean()) if len(v) else float("nan")
    std = float(v.std(ddof=1)) if len(v) > 1 else float("nan")
    return {"valid_day_count": int(len(v)), "mean_rank_ic": mean, "rank_ic_std": std,
            "icir": (mean / std if math.isfinite(std) and std > 0 else float("nan"))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "oos_mean_reproduction")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    per = {}
    labels_by_fold = {}
    for fold in ("F1", "F2", "F3", "F4"):
        labels_by_fold[fold] = load_labels(fold)

    # the frozen baseline's own OOS numbers, from baseline_v2/report.json
    base_rep = json.load(open(ART / "baseline_v2" / "report.json", encoding="utf-8")).get("oos", {})

    for carrier, ws in CARRIERS:
        rep_path = ART / ws / "report.json"
        if not rep_path.is_file():
            per[carrier] = {"skipped": f"{rep_path} absent"}
            continue
        signed = json.load(open(rep_path, encoding="utf-8")).get("oos", {})
        daily_parts, matched_parts = [], []
        for fold in ("F1", "F2", "F3", "F4"):
            cand = find_pred(ws, fold, "candidate")
            base = find_pred(ws, fold, "best_simple")
            if cand is None or base is None:
                daily_parts.append(None)
                continue
            keys = ["signal_date", "ts_code"]
            common = cand.loc[:, keys].merge(base.loc[:, keys], on=keys, how="inner", validate="one_to_one")
            cand_m = cand.merge(common, on=keys, how="inner", validate="one_to_one")
            base_m = base.merge(common, on=keys, how="inner", validate="one_to_one")
            matched_parts.append({"fold": fold, "candidate_rows_raw": int(len(cand)),
                                  "baseline_rows_raw": int(len(base)), "common_keys": int(len(common)),
                                  "candidate_rows_after_match": int(len(cand_m)),
                                  "baseline_rows_after_match": int(len(base_m)),
                                  "rows_removed_from_candidate": int(len(cand) - len(cand_m)),
                                  "rows_removed_from_baseline": int(len(base) - len(base_m))})
            daily_parts.append(daily_rank_ic(cand_m, labels_by_fold[fold]))
        daily_parts = [d for d in daily_parts if d is not None]
        if not daily_parts:
            per[carrier] = {"skipped": "no candidate/baseline prediction pairs found"}
            continue
        daily = pd.concat(daily_parts, ignore_index=True)
        got = summarise(daily)
        per[carrier] = {
            "workspace": ws,
            "signed": {k: signed.get(k) for k in ("valid_day_count", "mean_rank_ic", "icir",
                                                  "baseline_mean_rank_ic", "delta_rank_ic")},
            "reproduced": got,
            "abs_diff_valid_day_count": (int(got["valid_day_count"]) - int(signed["valid_day_count"]))
                                        if signed.get("valid_day_count") is not None else None,
            "abs_diff_mean_rank_ic": (abs(got["mean_rank_ic"] - float(signed["mean_rank_ic"]))
                                      if signed.get("mean_rank_ic") is not None else None),
            "abs_diff_icir": (abs(got["icir"] - float(signed["icir"]))
                              if signed.get("icir") is not None else None),
            "exact_mean_rank_ic": (got["mean_rank_ic"] == float(signed["mean_rank_ic"]))
                                  if signed.get("mean_rank_ic") is not None else None,
            "mean_rank_ic_hex_agreement": (
                float(got["mean_rank_ic"]).hex() == float(signed["mean_rank_ic"]).hex())
                if signed.get("mean_rank_ic") is not None else None,
            "row_matching": matched_parts,
        }
        print(f"[{carrier}] signed valid_day_count={signed.get('valid_day_count')} "
              f"mean={signed.get('mean_rank_ic')!r}", flush=True)
        print(f"           reproduced valid_day_count={got['valid_day_count']} "
              f"mean={got['mean_rank_ic']!r} icir={got['icir']!r}", flush=True)
        print(f"           |diff mean|={per[carrier]['abs_diff_mean_rank_ic']:.3e}  "
              f"hex-identical={per[carrier]['mean_rank_ic_hex_agreement']}  "
              f"|diff icir|={per[carrier]['abs_diff_icir']:.3e}", flush=True)
        for m in matched_parts:
            print(f"             {m['fold']}: candidate {m['candidate_rows_raw']} -> "
                  f"{m['candidate_rows_after_match']} (removed {m['rows_removed_from_candidate']}), "
                  f"baseline {m['baseline_rows_raw']} -> {m['baseline_rows_after_match']} "
                  f"(removed {m['rows_removed_from_baseline']})", flush=True)

    result = {
        "question": ("Does an independent re-implementation that reads only the frozen parquet artifacts "
                     "reproduce each carrier's signed OOS mean_rank_ic, valid_day_count and icir exactly, "
                     "once _match_prediction_coverage's row-level inner join is implemented?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False,
        "repository_module_imported_or_executed": False,
        "criteria": {"minimum_cross_section": MIN_CROSS_SECTION, "coverage_window": [COV_LO, COV_HI],
                     "unique_score_count_floor": MIN_UNIQUE, "score_std_floor": MIN_STD,
                     "source": "evaluation/gates.py:35 via evaluation/rank_ic.py:96-107"},
        "row_matching_source": "evaluation/model_evaluator.py:59-64 -> _match_prediction_coverage at :140-168",
        "frozen_baseline_report_oos": base_rep,
        "carriers": per,
        "summary": {
            "n_carriers_reproduced_hex_identical": sum(
                1 for v in per.values() if v.get("mean_rank_ic_hex_agreement") is True),
            "n_carriers_measured": sum(1 for v in per.values() if "reproduced" in v),
            "max_abs_diff_mean_rank_ic": max(
                (v["abs_diff_mean_rank_ic"] for v in per.values()
                 if v.get("abs_diff_mean_rank_ic") is not None), default=None),
            "all_valid_day_counts_equal_945": all(
                v.get("reproduced", {}).get("valid_day_count") == 945
                for v in per.values() if "reproduced" in v),
        },
    }
    out = args.out / "oos_mean_reproduction.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out}")
    print(json.dumps(result["summary"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
