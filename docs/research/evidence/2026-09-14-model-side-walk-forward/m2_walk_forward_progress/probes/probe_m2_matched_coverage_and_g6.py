#!/usr/bin/env python3
"""Read-only probe: M2's F1 coverage as the evaluator actually sees it, and the G6 it implies.

§AX projected, from source rules plus the frozen dataset artifacts, that M2 would lose 64 OOS days to
the 0.90 coverage floor and land at G6 = 0.9339525283797729 against a 0.95 gate. That was a projection:
§AX.8 says explicitly that it never triggered the ValueError and never observed the guard. M2's F1
predictions now exist on disk, so the mechanism can be observed on one fold instead of inferred on four.

What this probe measures, and what it found:

  * M2's OWN coverage against the full label frame is comfortable -- min 0.985229, median 0.992524,
    ZERO days below 0.90, so scored on its own rows M2 would keep all 242 F1 days.
  * After evaluation/model_evaluator.py:_match_prediction_coverage inner-joins M2's rows with the
    baseline's, M2's coverage becomes min 0.890015, median 0.921050, and ONE day falls below the floor
    -- the same day the baseline itself fails (2022-07-29). So the failure is inherited, then deepened.
  * The stacking is close to multiplicative: corr(matched M2 coverage, baseline coverage) = 0.992728 and
    the median ratio matched/baseline is 0.994613, against M2's own median coverage of 0.992524.
  * Feeding that one measured deficit into the baseline's coverage on the other three folds projects 65
    days below the floor and G6 = 0.9329205366 -- one day and 0.001 away from §AX's independently
    derived 64 and 0.9339525283797729. Two unrelated derivations agreeing to within one day out of 969
    is the point of running this.

It also checks the pre-registration's expected coverage values against what was observed, and finds that
the pre-registered min and median for F1 are the BASELINE's own coverage figures to ten decimal places,
so that clause cannot detect a candidate which scores fewer rows than the baseline.

Safety: reads parquet and JSON artifacts in place. No database access, no weight loading, no repository
module imported or executed, nothing written into any artifact directory, no process signalled.
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
KEYS = ["signal_date", "ts_code"]
FLOOR = 0.90
G6_GATE = 0.95


def load_labels(fold: str) -> pd.DataFrame:
    paths = sorted(glob.glob(str(ART / LABEL_WORKSPACE / "datasets" / FOLD_DATASET[fold]
                                 / "split=test" / "*" / "*.parquet")))
    frame = pd.concat([pd.read_parquet(p, columns=["signal_date", "ts_code", "raw_return_5d", "label_valid"])
                       for p in paths], ignore_index=True)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
    frame["label_valid"] = frame["label_valid"].fillna(False).astype(bool)
    frame["raw_return_5d"] = pd.to_numeric(frame["raw_return_5d"], errors="coerce")
    return frame


def get_pred(workspace: str, fold: str, kind: str) -> pd.DataFrame | None:
    for man in sorted(glob.glob(str(ART / workspace / "predictions" / "*" / "manifest.json"))):
        md = json.load(open(man, encoding="utf-8")).get("metadata", {})
        if md.get("kind") == kind and md.get("fold_id") == fold:
            f = pd.read_parquet(Path(man).parent / "scores.parquet", columns=["signal_date", "ts_code", "score"])
            f["signal_date"] = pd.to_datetime(f["signal_date"], errors="raise").dt.date
            f["score"] = pd.to_numeric(f["score"], errors="coerce")
            return f
    return None


def coverage_profile(pred: pd.DataFrame, labels: pd.DataFrame) -> pd.DataFrame:
    """Per-day coverage exactly as evaluation/rank_ic.py:77-84 computes it."""
    joined = labels.merge(pred, on=KEYS, how="left", validate="one_to_one")
    rows = []
    for d, g in joined.groupby("signal_date", sort=True):
        vl = g["label_valid"].to_numpy() & np.isfinite(g["raw_return_5d"].to_numpy())
        fs = vl & np.isfinite(g["score"].to_numpy(dtype=float))
        vlc = int(vl.sum())
        n = int(fs.sum())
        rows.append({"signal_date": d, "coverage": (n / vlc if vlc > 0 else 0.0),
                     "scored_count": n, "valid_label_count": vlc})
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "m2_matched_coverage_g6")
    ap.add_argument("--fold", default="F1", help="the only fold where M2 has predictions today")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fold = args.fold

    labels = load_labels(fold)
    m2 = get_pred("m2_v1", fold, "candidate")
    base = get_pred("m2_v1", fold, "best_simple")
    assert m2 is not None and base is not None, f"M2 {fold} predictions not found"

    common = m2.loc[:, KEYS].merge(base.loc[:, KEYS], on=KEYS, how="inner", validate="one_to_one")
    m2m = m2.merge(common, on=KEYS, how="inner", validate="one_to_one")
    basem = base.merge(common, on=KEYS, how="inner", validate="one_to_one")

    own = coverage_profile(m2, labels)
    matched = coverage_profile(m2m, labels)
    bl = coverage_profile(base, labels)
    blm = coverage_profile(basem, labels)

    ratio = (matched["coverage"] / bl["coverage"]).replace([np.inf, -np.inf], np.nan)
    corr = float(np.corrcoef(matched["coverage"], bl["coverage"])[0, 1])
    deficit = float(ratio.median())

    def prof(c: pd.DataFrame) -> dict:
        return {"days": int(len(c)), "coverage_min": float(c["coverage"].min()),
                "coverage_median": float(c["coverage"].median()), "coverage_max": float(c["coverage"].max()),
                "days_below_floor": int((c["coverage"] < FLOOR).sum()),
                "days_at_or_above_floor": int((c["coverage"] >= FLOOR).sum()),
                "G6": float((c["coverage"] >= FLOOR).mean())}

    failing = matched.loc[matched["coverage"] < FLOOR, "signal_date"].astype(str).tolist()

    # project the other folds with the deficit measured on this one
    thr = FLOOR / deficit
    proj = {}
    tot_days = tot_below = 0
    for f in ("F1", "F2", "F3", "F4"):
        if f == fold:
            below, days = int((matched["coverage"] < FLOOR).sum()), int(len(matched))
            src = "observed"
        else:
            bf = None
            for man in sorted(glob.glob(str(ART / "baseline_v2" / "baseline_oos" / f"{f}.parquet"))):
                bp = pd.read_parquet(man, columns=["signal_date", "ts_code", "score"])
                bp["signal_date"] = pd.to_datetime(bp["signal_date"]).dt.date
                bp["score"] = pd.to_numeric(bp["score"], errors="coerce")
                bf = coverage_profile(bp, load_labels(f))
            assert bf is not None, f"baseline OOS predictions missing for {f}"
            below = int((bf["coverage"] < thr).sum())
            days = int(len(bf))
            src = f"projected at threshold {thr:.6f}"
        proj[f] = {"days": days, "days_below_floor": below, "valid_days": days - below, "basis": src,
                   "baseline_days_below_0.90": None}
        tot_days += days
        tot_below += below
    g6_proj = (tot_days - tot_below) / tot_days

    # pre-registration comparison
    pre_path = (Path(__file__).resolve().parent.parent / "measurements" / "m2_f1_grid_and_standing.json")
    pre = None
    if pre_path.is_file():
        pre = json.load(open(pre_path, encoding="utf-8")).get(
            "preregistration_status", {}).get("expected_per_fold", {}).get(fold)

    result = {
        "question": ("What is M2's OOS coverage as the evaluator actually computes it, which mechanism "
                     "produces the days it loses, and what G6 does that imply for the remaining folds?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False,
        "repository_module_imported_or_executed": False,
        "fold_observed": fold,
        "row_matching": {
            "source": "evaluation/model_evaluator.py:59-64 -> _match_prediction_coverage at :140-168",
            "m2_candidate_rows_raw": int(len(m2)), "baseline_rows_raw": int(len(base)),
            "common_keys": int(len(common)), "m2_rows_after_match": int(len(m2m)),
            "m2_rows_removed": int(len(m2) - len(m2m)),
            "baseline_rows_removed": int(len(base) - len(basem)),
        },
        "coverage_profiles": {
            "M2_candidate_on_its_own_rows": prof(own),
            "M2_candidate_after_row_matching": prof(matched),
            "baseline_on_its_own_rows": prof(bl),
            "baseline_after_row_matching": prof(blm),
        },
        "the_stack_is_close_to_multiplicative": {
            "pearson_corr_matched_m2_vs_baseline_coverage": corr,
            "median_ratio_matched_over_baseline": deficit,
            "m2_own_coverage_median": float(own["coverage"].median()),
            "reading": ("M2 loses about 0.7% of rows on its own and the baseline is already at about "
                        "0.925 median coverage, so the matched product sits about 0.5% below the "
                        "baseline -- enough to move days across a floor they were already near"),
        },
        "days_m2_loses_after_matching": {
            "count": len(failing), "dates": failing,
            "same_days_the_baseline_also_fails": sorted(
                set(failing) & set(bl.loc[bl["coverage"] < FLOOR, "signal_date"].astype(str))),
        },
        "projection_to_all_four_folds": {
            "effective_threshold_after_the_measured_deficit": thr,
            "per_fold": proj,
            "total_days": tot_days, "total_days_below_floor": tot_below,
            "projected_valid_days": tot_days - tot_below,
            "projected_G6": g6_proj,
            "G6_gate": G6_GATE, "would_pass": g6_proj >= G6_GATE,
            "independent_comparison": {
                "AX_rule_based_projection_days_below_floor": 64,
                "AX_rule_based_projection_G6": 0.9339525283797729,
                "this_probe_deficit_based_days_below_floor": tot_below,
                "this_probe_deficit_based_G6": g6_proj,
                "difference_in_days": tot_below - 64,
                "difference_in_G6": g6_proj - 0.9339525283797729,
                "note": ("§AX derived its number from the temporal eligibility rule in "
                         "training/temporal_batches.py:186-192 applied to the frozen datasets; this probe "
                         "derives its from M2's actual F1 predictions and the baseline's measured "
                         "coverage. Neither used the other's input."),
            },
        },
        "preregistration_check": {
            "preregistered_for_this_fold": pre,
            "observed_matched_coverage_min": float(matched["coverage"].min()),
            "observed_matched_coverage_median": float(matched["coverage"].median()),
            "baseline_own_coverage_min": float(bl["coverage"].min()),
            "baseline_own_coverage_median": float(bl["coverage"].median()),
            "preregistered_min_equals_the_baseline_not_the_candidate": (
                bool(pre) and abs(float(pre["expected_coverage_min"]) - float(bl["coverage"].min())) < 1e-12),
            "preregistered_median_equals_the_baseline_not_the_candidate": (
                bool(pre) and abs(float(pre["expected_coverage_median"]) - float(bl["coverage"].median())) < 1e-12),
            "candidate_is_below_what_was_preregistered_by": (
                float(pre["expected_coverage_min"]) - float(matched["coverage"].min())) if pre else None,
        },
    }
    out = args.out / f"m2_matched_coverage_and_g6_{fold}.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    print(f"\nrow matching on {fold}: M2 {len(m2)} -> {len(m2m)} (removed {len(m2)-len(m2m)}), "
          f"baseline {len(base)} -> {len(basem)} (removed {len(base)-len(basem)})")
    print(f"\n{'series':<38}{'cov min':>10}{'cov med':>10}{'days<0.90':>11}{'G6':>14}")
    for nm, key in (("M2 candidate, own rows", "M2_candidate_on_its_own_rows"),
                    ("M2 candidate, AFTER row matching", "M2_candidate_after_row_matching"),
                    ("baseline, own rows", "baseline_on_its_own_rows"),
                    ("baseline, after row matching", "baseline_after_row_matching")):
        v = result["coverage_profiles"][key]
        print(f"{nm:<38}{v['coverage_min']:>10.6f}{v['coverage_median']:>10.6f}"
              f"{v['days_below_floor']:>11}{v['G6']:>14.10f}")
    st = result["the_stack_is_close_to_multiplicative"]
    print(f"\nstacking: corr={st['pearson_corr_matched_m2_vs_baseline_coverage']:.6f} "
          f"median matched/baseline={st['median_ratio_matched_over_baseline']:.6f} "
          f"M2 own median={st['m2_own_coverage_median']:.6f}")
    print(f"days M2 loses after matching: {result['days_m2_loses_after_matching']['dates']} "
          f"(also failed by the baseline: {result['days_m2_loses_after_matching']['same_days_the_baseline_also_fails']})")
    pr = result["projection_to_all_four_folds"]
    print(f"\nprojection: threshold {pr['effective_threshold_after_the_measured_deficit']:.6f} -> "
          f"{pr['total_days_below_floor']} days below floor of {pr['total_days']}, "
          f"G6 = {pr['projected_G6']:.10f} (gate {G6_GATE}) -> {'PASS' if pr['would_pass'] else 'FAIL'}")
    for f, v in pr["per_fold"].items():
        print(f"   {f}: {v['days_below_floor']:>3} below / {v['days']} days  ({v['basis']})")
    ic = pr["independent_comparison"]
    print(f"   §AX rule-based: {ic['AX_rule_based_projection_days_below_floor']} days, "
          f"G6 {ic['AX_rule_based_projection_G6']:.10f}")
    print(f"   difference: {ic['difference_in_days']:+d} day(s), G6 {ic['difference_in_G6']:+.10f}")
    pc = result["preregistration_check"]
    print(f"\npreregistration: min equals the BASELINE not the candidate = "
          f"{pc['preregistered_min_equals_the_baseline_not_the_candidate']}, "
          f"median likewise = {pc['preregistered_median_equals_the_baseline_not_the_candidate']}")
    print(f"   candidate is below the preregistered min by {pc['candidate_is_below_what_was_preregistered_by']:.6f}")


if __name__ == "__main__":
    main()
