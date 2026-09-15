#!/usr/bin/env python3
"""Read-only probe: sweep the coverage floor on the CANDIDATE side, and price §AY's three paths.

The decision sheet's §3 sweeps the 0.90 coverage floor, but only for the baseline artifact: it shows
that raising the floor by 0.005 takes G6 from 0.975232 to 0.931889 and fails every candidate. §BQ then
measured the other half of the picture -- M2 scored on its own rows keeps all 242 F1 days at
G6 = 1.0000000000, and only loses a day once _match_prediction_coverage intersects it with a baseline
whose own coverage is already marginal, with the stacking close to multiplicative at a median ratio of
0.994613.

Two questions follow that neither §3 nor §BQ answers, and both are inputs to a decision the owner has to
make:

  1. On the candidate side, what does the floor sweep look like, and how far would the floor have to be
     LOWERED before M2 clears the 0.95 gate? §3 only ever moved it up, and only for the baseline.
  2. §AY ranked its three paths by cost -- path 1 relaxes the guard, path 2 changes
     minimum_valid_history_length (expensive, because inference/builder.py:160-181 carries the same
     frozen constant), path 3 changes _scoreable_label_keys (invalidates every signed number). But §AY
     priced none of them by benefit. This probe measures the benefit: how much of M2's own row deficit
     has to be recovered, at the current floor, for M2 to pass G6.

The projection is validated before it is used: the same median-ratio rule is applied to F1, where M2's
matched coverage is OBSERVED rather than projected, and the projected day count is compared with the
observed one. A projection that cannot reproduce the fold it was fitted on is not used to price anything.

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
FOLDS = ("F1", "F2", "F3", "F4")
LABEL_WORKSPACE = "m1_full_v1"
KEYS = ["signal_date", "ts_code"]
CURRENT_FLOOR = 0.90
G6_GATE = 0.95
TOTAL_OOS_DAYS = 969


def load_labels(fold: str) -> pd.DataFrame:
    paths = sorted(glob.glob(str(ART / LABEL_WORKSPACE / "datasets" / FOLD_DATASET[fold]
                                 / "split=test" / "*" / "*.parquet")))
    frame = pd.concat([pd.read_parquet(p, columns=["signal_date", "ts_code", "raw_return_5d", "label_valid"])
                       for p in paths], ignore_index=True)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
    frame["label_valid"] = frame["label_valid"].fillna(False).astype(bool)
    frame["raw_return_5d"] = pd.to_numeric(frame["raw_return_5d"], errors="coerce")
    return frame


def coverage(pred: pd.DataFrame, labels: pd.DataFrame) -> pd.Series:
    joined = labels.merge(pred, on=KEYS, how="left", validate="one_to_one")
    out = {}
    for d, g in joined.groupby("signal_date", sort=True):
        vl = g["label_valid"].to_numpy() & np.isfinite(g["raw_return_5d"].to_numpy())
        fs = vl & np.isfinite(g["score"].to_numpy(dtype=float))
        vlc = int(vl.sum())
        out[d] = (int(fs.sum()) / vlc) if vlc > 0 else 0.0
    return pd.Series(out).sort_index()


def read_pred(path: Path) -> pd.DataFrame:
    f = pd.read_parquet(path, columns=["signal_date", "ts_code", "score"])
    f["signal_date"] = pd.to_datetime(f["signal_date"], errors="raise").dt.date
    f["score"] = pd.to_numeric(f["score"], errors="coerce")
    return f


def get_pred(workspace: str, fold: str, kind: str) -> pd.DataFrame | None:
    for man in sorted(glob.glob(str(ART / workspace / "predictions" / "*" / "manifest.json"))):
        md = json.load(open(man, encoding="utf-8")).get("metadata", {})
        if md.get("kind") == kind and md.get("fold_id") == fold:
            return read_pred(Path(man).parent / "scores.parquet")
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "m2_g6_floor_sweep")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    base_cov, m2_own_cov, m2_matched_cov = {}, {}, {}
    ratios = {}
    for fold in FOLDS:
        labels = load_labels(fold)
        bp = read_pred(ART / "baseline_v2" / "baseline_oos" / f"{fold}.parquet")
        base_cov[fold] = coverage(bp, labels)
        m2c = get_pred("m2_v1", fold, "candidate")
        m2b = get_pred("m2_v1", fold, "best_simple")
        if m2c is not None and m2b is not None:
            common = m2c.loc[:, KEYS].merge(m2b.loc[:, KEYS], on=KEYS, how="inner", validate="one_to_one")
            m2m = m2c.merge(common, on=KEYS, how="inner", validate="one_to_one")
            m2_own_cov[fold] = coverage(m2c, labels)
            m2_matched_cov[fold] = coverage(m2m, labels)
            ratios[fold] = float((m2_matched_cov[fold] / base_cov[fold]).median())
        del labels
    observed_folds = sorted(m2_matched_cov)
    assert observed_folds, "no fold has M2 predictions to fit the deficit on"
    fit_fold = observed_folds[0]
    r_hat = ratios[fit_fold]

    # ---- VALIDATE the projection on the fold it was fitted on before using it anywhere
    projected_fit = base_cov[fit_fold] * r_hat
    observed_fit = m2_matched_cov[fit_fold]
    validation = {
        "fitted_on_fold": fit_fold,
        "median_ratio_matched_over_baseline": r_hat,
        "projected_vs_observed_coverage": {
            "pearson_corr": float(np.corrcoef(projected_fit, observed_fit)[0, 1]),
            "mean_absolute_difference": float((projected_fit - observed_fit).abs().mean()),
            "max_absolute_difference": float((projected_fit - observed_fit).abs().max()),
        },
        "days_below_current_floor": {
            "observed": int((observed_fit < CURRENT_FLOOR).sum()),
            "projected": int((projected_fit < CURRENT_FLOOR).sum()),
            "agree": bool((observed_fit < CURRENT_FLOOR).sum() == (projected_fit < CURRENT_FLOOR).sum()),
        },
        "all_observed_folds": {f: {"ratio": ratios[f],
                                    "observed_days_below_floor": int((m2_matched_cov[f] < CURRENT_FLOOR).sum())}
                                for f in observed_folds},
    }
    assert validation["days_below_current_floor"]["agree"], \
        "the projection does not reproduce the day count on the fold it was fitted on; do not use it"

    # ---- the candidate-side coverage series: observed where available, projected elsewhere
    m2_cov = {}
    for fold in FOLDS:
        m2_cov[fold] = m2_matched_cov[fold] if fold in m2_matched_cov else base_cov[fold] * r_hat
    m2_all = pd.concat([m2_cov[f].rename(f) for f in FOLDS], axis=1).stack().sort_index()
    base_all = pd.concat([base_cov[f].rename(f) for f in FOLDS], axis=1).stack().sort_index()
    assert len(m2_all) == TOTAL_OOS_DAYS == len(base_all), f"expected {TOTAL_OOS_DAYS} days"

    def g6_at(series: pd.Series, floor: float) -> dict:
        below = int((series < floor).sum())
        valid = TOTAL_OOS_DAYS - below
        g6 = valid / TOTAL_OOS_DAYS
        return {"floor": floor, "days_below": below, "valid_days": valid, "G6": g6,
                "passes_0.95": g6 >= G6_GATE}

    sweep = [g6_at(m2_all, f) for f in
             (0.860, 0.870, 0.875, 0.880, 0.885, 0.890, 0.895, 0.900, 0.905, 0.910, 0.920)]
    sweep_baseline = [g6_at(base_all, f) for f in
                      (0.860, 0.870, 0.875, 0.880, 0.885, 0.890, 0.895, 0.900, 0.905, 0.910, 0.920)]

    # highest floor at which M2 still passes, on a fine grid
    fine = np.arange(0.8000, 0.9601, 0.0001)
    passing = [float(f) for f in fine if (TOTAL_OOS_DAYS - int((m2_all < f).sum())) / TOTAL_OOS_DAYS >= G6_GATE]
    max_passing_floor_m2 = max(passing) if passing else None
    passing_b = [float(f) for f in fine if (TOTAL_OOS_DAYS - int((base_all < f).sum())) / TOTAL_OOS_DAYS >= G6_GATE]
    max_passing_floor_baseline = max(passing_b) if passing_b else None
    days_needed = math.ceil(G6_GATE * TOTAL_OOS_DAYS)
    max_days_below = TOTAL_OOS_DAYS - days_needed

    # ---- price §AY's path 2: how much of M2's own deficit must be recovered to pass at floor 0.90?
    deficit_pricing = []
    for r in np.arange(0.9900, 1.00001, 0.0002):
        proj = pd.concat([(base_cov[f] * r).rename(f) if f not in m2_matched_cov else m2_matched_cov[f]
                          for f in FOLDS], axis=1).stack().sort_index()
        below = int((proj < CURRENT_FLOOR).sum())
        deficit_pricing.append({
            "ratio_matched_over_baseline": float(r),
            "implied_row_deficit_share": float(1 - r),
            "days_below_0.90": below,
            "G6": (TOTAL_OOS_DAYS - below) / TOTAL_OOS_DAYS,
            "passes": (TOTAL_OOS_DAYS - below) / TOTAL_OOS_DAYS >= G6_GATE,
        })
    first_pass = next((d for d in deficit_pricing if d["passes"]), None)
    # fully recovering the deficit means the candidate's coverage equals the baseline's
    no_deficit = g6_at(base_all, CURRENT_FLOOR)

    # ---- price the decision sheet's own options on the SAME axis. Option 1 (backfill the STAR industry
    # labels as a union) and Option 2 (drop the two industry factors) both raise the BASELINE's per-day
    # coverage; since the stacking measured above is multiplicative, raising the baseline raises the
    # candidate by the same factor. Their measured worst days are recorded in the decision sheet, so the
    # candidate-side G6 follows from the minimum alone -- no per-day series needed, because if the
    # minimum stays above the floor then no day can fall below it.
    OPTION_WORST_DAYS = {
        "status_quo": {"baseline_coverage_min": float(base_all.min()),
                       "source": "this probe, measured over all 969 OOS days"},
        "option_1_union_backfill": {"baseline_coverage_min": min(0.976751, 0.971115),
                                    "source": ("decision sheet: 0.976751 worst day over the 420 "
                                               "pre-cutover days after the union backfill; the 549 "
                                               "post-cutover days are unchanged at >= 0.971115")},
        "option_2_drop_the_two_industry_factors": {"baseline_coverage_min": 0.971370,
                                                   "source": ("decision sheet: measured coverage min over "
                                                              "all 969 days with 357 and 471 both removed")},
    }
    option_scenarios = {}
    for name, v in OPTION_WORST_DAYS.items():
        bmin = v["baseline_coverage_min"]
        m2min = bmin * r_hat
        below = 0 if m2min >= CURRENT_FLOOR else None
        option_scenarios[name] = {
            "baseline_coverage_min": bmin,
            "implied_M2_matched_coverage_min": m2min,
            "M2_days_below_floor": below,
            "M2_G6": (1.0 if below == 0 else None),
            "M2_passes_G6": (True if below == 0 else None),
            "margin_above_floor": m2min - CURRENT_FLOOR,
            "basis": v["source"],
            "note": ("computed from the minimum alone: if the worst day stays above the floor then no day "
                     "falls below it, so G6 is exactly 1.0 and no per-day series is required"
                     if below == 0 else
                     "the minimum is below the floor, so the day count needs the per-day series above"),
        }

    result = {
        "question": ("On the candidate side, how far would the coverage floor have to move for M2 to "
                     "clear G6, and how much of M2's own row deficit would have to be recovered to clear "
                     "it at the current floor -- i.e. what is the BENEFIT side of §AY's three paths, "
                     "which §AY priced only by cost?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False,
        "repository_module_imported_or_executed": False,
        "inputs": {"total_oos_days": TOTAL_OOS_DAYS, "current_floor": CURRENT_FLOOR, "G6_gate": G6_GATE,
                   "folds_with_observed_m2_predictions": observed_folds,
                   "folds_projected": [f for f in FOLDS if f not in observed_folds],
                   "projection_rule": "candidate_coverage = baseline_coverage x median(matched/baseline)"},
        "projection_validation": validation,
        "floor_sweep_candidate_side": sweep,
        "floor_sweep_baseline_side_for_contrast": sweep_baseline,
        "max_floor_at_which_each_still_passes": {
            "M2_candidate": max_passing_floor_m2,
            "baseline_own_rows": max_passing_floor_baseline,
            "current_floor": CURRENT_FLOOR,
            "days_needed_to_pass": days_needed,
            "max_days_below_floor_to_pass": max_days_below,
        },
        "at_the_current_floor": {
            "M2_candidate": g6_at(m2_all, CURRENT_FLOOR),
            "baseline_own_rows": g6_at(base_all, CURRENT_FLOOR),
            "M2_if_its_own_deficit_were_fully_recovered": no_deficit,
        },
        "pricing_AY_path_2_recover_the_row_deficit": {
            "what_it_means": ("§AY's path 2 changes minimum_valid_history_length, which is what creates "
                              "M2's row deficit. This prices the benefit: the smallest deficit recovery "
                              "that clears G6 at the UNCHANGED floor of 0.90."),
            "measured_deficit_at_the_fitted_fold": float(1 - r_hat),
            "grid": deficit_pricing,
            "first_ratio_that_passes": first_pass,
            "verdict": ("the whole deficit must be recovered" if first_pass is None or
                        first_pass["ratio_matched_over_baseline"] >= 0.99999 else
                        f"a ratio of {first_pass['ratio_matched_over_baseline']:.5f} suffices, i.e. "
                        f"recovering {100*(first_pass['ratio_matched_over_baseline']-r_hat)/(1-r_hat):.1f}% "
                        f"of the measured deficit"),
        },
        "decision_sheet_options_priced_on_the_candidate_side": {
            "method": ("the stacking is multiplicative at the measured ratio, so each option's measured "
                       "baseline worst day maps straight onto a candidate worst day"),
            "ratio_used": r_hat,
            "scenarios": option_scenarios,
        },
        "candidate_coverage_percentiles": {
            "min": float(m2_all.min()), "p01": float(m2_all.quantile(0.01)),
            "p05": float(m2_all.quantile(0.05)), "median": float(m2_all.median()),
            "n_below_0.90": int((m2_all < 0.90).sum()),
            "n_in_0.90_to_0.905": int(((m2_all >= 0.90) & (m2_all < 0.905)).sum()),
            "n_in_0.895_to_0.90": int(((m2_all >= 0.895) & (m2_all < 0.90)).sum()),
        },
        "baseline_coverage_percentiles": {
            "min": float(base_all.min()), "p01": float(base_all.quantile(0.01)),
            "median": float(base_all.median()),
            "n_below_0.90": int((base_all < 0.90).sum()),
            "n_in_0.90_to_0.905": int(((base_all >= 0.90) & (base_all < 0.905)).sum()),
        },
    }
    out = args.out / "m2_g6_floor_sweep_and_deficit_pricing.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    v = validation
    print(f"\nprojection fitted on {v['fitted_on_fold']} at ratio {v['median_ratio_matched_over_baseline']:.6f}")
    print(f"  validation on the fitted fold: corr {v['projected_vs_observed_coverage']['pearson_corr']:.6f}, "
          f"max |diff| {v['projected_vs_observed_coverage']['max_absolute_difference']:.6f}, "
          f"days below floor observed {v['days_below_current_floor']['observed']} vs "
          f"projected {v['days_below_current_floor']['projected']} "
          f"-> {'AGREE' if v['days_below_current_floor']['agree'] else 'DISAGREE'}")
    print(f"\n{'floor':>8}{'M2 days<':>10}{'M2 G6':>12}{'M2':>7}{'base days<':>12}{'base G6':>12}{'base':>7}")
    for a_, b_ in zip(sweep, sweep_baseline):
        print(f"{a_['floor']:>8.3f}{a_['days_below']:>10}{a_['G6']:>12.6f}"
              f"{'PASS' if a_['passes_0.95'] else 'FAIL':>7}"
              f"{b_['days_below']:>12}{b_['G6']:>12.6f}{'PASS' if b_['passes_0.95'] else 'FAIL':>7}")
    mx = result["max_floor_at_which_each_still_passes"]
    print(f"\nhighest floor that still passes: M2 {mx['M2_candidate']}, baseline {mx['baseline_own_rows']}, "
          f"current {mx['current_floor']}")
    print(f"to pass you need >= {mx['days_needed_to_pass']} valid days, i.e. <= {mx['max_days_below_floor_to_pass']} below")
    cur = result["at_the_current_floor"]
    print(f"\nat the CURRENT floor 0.90:")
    for k, val in cur.items():
        print(f"  {k:<48} days_below={val['days_below']:>3} G6={val['G6']:.6f} "
              f"{'PASS' if val['passes_0.95'] else 'FAIL'}")
    print("\ndecision sheet options priced on the candidate side (ratio %.6f):" % r_hat)
    print(f"  {'scenario':<40}{'base cov min':>14}{'M2 cov min':>12}{'M2 days<':>10}{'M2 G6':>10}{'verdict':>9}")
    for nm, sc in result["decision_sheet_options_priced_on_the_candidate_side"]["scenarios"].items():
        g6 = sc["M2_G6"]; dl = sc["M2_days_below_floor"]; vd = sc["M2_passes_G6"]
        print(f"  {nm:<40}{sc['baseline_coverage_min']:>14.6f}{sc['implied_M2_matched_coverage_min']:>12.6f}"
              f"{('0' if dl == 0 else 'needs per-day'):>10}{('1.000000' if g6 == 1.0 else '-'):>10}"
              f"{('PASS' if vd else ('FAIL' if vd is False else '-')):>9}")
    pr = result["pricing_AY_path_2_recover_the_row_deficit"]
    print(f"\npricing §AY path 2 (measured deficit {100*pr['measured_deficit_at_the_fitted_fold']:.4f}% of rows):")
    print(f"  verdict: {pr['verdict']}")
    if pr["first_ratio_that_passes"]:
        f_ = pr["first_ratio_that_passes"]
        print(f"  first passing ratio {f_['ratio_matched_over_baseline']:.5f} -> days_below {f_['days_below_0.90']}, "
              f"G6 {f_['G6']:.6f}")
    for d_ in pr["grid"][::5]:
        print(f"    ratio {d_['ratio_matched_over_baseline']:.5f} (deficit {100*d_['implied_row_deficit_share']:.3f}%) "
              f"-> days_below {d_['days_below_0.90']:>3} G6 {d_['G6']:.6f} {'PASS' if d_['passes'] else 'FAIL'}")


if __name__ == "__main__":
    main()
