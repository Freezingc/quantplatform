#!/usr/bin/env python3
"""Read-only probe: how many OOS days does require_score_health=True actually remove, and by which criterion?

§BP.3-续 read the three numeric criteria out of evaluation/gates.py:35 --
`0.90 <= coverage <= 1.0`, `unique_score_count >= 10`, `score_std > 1e-6` -- and closed the half of the
§AW.8 / §AX.8 open item that asks *what* the criteria are. The other half asks *whether they
independently change the day count*, and that needs numbers rather than source reading. This probe
produces them, on every carrier that has finished all four folds.

Why it matters: the OOS acceptance path passes require_score_health=True while both validation paths
pass False (§BP.1's table). So a day can count toward a validation mean_rank_ic and be dropped from an
OOS mean_rank_ic. Until now nobody had measured how often that happens, or whether the mean moves when
it does. If it never happens, the two estimators coincide in practice on this data and §BP.3 stays a
source-level caveat. If it happens, the size of the shift is a number the owner needs.

The per-day arithmetic is re-implemented line by line from evaluation/rank_ic.py:60-107 and
evaluation/gates.py:25-35 rather than imported, following this pack's rule that no repository module is
executed. The rank correlation is reproduced the same way the repository does it -- pandas
rank(method="average") on both sides then a Pearson correlation -- so the values are comparable to the
ones already in the signed reports rather than merely similar.

Safety: reads parquet artifacts in place. No database access, no weight loading, no repository module
imported or executed, nothing written into any artifact directory.
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
LABEL_WORKSPACE = "m1_full_v1"          # every carrier consumes the same four frozen datasets
MIN_CROSS_SECTION = 30                  # evaluation/rank_ic.py default, and what both paths pass
COV_LO, COV_HI = 0.90, 1.0              # evaluation/gates.py:35
MIN_UNIQUE = 10                         # evaluation/gates.py:35
MIN_STD = 1e-6                          # evaluation/gates.py:35

# carrier label -> (workspace, how to find its OOS candidate predictions)
CARRIERS = (
    ("baseline_B2", "baseline_v2", "baseline_oos"),
    ("M0R", "m0r_v2", "predictions"),
    ("M0R-replicate", "m0r_v3", "predictions"),
    ("M0-base", "m0_v2", "predictions"),
    ("M0-refined", "m0_v3", "predictions"),
    ("M1-full", "m1_full_v1", "predictions"),
    ("M2", "m2_v1", "predictions"),
)


def load_labels(fold: str) -> pd.DataFrame:
    ds = FOLD_DATASET[fold]
    paths = sorted(glob.glob(str(ART / LABEL_WORKSPACE / "datasets" / ds / "split=test" / "*" / "*.parquet")))
    assert paths, f"no test split for {fold}"
    frame = pd.concat(
        [pd.read_parquet(p, columns=["signal_date", "ts_code", "raw_return_5d", "label_valid"]) for p in paths],
        ignore_index=True)
    frame["signal_date"] = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
    frame["label_valid"] = frame["label_valid"].fillna(False).astype(bool)
    frame["raw_return_5d"] = pd.to_numeric(frame["raw_return_5d"], errors="coerce")
    return frame


def find_predictions(workspace: str, mode: str, fold: str) -> str | None:
    if mode == "baseline_oos":
        p = ART / workspace / "baseline_oos" / f"{fold}.parquet"
        return str(p) if p.is_file() else None
    for man in sorted(glob.glob(str(ART / workspace / "predictions" / "*" / "manifest.json"))):
        md = json.load(open(man, encoding="utf-8")).get("metadata", {})
        if md.get("kind") == "candidate" and md.get("fold_id") == fold:
            return str(Path(man).parent / "scores.parquet")
    return None


def per_day(labels: pd.DataFrame, pred_path: str) -> pd.DataFrame:
    """Reproduce evaluation/rank_ic.py:60-107 for one carrier-fold, keeping every diagnostic."""
    pred = pd.read_parquet(pred_path, columns=["signal_date", "ts_code", "score"])
    pred["signal_date"] = pd.to_datetime(pred["signal_date"], errors="raise").dt.date
    pred["score"] = pd.to_numeric(pred["score"], errors="coerce")
    assert not pred.duplicated(["signal_date", "ts_code"]).any(), "duplicate prediction keys"
    assert not labels.duplicated(["signal_date", "ts_code"]).any(), "duplicate label keys"
    joined = labels.merge(pred, on=["signal_date", "ts_code"], how="left", validate="one_to_one")

    rows = []
    for d, g in joined.groupby("signal_date", sort=True):
        valid_label = g["label_valid"].to_numpy() & np.isfinite(g["raw_return_5d"].to_numpy())
        score = g["score"].to_numpy(dtype=float)
        lab = g["raw_return_5d"].to_numpy(dtype=float)
        valid_label_count = int(valid_label.sum())
        finite_score = valid_label & np.isfinite(score)
        scored_count = int(finite_score.sum())
        coverage = scored_count / valid_label_count if valid_label_count > 0 else 0.0
        sc = score[finite_score]
        lb = lab[finite_score]
        unique_score_count = int(pd.Series(sc).nunique()) if scored_count else 0
        score_std = float(np.std(sc, ddof=0)) if scored_count else float("nan")

        health_cov = bool(math.isfinite(coverage) and COV_LO <= coverage <= COV_HI)
        health_uniq = bool(unique_score_count >= MIN_UNIQUE)
        health_std = bool(math.isfinite(score_std) and score_std > MIN_STD)
        health_all = health_cov and health_uniq and health_std
        floor_ok = scored_count >= MIN_CROSS_SECTION

        rank_ic = float("nan")
        if floor_ok:
            sr = pd.Series(sc).rank(method="average")
            lr = pd.Series(lb).rank(method="average")
            if sr.nunique(dropna=False) >= 2 and lr.nunique(dropna=False) >= 2:
                cand = sr.corr(lr, method="pearson")
                if pd.notna(cand) and math.isfinite(float(cand)):
                    rank_ic = float(cand)
        rows.append({
            "signal_date": d, "eligible_count": int(len(g)), "valid_label_count": valid_label_count,
            "scored_count": scored_count, "coverage": coverage,
            "unique_score_count": unique_score_count, "score_std": score_std,
            "floor_ok": floor_ok, "health_cov": health_cov, "health_uniq": health_uniq,
            "health_std": health_std, "health_all": health_all,
            "valid_validation_regime": floor_ok,
            "valid_oos_regime": floor_ok and health_all,
            "rank_ic": rank_ic,
            "rank_ic_computable": bool(math.isfinite(rank_ic)),
        })
    return pd.DataFrame(rows)


def summarise(daily: pd.DataFrame) -> dict:
    v = daily["valid_validation_regime"]
    o = daily["valid_oos_regime"]
    dropped = daily[v & ~o]
    ric = daily["rank_ic"]
    mean_v = float(ric[v & ric.notna()].mean()) if (v & ric.notna()).any() else float("nan")
    mean_o = float(ric[o & ric.notna()].mean()) if (o & ric.notna()).any() else float("nan")

    def failing(mask_col: str) -> int:
        return int((v & ~daily[mask_col]).sum())

    combos = {}
    if len(dropped):
        key = (dropped["health_cov"].map({True: "pass", False: "FAIL_coverage"}) + "|"
               + dropped["health_uniq"].map({True: "pass", False: "FAIL_unique<10"}) + "|"
               + dropped["health_std"].map({True: "pass", False: "FAIL_std<=1e-6"}))
        combos = key.value_counts().to_dict()
    return {
        "label_dates": int(len(daily)),
        "days_valid_in_the_validation_regime_floor_only": int(v.sum()),
        "days_valid_in_the_oos_regime_floor_plus_health": int(o.sum()),
        "days_dropped_by_score_health": int(len(dropped)),
        "share_of_floor_valid_days_dropped": float(len(dropped) / v.sum()) if v.sum() else None,
        "days_failing_each_criterion_among_floor_valid": {
            "coverage_outside_0.90_to_1.0": failing("health_cov"),
            "unique_score_count_below_10": failing("health_uniq"),
            "score_std_at_or_below_1e-6": failing("health_std"),
            "any_of_the_three": int((v & ~daily["health_all"]).sum()),
        },
        "failure_combinations_on_dropped_days": {k: int(n) for k, n in combos.items()},
        "mean_rank_ic_validation_regime": mean_v,
        "mean_rank_ic_oos_regime": mean_o,
        "mean_rank_ic_shift_oos_minus_validation": (mean_o - mean_v)
            if (math.isfinite(mean_v) and math.isfinite(mean_o)) else None,
        "rank_ic_on_dropped_days": {
            "n": int(dropped["rank_ic"].notna().sum()),
            "mean": float(dropped.loc[dropped["rank_ic"].notna(), "rank_ic"].mean())
                    if dropped["rank_ic"].notna().any() else None,
            "min": float(dropped["rank_ic"].min()) if len(dropped) else None,
            "max": float(dropped["rank_ic"].max()) if len(dropped) else None,
        },
        "coverage_distribution_on_floor_valid_days": {
            "min": float(daily.loc[v, "coverage"].min()) if v.any() else None,
            "p01": float(daily.loc[v, "coverage"].quantile(0.01)) if v.any() else None,
            "median": float(daily.loc[v, "coverage"].median()) if v.any() else None,
            "max": float(daily.loc[v, "coverage"].max()) if v.any() else None,
        },
        "unique_score_count_on_floor_valid_days": {
            "min": int(daily.loc[v, "unique_score_count"].min()) if v.any() else None,
            "median": float(daily.loc[v, "unique_score_count"].median()) if v.any() else None,
        },
        "score_std_on_floor_valid_days": {
            "min": float(daily.loc[v, "score_std"].min()) if v.any() else None,
            "median": float(daily.loc[v, "score_std"].median()) if v.any() else None,
        },
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "oos_score_health_day_effect")
    ap.add_argument("--per-day-csv", type=Path, default=None,
                    help="also write every per-day diagnostic row (default: alongside the JSON)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    per, all_daily = {}, []
    for fold in ("F1", "F2", "F3", "F4"):
        labels = load_labels(fold)
        print(f"[{fold}] labels {labels.shape}, distinct dates {labels['signal_date'].nunique()}", flush=True)
        for carrier, ws, mode in CARRIERS:
            path = find_predictions(ws, mode, fold)
            if path is None:
                continue
            daily = per_day(labels, path)
            daily.insert(0, "fold", fold)
            daily.insert(0, "carrier", carrier)
            per[f"{carrier}|{fold}"] = summarise(daily)
            per[f"{carrier}|{fold}"]["carrier"] = carrier
            per[f"{carrier}|{fold}"]["fold"] = fold
            per[f"{carrier}|{fold}"]["predictions"] = path
            all_daily.append(daily)
            s = per[f"{carrier}|{fold}"]
            print(f"  {carrier:<14} floor-valid {s['days_valid_in_the_validation_regime_floor_only']:>3} "
                  f"-> oos-valid {s['days_valid_in_the_oos_regime_floor_plus_health']:>3} "
                  f"(dropped {s['days_dropped_by_score_health']:>2})  "
                  f"mean {s['mean_rank_ic_validation_regime']:.6f} -> {s['mean_rank_ic_oos_regime']:.6f}",
                  flush=True)
        del labels

    daily_all = pd.concat(all_daily, ignore_index=True)
    csv_path = args.per_day_csv or (args.out / "oos_score_health_per_day.csv")
    daily_all.to_csv(csv_path, index=False)

    # roll up per carrier across folds
    rollup = {}
    for carrier, ws, _ in CARRIERS:
        keys = [k for k in per if k.startswith(carrier + "|")]
        if not keys:
            continue
        rollup[carrier] = {
            "folds_measured": len(keys),
            "total_floor_valid_days": sum(per[k]["days_valid_in_the_validation_regime_floor_only"] for k in keys),
            "total_oos_valid_days": sum(per[k]["days_valid_in_the_oos_regime_floor_plus_health"] for k in keys),
            "total_days_dropped_by_score_health": sum(per[k]["days_dropped_by_score_health"] for k in keys),
            "days_failing_coverage": sum(
                per[k]["days_failing_each_criterion_among_floor_valid"]["coverage_outside_0.90_to_1.0"] for k in keys),
            "days_failing_unique_below_10": sum(
                per[k]["days_failing_each_criterion_among_floor_valid"]["unique_score_count_below_10"] for k in keys),
            "days_failing_std_at_or_below_1e-6": sum(
                per[k]["days_failing_each_criterion_among_floor_valid"]["score_std_at_or_below_1e-6"] for k in keys),
            "mean_rank_ic_shift_per_fold": {
                k.split("|")[1]: per[k]["mean_rank_ic_shift_oos_minus_validation"] for k in keys},
            "largest_absolute_mean_shift": max(
                (abs(per[k]["mean_rank_ic_shift_oos_minus_validation"]) for k in keys
                 if per[k]["mean_rank_ic_shift_oos_minus_validation"] is not None), default=None),
        }

    result = {
        "question": ("Does require_score_health=True on the OOS path actually remove days that the "
                     "validation paths keep, which of its three criteria does the removing, and does the "
                     "mean_rank_ic move when it does?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False,
        "repository_module_imported_or_executed": False,
        "method": ("per-day arithmetic re-implemented line by line from evaluation/rank_ic.py:60-107 and "
                   "evaluation/gates.py:25-35; rank correlation reproduced as pandas rank(method='average') "
                   "on both sides followed by a Pearson correlation, which is what the repository does"),
        "criteria_under_test": {
            "minimum_cross_section": MIN_CROSS_SECTION,
            "coverage_window": [COV_LO, COV_HI],
            "unique_score_count_floor": MIN_UNIQUE,
            "score_std_floor": MIN_STD,
            "source": "evaluation/gates.py:35",
        },
        "carriers_measured": sorted({k.split("|")[0] for k in per}),
        "per_carrier_rollup": rollup,
        "per_carrier_fold": per,
        "per_day_csv": str(csv_path),
        "per_day_rows": int(len(daily_all)),
    }
    out = args.out / "oos_score_health_day_effect.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out}\nwrote {csv_path} ({len(daily_all)} rows)")
    print("\n=== per-carrier rollup across folds ===")
    print(f"{'carrier':<15}{'folds':>6}{'floor-valid':>13}{'oos-valid':>11}{'dropped':>9}"
          f"{'cov':>6}{'uniq<10':>9}{'std<=1e-6':>11}{'max |shift|':>13}")
    for c, v in rollup.items():
        print(f"{c:<15}{v['folds_measured']:>6}{v['total_floor_valid_days']:>13}"
              f"{v['total_oos_valid_days']:>11}{v['total_days_dropped_by_score_health']:>9}"
              f"{v['days_failing_coverage']:>6}{v['days_failing_unique_below_10']:>9}"
              f"{v['days_failing_std_at_or_below_1e-6']:>11}"
              f"{(v['largest_absolute_mean_shift'] if v['largest_absolute_mean_shift'] is not None else float('nan')):>13.2e}")


if __name__ == "__main__":
    main()
