"""Is the rank-regression carrier's advantage robust, or one lucky config?

+0.009149 was measured at a single fixed config with no validation search, so it carries no
selection bias -- but a single point can still be a lucky draw. This evaluates FOUR configs
INDEPENDENTLY and takes no maximum: each gets its own pooled paired delta and its own six-gate
verdict. The question is how many of them clear G4, not what the best one scored.

Uses the production LGBMRankRegBaseline, not a reimplementation. Read-only: no DB, no
artifact-store writes, no repo writes.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, "/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/src")

from research_v2.baselines.lgbm_ranker import LGBMHyperparameters  # noqa: E402
from research_v2.baselines.lgbm_rankreg import LGBMRankRegBaseline  # noqa: E402
from research_v2.evaluation.model_evaluator import evaluate_frozen_oos  # noqa: E402

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
DS = ART / "m0_v3" / "datasets"
BASE = ART / "baseline_v2" / "baseline_oos"
REPORT = json.loads((ART / "m0r_v2" / "report.json").read_text())
FOLD_DS = {f["fold_id"]: f["dataset_id"] for f in REPORT["folds"]}
OUT = Path("/tmp/es_metric_diag/config_robustness.json")
FOLDS = ("F1", "F2", "F3", "F4")
KEYS = ["signal_date", "ts_code"]
ORDER = tuple(f"f{i:02d}" for i in range(15))
SEEDS = (20260801, 20260802, 20260803)
ROUNDS, PATIENCE, MIN_CS = 500, 50, 30

# C1 is the config the +0.009149 was measured at and must reproduce it exactly; the other
# three span the capacity/learning-rate/regularisation region the M0 search explored.
CONFIGS = {
    "C1_measured": dict(num_leaves=15, max_depth=4, learning_rate=0.05,
                        min_child_samples=200, feature_fraction=0.85, lambda_l2=0.0),
    "C2_mid": dict(num_leaves=31, max_depth=6, learning_rate=0.03,
                   min_child_samples=200, feature_fraction=0.85, lambda_l2=0.0),
    "C3_large": dict(num_leaves=63, max_depth=6, learning_rate=0.03,
                     min_child_samples=200, feature_fraction=0.85, lambda_l2=0.0),
    "C4_small_reg": dict(num_leaves=15, max_depth=4, learning_rate=0.03,
                         min_child_samples=50, feature_fraction=0.85, lambda_l2=1.0),
}


def hp(config: dict) -> LGBMHyperparameters:
    return LGBMHyperparameters(
        num_leaves=config["num_leaves"], max_depth=config["max_depth"],
        learning_rate=config["learning_rate"], min_child_samples=config["min_child_samples"],
        feature_fraction=config["feature_fraction"], num_threads=16,
        lambda_l2=config["lambda_l2"])


def percentile(model, frame: pd.DataFrame) -> pd.DataFrame:
    scores = model.predict(frame.loc[:, KEYS + list(ORDER)])
    out = frame.loc[:, KEYS].copy()
    out["score"] = (scores.groupby(frame["signal_date"].to_numpy()).rank(pct=True).to_numpy())
    return out


def main() -> None:
    frames, bases, labels = {}, {}, {}
    for fold in FOLDS:
        ds = DS / FOLD_DS[fold]
        frames[fold] = {r: pd.read_parquet(ds / f"split={r}") for r in ("train", "validation", "test")}
        bases[fold] = pd.read_parquet(BASE / f"{fold}.parquet").loc[:, KEYS + ["score"]]
        labels[fold] = frames[fold]["test"].loc[:, KEYS + ["raw_return_5d", "label_valid"]]
        print(f"[{fold}] loaded", flush=True)

    results: dict[str, dict] = {}
    for name, config in CONFIGS.items():
        t0 = time.time()
        cand_by_fold, base_by_fold, lab_by_fold, per_fold = [], [], [], {}
        for fold in FOLDS:
            tr, va, te = frames[fold]["train"], frames[fold]["validation"], frames[fold]["test"]
            keys = KEYS + list(ORDER)
            lab_cols = KEYS + ["raw_return_5d", "label_valid"]
            pcts, trees = [], []
            for seed in SEEDS:
                model = LGBMRankRegBaseline.fit(
                    tr.loc[:, keys], tr.loc[:, lab_cols], factor_order=ORDER, role="TRAIN",
                    num_boost_round=ROUNDS, minimum_cross_section=MIN_CS,
                    hyperparameters=hp(config),
                    evaluation_features=va.loc[:, keys], evaluation_labels=va.loc[:, lab_cols],
                    early_stopping_patience=PATIENCE, seed=seed, allow_nan=True)
                trees.append(int(model.booster.num_trees()))
                pcts.append(percentile(model, te))
            merged = pcts[0].rename(columns={"score": "s0"})
            for i, extra in enumerate(pcts[1:], start=1):
                merged = merged.merge(extra.rename(columns={"score": f"s{i}"}), on=KEYS,
                                      validate="one_to_one")
            merged["score"] = merged[[f"s{i}" for i in range(len(pcts))]].mean(axis=1)
            cand = merged.loc[:, KEYS + ["score"]]
            ev = evaluate_frozen_oos(cand, labels[fold], baseline_predictions=bases[fold],
                                     role="TEST")
            per_fold[fold] = {"trees": trees, "paired_delta": ev.delta_rank_ic}
            cand_by_fold.append(cand); base_by_fold.append(bases[fold])
            lab_by_fold.append(labels[fold])
            print(f"  [{name}] {fold}: trees={trees} delta={ev.delta_rank_ic:+.5f}", flush=True)

        pooled = evaluate_frozen_oos(
            pd.concat(cand_by_fold, ignore_index=True),
            pd.concat(lab_by_fold, ignore_index=True),
            baseline_predictions=pd.concat(base_by_fold, ignore_index=True),
            role="TEST")
        gates = {g: r.status.value for g, r in pooled.gate_report.results.items()}
        results[name] = {
            "config": config, "per_fold": per_fold,
            "pooled": {"cand": pooled.summary.mean_rank_ic,
                       "base": pooled.baseline_mean_rank_ic,
                       "delta": pooled.delta_rank_ic,
                       "icir": pooled.summary.icir, "hac_t": pooled.hac.t_statistic,
                       "valid_days": pooled.summary.valid_day_count,
                       "yearly": {str(k): v for k, v in pooled.yearly_mean_rank_ic.items()},
                       "gates": gates, "all_passed": pooled.gate_report.all_passed},
            "seconds": round(time.time() - t0, 1),
        }
        print(f"  [{name}] POOLED delta={pooled.delta_rank_ic:+.6f} "
              f"all_passed={pooled.gate_report.all_passed} gates={gates} "
              f"({time.time()-t0:.0f}s)", flush=True)

    passing = [n for n, r in results.items() if r["pooled"]["all_passed"]]
    g4 = [n for n, r in results.items()
          if r["pooled"]["gates"]["G4"] == "PASS"]
    results["_summary"] = {
        "configs_evaluated": len(CONFIGS),
        "configs_passing_all_six_gates": passing,
        "configs_passing_g4": g4,
        "deltas": {n: r["pooled"]["delta"] for n, r in results.items() if n != "_summary"},
        "minimum_delta": min(r["pooled"]["delta"] for n, r in results.items() if n != "_summary"),
        "note": "each config evaluated independently; no maximum taken, so no selection bias",
    }
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n=== SUMMARY: {len(g4)}/{len(CONFIGS)} configs pass G4; "
          f"{len(passing)}/{len(CONFIGS)} pass all six ===", flush=True)
    for n, r in results.items():
        if n == "_summary":
            continue
        print(f"  {n:>14}: delta={r['pooled']['delta']:+.6f} "
              f"all_passed={r['pooled']['all_passed']}", flush=True)
    print(f"  minimum delta across configs = {results['_summary']['minimum_delta']:+.6f}", flush=True)
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
