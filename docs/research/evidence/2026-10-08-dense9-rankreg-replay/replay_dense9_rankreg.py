"""Under the BETTER carrier, do the six sparse factors contribute?

The ablation that put adjudication-4's premise in doubt was measured mostly under LambdaRank,
where the carrier itself was crippling everything (ctrl all15 +0.00109). Under rankreg sparse6
improved from -0.02122 to -0.00877 but stayed negative -- and dense9 under rankreg was never
measured at all. That missing cell decides the question:

  all15 > dense9  -> the better carrier does unlock the sparse factors, and adjudication-4's
                     hybrid design is vindicated rather than contradicted
  all15 <= dense9 -> the sparse factors contribute nothing even when the carrier is fixed, and
                     the candidate's whole gain is the dense factors plus the model form

Uses the production LGBMRankRegBaseline. One fixed config, no search, so no selection bias.
Read-only: no DB, no artifact-store writes, no repo writes.
"""
from __future__ import annotations

import hashlib
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
RUN_DIR = Path("/data/zhangyudong/qrv2_research_artifacts/dense9_rankreg_replay_20261008")
OUT = RUN_DIR / "analysis" / "dense9_replay_results.json"
FITS_DIR = RUN_DIR / "fits"
PREDS_DIR = RUN_DIR / "predictions"
LOGS_DIR = RUN_DIR / "logs"
PROGRESS_LOG = LOGS_DIR / "fits_progress.jsonl"
for _d in (RUN_DIR / "analysis", FITS_DIR, PREDS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
CODE_PATHS = (
    "/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/src/research_v2/baselines/lgbm_rankreg.py",
    "/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/src/research_v2/baselines/lgbm_ranker.py",
    "/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/src/research_v2/evaluation/model_evaluator.py",
    "/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/src/research_v2/evaluation/rank_ic.py",
)

def _sha(path: str) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def _progress(record: dict) -> None:
    """Append-only per-fit progress record (never rewrites earlier lines)."""
    with PROGRESS_LOG.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str, ensure_ascii=False) + "\n")
FOLDS = ("F1", "F2", "F3", "F4")
KEYS = ["signal_date", "ts_code"]
LAB = KEYS + ["raw_return_5d", "label_valid"]
SEEDS = (20260801, 20260802, 20260803)
ROUNDS, PATIENCE, MIN_CS = 500, 50, 30

# NaN-rate-confirmed column mapping on the frozen-v2 15-factor set
ALL15 = tuple(f"f{i:02d}" for i in range(15))
DENSE9 = ("f00", "f01", "f04", "f09", "f10", "f11", "f12", "f13", "f14")
SPARSE6 = ("f02", "f03", "f05", "f06", "f07", "f08")
SUBSETS = {"dense9": DENSE9}  # replay scope: dense9 only (the two other subsets are not re-run)
CONFIG = LGBMHyperparameters(num_leaves=15, max_depth=4, learning_rate=0.05,
                             min_child_samples=200, feature_fraction=0.85,
                             num_threads=16, lambda_l2=0.0)


def percentile(model, frame, order) -> pd.DataFrame:
    scores = model.predict(frame.loc[:, KEYS + list(order)])
    out = frame.loc[:, KEYS].copy()
    out["score"] = scores.groupby(frame["signal_date"].to_numpy()).rank(pct=True).to_numpy()
    return out


def main() -> None:
    frames, bases, labels = {}, {}, {}
    for fold in FOLDS:
        ds = DS / FOLD_DS[fold]
        frames[fold] = {r: pd.read_parquet(ds / f"split={r}") for r in ("train", "validation", "test")}
        for _role, _fr in frames[fold].items():
            (LOGS_DIR / f"inputs_{fold}_{_role}.json").write_text(json.dumps(
                {"fold_id": fold, "dataset_id": FOLD_DS[fold], "role": _role,
                 "rows": int(len(_fr)), "days": int(_fr["signal_date"].nunique()),
                 "columns": list(_fr.columns)}, indent=2))
        bases[fold] = pd.read_parquet(BASE / f"{fold}.parquet").loc[:, KEYS + ["score"]]
        labels[fold] = frames[fold]["test"].loc[:, LAB]
        print(f"[{fold}] loaded", flush=True)

    results: dict[str, dict] = {}
    for name, order in SUBSETS.items():
        t0 = time.time()
        cand_folds, base_folds, lab_folds, per_fold = [], [], [], {}
        for fold in FOLDS:
            tr, va, te = frames[fold]["train"], frames[fold]["validation"], frames[fold]["test"]
            feats = KEYS + list(order)
            pcts, trees = [], []
            for seed in SEEDS:
                fit_t0 = time.time()
                fit_dir = FITS_DIR / f"{fold}_seed{seed}"
                fit_dir.mkdir(parents=True, exist_ok=True)
                model = LGBMRankRegBaseline.fit(
                    tr.loc[:, feats], tr.loc[:, LAB], factor_order=order, role="TRAIN",
                    num_boost_round=ROUNDS, minimum_cross_section=MIN_CS,
                    hyperparameters=CONFIG,
                    evaluation_features=va.loc[:, feats], evaluation_labels=va.loc[:, LAB],
                    early_stopping_patience=PATIENCE, seed=seed, allow_nan=True)
                n_trees = int(model.booster.num_trees())
                trees.append(n_trees)
                pct = percentile(model, te, order)
                pcts.append(pct)
                fit_seconds = round(time.time() - fit_t0, 1)
                model.booster.save_model(str(fit_dir / "booster.txt"))
                pct.to_parquet(fit_dir / "test_percentile.parquet", index=False)
                (fit_dir / "fit_meta.json").write_text(json.dumps({
                    "fold_id": fold, "seed": seed, "factor_order": list(order),
                    "num_boost_round": ROUNDS, "early_stopping_patience": PATIENCE,
                    "minimum_cross_section": MIN_CS, "allow_nan": True,
                    "config": {"num_leaves": CONFIG.num_leaves, "max_depth": CONFIG.max_depth,
                               "learning_rate": CONFIG.learning_rate,
                               "min_child_samples": CONFIG.min_child_samples,
                               "feature_fraction": CONFIG.feature_fraction,
                               "num_threads": CONFIG.num_threads, "lambda_l2": CONFIG.lambda_l2},
                    "trees": n_trees, "best_iteration": model.best_iteration,
                    "training_day_count": model.training_day_count,
                    "eligible_training_day_count": model.eligible_training_day_count,
                    "fit_seconds": fit_seconds,
                    "inputs": {"train_rows": int(len(tr)), "validation_rows": int(len(va)),
                               "test_rows": int(len(te)),
                               "train_days": int(tr["signal_date"].nunique()),
                               "validation_days": int(va["signal_date"].nunique()),
                               "test_days": int(te["signal_date"].nunique()),
                               "columns_used": feats},
                    "dataset_id": FOLD_DS[fold],
                }, indent=2, default=str))
                _progress({"event": "fit_complete", "fold_id": fold, "seed": seed,
                           "trees": n_trees, "best_iteration": model.best_iteration,
                           "fit_seconds": fit_seconds,
                           "training_day_count": model.training_day_count,
                           "eligible_training_day_count": model.eligible_training_day_count})
                print(f"    [{name}] {fold} seed={seed} trees={n_trees} "
                      f"best_iter={model.best_iteration} ({fit_seconds}s)", flush=True)
            merged = pcts[0].rename(columns={"score": "s0"})
            for i, extra in enumerate(pcts[1:], start=1):
                merged = merged.merge(extra.rename(columns={"score": f"s{i}"}), on=KEYS,
                                      validate="one_to_one")
            merged["score"] = merged[[f"s{i}" for i in range(len(pcts))]].mean(axis=1)
            cand = merged.loc[:, KEYS + ["score"]]
            fold_pred_dir = PREDS_DIR / fold
            fold_pred_dir.mkdir(parents=True, exist_ok=True)
            cand.to_parquet(fold_pred_dir / "scores.parquet", index=False)
            (fold_pred_dir / "meta.json").write_text(json.dumps({
                "fold_id": fold, "dataset_id": FOLD_DS[fold], "factor_order": list(order),
                "seeds": list(SEEDS), "ensemble": "mean of per-seed within-day percentile ranks "
                "computed on the FULL test cross-section (never intersected with the baseline first)",
                "rows": int(len(cand)), "days": int(cand["signal_date"].nunique()),
                "trees": trees,
            }, indent=2, default=str))
            ev = evaluate_frozen_oos(cand, labels[fold], baseline_predictions=bases[fold], role="TEST")
            per_fold[fold] = {"trees": trees, "paired_delta": ev.delta_rank_ic}
            cand_folds.append(cand); base_folds.append(bases[fold]); lab_folds.append(labels[fold])
            print(f"  [{name}] {fold}: trees={trees} delta={ev.delta_rank_ic:+.5f}", flush=True)

        pooled = evaluate_frozen_oos(
            pd.concat(cand_folds, ignore_index=True), pd.concat(lab_folds, ignore_index=True),
            baseline_predictions=pd.concat(base_folds, ignore_index=True), role="TEST")
        results[name] = {
            "factor_count": len(order), "per_fold": per_fold,
            "pooled": {"cand": pooled.summary.mean_rank_ic, "base": pooled.baseline_mean_rank_ic,
                       "delta": pooled.delta_rank_ic, "icir": pooled.summary.icir,
                       "hac_t": pooled.hac.t_statistic, "valid_days": pooled.summary.valid_day_count,
                       "yearly": {str(k): v for k, v in pooled.yearly_mean_rank_ic.items()},
                       "gates": {g: r.status.value for g, r in pooled.gate_report.results.items()},
                       "all_passed": pooled.gate_report.all_passed},
            "seconds": round(time.time() - t0, 1),
        }
        print(f"  [{name}] POOLED cand={pooled.summary.mean_rank_ic:.5f} "
              f"delta={pooled.delta_rank_ic:+.6f} all_passed={pooled.gate_report.all_passed} "
              f"({time.time()-t0:.0f}s)", flush=True)

    # Replay scope is dense9 only: the cross-subset verdict is deliberately omitted
    # (it required the all15 and sparse6 cells, which are not re-run here).
    results["_code_summary"] = {
        "driver_path": str(Path(__file__).resolve()),
        "driver_sha256": _sha(str(Path(__file__).resolve())),
        "training_implementation": {p: _sha(p) for p in CODE_PATHS},
        "lightgbm_version": __import__("lightgbm").__version__,
        "pandas_version": pd.__version__,
        "python_version": sys.version.split()[0],
    }
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n=== dense9 pooled delta={results['dense9']['pooled']['delta']:+.6f} "
          f"cand={results['dense9']['pooled']['cand']:.6f}", flush=True)
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
