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
OUT = Path("/tmp/es_metric_diag/rankreg_subsets.json")
FOLDS = ("F1", "F2", "F3", "F4")
KEYS = ["signal_date", "ts_code"]
LAB = KEYS + ["raw_return_5d", "label_valid"]
SEEDS = (20260801, 20260802, 20260803)
ROUNDS, PATIENCE, MIN_CS = 500, 50, 30

# NaN-rate-confirmed column mapping on the frozen-v2 15-factor set
ALL15 = tuple(f"f{i:02d}" for i in range(15))
DENSE9 = ("f00", "f01", "f04", "f09", "f10", "f11", "f12", "f13", "f14")
SPARSE6 = ("f02", "f03", "f05", "f06", "f07", "f08")
SUBSETS = {"all15": ALL15, "dense9": DENSE9, "sparse6": SPARSE6}
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
                model = LGBMRankRegBaseline.fit(
                    tr.loc[:, feats], tr.loc[:, LAB], factor_order=order, role="TRAIN",
                    num_boost_round=ROUNDS, minimum_cross_section=MIN_CS,
                    hyperparameters=CONFIG,
                    evaluation_features=va.loc[:, feats], evaluation_labels=va.loc[:, LAB],
                    early_stopping_patience=PATIENCE, seed=seed, allow_nan=True)
                trees.append(int(model.booster.num_trees()))
                pcts.append(percentile(model, te, order))
            merged = pcts[0].rename(columns={"score": "s0"})
            for i, extra in enumerate(pcts[1:], start=1):
                merged = merged.merge(extra.rename(columns={"score": f"s{i}"}), on=KEYS,
                                      validate="one_to_one")
            merged["score"] = merged[[f"s{i}" for i in range(len(pcts))]].mean(axis=1)
            cand = merged.loc[:, KEYS + ["score"]]
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

    a = results["all15"]["pooled"]["delta"]
    d = results["dense9"]["pooled"]["delta"]
    results["_verdict"] = {
        "all15_delta": a, "dense9_delta": d, "sparse_marginal_contribution": a - d,
        "reading": ("the better carrier DOES unlock the sparse factors"
                    if a - d > 0.001 else
                    "the sparse factors add nothing measurable even under the better carrier"
                    if a - d < -0.001 else
                    "the sparse factors are within noise of contributing nothing"),
        "lambdarank_comparison": {"all15": 0.00109, "dense9": 0.00177, "sparse6": -0.02122},
    }
    OUT.write_text(json.dumps(results, indent=2, default=str))
    print(f"\n=== VERDICT: all15={a:+.6f} dense9={d:+.6f} marginal={a-d:+.6f}", flush=True)
    print(f"    {results['_verdict']['reading']}", flush=True)
    print(f"written {OUT}", flush=True)


if __name__ == "__main__":
    main()
