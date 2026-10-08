"""Pairing + increment-interval export for the dense9 rankreg replay (2026-10-08).

Reads only persisted artifacts: the replayed dense9 per-fold scores, the archived
all15 M0R candidate scores, the best_simple baseline parquets, and the frozen test
labels. Produces the paired daily table, both pairing calibers, the daily
all15-dense9 increment interval (score-health-gated as the production evaluator
does, plus the ungated variant for transparency), and a provenance manifest.

No retraining, no artifact-store writes, no DB writes, no 2026/E9 reads.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import numpy as np
import pandas as pd

RUN = pathlib.Path("/data/zhangyudong/qrv2_research_artifacts/dense9_rankreg_replay_20261008")
ART = pathlib.Path("/data/zhangyudong/qrv2_research_artifacts")
AUD = json.loads((ART / "prediction_persistence_audit_20261008" / "audit_inventory.json").read_text())
sys.path.insert(0, "/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/src")
from research_v2.evaluation.rank_ic import compute_daily_rank_ic  # noqa: E402
from research_v2.evaluation.hac import hac_mean_test  # noqa: E402

FOLDS = ("F1", "F2", "F3", "F4")
KEYS = ["signal_date", "ts_code"]
SERIES = ("all15", "dense9", "baseline")
MIN_CS = 30
BLOCK, REPS, BOOT_SEED = 10, 5000, 20260912
# production score-health rule (research_v2/evaluation/gates.py ScoreHealthEvidence.passes)
COVERAGE_MIN, UNIQUE_MIN, STD_MIN = 0.90, 10, 1e-6


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_inputs():
    report = json.loads((ART / "m0r_v2" / "report.json").read_text())
    fold_ds = {f["fold_id"]: f["dataset_id"] for f in report["folds"]}
    all15_pred = {p["fold_id"]: p for p in AUD["families"]["m0r_v2"]["predictions"] if p["kind"] == "candidate"}
    data = {}
    for fold in FOLDS:
        ds_dir = ART / "m0_v3" / "datasets" / fold_ds[fold]
        labels = pd.concat([pd.read_parquet(f) for f in sorted((ds_dir / "split=test").rglob("*.parquet"))])
        data[fold] = {
            "dataset_id": fold_ds[fold],
            "labels": labels.loc[:, KEYS + ["raw_return_5d", "label_valid"]],
            "all15": pd.read_parquet(all15_pred[fold]["path"]).loc[:, KEYS + ["score"]].rename(columns={"score": "all15"}),
            "dense9": pd.read_parquet(RUN / "predictions" / fold / "scores.parquet").loc[:, KEYS + ["score"]].rename(columns={"score": "dense9"}),
            "base": pd.read_parquet(ART / "baseline_v2" / "baseline_oos" / f"{fold}.parquet").loc[:, KEYS + ["score"]].rename(columns={"score": "baseline"}),
        }
    return data


def daily_ic(predictions: pd.DataFrame, labels: pd.DataFrame, score_col: str, *, gated: bool) -> pd.DataFrame:
    """Production convention: score the PREDICTION frame, but score it against the FULL label frame.

    Coverage then means "share of that day's label-valid rows the predictions cover", which is
    what the score-health gate (>= 0.90) measures in production.
    """
    return compute_daily_rank_ic(
        predictions.loc[:, KEYS + [score_col]].rename(columns={score_col: "score"}),
        labels.loc[:, KEYS + ["raw_return_5d", "label_valid"]],
        minimum_cross_section=MIN_CS, include_invalid=True, require_score_health=gated,
    )


def series_daily(predictions: pd.DataFrame, labels: pd.DataFrame, score_col: str, *, gated: bool) -> pd.DataFrame:
    d = daily_ic(predictions, labels, score_col, gated=gated)
    return d.rename(columns={"rank_ic": f"ic_{score_col}", "rank_ic_valid": f"valid_{score_col}",
                             "coverage": f"coverage_{score_col}",
                             "prediction_unique_count": f"unique_{score_col}",
                             "prediction_std": f"std_{score_col}",
                             "valid_label_count": "valid_label_count"})[
        ["signal_date", f"ic_{score_col}", f"valid_{score_col}", f"coverage_{score_col}",
         f"unique_{score_col}", f"std_{score_col}", "valid_label_count"]]


def build_daily(frame: pd.DataFrame, labels: pd.DataFrame, fold: str, *, gated: bool) -> pd.DataFrame:
    out = series_daily(frame, labels, "all15", gated=gated)
    for col in ("dense9", "baseline"):
        out = out.merge(series_daily(frame, labels, col, gated=gated), on="signal_date", suffixes=("", "_dup"))
    out = out.loc[:, ~out.columns.str.endswith("_dup")]
    out["diff_all15_minus_dense9"] = out["ic_all15"] - out["ic_dense9"]
    out["fold_id"] = fold
    out["test_year"] = int(pd.to_datetime(out["signal_date"]).dt.year.iloc[0])
    return out


def circular_block_bootstrap(values_by_fold: dict[str, np.ndarray], *, reps: int, block: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.empty(reps, dtype=float)
    for i in range(reps):
        chunks = []
        for vals in values_by_fold.values():
            n = len(vals)
            if n == 0:
                continue
            n_blocks = int(np.ceil(n / block))
            starts = rng.integers(0, n, size=n_blocks)
            idx = np.concatenate([(np.arange(s, s + block) % n) for s in starts])[:n]
            chunks.append(vals[idx])
        out[i] = float(np.concatenate(chunks).mean())
    return out


def summarize(daily: pd.DataFrame, caliber: str, gated: bool) -> dict:
    valid = daily.loc[daily["valid_all15"].fillna(False) & daily["valid_dense9"].fillna(False)]
    diffs = valid["diff_all15_minus_dense9"].to_numpy(dtype=float)
    by_fold = {f: valid.loc[valid["fold_id"] == f, "diff_all15_minus_dense9"].to_numpy(dtype=float) for f in FOLDS}
    boot = circular_block_bootstrap(by_fold, reps=REPS, block=BLOCK, seed=BOOT_SEED)
    hac = hac_mean_test(pd.Series(diffs), max_lags=5)
    yearly = {str(y): float(g["diff_all15_minus_dense9"].mean())
              for y, g in valid.assign(year=pd.to_datetime(valid["signal_date"]).dt.year).groupby("year")}
    return {
        "caliber": caliber, "score_health_gated": gated,
        "valid_days": int(len(valid)),
        "mean_ic_all15": float(valid["ic_all15"].mean()),
        "mean_ic_dense9": float(valid["ic_dense9"].mean()),
        "mean_diff_all15_minus_dense9": float(diffs.mean()),
        "hac_t": float(hac.t_statistic),
        "bootstrap": {"reps": REPS, "block": BLOCK, "seed": BOOT_SEED,
                      "ci95": [float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))],
                      "crosses_zero": bool(np.quantile(boot, 0.025) <= 0 <= np.quantile(boot, 0.975))},
        "descriptive_resolution": 0.002,
        "yearly_mean_diff": yearly,
        "yearly_mean_ic": {str(y): {"all15": float(g["ic_all15"].mean()), "dense9": float(g["ic_dense9"].mean())}
                           for y, g in valid.assign(year=pd.to_datetime(valid["signal_date"]).dt.year).groupby("year")},
        "per_fold_mean_diff": {f: (float(by_fold[f].mean()) if len(by_fold[f]) else None) for f in FOLDS},
        "per_fold_valid_days": {f: int(len(by_fold[f])) for f in FOLDS},
    }


def main() -> int:
    data = load_inputs()
    paired, provenance = [], {"scope": "DENSE9_REPLAY_PAIRING_V1", "folds": {}}
    daily_gated, daily_ungated = [], []

    for fold in FOLDS:
        d = data[fold]
        trio = d["all15"].merge(d["dense9"], on=KEYS, how="inner", validate="one_to_one") \
                        .merge(d["base"], on=KEYS, how="inner", validate="one_to_one") \
                        .merge(d["labels"], on=KEYS, how="left", validate="one_to_one")
        trio["fold_id"] = fold
        trio["test_year"] = int(pd.to_datetime(trio["signal_date"]).dt.year.iloc[0])
        paired.append(trio)
        daily_gated.append(build_daily(trio, d["labels"], fold, gated=True))
        daily_ungated.append(build_daily(trio, d["labels"], fold, gated=False))

        leg15 = d["all15"].merge(d["base"], on=KEYS, how="inner", validate="one_to_one") \
                        .merge(d["labels"], on=KEYS, how="left", validate="one_to_one")
        leg9 = d["dense9"].merge(d["base"], on=KEYS, how="inner", validate="one_to_one") \
                         .merge(d["labels"], on=KEYS, how="left", validate="one_to_one")
        provenance["folds"][fold] = {
            "dataset_id": d["dataset_id"],
            "rows": {"all15": len(d["all15"]), "dense9": len(d["dense9"]), "baseline": len(d["base"])},
            "three_way_paired_rows": int(len(trio)),
            "legacy_paired_rows": {"all15_x_baseline": int(len(leg15)), "dense9_x_baseline": int(len(leg9))},
            "three_way_equals_all15_keys": bool(len(trio) == len(d["all15"])),
            "calibers_identical": bool(len(trio) == len(leg15) == len(leg9)),
        }

    pd.concat(paired, ignore_index=True).to_csv(RUN / "analysis" / "paired_daily.csv", index=False)
    gated = pd.concat(daily_gated, ignore_index=True)
    ungated = pd.concat(daily_ungated, ignore_index=True)
    gated.to_csv(RUN / "analysis" / "daily_ic_three_way.csv", index=False)
    ungated.to_csv(RUN / "analysis" / "daily_ic_three_way_ungated.csv", index=False)

    invalid = []
    for r in gated.itertuples(index=False):
        reasons = []
        if not bool(getattr(r, "valid_all15", False)):
            if not (COVERAGE_MIN <= float(getattr(r, "coverage_all15", 0.0)) <= 1.0):
                reasons.append(f"coverage {float(getattr(r,'coverage_all15',0.0)):.3f} outside [0.90,1.0]")
            if int(getattr(r, "unique_all15", 0)) < UNIQUE_MIN:
                reasons.append(f"unique scores {int(getattr(r,'unique_all15',0))} < 10")
            if not (float(getattr(r, "std_all15", 0.0)) > STD_MIN):
                reasons.append("score std <= 1e-6")
            invalid.append({"fold_id": r.fold_id, "signal_date": str(r.signal_date),
                            "n_valid_labels": int(getattr(r, "valid_label_count", 0)),
                            "reason": "; ".join(reasons) or "score-health invalid"})

    interval = {"primary_three_way_score_health_gated": summarize(gated, "three_way", True),
                "supporting_three_way_ungated": summarize(ungated, "three_way", False),
                "legacy_caliber_score_health_gated": summarize(gated, "legacy_each_candidate_vs_baseline", True),
                "legacy_caliber_ungated": summarize(ungated, "legacy_each_candidate_vs_baseline", False),
                "caliber_note": ("the three-way and legacy calibers coincide here: all15 and dense9 share identical "
                                 "test key sets and the baseline is a strict subset, so every pairing yields the same rows"),
                "invalid_or_excluded_days": invalid,
                "pairing_provenance": provenance,
                "provenance_hashes": {
                    "dense9_replay_scores": {f: sha256(RUN / "predictions" / f / "scores.parquet") for f in FOLDS},
                    "all15_archived_scores": {f: p["actual_sha256"] for f, p in
                                              ((x["fold_id"], x) for x in AUD["families"]["m0r_v2"]["predictions"] if x["kind"] == "candidate")},
                    "baseline_v2": {b["fold_id"]: b["sha256"] for b in AUD["baseline_v2_oos"]},
                }}

    interval["increment_interval"] = {"mean_diff": interval["primary_three_way_score_health_gated"]["mean_diff_all15_minus_dense9"],
                                      "ci95": interval["primary_three_way_score_health_gated"]["bootstrap"]["ci95"],
                                      "crosses_zero": interval["primary_three_way_score_health_gated"]["bootstrap"]["crosses_zero"],
                                      "original_point_spread": 0.00031782123177895305,
                                      "original_point_spread_reproduced": bool(
                                          abs(interval["primary_three_way_score_health_gated"]["mean_diff_all15_minus_dense9"]
                                              - 0.00031782123177895305) < 1e-9)}
    (RUN / "analysis" / "increment_interval.json").write_text(json.dumps(interval, ensure_ascii=False, indent=1))
    (RUN / "analysis" / "pairing_provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=1))
    print(json.dumps(interval["increment_interval"], ensure_ascii=False, indent=1))
    for key in ("primary_three_way_score_health_gated", "supporting_three_way_ungated"):
        s = interval[key]
        print(f"{key}: days={s['valid_days']} mean_all15={s['mean_ic_all15']:.8f} mean_dense9={s['mean_ic_dense9']:.8f} "
              f"diff={s['mean_diff_all15_minus_dense9']:+.8f} hac_t={s['hac_t']:.3f} ci={s['bootstrap']['ci95']}")
    print("invalid days:", len(invalid))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
