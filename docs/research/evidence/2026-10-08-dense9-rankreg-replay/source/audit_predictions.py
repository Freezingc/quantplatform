"""Read-only persistence audit + independent recomputation (2026-10-08).

Audits whether the all15 / dense9 three-seed ensemble TEST predictions exist as
frozen artifacts, verifies their integrity (SHA-256 vs manifest), and recomputes
the walk-forward OOS numbers from the persisted predictions with the production
evaluator. Writes machine-readable outputs into this new directory only.
Never writes to the artifacts tree it audits, the database, or the 2026 locked
segment.
"""
from __future__ import annotations

import hashlib
import json
import pathlib
import sys

import pandas as pd

ART = pathlib.Path("/data/zhangyudong/qrv2_research_artifacts")
OUT = ART / "prediction_persistence_audit_20261008"
sys.path.insert(0, "/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/src")
from research_v2.evaluation.model_evaluator import evaluate_frozen_oos  # noqa: E402
from research_v2.evaluation.rank_ic import summarize_rank_ic  # noqa: E402
from research_v2.evaluation.hac import hac_mean_test  # noqa: E402

FOLDS = ["F1", "F2", "F3", "F4"]
YEAR = {"F1": 2022, "F2": 2023, "F3": 2024, "F4": 2025}


def sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_manifest(path: pathlib.Path) -> dict:
    return json.loads((path / "manifest.json").read_text())


def inventory_predictions(family_dir: pathlib.Path) -> list[dict]:
    rows = []
    root = family_dir / "predictions"
    if not root.exists():
        return rows
    for d in sorted(root.iterdir()):
        man = read_manifest(d)
        meta = man["metadata"]
        actual = sha256(d / "scores.parquet")
        rows.append({
            "family_dir": family_dir.name,
            "prediction_id": man["prediction_id"],
            "path": str(d / "scores.parquet"),
            "kind": meta["kind"],
            "fold_id": meta["fold_id"],
            "model_id": meta["model_id"],
            "dataset_id": meta["dataset_id"],
            "selection_id": meta["selection_id"],
            "declared_sha256": man["sha256"],
            "actual_sha256": actual,
            "integrity_ok": actual == man["sha256"],
            "bytes": (d / "scores.parquet").stat().st_size,
        })
    return rows


def inventory_models(family_dir: pathlib.Path) -> list[dict]:
    rows = []
    root = family_dir / "models"
    if not root.exists():
        return rows
    for mdir in sorted(root.iterdir()):
        declared = {}
        ck = mdir / "checksums.sha256"
        if ck.exists():
            for line in ck.read_text().splitlines():
                parts = line.split()
                if len(parts) == 2:
                    declared[parts[1]] = parts[0]
        for f in sorted(mdir.iterdir()):
            if f.is_dir():
                continue
            actual = sha256(f)
            rows.append({
                "family_dir": family_dir.name,
                "model_id": mdir.name,
                "file": f.name,
                "path": str(f),
                "actual_sha256": actual,
                "declared_sha256": declared.get(f.name),
                "matches_declaration": (declared.get(f.name) == actual) if f.name in declared else None,
                "bytes": f.stat().st_size,
            })
    return rows


def inventory_datasets(family_dir: pathlib.Path) -> list[dict]:
    rows = []
    root = family_dir / "datasets"
    if not root.exists():
        return rows
    for d in sorted(root.iterdir()):
        man = read_manifest(d)
        entry = {
            "family_dir": family_dir.name,
            "dataset_id": man["dataset_id"],
            "path": str(d),
            "row_count": man["row_count"],
            "n_features": len([c for c in man["column_order"] if c.startswith("f") and c[1:].isdigit()]),
            "column_order": man["column_order"],
            "content_hash": man["content_hash"],
            "splits": {},
        }
        for split in ("train", "validation", "test"):
            files = sorted((d / f"split={split}").rglob("*.parquet"))
            entry["splits"][split] = [
                {"path": str(f), "bytes": f.stat().st_size, "sha256": sha256(f)} for f in files
            ]
        rows.append(entry)
    return rows


def test_labels(dataset_dir: pathlib.Path) -> pd.DataFrame:
    files = sorted((dataset_dir / "split=test").rglob("*.parquet"))
    df = pd.concat([pd.read_parquet(f, columns=["signal_date", "ts_code", "raw_return_5d", "label_valid"]) for f in files])
    return df


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    audit: dict = {"generated_at": pd.Timestamp.now(tz="Asia/Shanghai").isoformat(), "artifacts_root": str(ART)}

    families = {}
    for name in ("m0_v3", "m0r_v2", "m0r_v3"):
        fdir = ART / name
        if not fdir.exists():
            continue
        families[name] = {
            "predictions": inventory_predictions(fdir),
            "models": inventory_models(fdir),
            "datasets": inventory_datasets(fdir),
        }
    audit["families"] = families

    baseline = []
    for fold in FOLDS:
        p = ART / "baseline_v2" / "baseline_oos" / f"{fold}.parquet"
        if p.exists():
            baseline.append({"fold_id": fold, "path": str(p), "sha256": sha256(p), "bytes": p.stat().st_size})
    audit["baseline_v2_oos"] = baseline
    audit["baseline_v2_report"] = {
        "path": str(ART / "baseline_v2" / "report.json"),
        "sha256": sha256(ART / "baseline_v2" / "report.json"),
    }

    # ---- independent recomputation from persisted artifacts ----
    verification = {}
    for family in ("m0_v3", "m0r_v2"):
        fdir = ART / family
        preds = {r["fold_id"]: r for r in families[family]["predictions"] if r["kind"] == "candidate"}
        daily_rows, per_fold = [], {}
        for fold in FOLDS:
            pr = preds[fold]
            ddir = fdir / "datasets" / pr["dataset_id"]
            labels = test_labels(ddir)
            cand = pd.read_parquet(pr["path"])
            base = pd.read_parquet(ART / "baseline_v2" / "baseline_oos" / f"{fold}.parquet")
            base = base.loc[:, ["signal_date", "ts_code", "score"]]
            ev = evaluate_frozen_oos(cand, labels, baseline_predictions=base, role="TEST")
            d = ev.daily_rank_ic.copy()
            d["fold_id"] = fold
            daily_rows.append(d)
            per_fold[fold] = {
                "candidate_mean_rank_ic": float(ev.summary.mean_rank_ic),
                "candidate_icir": float(ev.summary.icir),
                "candidate_valid_days": int(ev.summary.valid_day_count),
                "baseline_mean_rank_ic": float(ev.baseline_mean_rank_ic),
                "delta_rank_ic": float(ev.delta_rank_ic),
                "hac_t": float(ev.hac.t_statistic),
                "yearly": {str(k): float(v) for k, v in ev.yearly_mean_rank_ic.items()},
            }
        pooled_daily = pd.concat(daily_rows, ignore_index=True)
        valid = pooled_daily.loc[pooled_daily["rank_ic_valid"]]
        pooled_summary = summarize_rank_ic(valid)
        pooled_hac = hac_mean_test(valid["rank_ic"], max_lags=5)
        baseline_daily_rows = []
        for fold in FOLDS:
            pr = preds[fold]
            ddir = fdir / "datasets" / pr["dataset_id"]
            labels = test_labels(ddir)
            base = pd.read_parquet(ART / "baseline_v2" / "baseline_oos" / f"{fold}.parquet").loc[:, ["signal_date", "ts_code", "score"]]
            from research_v2.evaluation.rank_ic import compute_daily_rank_ic
            bd = compute_daily_rank_ic(base, labels, include_invalid=True, require_score_health=True)
            bd["fold_id"] = fold
            baseline_daily_rows.append(bd)
        pooled_base = pd.concat(baseline_daily_rows, ignore_index=True)
        valid_base = pooled_base.loc[pooled_base["rank_ic_valid"]]
        base_summary = summarize_rank_ic(valid_base)
        label_days = int((pooled_daily.groupby("signal_date")["valid_label_count"].first() >= 30).sum())
        verification[family] = {
            "per_fold": per_fold,
            "pooled": {
                "candidate_mean_rank_ic": float(pooled_summary.mean_rank_ic),
                "candidate_icir": float(pooled_summary.icir),
                "candidate_valid_days": int(pooled_summary.valid_day_count),
                "hac_t": float(pooled_hac.t_statistic),
                "baseline_mean_rank_ic": float(base_summary.mean_rank_ic),
                "delta_rank_ic": float(pooled_summary.mean_rank_ic - base_summary.mean_rank_ic),
                "valid_label_days": label_days,
                "g6_ratio": float(pooled_summary.valid_day_count / label_days) if label_days else None,
                "yearly": {
                    str(y): float(g["rank_ic"].mean())
                    for y, g in valid.assign(year=pd.to_datetime(valid["signal_date"]).dt.year).groupby("year")
                },
            },
            "daily_table_path": str(OUT / "daily_rank_ic" / f"{family}_daily_rank_ic.parquet"),
            "baseline_daily_table_path": str(OUT / "daily_rank_ic" / f"{family}_baseline_daily_rank_ic.parquet"),
        }
        pooled_daily.to_parquet(OUT / "daily_rank_ic" / f"{family}_daily_rank_ic.parquet", index=False)
        pooled_base.to_parquet(OUT / "daily_rank_ic" / f"{family}_baseline_daily_rank_ic.parquet", index=False)

    # compare against archived reports
    archived = {}
    m0_report = json.loads((ART / "m0_v3" / "report.json").read_text())
    archived["m0_v3_report_oos"] = m0_report["oos"]
    m0r_ev = json.loads((ART / "m0r_v2" / "promotion_evidence.json").read_text())["gate_evidence"]
    archived["m0r_v2_promotion_gate_evidence"] = m0r_ev
    verification["_archived"] = archived
    for fam, arch in (("m0_v3", archived["m0_v3_report_oos"]), ("m0r_v2", None)):
        if arch:
            v = verification[fam]["pooled"]
            verification[fam]["match_vs_archived"] = {
                "mean_rank_ic": {"recomputed": v["candidate_mean_rank_ic"], "archived": arch["mean_rank_ic"],
                                 "abs_diff": abs(v["candidate_mean_rank_ic"] - arch["mean_rank_ic"])},
                "delta_rank_ic": {"recomputed": v["delta_rank_ic"], "archived": arch["delta_rank_ic"],
                                  "abs_diff": abs(v["delta_rank_ic"] - arch["delta_rank_ic"])},
                "hac_t": {"recomputed": v["hac_t"], "archived": arch["hac_t_statistic"],
                          "abs_diff": abs(v["hac_t"] - arch["hac_t_statistic"])},
                "valid_days": {"recomputed": v["candidate_valid_days"], "archived": arch["valid_day_count"]},
            }
    ev = archived["m0r_v2_promotion_gate_evidence"]
    v = verification["m0r_v2"]["pooled"]
    verification["m0r_v2"]["match_vs_archived"] = {
        "mean_rank_ic": {"recomputed": v["candidate_mean_rank_ic"], "archived": ev["mean_rank_ic"],
                         "abs_diff": abs(v["candidate_mean_rank_ic"] - ev["mean_rank_ic"])},
        "delta_rank_ic": {"recomputed": v["delta_rank_ic"], "archived": ev["candidate_delta_rank_ic"],
                          "abs_diff": abs(v["delta_rank_ic"] - ev["candidate_delta_rank_ic"])},
        "hac_t": {"recomputed": v["hac_t"], "archived": ev["hac_t_statistic"],
                  "abs_diff": abs(v["hac_t"] - ev["hac_t_statistic"])},
        "valid_label_days": {"recomputed": v["valid_label_days"], "archived": ev["valid_test_day_count"]},
    }
    audit["verification"] = verification

    # dense9 status
    audit["dense9"] = {
        "status": "PREDICTIONS_NOT_PERSISTED",
        "searched": [
            "qrv2_research_artifacts/**/datasets/*/manifest.json (feature-count scan: 53 x 15-feature, 1 x 14-feature, 0 x 9-feature)",
            "qrv2_research_artifacts/*/models/*/ (no dense9 model bundle)",
            "qrv2_research_artifacts/*/predictions/*/manifest.json (all candidate manifests name 15-feature datasets)",
            "/tmp/es_metric_diag/ (does not exist)",
            "repo + all worktrees: grep 'dense9' -> 3 handoff docs + 1 code comment only, no ablation driver script",
        ],
        "only_surviving_evidence": "docs/handoffs/2026-09-08-m0-attribution-and-carrier-findings.md section 3 (summary table: dense9 tuned +0.00347, dense9 fixed +0.00177, all15 fixed +0.00109, sparse6 -0.02122)",
        "not_recoverable_without": "re-running the ablation (scripts lived in the deleted /tmp/es_metric_diag)",
    }
    (OUT / "audit_inventory.json").write_text(json.dumps(audit, ensure_ascii=False, indent=1) + "\n")
    print("wrote", OUT / "audit_inventory.json")
    for fam in ("m0_v3", "m0r_v2"):
        v = verification[fam]
        print(f"\n{fam} pooled recomputed: mean={v['pooled']['candidate_mean_rank_ic']:.6f} "
              f"delta={v['pooled']['delta_rank_ic']:+.6f} hac={v['pooled']['hac_t']:.3f} "
              f"valid_days={v['pooled']['candidate_valid_days']}")
        print(f"  per-fold delta: " + ", ".join(f"{f}={v['per_fold'][f]['delta_rank_ic']:+.5f}" for f in FOLDS))
        print("  match_vs_archived:", json.dumps(v["match_vs_archived"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
