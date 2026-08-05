"""Audit TRAIN-to-2025 performance drift for the SH/SZ admission candidates."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Sequence
from pathlib import Path

import pandas as pd

DEFAULT_FACTOR_IDS = (
    "yeslab_380",
    "yeslab_431",
    "yeslab_437",
    "yeslab_155",
    "yeslab_98",
    "yeslab_94",
    "yeslab_376",
    "yeslab_396",
    "yeslab_442",
)
DEFAULT_QUALITY_ARTIFACTS = (
    Path("research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_quality_16_v1.json"),
    Path("research_v2/artifacts/diagnostics/tushare_canonical_core_quality_v1.json"),
    Path("research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_quality_155_v1.json"),
    Path("research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_quality_94_98_v1.json"),
)


def _sign(value: float) -> int:
    return 1 if value > 0 else -1 if value < 0 else 0


def audit_performance_drift(
    frame: pd.DataFrame,
    *,
    minimum_validation_abs_mean_rank_ic: float = 0.02,
    minimum_validation_abs_icir: float = 0.50,
    minimum_abs_mean_rank_ic_ratio: float = 0.50,
    minimum_abs_icir_ratio: float = 0.50,
) -> dict[str, object]:
    """Check fixed-direction validation quality and relative TRAIN degradation."""

    required = {
        "factor_id",
        "train_mean_rank_ic",
        "train_icir",
        "validation_mean_rank_ic",
        "validation_icir",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"performance drift input is missing columns: {missing}")
    rows: list[dict[str, object]] = []
    for raw in frame.loc[:, sorted(required)].to_dict(orient="records"):
        factor_id = str(raw["factor_id"])
        train_mean = float(raw["train_mean_rank_ic"])
        train_icir = float(raw["train_icir"])
        validation_mean = float(raw["validation_mean_rank_ic"])
        validation_icir = float(raw["validation_icir"])
        values = (train_mean, train_icir, validation_mean, validation_icir)
        if not all(math.isfinite(value) for value in values):
            raise ValueError(f"performance metrics are not finite for {factor_id}")
        direction_consistent = _sign(train_mean) != 0 and _sign(train_mean) == _sign(validation_mean)
        mean_ratio = abs(validation_mean) / abs(train_mean) if train_mean else 0.0
        icir_ratio = abs(validation_icir) / abs(train_icir) if train_icir else 0.0
        validation_predictive = (
            direction_consistent
            and abs(validation_mean) >= minimum_validation_abs_mean_rank_ic
            and abs(validation_icir) >= minimum_validation_abs_icir
        )
        drift_pass = (
            direction_consistent
            and mean_ratio >= minimum_abs_mean_rank_ic_ratio
            and icir_ratio >= minimum_abs_icir_ratio
        )
        rows.append(
            {
                "factor_id": factor_id,
                "train_mean_rank_ic": train_mean,
                "train_icir": train_icir,
                "validation_mean_rank_ic": validation_mean,
                "validation_icir": validation_icir,
                "direction_consistent": direction_consistent,
                "abs_mean_rank_ic_ratio": mean_ratio,
                "abs_icir_ratio": icir_ratio,
                "validation_predictive_status": "PASS" if validation_predictive else "FAIL",
                "drift_status": "PASS" if drift_pass else "FAIL",
            }
        )
    rows.sort(key=lambda row: str(row["factor_id"]))
    direction_flips = [
        str(row["factor_id"]) for row in rows if not bool(row["direction_consistent"])
    ]
    validation_failures = [
        str(row["factor_id"])
        for row in rows
        if row["validation_predictive_status"] == "FAIL"
    ]
    drift_failures = [str(row["factor_id"]) for row in rows if row["drift_status"] == "FAIL"]
    return {
        "status": "PASS" if not (direction_flips or validation_failures or drift_failures) else "FAIL",
        "factor_count": len(rows),
        "rows": rows,
        "direction_flip_factor_ids": direction_flips,
        "validation_predictive_fail_factor_ids": validation_failures,
        "drift_fail_factor_ids": drift_failures,
        "thresholds": {
            "minimum_validation_abs_mean_rank_ic": minimum_validation_abs_mean_rank_ic,
            "minimum_validation_abs_icir": minimum_validation_abs_icir,
            "minimum_abs_mean_rank_ic_ratio": minimum_abs_mean_rank_ic_ratio,
            "minimum_abs_icir_ratio": minimum_abs_icir_ratio,
        },
        "database_writes": False,
        "locked_test_read": False,
    }


def _append_payload_rows(payload: dict[str, object], rows: dict[str, dict[str, object]]) -> None:
    for summary in payload.get("summaries", []):
        if not isinstance(summary, dict) or "factor_id" not in summary:
            continue
        factor_id = str(summary["factor_id"])
        if {"train_mean_rank_ic", "train_icir", "validation_mean_rank_ic", "validation_icir"}.issubset(summary):
            rows[factor_id] = {
                "factor_id": factor_id,
                "train_mean_rank_ic": summary["train_mean_rank_ic"],
                "train_icir": summary["train_icir"],
                "validation_mean_rank_ic": summary["validation_mean_rank_ic"],
                "validation_icir": summary["validation_icir"],
            }
        elif isinstance(summary.get("train"), dict) and isinstance(summary.get("validation_2025"), dict):
            train = summary["train"]
            validation = summary["validation_2025"]
            rows[factor_id] = {
                "factor_id": factor_id,
                "train_mean_rank_ic": train["mean_rank_ic"],
                "train_icir": train["icir"],
                "validation_mean_rank_ic": validation["mean_rank_ic"],
                "validation_icir": validation["icir"],
            }
    train = payload.get("train", {})
    validation = payload.get("validation", {})
    if isinstance(train, dict) and isinstance(validation, dict):
        metrics = train.get("metrics", {})
        validation_metrics = validation.get("metrics", {})
        if isinstance(metrics, dict) and isinstance(validation_metrics, dict):
            for factor_id, train_metric in metrics.items():
                validation_metric = validation_metrics.get(factor_id)
                if isinstance(train_metric, dict) and isinstance(validation_metric, dict):
                    rows[str(factor_id)] = {
                        "factor_id": str(factor_id),
                        "train_mean_rank_ic": train_metric["mean_rank_ic"],
                        "train_icir": train_metric["icir"],
                        "validation_mean_rank_ic": validation_metric["mean_rank_ic"],
                        "validation_icir": validation_metric["icir"],
                    }
    summary = payload.get("summary")
    if isinstance(summary, dict) and isinstance(summary.get("train"), dict) and isinstance(summary.get("validation_2025"), dict):
        rows[str(summary["factor_id"])] = {
            "factor_id": str(summary["factor_id"]),
            "train_mean_rank_ic": summary["train"]["mean_rank_ic"],
            "train_icir": summary["train"]["icir"],
            "validation_mean_rank_ic": summary["validation_2025"]["mean_rank_ic"],
            "validation_icir": summary["validation_2025"]["icir"],
        }


def load_quality_rows(paths: Sequence[Path], factor_ids: Sequence[str]) -> pd.DataFrame:
    rows: dict[str, dict[str, object]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"quality artifact must be an object: {path}")
        _append_payload_rows(payload, rows)
    missing = sorted(set(map(str, factor_ids)).difference(rows))
    if missing:
        raise ValueError(f"quality artifacts are missing factors: {missing}")
    return pd.DataFrame([rows[factor_id] for factor_id in factor_ids])


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--factor-id", action="append")
    parser.add_argument("--quality-artifact", type=Path, action="append")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    args.factor_id = tuple(dict.fromkeys(args.factor_id or DEFAULT_FACTOR_IDS))
    args.quality_artifact = tuple(args.quality_artifact or DEFAULT_QUALITY_ARTIFACTS)
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    frame = load_quality_rows(args.quality_artifact, args.factor_id)
    result = audit_performance_drift(frame)
    result.update(
        {
            "scope": "TUSHARE_SHSZ_PERFORMANCE_DRIFT_AUDIT_V1",
            "market_scope": ["SH", "SZ"],
            "excluded_exchanges": ["BJ"],
            "train_window": "2020-01-02/2024-12-31",
            "validation_window": "2025-01-02/2025-12-23",
            "quality_artifacts": [str(path) for path in args.quality_artifact],
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"status": result["status"], "output": str(args.output)}))
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
