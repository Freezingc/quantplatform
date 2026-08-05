"""Audit TRAIN-to-2025 performance drift for the SH/SZ admission candidates."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping, Sequence
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
    Path("research_v2/artifacts/diagnostics/tushare_canonical_core_quality_v1.json"),
    Path("research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_train_16_quality_v2/quality_summary.json"),
    Path("research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_validation_16_quality_v2/quality_summary.json"),
    Path("research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_train_legacy3_quality_v2/quality_summary.json"),
    Path("research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_validation_legacy3_quality_v2/quality_summary.json"),
    Path("research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_train_canonical2_quality_v2/quality_summary.json"),
    Path("research_v2/artifacts/diagnostics/tushare_shsz_rematerialized_validation_canonical2_quality_v2/quality_summary.json"),
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


def _store_period_row(
    rows: dict[str, dict[str, object]], factor_id: str, period: str, mean_rank_ic: object, icir: object
) -> None:
    rows[factor_id] = {
        "factor_id": factor_id,
        f"{period}_mean_rank_ic": mean_rank_ic,
        f"{period}_icir": icir,
    }


def _append_payload_rows(
    payload: dict[str, object],
    combined: dict[str, dict[str, object]],
    train: dict[str, dict[str, object]],
    validation: dict[str, dict[str, object]],
) -> None:
    for summary in payload.get("summaries", []):
        if not isinstance(summary, Mapping) or "factor_id" not in summary:
            continue
        factor_id = str(summary["factor_id"])
        if {"train_mean_rank_ic", "train_icir", "validation_mean_rank_ic", "validation_icir"}.issubset(summary):
            combined[factor_id] = {
                "factor_id": factor_id,
                "train_mean_rank_ic": summary["train_mean_rank_ic"],
                "train_icir": summary["train_icir"],
                "validation_mean_rank_ic": summary["validation_mean_rank_ic"],
                "validation_icir": summary["validation_icir"],
            }
        elif isinstance(summary.get("train"), Mapping) and isinstance(summary.get("validation_2025"), Mapping):
            train = summary["train"]
            validation = summary["validation_2025"]
            combined[factor_id] = {
                "factor_id": factor_id,
                "train_mean_rank_ic": train["mean_rank_ic"],
                "train_icir": train["icir"],
                "validation_mean_rank_ic": validation["mean_rank_ic"],
                "validation_icir": validation["icir"],
            }
        elif {"mean_rank_ic", "icir"}.issubset(summary):
            start = str(payload.get("start", ""))
            _store_period_row(
                train if start < "2025" else validation,
                factor_id,
                "train" if start < "2025" else "validation",
                summary["mean_rank_ic"],
                summary["icir"],
            )
    payload_train = payload.get("train", {})
    payload_validation = payload.get("validation", {})
    if isinstance(payload_train, Mapping) and isinstance(payload_validation, Mapping):
        metrics = payload_train.get("metrics", {})
        validation_metrics = payload_validation.get("metrics", {})
        if isinstance(metrics, Mapping) and isinstance(validation_metrics, Mapping):
            for factor_id, train_metric in metrics.items():
                validation_metric = validation_metrics.get(factor_id)
                if isinstance(train_metric, Mapping) and isinstance(validation_metric, Mapping):
                    combined[str(factor_id)] = {
                        "factor_id": str(factor_id),
                        "train_mean_rank_ic": train_metric["mean_rank_ic"],
                        "train_icir": train_metric["icir"],
                        "validation_mean_rank_ic": validation_metric["mean_rank_ic"],
                        "validation_icir": validation_metric["icir"],
                    }
    summary = payload.get("summary")
    if isinstance(summary, Mapping) and isinstance(summary.get("train"), Mapping) and isinstance(summary.get("validation_2025"), Mapping):
        combined[str(summary["factor_id"])] = {
            "factor_id": str(summary["factor_id"]),
            "train_mean_rank_ic": summary["train"]["mean_rank_ic"],
            "train_icir": summary["train"]["icir"],
            "validation_mean_rank_ic": summary["validation_2025"]["mean_rank_ic"],
            "validation_icir": summary["validation_2025"]["icir"],
        }


def load_quality_rows(paths: Sequence[Path], factor_ids: Sequence[str]) -> pd.DataFrame:
    combined: dict[str, dict[str, object]] = {}
    train: dict[str, dict[str, object]] = {}
    validation: dict[str, dict[str, object]] = {}
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"quality artifact must be an object: {path}")
        _append_payload_rows(payload, combined, train, validation)
    rows: dict[str, dict[str, object]] = {factor_id: dict(row) for factor_id, row in combined.items()}
    for factor_id, period_row in train.items():
        rows.setdefault(factor_id, {"factor_id": factor_id}).update(period_row)
    for factor_id, period_row in validation.items():
        rows.setdefault(factor_id, {"factor_id": factor_id}).update(period_row)
    normalized: list[dict[str, object]] = []
    for factor_id in factor_ids:
        row = rows.get(str(factor_id), {})
        row.setdefault("factor_id", str(factor_id))
        normalized.append(row)
    missing = sorted(
        factor_id
        for factor_id, row in zip(map(str, factor_ids), normalized)
        if not {"train_mean_rank_ic", "train_icir", "validation_mean_rank_ic", "validation_icir"}.issubset(row)
    )
    if missing:
        raise ValueError(f"quality artifacts are missing factors: {missing}")
    return pd.DataFrame(normalized)


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
