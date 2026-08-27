"""Build auditable model OOS RankIC result batches."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import date

import pandas as pd

from research_v2.db.publish import StagedBatch
from research_v2.evaluation.rank_ic import RankICSummary
from research_v2.runs.context import RunContext

_DAILY_CORE_COLUMNS = [
    "signal_date",
    "evaluation_id",
    "model_id",
    "fold_id",
    "role",
    "rank_ic",
    "sample_count",
]
_DAILY_DIAGNOSTIC_COLUMNS = [
    "eligible_count",
    "valid_label_count",
    "coverage",
    "prediction_unique_count",
    "prediction_std",
    "rank_ic_valid",
]
_LINEAGE_COLUMNS = [
    "run_id",
    "data_version",
    "created_at",
    "source_watermark_id",
    "record_hash",
]


def build_model_oos_daily_batch(
    daily_rank_ic: pd.DataFrame,
    context: RunContext,
    *,
    evaluation_id: str,
    model_id: str,
    fold_id: str,
    source_watermark_id: str,
    role: str = "TEST",
) -> StagedBatch:
    """Map frozen OOS daily RankIC observations into a publishable batch."""
    _validate_identity(evaluation_id, model_id, fold_id, source_watermark_id, role)
    required = {"signal_date", "rank_ic", "sample_count"}
    if missing := sorted(required.difference(daily_rank_ic.columns)):
        raise ValueError(f"daily OOS RankIC is missing columns: {missing}")
    if daily_rank_ic.empty:
        raise ValueError("daily OOS RankIC cannot be empty")
    signal_dates = pd.to_datetime(daily_rank_ic["signal_date"], errors="raise").dt.date
    diagnostics = {
        "eligible_count": _optional_integer(daily_rank_ic, "eligible_count"),
        "valid_label_count": _optional_integer(daily_rank_ic, "valid_label_count"),
        "coverage": _optional_numeric(daily_rank_ic, "coverage"),
        "prediction_unique_count": _optional_integer(daily_rank_ic, "prediction_unique_count"),
        "prediction_std": _optional_numeric(daily_rank_ic, "prediction_std"),
        "rank_ic_valid": _optional_boolean(daily_rank_ic, "rank_ic_valid"),
    }
    materialized = pd.DataFrame(
        {
            "signal_date": signal_dates.reset_index(drop=True),
            "evaluation_id": evaluation_id,
            "model_id": model_id,
            "fold_id": fold_id,
            "role": role,
            "rank_ic": _required_numeric(daily_rank_ic, "rank_ic"),
            "sample_count": pd.to_numeric(daily_rank_ic["sample_count"], errors="raise")
            .astype(int)
            .reset_index(drop=True),
        }
    )
    materialized = pd.concat(
        [materialized, pd.DataFrame(diagnostics)],
        axis=1,
    )
    _append_lineage(materialized, context, source_watermark_id)
    materialized = materialized.loc[
        :, _DAILY_CORE_COLUMNS + _LINEAGE_COLUMNS + _DAILY_DIAGNOSTIC_COLUMNS
    ]
    return StagedBatch.create(
        object_name="model_oos_rank_ic_daily",
        logical_partition=min(materialized["signal_date"]).strftime("%Y-%m"),
        data_version=context.data_version,
        frame=materialized,
        business_keys=("signal_date", "evaluation_id", "model_id"),
        partition_date_column="signal_date",
    )


def build_model_oos_summary_batch(
    summary: RankICSummary,
    context: RunContext,
    *,
    as_of_date: date,
    evaluation_id: str,
    model_id: str,
    fold_id: str,
    source_watermark_id: str,
    positive_day_share: float,
    mean_sample_count: float,
    role: str = "TEST",
    extra_metrics: Mapping[str, object] | None = None,
) -> StagedBatch:
    """Map one frozen OOS summary and optional test evidence into a batch."""
    _validate_identity(evaluation_id, model_id, fold_id, source_watermark_id, role)
    metrics: dict[str, object] = {
        "valid_day_count": summary.valid_day_count,
        "mean_rank_ic": summary.mean_rank_ic,
        "rank_ic_std": summary.rank_ic_std,
        "icir": summary.icir,
        "positive_day_share": positive_day_share,
        "mean_sample_count": mean_sample_count,
    }
    if extra_metrics:
        metrics.update(extra_metrics)
    materialized = pd.DataFrame(
        {
            "as_of_date": [as_of_date],
            "evaluation_id": [evaluation_id],
            "model_id": [model_id],
            "fold_id": [fold_id],
            "role": [role],
            "valid_day_count": [summary.valid_day_count],
            "mean_rank_ic": [summary.mean_rank_ic],
            "rank_ic_std": [summary.rank_ic_std],
            "icir": [summary.icir],
            "positive_day_share": [positive_day_share],
            "mean_sample_count": [mean_sample_count],
            "metrics_json": [json.dumps(metrics, ensure_ascii=False, sort_keys=True)],
        }
    )
    _append_lineage(materialized, context, source_watermark_id)
    return StagedBatch.create(
        object_name="model_oos_rank_ic_summary",
        logical_partition=as_of_date.strftime("%Y-%m"),
        data_version=context.data_version,
        frame=materialized,
        business_keys=("as_of_date", "evaluation_id", "model_id"),
        partition_date_column="as_of_date",
    )


def _validate_identity(
    evaluation_id: str,
    model_id: str,
    fold_id: str,
    source_watermark_id: str,
    role: str,
) -> None:
    if not all(value.strip() for value in (evaluation_id, model_id, fold_id, source_watermark_id)):
        raise ValueError("OOS identity fields cannot be empty")
    if role != "TEST":
        raise PermissionError("frozen model OOS materialization requires role TEST")


def _append_lineage(frame: pd.DataFrame, context: RunContext, source_watermark_id: str) -> None:
    frame["run_id"] = context.run_id
    frame["data_version"] = context.data_version
    frame["created_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    frame["source_watermark_id"] = source_watermark_id
    frame["record_hash"] = frame.apply(_record_hash, axis=1)


def _record_hash(row: pd.Series) -> str:
    payload = {
        column: (value.isoformat() if isinstance(value, (date, pd.Timestamp)) else value)
        for column, value in row.items()
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True).encode()
    ).hexdigest()


def _required_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(frame[column], errors="raise").reset_index(drop=True)


def _optional_numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=range(len(frame)), dtype="Float64")
    return pd.to_numeric(frame[column], errors="raise").reset_index(drop=True)


def _optional_integer(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=range(len(frame)), dtype="Int64")
    return pd.to_numeric(frame[column], errors="raise").astype("Int64").reset_index(drop=True)


def _optional_boolean(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(pd.NA, index=range(len(frame)), dtype="boolean")
    return frame[column].astype("boolean").reset_index(drop=True)
