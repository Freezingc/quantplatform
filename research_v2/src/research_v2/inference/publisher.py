"""Map healthy daily ensemble scores to publish-ready V2 prediction_daily batches."""

from __future__ import annotations

import hashlib
import json
from datetime import date

import numpy as np
import pandas as pd

from research_v2.db.publish import StagedBatch
from research_v2.runs.context import RunContext


def build_prediction_batch(
    ensemble: pd.DataFrame,
    *,
    model_id: str,
    model_version: int,
    universe_id: str,
    universe_version: int,
    inference_manifest_id: str,
    source_watermark_id: str,
    context: RunContext,
) -> StagedBatch:
    """Create one stable-rank daily prediction batch for the existing V2 two-phase publisher."""
    if not all((model_id, universe_id, inference_manifest_id, source_watermark_id)):
        raise ValueError("prediction identity fields cannot be empty")
    if model_version < 1 or universe_version < 1:
        raise ValueError("prediction model and universe versions must be positive")
    required = {"signal_date", "ts_code", "ensemble_score"}
    if missing := sorted(required.difference(ensemble.columns)):
        raise ValueError(f"ensemble predictions are missing columns: {missing}")
    keys = ["signal_date", "ts_code"]
    if ensemble.empty or ensemble.duplicated(keys).any():
        raise ValueError("ensemble predictions must be non-empty with unique business keys")
    ordered = ensemble.loc[:, keys + ["ensemble_score"]].copy()
    ordered["signal_date"] = pd.to_datetime(ordered["signal_date"], errors="raise").dt.date
    if ordered["signal_date"].nunique() != 1:
        raise ValueError("one prediction batch must contain exactly one signal_date")
    ordered["ensemble_score"] = pd.to_numeric(ordered["ensemble_score"], errors="coerce")
    if not np.isfinite(ordered["ensemble_score"]).all():
        raise ValueError("ensemble scores must be finite")
    ordered = ordered.sort_values(
        ["ensemble_score", "ts_code"],
        ascending=[False, True],
        kind="mergesort",
        ignore_index=True,
    )
    ordered["rank"] = pd.Series(range(1, len(ordered) + 1), dtype="int64")
    ordered["percentile_rank"] = ordered["ensemble_score"].rank(method="average", pct=True)
    materialized = pd.DataFrame(
        {
            "model_id": model_id,
            "model_version": model_version,
            "signal_date": ordered["signal_date"],
            "ts_code": ordered["ts_code"],
            "raw_score": ordered["ensemble_score"],
            "score": ordered["ensemble_score"],
            "rank": ordered["rank"],
            "percentile_rank": ordered["percentile_rank"],
            "universe_id": universe_id,
            "universe_version": universe_version,
            "inference_manifest_id": inference_manifest_id,
        }
    )
    materialized["run_id"] = context.run_id
    materialized["data_version"] = context.data_version
    signal_date_text = str(materialized.loc[0, "signal_date"])
    materialized["created_at"] = pd.Timestamp(signal_date_text)
    materialized["source_watermark_id"] = source_watermark_id
    materialized["record_hash"] = materialized.apply(_record_hash, axis=1)
    logical_partition = f"{signal_date_text}|{model_id}|{model_version}"
    return StagedBatch.create(
        object_name="prediction_daily",
        logical_partition=logical_partition,
        data_version=context.data_version,
        frame=materialized,
        business_keys=("signal_date", "model_id", "model_version", "ts_code"),
    )


def _record_hash(row: pd.Series) -> str:
    payload = {
        column: (value.isoformat() if isinstance(value, (date, pd.Timestamp)) else value)
        for column, value in row.items()
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True).encode()
    ).hexdigest()
