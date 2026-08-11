"""Build auditable V2 batches for factor RankIC and ICIR measurements."""

from __future__ import annotations

import hashlib
import json
from datetime import date

import pandas as pd

from research_v2.db.publish import StagedBatch
from research_v2.runs.context import RunContext

_DAILY_REQUIRED = {"factor_name", "signal_date", "rank_ic", "sample_count"}
_SUMMARY_REQUIRED = {
    "factor_name",
    "valid_day_count",
    "mean_rank_ic",
    "rank_ic_std",
    "icir",
    "positive_day_share",
    "mean_sample_count",
}


def build_factor_rank_ic_daily_batch(
    daily_rank_ic: pd.DataFrame,
    catalog: pd.DataFrame,
    context: RunContext,
    *,
    evaluation_scope: str,
    source_watermark_id: str,
    label_id: str = "forward_open_5d",
    label_version: int = 1,
) -> StagedBatch:
    """Map evaluator output into one monthly ``factor_rank_ic_daily`` batch."""
    if not evaluation_scope or not source_watermark_id:
        raise ValueError("evaluation_scope and source_watermark_id are required")
    if not label_id or label_version < 1:
        raise ValueError("label_id and positive label_version are required")
    _require_columns(daily_rank_ic, _DAILY_REQUIRED, "daily RankIC")
    mapped = _map_catalog(daily_rank_ic, catalog)
    if mapped.empty:
        raise ValueError("daily RankIC batch cannot be empty")
    mapped["signal_date"] = pd.to_datetime(mapped["signal_date"], errors="raise").dt.date
    materialized = pd.DataFrame(
        {
            "signal_date": mapped["signal_date"],
            "factor_id": mapped["factor_id"].astype(str),
            "factor_version": mapped["factor_version"].astype(int),
            "factor_name": mapped["factor_name"].astype(str),
            "rank_ic": pd.to_numeric(mapped["rank_ic"], errors="raise"),
            "sample_count": pd.to_numeric(mapped["sample_count"], errors="raise").astype(int),
            "evaluation_scope": evaluation_scope,
            "label_id": label_id,
            "label_version": label_version,
        }
    )
    _append_lineage(materialized, context, source_watermark_id)
    return StagedBatch.create(
        object_name="factor_rank_ic_daily",
        logical_partition=min(materialized["signal_date"]).strftime("%Y-%m"),
        data_version=context.data_version,
        frame=materialized,
        business_keys=("signal_date", "factor_id", "factor_version"),
        partition_date_column="signal_date",
    )


def build_factor_rank_ic_summary_batch(
    summary: pd.DataFrame,
    catalog: pd.DataFrame,
    context: RunContext,
    *,
    as_of_date: date,
    evaluation_scope: str,
    source_watermark_id: str,
    label_id: str = "forward_open_5d",
    label_version: int = 1,
) -> StagedBatch:
    """Map factor-level RankIC/ICIR metrics into an auditable summary batch."""
    if not evaluation_scope or not source_watermark_id:
        raise ValueError("evaluation_scope and source_watermark_id are required")
    if not label_id or label_version < 1:
        raise ValueError("label_id and positive label_version are required")
    _require_columns(summary, _SUMMARY_REQUIRED, "factor RankIC summary")
    mapped = _map_catalog(summary, catalog)
    if mapped.empty:
        raise ValueError("factor RankIC summary cannot be empty")
    materialized = pd.DataFrame(
        {
            "as_of_date": as_of_date,
            "factor_id": mapped["factor_id"].astype(str),
            "factor_version": mapped["factor_version"].astype(int),
            "factor_name": mapped["factor_name"].astype(str),
            "evaluation_scope": evaluation_scope,
            "valid_day_count": pd.to_numeric(
                mapped["valid_day_count"], errors="raise"
            ).astype(int),
            "mean_rank_ic": pd.to_numeric(mapped["mean_rank_ic"], errors="raise"),
            "rank_ic_std": pd.to_numeric(mapped["rank_ic_std"], errors="raise"),
            "icir": pd.to_numeric(mapped["icir"], errors="coerce"),
            "positive_day_share": pd.to_numeric(
                mapped["positive_day_share"], errors="raise"
            ),
            "mean_sample_count": pd.to_numeric(
                mapped["mean_sample_count"], errors="raise"
            ),
            "label_id": label_id,
            "label_version": label_version,
        }
    )
    _append_lineage(materialized, context, source_watermark_id)
    return StagedBatch.create(
        object_name="factor_rank_ic_summary",
        logical_partition=as_of_date.strftime("%Y-%m"),
        data_version=context.data_version,
        frame=materialized,
        business_keys=("as_of_date", "factor_id", "factor_version", "evaluation_scope"),
        partition_date_column="as_of_date",
    )


def _map_catalog(frame: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    _require_columns(catalog, {"factor_id", "factor_name", "factor_version"}, "factor catalog")
    if catalog["factor_name"].duplicated().any():
        raise ValueError("factor catalog contains duplicate factor_name")
    mapped = frame.copy(deep=True).merge(
        catalog.loc[:, ["factor_id", "factor_name", "factor_version"]],
        on="factor_name",
        how="left",
        validate="many_to_one",
        suffixes=("", "_catalog"),
    )
    if mapped["factor_id"].isna().any():
        unknown = sorted(mapped.loc[mapped["factor_id"].isna(), "factor_name"].unique())
        raise ValueError(f"unknown factor_name values: {unknown}")
    return mapped


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


def _require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"{label} is missing columns: {missing}")
