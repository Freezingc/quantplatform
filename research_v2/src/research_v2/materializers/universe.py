"""Build versioned, point-in-time universe membership batches."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any

import pandas as pd

from research_v2.contracts.source import SourceBatch
from research_v2.db.publish import StagedBatch
from research_v2.runs.context import RunContext
from research_v2.universe.builder import UniverseDefinition, build_universe_snapshot

_RUN_LINEAGE_COLUMNS = frozenset({"run_id", "data_version", "created_at"})


def _as_date(value: Any) -> date | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def _active_boards(security_scd: pd.DataFrame, signal_date: date) -> dict[str, str]:
    required = {"ts_code", "board", "valid_from", "valid_to"}
    missing_columns = sorted(required.difference(security_scd.columns))
    if missing_columns:
        raise ValueError(f"security SCD is missing columns: {missing_columns}")
    normalized = security_scd.loc[:, list(required)].copy()
    normalized["ts_code"] = normalized["ts_code"].astype(str)
    normalized["valid_from"] = pd.Series(
        normalized["valid_from"].map(_as_date), index=normalized.index, dtype="object"
    )
    normalized["valid_to"] = pd.Series(
        normalized["valid_to"].map(_as_date), index=normalized.index, dtype="object"
    )
    active = normalized.loc[
        normalized["valid_from"].notna()
        & (normalized["valid_from"] <= signal_date)
        & (normalized["valid_to"].isna() | (signal_date < normalized["valid_to"]))
    ]
    duplicate_codes = active.loc[active["ts_code"].duplicated(), "ts_code"]
    if not duplicate_codes.empty:
        raise ValueError(
            f"security SCD has multiple active boards on {signal_date.isoformat()}: "
            f"{sorted(duplicate_codes.unique())}"
        )
    return dict(zip(active["ts_code"], active["board"].astype(str), strict=True))


def _source_watermark_id(tradability: SourceBatch, security_scd: pd.DataFrame) -> str:
    security_watermarks = (
        sorted(security_scd["source_watermark_id"].dropna().astype(str).unique())
        if "source_watermark_id" in security_scd.columns
        else []
    )
    payload = "|".join([tradability.watermark_id, *security_watermarks])
    return hashlib.sha256(payload.encode()).hexdigest()


def _record_hash(row: pd.Series) -> str:
    payload = {
        column: (value.isoformat() if isinstance(value, (date, pd.Timestamp)) else value)
        for column, value in row.items()
        if column not in _RUN_LINEAGE_COLUMNS
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True).encode()
    ).hexdigest()


def build_universe_batch(
    tradability: SourceBatch,
    security_scd: pd.DataFrame,
    definition: UniverseDefinition,
    context: RunContext,
) -> StagedBatch:
    """Resolve every eligible signal-date cross-section under one immutable rule version."""
    if tradability.table_name != "tradability_daily":
        raise ValueError("universe requires a tradability_daily batch")
    required = {
        "trade_date",
        "ts_code",
        "is_listed",
        "is_st",
        "is_suspended",
        "listing_age_trading_days",
        "valid_days_20",
        "adv20",
        "reason_bitmap",
    }
    missing_columns = sorted(required.difference(tradability.frame.columns))
    if missing_columns:
        raise ValueError(f"tradability is missing columns: {missing_columns}")
    snapshots: list[pd.DataFrame] = []
    for signal_date, records in tradability.frame.groupby("trade_date", sort=True):
        resolved_date = _as_date(signal_date)
        if resolved_date is None:
            raise ValueError("tradability contains a null trade_date")
        candidates = records.copy(deep=True)
        candidates["board"] = candidates["ts_code"].astype(str).map(
            _active_boards(security_scd, resolved_date)
        )
        candidates["board"] = candidates["board"].fillna("OTHER")
        snapshots.append(build_universe_snapshot(candidates, definition))
    if not snapshots:
        raise ValueError("universe materialization requires at least one tradability row")
    materialized = pd.concat(snapshots, ignore_index=True)
    materialized["run_id"] = context.run_id
    materialized["data_version"] = context.data_version
    materialized["created_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    materialized["source_watermark_id"] = _source_watermark_id(tradability, security_scd)
    materialized["record_hash"] = materialized.apply(_record_hash, axis=1)
    logical_partition = pd.Timestamp(materialized["signal_date"].min()).strftime("%Y-%m")
    return StagedBatch.create(
        object_name="universe_daily",
        logical_partition=logical_partition,
        data_version=context.data_version,
        frame=materialized,
        business_keys=("signal_date", "universe_id", "universe_version", "ts_code"),
        partition_date_column="signal_date",
    )
