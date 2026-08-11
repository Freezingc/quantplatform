"""Build auditable point-in-time tradability batches."""

from __future__ import annotations

import hashlib
import json
from datetime import date

import pandas as pd

from research_v2.contracts.source import SourceBatch
from research_v2.db.publish import StagedBatch
from research_v2.pit.calendar import TradingCalendar
from research_v2.runs.context import RunContext
from research_v2.universe.tradability import build_tradability_daily


def _watermark_id(*batches: SourceBatch) -> str:
    payload = "|".join(sorted(batch.watermark_id for batch in batches))
    return hashlib.sha256(payload.encode()).hexdigest()


def _record_hash(row: pd.Series) -> str:
    payload = {
        column: (value.isoformat() if isinstance(value, (date, pd.Timestamp)) else value)
        for column, value in row.items()
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, default=str, sort_keys=True).encode()
    ).hexdigest()


def build_tradability_batch(
    market: SourceBatch,
    security_scd: pd.DataFrame,
    suspensions: SourceBatch,
    calendar: TradingCalendar,
    context: RunContext,
    *,
    rule_version: int = 1,
    minimum_listing_trading_days: int = 252,
    minimum_valid_days_20: int = 15,
) -> StagedBatch:
    """Map signal-date tradability into the immutable V2 storage contract."""
    if market.table_name != "market_daily_clean":
        raise ValueError("tradability requires a market_daily_clean batch")
    if suspensions.table_name != "stock_trading_suspension_info":
        raise ValueError("tradability requires a stock_trading_suspension_info batch")
    if rule_version < 1:
        raise ValueError("rule_version must be positive")
    resolved = build_tradability_daily(
        market.frame,
        security_scd,
        suspensions.frame,
        calendar,
        minimum_listing_trading_days=minimum_listing_trading_days,
        minimum_valid_days_20=minimum_valid_days_20,
    )
    materialized = resolved.copy(deep=True)
    materialized["rule_version"] = rule_version
    materialized["run_id"] = context.run_id
    materialized["data_version"] = context.data_version
    materialized["created_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    materialized["source_watermark_id"] = _watermark_id(market, suspensions)
    materialized["record_hash"] = materialized.apply(_record_hash, axis=1)
    logical_partition = pd.Timestamp(materialized["trade_date"].min()).strftime("%Y-%m")
    return StagedBatch.create(
        object_name="tradability_daily",
        logical_partition=logical_partition,
        data_version=context.data_version,
        frame=materialized,
        business_keys=("trade_date", "ts_code", "rule_version"),
        partition_date_column="trade_date",
    )
