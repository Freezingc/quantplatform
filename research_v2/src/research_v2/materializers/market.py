"""Build auditable market-clean publish batches from readonly source data."""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any

import numpy as np
import pandas as pd

from research_v2.contracts.source import SourceBatch
from research_v2.db.publish import StagedBatch
from research_v2.pit.calendar import TradingCalendar
from research_v2.pit.market import (
    MarketDataQualityError,
    add_market_history_features,
    clean_market_daily,
)
from research_v2.runs.context import RunContext

_RUN_LINEAGE_COLUMNS = frozenset({"run_id", "data_version", "created_at"})


def _as_date(value: Any) -> date | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def active_listing_dates(
    security_scd: pd.DataFrame,
    trade_date: date,
) -> dict[str, date]:
    """Index exactly one active SCD record per security for one trade date."""
    required_columns = {"ts_code", "list_date", "valid_from", "valid_to"}
    missing_columns = sorted(required_columns.difference(security_scd.columns))
    if missing_columns:
        raise MarketDataQualityError(
            f"security SCD is missing columns: {missing_columns}"
        )
    normalized = security_scd.loc[:, list(required_columns)].copy()
    normalized["ts_code"] = normalized["ts_code"].astype(str)
    normalized["valid_from"] = normalized["valid_from"].map(_as_date)
    normalized["valid_to"] = normalized["valid_to"].map(_as_date)
    normalized["list_date"] = normalized["list_date"].map(_as_date)
    active = normalized.loc[
        normalized["valid_from"].notna()
        & (normalized["valid_from"] <= trade_date)
        & (normalized["valid_to"].isna() | (trade_date < normalized["valid_to"]))
    ]
    if active["ts_code"].duplicated().any():
        duplicates = sorted(active.loc[active["ts_code"].duplicated(), "ts_code"].unique())
        raise MarketDataQualityError(
            f"multiple active security SCD records on {trade_date.isoformat()}: {duplicates}"
        )
    missing_list_date = active["list_date"].isna()
    if missing_list_date.any():
        codes = sorted(active.loc[missing_list_date, "ts_code"].astype(str))
        raise MarketDataQualityError(f"security SCD has no list_date: {codes}")
    listing_dates: dict[str, date] = {}
    for row in active.loc[:, ["ts_code", "list_date"]].itertuples(index=False):
        list_date = _as_date(row.list_date)
        if list_date is None:
            raise MarketDataQualityError(f"security SCD has no list_date: {row.ts_code}")
        listing_dates[str(row.ts_code)] = list_date
    return listing_dates


def _hash_record(row: pd.Series) -> str:
    payload = {
        column: (value.isoformat() if isinstance(value, (date, pd.Timestamp)) else value)
        for column, value in row.items()
        if column not in _RUN_LINEAGE_COLUMNS
    }
    return hashlib.sha256(
        json.dumps(payload, default=str, ensure_ascii=False, sort_keys=True).encode()
    ).hexdigest()


def _without_invalid_adjustment_factors(source: pd.DataFrame) -> pd.DataFrame:
    if "adj_factor" not in source.columns:
        return source.copy(deep=True)
    adjustment_factors = pd.to_numeric(source["adj_factor"], errors="coerce")
    valid = np.isfinite(adjustment_factors) & (adjustment_factors > 0)
    return pd.DataFrame(source.loc[valid]).copy(deep=True)


def _invalid_adjustment_rows(source: pd.DataFrame) -> pd.DataFrame:
    required = {"ts_code", "trade_date", "adj_factor"}
    missing_columns = sorted(required.difference(source.columns))
    if missing_columns:
        raise MarketDataQualityError(
            f"market source is missing columns: {missing_columns}"
        )
    adjustment_factors = pd.to_numeric(source["adj_factor"], errors="coerce")
    valid = np.isfinite(adjustment_factors) & (adjustment_factors > 0)
    return pd.DataFrame(source.loc[~valid, ["trade_date", "ts_code", "adj_factor"]]).copy(
        deep=True
    )


def build_market_batch(
    source: SourceBatch,
    security_scd: pd.DataFrame,
    calendar: TradingCalendar,
    context: RunContext,
    *,
    publish_start: date | None = None,
    publish_end: date | None = None,
    quarantine_inactive_security: bool = False,
) -> StagedBatch:
    """Clean market source data and map it exactly to the V2 market table schema."""
    if source.table_name != "stock_factor_daily":
        raise ValueError("market batch requires a stock_factor_daily source batch")
    valid_source = _without_invalid_adjustment_factors(source.frame)
    if valid_source.empty:
        raise MarketDataQualityError("market source has no valid adjustment factors")
    if (publish_start is None) != (publish_end is None):
        raise ValueError("publish_start and publish_end must be provided together")
    if publish_start is not None and publish_end is not None and publish_start > publish_end:
        raise ValueError("publish_start cannot be after publish_end")
    clean = add_market_history_features(clean_market_daily(valid_source))
    clean["trade_date"] = pd.to_datetime(clean["trade_date"], errors="raise").dt.date
    if publish_start is not None and publish_end is not None:
        clean = clean.loc[
            clean["trade_date"].between(publish_start, publish_end, inclusive="both")
        ].copy()
        if clean.empty:
            raise MarketDataQualityError("market source has no valid rows in publish window")
    trade_dates: set[date] = set()
    for value in clean["trade_date"]:
        trade_date = _as_date(value)
        if trade_date is None:
            raise MarketDataQualityError("clean market contains a null trade_date")
        trade_dates.add(trade_date)
    listing_dates_by_trade_date = {
        trade_date: active_listing_dates(security_scd, trade_date)
        for trade_date in sorted(trade_dates)
    }
    active_security: list[bool] = []
    for row in clean.loc[:, ["trade_date", "ts_code"]].itertuples(index=False):
        trade_date = _as_date(row.trade_date)
        if trade_date is None:
            raise MarketDataQualityError("clean market contains a null trade_date")
        active_security.append(str(row.ts_code) in listing_dates_by_trade_date[trade_date])
    if not all(active_security):
        if not quarantine_inactive_security:
            missing_index = next(
                index for index, active in enumerate(active_security) if not active
            )
            missing_row = clean.iloc[missing_index]
            trade_date = _as_date(missing_row["trade_date"])
            assert trade_date is not None
            raise MarketDataQualityError(
                f"{missing_row['ts_code']} has no active security SCD record on "
                f"{trade_date.isoformat()}"
            )
        clean = clean.loc[active_security].copy()
        if clean.empty:
            raise MarketDataQualityError("market publish window has no active security SCD rows")
    listing_ages: list[int] = []
    for row in clean.loc[:, ["trade_date", "ts_code"]].itertuples(index=False):
        trade_date = _as_date(row.trade_date)
        if trade_date is None:
            raise MarketDataQualityError("clean market contains a null trade_date")
        listing_date = listing_dates_by_trade_date[trade_date].get(str(row.ts_code))
        if listing_date is None:
            raise MarketDataQualityError(
                f"{row.ts_code} has no active security SCD record on {trade_date.isoformat()}"
            )
        listing_ages.append(calendar.count(listing_date, trade_date))
    clean["listing_age_trading_days"] = listing_ages
    volume_lots = pd.to_numeric(clean["vol"], errors="raise")
    volume_shares = volume_lots * 100.0
    rounded_volume_shares = np.rint(volume_shares)
    if not np.isclose(volume_shares, rounded_volume_shares, rtol=0.0, atol=1e-6).all():
        raise MarketDataQualityError("source volume cannot be converted from lots to whole shares")
    materialized = pd.DataFrame(
        {
            "trade_date": clean["trade_date"],
            "ts_code": clean["ts_code"],
            "raw_open": clean["open"],
            "raw_high": clean["high"],
            "raw_low": clean["low"],
            "raw_close": clean["close"],
            "volume": rounded_volume_shares.astype("int64"),
            "amount": clean["amount"],
            "adj_factor": clean["adj_factor"],
            "adjusted_open": clean["adjusted_open"],
            "valid_trade_day": clean["valid_trade_day"],
            "valid_days_20": clean["valid_days_20"],
            "adv20": clean["adv20"],
            "listing_age_trading_days": clean["listing_age_trading_days"],
            "source_flags": "YESLAB;VOLUME_LOTS_X100",
            "dedup_resolution": clean["dedup_resolution"],
        }
    )
    materialized["run_id"] = context.run_id
    materialized["data_version"] = context.data_version
    materialized["created_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    materialized["source_watermark_id"] = source.watermark_id
    materialized["record_hash"] = materialized.apply(_hash_record, axis=1)
    logical_partition = min(materialized["trade_date"]).strftime("%Y-%m")
    return StagedBatch.create(
        object_name="market_daily_clean",
        logical_partition=logical_partition,
        data_version=context.data_version,
        frame=materialized,
        business_keys=("trade_date", "ts_code"),
        partition_date_column="trade_date",
    )


def build_market_security_quarantine_batch(
    source: SourceBatch,
    security_scd: pd.DataFrame,
    context: RunContext,
    *,
    publish_start: date | None = None,
    publish_end: date | None = None,
) -> StagedBatch:
    """Audit valid market rows that lack an active security SCD record."""
    if source.table_name != "stock_factor_daily":
        raise ValueError("market security quarantine requires a stock_factor_daily source batch")
    if (publish_start is None) != (publish_end is None):
        raise ValueError("publish_start and publish_end must be provided together")
    if publish_start is not None and publish_end is not None and publish_start > publish_end:
        raise ValueError("publish_start cannot be after publish_end")
    candidates = _without_invalid_adjustment_factors(source.frame).loc[
        :, ["trade_date", "ts_code"]
    ].copy()
    candidates["trade_date"] = pd.to_datetime(candidates["trade_date"], errors="raise").dt.date
    if publish_start is not None and publish_end is not None:
        candidates = candidates.loc[
            candidates["trade_date"].between(publish_start, publish_end, inclusive="both")
        ].copy()
    if candidates.empty:
        raise MarketDataQualityError("market security quarantine has no candidate rows")
    active_by_date = {
        trade_date: active_listing_dates(security_scd, trade_date)
        for trade_date in sorted(set(candidates["trade_date"]))
    }
    missing = candidates.loc[
        [
            str(row.ts_code) not in active_by_date[row.trade_date]
            for row in candidates.itertuples(index=False)
        ]
    ].copy()
    if missing.empty:
        raise MarketDataQualityError("market security quarantine has no inactive security rows")
    materialized = pd.DataFrame(
        {
            "trade_date": missing["trade_date"],
            "ts_code": missing["ts_code"].astype(str),
            "field_name": "security_scd",
            "reason": "NO_ACTIVE_SECURITY_SCD",
            "raw_value": "",
        }
    )
    materialized["run_id"] = context.run_id
    materialized["data_version"] = context.data_version
    materialized["created_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    materialized["source_watermark_id"] = source.watermark_id
    materialized["record_hash"] = materialized.apply(_hash_record, axis=1)
    return StagedBatch.create(
        object_name="market_quarantine",
        logical_partition=min(materialized["trade_date"]).strftime("%Y-%m"),
        data_version=context.data_version,
        frame=materialized,
        business_keys=("trade_date", "ts_code", "field_name"),
        partition_date_column="trade_date",
    )


def build_market_quarantine_batch(source: SourceBatch, context: RunContext) -> StagedBatch:
    """Materialize invalid adjustment factors for audit without admitting them downstream."""
    if source.table_name != "stock_factor_daily":
        raise ValueError("market quarantine requires a stock_factor_daily source batch")
    invalid_rows = _invalid_adjustment_rows(source.frame)
    if invalid_rows.empty:
        raise MarketDataQualityError("market source contains no quarantinable rows")
    materialized = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(invalid_rows["trade_date"], errors="raise").dt.date,
            "ts_code": invalid_rows["ts_code"].astype(str),
            "field_name": "adj_factor",
            "reason": "INVALID_ADJ_FACTOR",
            "raw_value": invalid_rows["adj_factor"].map(
                lambda value: "" if pd.isna(value) else str(value)
            ),
        }
    )
    materialized["run_id"] = context.run_id
    materialized["data_version"] = context.data_version
    materialized["created_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)
    materialized["source_watermark_id"] = source.watermark_id
    materialized["record_hash"] = materialized.apply(_hash_record, axis=1)
    logical_partition = min(materialized["trade_date"]).strftime("%Y-%m")
    return StagedBatch.create(
        object_name="market_quarantine",
        logical_partition=logical_partition,
        data_version=context.data_version,
        frame=materialized,
        business_keys=("trade_date", "ts_code", "field_name"),
        partition_date_column="trade_date",
    )
