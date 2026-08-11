"""Deterministic market-data cleaning and point-in-time derived fields."""

from __future__ import annotations

import numpy as np
import pandas as pd


class MarketDataQualityError(ValueError):
    """Raised when source market data violates a blocking invariant."""


def clean_market_daily(raw: pd.DataFrame) -> pd.DataFrame:
    """Validate and deduplicate daily market observations."""
    required = {
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "vol",
        "amount",
        "adj_factor",
    }
    missing = sorted(required.difference(raw.columns))
    if missing:
        raise MarketDataQualityError(f"market source is missing required columns: {missing}")

    frame = raw.loc[:, sorted(required)].copy(deep=True)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="raise").dt.date
    numeric_columns = ["open", "high", "low", "close", "vol", "amount", "adj_factor"]
    for column in numeric_columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if not np.isfinite(frame.loc[:, numeric_columns].to_numpy(dtype=float)).all():
        raise MarketDataQualityError("market source contains non-finite numeric values")

    key_columns = ["trade_date", "ts_code"]
    if frame[key_columns].isna().any().any():
        raise MarketDataQualityError("market business keys cannot be null")
    duplicate_rows = frame.duplicated(key_columns, keep=False)
    unique = frame.loc[~duplicate_rows].copy()
    unique["dedup_resolution"] = "UNIQUE"
    duplicated = frame.loc[duplicate_rows]
    collapsed = duplicated.iloc[0:0].copy()
    if not duplicated.empty:
        for key, group in duplicated.groupby(key_columns, sort=True):
            if not group.eq(group.iloc[0]).all(axis=None):
                raise MarketDataQualityError(f"conflicting market duplicate for key {key}")
        collapsed = duplicated.drop_duplicates(key_columns, keep="first").copy()
        collapsed["dedup_resolution"] = "IDENTICAL_COLLAPSE"
    clean = pd.concat([unique, collapsed], ignore_index=True).sort_values(
        key_columns,
        ignore_index=True,
    )
    if (clean[["open", "high", "low", "close"]] <= 0).any().any():
        raise MarketDataQualityError("OHLC values must be positive")
    if (clean["high"] < clean[["open", "low", "close"]].max(axis=1)).any() or (
        clean["low"] > clean[["open", "high", "close"]].min(axis=1)
    ).any():
        raise MarketDataQualityError("OHLC relationship is invalid")
    if (clean[["vol", "amount"]] < 0).any().any() or (clean["adj_factor"] <= 0).any():
        raise MarketDataQualityError("volume, amount, and adj_factor are invalid")

    clean["adjusted_open"] = clean["open"] * clean["adj_factor"]
    return clean


def add_market_history_features(clean: pd.DataFrame) -> pd.DataFrame:
    """Add causal per-security liquidity and valid-trading-history features."""
    required = {"ts_code", "trade_date", "vol", "amount"}
    missing = sorted(required.difference(clean.columns))
    if missing:
        raise MarketDataQualityError(f"market history is missing required columns: {missing}")
    enriched = clean.copy(deep=True)
    enriched["trade_date"] = pd.to_datetime(enriched["trade_date"], errors="raise")
    enriched = enriched.sort_values(["ts_code", "trade_date"], ignore_index=True)
    enriched["valid_trade_day"] = (enriched["vol"] > 0) & (enriched["amount"] > 0)
    grouped = enriched.groupby("ts_code", sort=False)
    enriched["valid_days_20"] = grouped["valid_trade_day"].transform(
        lambda values: values.rolling(20, min_periods=1).sum()
    ).astype("int64")
    valid_amount = enriched["amount"].where(enriched["valid_trade_day"])
    enriched["adv20"] = valid_amount.groupby(enriched["ts_code"], sort=False).transform(
        lambda values: values.rolling(20, min_periods=15).mean()
    )
    return enriched
