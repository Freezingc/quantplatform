from __future__ import annotations

from time import perf_counter

import pandas as pd
import pytest

from research_v2.pit.market import (
    MarketDataQualityError,
    add_market_history_features,
    clean_market_daily,
)


def test_market_cleaner_collapses_identical_rows_and_calculates_adjusted_open() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["2025-01-02", "2025-01-02"],
            "open": [10.0, 10.0],
            "high": [10.5, 10.5],
            "low": [9.8, 9.8],
            "close": [10.2, 10.2],
            "vol": [1000.0, 1000.0],
            "amount": [10000.0, 10000.0],
            "adj_factor": [1.25, 1.25],
        }
    )

    clean = clean_market_daily(raw)

    assert len(clean) == 1
    assert clean.loc[0, "adjusted_open"] == pytest.approx(12.5)
    assert clean.loc[0, "dedup_resolution"] == "IDENTICAL_COLLAPSE"


def test_market_cleaner_rejects_conflicting_duplicate_values() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["2025-01-02", "2025-01-02"],
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.8, 10.8],
            "close": [10.2, 11.2],
            "vol": [1000.0, 1000.0],
            "amount": [10000.0, 10000.0],
            "adj_factor": [1.25, 1.25],
        }
    )

    with pytest.raises(MarketDataQualityError, match="conflicting"):
        clean_market_daily(raw)


def test_market_cleaner_rejects_invalid_ohlc_relationship() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["2025-01-02"],
            "open": [10.0],
            "high": [9.9],
            "low": [9.8],
            "close": [10.2],
            "vol": [1000.0],
            "amount": [10000.0],
            "adj_factor": [1.25],
        }
    )

    with pytest.raises(MarketDataQualityError, match="OHLC"):
        clean_market_daily(raw)


def test_history_features_are_grouped_by_security_and_use_no_future_rows() -> None:
    dates = pd.bdate_range("2025-01-01", periods=20)
    market = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 20 + ["000002.SZ"] * 20,
            "trade_date": list(dates) * 2,
            "vol": [100.0] * 20 + [0.0] * 20,
            "amount": [1000.0] * 20 + [0.0] * 20,
        }
    )

    enriched = add_market_history_features(market)

    first = enriched[enriched["ts_code"] == "000001.SZ"].iloc[-1]
    second = enriched[enriched["ts_code"] == "000002.SZ"].iloc[-1]
    assert first["valid_days_20"] == 20
    assert first["adv20"] == pytest.approx(1000.0)
    assert second["valid_days_20"] == 0
    assert pd.isna(second["adv20"])


def test_market_cleaner_handles_a_large_unique_cross_section_without_group_scan() -> None:
    row_count = 10_000
    raw = pd.DataFrame(
        {
            "ts_code": [f"{index:06d}.SZ" for index in range(row_count)],
            "trade_date": ["2025-01-02"] * row_count,
            "open": [10.0] * row_count,
            "high": [10.5] * row_count,
            "low": [9.8] * row_count,
            "close": [10.2] * row_count,
            "vol": [1000.0] * row_count,
            "amount": [10000.0] * row_count,
            "adj_factor": [1.25] * row_count,
        }
    )

    started_at = perf_counter()
    clean = clean_market_daily(raw)

    assert len(clean) == row_count
    assert perf_counter() - started_at < 1.0
