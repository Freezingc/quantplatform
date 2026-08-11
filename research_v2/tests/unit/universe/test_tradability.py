from __future__ import annotations

from datetime import date
from time import perf_counter

import pandas as pd

from research_v2.pit.calendar import TradingCalendar
from research_v2.universe.tradability import build_tradability_daily


def test_tradability_uses_only_signal_date_security_and_suspension_state() -> None:
    calendar = TradingCalendar.from_dates(
        [date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6)]
    )
    market = pd.DataFrame(
        [
            {
                "trade_date": date(2025, 1, 3),
                "ts_code": "000001.SZ",
                "valid_days_20": 15,
                "adv20": 100.0,
            },
            {
                "trade_date": date(2025, 1, 3),
                "ts_code": "000002.SZ",
                "valid_days_20": 15,
                "adv20": 100.0,
            },
            {
                "trade_date": date(2025, 1, 3),
                "ts_code": "000003.SZ",
                "valid_days_20": 14,
                "adv20": 100.0,
            },
            {
                "trade_date": date(2025, 1, 3),
                "ts_code": "000004.SZ",
                "valid_days_20": 15,
                "adv20": 100.0,
            },
        ]
    )
    security_scd = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "list_date": date(2020, 1, 2),
                "delist_date": pd.NaT,
                "is_st": False,
                "valid_from": date(2020, 1, 2),
                "valid_to": pd.NaT,
            },
            {
                "ts_code": "000002.SZ",
                "list_date": date(2020, 1, 2),
                "delist_date": pd.NaT,
                "is_st": True,
                "valid_from": date(2020, 1, 2),
                "valid_to": pd.NaT,
            },
            {
                "ts_code": "000003.SZ",
                "list_date": date(2025, 1, 2),
                "delist_date": pd.NaT,
                "is_st": False,
                "valid_from": date(2025, 1, 2),
                "valid_to": pd.NaT,
            },
            {
                "ts_code": "000004.SZ",
                "list_date": date(2020, 1, 2),
                "delist_date": date(2025, 1, 2),
                "is_st": False,
                "valid_from": date(2020, 1, 2),
                "valid_to": date(2025, 1, 3),
            },
        ]
    )
    suspensions = pd.DataFrame(
        [
            {"trade_date": date(2025, 1, 3), "ts_code": "000001.SZ", "suspend_type": "S"},
        ]
    )

    result = build_tradability_daily(
        market,
        security_scd,
        suspensions,
        calendar,
        minimum_listing_trading_days=2,
        minimum_valid_days_20=15,
    ).set_index("ts_code")

    assert result.loc["000001.SZ", "is_suspended"]
    assert result.loc["000002.SZ", "is_st"]
    assert result.loc["000003.SZ", "listing_age_trading_days"] == 2
    assert not result.loc["000004.SZ", "is_listed"]
    assert not result.loc["000001.SZ", "is_eligible_pre_liquidity"]
    assert result.loc["000003.SZ", "reason_bitmap"] == 16


def test_tradability_handles_a_large_cross_section_without_repeated_scd_scans() -> None:
    row_count = 5_000
    trade_date = date(2025, 1, 3)
    codes = [f"{index:06d}.SZ" for index in range(row_count)]
    market = pd.DataFrame(
        {
            "trade_date": [trade_date] * row_count,
            "ts_code": codes,
            "valid_days_20": [20] * row_count,
            "adv20": [100.0] * row_count,
        }
    )
    security_scd = pd.DataFrame(
        {
            "ts_code": codes,
            "list_date": [date(2020, 1, 2)] * row_count,
            "delist_date": [pd.NaT] * row_count,
            "is_st": [False] * row_count,
            "valid_from": [date(2020, 1, 2)] * row_count,
            "valid_to": [pd.NaT] * row_count,
        }
    )
    started_at = perf_counter()

    result = build_tradability_daily(
        market,
        security_scd,
        pd.DataFrame(columns=["trade_date", "ts_code", "suspend_type"]),
        TradingCalendar.from_dates([trade_date]),
        minimum_listing_trading_days=1,
        minimum_valid_days_20=1,
    )

    assert len(result) == row_count
    assert perf_counter() - started_at < 1.0


def test_tradability_normalizes_dolphindb_timestamp_trade_dates() -> None:
    trade_date = date(2025, 1, 3)

    result = build_tradability_daily(
        pd.DataFrame(
            {
                "trade_date": [pd.Timestamp(trade_date)],
                "ts_code": ["000001.SZ"],
                "valid_days_20": [20],
                "adv20": [100.0],
            }
        ),
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "list_date": [date(2020, 1, 2)],
                "delist_date": [pd.NaT],
                "is_st": [False],
                "valid_from": [date(2020, 1, 2)],
                "valid_to": [pd.NaT],
            }
        ),
        pd.DataFrame(columns=["trade_date", "ts_code", "suspend_type"]),
        TradingCalendar.from_dates([trade_date]),
        minimum_listing_trading_days=1,
        minimum_valid_days_20=1,
    )

    assert result.loc[0, "trade_date"] == trade_date
