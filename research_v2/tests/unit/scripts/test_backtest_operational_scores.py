"""Unit tests for the operational backtest's pure helpers.

Two things here are easy to get wrong in the same direction. The forward return has to use the
platform's T+1/T+6 open convention rather than anything derived from the signal day, or the whole
backtest leaks. And a signal whose exit price does not exist yet must be dropped rather than
scored, or the window reports a return for a trade that could not have been closed.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

import backtest_operational_scores as backtest
from research_v2.pit.calendar import TradingCalendar


def _calendar(count: int) -> TradingCalendar:
    return TradingCalendar.from_dates(
        [date(2026, 8, 3) + pd.Timedelta(days=index) for index in range(count)]
    )


def test_forward_returns_use_the_next_open_to_the_sixth_open() -> None:
    calendar = _calendar(8)
    days = calendar.dates
    market = pd.DataFrame(
        {
            "trade_date": list(days),
            "ts_code": ["a"] * 8,
            "adjusted_open": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 22.0, 30.0],
        }
    )
    returns = backtest.forward_returns(market, calendar, (days[0],))
    # entry is days[1] (open 11), exit is days[6] (open 22).
    assert returns[days[0]]["a"] == pytest.approx(22.0 / 11.0 - 1.0)


def test_forward_returns_skips_a_name_with_no_entry_or_exit_price() -> None:
    calendar = _calendar(8)
    days = calendar.dates
    market = pd.DataFrame(
        {
            "trade_date": [days[1], days[6], days[1]],
            "ts_code": ["a", "a", "b"],
            "adjusted_open": [11.0, 22.0, 5.0],
        }
    )
    returns = backtest.forward_returns(market, calendar, (days[0],))
    assert set(returns[days[0]]) == {"a"}


def test_measurable_signal_dates_drop_signals_whose_exit_has_not_priced() -> None:
    calendar = _calendar(10)
    days = calendar.dates
    measurable = backtest.measurable_signal_dates(
        calendar, tuple(days), priced_through=days[7]
    )
    # exit is shift(T, 6), so only T <= days[1] closes on or before days[7].
    assert measurable == (days[0], days[1])


def test_adjusted_open_is_the_raw_open_times_the_factor() -> None:
    market = pd.DataFrame({"open": [10.0, 5.0], "adj_factor": [1.5, 2.0]})
    adjusted = backtest.with_adjusted_open(market)
    assert adjusted["adjusted_open"].tolist() == [15.0, 10.0]


def test_parse_policies_reads_enter_exit_pairs() -> None:
    policies = backtest.parse_policies("10:50,30:150")
    assert [(policy.n_enter, policy.n_exit) for policy in policies] == [(10, 50), (30, 150)]


def test_parse_policies_refuses_a_narrower_buffer() -> None:
    with pytest.raises(ValueError):
        backtest.parse_policies("30:10")


def test_apply_trend_floor_overrides_every_policy() -> None:
    """The overlay can park the book in cash while the benchmark stays invested, so isolating
    stock selection from the overlay means being able to pin the floor at 1.0."""
    policies = backtest.apply_trend_floor(backtest.parse_policies("10:50,30:150"), floor=1.0)
    assert [policy.trend_exposure_floor for policy in policies] == [1.0, 1.0]
    assert [policy.n_enter for policy in policies] == [10, 30]
