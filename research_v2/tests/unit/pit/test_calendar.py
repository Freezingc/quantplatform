from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.pit.calendar import TradingCalendar, TradingCalendarBoundaryError


@pytest.fixture
def calendar() -> TradingCalendar:
    return TradingCalendar.from_dates(
        [
            date(2025, 1, 2),
            date(2025, 1, 3),
            date(2025, 1, 6),
            date(2025, 1, 7),
            date(2025, 1, 8),
            date(2025, 1, 9),
            date(2025, 1, 10),
            date(2025, 1, 13),
        ]
    )


def test_shift_uses_trading_days_across_weekends(calendar: TradingCalendar) -> None:
    assert calendar.shift(date(2025, 1, 3), 1) == date(2025, 1, 6)
    assert calendar.shift(date(2025, 1, 3), 6) == date(2025, 1, 13)


def test_window_and_count_are_inclusive(calendar: TradingCalendar) -> None:
    assert calendar.window(date(2025, 1, 3), date(2025, 1, 8)) == (
        date(2025, 1, 3),
        date(2025, 1, 6),
        date(2025, 1, 7),
        date(2025, 1, 8),
    )
    assert calendar.count(date(2025, 1, 3), date(2025, 1, 8)) == 4


def test_shift_rejects_dates_outside_the_calendar(calendar: TradingCalendar) -> None:
    with pytest.raises(TradingCalendarBoundaryError):
        calendar.shift(date(2025, 1, 13), 1)


def test_calendar_normalizes_timestamp_dates_at_the_source_boundary() -> None:
    calendar = TradingCalendar.from_dates(
        [pd.Timestamp("2025-01-02"), pd.Timestamp("2025-01-03")]
    )

    assert calendar.dates == (date(2025, 1, 2), date(2025, 1, 3))
    assert calendar.window(date(2025, 1, 2), date(2025, 1, 3)) == calendar.dates
