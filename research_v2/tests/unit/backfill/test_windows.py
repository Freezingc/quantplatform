from __future__ import annotations

from datetime import date

from research_v2.backfill.windows import resolve_monthly_backfill_windows
from research_v2.pit.calendar import TradingCalendar


def test_monthly_backfill_windows_use_actual_trading_month_boundaries_and_lookback() -> None:
    calendar = TradingCalendar.from_dates(
        [
            date(2019, 12, 27),
            date(2019, 12, 30),
            date(2019, 12, 31),
            date(2020, 1, 2),
            date(2020, 1, 3),
            date(2020, 1, 23),
            date(2020, 2, 3),
            date(2020, 2, 28),
        ]
    )

    windows = resolve_monthly_backfill_windows(
        calendar,
        start_date=date(2020, 1, 1),
        end_date=date(2020, 2, 29),
        lookback_trading_days=2,
    )

    assert [(item.month, item.target_start, item.target_end) for item in windows] == [
        ("2020-01", date(2020, 1, 2), date(2020, 1, 23)),
        ("2020-02", date(2020, 2, 3), date(2020, 2, 28)),
    ]
    assert windows[0].lookback_start == date(2019, 12, 30)
    assert windows[1].lookback_start == date(2020, 1, 3)
