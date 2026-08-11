"""Trading-calendar-based work units for resumable historical backfills."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date

from research_v2.pit.calendar import TradingCalendar, TradingCalendarBoundaryError


@dataclass(frozen=True)
class MonthlyBackfillWindow:
    """One publishable calendar month and the predecessor data needed for features."""

    month: str
    lookback_start: date
    target_start: date
    target_end: date


def resolve_monthly_backfill_windows(
    calendar: TradingCalendar,
    *,
    start_date: date,
    end_date: date,
    lookback_trading_days: int = 20,
) -> list[MonthlyBackfillWindow]:
    """Partition a closed date range by real trading months, not calendar-month ends."""
    if start_date > end_date:
        raise ValueError("start_date cannot be after end_date")
    if lookback_trading_days < 1:
        raise ValueError("lookback_trading_days must be positive")
    by_month: dict[str, list[date]] = {}
    for trade_date in calendar.window(start_date, end_date):
        by_month.setdefault(trade_date.strftime("%Y-%m"), []).append(trade_date)
    if not by_month:
        raise ValueError("requested range contains no trading days")
    windows: list[MonthlyBackfillWindow] = []
    for month, target_dates in by_month.items():
        target_start = target_dates[0]
        try:
            lookback_start = calendar.shift(target_start, -lookback_trading_days)
        except TradingCalendarBoundaryError as error:
            raise ValueError(
                f"insufficient trading-calendar history for {month} lookback"
            ) from error
        windows.append(
            MonthlyBackfillWindow(
                month=month,
                lookback_start=lookback_start,
                target_start=target_start,
                target_end=target_dates[-1],
            )
        )
    return windows
