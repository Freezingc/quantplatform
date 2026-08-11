"""Immutable trading-calendar operations."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from datetime import date, datetime


class TradingCalendarBoundaryError(ValueError):
    """Raised when a requested trading-day operation leaves the calendar."""


@dataclass(frozen=True)
class TradingCalendar:
    """A sorted set of open-market dates."""

    dates: tuple[date, ...]

    @classmethod
    def from_dates(cls, dates: list[date]) -> TradingCalendar:
        normalized = tuple(
            sorted(
                {
                    value.date() if isinstance(value, datetime) else value
                    for value in dates
                }
            )
        )
        if not normalized:
            raise ValueError("trading calendar cannot be empty")
        return cls(dates=normalized)

    def shift(self, current: date, offset: int) -> date:
        try:
            index = self.dates.index(current)
        except ValueError as error:
            raise TradingCalendarBoundaryError(f"not a trading day: {current}") from error
        shifted_index = index + offset
        if shifted_index < 0 or shifted_index >= len(self.dates):
            raise TradingCalendarBoundaryError("trading-day shift leaves calendar coverage")
        return self.dates[shifted_index]

    def window(self, start: date, end: date) -> tuple[date, ...]:
        if start > end:
            raise ValueError("start cannot be after end")
        return self.dates[bisect_left(self.dates, start) : bisect_right(self.dates, end)]

    def count(self, start: date, end: date) -> int:
        return len(self.window(start, end))
