"""Resumable monthly backfill planning and execution primitives."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import date
from typing import Literal

from research_v2.backfill.windows import MonthlyBackfillWindow, resolve_monthly_backfill_windows
from research_v2.pit.calendar import TradingCalendar

PointerReader = Callable[[str, str], str | None]
PublishCallback = Callable[[MonthlyBackfillWindow, tuple[str, ...]], Iterable[str]]


@dataclass(frozen=True)
class MonthlyBackfillPlan:
    """One monthly unit with the objects still requiring publication."""

    window: MonthlyBackfillWindow
    pending_objects: tuple[str, ...]

    @property
    def status(self) -> Literal["complete", "pending"]:
        """Return whether all required objects already have visible pointers."""
        return "complete" if not self.pending_objects else "pending"


@dataclass(frozen=True)
class MonthlyBackfillResult:
    """Outcome of one planned month."""

    month: str
    status: Literal["complete", "published"]
    published_objects: tuple[str, ...]


class MonthlyBackfillRunner:
    """Plan and execute monthly units while remaining safe to resume.

    The pointer reader is deliberately injected so this runner can be used with
    the V2 ``published_version`` table without coupling planning to a database
    client. A complete month is never sent to the publish callback.
    """

    def __init__(
        self,
        calendar: TradingCalendar,
        pointer_reader: PointerReader,
        *,
        required_objects: tuple[str, ...],
    ) -> None:
        if not required_objects or len(set(required_objects)) != len(required_objects):
            raise ValueError("required_objects must contain unique object names")
        self._calendar = calendar
        self._pointer_reader = pointer_reader
        self._required_objects = required_objects

    def plan(
        self,
        start_date: date,
        end_date: date,
        *,
        lookback_trading_days: int = 20,
    ) -> list[MonthlyBackfillPlan]:
        """Return monthly work units with only missing objects left pending."""
        windows = resolve_monthly_backfill_windows(
            self._calendar,
            start_date=start_date,
            end_date=end_date,
            lookback_trading_days=lookback_trading_days,
        )
        plans: list[MonthlyBackfillPlan] = []
        for window in windows:
            pending = tuple(
                object_name
                for object_name in self._required_objects
                if not self._pointer_reader(object_name, window.month)
            )
            plans.append(MonthlyBackfillPlan(window=window, pending_objects=pending))
        return plans

    def run(
        self,
        start_date: date,
        end_date: date,
        *,
        publish: PublishCallback,
        lookback_trading_days: int = 20,
    ) -> list[MonthlyBackfillResult]:
        """Publish only pending objects and return one result per month.

        The callback must report object names it actually published. Reporting
        an object outside the pending set is rejected to prevent accidental
        cross-month or duplicate writes.
        """
        results: list[MonthlyBackfillResult] = []
        for plan in self.plan(
            start_date,
            end_date,
            lookback_trading_days=lookback_trading_days,
        ):
            if not plan.pending_objects:
                results.append(
                    MonthlyBackfillResult(
                        month=plan.window.month,
                        status="complete",
                        published_objects=(),
                    )
                )
                continue
            published = tuple(publish(plan.window, plan.pending_objects))
            if not published or len(set(published)) != len(published):
                raise ValueError("publish callback must report unique object names")
            if any(object_name not in plan.pending_objects for object_name in published):
                raise ValueError("publish callback reported an object that was not pending")
            results.append(
                MonthlyBackfillResult(
                    month=plan.window.month,
                    status="published",
                    published_objects=published,
                )
            )
        return results
