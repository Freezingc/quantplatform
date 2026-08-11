"""Point-in-time factor availability and bounded fundamental-event expansion."""

from __future__ import annotations

from datetime import date
from typing import Any, cast

import pandas as pd

from research_v2.pit.calendar import TradingCalendar, TradingCalendarBoundaryError


def _as_date(value: Any) -> date:
    if isinstance(value, pd.Timestamp):
        return value.date()
    if isinstance(value, date):
        return value
    return pd.Timestamp(value).date()


def assign_fundamental_available_date(ann_date: date, calendar: TradingCalendar) -> date:
    """Make an announcement usable only from the next open-market day."""
    if ann_date in calendar.dates:
        return calendar.shift(ann_date, 1)
    future_days = calendar.window(ann_date, calendar.dates[-1])
    if not future_days:
        raise TradingCalendarBoundaryError("announcement date exceeds calendar coverage")
    return future_days[0]


def materialize_technical_availability(observations: pd.DataFrame) -> pd.DataFrame:
    """Technical observations remain available only on their own close-to-close date."""
    required = {"ts_code", "observation_date", "raw_value"}
    missing = sorted(required.difference(observations.columns))
    if missing:
        raise ValueError(f"technical observations are missing columns: {missing}")
    result = observations.copy(deep=True)
    result["observation_date"] = result["observation_date"].map(_as_date)
    result["available_date"] = result["observation_date"]
    result["signal_date"] = result["observation_date"]
    result["stale_days"] = 0
    return result


def expand_fundamental_events(
    events: pd.DataFrame,
    universe: pd.DataFrame,
    calendar: TradingCalendar,
    *,
    max_staleness_calendar_days: int = 180,
) -> pd.DataFrame:
    """Join event versions to actual universe dates without all-security Cartesian expansion."""
    required_events = {"ts_code", "report_period", "ann_date", "raw_value", "source_revision_id"}
    required_universe = {"signal_date", "ts_code"}
    if missing := sorted(required_events.difference(events.columns)):
        raise ValueError(f"fundamental events are missing columns: {missing}")
    if missing := sorted(required_universe.difference(universe.columns)):
        raise ValueError(f"universe is missing columns: {missing}")
    if max_staleness_calendar_days < 0:
        raise ValueError("max_staleness_calendar_days cannot be negative")
    normalized_events = events.copy(deep=True)
    normalized_events["ann_date"] = normalized_events["ann_date"].map(_as_date)
    normalized_events["report_period"] = normalized_events["report_period"].map(_as_date)
    normalized_events["available_date"] = normalized_events["ann_date"].map(
        lambda value: assign_fundamental_available_date(value, calendar)
    )
    rows: list[dict[str, object]] = []
    for universe_row in cast(list[dict[str, Any]], universe.to_dict(orient="records")):
        signal_date = _as_date(universe_row["signal_date"])
        ts_code = str(universe_row["ts_code"])
        candidates = normalized_events.loc[
            (normalized_events["ts_code"].astype(str) == ts_code)
            & (normalized_events["available_date"] <= signal_date)
        ]
        if candidates.empty:
            continue
        latest = candidates.sort_values(
            ["available_date", "source_revision_id"],
            ascending=[False, False],
            kind="mergesort",
        ).iloc[0]
        available_date = _as_date(latest["available_date"])
        stale_days = (signal_date - available_date).days
        if stale_days > max_staleness_calendar_days:
            continue
        rows.append(
            {
                "signal_date": signal_date,
                "ts_code": ts_code,
                "raw_value": latest["raw_value"],
                "report_period": _as_date(latest["report_period"]),
                "ann_date": _as_date(latest["ann_date"]),
                "available_date": available_date,
                "source_revision_id": latest["source_revision_id"],
                "stale_days": stale_days,
            }
        )
    return pd.DataFrame(rows).sort_values(["signal_date", "ts_code"], ignore_index=True)
