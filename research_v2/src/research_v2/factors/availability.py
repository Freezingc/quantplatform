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


EXPANSION_COLUMNS: tuple[str, ...] = (
    "signal_date",
    "ts_code",
    "raw_value",
    "report_period",
    "ann_date",
    "available_date",
    "source_revision_id",
    "stale_days",
)


def _empty_expansion() -> pd.DataFrame:
    """An expansion in which nothing qualified, with the same columns as a non-empty one.

    Building the frame from an empty row list yields a frame with *no* columns, so the
    trailing ``sort_values`` raised ``KeyError: 'signal_date'`` -- naming a column and so
    pointing nowhere near the actual cause, which is simply that no version was available
    within the staleness bound. That is an ordinary outcome, not an error: it happens for an
    early fold preceding the first announcement, for a newly listed security, and whenever a
    bound is tight enough that every version has expired. Callers concatenate expansions
    across factors and folds, so one empty result must not abort the whole materialization.
    """
    dtypes: dict[str, str] = {
        "signal_date": "object",
        "ts_code": "str",
        "raw_value": "float64",
        "report_period": "object",
        "ann_date": "object",
        "available_date": "object",
        "source_revision_id": "str",
        "stale_days": "int64",
    }
    return pd.DataFrame(
        {name: pd.Series(dtype=dtypes[name]) for name in EXPANSION_COLUMNS}
    )


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
        # ``ann_date`` is a fourth, last key and it is there to make the sort total, not to
        # rank anything. ``available_date`` is derived from ``ann_date`` through the calendar
        # and that map is many-to-one -- a Friday, its Saturday and its Sunday all roll to the
        # same next trading day -- so two events announced on different calendar days can tie
        # on ``available_date`` while carrying different ``ann_date``s. Tied on all three of
        # the original keys, a stable sort then resolved them by input row order, and since
        # ``ann_date`` is published as provenance the *recorded announcement date* depended on
        # the order the extract happened to arrive in even when every value agreed. Hypothesis
        # found that shape on 2026-09-06.
        #
        # This does not pre-empt decision B, which owns the precedence *among* the keys above
        # (whether a late restatement of an older period should supersede a newer period).
        # Appending a key that only fires once those three are already equal cannot reorder
        # them, and it is compatible with every ordering B could choose. It also leaves the
        # pinned value-conflict defect untouched: two events sharing one ``ann_date`` and
        # disagreeing on ``raw_value`` still tie on all four keys and are still resolved by
        # input order, which is what
        # ``test_two_events_tied_on_every_sort_key_are_resolved_by_input_order`` asserts.
        latest = candidates.sort_values(
            ["available_date", "report_period", "source_revision_id", "ann_date"],
            ascending=[False, False, False, False],
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
    if not rows:
        return _empty_expansion()
    return pd.DataFrame(rows).sort_values(["signal_date", "ts_code"], ignore_index=True)
