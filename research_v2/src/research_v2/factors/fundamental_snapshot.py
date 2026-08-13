"""Materialize announcement-safe fundamental values for a supplied universe."""

from __future__ import annotations

import pandas as pd

from research_v2.factors.availability import expand_fundamental_events
from research_v2.pit.calendar import TradingCalendar

_EVENT_COLUMNS = {
    "metric_id",
    "ts_code",
    "report_period",
    "ann_date",
    "source_revision_id",
    "raw_value",
}
_UNIVERSE_COLUMNS = {"signal_date", "ts_code"}


def build_fundamental_snapshot(
    events: pd.DataFrame,
    universe: pd.DataFrame,
    calendar: TradingCalendar,
    *,
    max_staleness_calendar_days: int = 180,
) -> pd.DataFrame:
    """Expand each metric's announcement events over a deduplicated universe.

    The input remains event-grained.  This function selects the latest
    announcement-safe value for each metric/security/signal date without
    changing any source table or calculating derived financial expressions.
    """
    if missing := sorted(_EVENT_COLUMNS.difference(events.columns)):
        raise ValueError(f"fundamental events are missing columns: {missing}")
    if missing := sorted(_UNIVERSE_COLUMNS.difference(universe.columns)):
        raise ValueError(f"fundamental universe is missing columns: {missing}")
    event_frame = events.loc[:, sorted(_EVENT_COLUMNS)].copy()
    universe_frame = universe.loc[:, ["signal_date", "ts_code"]].copy()
    universe_frame["signal_date"] = pd.to_datetime(
        universe_frame["signal_date"], errors="raise"
    ).dt.date
    universe_frame["ts_code"] = universe_frame["ts_code"].astype(str)
    if universe_frame.duplicated(["signal_date", "ts_code"], keep=False).any():
        raise ValueError("fundamental universe has duplicate signal_date/ts_code keys")

    results: list[pd.DataFrame] = []
    for metric_id, metric_events in event_frame.groupby("metric_id", sort=True):
        expanded = expand_fundamental_events(
            metric_events.drop(columns="metric_id"),
            universe_frame,
            calendar,
            max_staleness_calendar_days=max_staleness_calendar_days,
        )
        if expanded.empty:
            continue
        expanded.insert(2, "metric_id", str(metric_id))
        results.append(
            expanded.rename(columns={"raw_value": "value"}).loc[
                :,
                [
                    "signal_date",
                    "ts_code",
                    "metric_id",
                    "value",
                    "report_period",
                    "ann_date",
                    "available_date",
                    "source_revision_id",
                    "stale_days",
                ],
            ]
        )
    if not results:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "ts_code",
                "metric_id",
                "value",
                "report_period",
                "ann_date",
                "available_date",
                "source_revision_id",
                "stale_days",
            ]
        )
    result = pd.concat(results, ignore_index=True).sort_values(
        ["signal_date", "ts_code", "metric_id"], kind="mergesort"
    )
    if result.duplicated(["signal_date", "ts_code", "metric_id"], keep=False).any():
        raise ValueError("fundamental snapshot has duplicate business keys")
    return result.reset_index(drop=True)
