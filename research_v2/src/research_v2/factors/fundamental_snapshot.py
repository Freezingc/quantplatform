"""Materialize announcement-safe fundamental values for a supplied universe."""

from __future__ import annotations

import pandas as pd

from research_v2.factors.availability import assign_fundamental_available_date
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
        expanded = _expand_metric_events(
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


def _expand_metric_events(
    events: pd.DataFrame,
    universe: pd.DataFrame,
    calendar: TradingCalendar,
    *,
    max_staleness_calendar_days: int,
) -> pd.DataFrame:
    """Vectorized per-security equivalent of fundamental event expansion."""
    source = events.copy()
    source["ann_date"] = pd.to_datetime(source["ann_date"], errors="raise")
    source["report_period"] = pd.to_datetime(source["report_period"], errors="raise")
    source["available_date"] = source["ann_date"].map(
        lambda value: pd.Timestamp(assign_fundamental_available_date(value.date(), calendar))
    )
    source = source.sort_values(
        ["ts_code", "available_date", "report_period", "source_revision_id"],
        kind="mergesort",
    ).drop_duplicates(["ts_code", "available_date"], keep="last")
    targets = universe.copy()
    targets["signal_date"] = pd.to_datetime(targets["signal_date"], errors="raise")
    rows: list[pd.DataFrame] = []
    for ts_code, target in targets.groupby("ts_code", sort=False):
        source_rows = source.loc[source["ts_code"].astype(str) == str(ts_code)]
        if source_rows.empty:
            continue
        merged = pd.merge_asof(
            target.sort_values("signal_date", kind="mergesort"),
            source_rows.sort_values("available_date", kind="mergesort"),
            left_on="signal_date",
            right_on="available_date",
            direction="backward",
            allow_exact_matches=True,
        )
        merged["stale_days"] = (merged["signal_date"] - merged["available_date"]).dt.days
        rows.append(
            merged.loc[
                merged["raw_value"].notna()
                & (merged["stale_days"] <= max_staleness_calendar_days),
                [
                    "signal_date",
                    "ts_code_x",
                    "raw_value",
                    "report_period",
                    "ann_date",
                    "available_date",
                    "source_revision_id",
                    "stale_days",
                ],
            ].rename(columns={"ts_code_x": "ts_code"})
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "signal_date",
                "ts_code",
                "raw_value",
                "report_period",
                "ann_date",
                "available_date",
                "source_revision_id",
                "stale_days",
            ]
        )
    result = pd.concat(rows, ignore_index=True)
    for column in ("signal_date", "report_period", "ann_date", "available_date"):
        result[column] = pd.to_datetime(result[column], errors="raise").dt.date
    return result
