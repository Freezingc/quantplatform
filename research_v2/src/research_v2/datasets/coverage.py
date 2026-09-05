"""Per-day cross-section width of an assembled point-in-time frame.

The walk-forward runner drops every row lacking any factor
(``dropna(subset=factor_order)``) because B2/B3 raise on a non-finite feature. That
delete is silent: a day can lose most of its cross-section, or all of it, and nothing
downstream reports which. Measured 2026-09-05/06, the frozen 17 lose 15 train days in
F01 and F02 and 4 in F03 -- including 2015-08-24, where 2 of 1,466 liquid-universe
stocks had all 17 factors -- while the accepted legacy 14 lose exactly one day,
2017-10-09, and lose it entirely, to a single absent ``(factor, date)`` cell at source.
G1-G6 cannot see any of it, because they are computed only from TEST-period evidence.

This module owns the measurement so the runner, the pre-flight coverage gate and the
calibration audit all read the same quantity the same way rather than three copies that
can drift. Widths are counted on the **assembled** frame, i.e. after the universe and
label joins and the ``dropna`` the model actually trains through, which is strictly more
authoritative than counting ``factor_daily`` rows: the two disagree (F01 train spans
1,214 assembled days against 1,220 factor-daily days).

A day whose every row is dropped is reported with width 0 rather than omitted. Grouping
the post-dropna frame alone would erase it from the calendar, which is how 2015-08-24
and the two 2016 circuit-breaker days originally disappeared without any gate noticing.
"""

from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any, cast

import pandas as pd

# The same floor the evaluation path uses (``evaluate_frozen_oos(minimum_cross_section=30)``),
# so a day counted as lost here is a day the model genuinely could not be fitted or scored on.
DEFAULT_MINIMUM_CROSS_SECTION = 30


@dataclass(frozen=True)
class WidthSummary:
    """What one assembled segment's cross-section looked like, day by day."""

    rows: int
    complete_rows: int
    days: int
    days_below_floor: int
    empty_days: int
    median_width: int
    min_width: int

    @property
    def survival_ratio(self) -> float:
        """Share of assembled rows that kept every factor, i.e. survived the dropna."""
        return self.complete_rows / self.rows if self.rows else 0.0

    def to_json(self) -> dict[str, Any]:
        payload: dict[str, Any] = asdict(self)
        payload["survival_ratio"] = self.survival_ratio
        return payload


def factor_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """The ``fNN`` feature columns the runner passes to ``dropna(subset=...)``."""
    found = tuple(str(column) for column in columns if _is_feature_name(str(column)))
    if not found:
        raise ValueError("frame has no fNN feature columns; not an assembled walk-forward frame")
    return found


def daily_complete_widths(
    frame: pd.DataFrame, features: Sequence[str]
) -> dict[Any, int]:
    """Per-day surviving row count, applying exactly the runner's ``dropna``."""
    if not len(frame):
        return {}
    calendar = [pd.Timestamp(day) for day in frame["signal_date"].unique()]
    counts = frame.dropna(subset=list(features)).groupby("signal_date").size()
    # ``Series.items`` types its key as ``Hashable``; these are the frame's own
    # signal_date values, already accepted by ``pd.Timestamp`` on the line above.
    lookup = {pd.Timestamp(cast(Any, day)): int(value) for day, value in counts.items()}
    return {day: lookup.get(day, 0) for day in calendar}


def _is_feature_name(column: str) -> bool:
    return len(column) == 3 and column.startswith("f") and column[1:].isdigit()


def summarize_frame_coverage(
    frame: pd.DataFrame,
    columns: Sequence[str] | None = None,
    *,
    minimum_cross_section: int = DEFAULT_MINIMUM_CROSS_SECTION,
) -> WidthSummary:
    """Measure one assembled segment, deriving the feature columns when not given."""
    features = tuple(columns) if columns is not None else factor_columns(frame.columns.tolist())
    widths = daily_complete_widths(frame, features)
    return summarize_widths(
        widths, rows=int(len(frame)), minimum_cross_section=minimum_cross_section
    )


def summarize_widths(
    widths: Mapping[Any, int],
    *,
    rows: int,
    minimum_cross_section: int = DEFAULT_MINIMUM_CROSS_SECTION,
) -> WidthSummary:
    """Summarize an already-computed width histogram.

    Split out from :func:`summarize_frame_coverage` so a caller that needs both the
    per-day widths and their summary -- the calibration audit feeds the widths to the
    coverage gate's verdict function -- pays for the ``dropna`` once instead of twice
    on frames of up to 3.7 million rows.
    """
    if minimum_cross_section < 2:
        raise ValueError("minimum_cross_section must be at least 2")
    values = sorted(int(width) for width in widths.values())
    return WidthSummary(
        rows=rows,
        complete_rows=int(sum(values)),
        days=len(values),
        days_below_floor=sum(1 for width in values if width < minimum_cross_section),
        empty_days=sum(1 for width in values if width == 0),
        median_width=int(statistics.median(values)) if values else 0,
        min_width=values[0] if values else 0,
    )
