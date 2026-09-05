"""Property invariants for point-in-time factor availability (C plan C4.5 / C4.7).

The unit test pins three hand-built cases: same-day technical availability, the
next-trading-day fundamental rule with the 180-day expiry boundary, and latest-report-period
preference. Those are the right examples but they are examples. What the rest of the
platform relies on is stronger and was unproven -- that for *any* set of announcements and
*any* universe, the expansion never publishes a row that violates the point-in-time ordering,
never invents a (signal_date, ts_code) the universe did not contain, never emits two records
for one key, and does not depend on the order the announcement rows happened to arrive in.

Order independence matters for the same reason it does for deduplication: YesLab extracts
arrive per column, per month and per factor group, so an order dependence would not fail
reproducibly. It would quietly change which revision of a fundamental figure a past signal
date sees.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from hypothesis import given, settings
from hypothesis import strategies as st

from research_v2.factors.availability import (
    assign_fundamental_available_date,
    expand_fundamental_events,
)
from research_v2.pit.calendar import TradingCalendar

CODES = ("000001.SZ", "600000.SH", "600519.SH")
REPORT_PERIODS = (date(2024, 9, 30), date(2024, 12, 31), date(2025, 3, 31), date(2025, 6, 30))
REVISIONS = ("r1", "r2", "r3")
# Announcement dates stay clear of the final calendar day, whose next-trading-day shift
# would legitimately leave the calendar and raise.
ANNOUNCEMENTS = st.dates(min_value=date(2025, 1, 1), max_value=date(2025, 12, 15))


# Built once: the strategies below draw from it on every example, and rebuilding a
# 261-day calendar per draw costs more than the expansion being tested.
CALENDAR = TradingCalendar.from_dates(
    [timestamp.date() for timestamp in pd.bdate_range("2025-01-01", "2025-12-31")]
)


@st.composite
def event_frames(draw, min_rows: int = 1, max_rows: int = 12) -> pd.DataFrame:
    rows = [
        {
            "ts_code": draw(st.sampled_from(CODES)),
            "report_period": draw(st.sampled_from(REPORT_PERIODS)),
            "ann_date": draw(ANNOUNCEMENTS),
            "raw_value": draw(
                st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False)
            ),
            "source_revision_id": draw(st.sampled_from(REVISIONS)),
        }
        for _ in range(draw(st.integers(min_value=min_rows, max_value=max_rows)))
    ]
    return pd.DataFrame(rows)


@st.composite
def universe_frames(draw, min_rows: int = 1, max_rows: int = 20) -> pd.DataFrame:
    days = draw(
        st.lists(
            st.sampled_from(CALENDAR.dates),
            min_size=min_rows,
            max_size=max_rows,
        )
    )
    rows = [
        {"signal_date": day, "ts_code": draw(st.sampled_from(CODES))} for day in days
    ]
    return pd.DataFrame(rows).drop_duplicates(["signal_date", "ts_code"], keep="first")


@given(ann_date=ANNOUNCEMENTS)
@settings(max_examples=80, deadline=None)
def test_available_date_is_the_earliest_trading_day_strictly_after_the_announcement(
    ann_date: date,
) -> None:
    available = assign_fundamental_available_date(ann_date, CALENDAR)

    assert available > ann_date, "a value must never be usable on or before it is announced"
    assert available in CALENDAR.dates, "availability must land on an open-market day"
    earlier = [day for day in CALENDAR.dates if ann_date < day < available]
    assert not earlier, f"{available} is not the earliest trading day after {ann_date}: {earlier}"


@given(events=event_frames(), universe=universe_frames())
@settings(max_examples=40, deadline=None)
def test_every_published_row_satisfies_the_point_in_time_ordering(
    events: pd.DataFrame, universe: pd.DataFrame
) -> None:
    expanded = expand_fundamental_events(events, universe, CALENDAR)

    for row in expanded.to_dict(orient="records"):
        assert row["ann_date"] < row["available_date"] <= row["signal_date"]
        assert row["stale_days"] == (row["signal_date"] - row["available_date"]).days
        assert 0 <= row["stale_days"] <= 180


@given(events=event_frames(), universe=universe_frames())
@settings(max_examples=40, deadline=None)
def test_expansion_never_invents_a_key_or_duplicates_one(
    events: pd.DataFrame, universe: pd.DataFrame
) -> None:
    expanded = expand_fundamental_events(events, universe, CALENDAR)
    if expanded.empty:
        return

    published = set(
        map(tuple, expanded.loc[:, ["signal_date", "ts_code"]].to_numpy().tolist())
    )
    requested = set(map(tuple, universe.loc[:, ["signal_date", "ts_code"]].to_numpy().tolist()))

    assert published <= requested, "expansion published a key the universe never asked for"
    assert not expanded.duplicated(["signal_date", "ts_code"]).any()


@given(events=event_frames(), universe=universe_frames(), seed=st.integers(0, 2**31 - 1))
@settings(max_examples=40, deadline=None)
def test_event_row_order_does_not_change_the_expansion(
    events: pd.DataFrame, universe: pd.DataFrame, seed: int
) -> None:
    baseline = expand_fundamental_events(events, universe, CALENDAR)

    shuffled = events.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    permuted = expand_fundamental_events(shuffled, universe, CALENDAR)

    pd.testing.assert_frame_equal(baseline, permuted)


@given(events=event_frames(), universe=universe_frames())
@settings(max_examples=40, deadline=None)
def test_a_larger_staleness_bound_never_removes_a_row(
    events: pd.DataFrame, universe: pd.DataFrame
) -> None:
    strict = expand_fundamental_events(
        events, universe, CALENDAR, max_staleness_calendar_days=60
    )
    lenient = expand_fundamental_events(
        events, universe, CALENDAR, max_staleness_calendar_days=365
    )

    strict_keys = set(map(tuple, strict.loc[:, ["signal_date", "ts_code"]].to_numpy().tolist()))
    lenient_keys = set(map(tuple, lenient.loc[:, ["signal_date", "ts_code"]].to_numpy().tolist()))

    assert strict_keys <= lenient_keys, "widening the staleness bound dropped a record"


@given(events=event_frames(), universe=universe_frames())
@settings(max_examples=40, deadline=None)
def test_a_published_value_is_always_one_of_that_security_announced_values(
    events: pd.DataFrame, universe: pd.DataFrame
) -> None:
    """The expansion selects an announced version; it never interpolates or synthesises."""
    expanded = expand_fundamental_events(events, universe, CALENDAR)
    if expanded.empty:
        return
    announced: dict[str, set[float]] = {}
    for row in events.to_dict(orient="records"):
        announced.setdefault(str(row["ts_code"]), set()).add(float(row["raw_value"]))

    for row in expanded.to_dict(orient="records"):
        assert float(row["raw_value"]) in announced[str(row["ts_code"])]
