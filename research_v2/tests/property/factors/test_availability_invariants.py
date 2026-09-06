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

**Hypothesis found two input shapes that violated that last property, on the same day and after it
had passed for several sessions. One is fixed, one is still a pinned defect, and the difference
between them is the reason only one was fixed.**

*Shape 1 -- fixed.* ``available_date`` is derived from ``ann_date`` through the calendar and that
map is many-to-one: a Friday and the weekend behind it all roll to the same next trading day. Two
events announced on different calendar days could therefore tie on ``available_date``,
``report_period`` and ``source_revision_id`` while disagreeing on ``ann_date``, and a stable sort
with no fourth key resolved them by input row order -- publishing a different *provenance*
depending on how the extract arrived, with every value in agreement. ``ann_date`` is now the
fourth, last sort key, which makes the sort total without reordering the first three. Pinned by
``test_availability.py::test_announcement_dates_collapsing_to_one_available_date_are_order_invariant``.

*Shape 2 -- still a defect, still deferred to decision B.* Two events sharing one ``ann_date``, so
tied on all four keys, but carrying different ``raw_value``s, are still resolved by input row
order. They are excluded from the property below by ``_has_contradictory_tie`` and pinned
deterministically by
``test_availability.py::test_two_events_tied_on_every_sort_key_are_resolved_by_input_order``,
because what should happen when two revisions are indistinguishable is a semantic choice, and the
technical path's answer -- a deduplication and quarantine layer that refuses to pick between
conflicting values -- has no counterpart on the fundamental path at all.

Shape 1 was fixed rather than deferred because nothing semantic is chosen by it: no value competes,
and no precedence among ``available_date`` / ``report_period`` / ``source_revision_id`` is
restated, so it is compatible with every ordering decision B could reach. That the two shapes are
disjoint is measured, not assumed -- the shape-2 tripwire test passes unchanged on both sides of
the fix.
"""

from __future__ import annotations

from datetime import date

import pandas as pd
from hypothesis import assume, given, settings
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
    assume(not _has_contradictory_tie(events))

    baseline = expand_fundamental_events(events, universe, CALENDAR)

    shuffled = events.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    permuted = expand_fundamental_events(shuffled, universe, CALENDAR)

    pd.testing.assert_frame_equal(baseline, permuted)


def _has_contradictory_tie(events: pd.DataFrame) -> bool:
    """True when two events are indistinguishable to the sort yet disagree on the value.

    ``expand_fundamental_events`` orders candidates by ``available_date``, ``report_period``,
    ``source_revision_id`` and ``ann_date``, all descending, and takes the first with a stable
    sort, so rows tied on all four keep their input order. Grouping on
    ``(ts_code, report_period, ann_date, source_revision_id)`` is exactly that tie set, because
    ``available_date`` is a function of ``ann_date``: equal ``ann_date`` implies equal
    ``available_date``, so the derived column needs no place in the key.

    The implication runs only that way. The converse -- different ``ann_date`` implies different
    ``available_date`` -- is false, because the calendar rolls a non-trading day forward to the
    next trading one and is therefore many-to-one. Believing the converse is what let a second
    order-dependence through this filter until 2026-09-06; that shape is now closed by the fourth
    sort key rather than excluded here, so this filter describes the *only* remaining tie and must
    not be widened. Filtering these out is what lets the property above state something true; the
    excluded shape is a defect with its own deterministic test, not a case nobody thought about.
    """
    if events.empty:
        return False
    keys = ["ts_code", "report_period", "ann_date", "source_revision_id"]
    distinct_values = events.groupby(keys, sort=False, dropna=False)["raw_value"].nunique()
    return bool((distinct_values > 1).any())


def test_the_excluded_shape_is_really_the_one_that_breaks_order_invariance() -> None:
    """The filter above has to be narrow, or it would hide cases the property could cover.

    Without this, ``assume`` could be widened until the property tested nothing and still
    passed -- the same failure mode as a check that cannot fail. Two events tied on every sort
    key with the *same* value are not contradictory and must stay inside the property; only
    differing values are excluded.
    """
    tied_same_value = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "report_period": [date(2024, 9, 30)] * 2,
            "ann_date": [date(2025, 1, 1)] * 2,
            "raw_value": [1.5, 1.5],
            "source_revision_id": ["r1", "r1"],
        }
    )
    tied_different_value = tied_same_value.assign(raw_value=[1.5, 2.5])
    different_revision = tied_same_value.assign(source_revision_id=["r1", "r2"])

    assert not _has_contradictory_tie(tied_same_value)
    assert not _has_contradictory_tie(different_revision)
    assert _has_contradictory_tie(tied_different_value)
    assert not _has_contradictory_tie(pd.DataFrame(columns=tied_same_value.columns))


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
