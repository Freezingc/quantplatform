from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.factors.availability import (
    assign_fundamental_available_date,
    expand_fundamental_events,
    materialize_technical_availability,
)
from research_v2.pit.calendar import TradingCalendar


def _calendar() -> TradingCalendar:
    return TradingCalendar.from_dates(
        [
            date(2025, 3, 28),
            date(2025, 3, 31),
            date(2025, 4, 1),
            date(2025, 9, 26),
            date(2025, 9, 29),
        ]
    )


def test_technical_factor_is_available_only_on_its_observation_date() -> None:
    observations = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "observation_date": [date(2025, 3, 28)],
            "raw_value": [1.0],
        }
    )

    result = materialize_technical_availability(observations)

    assert result.loc[0, "available_date"] == date(2025, 3, 28)
    assert result.loc[0, "signal_date"] == date(2025, 3, 28)
    assert result.loc[0, "stale_days"] == 0


def test_fundamental_factor_becomes_available_on_next_trading_day_and_expires() -> None:
    calendar = _calendar()
    assert assign_fundamental_available_date(date(2025, 3, 28), calendar) == date(2025, 3, 31)
    events = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "report_period": [date(2024, 12, 31)],
            "ann_date": [date(2025, 3, 28)],
            "raw_value": [2.0],
            "source_revision_id": ["first"],
        }
    )
    universe = pd.DataFrame(
        {
            "signal_date": [
                date(2025, 3, 28),
                date(2025, 3, 31),
                date(2025, 9, 26),
                date(2025, 9, 29),
            ],
            "ts_code": ["000001.SZ"] * 4,
        }
    )

    result = expand_fundamental_events(events, universe, calendar, max_staleness_calendar_days=180)

    assert result["signal_date"].tolist() == [date(2025, 3, 31), date(2025, 9, 26)]
    assert result["stale_days"].tolist() == [0, 179]


def test_a_shared_announcement_date_tie_is_broken_on_the_latest_report_period() -> None:
    """A tie-break, not period precedence. Both versions share one ``ann_date``, so the sort
    on ``available_date`` cannot separate them and falls through to ``report_period``.

    The name this replaces claimed to pin "prefers latest report period before revision",
    which is exactly what ``test_a_late_restatement_of_an_older_period_supersedes_a_newer_period``
    below shows does not hold once the announcement dates differ. Renaming a test cannot
    change behaviour, so the append-only discipline that kept the old name in place while a
    bit-for-bit reproduction was running does not apply to it.
    """
    calendar = _calendar()
    events = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "report_period": [date(2024, 9, 30), date(2024, 12, 31)],
            "ann_date": [date(2025, 3, 28), date(2025, 3, 28)],
            "raw_value": [1.0, 2.0],
            "source_revision_id": ["z-earlier-period", "a-later-period"],
        }
    )
    universe = pd.DataFrame(
        {
            "signal_date": [date(2025, 3, 31)],
            "ts_code": ["000001.SZ"],
        }
    )

    result = expand_fundamental_events(events, universe, calendar)

    assert result.loc[0, "report_period"] == date(2024, 12, 31)
    assert result.loc[0, "raw_value"] == 2.0
    assert result.loc[0, "source_revision_id"] == "a-later-period"


def test_two_events_tied_on_every_sort_key_are_resolved_by_input_order() -> None:
    """DEFECT, pinned rather than fixed: a full tie lets the input row order pick the value.

    ``expand_fundamental_events`` orders candidates by ``available_date``, ``report_period``
    and ``source_revision_id``, all descending, with a stable sort, and takes the first. Two
    events that tie on all three therefore keep whatever order they arrived in -- and if they
    carry different values, the published number depends on the row order of the extract.
    Hypothesis found this on 2026-09-06 in
    ``test_event_row_order_does_not_change_the_expansion``, after that property had passed for
    several sessions; the minimal case is two rows identical in every key field with values
    0.0 and 1.0, which is what is reproduced here deterministically.

    Not fixed, for two reasons that are decisions rather than caution. What *should* happen
    when two revisions are indistinguishable is a semantic choice, and the ordering semantics
    of this function are already open as decision B, which would restate the sort key and with
    it the C4.3 interval formula. And the deeper gap is structural: the technical path meets
    contradictory duplicates with a whole deduplication and quarantine layer that refuses to
    choose between conflicting values, while the fundamental path has no counterpart at all --
    so the honest fix is probably "give this path the same layer", which is architecture.

    Either way this test is the tripwire: any tie-break, refusal, or dedup layer added here has
    to change it deliberately.
    """
    events = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "report_period": [date(2024, 9, 30), date(2024, 9, 30)],
            "ann_date": [date(2025, 1, 1), date(2025, 1, 1)],
            "raw_value": [0.0, 1.0],
            "source_revision_id": ["r1", "r1"],
        }
    )
    universe = pd.DataFrame([{"signal_date": date(2025, 1, 2), "ts_code": "000001.SZ"}])
    calendar = TradingCalendar.from_dates(
        [date(2025, 1, 1), date(2025, 1, 2), date(2025, 1, 3)]
    )

    forward = expand_fundamental_events(events, universe, calendar)
    backward = expand_fundamental_events(
        events.iloc[::-1].reset_index(drop=True), universe, calendar
    )

    assert len(forward) == 1 and len(backward) == 1
    assert forward.loc[0, "raw_value"] == 0.0
    assert backward.loc[0, "raw_value"] == 1.0, (
        "the same two events publish a different value depending only on the order they "
        "arrived in, which is the defect this test exists to keep visible"
    )


EXPANSION_COLUMNS = [
    "signal_date",
    "ts_code",
    "raw_value",
    "report_period",
    "ann_date",
    "available_date",
    "source_revision_id",
    "stale_days",
]


def test_expansion_before_any_announcement_is_available_returns_empty_not_key_error() -> None:
    """Nothing qualifies, and that is a normal outcome rather than an error.

    Reachable in production, not a corner case to tolerate: an early fold that precedes the
    first announcement, a newly listed security, or a staleness bound tight enough that every
    version has expired. The previous behaviour was ``KeyError: 'signal_date'`` raised by the
    trailing ``sort_values`` on a frame built from an empty row list -- which named a column
    and so pointed nowhere near the actual cause.
    """
    calendar = _calendar()
    events = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "report_period": [date(2025, 6, 30)],
            "ann_date": [date(2025, 9, 26)],  # available 2025-09-29
            "raw_value": [2.0],
            "source_revision_id": ["r1"],
        }
    )
    universe = pd.DataFrame(
        {"signal_date": [date(2025, 3, 31)], "ts_code": ["000001.SZ"]}
    )

    result = expand_fundamental_events(events, universe, calendar)

    assert result.empty
    assert list(result.columns) == EXPANSION_COLUMNS


def test_expansion_where_every_version_has_expired_returns_empty_not_key_error() -> None:
    """The second route to an empty result: available, but stale beyond the bound."""
    calendar = _calendar()
    events = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "report_period": [date(2024, 12, 31)],
            "ann_date": [date(2025, 3, 28)],  # available 2025-03-31
            "raw_value": [1.0],
            "source_revision_id": ["r1"],
        }
    )
    universe = pd.DataFrame(
        {"signal_date": [date(2025, 9, 29)], "ts_code": ["000001.SZ"]}
    )

    result = expand_fundamental_events(events, universe, calendar)

    # 2025-03-31 to 2025-09-29 is 182 calendar days, past the 180-day bound.
    assert (date(2025, 9, 29) - date(2025, 3, 31)).days == 182
    assert result.empty
    assert list(result.columns) == EXPANSION_COLUMNS


# --- C4.1/C4.3 clauses that were untested, and two behaviours the spec does not settle ----------


def test_a_missing_technical_day_is_left_missing_rather_than_filled_from_the_previous_one() -> None:
    """C4.1: 缺值保持缺值，不从前一日 ffill.

    A forward-filled day is indistinguishable from an observed one downstream, so a gap in the
    source would become a factor value that was never measured -- and for a technical factor the
    whole availability rule is that the value exists only on its own close-to-close date. Pinning
    the row count is what makes this a no-ffill test rather than a value test: filling would add a
    row, and no assertion about the rows that are present would notice.

    "不 ffill" has two readings and both are covered, because a test for only one passes under a
    regression of the other. A date with no observation at all must not gain a row (000001.SZ is
    missing 03-31); a date that *has* a row whose value is missing must keep that value missing
    rather than inherit the previous day's (000002.SZ on 03-31). The second is the one a
    ``groupby.ffill()`` would silently get wrong while leaving every row count intact.
    """
    observations = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ", "000002.SZ"],
            "observation_date": [
                date(2025, 3, 28),
                date(2025, 4, 1),
                date(2025, 3, 28),
                date(2025, 3, 31),
            ],
            "raw_value": [1.0, 3.0, 2.0, float("nan")],
        }
    )

    result = materialize_technical_availability(observations)

    assert len(result) == len(observations), "a row was invented for the missing 2025-03-31"

    first = result.loc[result["ts_code"] == "000001.SZ"]
    assert first["signal_date"].tolist() == [date(2025, 3, 28), date(2025, 4, 1)]
    assert first["available_date"].tolist() == [date(2025, 3, 28), date(2025, 4, 1)]
    assert first["stale_days"].tolist() == [0, 0]
    assert date(2025, 3, 31) not in set(first["signal_date"]), "the gap gained a row"

    second = result.loc[result["ts_code"] == "000002.SZ"]
    assert second["signal_date"].tolist() == [date(2025, 3, 28), date(2025, 3, 31)]
    assert second["raw_value"].isna().tolist() == [False, True], (
        "the missing value inherited the previous day's 2.0 instead of staying missing"
    )


def test_the_staleness_bound_is_inclusive_at_the_limit_and_exclusive_one_day_past() -> None:
    """C4.3: 第 181 个自然日值为空 -- which is only true if the comparison is ``>`` and not ``>=``.

    The default 180 and the dates in the neighbouring test leave the operator itself unpinned: they
    exercise stale=179 (kept) and stale=182 (dropped), so a bound written as ``>=`` with an
    off-by-one would pass both. Varying the bound against a fixed stale distance isolates the
    operator, which is the part that decides whether the 180th or the 181st day is the last usable.
    """
    calendar = _calendar()
    events = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "report_period": [date(2024, 12, 31)],
            "ann_date": [date(2025, 3, 28)],
            "raw_value": [2.0],
            "source_revision_id": ["first"],
        }
    )
    # available 2025-03-31; 2025-09-26 is exactly 179 calendar days later
    universe = pd.DataFrame({"signal_date": [date(2025, 9, 26)], "ts_code": ["000001.SZ"]})

    at_the_limit = expand_fundamental_events(
        events, universe, calendar, max_staleness_calendar_days=179
    )
    one_day_past = expand_fundamental_events(
        events, universe, calendar, max_staleness_calendar_days=178
    )

    assert len(at_the_limit) == 1 and at_the_limit.loc[0, "stale_days"] == 179
    assert one_day_past.empty, "the bound was treated as inclusive one day too far"


def test_a_late_restatement_of_an_older_period_supersedes_a_newer_period() -> None:
    """Current behaviour, and a defect the spec does not settle -- pinned, not fixed.

    FY2024 is announced 2025-03-28 (available 03-31). On 2025-04-10 the issuer restates the
    *older* Q3 2024 period (available 04-11). At signal date 2025-04-15 the expansion publishes
    the restated Q3 value and drops FY2024, because versions are ordered by ``available_date``
    first and the restatement is the most recently available thing.

    Under ordinary point-in-time fundamental semantics this is wrong: the value for a security on
    a date should come from the latest *reporting period* then available, with a revision settling
    ties inside that period. Publishing an older period's number because it happened to be
    restated later silently rewinds the reporting vintage.

    The spec is genuinely ambiguous, which is why this is recorded rather than corrected. C4.3
    defines the interval as ending at ``next_version_available_date - 1 trading day`` -- a single
    linear version sequence per security, which is exactly what the code implements. C4.4 only
    speaks to revisions *of the same report period*. Neither says what outranks what across
    periods. Note also that the neighbouring test named
    ``test_fundamental_event_prefers_latest_report_period_before_revision`` asserts a precedence
    the code does not implement in general: it holds there only because both versions share one
    ``available_date``, so the sort falls through to ``report_period``. The name reads as a
    guarantee about period precedence; it is a guarantee about a tie-break.

    Not changed here. ``expand_fundamental_events`` has no production caller (only
    ``materialize_technical_availability`` is wired, via the materializer), so nothing is
    mispricing today -- and choosing between the two readings would fix fundamental PIT semantics
    for every factor that is ever wired through it. That is an owner decision, and the unwired
    state is the cheap moment to make it.
    """
    calendar = TradingCalendar.from_dates(
        [
            date(2025, 3, 28),
            date(2025, 3, 31),
            date(2025, 4, 1),
            date(2025, 4, 10),
            date(2025, 4, 11),
            date(2025, 4, 15),
        ]
    )
    events = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"] * 2,
            "report_period": [date(2024, 12, 31), date(2024, 9, 30)],
            "ann_date": [date(2025, 3, 28), date(2025, 4, 10)],
            "raw_value": [2.0, 1.5],
            "source_revision_id": ["fy2024-original", "q3-2024-restated"],
        }
    )

    before = expand_fundamental_events(
        events,
        pd.DataFrame({"signal_date": [date(2025, 4, 1)], "ts_code": ["000001.SZ"]}),
        calendar,
    )
    after = expand_fundamental_events(
        events,
        pd.DataFrame({"signal_date": [date(2025, 4, 15)], "ts_code": ["000001.SZ"]}),
        calendar,
    )

    assert before.loc[0, "report_period"] == date(2024, 12, 31)
    assert before.loc[0, "raw_value"] == 2.0, "before the restatement, FY2024 is correctly latest"

    assert after.loc[0, "report_period"] == date(2024, 9, 30)
    assert after.loc[0, "raw_value"] == 1.5
    assert after.loc[0, "source_revision_id"] == "q3-2024-restated"
    assert (after["report_period"] < before["report_period"]).all(), (
        "the published reporting period moved backwards across the restatement"
    )


def test_a_fundamental_record_with_no_announcement_date_raises_not_quarantined() -> None:
    """C4.2 requires such a record to be *isolated*; today it aborts the expansion obscurely.

    The clause's second half does hold -- ``report_period`` is never substituted for a missing
    ``ann_date``, so there is no silent look-ahead. But the first half does not: a NaT
    announcement date surfaces as ``TypeError: Cannot compare NaT with datetime.date object``,
    naming a pandas comparison rather than the actual problem, and it takes the whole expansion
    down with it. That contradicts this module's own stated intent -- ``_empty_expansion`` exists
    precisely because "one empty result must not abort the whole materialization" -- and it is the
    opposite of quarantining, which keeps the batch alive and records why one row was held back.

    Quarantining is not implemented here because the function has nowhere to put a quarantined row:
    it returns one DataFrame and has no second channel or reason column, so honouring the clause
    means changing its contract. Given ``expand_fundamental_events`` has no production caller, the
    contract is cheaper to settle now than after something depends on its return shape.
    """
    calendar = _calendar()
    events = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "report_period": [date(2024, 12, 31)],
            "ann_date": [pd.NaT],
            "raw_value": [2.0],
            "source_revision_id": ["first"],
        }
    )
    universe = pd.DataFrame({"signal_date": [date(2025, 4, 1)], "ts_code": ["000001.SZ"]})

    with pytest.raises(TypeError, match="NaT"):
        expand_fundamental_events(events, universe, calendar)


# --- C4.2's delisting case, which the module satisfies only implicitly ---------------------------


def _calendar_april() -> TradingCalendar:
    return TradingCalendar.from_dates(
        [
            date(2025, 3, 28),
            date(2025, 3, 31),
            date(2025, 4, 1),
            date(2025, 4, 2),
            date(2025, 4, 3),
        ]
    )


def _fy2024_event(ts_code: str) -> dict[str, object]:
    """One FY2024 announcement, available from 2025-03-31 and fresh for 180 days after."""
    return {
        "ts_code": ts_code,
        "report_period": date(2024, 12, 31),
        "ann_date": date(2025, 3, 28),
        "raw_value": 2.0,
        "source_revision_id": f"fy2024-{ts_code}",
    }


def test_a_delisted_security_stops_expanding_while_its_value_is_still_fresh() -> None:
    """C4.2's 退市 case: membership, not staleness, decides when a value stops being published.

    The module has no ``delist`` logic at all -- it holds because rows are emitted only for
    (signal_date, ts_code) pairs the universe actually contains, which is the right design: the
    universe owns membership, so availability should not decide it a second time. But implicit and
    untested means unguarded, and the failure mode is concrete. If staleness were ever made the
    only gate -- a plausible "simplification" -- a delisted security would keep publishing its last
    announced fundamental for up to 180 more days, which is exactly a look-ahead-free but
    investability-blind factor value on a security that can no longer be traded.

    The control security is what makes this a test of the mechanism rather than of the window: it
    carries the identical event and *is* in the universe on 2025-04-03, so it proves the value was
    still fresh on the date the delisted one went silent. Without it, "no row" would be equally
    consistent with the staleness bound having closed.
    """
    calendar = _calendar_april()
    events = pd.DataFrame([_fy2024_event("000001.SZ"), _fy2024_event("000002.SZ")])
    universe = pd.DataFrame(
        [
            # 000001.SZ delists after 2025-04-02; 000002.SZ stays listed throughout
            {"signal_date": day, "ts_code": "000001.SZ"}
            for day in (date(2025, 3, 31), date(2025, 4, 1), date(2025, 4, 2))
        ]
        + [{"signal_date": day, "ts_code": "000002.SZ"} for day in calendar.dates[1:]]
    )

    result = expand_fundamental_events(events, universe, calendar)

    delisted = result.loc[result["ts_code"] == "000001.SZ", "signal_date"].tolist()
    assert delisted == [date(2025, 3, 31), date(2025, 4, 1), date(2025, 4, 2)]
    assert date(2025, 4, 3) not in delisted

    control = result.loc[result["ts_code"] == "000002.SZ"]
    assert date(2025, 4, 3) in control["signal_date"].tolist(), (
        "the control proves the staleness window was still open on 2025-04-03"
    )
    assert int(control.loc[control["signal_date"] == date(2025, 4, 3), "stale_days"].iloc[0]) == 3
    assert float(control["raw_value"].iloc[0]) == 2.0


def test_a_suspended_security_gaps_and_then_resumes_without_losing_the_value() -> None:
    """Membership can be non-contiguous; the value must reappear, not be treated as expired.

    A suspension is the same mechanism as a delisting except that membership returns. If the
    expansion carried any notion of "consecutive days since available", a gap would either extend
    staleness across it or drop the security permanently. Neither is right: staleness is measured
    from the available date in calendar days, so the value simply reappears with a larger
    stale_days, and a gap long enough to cross the bound expires it on the ordinary rule.

    The gap deliberately spans a **revision**, which is what makes the staleness assertion mean
    something. With a single event the available date never changes, so measuring staleness from
    the previous published row and measuring it from the winning version's own available date give
    the same number and the two are indistinguishable -- a first version of this test used one
    event, claimed in its docstring to pin "measured from the available date, not from the last
    published row", and did not. Across a revision the winner on 2025-04-03 is the later version
    (available 04-02, so stale 1), while the previous row carries the earlier version's available
    date (03-31, which would give stale 3). Same report period, so this stays inside C4.4's
    same-period revision semantics and does not depend on the unresolved cross-period question.
    """
    calendar = _calendar_april()
    events = pd.DataFrame(
        [
            _fy2024_event("000001.SZ"),
            {
                # a revision of the same report period, announced 2025-04-01 -> available 04-02
                "ts_code": "000001.SZ",
                "report_period": date(2024, 12, 31),
                "ann_date": date(2025, 4, 1),
                "raw_value": 3.0,
                "source_revision_id": "fy2024-revised",
            },
        ]
    )
    universe = pd.DataFrame(
        [
            {"signal_date": date(2025, 3, 31), "ts_code": "000001.SZ"},
            # absent 2025-04-01 and 2025-04-02 (suspended)
            {"signal_date": date(2025, 4, 3), "ts_code": "000001.SZ"},
        ]
    )

    result = expand_fundamental_events(events, universe, calendar)

    assert result["signal_date"].tolist() == [date(2025, 3, 31), date(2025, 4, 3)]
    assert result["stale_days"].tolist() == [0, 1], (
        "staleness is measured from the winning version's own available date (04-02), "
        "not from the previous published row's (03-31, which would give 3)"
    )
    assert result["raw_value"].tolist() == [2.0, 3.0]
    assert result["source_revision_id"].tolist() == [
        "fy2024-000001.SZ",
        "fy2024-revised",
    ], "before the revision the original is published; after it, the revision"
