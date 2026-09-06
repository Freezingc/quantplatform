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


def test_fundamental_event_prefers_latest_report_period_before_revision() -> None:
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
