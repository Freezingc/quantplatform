"""Future-perturbation leakage canaries for point-in-time factor availability (C plan C4.6).

The contract: adding or modifying a future announcement, a future revision, or a future
technical value must not change the clean factor record for any earlier signal date, and
deleting future universe dates must not change the past expansion. This is the property
that makes a walk-forward result interpretable at all -- if restating history could reach
backwards, every out-of-sample number would be contaminated by information that did not
exist when the signal date closed.

The case that matters most is a future *revision of a past report period*: an annual figure
announced in March and restated in September. Signal dates before the restatement must keep
seeing the March version, because that is what was knowable then.

Each test also asserts the perturbation really did change something after the cutoff. Without
that guard a leakage test can pass vacuously -- for example by perturbing a row the expansion
never reads -- and look like evidence while proving nothing.
"""

from __future__ import annotations

import hashlib
from datetime import date

import pandas as pd

from research_v2.factors.availability import (
    expand_fundamental_events,
    materialize_technical_availability,
)
from research_v2.pit.calendar import TradingCalendar

CUTOFF = date(2025, 6, 30)
CODES = ("000001.SZ", "600000.SH")
PAST_SIGNAL_DATES = (date(2025, 4, 1), date(2025, 5, 2), date(2025, 6, 2), CUTOFF)
FUTURE_SIGNAL_DATES = (date(2025, 9, 1), date(2025, 10, 1), date(2025, 11, 3))


def _calendar() -> TradingCalendar:
    """Every weekday of 2025; only ordering and shift semantics matter here."""
    return TradingCalendar.from_dates(
        [timestamp.date() for timestamp in pd.bdate_range("2025-01-01", "2025-12-31")]
    )


def _events() -> pd.DataFrame:
    """Annual figures announced 2025-03-28, available the next open day (2025-03-31)."""
    rows = [
        {
            "ts_code": code,
            "report_period": date(2024, 12, 31),
            "ann_date": date(2025, 3, 28),
            "raw_value": 1.0,
            "source_revision_id": "r1",
        }
        for code in CODES
    ]
    rows += [
        {
            "ts_code": code,
            "report_period": date(2025, 6, 30),
            "ann_date": date(2025, 8, 29),
            "raw_value": 2.0,
            "source_revision_id": "r1",
        }
        for code in CODES
    ]
    return pd.DataFrame(rows)


def _universe(signal_dates: tuple[date, ...] = ()) -> pd.DataFrame:
    dates = signal_dates or (*PAST_SIGNAL_DATES, *FUTURE_SIGNAL_DATES)
    return pd.DataFrame(
        [{"signal_date": day, "ts_code": code} for day in dates for code in CODES]
    )


def _digest(frame: pd.DataFrame) -> str:
    """Canonical digest of the records at or before the cutoff.

    Columns are sorted by name and rows by business key so the digest depends only on the
    published content, not on the order the expansion happened to emit it in.

    An empty scope is an error rather than a sentinel hash. Two empty scopes digest equal,
    so a fixture that accidentally published nothing before the cutoff would make every
    leakage assertion below pass without testing anything.
    """
    scoped = frame.loc[[day <= CUTOFF for day in frame["signal_date"]]]
    if scoped.empty:
        raise AssertionError(
            "no records at or before the cutoff; the fixture would make the leakage "
            "assertions vacuous"
        )
    ordered = scoped.loc[:, sorted(scoped.columns)].sort_values(
        ["signal_date", "ts_code"], kind="mergesort", ignore_index=True
    )
    return hashlib.sha256(ordered.to_csv(index=False).encode("utf-8")).hexdigest()


def _after_cutoff(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[[day > CUTOFF for day in frame["signal_date"]]]


def test_a_future_revision_of_a_past_report_period_does_not_rewrite_history() -> None:
    """The look-ahead case: restating the 2024 annual figure in September 2025."""
    calendar = _calendar()
    baseline = expand_fundamental_events(_events(), _universe(), calendar)

    restated = pd.concat(
        [
            _events(),
            pd.DataFrame(
                [
                    {
                        "ts_code": code,
                        "report_period": date(2024, 12, 31),
                        "ann_date": date(2025, 9, 30),
                        "raw_value": 99.0,
                        "source_revision_id": "r2",
                    }
                    for code in CODES
                ]
            ),
        ],
        ignore_index=True,
    )
    perturbed = expand_fundamental_events(restated, _universe(), calendar)

    assert _digest(perturbed) == _digest(baseline)
    past = perturbed.loc[[day <= CUTOFF for day in perturbed["signal_date"]]]
    assert set(past["raw_value"]) == {1.0}, "a restatement reached back before its announcement"
    assert set(past["source_revision_id"]) == {"r1"}
    # Guard: the restatement must take effect once it is actually available.
    later = _after_cutoff(perturbed)
    restated_rows = later.loc[later["signal_date"] >= date(2025, 10, 1)]
    assert 99.0 in set(restated_rows["raw_value"]), "perturbation had no effect anywhere"
    assert 99.0 not in set(_after_cutoff(baseline)["raw_value"])


def test_adding_a_wholly_future_announcement_leaves_earlier_records_unchanged() -> None:
    calendar = _calendar()
    baseline = expand_fundamental_events(_events(), _universe(), calendar)

    extended = pd.concat(
        [
            _events(),
            pd.DataFrame(
                [
                    {
                        "ts_code": code,
                        "report_period": date(2025, 9, 30),
                        "ann_date": date(2025, 10, 31),
                        "raw_value": 3.0,
                        "source_revision_id": "r1",
                    }
                    for code in CODES
                ]
            ),
        ],
        ignore_index=True,
    )
    perturbed = expand_fundamental_events(extended, _universe(), calendar)

    assert _digest(perturbed) == _digest(baseline)
    later = _after_cutoff(perturbed)
    assert 3.0 in set(later["raw_value"]), "perturbation had no effect anywhere"
    assert 3.0 not in set(_after_cutoff(baseline)["raw_value"])


def test_modifying_a_future_value_leaves_the_earlier_digest_unchanged() -> None:
    calendar = _calendar()
    baseline = expand_fundamental_events(_events(), _universe(), calendar)

    modified = _events()
    future_mask = modified["ann_date"] > CUTOFF
    modified.loc[future_mask, "raw_value"] = 1234.5
    perturbed = expand_fundamental_events(modified, _universe(), calendar)

    assert bool(future_mask.any()), "the fixture must contain a post-cutoff announcement"
    assert _digest(perturbed) == _digest(baseline)
    assert 1234.5 in set(_after_cutoff(perturbed)["raw_value"])


def test_deleting_future_universe_dates_does_not_change_the_past_expansion() -> None:
    calendar = _calendar()
    baseline = expand_fundamental_events(_events(), _universe(), calendar)

    truncated = expand_fundamental_events(
        _events(), _universe(PAST_SIGNAL_DATES), calendar
    )

    assert _digest(truncated) == _digest(baseline)
    assert not truncated.loc[[day > CUTOFF for day in truncated["signal_date"]]].shape[0]
    assert _after_cutoff(baseline).shape[0] > 0, "guard: baseline must have post-cutoff rows"


def test_a_future_technical_value_does_not_change_earlier_records() -> None:
    """Technical factors are same-day-close, so the argument is per-row rather than per-join."""
    past = pd.DataFrame(
        [
            {"ts_code": code, "observation_date": day, "raw_value": 1.0}
            for day in PAST_SIGNAL_DATES
            for code in CODES
        ]
    )
    baseline = materialize_technical_availability(past)

    extended = pd.concat(
        [
            past,
            pd.DataFrame(
                [
                    {"ts_code": code, "observation_date": day, "raw_value": 777.0}
                    for day in FUTURE_SIGNAL_DATES
                    for code in CODES
                ]
            ),
        ],
        ignore_index=True,
    )
    perturbed = materialize_technical_availability(extended)

    assert _digest(perturbed) == _digest(baseline)
    assert 777.0 in set(_after_cutoff(perturbed)["raw_value"])
    assert 777.0 not in set(_after_cutoff(baseline)["raw_value"])


def test_every_published_row_satisfies_the_point_in_time_ordering() -> None:
    """C4.5's invariant, asserted on the same fixtures: observation <= available <= signal."""
    calendar = _calendar()
    expanded = expand_fundamental_events(_events(), _universe(), calendar)

    assert not expanded.empty
    for row in expanded.to_dict(orient="records"):
        assert row["available_date"] <= row["signal_date"]
        assert row["ann_date"] < row["available_date"], (
            "a fundamental value must be usable only from the next open-market day"
        )
        assert row["stale_days"] == (row["signal_date"] - row["available_date"]).days

    technical = materialize_technical_availability(
        pd.DataFrame(
            [
                {"ts_code": "000001.SZ", "observation_date": date(2025, 4, 1), "raw_value": 1.0}
            ]
        )
    )
    row = technical.iloc[0]
    assert row["observation_date"] == row["available_date"] == row["signal_date"]
    assert row["stale_days"] == 0
