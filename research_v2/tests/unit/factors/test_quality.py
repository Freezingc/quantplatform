from __future__ import annotations

from datetime import date
from typing import Any

import numpy as np
import pandas as pd
import pytest

from research_v2.factors.quality import (
    FactorQualitySnapshot,
    QualityStatus,
)
from research_v2.factors.quality import (
    assess_factor_quality as assess_with_explicit_scope,
)


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 3 + [date(2025, 1, 3)] * 3,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"] * 2,
        }
    )


SCOPE_ID = "train-scope-2014-2019"
AS_OF = date(2025, 12, 31)


def assess_factor_quality(  # noqa: F811 - deliberate module-wide fixture, see docstring
    universe: pd.DataFrame, observations: pd.DataFrame, **kwargs: Any
) -> FactorQualitySnapshot:
    """Pin the scope and as-of date once so each case below stays about what it tests.

    C6.1 requires both on every snapshot, and they are identical for all nineteen call
    sites in this module; repeating them would bury the assertion each test is actually
    about. The wrapper only supplies defaults, so a case that needs a different factor
    version or an explicit scope still passes it through.
    """
    kwargs.setdefault("factor_id", "yeslab_1")
    kwargs.setdefault("factor_version", 1)
    kwargs.setdefault("scope_id", SCOPE_ID)
    kwargs.setdefault("as_of_date", AS_OF)
    return assess_with_explicit_scope(universe, observations, **kwargs)


def test_quality_uses_universe_members_as_coverage_denominator_and_expires_stale_values() -> None:
    observations = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2), date(2025, 1, 2), date(2025, 1, 3)],
            "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ"],
            "value": [1.0, 2.0, 3.0],
            "stale_days": [0, 0, 181],
        }
    )

    snapshot = assess_factor_quality(
        _universe(),
        observations,
        factor_id="yeslab_1",
        factor_version=1,
    )

    assert snapshot.metrics["universe_member_count"] == 6
    assert snapshot.metrics["finite_value_count"] == 2
    assert snapshot.metrics["coverage"] == 2 / 6
    assert snapshot.status is QualityStatus.ELIGIBILITY_FAIL


def test_quality_blocks_duplicate_business_keys_and_nonfinite_clean_values() -> None:
    observations = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "ts_code": ["000001.SZ", "000001.SZ"],
            "value": [1.0, 2.0],
            "stale_days": [0, 0],
        }
    )

    snapshot = assess_factor_quality(
        _universe(),
        observations,
        factor_id="yeslab_1",
        factor_version=1,
    )

    assert snapshot.status is QualityStatus.BLOCKER
    assert "DUPLICATE_BUSINESS_KEY" in snapshot.blockers


def test_quality_returns_blocker_snapshot_for_duplicate_universe_keys_instead_of_crashing() -> None:
    duplicated_universe = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "ts_code": ["000001.SZ", "000001.SZ"],
        }
    )
    observations = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)],
            "ts_code": ["000001.SZ"],
            "value": [1.0],
            "stale_days": [0],
        }
    )

    snapshot = assess_factor_quality(
        duplicated_universe,
        observations,
        factor_id="yeslab_1",
        factor_version=1,
    )

    assert snapshot.status is QualityStatus.BLOCKER
    assert "DUPLICATE_UNIVERSE_KEY" in snapshot.blockers


def _observations(**extra_columns: object) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2), date(2025, 1, 2), date(2025, 1, 3)],
            "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ"],
            "value": [1.0, 2.0, 3.0],
            "stale_days": [0, 0, 1],
        }
    )
    for name, values in extra_columns.items():
        frame[name] = values
    return frame


def test_structural_quality_is_invariant_to_a_perfectly_predictive_label_column() -> None:
    """Completion criterion: structural quality does not read labels.

    ``assess_factor_quality`` documents that it scores structure from the universe and the
    factor observations only, and implements it by projecting to
    signal_date/ts_code/value/stale_days before the merge. This pins that projection: a
    label column carrying a perfectly predictive signal -- one that would drive any
    label-aware metric to its extreme -- must not move the snapshot by anything.

    Without it, widening the projection or passing the whole frame through would silently
    let a factor's *eligibility* depend on outcomes, which is the exact inversion of what a
    label-free structural screen is for: eligibility is decided before any outcome is
    observed, so a factor could otherwise be admitted or rejected on information that did
    not exist at admission time.
    """
    baseline = assess_factor_quality(
        _universe(), _observations(), factor_id="yeslab_1", factor_version=1
    )

    values = [1.0, 2.0, 3.0]
    variants = {
        "perfectly_predictive": _observations(
            raw_return_5d=values, label_valid=[True, True, True]
        ),
        "perfectly_contradictory": _observations(
            raw_return_5d=[-value for value in values], label_valid=[True, True, True]
        ),
        "all_labels_invalid": _observations(
            raw_return_5d=values, label_valid=[False, False, False]
        ),
        # Non-finite labels matter separately from predictive ones: the finiteness check
        # is the one place a label column could change the verdict without changing any
        # metric, so a well-behaved label fixture alone would not detect that leak. Found
        # by mutation -- substituting the label into the finiteness check left every
        # finite-label variant identical and so passed unnoticed.
        "nonfinite_labels": _observations(
            raw_return_5d=[float("nan"), float("inf"), 1.0],
            label_valid=[True, True, True],
        ),
        "labels_absent_but_column_present": _observations(
            raw_return_5d=[None, None, None], label_valid=[False, False, False]
        ),
    }
    for name, observations in variants.items():
        snapshot = assess_factor_quality(
            _universe(), observations, factor_id="yeslab_1", factor_version=1
        )
        assert snapshot == baseline, f"{name} label column changed the structural verdict"


def test_structural_quality_reports_no_label_derived_metric() -> None:
    """A second, independent pin: the metric vocabulary itself must stay label-free."""
    snapshot = assess_factor_quality(
        _universe(),
        _observations(raw_return_5d=[1.0, 2.0, 3.0], label_valid=[True, True, True]),
        factor_id="yeslab_1",
        factor_version=1,
    )

    assert snapshot.metrics, "the snapshot should carry structural metrics"
    forbidden = ("label", "return", "rank_ic", "icir", "outcome")
    offending = sorted(
        name for name in snapshot.metrics if any(token in name.lower() for token in forbidden)
    )
    assert not offending, f"structural quality reported label-derived metrics: {offending}"


def test_an_empty_scope_reports_zero_coverage_and_an_explicit_status_never_nan() -> None:
    """C6.3: an empty date must yield 0 coverage and a definite status, not NaN read as PASS.

    NaN is the dangerous outcome because it is neither below nor above a threshold, so a
    comparison-based gate can treat it as passing and admit a factor that was never
    observed. Both routes to emptiness are covered: no observations for a non-empty
    universe, and an empty universe.
    """
    no_observations = assess_factor_quality(
        _universe(),
        _observations().iloc[0:0],
        factor_id="yeslab_1",
        factor_version=1,
    )
    empty_universe = assess_factor_quality(
        _universe().iloc[0:0],
        _observations(),
        factor_id="yeslab_1",
        factor_version=1,
    )

    for snapshot in (no_observations, empty_universe):
        assert snapshot.metrics["coverage"] == 0.0
        assert snapshot.metrics["finite_value_count"] == 0
        assert snapshot.status is not QualityStatus.PASS
        assert "LOW_COVERAGE" in snapshot.eligibility_failures
        for name, value in snapshot.metrics.items():
            assert value == value, f"metric {name} is NaN"


# --- C6.1: the metrics the plan names but the snapshot did not emit --------------

_DATES = (date(2025, 1, 2), date(2025, 1, 3), date(2025, 1, 6))
_CODES = ("000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ")


def _scoped_universe() -> pd.DataFrame:
    return pd.DataFrame(
        [{"signal_date": day, "ts_code": code} for day in _DATES for code in _CODES]
    )


def _scoped_observations() -> pd.DataFrame:
    """Hand-computable scope: 12 universe cells, 8 usable, 1 stale, 1 empty date.

    2025-01-02 carries 0/2/4/6 (one zero, median 3, MAD 2), 2025-01-03 carries four 1s
    (zero dispersion), and 2025-01-06 has a single value that is 181 days stale, so the
    date has no usable observation at all. Pooled usable values are therefore
    [0, 1, 1, 1, 1, 2, 4, 6], whose median is the mean of the 4th and 5th order
    statistics -- both 1 -- i.e. 1.0, and whose MAD about that median is 0.5.
    """
    rows: list[dict[str, object]] = []
    for code, value in zip(_CODES, (0.0, 2.0, 4.0, 6.0), strict=True):
        rows.append(
            {"signal_date": _DATES[0], "ts_code": code, "value": value, "stale_days": 0}
        )
    for code in _CODES:
        rows.append(
            {"signal_date": _DATES[1], "ts_code": code, "value": 1.0, "stale_days": 0}
        )
    rows.append(
        {"signal_date": _DATES[2], "ts_code": _CODES[0], "value": 9.0, "stale_days": 181}
    )
    return pd.DataFrame(rows)


def _scoped_snapshot() -> FactorQualitySnapshot:
    return assess_factor_quality(
        _scoped_universe(), _scoped_observations(), factor_id="yeslab_1", factor_version=1
    )


def test_c61_distribution_metrics_are_pooled_over_usable_values() -> None:
    pooled = [0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 4.0, 6.0]

    metrics = _scoped_snapshot().metrics

    assert metrics["finite_value_count"] == 8
    assert metrics["universe_member_count"] == 12
    assert metrics["coverage"] == 8 / 12
    assert metrics["zero_ratio"] == 1 / 8
    assert metrics["value_median"] == 1.0
    assert metrics["value_mad"] == 0.5
    assert metrics["value_p50"] == metrics["value_median"]
    assert metrics["value_p01"] == float(np.percentile(pooled, 1))
    assert metrics["value_p99"] == float(np.percentile(pooled, 99))


def test_c61_per_day_dispersion_and_stale_share_are_reported() -> None:
    metrics = _scoped_snapshot().metrics

    # daily MADs are 2.0 on 2025-01-02 and 0.0 on 2025-01-03; the empty date contributes
    # no dispersion sample rather than a zero, which would understate the median.
    assert metrics["median_daily_dispersion"] == 1.0
    assert metrics["zero_dispersion_date_share"] == 1 / 3
    # one universe cell held a finite value that was too stale to use
    assert metrics["stale_share"] == 1 / 12


def test_c61_date_and_security_coverage_and_the_longest_missing_streak() -> None:
    metrics = _scoped_snapshot().metrics

    assert metrics["date_coverage"] == 2 / 3
    assert metrics["security_coverage"] == 1.0
    assert metrics["longest_missing_date_streak"] == 1


def test_c61_longest_missing_streak_counts_consecutive_empty_dates() -> None:
    """A streak, not a count: three separate empty days are worse than one run of three.

    This is the metric that says whether a factor disappears once or goes dark for a week,
    which is the difference between a data hiccup and an unusable factor.
    """
    observations = _scoped_observations()
    # make the second date empty too, so 2025-01-03 and 2025-01-06 form a run of two
    observations = observations.loc[observations["signal_date"] != _DATES[1]].copy()

    snapshot = assess_factor_quality(
        _scoped_universe(), observations, factor_id="yeslab_1", factor_version=1
    )

    assert snapshot.metrics["longest_missing_date_streak"] == 2
    assert snapshot.metrics["date_coverage"] == 1 / 3


def test_c61_duplicate_rate_is_a_ratio_and_not_only_a_boolean() -> None:
    """C6.1 asks for a duplicate rate; C6.2's blocker alone cannot say how widespread it is."""
    observations = _scoped_observations()
    clean_rate = assess_factor_quality(
        _scoped_universe(), observations, factor_id="yeslab_1", factor_version=1
    ).metrics["duplicate_observation_rate"]
    assert clean_rate == 0.0

    duplicated = pd.concat(
        [observations, observations.iloc[[0]]], ignore_index=True
    )
    snapshot = assess_factor_quality(
        _scoped_universe(), duplicated, factor_id="yeslab_1", factor_version=1
    )

    # 9 rows plus one repeated row is 10 rows over 9 distinct business keys.
    assert snapshot.metrics["duplicate_observation_rate"] == 1 / 10
    assert "DUPLICATE_BUSINESS_KEY" in snapshot.blockers
    assert snapshot.status is QualityStatus.BLOCKER
    # the duplicate must not inflate coverage: it is the same business key
    assert snapshot.metrics["coverage"] == 8 / 12


# --- C6.2: the three BLOCKER conditions that were missing -----------------------


def _obs(day: date, code: str, value: float, stale_days: int = 0) -> dict[str, object]:
    return {"signal_date": day, "ts_code": code, "value": value, "stale_days": stale_days}


def test_a_value_available_after_its_signal_date_is_a_pit_blocker() -> None:
    """stale_days is signal_date - available_date, so a negative value is a PIT violation.

    This is the one blocker that needs no extra input, and the most serious of the three:
    a factor usable before it existed would let a model train on the future, and no
    coverage or dispersion metric would reveal it.
    """
    observations = pd.DataFrame(
        [
            _obs(date(2025, 1, 2), "000001.SZ", 1.0, -1),
            _obs(date(2025, 1, 2), "000002.SZ", 2.0),
            _obs(date(2025, 1, 3), "000001.SZ", 3.0, 1),
        ]
    )

    snapshot = assess_factor_quality(
        _universe(), observations, factor_id="yeslab_1", factor_version=1
    )

    assert "PIT_VIOLATION" in snapshot.blockers
    assert snapshot.status is QualityStatus.BLOCKER
    # The violating row must not count as usable either, or the blocker would report a
    # problem while coverage quietly took credit for the row.
    assert snapshot.metrics["finite_value_count"] == 2
    assert snapshot.metrics["coverage"] == 2 / 6


def test_a_conflicted_key_that_reached_clean_is_a_blocker() -> None:
    """Deduplication quarantines conflicts upstream; this asserts it actually did.

    Relying on "the upstream step probably handled it" is exactly how a conflicting value
    ends up published: nothing downstream would notice, because a single surviving row
    looks perfectly clean.
    """
    observations = pd.DataFrame(
        [
            _obs(date(2025, 1, 2), "000001.SZ", 1.0),
            _obs(date(2025, 1, 2), "000002.SZ", 2.0),
        ]
    )

    snapshot = assess_factor_quality(
        _universe(),
        observations,
        factor_id="yeslab_1",
        factor_version=1,
        quarantined_keys={(date(2025, 1, 2), "000001.SZ")},
    )

    assert "CONFLICT_KEY_IN_CLEAN" in snapshot.blockers
    assert snapshot.status is QualityStatus.BLOCKER


# --- C6.1's snapshot key and the two remaining metrics ---------------------------


def test_a_snapshot_records_the_scope_and_as_of_date_it_describes() -> None:
    """Without these a snapshot cannot be interpreted, only compared to itself."""
    snapshot = assess_factor_quality(_universe(), _observations())

    assert snapshot.scope_id == SCOPE_ID
    assert snapshot.as_of_date == AS_OF
    assert snapshot.factor_id == "yeslab_1"
    assert snapshot.factor_version == 1


def test_an_empty_scope_is_rejected_rather_than_recorded_as_anonymous() -> None:
    with pytest.raises(ValueError, match="scope_id cannot be empty"):
        assess_with_explicit_scope(
            _universe(),
            _observations(),
            factor_id="yeslab_1",
            factor_version=1,
            scope_id="",
            as_of_date=AS_OF,
        )


def test_finite_ratio_describes_the_source_while_coverage_describes_the_universe() -> None:
    """The two answer different questions and must not be collapsed into one number.

    A factor can be entirely finite and still cover almost nothing, or cover the whole
    universe while half of what the source delivered was unusable. Only the pair says which.
    """
    observations = pd.DataFrame(
        [
            _obs(date(2025, 1, 2), "000001.SZ", 1.0),
            _obs(date(2025, 1, 2), "000002.SZ", float("inf")),
            _obs(date(2025, 1, 2), "000003.SZ", float("nan")),
        ]
    )

    metrics = assess_factor_quality(_universe(), observations).metrics

    # one finite value out of the two the source actually delivered (inf counts as delivered)
    assert metrics["finite_ratio"] == 0.5
    # ...but only one of the six universe cells is usable
    assert metrics["coverage"] == 1 / 6


def test_conflict_key_rate_is_emitted_only_when_it_was_actually_measured() -> None:
    """A rate of 0.0 and an unmeasured rate are different claims and must not look alike."""
    unmeasured = assess_factor_quality(_universe(), _observations())

    assert "conflict_key_rate" not in unmeasured.metrics
    assert "CONFLICT_KEY_IN_CLEAN" in unmeasured.unchecked_blockers

    measured = assess_factor_quality(
        _universe(),
        _observations(),
        quarantined_keys={(date(2025, 1, 3), "000002.SZ")},
        known_factor_versions={1},
    )

    assert measured.metrics["conflict_key_rate"] == 1 / 3
    assert measured.unchecked_blockers == ()


def test_a_quarantined_key_absent_from_clean_is_not_a_blocker() -> None:
    observations = pd.DataFrame(
        [
            _obs(date(2025, 1, 2), "000001.SZ", 1.0),
        ]
    )

    snapshot = assess_factor_quality(
        _universe(),
        observations,
        factor_id="yeslab_1",
        factor_version=1,
        quarantined_keys={(date(2025, 1, 3), "000002.SZ")},
        known_factor_versions={1},
    )

    assert snapshot.blockers == ()
    assert snapshot.unchecked_blockers == ()


def test_an_unknown_registry_version_is_a_blocker() -> None:
    observations = pd.DataFrame(
        [
            _obs(date(2025, 1, 2), "000001.SZ", 1.0),
        ]
    )

    snapshot = assess_factor_quality(
        _universe(),
        observations,
        factor_id="yeslab_1",
        factor_version=7,
        known_factor_versions={1, 2},
    )

    assert "UNKNOWN_REGISTRY_VERSION" in snapshot.blockers
    assert snapshot.status is QualityStatus.BLOCKER


def test_checks_that_cannot_run_are_recorded_instead_of_silently_skipped() -> None:
    """Two of the six blockers need inputs the caller may not have.

    Skipping them quietly would reproduce the failure this whole plan keeps guarding
    against: a gate that passes because it never looked. Recording them lets a caller
    decide whether an unperformed check is acceptable, and makes the omission visible in
    the archived snapshot.
    """
    snapshot = assess_factor_quality(
        _universe(), _scoped_observations(), factor_id="yeslab_1", factor_version=1
    )

    assert set(snapshot.unchecked_blockers) == {
        "CONFLICT_KEY_IN_CLEAN",
        "UNKNOWN_REGISTRY_VERSION",
    }
    assert snapshot.blockers == ()


def test_supplying_both_inputs_clears_the_unchecked_record() -> None:
    snapshot = assess_factor_quality(
        _universe(),
        _scoped_observations(),
        factor_id="yeslab_1",
        factor_version=1,
        quarantined_keys=set(),
        known_factor_versions={1},
    )

    assert snapshot.unchecked_blockers == ()
    assert snapshot.blockers == ()


def test_a_leaked_key_is_caught_across_date_and_timestamp_representations() -> None:
    """The two sides must be normalised, or a type mismatch reads as "no overlap".

    Assembled frames in this codebase carry ``datetime.date`` while other paths carry
    ``pandas.Timestamp``, and the failure direction is the silent one: unequal key types
    never intersect, so a conflicted value that reached clean would be reported as clean.
    """
    observations = pd.DataFrame([_obs(date(2025, 1, 2), "000001.SZ", 1.0)])

    snapshot = assess_factor_quality(
        _universe(),
        observations,
        factor_id="yeslab_1",
        factor_version=1,
        quarantined_keys={(pd.Timestamp("2025-01-02"), "000001.SZ")},
        known_factor_versions={1},
    )

    assert "CONFLICT_KEY_IN_CLEAN" in snapshot.blockers
    assert snapshot.status is QualityStatus.BLOCKER
