from __future__ import annotations

from datetime import date

import pandas as pd

from research_v2.factors.quality import QualityStatus, assess_factor_quality


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 3 + [date(2025, 1, 3)] * 3,
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"] * 2,
        }
    )


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
