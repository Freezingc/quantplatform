from __future__ import annotations

from datetime import date

import pandas as pd

from research_v2.monitoring.data_health import (
    FeatureReference,
    evaluate_feature_data_health,
)
from research_v2.monitoring.drift import DriftSeverity
from research_v2.monitoring.policy import MonitoringPolicy


def _reference() -> FeatureReference:
    return FeatureReference(
        feature_name="quality",
        bin_edges=(0.0, 0.5, 1.0),
        reference_bin_masses=(0.5, 0.5),
        reference_missing_rate=0.0,
        minimum_finite_coverage=0.95,
    )


def _frame(signal_date: date) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": [signal_date] * 20,
            "ts_code": [f"S{index:03d}" for index in range(20)],
            "quality": [0.1] * 10 + [0.9] * 10,
            "available_date": [signal_date] * 20,
        }
    )


def test_feature_data_health_blocks_missing_keys_future_data_and_low_row_coverage() -> None:
    signal_date = date(2026, 8, 27)
    healthy = _frame(signal_date)

    report = evaluate_feature_data_health(
        healthy,
        signal_date=signal_date,
        references=(_reference(),),
        expected_universe_size=20,
        recent_universe_size_median=20,
    )

    assert not report.blocking_reasons
    assert report.features[0].psi.severity is DriftSeverity.INFO

    malformed = healthy.drop(columns=["quality"]).iloc[:18].copy()
    malformed.loc[0, "available_date"] = date(2026, 8, 28)
    malformed.loc[1, "ts_code"] = malformed.loc[0, "ts_code"]
    blocked = evaluate_feature_data_health(
        malformed,
        signal_date=signal_date,
        references=(_reference(),),
        expected_universe_size=20,
        recent_universe_size_median=20,
    )

    assert set(blocked.blocking_reasons) == {
        "DUPLICATE_BUSINESS_KEY",
        "FUTURE_AVAILABLE_DATE",
        "INSUFFICIENT_ROW_COVERAGE",
        "MISSING_REQUIRED_FEATURE:quality",
    }


def test_feature_data_health_reports_factor_coverage_and_psi_without_blocking() -> None:
    signal_date = date(2026, 8, 27)
    frame = _frame(signal_date)
    frame.loc[:2, "quality"] = None
    frame.loc[3:, "quality"] = 0.99

    report = evaluate_feature_data_health(
        frame,
        signal_date=signal_date,
        references=(_reference(),),
        expected_universe_size=20,
        recent_universe_size_median=20,
    )

    assert not report.blocking_reasons
    assert report.features[0].finite_coverage == 0.85
    assert report.features[0].psi.severity is DriftSeverity.ERROR
    assert report.investigation_alerts == (
        "LOW_FACTOR_COVERAGE:quality",
        "MISSING_RATE_INCREASE:quality",
        "PSI_ERROR:quality",
    )


def test_feature_data_health_counts_values_outside_reference_bins_as_tail_drift() -> None:
    signal_date = date(2026, 8, 27)
    frame = _frame(signal_date).assign(quality=2.0)

    report = evaluate_feature_data_health(
        frame,
        signal_date=signal_date,
        references=(_reference(),),
        expected_universe_size=20,
        recent_universe_size_median=20,
    )

    assert report.features[0].psi.severity is DriftSeverity.ERROR


def test_feature_data_health_blocks_missing_available_dates_and_blank_business_keys() -> None:
    signal_date = date(2026, 8, 27)
    frame = _frame(signal_date)
    frame.loc[0, "available_date"] = None
    frame.loc[1, "ts_code"] = " "

    report = evaluate_feature_data_health(
        frame,
        signal_date=signal_date,
        references=(_reference(),),
        expected_universe_size=20,
        recent_universe_size_median=20,
    )

    assert {"MISSING_AVAILABLE_DATE", "INVALID_BUSINESS_KEY"}.issubset(
        report.blocking_reasons
    )


def test_feature_data_health_uses_loaded_policy_thresholds() -> None:
    signal_date = date(2026, 8, 27)
    report = evaluate_feature_data_health(
        _frame(signal_date).iloc[:18],
        signal_date=signal_date,
        references=(_reference(),),
        expected_universe_size=20,
        recent_universe_size_median=20,
        policy=MonitoringPolicy(minimum_row_coverage=0.90),
    )

    assert not report.blocking_reasons
