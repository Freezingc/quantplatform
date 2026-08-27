from __future__ import annotations

from datetime import date
from pathlib import Path

import pytest
import yaml

from research_v2.monitoring.drift import (
    DailyDriftStatus,
    DriftReviewApprovalStore,
    DriftReviewStatus,
    DriftSeverity,
    evaluate_drift_review,
    population_stability_index,
)
from research_v2.monitoring.policy import load_monitoring_policy


def test_psi_is_zero_for_identical_distributions_and_escalates_for_shift() -> None:
    unchanged = population_stability_index((0.5, 0.5), (0.5, 0.5))
    shifted = population_stability_index((0.99, 0.01), (0.01, 0.99))

    assert unchanged.value == pytest.approx(0.0)
    assert unchanged.severity is DriftSeverity.INFO
    assert shifted.value > 0.25
    assert shifted.severity is DriftSeverity.ERROR


def test_review_requires_a_signed_approval_after_three_psi_error_days(tmp_path: Path) -> None:
    trading_dates = tuple(date(2026, 8, day) for day in (25, 26, 27, 28))
    review = evaluate_drift_review(
        (
            DailyDriftStatus(date(2026, 8, 25), True),
            DailyDriftStatus(date(2026, 8, 26), True),
            DailyDriftStatus(date(2026, 8, 27), True),
        ),
        trading_dates=trading_dates,
    )
    still_blocked = evaluate_drift_review(
        (
            DailyDriftStatus(date(2026, 8, 25), True),
            DailyDriftStatus(date(2026, 8, 26), True),
            DailyDriftStatus(date(2026, 8, 27), True),
            DailyDriftStatus(date(2026, 8, 28), False),
        ),
        trading_dates=trading_dates,
    )
    approvals = DriftReviewApprovalStore(tmp_path / "approvals", signing_key=b"review-key")
    approval = approvals.issue(
        review_required_since=date(2026, 8, 27),
        actor="risk-approver",
        evidence_hash="a" * 64,
        acknowledged_on=date(2026, 8, 28),
    )
    acknowledged = evaluate_drift_review(
        (
            DailyDriftStatus(date(2026, 8, 25), True),
            DailyDriftStatus(date(2026, 8, 26), True),
            DailyDriftStatus(date(2026, 8, 27), True),
            DailyDriftStatus(date(2026, 8, 28), False),
        ),
        trading_dates=trading_dates,
        approval=approval,
        approval_store=approvals,
    )

    assert review.status is DriftReviewStatus.REVIEW_REQUIRED
    assert review.consecutive_error_days == 3
    assert still_blocked.status is DriftReviewStatus.REVIEW_REQUIRED
    assert acknowledged.status is DriftReviewStatus.CLEAR


def test_drift_review_rejects_an_unverified_approval() -> None:
    with pytest.raises(ValueError, match="approval_store"):
        evaluate_drift_review(
            (
                DailyDriftStatus(date(2026, 8, 25), True),
                DailyDriftStatus(date(2026, 8, 26), True),
                DailyDriftStatus(date(2026, 8, 27), True),
            ),
            trading_dates=(date(2026, 8, 25), date(2026, 8, 26), date(2026, 8, 27)),
            approval=object(),  # type: ignore[arg-type]
        )


def test_drift_review_rejects_an_approval_signed_with_a_different_key(tmp_path: Path) -> None:
    forged = DriftReviewApprovalStore(tmp_path / "forged", signing_key=b"untrusted-key").issue(
        review_required_since=date(2026, 8, 27),
        actor="risk-approver",
        evidence_hash="a" * 64,
        acknowledged_on=date(2026, 8, 27),
    )
    trusted = DriftReviewApprovalStore(tmp_path / "forged", signing_key=b"trusted-key")

    with pytest.raises(ValueError, match="signature"):
        evaluate_drift_review(
            (
                DailyDriftStatus(date(2026, 8, 25), True),
                DailyDriftStatus(date(2026, 8, 26), True),
                DailyDriftStatus(date(2026, 8, 27), True),
            ),
            trading_dates=(date(2026, 8, 25), date(2026, 8, 26), date(2026, 8, 27)),
            approval=forged,
            approval_store=trusted,
        )


def test_drift_review_rejects_missing_trading_day_observations() -> None:
    with pytest.raises(ValueError, match="contiguous"):
        evaluate_drift_review(
            (
                DailyDriftStatus(date(2026, 8, 25), True),
                DailyDriftStatus(date(2026, 8, 27), True),
            ),
            trading_dates=(date(2026, 8, 25), date(2026, 8, 26), date(2026, 8, 27)),
        )


def test_loaded_monitoring_policy_changes_psi_and_drift_review_thresholds(tmp_path: Path) -> None:
    path = tmp_path / "monitoring.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "psi": {"epsilon": 1e-6, "info_upper_bound": 0.50, "warn_upper_bound": 0.75},
                "feature_health": {
                    "minimum_row_coverage": 0.90,
                    "universe_size_lower_ratio": 0.80,
                    "universe_size_upper_ratio": 1.20,
                    "maximum_missing_rate_increase": 0.10,
                },
                "drift_review": {"consecutive_psi_error_days": 2},
            }
        ),
        encoding="utf-8",
    )
    policy = load_monitoring_policy(path)

    psi = population_stability_index((0.9, 0.1), (0.65, 0.35), policy=policy)
    review = evaluate_drift_review(
        (
            DailyDriftStatus(date(2026, 8, 26), True),
            DailyDriftStatus(date(2026, 8, 27), True),
        ),
        trading_dates=(date(2026, 8, 26), date(2026, 8, 27)),
        policy=policy,
    )

    assert psi.severity is DriftSeverity.INFO
    assert review.status is DriftReviewStatus.REVIEW_REQUIRED


def test_monitoring_policy_rejects_boolean_or_fractional_consecutive_day_thresholds() -> None:
    from research_v2.monitoring.policy import MonitoringPolicy

    with pytest.raises(ValueError, match="consecutive"):
        MonitoringPolicy(consecutive_psi_error_days=True)
    with pytest.raises(ValueError, match="consecutive"):
        MonitoringPolicy(consecutive_psi_error_days=1.5)  # type: ignore[arg-type]
