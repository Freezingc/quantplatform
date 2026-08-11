from __future__ import annotations

from datetime import date, timedelta

from research_v2.monitoring.shadow import (
    ShadowDay,
    ShadowQualificationStatus,
    evaluate_shadow_qualification,
)


def test_shadow_qualification_requires_sixty_distinct_healthy_trading_days() -> None:
    start = date(2025, 1, 2)
    days = tuple(
        ShadowDay(
            signal_date=start + timedelta(days=index),
            is_trading_day=True,
            pipeline_success=True,
            prediction_published=True,
            has_blocker=False,
            report_verified=True,
            psi_error=False,
        )
        for index in range(60)
    )

    ready = evaluate_shadow_qualification(days)
    assert ready.valid_day_count == 60
    assert ready.status is ShadowQualificationStatus.READY_FOR_REVIEW
    incomplete = evaluate_shadow_qualification(days[:-1])
    assert incomplete.status is ShadowQualificationStatus.INCOMPLETE


def test_three_consecutive_psi_errors_require_review_even_at_sixty_days() -> None:
    start = date(2025, 1, 2)
    days = tuple(
        ShadowDay(
            signal_date=start + timedelta(days=index),
            is_trading_day=True,
            pipeline_success=True,
            prediction_published=True,
            has_blocker=False,
            report_verified=True,
            psi_error=index in {10, 11, 12},
        )
        for index in range(60)
    )

    assert evaluate_shadow_qualification(days).status is ShadowQualificationStatus.REVIEW_REQUIRED
