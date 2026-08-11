"""Label-free 60-day shadow-operation qualification accounting."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from enum import StrEnum


class ShadowQualificationStatus(StrEnum):
    INCOMPLETE = "INCOMPLETE"
    READY_FOR_REVIEW = "READY_FOR_REVIEW"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"


@dataclass(frozen=True)
class ShadowDay:
    signal_date: date
    is_trading_day: bool
    pipeline_success: bool
    prediction_published: bool
    has_blocker: bool
    report_verified: bool
    psi_error: bool

    @property
    def is_valid_shadow_day(self) -> bool:
        return (
            self.is_trading_day
            and self.pipeline_success
            and self.prediction_published
            and not self.has_blocker
            and self.report_verified
        )


@dataclass(frozen=True)
class ShadowQualification:
    valid_day_count: int
    status: ShadowQualificationStatus
    max_consecutive_psi_errors: int


def evaluate_shadow_qualification(days: tuple[ShadowDay, ...]) -> ShadowQualification:
    """Count only verified healthy days; never infer performance from labels or returns."""
    dates = tuple(day.signal_date for day in days)
    if dates != tuple(sorted(dates)) or len(set(dates)) != len(dates):
        raise ValueError("shadow days must have unique ascending signal dates")
    valid_day_count = sum(day.is_valid_shadow_day for day in days)
    maximum = 0
    current = 0
    for day in days:
        if day.is_valid_shadow_day and day.psi_error:
            current += 1
            maximum = max(maximum, current)
        else:
            current = 0
    if maximum >= 3:
        status = ShadowQualificationStatus.REVIEW_REQUIRED
    elif valid_day_count >= 60:
        status = ShadowQualificationStatus.READY_FOR_REVIEW
    else:
        status = ShadowQualificationStatus.INCOMPLETE
    return ShadowQualification(valid_day_count, status, maximum)
