"""Population stability index calculations for non-blocking inference drift monitoring."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum


class DriftSeverity(StrEnum):
    INFO = "INFO"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass(frozen=True)
class PsiResult:
    value: float
    severity: DriftSeverity


def population_stability_index(
    reference_bins: Sequence[float],
    current_bins: Sequence[float],
    *,
    epsilon: float = 1e-6,
) -> PsiResult:
    """Calculate PSI from bin masses with epsilon smoothing and normalized distributions."""
    if len(reference_bins) != len(current_bins) or not reference_bins or epsilon <= 0:
        raise ValueError("PSI inputs must have equal non-empty bins and positive epsilon")
    reference = _normalize(reference_bins, epsilon)
    current = _normalize(current_bins, epsilon)
    value = sum(
        (actual - expected) * math.log(actual / expected)
        for expected, actual in zip(reference, current, strict=True)
    )
    if value < 0.10:
        severity = DriftSeverity.INFO
    elif value <= 0.25:
        severity = DriftSeverity.WARN
    else:
        severity = DriftSeverity.ERROR
    return PsiResult(value=float(value), severity=severity)


def _normalize(values: Sequence[float], epsilon: float) -> tuple[float, ...]:
    if any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("PSI bin masses must be finite and non-negative")
    smoothed = [float(value) + epsilon for value in values]
    total = sum(smoothed)
    return tuple(value / total for value in smoothed)
