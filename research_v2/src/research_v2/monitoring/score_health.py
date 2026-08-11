"""Blocking health checks for label-free daily ensemble score publication."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ScoreHealthReport:
    finite_coverage: float
    unique_score_count: int
    score_std: float
    blocking_reasons: tuple[str, ...]


def evaluate_score_health(
    predictions: pd.DataFrame,
    *,
    expected_key_count: int,
) -> ScoreHealthReport:
    """Reject incomplete, duplicate, non-finite, or collapsed score cross-sections."""
    if expected_key_count < 1:
        raise ValueError("expected_key_count must be positive")
    required = {"signal_date", "ts_code", "ensemble_score"}
    if missing := sorted(required.difference(predictions.columns)):
        raise ValueError(f"predictions are missing columns: {missing}")
    keys = ["signal_date", "ts_code"]
    reasons: list[str] = []
    if predictions.duplicated(keys).any():
        reasons.append("DUPLICATE_BUSINESS_KEY")
    numeric_scores = pd.to_numeric(predictions["ensemble_score"], errors="coerce")
    finite = np.isfinite(numeric_scores)
    finite_coverage = float(finite.sum() / expected_key_count)
    if finite_coverage < 0.95:
        reasons.append("INSUFFICIENT_FINITE_SCORE_COVERAGE")
    finite_scores = numeric_scores.loc[finite]
    unique_count = int(finite_scores.nunique())
    if unique_count < 10:
        reasons.append("INSUFFICIENT_UNIQUE_SCORES")
    score_std = float(finite_scores.std(ddof=0)) if not finite_scores.empty else float("nan")
    if not np.isfinite(score_std) or score_std <= 1e-6:
        reasons.append("SCORE_STD_COLLAPSE")
    return ScoreHealthReport(
        finite_coverage=finite_coverage,
        unique_score_count=unique_count,
        score_std=score_std,
        blocking_reasons=tuple(reasons),
    )
