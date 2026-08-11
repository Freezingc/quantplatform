"""Validation-only selection for the simplest frozen-direction factor baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from research_v2.evaluation.rank_ic import compute_daily_rank_ic, summarize_rank_ic


@dataclass(frozen=True)
class SingleFactorSelection:
    factor_name: str
    frozen_direction: int
    validation_mean_rank_ic: float


def select_single_factor(
    validation_features: pd.DataFrame,
    validation_labels: pd.DataFrame,
    *,
    frozen_directions: dict[str, int],
    minimum_cross_section: int = 30,
) -> SingleFactorSelection:
    """Choose one factor using validation only; output direction is never re-estimated."""
    required_keys = {"signal_date", "ts_code"}
    if missing := sorted(required_keys.difference(validation_features.columns)):
        raise ValueError(f"validation features are missing columns: {missing}")
    if not frozen_directions:
        raise ValueError("at least one frozen factor direction is required")
    candidates: list[SingleFactorSelection] = []
    for factor_name, direction in sorted(frozen_directions.items()):
        if factor_name not in validation_features.columns or direction not in {-1, 1}:
            raise ValueError("validation factor columns or frozen directions are invalid")
        predictions = validation_features.loc[:, ["signal_date", "ts_code"]].copy()
        predictions["score"] = pd.to_numeric(
            validation_features[factor_name],
            errors="coerce",
        ) * direction
        daily = compute_daily_rank_ic(
            predictions,
            validation_labels,
            minimum_cross_section=minimum_cross_section,
        )
        summary = summarize_rank_ic(daily)
        if math.isfinite(summary.mean_rank_ic):
            candidates.append(
                SingleFactorSelection(
                    factor_name=factor_name,
                    frozen_direction=direction,
                    validation_mean_rank_ic=summary.mean_rank_ic,
                )
            )
    if not candidates:
        raise ValueError("no factor has enough valid validation RankIC observations")
    return min(
        candidates,
        key=lambda candidate: (-candidate.validation_mean_rank_ic, candidate.factor_name),
    )
