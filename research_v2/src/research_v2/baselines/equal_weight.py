"""Equal-weight factor-score baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pandas as pd

from research_v2.baselines.base import BaselineDataset, FittedBaseline, PredictionFrame
from research_v2.baselines.rank_ensemble import directional_rank_scores


@dataclass(frozen=True)
class FittedEqualWeightBaseline:
    """Frozen B1 factor directions and order."""

    factor_order: tuple[str, ...]
    directions: tuple[int, ...]
    minimum_factor_coverage: float = 0.60
    model_id: str = "B1"

    def predict(self, data: BaselineDataset) -> PredictionFrame:
        _validate_feature_columns(data, self.factor_order)
        scores = directional_rank_scores(
            data.features,
            self.factor_order,
            self.directions,
            minimum_factor_coverage=self.minimum_factor_coverage,
        )
        return PredictionFrame.from_scores(
            data.features,
            scores,
            model_id=self.model_id,
            fold_id=data.fold_id,
            dataset_id=data.dataset_id,
        )


@dataclass(frozen=True)
class EqualWeightBaseline:
    """B1: signed equal-weight factor ranks with a per-row coverage floor."""

    factor_order: tuple[str, ...]
    frozen_directions: dict[str, int]
    minimum_factor_coverage: float = 0.60
    model_id: str = "B1"

    def fit(
        self,
        train: BaselineDataset,
        validation: BaselineDataset,
    ) -> FittedBaseline:
        if train.role != "TRAIN" or validation.role != "VALIDATION":
            raise PermissionError("equal-weight baseline accepts only TRAIN and VALIDATION data")
        if train.fold_id != validation.fold_id:
            raise ValueError("train and validation datasets must belong to one fold")
        if not 0 < self.minimum_factor_coverage <= 1:
            raise ValueError("minimum_factor_coverage must be in (0, 1]")
        _validate_feature_columns(train, self.factor_order)
        _validate_feature_columns(validation, self.factor_order)
        if (
            tuple(self.frozen_directions) != tuple(self.factor_order)
            or any(self.frozen_directions[name] not in {-1, 1} for name in self.factor_order)
        ):
            raise ValueError("frozen directions must align with factor_order")
        return FittedEqualWeightBaseline(
            factor_order=self.factor_order,
            directions=tuple(self.frozen_directions[name] for name in self.factor_order),
            minimum_factor_coverage=self.minimum_factor_coverage,
            model_id=self.model_id,
        )


def _validate_feature_columns(data: BaselineDataset, factor_order: tuple[str, ...]) -> None:
    if not factor_order or len(set(factor_order)) != len(factor_order):
        raise ValueError("factor_order must be non-empty and unique")
    if tuple(data.features.columns) != ("signal_date", "ts_code", *factor_order):
        raise ValueError("baseline feature columns must exactly match factor_order")

def equal_weight_scores(
    features: pd.DataFrame,
    factor_order: tuple[str, ...],
    *,
    minimum_factor_coverage: float = 0.60,
) -> pd.Series:
    """Average finite feature values subject to the B1 row coverage floor."""
    if tuple(features.columns) != factor_order:
        raise ValueError("feature columns must exactly match factor_order")
    if not 0 < minimum_factor_coverage <= 1:
        raise ValueError("minimum_factor_coverage must be in (0, 1]")
    numeric = features.replace([float("inf"), float("-inf")], float("nan")).apply(
        pd.to_numeric,
        errors="coerce",
    )
    minimum_valid_factor_count = int(np.ceil(minimum_factor_coverage * len(factor_order)))
    valid_factor_count = numeric.notna().sum(axis=1)
    return cast(
        "pd.Series[Any]",
        numeric.mean(axis=1, skipna=True).where(valid_factor_count >= minimum_valid_factor_count),
    )
