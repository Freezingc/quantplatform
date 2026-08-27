"""Validation-only selection for the simplest frozen-direction factor baseline."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from research_v2.baselines.base import BaselineDataset, FittedBaseline, PredictionFrame
from research_v2.evaluation.rank_ic import compute_daily_rank_ic, summarize_rank_ic


@dataclass(frozen=True)
class SingleFactorSelection:
    factor_name: str
    frozen_direction: int
    validation_mean_rank_ic: float


@dataclass(frozen=True)
class FittedSingleFactorBaseline:
    """One factor and direction selected before the outer test is available."""

    factor_name: str
    frozen_direction: int
    model_id: str = "B0"

    def predict(self, data: BaselineDataset) -> PredictionFrame:
        if self.factor_name not in data.features.columns:
            raise ValueError(f"baseline features are missing selected factor: {self.factor_name}")
        scores = pd.to_numeric(data.features[self.factor_name], errors="coerce")
        scores = scores.replace([float("inf"), float("-inf")], float("nan"))
        return PredictionFrame.from_scores(
            data.features,
            scores * self.frozen_direction,
            model_id=self.model_id,
            fold_id=data.fold_id,
            dataset_id=data.dataset_id,
        )


@dataclass(frozen=True)
class SingleFactorBaseline:
    """B0: select the strongest factor on validation with train-only tie-break evidence."""

    factor_order: tuple[str, ...]
    frozen_directions: dict[str, int]
    minimum_cross_section: int = 30
    model_id: str = "B0"

    def fit(
        self,
        train: BaselineDataset,
        validation: BaselineDataset,
    ) -> FittedBaseline:
        _validate_baseline_inputs(
            train,
            validation,
            self.factor_order,
            self.frozen_directions,
            self.minimum_cross_section,
        )
        candidates: list[tuple[float, float, float, str]] = []
        for factor_name in self.factor_order:
            direction = self.frozen_directions[factor_name]
            train_daily = _factor_daily_rank_ic(
                train,
                factor_name,
                direction,
                minimum_cross_section=self.minimum_cross_section,
            )
            validation_daily = _factor_daily_rank_ic(
                validation,
                factor_name,
                direction,
                minimum_cross_section=self.minimum_cross_section,
            )
            train_summary = summarize_rank_ic(train_daily)
            validation_summary = summarize_rank_ic(validation_daily)
            validation_mean = validation_summary.mean_rank_ic
            if not math.isfinite(validation_mean):
                continue
            train_mean = train_summary.mean_rank_ic
            train_coverage = _mean_coverage(train_daily)
            candidates.append(
                (
                    -validation_mean,
                    -train_mean if math.isfinite(train_mean) else math.inf,
                    -train_coverage,
                    factor_name,
                )
            )
        if not candidates:
            raise ValueError("no factor has enough valid validation RankIC observations")
        selected_factor = min(candidates)[3]
        return FittedSingleFactorBaseline(
            factor_name=selected_factor,
            frozen_direction=self.frozen_directions[selected_factor],
            model_id=self.model_id,
        )


def select_single_factor(
    validation_features: pd.DataFrame,
    validation_labels: pd.DataFrame,
    *,
    frozen_directions: dict[str, int],
    train_features: pd.DataFrame | None = None,
    train_labels: pd.DataFrame | None = None,
    minimum_cross_section: int = 30,
) -> SingleFactorSelection:
    """Choose one factor on validation with optional train-only tie-break evidence.

    frozen_directions must be supplied by the upstream Train selection stage.
    Optional Train frames are used only after a Validation tie and never change
    the selected direction.
    """
    required_keys = {"signal_date", "ts_code"}
    if missing := sorted(required_keys.difference(validation_features.columns)):
        raise ValueError(f"validation features are missing columns: {missing}")
    if not frozen_directions:
        raise ValueError("at least one frozen factor direction is required")
    train_tie_break: dict[str, tuple[float, float]] = {}
    if (train_features is None) != (train_labels is None):
        raise ValueError("train_features and train_labels must be supplied together")
    train_frame: BaselineDataset | None = None
    if train_features is not None and train_labels is not None:
        train_frame = BaselineDataset(
            role="TRAIN",
            fold_id="single-factor",
            dataset_id="single-factor-train",
            features=train_features,
            labels=train_labels,
        )
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
            if train_frame is not None:
                train_daily = _factor_daily_rank_ic(
                    train_frame,
                    factor_name,
                    direction,
                    minimum_cross_section=minimum_cross_section,
                )
                train_summary = summarize_rank_ic(train_daily)
                train_mean = (
                    train_summary.mean_rank_ic
                    if math.isfinite(train_summary.mean_rank_ic)
                    else float("-inf")
                )
                train_tie_break[factor_name] = (train_mean, _mean_coverage(train_daily))
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
        key=lambda candidate: (
            -candidate.validation_mean_rank_ic,
            -train_tie_break.get(candidate.factor_name, (float("-inf"), 0.0))[0],
            -train_tie_break.get(candidate.factor_name, (float("-inf"), 0.0))[1],
            candidate.factor_name,
        ),
    )


def _factor_daily_rank_ic(
    dataset: BaselineDataset,
    factor_name: str,
    direction: int,
    *,
    minimum_cross_section: int,
) -> pd.DataFrame:
    predictions = dataset.features.loc[:, ["signal_date", "ts_code"]].copy()
    predictions["score"] = (
        pd.to_numeric(dataset.features[factor_name], errors="coerce")
        .replace([float("inf"), float("-inf")], float("nan"))
        * direction
    )
    return compute_daily_rank_ic(
        predictions,
        dataset.labels,
        minimum_cross_section=minimum_cross_section,
        include_invalid=True,
        require_score_health=False,
    )


def _mean_coverage(daily: pd.DataFrame) -> float:
    if daily.empty or "coverage" not in daily.columns:
        return 0.0
    coverage = pd.to_numeric(daily["coverage"], errors="coerce")
    return float(coverage.mean()) if coverage.notna().any() else 0.0


def _validate_baseline_inputs(
    train: BaselineDataset,
    validation: BaselineDataset,
    factor_order: tuple[str, ...],
    frozen_directions: dict[str, int],
    minimum_cross_section: int,
) -> None:
    if train.role != "TRAIN" or validation.role != "VALIDATION":
        raise PermissionError("single-factor baseline accepts only TRAIN and VALIDATION data")
    if train.fold_id != validation.fold_id:
        raise ValueError("train and validation datasets must belong to one fold")
    if (
        minimum_cross_section < 2
        or not factor_order
        or len(set(factor_order)) != len(factor_order)
        or tuple(frozen_directions) != tuple(factor_order)
        or any(frozen_directions[name] not in {-1, 1} for name in factor_order)
    ):
        raise ValueError("factor order, directions, or minimum cross-section is invalid")
    expected = ("signal_date", "ts_code", *factor_order)
    if tuple(train.features.columns) != expected or tuple(validation.features.columns) != expected:
        raise ValueError("baseline feature columns must exactly match factor_order")
