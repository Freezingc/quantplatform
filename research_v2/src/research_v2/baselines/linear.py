"""Train-only date-equal Ridge baseline with frozen factor order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Ridge  # type: ignore[import-untyped]

from research_v2.baselines.base import BaselineDataset, FittedBaseline, PredictionFrame
from research_v2.baselines.cross_validation import expanding_inner_folds
from research_v2.evaluation.rank_ic import compute_daily_rank_ic, summarize_rank_ic

FitRole = Literal["TRAIN", "VALIDATION", "TEST", "LOCKED"]
LINEAR_ALPHA_GRID = (0.01, 0.1, 1.0, 10.0, 100.0)
LINEAR_L1_RATIO_GRID = (0.0, 0.1, 0.5, 0.9, 1.0)


@dataclass(frozen=True)
class LinearHyperparameters:
    alpha: float
    l1_ratio: float

    def __post_init__(self) -> None:
        if self.alpha <= 0 or not np.isfinite(self.alpha):
            raise ValueError("linear alpha must be finite and positive")
        if not 0 <= self.l1_ratio <= 1 or not np.isfinite(self.l1_ratio):
            raise ValueError("linear l1_ratio must be finite and in [0, 1]")


@dataclass(frozen=True)
class LinearRidgeBaseline:
    factor_order: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    alpha: float
    l1_ratio: float = 0.0

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        *,
        factor_order: tuple[str, ...],
        role: FitRole,
        alpha: float = 1.0,
        l1_ratio: float = 0.0,
    ) -> LinearRidgeBaseline:
        """Fit only on TRAIN rows with date-equal demeaned raw-return targets."""
        if role != "TRAIN":
            raise PermissionError("LinearRidgeBaseline.fit is permitted only for TRAIN data")
        if (
            alpha < 0
            or not np.isfinite(alpha)
            or not 0 <= l1_ratio <= 1
            or not np.isfinite(l1_ratio)
            or (l1_ratio > 0 and alpha <= 0)
            or not factor_order
            or len(set(factor_order)) != len(factor_order)
        ):
            raise ValueError("alpha and factor_order are invalid")
        _validate_feature_schema(features, factor_order)
        required_labels = {"signal_date", "ts_code", "raw_return_5d", "label_valid"}
        if missing := sorted(required_labels.difference(labels.columns)):
            raise ValueError(f"labels are missing columns: {missing}")
        keys = ["signal_date", "ts_code"]
        if features.duplicated(keys).any() or labels.duplicated(keys).any():
            raise ValueError("features and labels must have unique business keys")
        joined = features.merge(
            labels.loc[:, keys + ["raw_return_5d", "label_valid"]],
            on=keys,
            how="inner",
            validate="one_to_one",
        )
        joined["label_valid"] = joined["label_valid"].fillna(False).astype(bool)
        joined = joined.loc[joined["label_valid"]].copy()
        values = _finite_matrix(joined, factor_order)
        raw_target = pd.to_numeric(joined["raw_return_5d"], errors="coerce")
        if raw_target.empty or not np.isfinite(raw_target.to_numpy(dtype=float)).all():
            raise ValueError("TRAIN labels must be finite and non-empty")
        target = (
            raw_target
            - raw_target.groupby(joined["signal_date"], sort=False).transform("mean")
        ).to_numpy(dtype=float)
        counts = joined.groupby("signal_date", sort=False)["ts_code"].transform("count")
        weights = 1.0 / counts.to_numpy(dtype=float)
        if l1_ratio == 0:
            model = Ridge(alpha=alpha, fit_intercept=True)
        else:
            model = ElasticNet(
                alpha=alpha,
                l1_ratio=l1_ratio,
                fit_intercept=True,
                max_iter=10_000,
                tol=1e-7,
                selection="cyclic",
                random_state=20260801,
            )
        model.fit(values, target, sample_weight=weights)
        return cls(
            factor_order=factor_order,
            coefficients=tuple(float(value) for value in model.coef_),
            intercept=float(model.intercept_),
            alpha=alpha,
            l1_ratio=l1_ratio,
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Score a validation or test frame using the frozen train-derived linear weights."""
        _validate_feature_schema(features, self.factor_order)
        values = _finite_matrix(features, self.factor_order)
        return pd.Series(
            values @ np.asarray(self.coefficients) + self.intercept,
            index=features.index,
            name="score",
        )


@dataclass(frozen=True)
class FittedRidgeBaseline:
    """Adapter that exposes the train-fitted Ridge model through the baseline protocol."""

    model: LinearRidgeBaseline
    model_id: str = "B2"

    def predict(self, data: BaselineDataset) -> PredictionFrame:
        scores = self.model.predict(data.features)
        return PredictionFrame.from_scores(
            data.features,
            scores,
            model_id=self.model_id,
            fold_id=data.fold_id,
            dataset_id=data.dataset_id,
        )


@dataclass(frozen=True)
class RidgeBaselineCandidate:
    """B2 candidate using one explicit Ridge hyperparameter."""

    factor_order: tuple[str, ...]
    alpha: float = 1.0
    model_id: str = "B2"

    def fit(
        self,
        train: BaselineDataset,
        validation: BaselineDataset,
    ) -> FittedBaseline:
        _validate_candidate_views(train, validation, self.factor_order)
        if not np.isfinite(self.alpha) or self.alpha < 0:
            raise ValueError("Ridge alpha must be finite and non-negative")
        model = LinearRidgeBaseline.fit(
            train.features,
            train.labels,
            factor_order=self.factor_order,
            role="TRAIN",
            alpha=self.alpha,
        )
        return FittedRidgeBaseline(model=model, model_id=self.model_id)


@dataclass(frozen=True)
class LinearBaselineCandidate:
    """B2: select Ridge/ElasticNet hyperparameters in three inner Train folds."""

    factor_order: tuple[str, ...]
    alpha_grid: tuple[float, ...] = LINEAR_ALPHA_GRID
    l1_ratio_grid: tuple[float, ...] = LINEAR_L1_RATIO_GRID
    minimum_cross_section: int = 30
    model_id: str = "B2"

    def fit(
        self,
        train: BaselineDataset,
        validation: BaselineDataset,
    ) -> FittedBaseline:
        _validate_candidate_views(train, validation, self.factor_order)
        if self.minimum_cross_section < 2:
            raise ValueError("minimum_cross_section must be at least 2")
        grid = _linear_grid(self.alpha_grid, self.l1_ratio_grid)
        inner_folds = expanding_inner_folds(train)
        scored: list[tuple[float, float, float, float, LinearHyperparameters]] = []
        for parameters in grid:
            fold_scores: list[float] = []
            for inner_fold in inner_folds:
                model = LinearRidgeBaseline.fit(
                    inner_fold.train.features,
                    inner_fold.train.labels,
                    factor_order=self.factor_order,
                    role="TRAIN",
                    alpha=parameters.alpha,
                    l1_ratio=parameters.l1_ratio,
                )
                predictions = _prediction_frame(
                    inner_fold.validation,
                    model.predict(inner_fold.validation.features),
                )
                daily = compute_daily_rank_ic(
                    predictions,
                    inner_fold.validation.labels,
                    minimum_cross_section=self.minimum_cross_section,
                )
                summary = summarize_rank_ic(daily)
                if np.isfinite(summary.mean_rank_ic):
                    fold_scores.append(summary.mean_rank_ic)
            if fold_scores:
                scored.append(
                    (
                        -float(np.mean(fold_scores)),
                        -parameters.alpha,
                        parameters.l1_ratio,
                        -float(len(fold_scores)),
                        parameters,
                    )
                )
        if not scored:
            raise ValueError("B2 has no valid inner-fold RankIC observations")
        selected = min(scored, key=lambda item: item[:4])[-1]
        model = LinearRidgeBaseline.fit(
            train.features,
            train.labels,
            factor_order=self.factor_order,
            role="TRAIN",
            alpha=selected.alpha,
            l1_ratio=selected.l1_ratio,
        )
        return FittedRidgeBaseline(model=model, model_id=self.model_id)


def _validate_feature_schema(features: pd.DataFrame, factor_order: tuple[str, ...]) -> None:
    expected_columns = ("signal_date", "ts_code", *factor_order)
    if tuple(features.columns) != expected_columns:
        raise ValueError("feature order must exactly match signal_date, ts_code, and factor_order")


def _finite_matrix(frame: pd.DataFrame, factor_order: tuple[str, ...]) -> np.ndarray:
    numeric = frame.loc[:, list(factor_order)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("baseline feature values must be finite")
    return cast(np.ndarray, values)


def _validate_candidate_views(
    train: BaselineDataset,
    validation: BaselineDataset,
    factor_order: tuple[str, ...],
) -> None:
    if train.role != "TRAIN" or validation.role != "VALIDATION":
        raise PermissionError("Ridge candidate accepts only TRAIN and VALIDATION data")
    if train.fold_id != validation.fold_id:
        raise ValueError("train and validation datasets must belong to one fold")
    _validate_feature_schema(train.features, factor_order)
    _validate_feature_schema(validation.features, factor_order)


def _linear_grid(
    alpha_grid: tuple[float, ...],
    l1_ratio_grid: tuple[float, ...],
) -> tuple[LinearHyperparameters, ...]:
    if not alpha_grid or not l1_ratio_grid:
        raise ValueError("linear hyperparameter grids cannot be empty")
    return tuple(
        LinearHyperparameters(alpha=alpha, l1_ratio=l1_ratio)
        for alpha in alpha_grid
        for l1_ratio in l1_ratio_grid
    )


def _prediction_frame(dataset: BaselineDataset, scores: pd.Series) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": dataset.features["signal_date"],
            "ts_code": dataset.features["ts_code"],
            "score": scores,
        }
    )
