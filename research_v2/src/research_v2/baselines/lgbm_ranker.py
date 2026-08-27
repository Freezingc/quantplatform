"""Train-only LightGBM LambdaRank baseline with date-defined ranking groups."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import lightgbm as lgb
import numpy as np
import pandas as pd

from research_v2.baselines.base import BaselineDataset, FittedBaseline, PredictionFrame
from research_v2.baselines.cross_validation import expanding_inner_folds
from research_v2.baselines.linear import FitRole
from research_v2.evaluation.rank_ic import compute_daily_rank_ic, summarize_rank_ic

LGBM_NUM_LEAVES_GRID = (15, 31)
LGBM_MAX_DEPTH_GRID = (4, 6)
LGBM_LEARNING_RATE_GRID = (0.03, 0.05)
LGBM_MIN_CHILD_SAMPLES_GRID = (50, 200)
LGBM_MAX_ESTIMATORS = 500
LGBM_EARLY_STOPPING_PATIENCE = 50


@dataclass(frozen=True)
class LGBMHyperparameters:
    num_leaves: int
    max_depth: int
    learning_rate: float
    min_child_samples: int

    def __post_init__(self) -> None:
        if self.num_leaves < 1 or self.max_depth < 1 or self.min_child_samples < 1:
            raise ValueError("LightGBM tree parameters must be positive")
        if self.learning_rate <= 0 or not np.isfinite(self.learning_rate):
            raise ValueError("LightGBM learning_rate must be finite and positive")


_DIRECT_FIT_DEFAULTS = LGBMHyperparameters(
    num_leaves=7,
    max_depth=4,
    learning_rate=0.05,
    min_child_samples=1,
)
LGBM_HYPERPARAMETER_GRID = tuple(
    LGBMHyperparameters(
        num_leaves=num_leaves,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_child_samples=min_child_samples,
    )
    for num_leaves in LGBM_NUM_LEAVES_GRID
    for max_depth in LGBM_MAX_DEPTH_GRID
    for learning_rate in LGBM_LEARNING_RATE_GRID
    for min_child_samples in LGBM_MIN_CHILD_SAMPLES_GRID
)


@dataclass(frozen=True)
class LGBMRankerBaseline:
    factor_order: tuple[str, ...]
    booster: object
    num_boost_round: int
    hyperparameters: LGBMHyperparameters = _DIRECT_FIT_DEFAULTS
    training_day_count: int = 0
    eligible_training_day_count: int = 0
    best_iteration: int | None = None

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        *,
        factor_order: tuple[str, ...],
        role: FitRole,
        num_boost_round: int = LGBM_MAX_ESTIMATORS,
        minimum_cross_section: int = 30,
        hyperparameters: LGBMHyperparameters | None = None,
        evaluation_features: pd.DataFrame | None = None,
        evaluation_labels: pd.DataFrame | None = None,
        early_stopping_patience: int = LGBM_EARLY_STOPPING_PATIENCE,
    ) -> LGBMRankerBaseline:
        """Fit LambdaRank only on TRAIN data, with one ranking group per date."""
        if role != "TRAIN":
            raise PermissionError("LGBMRankerBaseline.fit is permitted only for TRAIN data")
        if (
            not factor_order
            or len(set(factor_order)) != len(factor_order)
            or num_boost_round < 1
            or minimum_cross_section < 2
            or early_stopping_patience < 1
        ):
            raise ValueError("factor_order or training parameters are invalid")
        selected_parameters = hyperparameters or _DIRECT_FIT_DEFAULTS
        _validate_feature_schema(features, factor_order)
        normalized_features = _normalize_dates(features)
        normalized_labels = _normalize_dates(labels)
        joined, training_day_count, eligible_training_day_count = _prepare_ranker_rows(
            normalized_features,
            normalized_labels,
            factor_order=factor_order,
            minimum_cross_section=minimum_cross_section,
        )
        values = _finite_matrix(joined, factor_order)
        raw_target = pd.to_numeric(joined["raw_return_5d"], errors="coerce")
        relevance, group_sizes = _relevance_and_groups(joined, raw_target)
        dataset = lgb.Dataset(values, label=relevance.astype(float), group=group_sizes)
        evaluation_set: lgb.Dataset | None = None
        if (evaluation_features is None) != (evaluation_labels is None):
            raise ValueError("evaluation_features and evaluation_labels must be supplied together")
        if evaluation_features is not None and evaluation_labels is not None:
            _validate_feature_schema(evaluation_features, factor_order)
            evaluation_joined, _, _ = _prepare_ranker_rows(
                _normalize_dates(evaluation_features),
                _normalize_dates(evaluation_labels),
                factor_order=factor_order,
                minimum_cross_section=minimum_cross_section,
            )
            evaluation_values = _finite_matrix(evaluation_joined, factor_order)
            evaluation_target = pd.to_numeric(
                evaluation_joined["raw_return_5d"], errors="coerce"
            )
            evaluation_relevance, evaluation_groups = _relevance_and_groups(
                evaluation_joined,
                evaluation_target,
            )
            evaluation_set = lgb.Dataset(
                evaluation_values,
                label=evaluation_relevance.astype(float),
                group=evaluation_groups,
                reference=dataset,
            )
        params = _training_parameters(selected_parameters)
        callbacks: list[Callable[..., Any]] = (
            [lgb.early_stopping(early_stopping_patience, verbose=False)]
            if evaluation_set is not None
            else []
        )
        valid_sets = [evaluation_set] if evaluation_set is not None else None
        booster = lgb.train(
            params,
            dataset,
            num_boost_round=num_boost_round,
            valid_sets=valid_sets,
            valid_names=["inner_validation"] if evaluation_set is not None else None,
            callbacks=callbacks,
        )
        best_iteration = getattr(booster, "best_iteration", None)
        return cls(
            factor_order=factor_order,
            booster=booster,
            num_boost_round=num_boost_round,
            hyperparameters=selected_parameters,
            training_day_count=training_day_count,
            eligible_training_day_count=eligible_training_day_count,
            best_iteration=int(best_iteration) if best_iteration else None,
        )

    def predict(self, features: pd.DataFrame) -> pd.Series:
        """Score a validation or test frame with the immutable trained booster."""
        _validate_feature_schema(features, self.factor_order)
        values = _finite_matrix(features, self.factor_order)
        predict = getattr(self.booster, "predict", None)
        if not callable(predict):
            raise ValueError("stored LightGBM booster cannot predict")
        scores = cast(np.ndarray, predict(values))
        if not np.isfinite(scores).all():
            raise ValueError("LightGBM emitted non-finite scores")
        return pd.Series(scores, index=features.index, name="score")


@dataclass(frozen=True)
class FittedLGBMRankerBaseline:
    """Adapter that exposes the train-fitted ranker through the baseline protocol."""

    model: LGBMRankerBaseline
    model_id: str = "B3"

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
class LGBMRankerCandidate:
    """B3: select the fixed LambdaRank grid in three inner Train folds."""

    factor_order: tuple[str, ...]
    hyperparameter_grid: tuple[LGBMHyperparameters, ...] = LGBM_HYPERPARAMETER_GRID
    num_boost_round: int = LGBM_MAX_ESTIMATORS
    minimum_cross_section: int = 30
    early_stopping_patience: int = LGBM_EARLY_STOPPING_PATIENCE
    model_id: str = "B3"

    def fit(
        self,
        train: BaselineDataset,
        validation: BaselineDataset,
    ) -> FittedBaseline:
        if train.role != "TRAIN" or validation.role != "VALIDATION":
            raise PermissionError("LightGBM candidate accepts only TRAIN and VALIDATION data")
        if train.fold_id != validation.fold_id:
            raise ValueError("train and validation datasets must belong to one fold")
        if (
            not self.hyperparameter_grid
            or self.minimum_cross_section < 2
            or self.num_boost_round < 1
            or self.early_stopping_patience < 1
        ):
            raise ValueError("LightGBM candidate training parameters are invalid")
        _validate_feature_schema(train.features, self.factor_order)
        _validate_feature_schema(validation.features, self.factor_order)
        inner_folds = expanding_inner_folds(train)
        scored: list[tuple[float, int, LGBMHyperparameters, list[int]]] = []
        for grid_index, parameters in enumerate(self.hyperparameter_grid):
            fold_scores: list[float] = []
            best_iterations: list[int] = []
            for inner_fold in inner_folds:
                inner_model = LGBMRankerBaseline.fit(
                    inner_fold.train.features,
                    inner_fold.train.labels,
                    factor_order=self.factor_order,
                    role="TRAIN",
                    num_boost_round=self.num_boost_round,
                    minimum_cross_section=self.minimum_cross_section,
                    hyperparameters=parameters,
                    evaluation_features=inner_fold.validation.features,
                    evaluation_labels=inner_fold.validation.labels,
                    early_stopping_patience=self.early_stopping_patience,
                )
                predictions = pd.DataFrame(
                    {
                        "signal_date": inner_fold.validation.features["signal_date"],
                        "ts_code": inner_fold.validation.features["ts_code"],
                        "score": inner_model.predict(inner_fold.validation.features),
                    }
                )
                daily = compute_daily_rank_ic(
                    predictions,
                    inner_fold.validation.labels,
                    minimum_cross_section=self.minimum_cross_section,
                )
                summary = summarize_rank_ic(daily)
                if np.isfinite(summary.mean_rank_ic):
                    fold_scores.append(summary.mean_rank_ic)
                if inner_model.best_iteration is not None:
                    best_iterations.append(inner_model.best_iteration)
            if fold_scores:
                scored.append(
                    (
                        -float(np.mean(fold_scores)),
                        grid_index,
                        parameters,
                        best_iterations,
                    )
                )
        if not scored:
            raise ValueError("B3 has no valid inner-fold RankIC observations")
        _, _, selected_parameters, best_iterations = min(
            scored,
            key=lambda item: item[:2],
        )
        final_rounds = (
            int(np.median(best_iterations))
            if best_iterations
            else self.num_boost_round
        )
        final_rounds = max(1, min(self.num_boost_round, final_rounds))
        model = LGBMRankerBaseline.fit(
            train.features,
            train.labels,
            factor_order=self.factor_order,
            role="TRAIN",
            num_boost_round=final_rounds,
            minimum_cross_section=self.minimum_cross_section,
            hyperparameters=selected_parameters,
        )
        return FittedLGBMRankerBaseline(model=model, model_id=self.model_id)


def _validate_feature_schema(features: pd.DataFrame, factor_order: tuple[str, ...]) -> None:
    if tuple(features.columns) != ("signal_date", "ts_code", *factor_order):
        raise ValueError("feature order must exactly match signal_date, ts_code, and factor_order")


def _finite_matrix(frame: pd.DataFrame, factor_order: tuple[str, ...]) -> np.ndarray:
    numeric = frame.loc[:, list(factor_order)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("baseline feature values must be finite")
    return cast(np.ndarray, values)


def _normalize_dates(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.copy(deep=True)
    if "signal_date" in normalized.columns:
        normalized["signal_date"] = pd.to_datetime(
            normalized["signal_date"], errors="raise"
        ).dt.date
    return normalized


def _prepare_ranker_rows(
    features: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    factor_order: tuple[str, ...],
    minimum_cross_section: int,
) -> tuple[pd.DataFrame, int, int]:
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
    joined["raw_return_5d"] = pd.to_numeric(joined["raw_return_5d"], errors="coerce")
    joined = joined.loc[
        joined["label_valid"] & np.isfinite(joined["raw_return_5d"])
    ].copy()
    if joined.empty:
        raise ValueError("TRAIN labels must contain at least one valid observation")
    training_day_count = int(joined["signal_date"].nunique())
    counts = joined.groupby("signal_date", sort=False)["ts_code"].size()
    eligible_dates = counts.loc[counts >= minimum_cross_section].index
    joined = joined.loc[joined["signal_date"].isin(eligible_dates)].copy()
    if joined.empty:
        raise ValueError("no training date meets minimum_cross_section")
    joined = joined.sort_values(keys, kind="mergesort", ignore_index=True)
    return joined, training_day_count, int(len(eligible_dates))


def _relevance_and_groups(
    joined: pd.DataFrame,
    raw_target: pd.Series,
) -> tuple[np.ndarray, list[int]]:
    group_sizes = joined.groupby("signal_date", sort=False).size().astype(int).tolist()
    if sum(group_sizes) != len(joined):
        raise ValueError("LightGBM query group sizes must sum to the row count")
    rank = raw_target.groupby(joined["signal_date"], sort=False).rank(method="first")
    group_size_by_row = joined["signal_date"].map(joined["signal_date"].value_counts())
    relevance = np.floor(
        (rank.to_numpy(dtype=float) - 1.0)
        * 5.0
        / group_size_by_row.to_numpy(dtype=float)
    )
    return np.clip(relevance, 0, 4).astype(np.int32), group_sizes


def _training_parameters(parameters: LGBMHyperparameters) -> dict[str, object]:
    return {
        "boosting_type": "gbdt",
        "deterministic": True,
        "feature_pre_filter": False,
        "force_col_wise": True,
        "learning_rate": parameters.learning_rate,
        "max_depth": parameters.max_depth,
        "min_child_samples": parameters.min_child_samples,
        "num_leaves": parameters.num_leaves,
        "num_threads": 1,
        "objective": "lambdarank",
        "label_gain": [0, 1, 2, 3, 4],
        "seed": 20260801,
        "verbosity": -1,
    }
