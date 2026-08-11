"""Train-only date-equal Ridge baseline with frozen factor order."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, cast

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge  # type: ignore[import-untyped]

FitRole = Literal["TRAIN", "VALIDATION", "TEST", "LOCKED"]


@dataclass(frozen=True)
class LinearRidgeBaseline:
    factor_order: tuple[str, ...]
    coefficients: tuple[float, ...]
    intercept: float
    alpha: float

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        *,
        factor_order: tuple[str, ...],
        role: FitRole,
        alpha: float = 1.0,
    ) -> LinearRidgeBaseline:
        """Fit only on TRAIN rows, assigning each signal date equal aggregate weight."""
        if role != "TRAIN":
            raise PermissionError("LinearRidgeBaseline.fit is permitted only for TRAIN data")
        if alpha < 0 or not factor_order or len(set(factor_order)) != len(factor_order):
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
        joined = joined.loc[joined["label_valid"]].copy()
        values = _finite_matrix(joined, factor_order)
        target = pd.to_numeric(joined["raw_return_5d"], errors="coerce").to_numpy(dtype=float)
        if not len(target) or not np.isfinite(target).all():
            raise ValueError("TRAIN labels must be finite and non-empty")
        counts = joined.groupby("signal_date", sort=False)["ts_code"].transform("count")
        weights = 1.0 / counts.to_numpy(dtype=float)
        model = Ridge(alpha=alpha, fit_intercept=True)
        model.fit(values, target, sample_weight=weights)
        return cls(
            factor_order=factor_order,
            coefficients=tuple(float(value) for value in model.coef_),
            intercept=float(model.intercept_),
            alpha=alpha,
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
