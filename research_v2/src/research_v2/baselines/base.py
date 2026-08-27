"""Shared data and prediction contracts for the simple baseline suite."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Literal, Protocol

import numpy as np
import pandas as pd

BaselineRole = Literal["TRAIN", "VALIDATION", "TEST"]
_BASELINE_ROLES = frozenset({"TRAIN", "VALIDATION", "TEST"})
_BUSINESS_KEYS = ("signal_date", "ts_code")
_PREDICTION_COLUMNS = (
    "signal_date",
    "ts_code",
    "score",
    "model_id",
    "fold_id",
    "dataset_id",
)


@dataclass(frozen=True)
class BaselineDataset:
    """One immutable, role-labelled dataset view supplied to a baseline."""

    role: BaselineRole
    fold_id: str
    dataset_id: str
    features: pd.DataFrame
    labels: pd.DataFrame

    def __post_init__(self) -> None:
        if self.role not in _BASELINE_ROLES:
            raise ValueError("baseline dataset role must be TRAIN, VALIDATION, or TEST")
        if not self.fold_id.strip():
            raise ValueError("fold_id cannot be empty")
        if not self.dataset_id.strip():
            raise ValueError("dataset_id cannot be empty")
        if not isinstance(self.features, pd.DataFrame) or not isinstance(self.labels, pd.DataFrame):
            raise TypeError("baseline features and labels must be pandas DataFrames")
        features = _normalize_keyed_frame(self.features, "features")
        labels = _normalize_keyed_frame(self.labels, "labels")
        required_labels = {"raw_return_5d", "label_valid"}
        if missing := sorted(required_labels.difference(labels.columns)):
            raise ValueError(f"labels are missing columns: {missing}")
        if features.empty:
            raise ValueError("baseline features cannot be empty")
        if labels.empty:
            raise ValueError("baseline labels cannot be empty")
        _require_matching_keys(features, labels)
        _reject_locked_dates(self.fold_id, self.role, features, labels)
        object.__setattr__(self, "features", features)
        object.__setattr__(self, "labels", labels)


@dataclass(frozen=True)
class PredictionFrame:
    """Business-keyed baseline scores with model, fold, and dataset lineage."""

    frame: pd.DataFrame

    def __post_init__(self) -> None:
        if not isinstance(self.frame, pd.DataFrame):
            raise TypeError("prediction frame must be a pandas DataFrame")
        if tuple(self.frame.columns) != _PREDICTION_COLUMNS:
            raise ValueError("prediction frame columns must exactly match the baseline contract")
        if self.frame.empty:
            raise ValueError("prediction frame cannot be empty")
        normalized = self.frame.copy(deep=True)
        normalized["signal_date"] = pd.to_datetime(
            normalized["signal_date"], errors="raise"
        ).dt.date
        if normalized["signal_date"].isna().any():
            raise ValueError("prediction signal_date cannot be null")
        if normalized["ts_code"].isna().any() or normalized["ts_code"].astype(str).eq("").any():
            raise ValueError("prediction ts_code cannot be null or empty")
        if normalized.duplicated(list(_BUSINESS_KEYS)).any():
            raise ValueError("prediction frame must have unique signal-date/security keys")
        for column in ("model_id", "fold_id", "dataset_id"):
            if normalized[column].isna().any() or normalized[column].astype(str).eq("").any():
                raise ValueError(f"prediction {column} cannot be null or empty")
        numeric = pd.to_numeric(normalized["score"], errors="coerce")
        invalid_non_null = normalized["score"].notna() & numeric.isna()
        if invalid_non_null.any():
            raise ValueError("prediction scores must be numeric")
        numeric_values = numeric.to_numpy(dtype=float, na_value=np.nan)
        if np.isinf(numeric_values).any():
            raise ValueError("prediction scores must be finite and non-finite values are forbidden")
        normalized["score"] = numeric
        object.__setattr__(self, "frame", normalized.loc[:, list(_PREDICTION_COLUMNS)])

    @classmethod
    def from_scores(
        cls,
        features: pd.DataFrame,
        scores: Sequence[Any] | pd.Series[Any],
        *,
        model_id: str,
        fold_id: str,
        dataset_id: str,
    ) -> PredictionFrame:
        """Attach scores to feature keys without changing their positional alignment."""
        if not model_id.strip() or not fold_id.strip() or not dataset_id.strip():
            raise ValueError("prediction lineage fields cannot be empty")
        normalized_features = _normalize_keyed_frame(features, "features")
        values = pd.Series(scores).reset_index(drop=True)
        if len(values) != len(normalized_features):
            raise ValueError("score count must match feature row count")
        frame = normalized_features.loc[:, list(_BUSINESS_KEYS)].reset_index(drop=True)
        frame["score"] = values
        frame["model_id"] = model_id
        frame["fold_id"] = fold_id
        frame["dataset_id"] = dataset_id
        return cls(frame.loc[:, list(_PREDICTION_COLUMNS)])


class FittedBaseline(Protocol):
    """A train-fitted baseline that can score a role-labelled dataset."""

    def predict(self, data: BaselineDataset) -> PredictionFrame: ...


class Baseline(Protocol):
    """A baseline whose fit method can receive only TRAIN and VALIDATION."""

    def fit(self, train: BaselineDataset, validation: BaselineDataset) -> FittedBaseline: ...


def _normalize_keyed_frame(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = set(_BUSINESS_KEYS).difference(frame.columns)
    if missing:
        raise ValueError(f"{name} are missing business-key columns: {sorted(missing)}")
    normalized = frame.copy(deep=True)
    normalized["signal_date"] = pd.to_datetime(normalized["signal_date"], errors="raise").dt.date
    if normalized["signal_date"].isna().any():
        raise ValueError(f"{name} signal_date cannot be null")
    if normalized["ts_code"].isna().any() or normalized["ts_code"].astype(str).eq("").any():
        raise ValueError(f"{name} ts_code cannot be null or empty")
    if normalized.duplicated(list(_BUSINESS_KEYS)).any():
        raise ValueError(f"{name} must have unique signal-date/security keys")
    return normalized.reset_index(drop=True)


def _require_matching_keys(features: pd.DataFrame, labels: pd.DataFrame) -> None:
    feature_keys = (
        features.loc[:, list(_BUSINESS_KEYS)]
        .sort_values(
            list(_BUSINESS_KEYS),
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    label_keys = (
        labels.loc[:, list(_BUSINESS_KEYS)]
        .sort_values(
            list(_BUSINESS_KEYS),
            kind="mergesort",
        )
        .reset_index(drop=True)
    )
    if not feature_keys.equals(label_keys):
        raise ValueError("baseline features and labels must have identical business keys")


def _reject_locked_dates(
    fold_id: str,
    role: BaselineRole,
    features: pd.DataFrame,
    labels: pd.DataFrame,
) -> None:
    for name, frame in (("features", features), ("labels", labels)):
        years = pd.to_datetime(frame["signal_date"], errors="raise").dt.year
        if (years >= 2026).any():
            if fold_id == "FINAL" and role == "TEST":
                raise PermissionError("FINAL test baseline data is locked")
            raise PermissionError(
                f"ordinary baseline data cannot consume 2026 or future dates: {name}"
            )
