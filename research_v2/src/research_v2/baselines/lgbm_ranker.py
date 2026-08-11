"""Train-only LightGBM LambdaRank baseline with date-defined ranking groups."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import lightgbm as lgb
import numpy as np
import pandas as pd

from research_v2.baselines.linear import FitRole


@dataclass(frozen=True)
class LGBMRankerBaseline:
    factor_order: tuple[str, ...]
    booster: object
    num_boost_round: int

    @classmethod
    def fit(
        cls,
        features: pd.DataFrame,
        labels: pd.DataFrame,
        *,
        factor_order: tuple[str, ...],
        role: FitRole,
        num_boost_round: int = 100,
    ) -> LGBMRankerBaseline:
        """Fit LambdaRank only on TRAIN dates, with one ranking group per date."""
        if role != "TRAIN":
            raise PermissionError("LGBMRankerBaseline.fit is permitted only for TRAIN data")
        if not factor_order or len(set(factor_order)) != len(factor_order) or num_boost_round < 1:
            raise ValueError("factor_order or num_boost_round is invalid")
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
        if joined.empty:
            raise ValueError("TRAIN labels must contain at least one valid observation")
        joined = joined.sort_values(keys, kind="mergesort", ignore_index=True)
        values = _finite_matrix(joined, factor_order)
        raw_target = pd.to_numeric(joined["raw_return_5d"], errors="coerce")
        if not np.isfinite(raw_target.to_numpy(dtype=float)).all():
            raise ValueError("TRAIN labels must be finite")
        group_sizes = joined.groupby("signal_date", sort=False).size().astype(int).tolist()
        rank = raw_target.groupby(joined["signal_date"], sort=False).rank(method="first")
        # LightGBM's default LambdaRank label mapping is finite.  Encode each
        # date's continuous return ordering into five deterministic relevance
        # buckets (0..4), which also matches the approved B3 specification.
        group_size_by_row = joined["signal_date"].map(joined["signal_date"].value_counts())
        relevance = np.floor((rank.to_numpy(dtype=float) - 1.0) * 5.0 / group_size_by_row)
        relevance = np.clip(relevance, 0, 4).astype(np.int32)
        dataset = lgb.Dataset(values, label=relevance.astype(float), group=group_sizes)
        booster = lgb.train(
            {
                "boosting_type": "gbdt",
                "feature_pre_filter": False,
                "force_col_wise": True,
                "learning_rate": 0.05,
                "min_data_in_leaf": 1,
                "num_leaves": 7,
                "objective": "lambdarank",
                "label_gain": [0, 1, 2, 3, 4],
                "seed": 20260801,
                "verbosity": -1,
            },
            dataset,
            num_boost_round=num_boost_round,
        )
        return cls(factor_order=factor_order, booster=booster, num_boost_round=num_boost_round)

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


def _validate_feature_schema(features: pd.DataFrame, factor_order: tuple[str, ...]) -> None:
    if tuple(features.columns) != ("signal_date", "ts_code", *factor_order):
        raise ValueError("feature order must exactly match signal_date, ts_code, and factor_order")


def _finite_matrix(frame: pd.DataFrame, factor_order: tuple[str, ...]) -> np.ndarray:
    numeric = frame.loc[:, list(factor_order)].apply(pd.to_numeric, errors="coerce")
    values = numeric.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError("baseline feature values must be finite")
    return cast(np.ndarray, values)
