from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import pytest

from research_v2.baselines.base import BaselineDataset
from research_v2.baselines.cross_validation import expanding_inner_folds


def _training_dataset() -> BaselineDataset:
    features: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    for day_index in range(10):
        signal_date = date(2020, 1, 1) + timedelta(days=day_index)
        for stock_index in range(3):
            key = {"signal_date": signal_date, "ts_code": f"S{stock_index}"}
            features.append({**key, "f1": float(stock_index)})
            labels.append({**key, "raw_return_5d": float(stock_index), "label_valid": True})
    return BaselineDataset(
        role="TRAIN",
        fold_id="F01",
        dataset_id="train-v1",
        features=pd.DataFrame(features),
        labels=pd.DataFrame(labels),
    )


def test_expanding_inner_folds_are_chronological_and_disjoint() -> None:
    folds = expanding_inner_folds(_training_dataset())

    assert [fold.inner_fold_id for fold in folds] == ["F01.I1", "F01.I2", "F01.I3"]
    for fold in folds:
        train_dates = set(fold.train.features["signal_date"])
        validation_dates = set(fold.validation.features["signal_date"])
        assert train_dates.isdisjoint(validation_dates)
        assert max(train_dates) < min(validation_dates)
        assert fold.train.role == "TRAIN"
        assert fold.validation.role == "VALIDATION"


def test_expanding_inner_folds_rejects_non_train_or_short_input() -> None:
    training = _training_dataset()
    validation = BaselineDataset(
        role="VALIDATION",
        fold_id="F01",
        dataset_id="validation-v1",
        features=training.features,
        labels=training.labels,
    )
    with pytest.raises(PermissionError, match="TRAIN"):
        expanding_inner_folds(validation)

    short = BaselineDataset(
        role="TRAIN",
        fold_id="F01",
        dataset_id="short-v1",
        features=training.features.loc[training.features["signal_date"] < date(2020, 1, 5)],
        labels=training.labels.loc[training.labels["signal_date"] < date(2020, 1, 5)],
    )
    with pytest.raises(ValueError, match="five"):
        expanding_inner_folds(short)
