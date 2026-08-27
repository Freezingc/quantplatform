from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.baselines.base import BaselineDataset, PredictionFrame


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "ts_code": ["A", "B"],
            "f1": [1.0, 2.0],
        }
    )


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "ts_code": ["A", "B"],
            "raw_return_5d": [1.0, 2.0],
            "label_valid": [True, True],
        }
    )


def test_baseline_dataset_validates_role_and_identity() -> None:
    dataset = BaselineDataset(
        role="TRAIN",
        fold_id="F01",
        dataset_id="train-v1",
        features=_features(),
        labels=_labels(),
    )

    assert dataset.role == "TRAIN"
    assert dataset.fold_id == "F01"
    with pytest.raises(ValueError, match="dataset_id"):
        BaselineDataset(
            role="TRAIN",
            fold_id="F01",
            dataset_id="",
            features=_features(),
            labels=_labels(),
        )


def test_prediction_frame_contains_only_keyed_scores_and_lineage() -> None:
    prediction = PredictionFrame.from_scores(
        _features(),
        pd.Series([0.1, pd.NA]),
        model_id="B1",
        fold_id="F01",
        dataset_id="validation-v1",
    )

    assert list(prediction.frame.columns) == [
        "signal_date",
        "ts_code",
        "score",
        "model_id",
        "fold_id",
        "dataset_id",
    ]
    assert prediction.frame.loc[0, "model_id"] == "B1"
    assert pd.isna(prediction.frame.loc[1, "score"])

    with pytest.raises(ValueError, match="non-finite"):
        PredictionFrame.from_scores(
            _features(),
            pd.Series([0.1, float("inf")]),
            model_id="B1",
            fold_id="F01",
            dataset_id="validation-v1",
        )
