from __future__ import annotations

import math

import pandas as pd

from research_v2.baselines.base import BaselineDataset
from research_v2.baselines.equal_weight import EqualWeightBaseline, equal_weight_scores


def test_equal_weight_preserves_all_missing_rows() -> None:
    features = pd.DataFrame({"f1": [1.0, None, 1.0], "f2": [3.0, None, None]})

    scores = equal_weight_scores(features, ("f1", "f2"))

    assert scores.iloc[0] == 2.0
    assert math.isnan(scores.iloc[1])
    assert math.isnan(scores.iloc[2])


def test_equal_weight_baseline_applies_frozen_directions_and_coverage_floor() -> None:
    features = pd.DataFrame(
        {
            "signal_date": ["2025-01-02"] * 3,
            "ts_code": ["A", "B", "C"],
            "f1": [1.0, 2.0, 3.0],
            "f2": [10.0, 20.0, None],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": ["2025-01-02"] * 3,
            "ts_code": ["A", "B", "C"],
            "raw_return_5d": [1.0, 2.0, 3.0],
            "label_valid": [True] * 3,
        }
    )
    train = BaselineDataset("TRAIN", "F01", "train-v1", features, labels)
    dataset = BaselineDataset("VALIDATION", "F01", "validation-v1", features, labels)
    fitted = EqualWeightBaseline(
        factor_order=("f1", "f2"),
        frozen_directions={"f1": 1, "f2": 1},
    ).fit(train, dataset)

    scores = fitted.predict(dataset).frame["score"]

    assert scores.iloc[:2].tolist() == [0.5, 1.0]
    assert pd.isna(scores.iloc[2])
