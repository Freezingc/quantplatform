from __future__ import annotations

from datetime import date

import pandas as pd

from research_v2.baselines.base import BaselineDataset
from research_v2.baselines.single_factor import SingleFactorBaseline, select_single_factor


def test_single_factor_selection_uses_validation_rank_ic_and_stable_tie_break() -> None:
    features = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 3,
            "ts_code": ["a", "b", "c"],
            "f1": [1.0, 2.0, 3.0],
            "f2": [3.0, 2.0, 1.0],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 3,
            "ts_code": ["a", "b", "c"],
            "raw_return_5d": [1.0, 2.0, 3.0],
            "label_valid": [True] * 3,
        }
    )

    selected = select_single_factor(
        features,
        labels,
        frozen_directions={"f1": 1, "f2": -1},
        minimum_cross_section=3,
    )

    assert selected.factor_name == "f1"
    assert selected.validation_mean_rank_ic == 1.0


def test_single_factor_baseline_uses_validation_then_train_tie_break() -> None:
    train_features = pd.DataFrame(
        {
            "signal_date": [date(2024, 1, 2)] * 3,
            "ts_code": ["a", "b", "c"],
            "f1": [1.0, 2.0, 3.0],
            "f2": [3.0, 2.0, 1.0],
        }
    )
    train_labels = pd.DataFrame(
        {
            "signal_date": [date(2024, 1, 2)] * 3,
            "ts_code": ["a", "b", "c"],
            "raw_return_5d": [1.0, 2.0, 3.0],
            "label_valid": [True] * 3,
        }
    )
    validation_features = train_features.copy()
    validation_features["signal_date"] = date(2025, 1, 2)
    validation_labels = train_labels.copy()
    validation_labels["signal_date"] = date(2025, 1, 2)
    validation_labels["raw_return_5d"] = [3.0, 2.0, 1.0]
    fitted = SingleFactorBaseline(
        factor_order=("f1", "f2"),
        frozen_directions={"f1": 1, "f2": -1},
        minimum_cross_section=3,
    ).fit(
        BaselineDataset("TRAIN", "F01", "train-v1", train_features, train_labels),
        BaselineDataset(
            "VALIDATION", "F01", "validation-v1", validation_features, validation_labels
        ),
    )

    assert fitted.factor_name == "f1"
    assert fitted.frozen_direction == 1
