from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from research_v2.baselines.base import BaselineDataset
from research_v2.baselines.linear import (
    LINEAR_ALPHA_GRID,
    LINEAR_L1_RATIO_GRID,
    FittedRidgeBaseline,
    LinearBaselineCandidate,
    LinearRidgeBaseline,
)


def _dataset(role: str, dataset_id: str, start_day: int) -> BaselineDataset:
    features: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    for day_index in range(10):
        signal_date = date(2020, 1, 1) + timedelta(days=start_day + day_index)
        for stock_index in range(30):
            key = {"signal_date": signal_date, "ts_code": f"S{stock_index:03d}"}
            features.append({**key, "f1": float(stock_index)})
            labels.append(
                {
                    **key,
                    "raw_return_5d": float(stock_index + day_index),
                    "label_valid": True,
                }
            )
    return BaselineDataset(role, "F01", dataset_id, pd.DataFrame(features), pd.DataFrame(labels))


def test_linear_baseline_demeans_raw_return_within_each_training_day() -> None:
    features = pd.DataFrame(
        {
            "signal_date": [date(2020, 1, 1)] * 3 + [date(2020, 1, 2)] * 3,
            "ts_code": ["A", "B", "C"] * 2,
            "f1": [0.0, 1.0, 2.0] * 2,
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2020, 1, 1)] * 3 + [date(2020, 1, 2)] * 3,
            "ts_code": ["A", "B", "C"] * 2,
            "raw_return_5d": [10.0, 11.0, 12.0, 100.0, 101.0, 102.0],
            "label_valid": [True] * 6,
        }
    )
    model = LinearRidgeBaseline.fit(
        features,
        labels,
        factor_order=("f1",),
        role="TRAIN",
        alpha=1e-8,
    )

    scores = model.predict(features)

    assert model.intercept == pytest.approx(-1.0)
    assert scores.groupby(features["signal_date"]).mean().tolist() == pytest.approx([0.0, 0.0])


def test_linear_candidate_uses_the_fixed_grid_and_deterministic_tie_break() -> None:
    assert LINEAR_ALPHA_GRID == (0.01, 0.1, 1.0, 10.0, 100.0)
    assert LINEAR_L1_RATIO_GRID == (0.0, 0.1, 0.5, 0.9, 1.0)
    train = _dataset("TRAIN", "train-v1", 0)
    validation = BaselineDataset(
        "VALIDATION",
        "F01",
        "validation-v1",
        train.features.copy(),
        train.labels.copy(),
    )

    fitted = LinearBaselineCandidate(
        factor_order=("f1",),
        alpha_grid=(0.1, 10.0),
        l1_ratio_grid=(0.0,),
        minimum_cross_section=30,
    ).fit(train, validation)

    assert isinstance(fitted, FittedRidgeBaseline)
    assert fitted.model.alpha == 10.0
    assert fitted.model.l1_ratio == 0.0
    assert np.isfinite(fitted.model.coefficients).all()
