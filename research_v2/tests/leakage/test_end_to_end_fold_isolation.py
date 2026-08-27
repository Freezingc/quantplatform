from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.baselines.base import BaselineDataset
from research_v2.evaluation.leakage import (
    assert_train_scope_isolation,
    fingerprint_train_scope,
)


def test_future_perfect_rows_do_not_change_train_preprocessor_fingerprint() -> None:
    train = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0],
            "f2": [4.0, 3.0, 2.0, 1.0],
        }
    )
    future_perfect = pd.DataFrame(
        {
            "f1": [100_000.0, 100_001.0],
            "f2": [-100_000.0, -100_001.0],
        }
    )

    original = fingerprint_train_scope(
        train_features=train,
        factor_order=("f1", "f2"),
    )
    contaminated = fingerprint_train_scope(
        train_features=pd.concat([train, future_perfect], ignore_index=True).iloc[: len(train)],
        factor_order=("f1", "f2"),
    )

    assert_train_scope_isolation(original, contaminated)


def test_ordinary_train_baseline_rejects_future_perfect_2026_rows() -> None:
    features = pd.DataFrame(
        {
            "signal_date": [date(2025, 12, 31), date(2026, 1, 2)],
            "ts_code": ["A", "A"],
            "f1": [1.0, 100_000.0],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2025, 12, 31), date(2026, 1, 2)],
            "ts_code": ["A", "A"],
            "raw_return_5d": [1.0, 100_000.0],
            "label_valid": [True, True],
        }
    )

    with pytest.raises(PermissionError, match="2026|future"):
        BaselineDataset(
            role="TRAIN",
            fold_id="F06",
            dataset_id="train-v1",
            features=features,
            labels=labels,
        )
