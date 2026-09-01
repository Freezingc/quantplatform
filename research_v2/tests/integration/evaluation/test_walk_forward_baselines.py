"""Synthetic seven-year end-to-end baseline orchestration evidence (D8.6)."""

from __future__ import annotations

import os
from datetime import date

import numpy as np
import pandas as pd
import pytest

from research_v2.baselines.base import BaselineDataset
from research_v2.baselines.lgbm_ranker import LGBMHyperparameters, LGBMRankerCandidate
from research_v2.baselines.orchestration import (
    BaselineFold,
    BaselineFoldResult,
    build_default_baseline_candidates,
    run_baseline_fold,
)

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("QRV2_INTEGRATION") != "1",
        reason="set QRV2_INTEGRATION=1 to run baseline walk-forward integration tests",
    ),
]

_FACTOR_ORDER = ("f1", "f2", "f3")
_DIRECTIONS = {"f1": 1, "f2": 1, "f3": 1}
_STOCK_COUNT = 40


def _business_days(start: str, end: str) -> list[date]:
    return [day.date() for day in pd.bdate_range(start, end)]


def _panel(days: list[date], seed: int, *, label_seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    generator = np.random.default_rng(seed)
    label_generator = np.random.default_rng(label_seed)
    alpha = generator.standard_normal(_STOCK_COUNT)
    rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for signal_date in days:
        f1 = alpha + generator.standard_normal(_STOCK_COUNT) * 0.5
        f2 = generator.standard_normal(_STOCK_COUNT)
        f3 = generator.standard_normal(_STOCK_COUNT)
        returns = alpha + label_generator.standard_normal(_STOCK_COUNT) * 0.5
        for stock_index in range(_STOCK_COUNT):
            code = f"s{stock_index:02d}"
            rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": code,
                    "f1": f1[stock_index],
                    "f2": f2[stock_index],
                    "f3": f3[stock_index],
                }
            )
            label_rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": code,
                    "raw_return_5d": returns[stock_index],
                    "label_valid": True,
                }
            )
    features = pd.DataFrame(rows)
    labels = pd.DataFrame(label_rows)
    return features, labels


def _dataset(
    role: str, days: list[date], *, seed: int, label_seed: int, dataset_id: str
) -> BaselineDataset:
    features, labels = _panel(days, seed, label_seed=label_seed)
    return BaselineDataset(
        role=role,
        fold_id="SYN01",
        dataset_id=dataset_id,
        features=features,
        labels=labels,
    )


def _smoke_candidates(fold: BaselineFold) -> dict[str, object]:
    candidates = dict(build_default_baseline_candidates(fold, minimum_cross_section=30))
    candidates["B3"] = LGBMRankerCandidate(
        factor_order=_FACTOR_ORDER,
        hyperparameter_grid=(
            LGBMHyperparameters(
                num_leaves=15, max_depth=4, learning_rate=0.05, min_child_samples=50
            ),
        ),
        num_boost_round=20,
        minimum_cross_section=30,
        early_stopping_patience=10,
    )
    return candidates


def _run_pipeline(test_dataset: BaselineDataset) -> BaselineFoldResult:
    train = _dataset(
        "TRAIN",
        _business_days("2015-01-05", "2019-12-31"),
        seed=7,
        label_seed=8,
        dataset_id="synthetic-train",
    )
    validation = _dataset(
        "VALIDATION",
        _business_days("2020-01-01", "2020-12-31"),
        seed=9,
        label_seed=10,
        dataset_id="synthetic-validation",
    )
    fold = BaselineFold(
        fold_id="SYN01",
        train=train,
        validation=validation,
        factor_order=_FACTOR_ORDER,
        frozen_directions=_DIRECTIONS,
        test_loader=lambda: test_dataset,
    )
    records: list[object] = []
    return run_baseline_fold(
        fold,
        candidates=_smoke_candidates(fold),
        minimum_cross_section=30,
        selection_recorder=records.append,
    )


def test_end_to_end_baseline_pipeline_is_deterministic() -> None:
    test_dataset = _dataset(
        "TEST",
        _business_days("2021-01-01", "2021-12-31"),
        seed=11,
        label_seed=12,
        dataset_id="synthetic-test",
    )

    first = _run_pipeline(test_dataset)
    second = _run_pipeline(test_dataset)

    assert first.selection_record == second.selection_record
    assert first.test_evaluation is not None and second.test_evaluation is not None
    assert first.test_evaluation.baseline_id == second.test_evaluation.baseline_id
    assert first.test_evaluation.summary == second.test_evaluation.summary
    pd.testing.assert_frame_equal(
        first.test_evaluation.predictions.frame, second.test_evaluation.predictions.frame
    )


def test_end_to_end_selection_is_invariant_to_test_perturbation() -> None:
    clean_test = _dataset(
        "TEST",
        _business_days("2021-01-01", "2021-12-31"),
        seed=11,
        label_seed=12,
        dataset_id="synthetic-test",
    )
    perturbed_labels = clean_test.labels.copy()
    perturbed_labels["raw_return_5d"] = perturbed_labels["raw_return_5d"] * -100.0
    perturbed_test = BaselineDataset(
        role="TEST",
        fold_id="SYN01",
        dataset_id="synthetic-test",
        features=clean_test.features,
        labels=perturbed_labels,
    )

    clean_result = _run_pipeline(clean_test)
    perturbed_result = _run_pipeline(perturbed_test)

    assert clean_result.selection_record == perturbed_result.selection_record
    assert clean_result.test_evaluation is not None
    assert perturbed_result.test_evaluation is not None
    assert clean_result.test_evaluation.summary != perturbed_result.test_evaluation.summary


def test_end_to_end_selection_records_all_validation_scores() -> None:
    test_dataset = _dataset(
        "TEST",
        _business_days("2021-01-01", "2021-12-31"),
        seed=11,
        label_seed=12,
        dataset_id="synthetic-test",
    )

    result = _run_pipeline(test_dataset)

    record = result.selection_record
    assert record.fold_id == "SYN01"
    assert record.selected_baseline_id in {"B0", "B1", "B2", "B3"}
    assert {score.baseline_id for score in record.validation_scores} == {"B0", "B1", "B2", "B3"}
    assert {reason[0] for reason in record.unselected_reasons} == (
        {"B0", "B1", "B2", "B3"} - {record.selected_baseline_id}
    )
    assert result.test_evaluation is not None
    assert result.test_evaluation.baseline_id == record.selected_baseline_id
    assert result.test_evaluation.summary.valid_day_count > 0
