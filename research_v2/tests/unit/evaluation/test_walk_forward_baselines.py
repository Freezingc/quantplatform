from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from research_v2.baselines.base import BaselineDataset, PredictionFrame
from research_v2.baselines.orchestration import BaselineFold
from research_v2.evaluation.walk_forward import (
    ORDINARY_TEST_FOLDS,
    run_walk_forward_baselines,
)


def _dataset(role: str, fold_id: str, dataset_id: str, signal_date: date) -> BaselineDataset:
    # Two dates per dataset so four folds clear the >5 valid-day OOS gate.
    dates = [signal_date] * 30 + [signal_date + timedelta(days=1)] * 30
    codes = [f"S{index:03d}" for index in range(30)]
    features = pd.DataFrame(
        {
            "signal_date": dates,
            "ts_code": codes * 2,
            "f1": [float(index) for index in range(30)] * 2,
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": dates,
            "ts_code": codes * 2,
            "raw_return_5d": [float(index) for index in range(30)] * 2,
            "label_valid": [True] * 60,
        }
    )
    return BaselineDataset(role, fold_id, dataset_id, features, labels)


class _Fitted:
    def __init__(self, baseline_id: str) -> None:
        self._baseline_id = baseline_id

    def predict(self, data: BaselineDataset) -> PredictionFrame:
        return PredictionFrame.from_scores(
            data.features,
            data.features["f1"],
            model_id=self._baseline_id,
            fold_id=data.fold_id,
            dataset_id=data.dataset_id,
        )


class _Candidate:
    def __init__(self, baseline_id: str) -> None:
        self._baseline_id = baseline_id

    def fit(self, train: BaselineDataset, validation: BaselineDataset) -> _Fitted:
        return _Fitted(self._baseline_id)


def test_walk_forward_baselines_runs_four_winners_and_marks_g4_not_applicable() -> None:
    load_counts = {fold_id: 0 for fold_id in ORDINARY_TEST_FOLDS}
    folds: dict[str, BaselineFold] = {}
    for index, fold_id in enumerate(ORDINARY_TEST_FOLDS, start=1):
        train = _dataset("TRAIN", fold_id, f"{fold_id}-train", date(2010 + index, 1, 2))
        validation = _dataset(
            "VALIDATION",
            fold_id,
            f"{fold_id}-validation",
            date(2018 + index, 1, 2),
        )
        test = _dataset("TEST", fold_id, f"{fold_id}-test", date(2021 + index, 1, 2))

        def load_test(fold_id: str = fold_id, test: BaselineDataset = test) -> BaselineDataset:
            load_counts[fold_id] += 1
            return test

        folds[fold_id] = BaselineFold(
            fold_id=fold_id,
            train=train,
            validation=validation,
            factor_order=("f1",),
            frozen_directions={"f1": 1},
            test_loader=load_test,
        )

    result = run_walk_forward_baselines(
        folds,
        candidates_factory=lambda fold: {
            baseline_id: _Candidate(baseline_id)
            for baseline_id in ("B0", "B1", "B2", "B3")
        },
        minimum_cross_section=30,
    )

    assert set(result.fold_results) == set(ORDINARY_TEST_FOLDS)
    assert [record.selected_baseline_id for record in result.selection_records] == ["B0"] * 4
    assert load_counts == {fold_id: 1 for fold_id in ORDINARY_TEST_FOLDS}
    assert result.evaluation.summary.valid_day_count == 8
    assert result.evaluation.gate_report.results["G4"].status.value == "NOT_APPLICABLE"
