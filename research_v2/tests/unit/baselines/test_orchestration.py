from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.baselines.base import BaselineDataset, PredictionFrame
from research_v2.baselines.lgbm_ranker import LGBMRankerCandidate
from research_v2.baselines.linear import LinearBaselineCandidate
from research_v2.baselines.orchestration import (
    BaselineFold,
    build_default_baseline_candidates,
    run_baseline_fold,
)


def _dataset(role: str, dataset_id: str, year: int, reverse: bool = False) -> BaselineDataset:
    rows: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    for day_index in range(2):
        signal_date = date(year, 1, 2 + day_index)
        for stock_index in range(3):
            code = f"S{stock_index}"
            rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": code,
                    "f1": float(stock_index),
                }
            )
            label_value = float(stock_index if not reverse else 2 - stock_index)
            labels.append(
                {
                    "signal_date": signal_date,
                    "ts_code": code,
                    "raw_return_5d": label_value,
                    "label_valid": True,
                }
            )
    return BaselineDataset(
        role=role,
        fold_id="F01",
        dataset_id=dataset_id,
        features=pd.DataFrame(rows),
        labels=pd.DataFrame(labels),
    )


class _FakeFitted:
    def __init__(self, model_id: str, direction: int, events: list[str]) -> None:
        self.model_id = model_id
        self._direction = direction
        self._events = events

    def predict(self, data: BaselineDataset) -> PredictionFrame:
        self._events.append(f"predict:{self.model_id}:{data.role}")
        scores = data.features["f1"] * self._direction
        return PredictionFrame.from_scores(
            data.features,
            scores,
            model_id=self.model_id,
            fold_id=data.fold_id,
            dataset_id=data.dataset_id,
        )


class _FakeBaseline:
    def __init__(self, model_id: str, direction: int, events: list[str]) -> None:
        self.model_id = model_id
        self._direction = direction
        self._events = events

    def fit(self, train: BaselineDataset, validation: BaselineDataset) -> _FakeFitted:
        self._events.append(f"fit:{self.model_id}:{train.role}:{validation.role}")
        return _FakeFitted(self.model_id, self._direction, self._events)


def _candidates(events: list[str]) -> dict[str, _FakeBaseline]:
    return {
        model_id: _FakeBaseline(model_id, direction, events)
        for model_id, direction in (("B0", 1), ("B1", 1), ("B2", -1), ("B3", 1))
    }


def test_fold_selection_uses_validation_tie_break_and_delays_test_access() -> None:
    events: list[str] = []
    records = []
    test = _dataset("TEST", "test-v1", 2025)

    def load_test() -> BaselineDataset:
        events.append("load-test")
        return test

    def record_selection(record: object) -> None:
        records.append(record)
        events.append("selection-recorded")

    result = run_baseline_fold(
        BaselineFold(
            fold_id="F01",
            train=_dataset("TRAIN", "train-v1", 2020),
            validation=_dataset("VALIDATION", "validation-v1", 2024),
            factor_order=("f1",),
            frozen_directions={"f1": 1},
            test_loader=load_test,
        ),
        candidates=_candidates(events),
        minimum_cross_section=3,
        selection_recorder=record_selection,
    )

    assert records == [result.selection_record]
    assert result.selection_record.selected_baseline_id == "B0"
    assert [item.baseline_id for item in result.selection_record.validation_scores] == [
        "B0",
        "B1",
        "B2",
        "B3",
    ]
    assert result.test_evaluation is not None
    assert result.test_evaluation.predictions.frame["model_id"].unique().tolist() == ["B0"]
    test_load_index = events.index("load-test")
    assert events.index("selection-recorded") < test_load_index
    assert test_load_index > max(
        index
        for index, event in enumerate(events)
        if event.startswith("predict:") and event.endswith(":VALIDATION")
    )
    assert [event for event in events if event.endswith(":TEST")] == ["predict:B0:TEST"]


def test_test_label_perturbation_cannot_change_validation_selection() -> None:
    events_a: list[str] = []
    events_b: list[str] = []

    result_a = run_baseline_fold(
        BaselineFold(
            fold_id="F01",
            train=_dataset("TRAIN", "train-v1", 2020),
            validation=_dataset("VALIDATION", "validation-v1", 2024),
            factor_order=("f1",),
            frozen_directions={"f1": 1},
            test_loader=lambda: _dataset("TEST", "test-v1", 2025, reverse=False),
        ),
        candidates=_candidates(events_a),
        minimum_cross_section=3,
        selection_recorder=lambda record: None,
    )
    result_b = run_baseline_fold(
        BaselineFold(
            fold_id="F01",
            train=_dataset("TRAIN", "train-v1", 2020),
            validation=_dataset("VALIDATION", "validation-v1", 2024),
            factor_order=("f1",),
            frozen_directions={"f1": 1},
            test_loader=lambda: _dataset("TEST", "test-v1", 2025, reverse=True),
        ),
        candidates=_candidates(events_b),
        minimum_cross_section=3,
        selection_recorder=lambda record: None,
    )

    assert result_a.selection_record == result_b.selection_record
    assert result_a.test_evaluation is not None
    assert result_b.test_evaluation is not None
    assert (
        result_a.test_evaluation.summary.mean_rank_ic
        == pytest.approx(-result_b.test_evaluation.summary.mean_rank_ic)
    )


def test_fold_requires_all_four_simple_baselines_before_selection() -> None:
    with pytest.raises(ValueError, match="B3"):
        run_baseline_fold(
            BaselineFold(
                fold_id="F01",
                train=_dataset("TRAIN", "train-v1", 2020),
                validation=_dataset("VALIDATION", "validation-v1", 2024),
                factor_order=("f1",),
                frozen_directions={"f1": 1},
            ),
            candidates={key: value for key, value in _candidates([]).items() if key != "B3"},
            minimum_cross_section=3,
        )


def test_fold_refuses_test_access_without_a_selection_recorder() -> None:
    with pytest.raises(ValueError, match="selection_recorder"):
        run_baseline_fold(
            BaselineFold(
                fold_id="F01",
                train=_dataset("TRAIN", "train-v1", 2020),
                validation=_dataset("VALIDATION", "validation-v1", 2024),
                factor_order=("f1",),
                frozen_directions={"f1": 1},
                test_loader=lambda: _dataset("TEST", "test-v1", 2025),
            ),
            candidates=_candidates([]),
            minimum_cross_section=3,
        )


def test_default_candidates_propagate_the_minimum_cross_section_to_b2_and_b3() -> None:
    fold = BaselineFold(
        fold_id="F01",
        train=_dataset("TRAIN", "train-v1", 2020),
        validation=_dataset("VALIDATION", "validation-v1", 2024),
        factor_order=("f1",),
        frozen_directions={"f1": 1},
    )

    candidates = build_default_baseline_candidates(fold, minimum_cross_section=17)

    assert isinstance(candidates["B2"], LinearBaselineCandidate)
    assert candidates["B2"].minimum_cross_section == 17
    assert isinstance(candidates["B3"], LGBMRankerCandidate)
    assert candidates["B3"].minimum_cross_section == 17
