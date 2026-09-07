from __future__ import annotations

from datetime import date

import pandas as pd

from research_v2.baselines.base import BaselineDataset, PredictionFrame
from research_v2.baselines.orchestration import (
    BaselineFoldResult,
    BaselineSelectionRecord,
    BaselineTestEvaluation,
    BaselineValidationScore,
)
from research_v2.evaluation.rank_ic import RankICSummary
from research_v2.evaluation.walk_forward import (
    BaselineWalkForwardResult,
    OosFold,
    evaluate_walk_forward_oos,
)
from research_v2.reporting.experiment import build_experiment_report


def _oos_fold(year: int, fold_index: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    signal_date = date(year, 1, 2)
    # Two test dates per fold so four folds clear the >5 valid-day OOS gate.
    dates = [signal_date] * 30 + [date(year, 1, 3)] * 30
    keys = {
        "signal_date": dates,
        "ts_code": [f"S{index:03d}" for index in range(30)] * 2,
    }
    dataset_id = f"dataset-{year}"
    fold_id = f"F{fold_index}"
    labels = pd.DataFrame(
        {
            **keys,
            "raw_return_5d": [float(index) for index in range(30)] * 2,
            "label_valid": True,
            "fold_id": fold_id,
            "dataset_id": dataset_id,
        }
    )
    predictions = pd.DataFrame(
        {
            **keys,
            "score": [float(index + fold_index) for index in range(30)] * 2,
            "model_id": "candidate",
            "fold_id": fold_id,
            "dataset_id": dataset_id,
        }
    )
    return predictions, labels


def test_experiment_report_contains_fold_and_overall_evidence() -> None:
    oos_folds = {}
    fold_results: dict[str, BaselineFoldResult] = {}
    selections: list[BaselineSelectionRecord] = []
    for fold_index, fold_id in enumerate(("F1", "F2", "F3", "F4"), start=1):
        prediction, labels = _oos_fold(2021 + fold_index, fold_index)
        oos_folds[fold_id] = OosFold(
            candidate_predictions=prediction,
            labels=labels,
            baseline_predictions=prediction,
        )
        selection = BaselineSelectionRecord(
            fold_id=fold_id,
            selected_baseline_id="B0",
            validation_scores=tuple(
                BaselineValidationScore(model_id, 0.1 if model_id == "B0" else 0.05, 1, 0)
                for model_id in ("B0", "B1", "B2", "B3")
            ),
            unselected_reasons=(
                ("B1", "LOWER_VALIDATION_RANKIC"),
                ("B2", "LOWER_VALIDATION_RANKIC"),
                ("B3", "LOWER_VALIDATION_RANKIC"),
            ),
        )
        baseline_dataset = BaselineDataset(
            "TEST",
            fold_id,
            f"{fold_id}-test",
            prediction.loc[:, ["signal_date", "ts_code", "score"]].rename(
                columns={"score": "f1"}
            ),
            labels,
        )
        keyed_prediction = PredictionFrame.from_scores(
            baseline_dataset.features,
            baseline_dataset.features["f1"],
            model_id="B0",
            fold_id=fold_id,
            dataset_id=baseline_dataset.dataset_id,
        )
        test_evaluation = BaselineTestEvaluation(
            baseline_id="B0",
            predictions=keyed_prediction,
            labels=labels,
            daily_rank_ic=pd.DataFrame(
                {
                    "signal_date": [date(2019 + fold_index, 1, 2)],
                    "rank_ic": [1.0],
                    "sample_count": [30],
                    "coverage": [1.0],
                    "prediction_unique_count": [30],
                    "prediction_std": [8.0],
                    "rank_ic_valid": [True],
                }
            ),
            summary=RankICSummary(
                valid_day_count=1,
                mean_rank_ic=1.0,
                rank_ic_std=float("nan"),
                icir=float("nan"),
            ),
        )
        selections.append(selection)
        fold_results[fold_id] = BaselineFoldResult(selection, test_evaluation)

    evaluation = evaluate_walk_forward_oos(oos_folds)
    report = build_experiment_report(
        BaselineWalkForwardResult(fold_results, tuple(selections), evaluation),
        experiment_id="sha256:" + "a" * 64,
        fold_metadata={
            fold_id: {
                "train_years": [2014],
                "validation_years": [2019],
                "test_years": [2020],
                "purge": 6,
            }
            for fold_id in oos_folds
        },
        factor_set={"factor_set_id": "set-v1", "factors": ["f1"]},
        preprocessing={"hash": "preprocess-v1"},
        environment={"python": "3.11"},
        artifact_links={"dataset": "artifacts/dataset-v1"},
    )

    payload = report.payload
    assert payload["experiment_id"] == "sha256:" + "a" * 64
    assert set(payload["folds"]) == set(oos_folds)
    assert set(payload["folds"]["F1"]["baselines"]) == {"B0", "B1", "B2", "B3"}
    assert payload["folds"]["F1"]["baselines"]["B0"]["test"]["evaluated"] is True
    assert payload["folds"]["F1"]["baselines"]["B1"]["test"]["evaluated"] is False
    assert len(payload["overall_oos"]["yearly_summary"]) == 4
    assert payload["overall_oos"]["hac"]["max_lags"] == 5
    assert set(payload["overall_oos"]["gates"]) == {"G1", "G2", "G3", "G4", "G5", "G6"}
    assert payload["overall_oos"]["score_health"]["daily"][0]["coverage"] == 1.0
    assert "overall_oos" in report.to_html()
