from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.evaluation.walk_forward import OosFold, evaluate_walk_forward_oos


def _fold(year: int) -> OosFold:
    rows: list[dict[str, object]] = []
    baseline: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    # Two test dates per fold: evaluate_frozen_oos refuses OOS with <=5 valid days,
    # and four folds at one date each only reach four.
    for signal_date in (date(year, 1, 2), date(year, 1, 3)):
        for index in range(30):
            key = {"signal_date": signal_date, "ts_code": f"S{index:03d}"}
            fold_id = f"F{year - 2021}"
            dataset_id = f"dataset-{year}"
            rows.append(
                {
                    **key,
                    "score": float(index),
                    "model_id": "candidate",
                    "fold_id": fold_id,
                    "dataset_id": dataset_id,
                }
            )
            baseline.append(
                {
                    **key,
                    "score": float(-index),
                    "model_id": "baseline",
                    "fold_id": fold_id,
                    "dataset_id": dataset_id,
                }
            )
            labels.append(
                {
                    **key,
                    "raw_return_5d": float(index),
                    "label_valid": True,
                    "fold_id": fold_id,
                    "dataset_id": dataset_id,
                }
            )
    return OosFold(
        candidate_predictions=pd.DataFrame(rows),
        labels=pd.DataFrame(labels),
        baseline_predictions=pd.DataFrame(baseline),
    )


def test_walk_forward_requires_all_four_ordinary_test_folds() -> None:
    folds = {f"F{index}": _fold(2021 + index) for index in range(1, 5)}

    evaluation = evaluate_walk_forward_oos(folds)

    assert evaluation.summary.valid_day_count == 8
    assert evaluation.delta_rank_ic == pytest.approx(2.0)
    with pytest.raises(ValueError, match="F4"):
        evaluate_walk_forward_oos({key: value for key, value in folds.items() if key != "F4"})


def test_walk_forward_rejects_a_fold_with_dates_outside_its_test_year() -> None:
    folds = {f"F{index}": _fold(2021 + index) for index in range(1, 5)}
    folds["F3"] = _fold(2022)

    with pytest.raises(ValueError, match="F3|2024"):
        evaluate_walk_forward_oos(folds)


def test_walk_forward_rejects_mismatched_label_and_prediction_keys() -> None:
    folds = {f"F{index}": _fold(2021 + index) for index in range(1, 5)}
    mismatched = folds["F1"].labels.copy()
    mismatched.loc[0, "ts_code"] = "MISSING"
    folds["F1"] = OosFold(
        candidate_predictions=folds["F1"].candidate_predictions,
        labels=mismatched,
        baseline_predictions=folds["F1"].baseline_predictions,
    )

    with pytest.raises(ValueError, match="identical business keys"):
        evaluate_walk_forward_oos(folds)


def test_walk_forward_rejects_empty_oos_frames() -> None:
    folds = {f"F{index}": _fold(2021 + index) for index in range(1, 5)}
    empty = folds["F1"].candidate_predictions.iloc[0:0].copy()
    folds["F1"] = OosFold(
        candidate_predictions=empty,
        labels=folds["F1"].labels,
        baseline_predictions=folds["F1"].baseline_predictions,
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate_walk_forward_oos(folds)
