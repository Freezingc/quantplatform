from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.evaluation.walk_forward import OosFold, evaluate_walk_forward_oos


def _fold(year: int) -> OosFold:
    rows: list[dict[str, object]] = []
    baseline: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    signal_date = date(year, 1, 2)
    for index in range(30):
        key = {"signal_date": signal_date, "ts_code": f"S{index:03d}"}
        rows.append({**key, "score": float(index)})
        baseline.append({**key, "score": float(-index)})
        labels.append({**key, "raw_return_5d": float(index), "label_valid": True})
    return OosFold(
        candidate_predictions=pd.DataFrame(rows),
        labels=pd.DataFrame(labels),
        baseline_predictions=pd.DataFrame(baseline),
    )


def test_walk_forward_requires_all_six_ordinary_test_folds() -> None:
    folds = {f"F{index:02d}": _fold(2019 + index) for index in range(1, 7)}

    evaluation = evaluate_walk_forward_oos(folds)

    assert evaluation.summary.valid_day_count == 6
    assert evaluation.delta_rank_ic == pytest.approx(2.0)
    with pytest.raises(ValueError, match="F06"):
        evaluate_walk_forward_oos({key: value for key, value in folds.items() if key != "F06"})


def test_walk_forward_rejects_a_fold_with_dates_outside_its_test_year() -> None:
    folds = {f"F{index:02d}": _fold(2019 + index) for index in range(1, 7)}
    folds["F03"] = _fold(2025)

    with pytest.raises(ValueError, match="F03|2022"):
        evaluate_walk_forward_oos(folds)
