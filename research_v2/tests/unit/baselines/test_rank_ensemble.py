from __future__ import annotations

import pandas as pd
import pytest

from research_v2.baselines.rank_ensemble import directional_rank_scores


def test_directional_rank_scores_ranks_within_each_signal_date() -> None:
    features = pd.DataFrame(
        [
            {"signal_date": "2025-01-02", "ts_code": "A", "f1": 1.0, "f2": 10.0},
            {"signal_date": "2025-01-02", "ts_code": "B", "f1": 2.0, "f2": 20.0},
            {"signal_date": "2025-01-03", "ts_code": "A", "f1": 100.0, "f2": 1.0},
            {"signal_date": "2025-01-03", "ts_code": "B", "f1": 50.0, "f2": 2.0},
        ]
    )

    scores = directional_rank_scores(features, ("f1", "f2"), (-1, 1))

    assert scores.tolist() == pytest.approx([0.5, 0.5, 0.0, 1.0])
    assert scores.index.equals(features.index)


def test_directional_rank_scores_ignores_missing_factor_values() -> None:
    features = pd.DataFrame(
        [
            {"signal_date": "2025-01-02", "ts_code": "A", "f1": 1.0, "f2": None},
            {"signal_date": "2025-01-02", "ts_code": "B", "f1": 2.0, "f2": 20.0},
        ]
    )

    scores = directional_rank_scores(features, ("f1", "f2"), (-1, 1))

    assert scores.tolist() == pytest.approx([1.0, 0.25])


def test_directional_rank_scores_preserves_all_missing_rows_as_nan() -> None:
    features = pd.DataFrame(
        [{"signal_date": "2025-01-02", "ts_code": "A", "f1": None, "f2": None}]
    )

    scores = directional_rank_scores(features, ("f1", "f2"), (-1, 1))

    assert pd.isna(scores.iloc[0])
