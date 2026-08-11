from __future__ import annotations

from datetime import date

import pandas as pd

from research_v2.baselines.single_factor import select_single_factor


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
