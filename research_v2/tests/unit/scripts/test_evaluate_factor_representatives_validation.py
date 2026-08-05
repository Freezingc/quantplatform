from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest
from evaluate_factor_representatives_validation import (
    build_fixed_direction_validation_report,
    normalize_validation_daily,
)


def test_validation_report_preserves_train_direction_and_marks_missing_factor() -> None:
    dates = pd.date_range("2025-01-02", periods=80, freq="B")
    daily = pd.DataFrame(
        {
            "factor_name": ["yeslab_1"] * len(dates),
            "signal_date": dates,
            "rank_ic": -0.05 + np.sin(np.arange(len(dates))) * 0.01,
        }
    )

    result = build_fixed_direction_validation_report(
        daily,
        factor_ids=("yeslab_1", "yeslab_2"),
        train_directions={"yeslab_1": -1, "yeslab_2": 1},
    ).set_index("factor_id")

    assert result.loc["yeslab_1", "direction"] == -1
    assert result.loc["yeslab_1", "signed_mean_rank_ic"] > 0
    assert result.loc["yeslab_2", "decision_status"] == "VALIDATION_2025_NO_RANKIC"
    assert "NO_VALIDATION_RANKIC" in result.loc["yeslab_2", "reason_codes"]


def test_validation_daily_rejects_locked_test_dates() -> None:
    daily = pd.DataFrame(
        {
            "factor_id": ["yeslab_1"],
            "signal_date": [date(2026, 1, 2)],
            "rank_ic": [0.1],
        }
    )

    with pytest.raises(ValueError, match="outside validation window"):
        normalize_validation_daily(
            daily,
            factor_ids=("yeslab_1",),
            start=date(2025, 1, 1),
            end=date(2025, 12, 31),
        )
