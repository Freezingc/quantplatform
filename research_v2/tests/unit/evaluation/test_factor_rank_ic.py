from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.evaluation.factor_rank_ic import (
    compute_factor_daily_rank_ic,
    summarize_factor_rank_ic,
)


def test_compute_factor_daily_rank_ic_returns_one_row_per_factor_and_date() -> None:
    factors = pd.DataFrame(
        {
            "signal_date": [date(2024, 1, 2)] * 4,
            "ts_code": ["A", "B", "C", "D"],
            "factor_name": ["f1", "f1", "f1", "f1"],
            "value": [1.0, 2.0, 3.0, 4.0],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2024, 1, 2)] * 4,
            "ts_code": ["A", "B", "C", "D"],
            "raw_return": [0.4, 0.3, 0.2, 0.1],
            "label_valid": [True] * 4,
        }
    )

    result = compute_factor_daily_rank_ic(factors, labels, minimum_cross_section=3)

    assert list(result.columns) == ["factor_name", "signal_date", "rank_ic", "sample_count"]
    assert result.loc[0, "factor_name"] == "f1"
    assert result.loc[0, "rank_ic"] == pytest.approx(-1.0)
    assert result.loc[0, "sample_count"] == 4


def test_factor_summary_uses_daily_values_and_counts_coverage() -> None:
    daily = pd.DataFrame(
        {
            "factor_name": ["f1", "f1", "f2"],
            "signal_date": [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 2)],
            "rank_ic": [0.2, 0.4, -0.1],
            "sample_count": [100, 90, 80],
        }
    )

    result = summarize_factor_rank_ic(daily)

    f1 = result.loc[result["factor_name"] == "f1"].iloc[0]
    assert f1["valid_day_count"] == 2
    assert f1["mean_rank_ic"] == pytest.approx(0.3)
    assert f1["icir"] == pytest.approx(0.3 / (0.02**0.5))
    assert f1["mean_sample_count"] == pytest.approx(95.0)


def test_compute_factor_daily_rank_ic_skips_constant_cross_sections() -> None:
    factors = pd.DataFrame(
        {
            "signal_date": [date(2024, 1, 2)] * 3,
            "ts_code": ["A", "B", "C"],
            "factor_name": ["flat"] * 3,
            "value": [1.0, 1.0, 1.0],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2024, 1, 2)] * 3,
            "ts_code": ["A", "B", "C"],
            "raw_return": [0.1, 0.2, 0.3],
            "label_valid": [True] * 3,
        }
    )

    result = compute_factor_daily_rank_ic(factors, labels, minimum_cross_section=3)

    assert result.empty
