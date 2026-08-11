from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.evaluation.rank_ic import compute_daily_rank_ic, summarize_rank_ic


def test_rank_ic_is_computed_cross_sectionally_per_signal_date() -> None:
    predictions = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 3 + [date(2025, 1, 3)] * 3,
            "ts_code": ["a", "b", "c"] * 2,
            "score": [1.0, 2.0, 3.0, 1.0, 2.0, 3.0],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 3 + [date(2025, 1, 3)] * 3,
            "ts_code": ["a", "b", "c"] * 2,
            "raw_return_5d": [1.0, 2.0, 3.0, 3.0, 2.0, 1.0],
            "label_valid": [True] * 6,
        }
    )

    daily = compute_daily_rank_ic(predictions, labels, minimum_cross_section=3)
    summary = summarize_rank_ic(daily)

    assert daily["rank_ic"].tolist() == pytest.approx([1.0, -1.0])
    assert summary.mean_rank_ic == pytest.approx(0.0)
    assert summary.icir == pytest.approx(0.0)


def test_rank_ic_excludes_invalid_labels_and_undersized_cross_sections() -> None:
    predictions = pd.DataFrame(
        {"signal_date": [date(2025, 1, 2)] * 2, "ts_code": ["a", "b"], "score": [1.0, 2.0]}
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 2,
            "ts_code": ["a", "b"],
            "raw_return_5d": [1.0, 2.0],
            "label_valid": [True, False],
        }
    )

    daily = compute_daily_rank_ic(predictions, labels, minimum_cross_section=3)

    assert daily.empty
