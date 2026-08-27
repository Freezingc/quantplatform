from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.evaluation.rank_ic import (
    compute_daily_rank_ic,
    summarize_rank_ic,
    summarize_rank_ic_by_year,
)


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

    assert list(daily.columns) == ["signal_date", "rank_ic", "sample_count"]
    assert isinstance(daily.loc[0, "signal_date"], date)
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


def test_rank_ic_normalizes_intraday_timestamps_before_matching_business_keys() -> None:
    predictions = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2025-01-02 09:30")] * 3,
            "ts_code": ["a", "b", "c"],
            "score": [1.0, 2.0, 3.0],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": [pd.Timestamp("2025-01-02 15:00")] * 3,
            "ts_code": ["a", "b", "c"],
            "raw_return_5d": [1.0, 2.0, 3.0],
            "label_valid": [True] * 3,
        }
    )

    daily = compute_daily_rank_ic(predictions, labels, minimum_cross_section=3)

    assert daily["rank_ic"].tolist() == pytest.approx([1.0])
    assert daily.loc[0, "signal_date"] == date(2025, 1, 2)


def test_rank_ic_reports_score_health_and_excludes_collapsed_days() -> None:
    predictions = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 30 + [date(2025, 1, 3)] * 30,
            "ts_code": [f"S{index:03d}" for index in range(30)] * 2,
            "score": list(range(30)) + [1.0] * 30,
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 30 + [date(2025, 1, 3)] * 30,
            "ts_code": [f"S{index:03d}" for index in range(30)] * 2,
            "raw_return_5d": list(range(30)) * 2,
            "label_valid": [True] * 60,
        }
    )

    daily = compute_daily_rank_ic(
        predictions,
        labels,
        minimum_cross_section=30,
        include_invalid=True,
        require_score_health=True,
    )
    summary = summarize_rank_ic(daily)
    yearly = summarize_rank_ic_by_year(daily)

    assert daily["coverage"].tolist() == pytest.approx([1.0, 1.0])
    assert daily["prediction_unique_count"].tolist() == [30, 1]
    assert daily.loc[0, "prediction_std"] > 1e-6
    assert daily.loc[0, "rank_ic_valid"]
    assert not daily.loc[1, "rank_ic_valid"]
    assert pd.isna(daily.loc[1, "rank_ic"])
    assert summary.valid_day_count == 1
    assert summary.invalid_day_count == 1
    assert yearly.loc[0, "year"] == 2025
    assert yearly.loc[0, "valid_day_count"] == 1
    assert yearly.loc[0, "invalid_day_count"] == 1
    assert yearly.loc[0, "mean_rank_ic"] == pytest.approx(1.0)
    assert pd.isna(yearly.loc[0, "rank_ic_std"])
    assert pd.isna(yearly.loc[0, "icir"])
    assert yearly.loc[0, "positive_day_share"] == pytest.approx(1.0)
    assert yearly.loc[0, "mean_sample_count"] == pytest.approx(30.0)
    assert yearly.loc[0, "sample_count_p05"] == pytest.approx(30.0)
    assert yearly.loc[0, "sample_count_median"] == pytest.approx(30.0)
    assert yearly.loc[0, "sample_count_p95"] == pytest.approx(30.0)


def test_rank_ic_excludes_near_collapsed_scores_from_metric_evidence() -> None:
    predictions = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 30,
            "ts_code": [f"S{index:03d}" for index in range(30)],
            "score": [float(index % 9) for index in range(30)],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)] * 30,
            "ts_code": [f"S{index:03d}" for index in range(30)],
            "raw_return_5d": list(range(30)),
            "label_valid": [True] * 30,
        }
    )

    daily = compute_daily_rank_ic(
        predictions,
        labels,
        minimum_cross_section=30,
        include_invalid=True,
    )

    assert daily.loc[0, "prediction_unique_count"] == 9
    assert not daily.loc[0, "rank_ic_valid"]
    assert pd.isna(daily.loc[0, "rank_ic"])
