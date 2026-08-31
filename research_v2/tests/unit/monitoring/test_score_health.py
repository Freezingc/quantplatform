from __future__ import annotations

import pandas as pd
import pytest

from research_v2.monitoring.score_health import evaluate_score_health
from research_v2.training.ensemble import REQUIRED_SEEDS


def _seed_ranks(values: list[float]) -> dict[int, pd.DataFrame]:
    return {
        seed: pd.DataFrame(
            {
                "signal_date": ["2025-01-02"] * len(values),
                "ts_code": [f"S{index:03d}" for index in range(len(values))],
                "percentile_rank": seed_values,
            }
        )
        for seed, seed_values in zip(
            REQUIRED_SEEDS,
            (values, values, values),
            strict=True,
        )
    }


def _prediction_frame(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": ["2025-01-02"] * count,
            "ts_code": [f"S{index:03d}" for index in range(count)],
            "ensemble_score": [float(index) for index in range(count)],
        }
    )


def test_score_health_blocks_duplicate_low_coverage_and_score_collapse() -> None:
    healthy = pd.DataFrame(
        {
            "signal_date": ["2025-01-02"] * 10,
            "ts_code": [f"S{index:03d}" for index in range(10)],
            "ensemble_score": [index / 10 for index in range(10)],
        }
    )

    report = evaluate_score_health(healthy, expected_key_count=10)
    assert not report.blocking_reasons
    duplicate = pd.concat([healthy, healthy.iloc[[0]]], ignore_index=True)
    assert "DUPLICATE_BUSINESS_KEY" in evaluate_score_health(
        duplicate,
        expected_key_count=10,
    ).blocking_reasons
    collapsed = healthy.assign(ensemble_score=0.5)
    assert "INSUFFICIENT_UNIQUE_SCORES" in evaluate_score_health(
        collapsed,
        expected_key_count=10,
    ).blocking_reasons


def test_score_health_reports_pairwise_spearman_for_all_required_seeds() -> None:
    predictions = _prediction_frame(10)
    report = evaluate_score_health(
        predictions,
        expected_key_count=10,
        seed_percentile_ranks=_seed_ranks([index / 10 for index in range(10)]),
    )

    expected_pairs = {
        (REQUIRED_SEEDS[0], REQUIRED_SEEDS[1]),
        (REQUIRED_SEEDS[0], REQUIRED_SEEDS[2]),
        (REQUIRED_SEEDS[1], REQUIRED_SEEDS[2]),
    }
    assert set(report.pairwise_spearman) == expected_pairs
    assert all(value == pytest.approx(1.0) for value in report.pairwise_spearman.values())
    assert not report.error_signals


def test_score_health_reports_low_seed_agreement_as_non_blocking_error() -> None:
    values = [index / 10 for index in range(10)]
    seed_ranks = _seed_ranks(values)
    seed_ranks[REQUIRED_SEEDS[1]] = seed_ranks[REQUIRED_SEEDS[1]].assign(
        percentile_rank=list(reversed(values))
    )

    report = evaluate_score_health(
        _prediction_frame(10),
        expected_key_count=10,
        seed_percentile_ranks=seed_ranks,
    )

    assert report.pairwise_spearman[(REQUIRED_SEEDS[0], REQUIRED_SEEDS[1])] == pytest.approx(-1.0)
    assert "SEED_AGREEMENT_ERROR" in report.error_signals
    assert "SEED_AGREEMENT_ERROR" not in report.blocking_reasons


def test_score_health_reports_nonfinite_seed_agreement_as_non_blocking_error() -> None:
    seed_ranks = _seed_ranks([index / 10 for index in range(10)])
    seed_ranks[REQUIRED_SEEDS[1]] = seed_ranks[REQUIRED_SEEDS[1]].assign(
        percentile_rank=0.5
    )

    report = evaluate_score_health(
        _prediction_frame(10),
        expected_key_count=10,
        seed_percentile_ranks=seed_ranks,
    )

    assert not pd.notna(report.pairwise_spearman[(REQUIRED_SEEDS[0], REQUIRED_SEEDS[1])])
    assert "SEED_AGREEMENT_ERROR" in report.error_signals
    assert "SEED_AGREEMENT_ERROR" not in report.blocking_reasons


def test_score_health_rejects_invalid_required_seed_coverage() -> None:
    predictions = _prediction_frame(10)
    seed_ranks = _seed_ranks([index / 10 for index in range(10)])

    with pytest.raises(ValueError, match="required seeds"):
        evaluate_score_health(
            predictions,
            expected_key_count=10,
            seed_percentile_ranks={REQUIRED_SEEDS[0]: seed_ranks[REQUIRED_SEEDS[0]]},
        )

    duplicate = pd.concat([seed_ranks[REQUIRED_SEEDS[0]], seed_ranks[REQUIRED_SEEDS[0]].iloc[[0]]])
    with pytest.raises(ValueError, match="duplicate"):
        evaluate_score_health(
            predictions,
            expected_key_count=10,
            seed_percentile_ranks={
                REQUIRED_SEEDS[0]: duplicate,
                REQUIRED_SEEDS[1]: seed_ranks[REQUIRED_SEEDS[1]],
                REQUIRED_SEEDS[2]: seed_ranks[REQUIRED_SEEDS[2]],
            },
        )

    mismatched = seed_ranks[REQUIRED_SEEDS[2]].iloc[:-1]
    with pytest.raises(ValueError, match="identical"):
        evaluate_score_health(
            predictions,
            expected_key_count=10,
            seed_percentile_ranks={
                REQUIRED_SEEDS[0]: seed_ranks[REQUIRED_SEEDS[0]],
                REQUIRED_SEEDS[1]: seed_ranks[REQUIRED_SEEDS[1]],
                REQUIRED_SEEDS[2]: mismatched,
            },
        )

    truncated = {
        seed: frame.iloc[:-1]
        for seed, frame in seed_ranks.items()
    }
    with pytest.raises(ValueError, match="prediction business-key"):
        evaluate_score_health(
            predictions,
            expected_key_count=10,
            seed_percentile_ranks=truncated,
        )


def test_score_health_blocks_extreme_missingness_pattern_collapse() -> None:
    predictions = _prediction_frame(200)
    patterns = ["baseline"] * 198 + ["collapsed"] * 2

    report = evaluate_score_health(
        predictions,
        expected_key_count=200,
        missingness_patterns=patterns,
    )

    assert "EXTREME_MISSINGNESS_PATTERN_COLLAPSE" in report.blocking_reasons


def test_score_health_skips_extreme_pattern_check_for_small_cross_sections() -> None:
    report = evaluate_score_health(
        _prediction_frame(50),
        expected_key_count=50,
        missingness_patterns=["only-pattern"] * 50,
    )

    assert "EXTREME_MISSINGNESS_PATTERN_COLLAPSE" not in report.blocking_reasons


def test_score_health_skips_extreme_pattern_check_when_each_tail_has_one_row() -> None:
    report = evaluate_score_health(
        _prediction_frame(100),
        expected_key_count=100,
        missingness_patterns=["only-pattern"] * 100,
    )

    assert "EXTREME_MISSINGNESS_PATTERN_COLLAPSE" not in report.blocking_reasons
