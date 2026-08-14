"""Tests for the isolated canonical Alpha484 admission evaluator."""

from __future__ import annotations

from datetime import date

import evaluate_alpha484_canonical as evaluation
import pandas as pd
import pytest


def test_candidate_identity_is_isolated_from_historical_factor() -> None:
    assert evaluation.CANDIDATE_ID == "yeslab_484_tushare_shsz_v1"
    assert evaluation.SOURCE_FACTOR_ID == "yeslab_484"
    assert evaluation.CANDIDATE_ID != evaluation.SOURCE_FACTOR_ID


def test_parse_args_rejects_the_locked_test_boundary() -> None:
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--end", "2026-01-01", "--output", "/tmp/out"])


def test_validate_full_market_dates_rejects_a_missing_open_date() -> None:
    market = pd.DataFrame({"trade_date": [pd.Timestamp("2025-01-02")]})

    with pytest.raises(ValueError, match="full SH/SZ"):
        evaluation.validate_full_market_dates(
            market, (date(2025, 1, 2), date(2025, 1, 3))
        )


def test_summarize_coverage_rejects_a_month_below_seventy_percent() -> None:
    labels = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2025-01-02"] * 10 + ["2025-02-03"] * 10),
            "ts_code": [f"0000{index:02d}.SZ" for index in range(10)] * 2,
        }
    )
    values = labels.iloc[:16].rename(columns={"signal_date": "trade_date"})
    values["replay_value"] = 1.0

    result = evaluation.summarize_coverage(values, labels)

    assert result["status"] == "FAIL"
    assert result["monthly_min_ratio"] == pytest.approx(0.60)


def test_freeze_training_direction_uses_only_training_rank_ic() -> None:
    training = pd.DataFrame(
        {
            "signal_date": pd.date_range("2020-01-02", periods=3),
            "rank_ic": [-0.1, -0.2, 0.0],
        }
    )

    assert evaluation.freeze_training_direction(training) == -1


def test_build_review_never_makes_candidate_a_model_input() -> None:
    review = evaluation.build_review(
        training={
            "signed_mean_rank_ic": 0.04,
            "signed_icir": 0.60,
            "directional_day_share": 0.70,
            "annual_gate_years": 5,
        },
        validation={
            "signed_mean_rank_ic": 0.03,
            "signed_icir": 0.55,
            "directional_day_share": 0.60,
        },
        coverage={"status": "PASS"},
        train_similarity={"status": "PASS"},
        validation_similarity={"status": "PASS"},
    )

    assert review["status"] == "READY_FOR_EXPLICIT_ADMISSION_REVIEW"
    assert review["model_eligible"] is False
    assert review["formal_factor_set_changed"] is False
    assert review["locked_test_read"] is False
