from __future__ import annotations

import evaluate_alpha95_relative_amount_volatility as evaluation
import numpy as np
import pandas as pd
import pytest


def make_market(values: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(values))
    return pd.DataFrame(
        {
            "trade_date": dates.repeat(2),
            "ts_code": ["000001.SZ", "600000.SH"] * len(values),
            "amount": np.repeat(values, 2),
        }
    )


def test_relative_amount_volatility_uses_complete_trailing_window() -> None:
    market = pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=4).repeat(2),
            "ts_code": ["000001.SZ", "600000.SH"] * 4,
            "amount": [1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0],
        }
    )

    result = evaluation.compute_relative_amount_volatility(market, window=3)

    assert result.iloc[:2].isna().all().all()
    np.testing.assert_allclose(result.iloc[2].to_numpy(), [0.5, 0.5])


def test_relative_amount_volatility_replaces_zero_baseline_with_missing() -> None:
    result = evaluation.compute_relative_amount_volatility(make_market([0.0, 0.0, 0.0]), 3)

    assert result.iloc[2].isna().all()


def test_relative_amount_volatility_does_not_use_future_amount() -> None:
    base = make_market([1.0, 2.0, 3.0, 4.0, 5.0])
    changed = base.copy()
    changed.loc[changed["trade_date"] == pd.Timestamp("2025-01-05"), "amount"] = 999.0

    before = evaluation.compute_relative_amount_volatility(base, 3)
    after = evaluation.compute_relative_amount_volatility(changed, 3)

    assert before.iloc[2].equals(after.iloc[2])


def test_select_training_candidate_requires_every_gate() -> None:
    rows = {
        "v10": {
            "signed_mean_rank_ic": 0.04,
            "signed_icir": 0.60,
            "directional_day_share": 0.70,
            "annual_double_gate_pass_years": 4,
            "max_similarity": 0.40,
        },
        "v20": {
            "signed_mean_rank_ic": 0.06,
            "signed_icir": 0.70,
            "directional_day_share": 0.70,
            "annual_double_gate_pass_years": 3,
            "max_similarity": 0.30,
        },
    }

    assert evaluation.select_training_candidate(rows) == "v10"


def test_select_training_candidate_breaks_icir_tie_by_lower_similarity() -> None:
    rows = {
        "v10": {
            "signed_mean_rank_ic": 0.04,
            "signed_icir": 0.60,
            "directional_day_share": 0.70,
            "annual_double_gate_pass_years": 4,
            "max_similarity": 0.30,
        },
        "v20": {
            "signed_mean_rank_ic": 0.05,
            "signed_icir": 0.60,
            "directional_day_share": 0.70,
            "annual_double_gate_pass_years": 4,
            "max_similarity": 0.20,
        },
    }

    assert evaluation.select_training_candidate(rows) == "v20"


def test_parse_args_rejects_locked_test_dates() -> None:
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--end", "2026-01-01", "--output", "/tmp/result"])


def test_build_review_rejects_validation_without_changing_formal_set() -> None:
    review = evaluation.build_review(
        selected_id="yeslab_163_relative_amount_volatility_20_v1",
        training={"selected": True},
        validation={
            "signed_mean_rank_ic": 0.01,
            "signed_icir": 0.60,
            "directional_day_share": 0.70,
            "max_similarity": 0.20,
        },
    )

    assert review["status"] == "REJECT_VALIDATION_GATE"
    assert review["formal_factor_set_changed"] is False
    assert review["locked_test_read"] is False
    assert review["model_training_started"] is False


def test_summarize_training_candidate_freezes_direction_and_counts_annual_gates() -> None:
    daily = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(
                ["2020-01-02", "2020-01-03", "2021-01-04", "2021-01-05"]
            ),
            "rank_ic": [-0.10, -0.20, -0.10, -0.20],
            "sample_count": [100, 100, 100, 100],
        }
    )

    summary = evaluation.summarize_training_candidate(daily)

    assert summary["direction"] == -1
    assert summary["annual_double_gate_pass_years"] == 2
    assert summary["signed_mean_rank_ic"] == pytest.approx(0.15)
    assert summary["signed_icir"] > 0.50


def test_max_formal_similarity_selects_only_formal_edges() -> None:
    pairs = pd.DataFrame(
        {
            "factor_id_left": ["candidate", "candidate", "candidate"],
            "factor_id_right": ["yeslab_380", "yeslab_431", "other_candidate"],
            "median_abs_spearman": [0.30, 0.70, 0.95],
            "valid_day_count": [100, 100, 100],
        }
    )

    result = evaluation.max_formal_similarity(pairs, "candidate")

    assert result == {"factor_id": "yeslab_431", "value": 0.70, "valid_day_count": 100}


def test_max_similarity_to_references_includes_other_alpha95_variants() -> None:
    pairs = pd.DataFrame(
        {
            "factor_id_left": ["candidate", "candidate"],
            "factor_id_right": ["yeslab_431", "other_variant"],
            "median_abs_spearman": [0.70, 0.80],
            "valid_day_count": [100, 100],
        }
    )

    result = evaluation.max_similarity_to_references(
        pairs,
        "candidate",
        {"yeslab_431", "other_variant"},
    )

    assert result == {"factor_id": "other_variant", "value": 0.80, "valid_day_count": 100}


def test_panel_to_replay_values_excludes_warmup_dates() -> None:
    panel = pd.DataFrame(
        {"000001.SZ": [1.0, 2.0, 3.0]},
        index=pd.to_datetime(["2019-12-31", "2020-01-02", "2020-01-03"]),
    )

    result = evaluation._panel_to_replay_values(
        panel,
        start=pd.Timestamp("2020-01-02"),
        end=pd.Timestamp("2020-01-03"),
    )

    assert result["trade_date"].min() == pd.Timestamp("2020-01-02")
    assert result["trade_date"].max() == pd.Timestamp("2020-01-03")
    assert len(result) == 2
