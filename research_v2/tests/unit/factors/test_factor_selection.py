from __future__ import annotations

import pandas as pd

from research_v2.factors.selection import (
    FactorSelectionRule,
    select_factors_by_dual_threshold,
    select_stable_factors,
    select_tushare_canonical_stable_factors,
)


def test_dual_threshold_selection_requires_rank_ic_and_icir_gates() -> None:
    summary = pd.DataFrame(
        {
            "factor_id": ["f1", "f2", "f3", "f4"],
            "factor_version": [1, 1, 1, 1],
            "factor_name": ["good", "weak_ic", "weak_icir", "low_coverage"],
            "mean_rank_ic": [0.04, 0.01, 0.04, 0.05],
            "icir": [0.8, 1.0, 0.2, 0.9],
            "valid_day_count": [120, 120, 120, 20],
            "positive_day_share": [0.75, 0.75, 0.75, 0.75],
        }
    )

    selected = select_factors_by_dual_threshold(
        summary,
        FactorSelectionRule(
            min_abs_mean_rank_ic=0.02,
            min_abs_icir=0.5,
            min_valid_day_count=60,
            min_directional_day_share=0.60,
        ),
    )

    assert selected.factor_id.tolist() == ["f1"]
    assert selected.loc[0, "frozen_direction"] == 1
    assert selected.loc[0, "selection_reason"] == "DUAL_RANKIC_ICIR_GATE"


def test_dual_threshold_selection_freezes_negative_direction_deterministically() -> None:
    summary = pd.DataFrame(
        {
            "factor_id": ["f2"],
            "factor_version": [1],
            "factor_name": ["negative"],
            "mean_rank_ic": [-0.05],
            "icir": [-0.9],
            "valid_day_count": [80],
            "positive_day_share": [0.20],
        }
    )

    selected = select_factors_by_dual_threshold(summary, FactorSelectionRule())

    assert selected.loc[0, "frozen_direction"] == -1


def test_stable_selection_requires_both_periods_and_direction_agreement() -> None:
    train = pd.DataFrame(
        {
            "factor_id": ["f1", "f2"],
            "factor_version": [1, 1],
            "factor_name": ["stable", "flip"],
            "mean_rank_ic": [0.04, 0.05],
            "icir": [0.8, 0.9],
            "valid_day_count": [80, 80],
            "positive_day_share": [0.75, 0.75],
        }
    )
    validation = train.copy()
    validation.loc[1, "mean_rank_ic"] = -0.05
    validation.loc[1, "icir"] = -0.9
    validation.loc[1, "positive_day_share"] = 0.25

    selected = select_stable_factors(
        train,
        validation,
        FactorSelectionRule(min_valid_day_count=60),
    )

    assert selected.factor_id.tolist() == ["f1"]
    assert selected.loc[0, "selection_reason"] == "DUAL_GATE_STABLE_DIRECTION"


def test_dual_threshold_selection_excludes_explicit_factor_ids() -> None:
    summary = pd.DataFrame(
        {
            "factor_id": ["production", "synthetic"],
            "factor_version": [1, 1],
            "factor_name": ["good", "test_factor"],
            "mean_rank_ic": [0.04, 0.05],
            "icir": [0.8, 0.9],
            "valid_day_count": [120, 120],
            "positive_day_share": [0.75, 0.75],
        }
    )

    selected = select_factors_by_dual_threshold(
        summary,
        FactorSelectionRule(excluded_factor_ids=frozenset({"synthetic"})),
    )

    assert selected.factor_id.tolist() == ["production"]


def test_canonical_selection_requires_every_replay_window_to_pass() -> None:
    train = pd.DataFrame(
        {
            "factor_id": ["f1", "f2"],
            "factor_version": [1, 1],
            "factor_name": ["pass", "blocked"],
            "mean_rank_ic": [0.04, 0.05],
            "icir": [0.8, 0.9],
            "valid_day_count": [80, 80],
            "positive_day_share": [0.75, 0.75],
        }
    )
    validation = train.copy()
    replay = pd.DataFrame(
        {
            "factor_id": ["f1", "f1", "f2", "f2"],
            "window": ["2020", "2025", "2020", "2025"],
            "decision_status": [
                "FORMULA_REPLAY_PASS",
                "FORMULA_REPLAY_PASS",
                "FORMULA_REPLAY_PASS",
                "FORMULA_REPLAY_BLOCKED",
            ],
        }
    )

    selected = select_tushare_canonical_stable_factors(train, validation, replay)

    assert selected.factor_id.tolist() == ["f1"]
    assert selected.loc[0, "canonical_replay_status"] == "FORMULA_REPLAY_PASS"


def test_canonical_selection_fails_closed_for_missing_replay_rows() -> None:
    summary = pd.DataFrame(
        {
            "factor_id": ["f1"],
            "factor_version": [1],
            "factor_name": ["missing_audit"],
            "mean_rank_ic": [0.04],
            "icir": [0.8],
            "valid_day_count": [80],
            "positive_day_share": [0.75],
        }
    )

    selected = select_tushare_canonical_stable_factors(
        summary, summary.copy(), pd.DataFrame(columns=["factor_id", "decision_status"])
    )

    assert selected.empty
