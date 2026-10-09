"""Contract tests for the portfolio policy layer.

The rules here are the ones the 2022-2025 backtest measured, so these tests pin the exact
semantics the replay verification depends on: buffer ordering, tie-breaking, cold-start
exposure, and the refusal of the two configurations that were measured and refuted.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from research_v2.portfolio.policy import (
    DEFAULT_PORTFOLIO_POLICY,
    WEIGHTING_EQUAL_RAW_SCORE,
    PortfolioPolicy,
    combine_scores,
    cost_of,
    load_portfolio_policy,
    position_weights,
    rank_descending,
    select_holdings,
    trend_exposure,
    turnover,
)

CONFIG = Path(__file__).resolve().parents[3] / "configs" / "portfolio" / "default.yaml"


# --------------------------------------------------------------------- validation
def test_default_policy_is_usable_and_buffered() -> None:
    policy = DEFAULT_PORTFOLIO_POLICY
    assert policy.n_enter == 30
    assert policy.n_exit == 150
    assert policy.n_exit > policy.n_enter
    assert policy.ensemble_members == ("M0R", "M1")
    assert policy.ensemble_weighting == WEIGHTING_EQUAL_RAW_SCORE


def test_shipped_config_loads_and_carries_the_documented_settings() -> None:
    # the shipped policy is deliberately NOT the dataclass default: the default keeps the
    # overlay at its simplest (MA10, floor 0) while the shipped config uses the cell chosen
    # for parameter robustness. Pinning the shipped values here means an edit to
    # default.yaml cannot silently change what the daily CLI trades.
    policy = load_portfolio_policy(CONFIG)
    assert policy.n_enter == 30
    assert policy.n_exit == 150
    assert policy.ensemble_members == ("M0R", "M1")
    assert policy.ensemble_weighting == WEIGHTING_EQUAL_RAW_SCORE
    assert policy.trend_index_code == "932000.CSI"
    assert policy.trend_ma_window == 20
    assert policy.trend_exposure_floor == 0.5
    assert policy.round_trip_bp == 15.0
    assert policy.switch_cost_factor == 0.5


def test_dataclass_default_still_loads_and_is_a_valid_policy() -> None:
    assert PortfolioPolicy() == DEFAULT_PORTFOLIO_POLICY
    assert DEFAULT_PORTFOLIO_POLICY.n_exit > DEFAULT_PORTFOLIO_POLICY.n_enter


def test_buffer_narrower_than_the_book_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be >= n_enter"):
        PortfolioPolicy(n_enter=30, n_exit=29)


@pytest.mark.parametrize("weighting", ["zscore", "zscore_mean", "vol_target", "vol_targeting"])
def test_refuted_configurations_raise_with_the_measurement_that_refuted_them(
    weighting: str,
) -> None:
    with pytest.raises(ValueError, match="refused"):
        PortfolioPolicy(ensemble_weighting=weighting)


def test_unknown_weighting_is_rejected_with_the_supported_value_named() -> None:
    with pytest.raises(ValueError, match="equal_raw_score"):
        PortfolioPolicy(ensemble_weighting="inverse_variance")


@pytest.mark.parametrize(
    "kwargs",
    [
        {"n_enter": 0},
        {"n_enter": True},
        {"trend_ma_window": 1},
        {"trend_exposure_floor": -0.1},
        {"trend_exposure_floor": 1.1},
        {"round_trip_bp": -1.0},
        {"ensemble_members": ()},
        {"ensemble_members": ("M0R", "M0R")},
        {"trend_index_code": ""},
    ],
)
def test_invalid_parameters_are_rejected(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        PortfolioPolicy(**kwargs)  # type: ignore[arg-type]


def test_loader_rejects_unknown_top_level_keys(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(
        "selection:\n  n_enter: 30\n  n_exit: 150\n"
        "ensemble:\n  members: [M0R]\n  weighting: equal_raw_score\n"
        "exposure:\n  index_code: 932000.CSI\n  ma_window: 10\n  floor: 0.0\n"
        "cost:\n  round_trip_bp: 15.0\nextra: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="top-level keys"):
        load_portfolio_policy(path)


def test_loader_rejects_unknown_nested_keys(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(
        "selection:\n  n_enter: 30\n  n_exit: 150\n  smooth_k: 2\n"
        "ensemble:\n  members: [M0R]\n  weighting: equal_raw_score\n"
        "exposure:\n  index_code: 932000.CSI\n  ma_window: 10\n  floor: 0.0\n"
        "cost:\n  round_trip_bp: 15.0\n  switch_cost_factor: 0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="selection must have exactly the keys"):
        load_portfolio_policy(path)


def test_loader_rejects_a_refuted_weighting_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "policy.yaml"
    path.write_text(
        "selection:\n  n_enter: 30\n  n_exit: 150\n"
        "ensemble:\n  members: [M0R, M1]\n  weighting: zscore\n"
        "exposure:\n  index_code: 932000.CSI\n  ma_window: 10\n  floor: 0.0\n"
        "cost:\n  round_trip_bp: 15.0\n  switch_cost_factor: 0.5\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="refused"):
        load_portfolio_policy(path)


def test_loader_reports_a_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unable to load portfolio policy"):
        load_portfolio_policy(tmp_path / "absent.yaml")


# --------------------------------------------------------------------- combining
def test_combine_scores_averages_only_over_names_every_member_scored() -> None:
    policy = PortfolioPolicy(ensemble_members=("M0R", "M1"))
    out = combine_scores({"M0R": {"A": 1.0, "B": 3.0}, "M1": {"A": 3.0, "C": 9.0}}, policy)
    assert out == {"A": 2.0}


def test_combine_scores_is_deterministic_in_key_order() -> None:
    policy = PortfolioPolicy(ensemble_members=("M0R", "M1"))
    scores = {"M0R": {"Z": 1.0, "A": 1.0}, "M1": {"Z": 1.0, "A": 1.0}}
    assert list(combine_scores(scores, policy)) == ["A", "Z"]


def test_rank_descending_breaks_ties_by_code_so_replays_are_reproducible() -> None:
    assert rank_descending({"B": 1.0, "A": 1.0, "C": 2.0}) == ["C", "A", "B"]


# --------------------------------------------------------------------- buffer
def test_buffer_keeps_an_incumbent_that_fell_out_of_the_entry_rank_but_not_the_exit_rank() -> None:
    policy = PortfolioPolicy(n_enter=2, n_exit=4)
    ranked = ["N1", "N2", "N3", "OLD", "N5"]
    assert select_holdings(ranked, ["OLD", "GONE"], policy) == ["OLD", "N1"]


def test_buffer_drops_an_incumbent_that_fell_out_of_the_exit_rank() -> None:
    policy = PortfolioPolicy(n_enter=2, n_exit=3)
    ranked = ["N1", "N2", "N3", "OLD"]
    assert select_holdings(ranked, ["OLD", "GONE"], policy) == ["N1", "N2"]


def test_buffer_preserves_previous_order_for_kept_names() -> None:
    # the kept names stay in YESTERDAY's order rather than being re-sorted by today's
    # score; that ordering is part of the measured behaviour
    policy = PortfolioPolicy(n_enter=3, n_exit=5)
    ranked = ["A", "B", "C", "D", "E"]
    assert select_holdings(ranked, ["D", "B"], policy) == ["D", "B", "A"]


def test_buffer_never_returns_more_than_n_enter_names() -> None:
    policy = PortfolioPolicy(n_enter=3, n_exit=10)
    ranked = [f"N{i}" for i in range(20)]
    assert len(select_holdings(ranked, ranked[:10], policy)) == 3


def test_first_build_from_empty_holdings_takes_the_top_n_enter() -> None:
    policy = PortfolioPolicy(n_enter=3, n_exit=10)
    assert select_holdings(["A", "B", "C", "D"], [], policy) == ["A", "B", "C"]


def test_a_thin_cross_section_holds_everything_available() -> None:
    policy = PortfolioPolicy(n_enter=30, n_exit=150)
    assert select_holdings(["A", "B"], [], policy) == ["A", "B"]


# --------------------------------------------------------------------- turnover
def test_turnover_boundaries() -> None:
    assert turnover(["A", "B"], []) == 1.0
    assert turnover(["A", "B"], ["A", "B"]) == 0.0
    assert turnover(["A", "B"], ["A", "C"]) == 0.5
    assert turnover([], ["A"]) == 0.0


# --------------------------------------------------------------------- exposure
def test_exposure_is_full_above_the_ma_and_floored_below_it() -> None:
    policy = PortfolioPolicy(trend_ma_window=3, trend_exposure_floor=0.0)
    assert trend_exposure([1.0, 2.0, 6.0], policy) == 1.0
    assert trend_exposure([6.0, 5.0, 1.0], policy) == 0.0


def test_exposure_uses_only_the_trailing_window() -> None:
    policy = PortfolioPolicy(trend_ma_window=3, trend_exposure_floor=0.0)
    # a huge older value must not leak into the trailing average
    assert trend_exposure([1000.0, 1.0, 2.0, 6.0], policy) == 1.0


def test_exposure_honours_a_partial_floor() -> None:
    policy = PortfolioPolicy(trend_ma_window=2, trend_exposure_floor=0.5)
    assert trend_exposure([5.0, 1.0], policy) == 0.5


def test_cold_start_degrades_to_fully_invested_not_to_an_arbitrary_cash_position() -> None:
    policy = PortfolioPolicy(trend_ma_window=10)
    assert trend_exposure([1.0, 2.0], policy) == 1.0
    assert trend_exposure([], policy) == 1.0


def test_non_finite_closes_degrade_to_fully_invested() -> None:
    policy = PortfolioPolicy(trend_ma_window=2)
    assert trend_exposure([float("nan"), 1.0], policy) == 1.0


def test_a_flat_index_stays_invested_because_the_rule_is_strictly_above() -> None:
    policy = PortfolioPolicy(trend_ma_window=3, trend_exposure_floor=0.0)
    assert trend_exposure([5.0, 5.0, 5.0], policy) == 0.0


# --------------------------------------------------------------------- weights/cost
def test_position_weights_sum_to_the_exposure() -> None:
    weights = position_weights(["A", "B", "C", "D"], 0.5)
    assert len(weights) == 4
    assert sum(weights.values()) == pytest.approx(0.5)


def test_zero_exposure_holds_nothing() -> None:
    assert position_weights(["A", "B"], 0.0) == {}
    assert position_weights([], 1.0) == {}


def test_cost_charges_the_cash_switch_as_one_leg_not_a_full_round_trip() -> None:
    policy = PortfolioPolicy(round_trip_bp=100.0)
    assert cost_of(["A", "B"], ["A", "B"], 1.0, 1.0, policy) == pytest.approx(0.0)
    # identical names, but going flat is a liquidation: one leg, so half a round trip
    assert cost_of(["A", "B"], ["A", "B"], 0.0, 1.0, policy) == pytest.approx(50.0 / 10_000.0)
    assert cost_of(["A", "B"], ["A", "B"], 1.0, 0.0, policy) == pytest.approx(50.0 / 10_000.0)


def test_the_research_switch_convention_is_still_reachable() -> None:
    # the published overlay numbers charged a full round trip per switch; keeping that
    # reachable is what lets verify_portfolio_policy_replay.py reproduce them exactly
    policy = PortfolioPolicy(round_trip_bp=100.0, switch_cost_factor=1.0)
    assert cost_of(["A", "B"], ["A", "B"], 0.0, 1.0, policy) == pytest.approx(100.0 / 10_000.0)


def test_a_partial_exposure_step_is_charged_pro_rata() -> None:
    policy = PortfolioPolicy(round_trip_bp=100.0, switch_cost_factor=0.5)
    assert cost_of(["A", "B"], ["A", "B"], 0.5, 1.0, policy) == pytest.approx(25.0 / 10_000.0)


@pytest.mark.parametrize("factor", [-0.1, 1.1, float("nan")])
def test_invalid_switch_cost_factor_is_rejected(factor: float) -> None:
    with pytest.raises(ValueError):
        PortfolioPolicy(switch_cost_factor=factor)


def test_first_day_charges_no_switch_because_there_was_no_prior_exposure() -> None:
    policy = PortfolioPolicy(round_trip_bp=100.0)
    assert cost_of(["A"], [], 1.0, None, policy) == pytest.approx(100.0 / 10_000.0)
