"""Deterministic dual-threshold factor screening rules."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FactorSelectionRule:
    """Stable screening gates applied only to non-test evaluation summaries."""

    min_abs_mean_rank_ic: float = 0.02
    min_abs_icir: float = 0.50
    min_valid_day_count: int = 60
    min_directional_day_share: float = 0.55
    excluded_factor_ids: frozenset[str] = frozenset()

    def __post_init__(self) -> None:
        if self.min_abs_mean_rank_ic < 0 or self.min_abs_icir < 0:
            raise ValueError("absolute RankIC and ICIR thresholds cannot be negative")
        if self.min_valid_day_count < 1:
            raise ValueError("min_valid_day_count must be positive")
        if not 0.5 <= self.min_directional_day_share <= 1:
            raise ValueError("directional day share must be in [0.5, 1]")
        if any(not factor_id for factor_id in self.excluded_factor_ids):
            raise ValueError("excluded factor IDs cannot be empty")


def select_factors_by_dual_threshold(
    summary: pd.DataFrame,
    rules: FactorSelectionRule | None = None,
) -> pd.DataFrame:
    """Select factors passing RankIC, ICIR, coverage, and direction gates.

    The empirical direction is frozen from the sign of mean RankIC. The input
    summary must be produced from a training/validation scope; this function
    intentionally has no access to locked-test rows.
    """
    if rules is None:
        rules = FactorSelectionRule()
    required = {
        "factor_id",
        "factor_version",
        "factor_name",
        "mean_rank_ic",
        "icir",
        "valid_day_count",
        "positive_day_share",
    }
    if missing := sorted(required.difference(summary.columns)):
        raise ValueError(f"factor summary is missing columns: {missing}")
    if summary["factor_id"].duplicated().any():
        raise ValueError("factor summary contains duplicate factor_id values")
    frame = summary.loc[:, sorted(required)].copy()
    for column in ["mean_rank_ic", "icir", "valid_day_count", "positive_day_share"]:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    finite = np.isfinite(frame["mean_rank_ic"])
    finite &= np.isfinite(frame["icir"])
    finite &= np.isfinite(frame["valid_day_count"])
    finite &= np.isfinite(frame["positive_day_share"])
    frame = frame.loc[finite].copy()
    if rules.excluded_factor_ids:
        frame = frame.loc[~frame["factor_id"].isin(rules.excluded_factor_ids)].copy()
    frame["frozen_direction"] = np.where(frame["mean_rank_ic"] >= 0, 1, -1)
    directional_share = np.where(
        frame["frozen_direction"] > 0,
        frame["positive_day_share"],
        1.0 - frame["positive_day_share"],
    )
    frame["directional_day_share"] = directional_share
    passes = (
        frame["mean_rank_ic"].abs() >= rules.min_abs_mean_rank_ic
    ) & (
        frame["icir"].abs() >= rules.min_abs_icir
    ) & (
        frame["valid_day_count"] >= rules.min_valid_day_count
    ) & (
        frame["directional_day_share"] >= rules.min_directional_day_share
    )
    selected = frame.loc[passes].copy()
    selected["selection_reason"] = "DUAL_RANKIC_ICIR_GATE"
    return pd.DataFrame(
        selected.sort_values(
            ["factor_id", "factor_version"],
            kind="mergesort",
            ignore_index=True,
        )
    )


def select_stable_factors(
    training_summary: pd.DataFrame,
    validation_summary: pd.DataFrame,
    rules: FactorSelectionRule | None = None,
) -> pd.DataFrame:
    """Keep dual-gate factors whose frozen direction agrees across two periods."""
    train_selected = select_factors_by_dual_threshold(training_summary, rules).rename(
        columns={
            "frozen_direction": "training_direction",
            "mean_rank_ic": "training_mean_rank_ic",
            "icir": "training_icir",
        }
    )
    validation_selected = select_factors_by_dual_threshold(validation_summary, rules).rename(
        columns={
            "frozen_direction": "validation_direction",
            "mean_rank_ic": "validation_mean_rank_ic",
            "icir": "validation_icir",
        }
    )
    merged = train_selected.merge(
        validation_selected.loc[
            :,
            [
                "factor_id",
                "factor_version",
                "validation_direction",
                "validation_mean_rank_ic",
                "validation_icir",
            ],
        ],
        on=["factor_id", "factor_version"],
        how="inner",
        validate="one_to_one",
    )
    stable = merged.loc[merged["training_direction"] == merged["validation_direction"]].copy()
    stable["frozen_direction"] = stable["training_direction"]
    stable["selection_reason"] = "DUAL_GATE_STABLE_DIRECTION"
    return pd.DataFrame(
        stable.sort_values(
            ["factor_id", "factor_version"],
            kind="mergesort",
            ignore_index=True,
        )
    )


def select_tushare_canonical_stable_factors(
    training_summary: pd.DataFrame,
    validation_summary: pd.DataFrame,
    replay_summary: pd.DataFrame,
    rules: FactorSelectionRule | None = None,
) -> pd.DataFrame:
    """Select stable factors only after every Tushare replay window passes.

    ``replay_summary`` is expected in long form with one or more rows per
    factor and a ``decision_status`` column. Missing factor rows and any
    status other than ``FORMULA_REPLAY_PASS`` fail closed. The existing
    selection functions remain available for non-canonical historical screens.
    """

    train_gated = _attach_tushare_replay_gate(training_summary, replay_summary)
    validation_gated = _attach_tushare_replay_gate(validation_summary, replay_summary)
    stable = select_stable_factors(train_gated, validation_gated, rules)
    if stable.empty:
        return stable.assign(
            canonical_replay_status=pd.Series(dtype=str),
            canonical_replay_window_count=pd.Series(dtype=int),
        )
    gate = train_gated.loc[
        :, ["factor_id", "factor_version", "canonical_replay_status", "canonical_replay_window_count"]
    ].drop_duplicates(["factor_id", "factor_version"])
    stable = stable.merge(
        gate,
        on=["factor_id", "factor_version"],
        how="inner",
        validate="one_to_one",
    )
    stable = stable.loc[stable["canonical_replay_status"].eq("FORMULA_REPLAY_PASS")].copy()
    stable["selection_reason"] = "DUAL_GATE_STABLE_DIRECTION_TUSHARE_CANONICAL"
    return stable.sort_values(
        ["factor_id", "factor_version"], kind="mergesort", ignore_index=True
    )


def _attach_tushare_replay_gate(
    summary: pd.DataFrame,
    replay_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Attach an all-windows replay status to a factor summary."""

    if "factor_id" not in summary.columns:
        raise ValueError("factor summary is missing factor_id")
    if summary["factor_id"].duplicated().any():
        raise ValueError("factor summary contains duplicate factor_id values")
    required = {"factor_id", "decision_status"}
    if missing := sorted(required.difference(replay_summary.columns)):
        raise ValueError(f"replay summary is missing columns: {missing}")
    replay = replay_summary.loc[:, ["factor_id", "decision_status"]].copy()
    replay["factor_id"] = replay["factor_id"].astype(str)
    if replay.empty:
        gate = pd.DataFrame(
            columns=["factor_id", "canonical_replay_status", "canonical_replay_window_count"]
        )
    else:
        grouped = replay.groupby("factor_id", sort=True)
        gate = grouped["decision_status"].agg(
            canonical_replay_window_count="size",
            _all_pass=lambda values: bool(values.eq("FORMULA_REPLAY_PASS").all()),
        ).reset_index()
        gate["canonical_replay_status"] = np.where(
            gate["_all_pass"], "FORMULA_REPLAY_PASS", "FORMULA_REPLAY_BLOCKED"
        )
        gate = gate.drop(columns=["_all_pass"])
    merged = summary.merge(gate, on="factor_id", how="left", validate="one_to_one")
    merged["canonical_replay_status"] = merged["canonical_replay_status"].fillna(
        "FORMULA_REPLAY_MISSING"
    )
    merged["canonical_replay_window_count"] = pd.to_numeric(
        merged["canonical_replay_window_count"], errors="coerce"
    ).fillna(0).astype(int)
    return merged
