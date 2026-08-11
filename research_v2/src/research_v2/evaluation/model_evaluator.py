"""Frozen test-period RankIC evaluation and promotion-gate evidence assembly."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import pandas as pd

from research_v2.evaluation.gates import GateEvidence, GateReport, evaluate_oos_gates
from research_v2.evaluation.hac import HACMeanTest, hac_mean_test
from research_v2.evaluation.rank_ic import RankICSummary, compute_daily_rank_ic, summarize_rank_ic

EvaluationRole = Literal["TEST", "VALIDATION", "TRAIN", "LOCKED"]


@dataclass(frozen=True)
class FrozenOosEvaluation:
    daily_rank_ic: pd.DataFrame
    summary: RankICSummary
    hac: HACMeanTest
    yearly_mean_rank_ic: dict[int, float]
    baseline_mean_rank_ic: float | None
    delta_rank_ic: float | None
    gate_report: GateReport


def evaluate_frozen_oos(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    baseline_predictions: pd.DataFrame | None = None,
    role: EvaluationRole,
    minimum_cross_section: int = 30,
) -> FrozenOosEvaluation:
    """Evaluate one frozen ordinary test dataset; validation and train roles are forbidden."""
    if role != "TEST":
        raise PermissionError("frozen OOS evaluation is permitted only for TEST data")
    candidate_predictions = predictions
    baseline_predictions_matched = baseline_predictions
    if baseline_predictions is not None:
        candidate_predictions, baseline_predictions_matched = _match_prediction_coverage(
            predictions,
            baseline_predictions,
        )
    daily = compute_daily_rank_ic(
        candidate_predictions,
        labels,
        minimum_cross_section=minimum_cross_section,
    )
    if len(daily) <= 5:
        raise ValueError("OOS evaluation requires more than five valid daily RankIC observations")
    summary = summarize_rank_ic(daily)
    hac = hac_mean_test(daily["rank_ic"], max_lags=5)
    yearly = _yearly_mean_rank_ic(daily)
    baseline_mean: float | None = None
    delta: float | None = None
    if baseline_predictions_matched is not None:
        baseline_daily = compute_daily_rank_ic(
            baseline_predictions_matched,
            labels,
            minimum_cross_section=minimum_cross_section,
        )
        baseline_summary = summarize_rank_ic(baseline_daily)
        if not math.isfinite(baseline_summary.mean_rank_ic):
            raise ValueError("baseline has no valid OOS RankIC observations")
        baseline_mean = baseline_summary.mean_rank_ic
        delta = summary.mean_rank_ic - baseline_mean
    valid_label_days = _valid_label_day_count(labels, minimum_cross_section)
    evidence = GateEvidence(
        mean_rank_ic=summary.mean_rank_ic,
        hac_t_statistic=hac.t_statistic,
        yearly_mean_rank_ic=yearly,
        candidate_delta_rank_ic=delta,
        valid_test_day_count=valid_label_days,
        good_score_day_count=len(daily),
    )
    return FrozenOosEvaluation(
        daily_rank_ic=daily,
        summary=summary,
        hac=hac,
        yearly_mean_rank_ic=yearly,
        baseline_mean_rank_ic=baseline_mean,
        delta_rank_ic=delta,
        gate_report=evaluate_oos_gates(evidence),
    )


def _yearly_mean_rank_ic(daily: pd.DataFrame) -> dict[int, float]:
    dates = pd.to_datetime(daily["signal_date"], errors="raise")
    values = pd.to_numeric(daily["rank_ic"], errors="coerce")
    grouped = pd.DataFrame({"year": dates.dt.year, "rank_ic": values}).groupby("year", sort=True)
    return {int(cast(Any, year)): float(group["rank_ic"].mean()) for year, group in grouped}


def _match_prediction_coverage(
    candidate_predictions: pd.DataFrame,
    baseline_predictions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Restrict candidate and baseline scores to the identical test date/security keys."""
    keys = ["signal_date", "ts_code"]
    for name, frame in (("candidate", candidate_predictions), ("baseline", baseline_predictions)):
        if missing := sorted(set(keys).difference(frame.columns)):
            raise ValueError(f"{name} predictions are missing columns: {missing}")
        if frame.duplicated(keys).any():
            raise ValueError(f"{name} predictions contain duplicate business keys")
    common_keys = candidate_predictions.loc[:, keys].merge(
        baseline_predictions.loc[:, keys],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    return (
        candidate_predictions.merge(common_keys, on=keys, how="inner", validate="one_to_one"),
        baseline_predictions.merge(common_keys, on=keys, how="inner", validate="one_to_one"),
    )


def _valid_label_day_count(labels: pd.DataFrame, minimum_cross_section: int) -> int:
    required = {"signal_date", "ts_code", "raw_return_5d", "label_valid"}
    if missing := sorted(required.difference(labels.columns)):
        raise ValueError(f"labels are missing columns: {missing}")
    raw_returns = pd.to_numeric(labels["raw_return_5d"], errors="coerce")
    valid = labels["label_valid"].astype(bool) & np.isfinite(raw_returns)
    counts = labels.loc[valid].groupby("signal_date", sort=False).size()
    return int((counts >= minimum_cross_section).sum())
