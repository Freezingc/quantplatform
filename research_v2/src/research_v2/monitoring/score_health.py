"""Blocking health checks for label-free daily ensemble score publication."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date
from itertools import combinations
from types import MappingProxyType

import numpy as np
import pandas as pd

from research_v2.monitoring.data_health import ScoreDistributionReference
from research_v2.monitoring.drift import PsiResult, evaluate_distribution_drift
from research_v2.monitoring.policy import DEFAULT_MONITORING_POLICY, MonitoringPolicy
from research_v2.training.ensemble import REQUIRED_SEEDS


@dataclass(frozen=True)
class ScoreHealthReport:
    finite_coverage: float
    unique_score_count: int
    score_std: float
    blocking_reasons: tuple[str, ...]
    pairwise_spearman: Mapping[tuple[int, int], float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    error_signals: tuple[str, ...] = ()
    daily_psi: PsiResult | None = None
    rolling_20d_psi: PsiResult | None = None
    extreme_rate: float | None = None


def evaluate_score_health(
    predictions: pd.DataFrame,
    *,
    expected_key_count: int,
    seed_percentile_ranks: Mapping[int, pd.DataFrame] | None = None,
    missingness_patterns: Sequence[object] | pd.Series | str | None = None,
    score_reference: ScoreDistributionReference | None = None,
    rolling_predictions: pd.DataFrame | None = None,
    policy: MonitoringPolicy = DEFAULT_MONITORING_POLICY,
) -> ScoreHealthReport:
    """Reject unhealthy score cross-sections and report label-free diagnostics.

    ``seed_percentile_ranks`` is optional for compatibility with legacy callers. When
    supplied, it must contain exactly the fixed required seeds and matching business
    keys; every pairwise Spearman value is reported, with low or non-finite values
    recorded as an error signal without becoming a blocking reason. ``missingness_patterns``
    is an optional sequence aligned positionally with ``predictions`` (or a column name).
    Its extreme-tail check is skipped when fewer than 100 finite scores are available or
    when a 1% tail has fewer than two observations, because that tail is too small to
    distinguish a missingness-pattern collapse from a single observation.
    """
    if expected_key_count < 1:
        raise ValueError("expected_key_count must be positive")
    if rolling_predictions is not None and score_reference is None:
        raise ValueError("rolling_predictions requires score_reference")
    required = {"signal_date", "ts_code", "ensemble_score"}
    if missing := sorted(required.difference(predictions.columns)):
        raise ValueError(f"predictions are missing columns: {missing}")
    keys = ["signal_date", "ts_code"]
    reasons: list[str] = []
    if predictions.duplicated(keys).any():
        reasons.append("DUPLICATE_BUSINESS_KEY")
    numeric_scores = pd.to_numeric(predictions["ensemble_score"], errors="coerce")
    finite = np.isfinite(numeric_scores)
    finite_coverage = float(finite.sum() / expected_key_count)
    if finite_coverage < 0.95:
        reasons.append("INSUFFICIENT_FINITE_SCORE_COVERAGE")
    finite_scores = numeric_scores.loc[finite]
    unique_count = int(finite_scores.nunique())
    if unique_count < 10:
        reasons.append("INSUFFICIENT_UNIQUE_SCORES")
    score_std = float(finite_scores.std(ddof=0)) if not finite_scores.empty else float("nan")
    if not np.isfinite(score_std) or score_std <= 1e-6:
        reasons.append("SCORE_STD_COLLAPSE")
    pairwise_spearman: Mapping[tuple[int, int], float] = MappingProxyType({})
    error_signals: list[str] = []
    if seed_percentile_ranks is not None:
        prediction_keys = set(zip(predictions["signal_date"], predictions["ts_code"], strict=True))
        normalized_seed_ranks = _validate_seed_percentile_ranks(
            seed_percentile_ranks,
            expected_keys=prediction_keys,
        )
        pairwise_values = {
            pair: _pairwise_spearman(normalized_seed_ranks, *pair)
            for pair in combinations(REQUIRED_SEEDS, 2)
        }
        pairwise_spearman = MappingProxyType(pairwise_values)
        if any(
            not np.isfinite(value) or value < 0.20
            for value in pairwise_values.values()
        ):
            error_signals.append("SEED_AGREEMENT_ERROR")
    if missingness_patterns is not None:
        _append_extreme_pattern_reason(
            predictions,
            numeric_scores,
            missingness_patterns,
            reasons,
        )
    daily_psi: PsiResult | None = None
    rolling_20d_psi: PsiResult | None = None
    extreme_rate: float | None = None
    if score_reference is not None:
        if not isinstance(score_reference, ScoreDistributionReference):
            raise ValueError("score_reference must be a ScoreDistributionReference")
        distribution = evaluate_distribution_drift(
            reference_bin_edges=score_reference.bin_edges,
            reference_bin_masses=score_reference.bin_masses,
            daily_values=numeric_scores.tolist(),
            rolling_values_by_date=_rolling_score_values(
                rolling_predictions,
                latest_signal_date=_latest_signal_date(predictions),
            ),
            policy=policy,
        )
        daily_psi = distribution.daily_psi
        rolling_20d_psi = distribution.rolling_20d_psi
        finite_array = numeric_scores.to_numpy(dtype=float)[finite.to_numpy()]
        extreme_rate = _extreme_rate(
            finite_array,
            lower=score_reference.bin_edges[0],
            upper=score_reference.bin_edges[-1],
        )
    return ScoreHealthReport(
        finite_coverage=finite_coverage,
        unique_score_count=unique_count,
        score_std=score_std,
        blocking_reasons=tuple(reasons),
        pairwise_spearman=pairwise_spearman,
        error_signals=tuple(error_signals),
        daily_psi=daily_psi,
        rolling_20d_psi=rolling_20d_psi,
        extreme_rate=extreme_rate,
    )


def _validate_seed_percentile_ranks(
    seed_percentile_ranks: Mapping[int, pd.DataFrame],
    *,
    expected_keys: set[tuple[object, object]],
) -> dict[int, pd.DataFrame]:
    if not isinstance(seed_percentile_ranks, Mapping):
        raise ValueError("seed percentile ranks must be a mapping keyed by seed")
    if set(seed_percentile_ranks) != set(REQUIRED_SEEDS):
        raise ValueError("all required seeds must be present exactly once")
    keys = ["signal_date", "ts_code"]
    required = set(keys + ["percentile_rank"])
    normalized: dict[int, pd.DataFrame] = {}
    reference_keys: set[tuple[object, object]] | None = None
    for seed in REQUIRED_SEEDS:
        frame = seed_percentile_ranks[seed]
        if not isinstance(frame, pd.DataFrame):
            raise ValueError("seed percentile ranks must be pandas data frames")
        if missing := sorted(required.difference(frame.columns)):
            raise ValueError(f"seed predictions are missing columns: {missing}")
        if frame.duplicated(keys).any():
            raise ValueError("seed predictions contain duplicate business keys")
        ranks = pd.to_numeric(frame["percentile_rank"], errors="coerce")
        if not np.isfinite(ranks).all() or ((ranks < 0) | (ranks > 1)).any():
            raise ValueError("seed percentile ranks must be finite values in [0, 1]")
        cleaned = frame.loc[:, keys].copy()
        cleaned["percentile_rank"] = ranks.to_numpy()
        frame_keys = set(zip(cleaned["signal_date"], cleaned["ts_code"], strict=True))
        if reference_keys is None:
            reference_keys = frame_keys
        elif frame_keys != reference_keys:
            raise ValueError("all seed predictions must have identical business-key coverage")
        normalized[seed] = cleaned
    if reference_keys != expected_keys:
        raise ValueError("seed predictions must match prediction business-key coverage")
    return normalized


def _pairwise_spearman(
    seed_ranks: Mapping[int, pd.DataFrame],
    left_seed: int,
    right_seed: int,
) -> float:
    keys = ["signal_date", "ts_code"]
    left_column = f"percentile_rank_{left_seed}"
    right_column = f"percentile_rank_{right_seed}"
    aligned = seed_ranks[left_seed].rename(columns={"percentile_rank": left_column}).merge(
        seed_ranks[right_seed].rename(columns={"percentile_rank": right_column}),
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    left_values = aligned[left_column].to_numpy(dtype=float)
    right_values = aligned[right_column].to_numpy(dtype=float)
    if len(aligned) < 2 or np.unique(left_values).size < 2 or np.unique(right_values).size < 2:
        return float("nan")
    left_ranks = pd.Series(left_values).rank(method="average").to_numpy(dtype=float)
    right_ranks = pd.Series(right_values).rank(method="average").to_numpy(dtype=float)
    left_centered = left_ranks - left_ranks.mean()
    right_centered = right_ranks - right_ranks.mean()
    denominator = float(
        np.sqrt(np.square(left_centered).sum() * np.square(right_centered).sum())
    )
    if denominator == 0:
        return float("nan")
    return float(np.dot(left_centered, right_centered) / denominator)


def _append_extreme_pattern_reason(
    predictions: pd.DataFrame,
    numeric_scores: pd.Series,
    missingness_patterns: Sequence[object] | pd.Series | str,
    reasons: list[str],
) -> None:
    if isinstance(missingness_patterns, str):
        if missingness_patterns not in predictions.columns:
            raise ValueError(
                f"predictions are missing missingness pattern column: {missingness_patterns}"
            )
        raw_patterns = predictions[missingness_patterns]
    else:
        raw_patterns = pd.Series(missingness_patterns)
    if len(raw_patterns) != len(predictions):
        raise ValueError("missingness_patterns must align with predictions")
    patterns = pd.Series(raw_patterns.to_numpy(), index=predictions.index)
    if patterns.isna().any():
        reasons.append("MISSINGNESS_PATTERN_UNAVAILABLE")
        return
    finite = np.isfinite(numeric_scores).to_numpy()
    finite_count = int(finite.sum())
    extreme_count = int(np.ceil(finite_count * 0.01))
    if finite_count < 100 or extreme_count < 2:
        return
    finite_scores = numeric_scores.to_numpy(dtype=float)[finite]
    finite_patterns = patterns.to_numpy()[finite]
    order = np.argsort(finite_scores, kind="mergesort")
    tails = (order[:extreme_count], order[-extreme_count:])
    if any(pd.unique(finite_patterns[tail]).size == 1 for tail in tails):
        reasons.append("EXTREME_MISSINGNESS_PATTERN_COLLAPSE")


def _rolling_score_values(
    frame: pd.DataFrame | None,
    *,
    latest_signal_date: date,
) -> dict[date, tuple[float, ...]] | None:
    if frame is None:
        return None
    required = {"signal_date", "ensemble_score"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"rolling predictions are missing columns: {missing}")
    normalized = frame.loc[:, ["signal_date", "ensemble_score"]].copy().reset_index(drop=True)
    normalized_dates = pd.to_datetime(normalized["signal_date"], errors="raise").dt.date
    if normalized_dates.isna().any():
        raise ValueError("rolling predictions contain invalid signal dates")
    if (normalized_dates > latest_signal_date).any():
        raise ValueError("rolling predictions contain future signal dates")
    normalized["signal_date"] = normalized_dates
    normalized["ensemble_score"] = pd.to_numeric(
        normalized["ensemble_score"], errors="coerce"
    )
    result: dict[date, tuple[float, ...]] = {}
    for signal_date, group in normalized.groupby("signal_date", sort=True):
        if not isinstance(signal_date, date):
            raise ValueError("rolling prediction signal dates must be dates")
        result[signal_date] = tuple(
            float(value) for value in group["ensemble_score"].to_numpy(dtype=float)
        )
    return result


def _latest_signal_date(predictions: pd.DataFrame) -> date:
    dates = pd.to_datetime(predictions["signal_date"], errors="raise").dt.date
    if dates.isna().any():
        raise ValueError("predictions contain invalid signal dates")
    latest = dates.max()
    if not isinstance(latest, date):
        raise ValueError("predictions require a signal date")
    return latest


def _extreme_rate(values: np.ndarray, *, lower: float, upper: float) -> float:
    if values.size == 0:
        return float("nan")
    return float(((values < lower) | (values > upper)).sum() / values.size)
