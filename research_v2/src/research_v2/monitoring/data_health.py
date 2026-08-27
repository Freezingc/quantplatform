"""Feature-frame health checks kept separate from inference publication policy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import date

import numpy as np
import pandas as pd

from research_v2.monitoring.drift import DriftSeverity, PsiResult, population_stability_index
from research_v2.monitoring.policy import DEFAULT_MONITORING_POLICY, MonitoringPolicy


@dataclass(frozen=True)
class FeatureReference:
    feature_name: str
    bin_edges: tuple[float, ...]
    reference_bin_masses: tuple[float, ...]
    reference_missing_rate: float
    minimum_finite_coverage: float

    def __post_init__(self) -> None:
        if not self.feature_name.strip():
            raise ValueError("feature_name cannot be empty")
        if len(self.bin_edges) < 2 or len(self.reference_bin_masses) != len(self.bin_edges) - 1:
            raise ValueError("feature reference bins are invalid")
        if any(not math.isfinite(value) for value in self.bin_edges) or any(
        right <= left for left, right in zip(self.bin_edges, self.bin_edges[1:], strict=False)
        ):
            raise ValueError("feature reference bin edges must be finite and increasing")
        if any(not math.isfinite(value) or value < 0 for value in self.reference_bin_masses):
            raise ValueError("feature reference bin masses are invalid")
        if not 0 <= self.reference_missing_rate <= 1:
            raise ValueError("reference_missing_rate must be between zero and one")
        if not 0 <= self.minimum_finite_coverage <= 1:
            raise ValueError("minimum_finite_coverage must be between zero and one")


@dataclass(frozen=True)
class FeatureHealth:
    feature_name: str
    finite_coverage: float
    missing_rate: float
    psi: PsiResult


@dataclass(frozen=True)
class DataHealthReport:
    row_coverage: float
    universe_size: int
    blocking_reasons: tuple[str, ...]
    investigation_alerts: tuple[str, ...]
    features: tuple[FeatureHealth, ...]


def evaluate_feature_data_health(
    frame: pd.DataFrame,
    *,
    signal_date: date,
    references: Sequence[FeatureReference],
    expected_universe_size: int,
    recent_universe_size_median: float,
    policy: MonitoringPolicy = DEFAULT_MONITORING_POLICY,
) -> DataHealthReport:
    """Evaluate PIT business keys and feature distributions without choosing publish policy."""
    if not isinstance(signal_date, date):
        raise ValueError("signal_date must be a date")
    if expected_universe_size < 1 or recent_universe_size_median <= 0:
        raise ValueError("expected universe size and recent median must be positive")
    if (
        not references
        or len({reference.feature_name for reference in references}) != len(references)
    ):
        raise ValueError("feature references must be non-empty and unique")
    required = {"signal_date", "ts_code", "available_date"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"feature frame is missing columns: {missing}")
    normalized = frame.copy(deep=True)
    normalized["signal_date"] = pd.to_datetime(normalized["signal_date"], errors="raise").dt.date
    normalized["available_date"] = pd.to_datetime(
        normalized["available_date"], errors="raise"
    ).dt.date
    scoped = normalized.loc[normalized["signal_date"] == signal_date].copy()
    blockers: list[str] = []
    alerts: list[str] = []
    keys = ["signal_date", "ts_code"]
    if scoped.duplicated(keys).any():
        blockers.append("DUPLICATE_BUSINESS_KEY")
    if scoped["ts_code"].isna().any() or scoped["ts_code"].astype(str).str.strip().eq("").any():
        blockers.append("INVALID_BUSINESS_KEY")
    if scoped["available_date"].isna().any():
        blockers.append("MISSING_AVAILABLE_DATE")
    if (scoped["available_date"] > signal_date).any():
        blockers.append("FUTURE_AVAILABLE_DATE")
    universe_size = int(len(scoped))
    row_coverage = universe_size / expected_universe_size
    if row_coverage < policy.minimum_row_coverage:
        blockers.append("INSUFFICIENT_ROW_COVERAGE")
    lower_bound = policy.universe_size_lower_ratio * recent_universe_size_median
    upper_bound = policy.universe_size_upper_ratio * recent_universe_size_median
    if universe_size < lower_bound or universe_size > upper_bound:
        blockers.append("UNIVERSE_SIZE_OUT_OF_RANGE")
    features: list[FeatureHealth] = []
    for reference in references:
        if reference.feature_name not in scoped.columns:
            blockers.append(f"MISSING_REQUIRED_FEATURE:{reference.feature_name}")
            continue
        values = pd.to_numeric(scoped[reference.feature_name], errors="coerce")
        finite = np.isfinite(values)
        finite_values = values.loc[finite].to_numpy(dtype=float)
        finite_coverage = float(finite.sum() / expected_universe_size)
        missing_rate = 1.0 - finite_coverage
        bounded_values = np.clip(finite_values, reference.bin_edges[0], reference.bin_edges[-1])
        bin_masses = tuple(
            float(value)
            for value in np.histogram(bounded_values, bins=np.asarray(reference.bin_edges))[0]
        )
        psi = population_stability_index(
            reference.reference_bin_masses,
            bin_masses,
            policy=policy,
        )
        features.append(
            FeatureHealth(reference.feature_name, finite_coverage, missing_rate, psi)
        )
        if finite_coverage < reference.minimum_finite_coverage:
            alerts.append(f"LOW_FACTOR_COVERAGE:{reference.feature_name}")
        if (
            missing_rate - reference.reference_missing_rate
            > policy.maximum_missing_rate_increase
        ):
            alerts.append(f"MISSING_RATE_INCREASE:{reference.feature_name}")
        if psi.severity is DriftSeverity.ERROR:
            alerts.append(f"PSI_ERROR:{reference.feature_name}")
    return DataHealthReport(
        row_coverage=row_coverage,
        universe_size=universe_size,
        blocking_reasons=tuple(sorted(set(blockers))),
        investigation_alerts=tuple(alerts),
        features=tuple(features),
    )
