"""Typed runtime policy for inference monitoring thresholds."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class MonitoringPolicy:
    psi_epsilon: float = 1e-6
    psi_info_upper_bound: float = 0.10
    psi_warn_upper_bound: float = 0.25
    minimum_row_coverage: float = 0.95
    universe_size_lower_ratio: float = 0.80
    universe_size_upper_ratio: float = 1.20
    maximum_missing_rate_increase: float = 0.10
    consecutive_psi_error_days: int = 3

    def __post_init__(self) -> None:
        finite_values = (
            self.psi_epsilon,
            self.psi_info_upper_bound,
            self.psi_warn_upper_bound,
            self.minimum_row_coverage,
            self.universe_size_lower_ratio,
            self.universe_size_upper_ratio,
            self.maximum_missing_rate_increase,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int | float)
            for value in finite_values
        ):
            raise ValueError("monitoring thresholds must be numeric")
        if any(not math.isfinite(value) for value in finite_values):
            raise ValueError("monitoring thresholds must be finite")
        if self.psi_epsilon <= 0 or not 0 < self.psi_info_upper_bound <= self.psi_warn_upper_bound:
            raise ValueError("PSI thresholds are invalid")
        if not 0 < self.minimum_row_coverage <= 1:
            raise ValueError("minimum_row_coverage must be in (0, 1]")
        if not 0 < self.universe_size_lower_ratio <= self.universe_size_upper_ratio:
            raise ValueError("universe size ratios are invalid")
        if not 0 <= self.maximum_missing_rate_increase <= 1:
            raise ValueError("maximum_missing_rate_increase must be in [0, 1]")
        if (
            isinstance(self.consecutive_psi_error_days, bool)
            or not isinstance(self.consecutive_psi_error_days, int)
            or self.consecutive_psi_error_days < 1
        ):
            raise ValueError("consecutive_psi_error_days must be positive")


DEFAULT_MONITORING_POLICY = MonitoringPolicy()


def load_monitoring_policy(path: Path) -> MonitoringPolicy:
    """Load one complete monitoring policy and reject unknown or malformed thresholds."""
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"unable to load monitoring policy: {path}") from error
    if not isinstance(raw, Mapping) or set(raw) != {"psi", "feature_health", "drift_review"}:
        raise ValueError("monitoring policy has an invalid top-level schema")
    psi = _mapping(raw.get("psi"), "psi")
    feature_health = _mapping(raw.get("feature_health"), "feature_health")
    drift_review = _mapping(raw.get("drift_review"), "drift_review")
    _require_exact_keys(psi, {"epsilon", "info_upper_bound", "warn_upper_bound"}, "psi")
    _require_exact_keys(
        feature_health,
        {
            "minimum_row_coverage",
            "universe_size_lower_ratio",
            "universe_size_upper_ratio",
            "maximum_missing_rate_increase",
        },
        "feature_health",
    )
    _require_exact_keys(drift_review, {"consecutive_psi_error_days"}, "drift_review")
    return MonitoringPolicy(
        psi_epsilon=_number(psi["epsilon"], "psi.epsilon"),
        psi_info_upper_bound=_number(psi["info_upper_bound"], "psi.info_upper_bound"),
        psi_warn_upper_bound=_number(psi["warn_upper_bound"], "psi.warn_upper_bound"),
        minimum_row_coverage=_number(
            feature_health["minimum_row_coverage"], "feature_health.minimum_row_coverage"
        ),
        universe_size_lower_ratio=_number(
            feature_health["universe_size_lower_ratio"], "feature_health.universe_size_lower_ratio"
        ),
        universe_size_upper_ratio=_number(
            feature_health["universe_size_upper_ratio"], "feature_health.universe_size_upper_ratio"
        ),
        maximum_missing_rate_increase=_number(
            feature_health["maximum_missing_rate_increase"],
            "feature_health.maximum_missing_rate_increase",
        ),
        consecutive_psi_error_days=_positive_integer(
            drift_review["consecutive_psi_error_days"],
            "drift_review.consecutive_psi_error_days",
        ),
    )


def _mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError(f"monitoring policy section is invalid: {name}")
    return value


def _require_exact_keys(value: Mapping[str, object], expected: set[str], name: str) -> None:
    if set(value) != expected:
        raise ValueError(f"monitoring policy section has invalid keys: {name}")


def _number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"monitoring policy value is not numeric: {name}")
    return float(value)


def _positive_integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"monitoring policy value is not a positive integer: {name}")
    return value
