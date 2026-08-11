"""Fail-closed OOS walk-forward promotion gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class GateEvidence:
    mean_rank_ic: float
    hac_t_statistic: float
    yearly_mean_rank_ic: dict[int, float]
    candidate_delta_rank_ic: float | None
    valid_test_day_count: int
    good_score_day_count: int


@dataclass(frozen=True)
class GateResult:
    status: GateStatus
    observed_value: float | None
    threshold: float | None


@dataclass(frozen=True)
class GateReport:
    results: dict[str, GateResult]

    @property
    def all_passed(self) -> bool:
        return all(
            result.status in {GateStatus.PASS, GateStatus.NOT_APPLICABLE}
            for result in self.results.values()
        )


def _finite_at_least(value: float, threshold: float) -> GateResult:
    return GateResult(
        status=GateStatus.PASS if math.isfinite(value) and value >= threshold else GateStatus.FAIL,
        observed_value=value if math.isfinite(value) else None,
        threshold=threshold,
    )


def evaluate_oos_gates(
    evidence: GateEvidence,
    *,
    is_baseline: bool = False,
    required_years: tuple[int, ...] = (2020, 2021, 2022, 2023, 2024, 2025),
) -> GateReport:
    """Evaluate G1–G6 using raw OOS statistics without any formatting-rounding effects."""
    results: dict[str, GateResult] = {
        "G1": _finite_at_least(evidence.mean_rank_ic, 0.030),
        "G2": _finite_at_least(evidence.hac_t_statistic, 2.000),
    }
    yearly_values = [evidence.yearly_mean_rank_ic.get(year) for year in required_years]
    valid_years = len(yearly_values) == len(required_years) and all(
        value is not None and math.isfinite(value) for value in yearly_values
    )
    positive_ratio = (
        sum(value > 0 for value in yearly_values if value is not None) / len(required_years)
        if valid_years and required_years
        else float("nan")
    )
    results["G3"] = _finite_at_least(positive_ratio, 0.70)
    if is_baseline:
        results["G4"] = GateResult(GateStatus.NOT_APPLICABLE, None, None)
    else:
        delta = evidence.candidate_delta_rank_ic
        results["G4"] = (
            _finite_at_least(delta, 0.005)
            if delta is not None
            else GateResult(GateStatus.FAIL, None, 0.005)
        )
    positive_contributions = [
        max(float(value), 0.0) for value in yearly_values if value is not None
    ]
    denominator = sum(positive_contributions) if valid_years else 0.0
    contribution_share = (
        max(positive_contributions) / denominator if denominator > 0 else float("nan")
    )
    results["G5"] = GateResult(
        status=GateStatus.PASS
        if math.isfinite(contribution_share) and contribution_share <= 0.50
        else GateStatus.FAIL,
        observed_value=contribution_share if math.isfinite(contribution_share) else None,
        threshold=0.50,
    )
    score_day_ratio = (
        evidence.good_score_day_count / evidence.valid_test_day_count
        if evidence.valid_test_day_count > 0
        else float("nan")
    )
    results["G6"] = _finite_at_least(score_day_ratio, 0.95)
    return GateReport(results=results)
