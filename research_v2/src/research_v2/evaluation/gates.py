"""Fail-closed OOS walk-forward promotion gates."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from numbers import Integral, Real


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


@dataclass(frozen=True)
class ScoreHealthEvidence:
    """Per-day score diagnostics used by the OOS collapse gate."""

    coverage: float
    unique_score_count: int
    score_std: float

    def passes(self) -> bool:
        """Return whether this day satisfies every score-health requirement."""
        if not _is_finite_real(self.coverage) or not _is_finite_real(self.score_std):
            return False
        if not isinstance(self.unique_score_count, Integral) or isinstance(
            self.unique_score_count, bool
        ):
            return False
        coverage = float(self.coverage)
        score_std = float(self.score_std)
        return 0.90 <= coverage <= 1.0 and self.unique_score_count >= 10 and score_std > 1e-6


@dataclass(frozen=True)
class GateEvidence:
    mean_rank_ic: float
    hac_t_statistic: float
    yearly_mean_rank_ic: dict[int, float]
    candidate_delta_rank_ic: float | None
    valid_test_day_count: int
    score_health: tuple[ScoreHealthEvidence, ...] = ()


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


def _ordinary_test_years() -> tuple[int, ...]:
    """The ordinary (non-locked) folds' test years, derived from the fold contract.

    Imported lazily so this module stays dependency-light; the fold contract lives in
    datasets.splits. Deriving it here is what keeps G3/G5 aligned with the current grid:
    the default used to be the hardcoded F01-F06 years (2020-2025), which made a v2
    (F1-F4, test years 2022-2025) OOS evaluate G3/G5 as NaN -> FAIL with observed=None
    simply because 2020 and 2021 were absent from the evidence.
    """
    from research_v2.datasets.splits import annual_folds

    return tuple(fold.test_years[0] for fold in annual_folds() if not fold.locked)


def evaluate_oos_gates(
    evidence: GateEvidence,
    *,
    is_baseline: bool = False,
    required_years: tuple[int, ...] | None = None,
) -> GateReport:
    """Evaluate G1–G6 using raw OOS statistics without any formatting-rounding effects.

    ``required_years`` defaults to the ordinary folds' test years from the fold
    contract (currently F1-F4 -> 2022-2025), so G3/G5 are evaluated over the years
    the current grid actually covers; callers testing gate logic against a fixed
    year set pass it explicitly.
    """
    if required_years is None:
        required_years = _ordinary_test_years()
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
    score_day_ratio = _score_health_ratio(evidence)
    results["G6"] = _finite_at_least(score_day_ratio, 0.95)
    return GateReport(results=results)


def _score_health_ratio(evidence: GateEvidence) -> float:
    """Compute G6 only from complete per-day evidence, failing closed otherwise."""
    if not isinstance(evidence.valid_test_day_count, Integral) or isinstance(
        evidence.valid_test_day_count, bool
    ):
        return float("nan")
    if evidence.valid_test_day_count <= 0:
        return float("nan")
    if not isinstance(evidence.score_health, tuple):
        return float("nan")
    if len(evidence.score_health) != evidence.valid_test_day_count:
        return float("nan")
    if not all(isinstance(item, ScoreHealthEvidence) for item in evidence.score_health):
        return float("nan")
    return sum(item.passes() for item in evidence.score_health) / evidence.valid_test_day_count


def _is_finite_real(value: object) -> bool:
    try:
        return (
            isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))
        )
    except (TypeError, ValueError, OverflowError):
        return False
