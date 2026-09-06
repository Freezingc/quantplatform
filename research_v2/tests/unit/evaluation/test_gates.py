from __future__ import annotations

from dataclasses import replace
from typing import Any, cast

import pytest

from research_v2.evaluation.gates import (
    GateEvidence,
    GateStatus,
    ScoreHealthEvidence,
    evaluate_oos_gates,
)

# Reused below: one day that satisfies every per-day score-health requirement and one that fails
# the coverage floor, mirroring the 95/5 split in _evidence().
_HEALTHY_DAY = ScoreHealthEvidence(0.90, 10, 1e-6 + 1e-9)
_UNHEALTHY_DAY = ScoreHealthEvidence(0.89, 10, 1e-6 + 1e-9)


def _evidence() -> GateEvidence:
    return GateEvidence(
        mean_rank_ic=0.030,
        hac_t_statistic=2.000,
        yearly_mean_rank_ic={
            2020: 0.01,
            2021: 0.01,
            2022: 0.01,
            2023: 0.01,
            2024: 0.01,
            2025: -0.01,
        },
        candidate_delta_rank_ic=0.005,
        valid_test_day_count=100,
        score_health=tuple(
            [ScoreHealthEvidence(0.90, 10, 1e-6 + 1e-9)] * 95
            + [ScoreHealthEvidence(0.89, 10, 1e-6 + 1e-9)] * 5
        ),
    )


def test_double_gate_boundaries_and_all_other_oos_gates_pass_at_exact_thresholds() -> None:
    report = evaluate_oos_gates(_evidence())

    assert report.results["G1"].status is GateStatus.PASS
    assert report.results["G2"].status is GateStatus.PASS
    assert report.results["G3"].status is GateStatus.PASS
    assert report.results["G4"].status is GateStatus.PASS
    assert report.results["G5"].status is GateStatus.PASS
    assert report.results["G6"].status is GateStatus.PASS
    assert report.all_passed


def test_baseline_marks_relative_gate_not_applicable_and_missing_year_fails_closed() -> None:
    baseline = evaluate_oos_gates(_evidence(), is_baseline=True)
    incomplete = evaluate_oos_gates(
        GateEvidence(
            mean_rank_ic=0.1,
            hac_t_statistic=3.0,
            yearly_mean_rank_ic={2020: 0.1},
            candidate_delta_rank_ic=None,
            valid_test_day_count=10,
            score_health=tuple(ScoreHealthEvidence(0.95, 10, 0.1) for _ in range(10)),
        )
    )

    assert baseline.results["G4"].status is GateStatus.NOT_APPLICABLE
    assert baseline.all_passed
    assert incomplete.results["G3"].status is GateStatus.FAIL
    assert not incomplete.all_passed


def test_score_health_requires_coverage_unique_scores_and_dispersion() -> None:
    for health in (
        ScoreHealthEvidence(0.899, 20, 0.1),
        ScoreHealthEvidence(1.001, 20, 0.1),
        ScoreHealthEvidence(0.95, 9, 0.1),
        ScoreHealthEvidence(0.95, 20, 1e-6),
    ):
        report = evaluate_oos_gates(
            GateEvidence(
                mean_rank_ic=0.03,
                hac_t_statistic=2.0,
                yearly_mean_rank_ic={year: 0.01 for year in range(2020, 2026)},
                candidate_delta_rank_ic=0.005,
                valid_test_day_count=1,
                score_health=(health,),
            )
        )

        assert report.results["G6"].status is GateStatus.FAIL


def test_score_health_missing_or_misaligned_evidence_fails_closed() -> None:
    evidence = _evidence()
    missing = evaluate_oos_gates(
        GateEvidence(
            mean_rank_ic=evidence.mean_rank_ic,
            hac_t_statistic=evidence.hac_t_statistic,
            yearly_mean_rank_ic=evidence.yearly_mean_rank_ic,
            candidate_delta_rank_ic=evidence.candidate_delta_rank_ic,
            valid_test_day_count=1,
            score_health=(),
        )
    )
    misaligned = evaluate_oos_gates(
        GateEvidence(
            mean_rank_ic=evidence.mean_rank_ic,
            hac_t_statistic=evidence.hac_t_statistic,
            yearly_mean_rank_ic=evidence.yearly_mean_rank_ic,
            candidate_delta_rank_ic=evidence.candidate_delta_rank_ic,
            valid_test_day_count=2,
            score_health=evidence.score_health[:1],
        )
    )

    assert missing.results["G6"].status is GateStatus.FAIL
    assert misaligned.results["G6"].status is GateStatus.FAIL


def test_score_health_malformed_runtime_evidence_fails_closed() -> None:
    for health in (
        ScoreHealthEvidence(cast(Any, None), 20, 0.1),
        ScoreHealthEvidence(0.95, cast(Any, "20"), 0.1),
        ScoreHealthEvidence(0.95, 20, cast(Any, "not-a-number")),
    ):
        report = evaluate_oos_gates(
            GateEvidence(
                mean_rank_ic=0.03,
                hac_t_statistic=2.0,
                yearly_mean_rank_ic={year: 0.01 for year in range(2020, 2026)},
                candidate_delta_rank_ic=0.005,
                valid_test_day_count=1,
                score_health=(health,),
            )
        )

        assert report.results["G6"].status is GateStatus.FAIL

    malformed = evaluate_oos_gates(
        GateEvidence(
            mean_rank_ic=0.03,
            hac_t_statistic=2.0,
            yearly_mean_rank_ic={year: 0.01 for year in range(2020, 2026)},
            candidate_delta_rank_ic=0.005,
            valid_test_day_count=1,
            score_health=cast(Any, None),
        )
    )

    assert malformed.results["G6"].status is GateStatus.FAIL


# --- G5: the year-concentration gate -----------------------------------------
#
# G5 exists for one reason: an alpha carried by a single year is not an alpha, it is
# one lucky year. The five tests above never take G5 off PASS -- _evidence()'s positive
# yearly contributions are all 0.01, so its share is 1/5 and max, min, a clipped or an
# unclipped denominator, and a strict or an inclusive comparator all agree. Everything
# below is a fixture where those choices diverge.


def test_g5_fails_when_one_year_carries_the_whole_signal() -> None:
    """G5 measures the LARGEST year's share; the smallest would pass this easily."""
    dominated = replace(
        _evidence(),
        yearly_mean_rank_ic={
            2020: 0.10,
            2021: 0.001,
            2022: 0.001,
            2023: 0.001,
            2024: 0.001,
            2025: 0.001,
        },
    )

    report = evaluate_oos_gates(dominated)

    # Guard the guard: the largest year really does dominate -- 0.10 of 0.105.
    assert report.results["G5"].observed_value is not None
    assert report.results["G5"].observed_value > 0.95
    assert report.results["G5"].status is GateStatus.FAIL
    assert not report.all_passed


def test_g5_passes_at_exactly_half_and_fails_just_above() -> None:
    """The 0.50 bound is inclusive, so exactly half passes and a hair more does not."""
    exactly_half = replace(
        _evidence(),
        yearly_mean_rank_ic={2020: 0.10, 2021: 0.10, 2022: 0.0, 2023: 0.0, 2024: 0.0, 2025: 0.0},
    )
    just_above = replace(
        _evidence(),
        yearly_mean_rank_ic={
            2020: 0.1001,
            2021: 0.10,
            2022: 0.0,
            2023: 0.0,
            2024: 0.0,
            2025: 0.0,
        },
    )

    at_bound = evaluate_oos_gates(exactly_half).results["G5"]
    above = evaluate_oos_gates(just_above).results["G5"]

    assert at_bound.observed_value is not None and abs(at_bound.observed_value - 0.50) < 1e-12
    assert at_bound.status is GateStatus.PASS
    assert above.status is GateStatus.FAIL


def test_g5_clips_negative_years_out_of_the_denominator() -> None:
    """A losing year must not shrink the denominator and inflate another year's share.

    Three years at 0.20 and one at -0.30: clipped, the largest share is 0.20/0.60 and
    G5 passes; unclipped the denominator drops to 0.30, the same year reads as 0.667 and
    G5 fails. Whether a negative year can veto an otherwise unconcentrated alpha is a
    real semantic choice, so it is pinned rather than left to the arithmetic.
    """
    mixed = replace(
        _evidence(),
        yearly_mean_rank_ic={
            2020: 0.20,
            2021: 0.20,
            2022: 0.20,
            2023: -0.30,
            2024: 0.0,
            2025: 0.0,
        },
    )

    result = evaluate_oos_gates(mixed).results["G5"]

    assert result.observed_value is not None
    # 0.2 + 0.2 + 0.2 is not exactly 0.6 in binary floating point, so compare loosely.
    assert result.observed_value == pytest.approx(0.20 / 0.60)
    assert result.status is GateStatus.PASS


# --- thresholds: the side the exact-boundary test cannot see ------------------
#
# test_double_gate_boundaries... proves the gates pass AT their thresholds, which
# catches a threshold drifting upward (stricter) and cannot catch it drifting downward
# (looser) -- and looser is the direction that promotes a model that should not be.
# Each gate below is fed a value a hair under its threshold.


def test_g1_g4_and_g6_fail_just_below_their_thresholds() -> None:
    below_g1 = evaluate_oos_gates(replace(_evidence(), mean_rank_ic=0.0299))
    below_g4 = evaluate_oos_gates(replace(_evidence(), candidate_delta_rank_ic=0.0049))
    below_g6 = evaluate_oos_gates(
        replace(
            _evidence(),
            score_health=(_HEALTHY_DAY,) * 94 + (_UNHEALTHY_DAY,) * 6,
        )
    )

    assert below_g1.results["G1"].status is GateStatus.FAIL
    assert below_g4.results["G4"].status is GateStatus.FAIL
    # 94 healthy days of 100 is exactly 0.01 under the 0.95 floor.
    assert below_g6.results["G6"].observed_value == cast(Any, 0.94)
    assert below_g6.results["G6"].status is GateStatus.FAIL
    for report in (below_g1, below_g4, below_g6):
        assert not report.all_passed


def test_g2_and_g3_fail_just_below_their_thresholds() -> None:
    """The two thresholds the first measurement round only inferred about, now measured.

    G3's pass fixture sits at 5/6 = 0.8333, well clear of 0.70, so before this test a
    drift anywhere below 0.8333 was invisible: four positive years of six is 0.6667 and
    must fail. G2 sits exactly on 2.000, so 1.999 must fail.
    """
    below_g2 = evaluate_oos_gates(replace(_evidence(), hac_t_statistic=1.999))
    below_g3 = evaluate_oos_gates(
        replace(
            _evidence(),
            yearly_mean_rank_ic={
                2020: 0.01,
                2021: 0.01,
                2022: 0.01,
                2023: 0.01,
                2024: -0.01,
                2025: -0.01,
            },
        )
    )

    assert below_g2.results["G2"].status is GateStatus.FAIL
    assert below_g3.results["G3"].observed_value == cast(Any, 4 / 6)
    assert below_g3.results["G3"].status is GateStatus.FAIL


# --- fail-closed readings that must not be softened ---------------------------


def test_g3_does_not_count_a_zero_year_as_positive() -> None:
    """Four positive years, one exactly zero, one negative: 4/6 must fail G3.

    Counting the zero year would give 5/6 = 0.8333 and pass. A year whose mean RankIC is
    exactly zero is not evidence the alpha worked that year, and G3 is the gate that asks
    whether it worked in most years.
    """
    with_zero_year = replace(
        _evidence(),
        yearly_mean_rank_ic={
            2020: 0.01,
            2021: 0.01,
            2022: 0.01,
            2023: 0.01,
            2024: 0.0,
            2025: -0.01,
        },
    )

    result = evaluate_oos_gates(with_zero_year).results["G3"]

    assert result.observed_value == cast(Any, 4 / 6)
    assert result.status is GateStatus.FAIL


def test_missing_candidate_delta_fails_g4_and_the_whole_report() -> None:
    """No baseline means no relative evidence, and that must read as FAIL, not as excused.

    ``GateReport.all_passed`` accepts NOT_APPLICABLE -- correctly, so that a baseline is
    not penalised for a gate that only applies to candidates. That is exactly why a
    candidate with no delta must not be given NOT_APPLICABLE: the driver's own admission
    check reads ``oos.all_passed`` out of the run report, so excusing a missing delta
    would turn "no relative evidence, refused" into "no relative evidence, all gates
    passed". The registry path cannot reach this (its evidence parser demands a finite
    delta), but ``evaluate_frozen_oos`` accepts ``baseline_predictions=None`` with
    ``is_baseline=False``, and this is the gate that path lands on.
    """
    without_delta = replace(_evidence(), candidate_delta_rank_ic=None)

    report = evaluate_oos_gates(without_delta, is_baseline=False)

    assert report.results["G4"].status is GateStatus.FAIL
    assert report.results["G4"].threshold == 0.005
    assert not report.all_passed
    # And the exemption still belongs to the baseline alone.
    assert evaluate_oos_gates(without_delta, is_baseline=True).results["G4"].status is (
        GateStatus.NOT_APPLICABLE
    )


def test_incomplete_evidence_reports_not_computable_rather_than_a_number() -> None:
    """A gate that cannot be computed must say so in observed_value, not invent a value.

    With only 2020 present, G3 and G5 both fail -- but they fail because the required six
    years are missing, and ``observed_value=None`` is what distinguishes that from failing
    on a computed ratio. Dropping the completeness check keeps both gates at FAIL for this
    input while silently starting to report numbers (1/6 and 1.0), which is how a gate
    that stopped being computable goes unnoticed.
    """
    single_year = replace(
        _evidence(),
        yearly_mean_rank_ic={2020: 0.1},
        candidate_delta_rank_ic=None,
        valid_test_day_count=10,
        score_health=(_HEALTHY_DAY,) * 10,
    )

    report = evaluate_oos_gates(single_year)

    assert report.results["G3"].status is GateStatus.FAIL
    assert report.results["G3"].observed_value is None
    assert report.results["G5"].status is GateStatus.FAIL
    assert report.results["G5"].observed_value is None


def test_non_positive_valid_day_count_fails_g6_instead_of_dividing_by_zero() -> None:
    """The day-count guard is fail-closed; without it the ratio raises ZeroDivisionError."""
    no_days = replace(_evidence(), valid_test_day_count=0, score_health=())

    report = evaluate_oos_gates(no_days)

    assert report.results["G6"].status is GateStatus.FAIL
    assert report.results["G6"].observed_value is None
    assert not report.all_passed
