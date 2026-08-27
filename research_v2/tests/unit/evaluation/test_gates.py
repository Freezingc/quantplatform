from __future__ import annotations

from typing import Any, cast

from research_v2.evaluation.gates import (
    GateEvidence,
    GateStatus,
    ScoreHealthEvidence,
    evaluate_oos_gates,
)


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
