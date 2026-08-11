from __future__ import annotations

from research_v2.evaluation.gates import GateEvidence, GateStatus, evaluate_oos_gates


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
        good_score_day_count=95,
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
            good_score_day_count=10,
        )
    )

    assert baseline.results["G4"].status is GateStatus.NOT_APPLICABLE
    assert baseline.all_passed
    assert incomplete.results["G3"].status is GateStatus.FAIL
    assert not incomplete.all_passed
