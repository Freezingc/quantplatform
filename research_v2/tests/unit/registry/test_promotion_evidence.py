"""Tests for the registry's promotion-evidence producer.

The producer exists because the registry had a consumer and no producer: every candidate,
including one that passed all six gates, stopped at the run report. These tests pin the
round-trip through the real loader, the refusal paths, and one specific trap -- the report's
valid_day_count and the evidence's valid_test_day_count are different measures, and a
cross-check that equated them would reject every real run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from research_v2.evaluation.gates import (
    GateEvidence,
    ScoreHealthEvidence,
    evaluate_oos_gates,
)
from research_v2.evaluation.hac import HACMeanTest
from research_v2.evaluation.model_evaluator import FrozenOosEvaluation
from research_v2.evaluation.rank_ic import RankICSummary
from research_v2.registry.model_registry import (
    build_walk_forward_evidence_document,
    load_walk_forward_evidence,
)
from research_v2.registry.promotion_evidence import (
    PromotionEvidenceError,
    _cross_check_against_report,
    _load_full_report,
    _prediction_ids_by_fold_and_kind,
    emit_promotion_evidence,
)

# The v2 fold contract: 969 label days wide enough to score, 945 of them producing a usable
# RankIC, so G6 = 945/969 = 0.9752. Those are the real numbers from the M0R run.
LABEL_DAYS = 969
SCORING_DAYS = 945
CONTRACT_YEARS = (2022, 2023, 2024, 2025)
KEYS = ["signal_date", "ts_code"]


def _passing_evidence(*, delta: float | None = 0.009149) -> GateEvidence:
    health = tuple(
        [
            ScoreHealthEvidence(coverage=0.98, unique_score_count=1800, score_std=0.29)
            for _ in range(SCORING_DAYS)
        ]
        + [
            # 24 days the candidate could not score: coverage below the 0.90 floor
            ScoreHealthEvidence(coverage=0.50, unique_score_count=1800, score_std=0.29)
            for _ in range(LABEL_DAYS - SCORING_DAYS)
        ]
    )
    return GateEvidence(
        mean_rank_ic=0.094376,
        hac_t_statistic=12.905,
        yearly_mean_rank_ic={year: 0.09 for year in CONTRACT_YEARS},
        candidate_delta_rank_ic=delta,
        valid_test_day_count=LABEL_DAYS,
        score_health=health,
    )


def _evaluation(evidence: GateEvidence) -> FrozenOosEvaluation:
    daily = pd.DataFrame(
        {"signal_date": [pd.Timestamp("2022-01-04")], "mean_rank_ic": [evidence.mean_rank_ic]}
    )
    return FrozenOosEvaluation(
        daily_rank_ic=daily,
        summary=RankICSummary(
            valid_day_count=SCORING_DAYS,
            mean_rank_ic=evidence.mean_rank_ic,
            rank_ic_std=0.126,
            icir=0.749,
        ),
        hac=HACMeanTest(
            observation_count=SCORING_DAYS,
            max_lags=5,
            mean=evidence.mean_rank_ic,
            standard_error=0.0073,
            t_statistic=evidence.hac_t_statistic,
            p_value=1e-30,
        ),
        yearly_mean_rank_ic=dict(evidence.yearly_mean_rank_ic),
        baseline_mean_rank_ic=0.085227,
        delta_rank_ic=evidence.candidate_delta_rank_ic,
        gate_evidence=evidence,
        gate_report=evaluate_oos_gates(evidence, is_baseline=False),
        yearly_summary=daily,
    )


def _report_from(evidence: GateEvidence) -> dict[str, object]:
    """The shape the candidate driver writes -- note valid_day_count is the SCORING days."""
    return {
        "family": "M0R",
        "mode": "full",
        "folds": [
            {"fold_id": fold, "model_selection_id": f"sel-{fold.lower()}"}
            for fold in ("F1", "F2", "F3", "F4")
        ],
        "oos": {
            "valid_day_count": SCORING_DAYS,
            "mean_rank_ic": evidence.mean_rank_ic,
            "icir": 0.749,
            "hac_t_statistic": evidence.hac_t_statistic,
            "baseline_mean_rank_ic": 0.085227,
            "delta_rank_ic": evidence.candidate_delta_rank_ic,
            "yearly_mean_rank_ic": {
                str(year): value for year, value in evidence.yearly_mean_rank_ic.items()
            },
            "gates": {
                gate: {
                    "status": result.status.value,
                    "observed_value": result.observed_value,
                    "threshold": result.threshold,
                }
                for gate, result in evaluate_oos_gates(evidence, is_baseline=False).results.items()
            },
            "all_passed": True,
        },
    }


def test_document_round_trips_through_the_real_registry_loader(tmp_path: Path) -> None:
    """The producer's output must be exactly what the consumer accepts, and it must promote."""
    evidence = _passing_evidence()
    document = build_walk_forward_evidence_document(_evaluation(evidence))
    path = tmp_path / "promotion_evidence.json"
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")

    loaded, digest, report = load_walk_forward_evidence(path)
    assert loaded.mean_rank_ic == evidence.mean_rank_ic
    assert loaded.hac_t_statistic == evidence.hac_t_statistic
    assert loaded.candidate_delta_rank_ic == evidence.candidate_delta_rank_ic
    assert loaded.valid_test_day_count == evidence.valid_test_day_count
    assert loaded.yearly_mean_rank_ic == evidence.yearly_mean_rank_ic
    assert loaded.score_health == evidence.score_health
    assert len(digest) == 64
    assert all(result.status.value == "PASS" for result in report.results.values())


def test_document_is_refused_for_a_baselines_own_walk_forward(tmp_path: Path) -> None:
    """A baseline run has no delta; presenting one as promotable evidence must fail."""
    baseline = _passing_evidence(delta=None)
    with pytest.raises(ValueError, match="candidate delta"):
        build_walk_forward_evidence_document(_evaluation(baseline))


def test_document_carries_the_schema_the_loader_demands() -> None:
    document = build_walk_forward_evidence_document(_evaluation(_passing_evidence()))
    assert document["schema_version"] == 1
    assert document["role"] == "ORDINARY_TEST"
    payload = document["gate_evidence"]
    assert isinstance(payload, dict)
    assert set(payload) == {
        "mean_rank_ic",
        "hac_t_statistic",
        "yearly_mean_rank_ic",
        "candidate_delta_rank_ic",
        "valid_test_day_count",
        "score_health",
    }
    assert set(payload["yearly_mean_rank_ic"]) == {str(year) for year in CONTRACT_YEARS}


def test_cross_check_accepts_the_two_different_day_counts(tmp_path: Path) -> None:
    """The trap: report valid_day_count=945 against evidence valid_test_day_count=969.

    Both are correct and they are not the same quantity -- their ratio is G6. A cross-check
    that compared them would reject every real run, so this pins that it does not.
    """
    evidence = _passing_evidence()
    document = build_walk_forward_evidence_document(_evaluation(evidence))
    path = tmp_path / "promotion_evidence.json"
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    report = _report_from(evidence)
    assert report["oos"]["valid_day_count"] == SCORING_DAYS
    assert evidence.valid_test_day_count == LABEL_DAYS
    _cross_check_against_report(report, path)  # must not raise


def _mutate_yearly(oos: dict[str, object]) -> None:
    yearly = oos["yearly_mean_rank_ic"]
    assert isinstance(yearly, dict)
    oos["yearly_mean_rank_ic"] = {**yearly, "2023": 0.01}


@pytest.mark.parametrize(
    "mutate",
    [
        pytest.param(lambda oos: oos.update(mean_rank_ic=0.05), id="mean_rank_ic"),
        pytest.param(lambda oos: oos.update(delta_rank_ic=0.001), id="delta"),
        pytest.param(lambda oos: oos.update(hac_t_statistic=1.0), id="hac_t"),
        pytest.param(_mutate_yearly, id="yearly"),
        pytest.param(
            lambda oos: oos.update(
                gates={**oos["gates"], "G4": {**oos["gates"]["G4"], "status": "FAIL"}}
            ),
            id="gate_status",
        ),
        pytest.param(
            lambda oos: oos.update(
                gates={**oos["gates"], "G1": {**oos["gates"]["G1"], "observed_value": 0.01}}
            ),
            id="gate_observed_value",
        ),
    ],
)
def test_cross_check_refuses_a_report_that_disagrees(tmp_path: Path, mutate) -> None:
    evidence = _passing_evidence()
    document = build_walk_forward_evidence_document(_evaluation(evidence))
    path = tmp_path / "promotion_evidence.json"
    path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    report = _report_from(evidence)
    oos = report["oos"]
    assert isinstance(oos, dict)
    mutate(oos)
    with pytest.raises(PromotionEvidenceError, match="disagrees"):
        _cross_check_against_report(report, path)


def _write_prediction_manifest(root: Path, prediction_id: str, fold: str, kind: str) -> None:
    directory = root / prediction_id
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "manifest.json").write_text(
        json.dumps(
            {
                "prediction_id": prediction_id,
                "metadata": {"fold_id": fold, "kind": kind},
                "sha256": "0" * 64,
                "signature": "0" * 64,
            }
        ),
        encoding="utf-8",
    )


def test_prediction_index_maps_fold_and_kind(tmp_path: Path) -> None:
    root = tmp_path / "predictions"
    _write_prediction_manifest(root, "a" * 64, "F1", "candidate")
    _write_prediction_manifest(root, "b" * 64, "F1", "best_simple")
    index = _prediction_ids_by_fold_and_kind(tmp_path)
    assert index == {("F1", "candidate"): "a" * 64, ("F1", "best_simple"): "b" * 64}


def test_prediction_index_refuses_two_candidates_for_one_fold(tmp_path: Path) -> None:
    root = tmp_path / "predictions"
    _write_prediction_manifest(root, "a" * 64, "F1", "candidate")
    _write_prediction_manifest(root, "c" * 64, "F1", "candidate")
    with pytest.raises(PromotionEvidenceError, match="two candidate predictions"):
        _prediction_ids_by_fold_and_kind(tmp_path)


def test_prediction_index_requires_the_directory(tmp_path: Path) -> None:
    with pytest.raises(PromotionEvidenceError, match="no frozen prediction artifacts"):
        _prediction_ids_by_fold_and_kind(tmp_path)


def test_report_must_be_a_full_run(tmp_path: Path) -> None:
    smoke = tmp_path / "smoke.json"
    smoke.write_text(json.dumps({"family": "M0R", "mode": "smoke"}), encoding="utf-8")
    with pytest.raises(PromotionEvidenceError, match="full run report"):
        _load_full_report(smoke)
    with pytest.raises(PromotionEvidenceError, match="does not exist"):
        _load_full_report(tmp_path / "absent.json")


def test_emit_requires_a_signing_key(tmp_path: Path) -> None:
    report = tmp_path / "report.json"
    report.write_text(json.dumps(_report_from(_passing_evidence())), encoding="utf-8")
    with pytest.raises(PromotionEvidenceError, match="signing key"):
        emit_promotion_evidence(
            tmp_path, report, tmp_path / "promotion_evidence.json", signing_key=b""
        )


def test_emit_refuses_a_workspace_missing_a_fold_prediction(tmp_path: Path) -> None:
    """Fail closed on an incomplete workspace rather than emitting partial evidence."""
    report_path = tmp_path / "report.json"
    report_path.write_text(json.dumps(_report_from(_passing_evidence())), encoding="utf-8")
    root = tmp_path / "predictions"
    for fold in ("F1", "F2", "F3", "F4"):
        _write_prediction_manifest(root, f"{fold.lower()}-cand", fold, "candidate")
    # F4's baseline prediction is missing
    for fold in ("F1", "F2", "F3"):
        _write_prediction_manifest(root, f"{fold.lower()}-base", fold, "best_simple")
    with pytest.raises(PromotionEvidenceError, match="best_simple"):
        emit_promotion_evidence(
            tmp_path, report_path, tmp_path / "promotion_evidence.json", signing_key=b"trust-root"
        )
