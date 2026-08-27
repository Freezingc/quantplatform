from __future__ import annotations

import json
from datetime import date

import pytest

from research_v2.evaluation.gates import GateEvidence, ScoreHealthEvidence
from research_v2.monitoring.shadow import ShadowDeploymentApprovalStore
from research_v2.registry.model_registry import ModelRegistry, ModelStatus
from research_v2.reporting.shadow import build_shadow_daily_report

_MODEL_ID = "sha256:" + "a" * 64


def _gate_evidence() -> GateEvidence:
    return GateEvidence(
        mean_rank_ic=0.04,
        hac_t_statistic=2.5,
        yearly_mean_rank_ic={year: 0.04 for year in range(2020, 2026)},
        candidate_delta_rank_ic=0.006,
        valid_test_day_count=6,
        score_health=tuple(
            ScoreHealthEvidence(coverage=0.95, unique_score_count=30, score_std=0.1)
            for _ in range(6)
        ),
    )


def test_shadow_transition_requires_a_verified_deployment_approval(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QRV2_SHADOW_APPROVAL_HMAC_KEY", "shadow-approval-key")
    monkeypatch.setenv("QRV2_SHADOW_DEPLOYER_ID", "shadow-controller")
    registry = ModelRegistry()
    registry.register_draft(_MODEL_ID, actor="trainer")
    registry.promote_walk_forward(_MODEL_ID, actor="reviewer", evidence=_gate_evidence())
    approvals = ShadowDeploymentApprovalStore(
        tmp_path / "shadow-approvals",  # type: ignore[operator]
        signing_key=b"shadow-approval-key",
    )
    approval = approvals.issue(
        model_id=_MODEL_ID,
        actor="risk-approver",
        walk_forward_evidence_hash=registry.history(_MODEL_ID)[-1].evidence_hashes[0],
        artifact_bundle_hash="c" * 64,
        inference_contract_hash="d" * 64,
        v2test_drill_hash="e" * 64,
        rollback_plan_hash="f" * 64,
    )

    shadow = registry.enter_shadow(
        _MODEL_ID,
        approval=approval,
        approval_root=tmp_path / "shadow-approvals",  # type: ignore[operator]
    )

    assert shadow.status is ModelStatus.SHADOW
    assert registry.history(_MODEL_ID)[-1].evidence_hashes == (approval.approval_id,)


def test_generic_registry_transition_cannot_bypass_shadow_deployment_approval() -> None:
    registry = ModelRegistry()
    registry.register_draft(_MODEL_ID, actor="trainer")
    registry.promote_walk_forward(_MODEL_ID, actor="reviewer", evidence=_gate_evidence())

    with pytest.raises(ValueError, match="deployment approval"):
        registry.transition(
            _MODEL_ID,
            ModelStatus.SHADOW,
            actor="attacker",
            evidence_hashes=("a" * 64,),
            reason="bypass approval",
        )


def test_shadow_transition_binds_the_approval_to_actual_walk_forward_evidence(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QRV2_SHADOW_APPROVAL_HMAC_KEY", "shadow-approval-key")
    monkeypatch.setenv("QRV2_SHADOW_DEPLOYER_ID", "shadow-controller")
    registry = ModelRegistry()
    registry.register_draft(_MODEL_ID, actor="trainer")
    registry.promote_walk_forward(_MODEL_ID, actor="reviewer", evidence=_gate_evidence())
    approvals = ShadowDeploymentApprovalStore(
        tmp_path / "shadow-approvals",  # type: ignore[operator]
        signing_key=b"shadow-approval-key",
    )
    mismatched = approvals.issue(
        model_id=_MODEL_ID,
        actor="risk-approver",
        walk_forward_evidence_hash="b" * 64,
        artifact_bundle_hash="c" * 64,
        inference_contract_hash="d" * 64,
        v2test_drill_hash="e" * 64,
        rollback_plan_hash="f" * 64,
    )

    with pytest.raises(ValueError, match="walk-forward"):
        registry.enter_shadow(
            _MODEL_ID,
            approval=mismatched,
            approval_root=tmp_path / "shadow-approvals",  # type: ignore[operator]
        )


def test_shadow_transition_requires_a_separate_deployment_operator(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QRV2_SHADOW_APPROVAL_HMAC_KEY", "shadow-approval-key")
    monkeypatch.setenv("QRV2_SHADOW_DEPLOYER_ID", "risk-approver")
    registry = ModelRegistry()
    registry.register_draft(_MODEL_ID, actor="trainer")
    registry.promote_walk_forward(_MODEL_ID, actor="reviewer", evidence=_gate_evidence())
    approvals = ShadowDeploymentApprovalStore(
        tmp_path / "shadow-approvals",  # type: ignore[operator]
        signing_key=b"shadow-approval-key",
    )
    approval = approvals.issue(
        model_id=_MODEL_ID,
        actor="risk-approver",
        walk_forward_evidence_hash=registry.history(_MODEL_ID)[-1].evidence_hashes[0],
        artifact_bundle_hash="c" * 64,
        inference_contract_hash="d" * 64,
        v2test_drill_hash="e" * 64,
        rollback_plan_hash="f" * 64,
    )

    with pytest.raises(ValueError, match="separate"):
        registry.enter_shadow(
            _MODEL_ID,
            approval=approval,
            approval_root=tmp_path / "shadow-approvals",  # type: ignore[operator]
        )


def test_shadow_transition_rejects_a_caller_signed_approval(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QRV2_SHADOW_APPROVAL_HMAC_KEY", "trusted-approval-key")
    monkeypatch.setenv("QRV2_SHADOW_DEPLOYER_ID", "shadow-controller")
    registry = ModelRegistry()
    registry.register_draft(_MODEL_ID, actor="trainer")
    registry.promote_walk_forward(_MODEL_ID, actor="reviewer", evidence=_gate_evidence())
    root = tmp_path / "shadow-approvals"  # type: ignore[operator]
    approval = ShadowDeploymentApprovalStore(root, signing_key=b"attacker-key").issue(
        model_id=_MODEL_ID,
        actor="risk-approver",
        walk_forward_evidence_hash=registry.history(_MODEL_ID)[-1].evidence_hashes[0],
        artifact_bundle_hash="c" * 64,
        inference_contract_hash="d" * 64,
        v2test_drill_hash="e" * 64,
        rollback_plan_hash="f" * 64,
    )

    with pytest.raises(ValueError, match="signature"):
        registry.enter_shadow(
            _MODEL_ID,
            approval=approval,
            approval_root=root,
        )


def test_shadow_daily_report_is_label_free_content_addressed_evidence() -> None:
    report = build_shadow_daily_report(
        signal_date=date(2026, 8, 27),
        model_id=_MODEL_ID,
        model_version=1,
        pipeline_status="SUCCESS",
        prediction_published=True,
        inference_manifest_id="sha256:" + "b" * 64,
        artifact_bundle_hash="c" * 64,
        source_watermark_id="d" * 64,
        latency_seconds=121.5,
        universe_size=1_200,
        feature_coverage={"quality": 0.98},
        score_health={"finite_coverage": 0.99, "unique_score_count": 800, "score_std": 0.15},
        seed_agreement={"minimum_spearman": 0.42},
        drift_alert_keys=("psi:quality",),
        event_ids=("e" * 64,),
    )

    report.verify()
    payload = json.loads(report.to_json())

    assert payload["performance_status"] == "LOCKED_NOT_COMPUTED"
    assert payload["prediction_published"] is True
    assert payload["latency_slo_seconds"] > 0
    assert payload["report_id"] == report.report_id
    assert "label" not in report.to_json().lower()
    assert "rankic" not in report.to_json().lower()


def test_shadow_daily_report_rejects_label_or_return_data_in_values() -> None:
    with pytest.raises(ValueError, match="payload"):
        build_shadow_daily_report(
            signal_date=date(2026, 8, 27),
            model_id=_MODEL_ID,
            model_version=1,
            pipeline_status="SUCCESS",
            prediction_published=True,
            inference_manifest_id="sha256:" + "b" * 64,
            artifact_bundle_hash="c" * 64,
            source_watermark_id="d" * 64,
            latency_seconds=121.5,
            universe_size=1_200,
            feature_coverage={"quality": 0.98},
            score_health={"finite_coverage": 0.99, "unique_score_count": 800, "score_std": 0.15},
            seed_agreement={"minimum_spearman": 0.42},
            drift_alert_keys=(),
            event_ids=("label-derived-event",),
        )


def test_shadow_daily_report_rejects_a_failed_pipeline_that_claims_publication() -> None:
    with pytest.raises(ValueError, match="pipeline"):
        build_shadow_daily_report(
            signal_date=date(2026, 8, 27),
            model_id=_MODEL_ID,
            model_version=1,
            pipeline_status="FAILED",
            prediction_published=True,
            inference_manifest_id="sha256:" + "b" * 64,
            artifact_bundle_hash="c" * 64,
            source_watermark_id="d" * 64,
            latency_seconds=121.5,
            universe_size=1_200,
            feature_coverage={"quality": 0.98},
            score_health={"finite_coverage": 0.99, "unique_score_count": 800, "score_std": 0.15},
            seed_agreement={"minimum_spearman": 0.42},
            drift_alert_keys=(),
            event_ids=(),
        )
