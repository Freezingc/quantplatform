from __future__ import annotations

import os
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from research_v2.evaluation.gates import GateEvidence, ScoreHealthEvidence
from research_v2.monitoring.shadow import (
    ShadowDeploymentApprovalStore,
    ShadowEvidenceStore,
    ShadowQualification,
    ShadowQualificationStatus,
)
from research_v2.registry.active_pointer import (
    ActiveModelPointerStore,
    PointerEligibilityGuard,
    PointerSwitchEvidence,
)
from research_v2.registry.locked_test import (
    IssuedLabelMaturityWatermark,
    LabelMaturityWatermark,
    LockedTestAuthority,
    LockedTestEvaluator,
    LockedTestMaturityEvidenceStore,
    LockedTestMetrics,
    LockedTestOperatorIdentityStore,
    LockedTestPreregistration,
    LockedTestPreregistrationStore,
    LockedTestTradingCalendar,
    ProductionDeploymentApprovalStore,
)
from research_v2.registry.model_registry import ModelRegistry, ModelStatus

pytestmark = pytest.mark.integration

_MODEL_ID = "sha256:" + "a" * 64
_RUN_ID = "00000000-0000-0000-0000-000000000001"


def _preregistration() -> LockedTestPreregistration:
    return LockedTestPreregistration(
        model_id=_MODEL_ID,
        lineage_id=_MODEL_ID,
        seed_artifact_hashes=("a" * 64, "b" * 64, "c" * 64),
        artifact_bundle_hash="a" * 64,
        final_train_manifest_hash="d" * 64,
        final_validation_manifest_hash="e" * 64,
        factor_set_id="factor-set-final-v1",
        preprocessor_hash="f" * 64,
        universe_id="cn-liquid-a-2026-v1",
        label_definition_hash="1" * 64,
        evaluation_code_hash="2" * 64,
        required_gates=("G1", "G2", "G3", "G4", "G5", "G6"),
        command=("qr-v2", "registry", "locked-test", "evaluate"),
        applicant_role="research-owner",
        approver_role="risk-approver",
    )


def _walk_forward_evidence() -> GateEvidence:
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


def _maturity_evidence(
    root: Path,
) -> tuple[
    LockedTestMaturityEvidenceStore,
    LockedTestTradingCalendar,
    IssuedLabelMaturityWatermark,
]:
    store = LockedTestMaturityEvidenceStore(
        root / "maturity",
        signing_key=b"integration-maturity-key",
    )
    calendar = store.issue_calendar(
        (
            date(2026, 12, 23),
            date(2026, 12, 24),
            date(2026, 12, 25),
            date(2026, 12, 28),
            date(2026, 12, 29),
            date(2026, 12, 30),
            date(2026, 12, 31),
            date(2027, 1, 4),
            date(2027, 1, 5),
            date(2027, 1, 6),
            date(2027, 1, 7),
            date(2027, 1, 8),
            date(2027, 1, 11),
        )
    )
    watermark = store.issue_watermark(
        LabelMaturityWatermark(
            watermark_id="6" * 64,
            exit_date=date(2027, 1, 11),
            adjusted_open_stable=True,
            reconciled=True,
            dq_passed=True,
            published=True,
        )
    )
    return store, calendar, watermark


@pytest.mark.skipif(
    os.environ.get("QRV2_INTEGRATION") != "1",
    reason="set QRV2_INTEGRATION=1 to run the locked-test integration workflow",
)
def test_signed_locked_test_workflow_reaches_a_production_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QRV2_SHADOW_APPROVAL_HMAC_KEY", "integration-shadow-approval-key")
    monkeypatch.setenv("QRV2_SHADOW_EVIDENCE_HMAC_KEY", "integration-shadow-evidence-key")
    monkeypatch.setenv("QRV2_SHADOW_DEPLOYER_ID", "integration-shadow-controller")
    monkeypatch.setenv("QRV2_PRODUCTION_APPROVAL_HMAC_KEY", "integration-production-approval-key")
    monkeypatch.setenv("QRV2_LOCKED_TEST_ACCESS_HMAC_KEY", "integration-locked-test-access-key")
    monkeypatch.setenv("QRV2_LOCKED_TEST_MATURITY_HMAC_KEY", "integration-maturity-key")
    monkeypatch.setenv("QRV2_LOCKED_TEST_IDENTITY_HMAC_KEY", "integration-identity-key")
    preregistration = _preregistration()
    preregistration_store = LockedTestPreregistrationStore(
        tmp_path / "preregistrations", signing_key=b"integration-preregistration-key"
    )
    issued = preregistration_store.issue(preregistration)
    registry = ModelRegistry()
    registry.register_draft(
        preregistration.model_id,
        actor="trainer",
    )
    registry.promote_walk_forward(
        preregistration.model_id,
        actor="reviewer",
        evidence=_walk_forward_evidence(),
    )
    shadow_approval_root = tmp_path / "shadow-approvals"
    shadow_approval = ShadowDeploymentApprovalStore(
        shadow_approval_root,
        signing_key=b"integration-shadow-approval-key",
    ).issue(
        model_id=preregistration.model_id,
        actor="shadow-risk-approver",
        walk_forward_evidence_hash=registry.history(preregistration.model_id)[-1].evidence_hashes[0],
        artifact_bundle_hash="a" * 64,
        inference_contract_hash="3" * 64,
        v2test_drill_hash="4" * 64,
        rollback_plan_hash="5" * 64,
    )
    registry.enter_shadow(
        preregistration.model_id,
        approval=shadow_approval,
        approval_root=shadow_approval_root,
    )
    _, maturity_calendar, watermark = _maturity_evidence(tmp_path)
    identities = LockedTestOperatorIdentityStore(
        tmp_path / "operator-identities",
        signing_key=b"integration-identity-key",
    )
    authority = LockedTestAuthority(
        tmp_path / "locked-test-authority",
        preregistration_store=preregistration_store,
        registry=registry,
        maturity_calendar=maturity_calendar,
        maturity_evidence_root=tmp_path / "maturity",
        operator_identity_root=tmp_path / "operator-identities",
        signing_key=b"integration-locked-test-authority-key",
        access_signing_key=b"integration-locked-test-access-key",
        clock=lambda: datetime(2027, 1, 12, tzinfo=UTC),
    )
    request = authority.request_unlock(
        issued,
        applicant=identities.issue(actor_id="research-principal", role="research-owner"),
        current_date=date(2027, 1, 12),
        watermark=watermark,
    )
    shadow_evidence_root = tmp_path / "shadow-evidence"
    valid_dates = tuple(date(2026, 1, 1) + timedelta(days=index) for index in range(60))
    shadow_evidence = ShadowEvidenceStore(
        shadow_evidence_root,
        signing_key=b"integration-shadow-evidence-key",
    ).issue(
        ShadowQualification(
            model_id=preregistration.model_id,
            trading_calendar_id="7" * 64,
            valid_day_count=60,
            status=ShadowQualificationStatus.READY_FOR_REVIEW,
            max_consecutive_psi_errors=0,
            valid_signal_dates=valid_dates,
            invalid_signal_dates=(),
            valid_report_ids=tuple(f"{index:064x}" for index in range(60)),
            valid_runtime_day_ids=tuple(f"{index + 100:064x}" for index in range(60)),
            invalid_runtime_day_ids=(),
        )
    )
    capability = authority.approve(
        request,
        approver=identities.issue(actor_id="risk-principal", role="risk-approver"),
        shadow_evidence=shadow_evidence,
        shadow_evidence_root=shadow_evidence_root,
    )
    result_root = tmp_path / "locked-test-results"
    result = LockedTestEvaluator(
        result_root,
        authority=authority,
        registry=registry,
        signing_key=b"integration-locked-test-results-key",
        artifact_bundle_hash="a" * 64,
    ).evaluate(
        capability,
        issued,
        actor="locked-test-evaluator",
        metrics_reader=lambda _access: LockedTestMetrics(
            mean_rank_ic=0.04,
            hac_t_statistic=2.5,
            candidate_delta_rank_ic=0.006,
            valid_test_day_count=6,
            score_health=tuple(
                ScoreHealthEvidence(coverage=0.95, unique_score_count=30, score_std=0.1)
                for _ in range(6)
            ),
        ),
    )
    approval_root = tmp_path / "production-approvals"
    approval = ProductionDeploymentApprovalStore(
        approval_root,
        registry=registry,
        result_root=result_root,
        result_verification_key=b"integration-locked-test-results-key",
        signing_key=b"integration-production-approval-key",
    ).issue(
        result,
        actor="production-risk-approver",
    )
    pointer_store = ActiveModelPointerStore(
        eligibility=PointerEligibilityGuard(registry, bundle_digest=lambda _model_id: "a" * 64),
        clock=lambda: datetime(2027, 1, 12, tzinfo=UTC),
    )
    update = pointer_store.compare_and_swap(
        consumer_id="paper-sim",
        environment="production",
        expected_current=None,
        model_id=preregistration.model_id,
        model_version=1,
        actor="production-operator",
        reason="integration locked-test release",
        evidence=PointerSwitchEvidence(
            production_approval=approval,
            production_approval_root=approval_root,
        ),
        run_id=_RUN_ID,
    )

    assert registry.get(preregistration.model_id).status is ModelStatus.PRODUCTION
    assert update is not None
    assert update.event.production_approval_id == approval.approval_id
