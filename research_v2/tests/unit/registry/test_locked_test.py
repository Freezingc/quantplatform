from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import pytest

from research_v2.datasets.builder import LockedDatasetError
from research_v2.evaluation.gates import GateEvidence, ScoreHealthEvidence
from research_v2.monitoring.shadow import (
    ShadowDeploymentApprovalStore,
    ShadowEvidence,
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
    IssuedPreregistration,
    LabelMaturityWatermark,
    LockedTestAuthority,
    LockedTestCapability,
    LockedTestEvaluator,
    LockedTestMaturityEvidenceStore,
    LockedTestMaturityGuard,
    LockedTestMetrics,
    LockedTestOperatorIdentityStore,
    LockedTestPreregistration,
    LockedTestPreregistrationStore,
    LockedTestTradingCalendar,
    ProductionDeploymentApprovalStore,
    _canonical_json,
)
from research_v2.registry.model_registry import ModelRegistry, ModelStatus

_MODEL_ID = "sha256:" + "a" * 64


@pytest.fixture(autouse=True)
def _trusted_shadow_evidence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QRV2_SHADOW_APPROVAL_HMAC_KEY", "locked-test-approval-key")
    monkeypatch.setenv("QRV2_SHADOW_EVIDENCE_HMAC_KEY", "locked-test-evidence-key")
    monkeypatch.setenv("QRV2_SHADOW_DEPLOYER_ID", "locked-test-shadow-controller")
    monkeypatch.setenv("QRV2_PRODUCTION_APPROVAL_HMAC_KEY", "production-approval-key")
    monkeypatch.setenv("QRV2_LOCKED_TEST_ACCESS_HMAC_KEY", "locked-test-access-key")
    monkeypatch.setenv("QRV2_LOCKED_TEST_MATURITY_HMAC_KEY", "maturity-key")
    monkeypatch.setenv("QRV2_LOCKED_TEST_IDENTITY_HMAC_KEY", "identity-key")


def _preregistration(
    *,
    model_id: str = _MODEL_ID,
    lineage_id: str = _MODEL_ID,
) -> LockedTestPreregistration:
    return LockedTestPreregistration(
        model_id=model_id,
        lineage_id=lineage_id,
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


def _shadow_registry(preregistration: LockedTestPreregistration, root: Path) -> ModelRegistry:
    registry = ModelRegistry()
    _enter_shadow(registry, preregistration, root)
    return registry


def _enter_shadow(
    registry: ModelRegistry,
    preregistration: LockedTestPreregistration,
    root: Path,
    *,
    parent_model_id: str | None = None,
) -> None:
    registry.register_draft(
        preregistration.model_id,
        actor="trainer",
        parent_model_id=parent_model_id,
    )
    registry.promote_walk_forward(
        preregistration.model_id,
        actor="reviewer",
        evidence=_walk_forward_evidence(),
    )
    approval = ShadowDeploymentApprovalStore(
        root / "shadow-approvals", signing_key=b"locked-test-approval-key"
    ).issue(
        model_id=preregistration.model_id,
        actor="shadow-risk-approver",
        walk_forward_evidence_hash=registry.history(preregistration.model_id)[-1].evidence_hashes[0],
        artifact_bundle_hash="3" * 64,
        inference_contract_hash="4" * 64,
        v2test_drill_hash="5" * 64,
        rollback_plan_hash="6" * 64,
    )
    registry.enter_shadow(
        preregistration.model_id,
        approval=approval,
        approval_root=root / "shadow-approvals",
    )


def _ready_shadow_evidence(root: Path, model_id: str) -> ShadowEvidence:
    dates = tuple(date(2026, 1, 1) + timedelta(days=index) for index in range(60))
    qualification = ShadowQualification(
        model_id=model_id,
        trading_calendar_id="7" * 64,
        valid_day_count=60,
        status=ShadowQualificationStatus.READY_FOR_REVIEW,
        max_consecutive_psi_errors=0,
        valid_signal_dates=dates,
        invalid_signal_dates=(),
        valid_report_ids=tuple(f"{index:064x}" for index in range(60)),
        valid_runtime_day_ids=tuple(f"{index + 100:064x}" for index in range(60)),
        invalid_runtime_day_ids=(),
    )
    return ShadowEvidenceStore(root, signing_key=b"locked-test-evidence-key").issue(qualification)


def _maturity_evidence(
    root: Path,
) -> tuple[
    LockedTestMaturityEvidenceStore,
    LockedTestTradingCalendar,
    IssuedLabelMaturityWatermark,
]:
    store = LockedTestMaturityEvidenceStore(root / "maturity", signing_key=b"maturity-key")
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
    return store, calendar, store.issue_watermark(_watermark())


def _operator_identities(root: Path) -> LockedTestOperatorIdentityStore:
    return LockedTestOperatorIdentityStore(
        root / "operator-identities",
        signing_key=b"identity-key",
    )


def _watermark() -> LabelMaturityWatermark:
    return LabelMaturityWatermark(
        watermark_id="8" * 64,
        exit_date=date(2027, 1, 11),
        adjusted_open_stable=True,
        reconciled=True,
        dq_passed=True,
        published=True,
    )


def _approved_capability(
    tmp_path: Path,
    *,
    clock: Callable[[], datetime] | None = None,
) -> tuple[
    LockedTestAuthority,
    ModelRegistry,
    LockedTestCapability,
    IssuedPreregistration,
    LockedTestPreregistrationStore,
]:
    preregistration_store = LockedTestPreregistrationStore(
        tmp_path / "preregistrations", signing_key=b"preregistration-key"
    )
    preregistration = _preregistration()
    issued = preregistration_store.issue(preregistration)
    registry = _shadow_registry(preregistration, tmp_path)
    _, maturity_calendar, watermark = _maturity_evidence(tmp_path)
    identities = _operator_identities(tmp_path)
    authority = LockedTestAuthority(
        tmp_path / "locked-test-authority",
        preregistration_store=preregistration_store,
        registry=registry,
        maturity_calendar=maturity_calendar,
        maturity_evidence_root=tmp_path / "maturity",
        operator_identity_root=tmp_path / "operator-identities",
        signing_key=b"locked-test-authority-key",
        access_signing_key=b"locked-test-access-key",
        clock=clock or (lambda: datetime(2027, 1, 12, tzinfo=UTC)),
    )
    request = authority.request_unlock(
        issued,
        applicant=identities.issue(actor_id="research-principal", role="research-owner"),
        current_date=date(2027, 1, 12),
        watermark=watermark,
    )
    evidence_root = tmp_path / "shadow-evidence"
    capability = authority.approve(
        request,
        approver=identities.issue(actor_id="risk-principal", role="risk-approver"),
        shadow_evidence=_ready_shadow_evidence(evidence_root, preregistration.model_id),
        shadow_evidence_root=evidence_root,
    )
    return authority, registry, capability, issued, preregistration_store


def _locked_test_metrics(*, rank_ic: float = 0.04) -> LockedTestMetrics:
    return LockedTestMetrics(
        mean_rank_ic=rank_ic,
        hac_t_statistic=2.5,
        candidate_delta_rank_ic=0.006,
        valid_test_day_count=6,
        score_health=tuple(
            ScoreHealthEvidence(coverage=0.95, unique_score_count=30, score_std=0.1)
            for _ in range(6)
        ),
    )


def test_preregistration_is_signed_immutable_and_has_no_result_fields(tmp_path: Path) -> None:
    store = LockedTestPreregistrationStore(tmp_path, signing_key=b"preregistration-key")
    preregistration = _preregistration()

    issued = store.issue(preregistration)

    assert issued.preregistration_hash == preregistration.preregistration_hash
    assert store.verify(issued) == preregistration
    assert "result" not in issued.to_json().lower()
    with pytest.raises(FileExistsError, match="immutable"):
        store.issue(preregistration)


def test_maturity_guard_requires_six_subsequent_trading_days_and_clean_label_watermark(
    tmp_path: Path,
) -> None:
    store, calendar, watermark = _maturity_evidence(tmp_path)
    guard = LockedTestMaturityGuard(calendar, evidence_store=store)

    with pytest.raises(ValueError, match="not mature"):
        guard.validate(date(2027, 1, 11), watermark)
    maturity = guard.validate(date(2027, 1, 12), watermark)

    with pytest.raises(ValueError, match="watermark"):
        guard.validate(date(2027, 1, 12), _watermark())  # type: ignore[arg-type]

    assert maturity.last_2026_trading_date == date(2026, 12, 31)
    assert maturity.required_exit_date == date(2027, 1, 11)
    assert maturity.label_watermark_id == watermark.watermark.watermark_id


def test_two_person_unlock_consumes_once_and_exposes_lineage_before_label_access(
    tmp_path: Path,
) -> None:
    preregistration_store = LockedTestPreregistrationStore(
        tmp_path / "preregistrations", signing_key=b"preregistration-key"
    )
    preregistration = _preregistration()
    issued = preregistration_store.issue(preregistration)
    registry = _shadow_registry(preregistration, tmp_path)
    _, maturity_calendar, watermark = _maturity_evidence(tmp_path)
    identities = _operator_identities(tmp_path)
    authority = LockedTestAuthority(
        tmp_path / "locked-test-authority",
        preregistration_store=preregistration_store,
        registry=registry,
        maturity_calendar=maturity_calendar,
        maturity_evidence_root=tmp_path / "maturity",
        operator_identity_root=tmp_path / "operator-identities",
        signing_key=b"locked-test-authority-key",
        access_signing_key=b"locked-test-access-key",
        clock=lambda: datetime(2027, 1, 12, tzinfo=UTC),
    )
    request = authority.request_unlock(
        issued,
        applicant=identities.issue(actor_id="research-principal", role="research-owner"),
        current_date=date(2027, 1, 12),
        watermark=watermark,
    )
    evidence_root = tmp_path / "shadow-evidence"
    shadow_evidence = _ready_shadow_evidence(evidence_root, preregistration.model_id)

    with pytest.raises(ValueError, match="different"):
        authority.approve(
            request,
            approver=identities.issue(actor_id="research-principal", role="risk-approver"),
            shadow_evidence=shadow_evidence,
            shadow_evidence_root=evidence_root,
        )

    capability = authority.approve(
        request,
        approver=identities.issue(actor_id="risk-principal", role="risk-approver"),
        shadow_evidence=shadow_evidence,
        shadow_evidence_root=evidence_root,
    )

    assert registry.get(preregistration.model_id).status is ModelStatus.LOCKED_TEST_APPROVED
    access = authority.consume(capability, issued, consumer="locked-test-evaluator")
    assert access.locked_year == 2026
    assert authority.is_lineage_exposed(preregistration.lineage_id)
    access.consume_once()
    assert (
        tmp_path
        / "locked-test-authority"
        / "access-consumed"
        / f"{capability.capability_id}.json"
    ).is_file()
    with pytest.raises(LockedDatasetError, match="already consumed"):
        access.consume_once()
    with pytest.raises(ValueError, match="consumed"):
        authority.consume(capability, issued, consumer="locked-test-evaluator")
    assert authority.audit_events()[-1].outcome == "DENIED"


def test_failed_capability_persistence_does_not_advance_the_model_lifecycle(
    tmp_path: Path,
) -> None:
    preregistration_store = LockedTestPreregistrationStore(
        tmp_path / "preregistrations", signing_key=b"preregistration-key"
    )
    preregistration = _preregistration()
    issued = preregistration_store.issue(preregistration)
    registry = _shadow_registry(preregistration, tmp_path)
    _, maturity_calendar, watermark = _maturity_evidence(tmp_path)
    identities = _operator_identities(tmp_path)
    authority_root = tmp_path / "locked-test-authority"
    authority = LockedTestAuthority(
        authority_root,
        preregistration_store=preregistration_store,
        registry=registry,
        maturity_calendar=maturity_calendar,
        maturity_evidence_root=tmp_path / "maturity",
        operator_identity_root=tmp_path / "operator-identities",
        signing_key=b"locked-test-authority-key",
        access_signing_key=b"locked-test-access-key",
        clock=lambda: datetime(2027, 1, 12, tzinfo=UTC),
    )
    request = authority.request_unlock(
        issued,
        applicant=identities.issue(actor_id="research-principal", role="research-owner"),
        current_date=date(2027, 1, 12),
        watermark=watermark,
    )
    capability_directory = authority_root / "capabilities"
    capability_directory.parent.mkdir(parents=True, exist_ok=True)
    capability_directory.write_text("blocked", encoding="utf-8")
    evidence_root = tmp_path / "shadow-evidence"

    with pytest.raises(FileExistsError):
        authority.approve(
            request,
            approver=identities.issue(actor_id="risk-principal", role="risk-approver"),
            shadow_evidence=_ready_shadow_evidence(evidence_root, preregistration.model_id),
            shadow_evidence_root=evidence_root,
        )

    assert registry.get(preregistration.model_id).status is ModelStatus.SHADOW


def test_evaluator_discloses_only_final_aggregate_metrics_after_a_successful_one_shot_run(
    tmp_path: Path,
) -> None:
    authority, registry, capability, issued, _ = _approved_capability(tmp_path)
    evaluator = LockedTestEvaluator(
        tmp_path / "locked-test-results",
        authority=authority,
        registry=registry,
        signing_key=b"locked-test-results-key",
        artifact_bundle_hash="a" * 64,
    )

    result = evaluator.evaluate(
        capability,
        issued,
        actor="locked-test-evaluator",
        metrics_reader=lambda _access: _locked_test_metrics(),
    )

    assert result.passed
    assert result.aggregate.mean_rank_ic == 0.04
    assert result.aggregate.score_health_pass_ratio == 1.0
    assert registry.get(_MODEL_ID).status is ModelStatus.LOCKED_TEST_PASSED


def test_evaluator_suppresses_callback_stdout_until_the_caller_receives_the_final_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    authority, registry, capability, issued, _ = _approved_capability(tmp_path)
    evaluator = LockedTestEvaluator(
        tmp_path / "locked-test-results",
        authority=authority,
        registry=registry,
        signing_key=b"locked-test-results-key",
        artifact_bundle_hash="a" * 64,
    )

    def noisy_metrics_reader(_access: object) -> LockedTestMetrics:
        print("intermediate label-derived output")
        return _locked_test_metrics()

    result = evaluator.evaluate(
        capability,
        issued,
        actor="locked-test-evaluator",
        metrics_reader=noisy_metrics_reader,
    )

    assert result.passed
    assert capsys.readouterr().out == ""


def test_evaluator_crash_consumes_the_capability_and_records_a_failed_locked_test(
    tmp_path: Path,
) -> None:
    authority, registry, capability, issued, _ = _approved_capability(tmp_path)
    evaluator = LockedTestEvaluator(
        tmp_path / "locked-test-results",
        authority=authority,
        registry=registry,
        signing_key=b"locked-test-results-key",
        artifact_bundle_hash="a" * 64,
    )

    def crash(_access: object) -> LockedTestMetrics:
        raise RuntimeError("reader crashed before aggregate output")

    with pytest.raises(RuntimeError, match="reader crashed"):
        evaluator.evaluate(
            capability,
            issued,
            actor="locked-test-evaluator",
            metrics_reader=crash,
        )

    assert registry.get(_MODEL_ID).status is ModelStatus.LOCKED_TEST_FAILED
    with pytest.raises(ValueError, match="consumed"):
        authority.consume(capability, issued, consumer="locked-test-evaluator")


def test_new_evaluator_recovers_a_consumed_capability_without_a_terminal_result(
    tmp_path: Path,
) -> None:
    authority, registry, capability, issued, _ = _approved_capability(tmp_path)
    result_root = tmp_path / "locked-test-results"
    authority.consume(
        capability,
        issued,
        consumer="locked-test-evaluator",
        result_root=result_root,
    )

    LockedTestEvaluator(
        result_root,
        authority=authority,
        registry=registry,
        signing_key=b"locked-test-results-key",
        artifact_bundle_hash="a" * 64,
    )

    assert registry.get(_MODEL_ID).status is ModelStatus.LOCKED_TEST_FAILED


def test_unlock_rejects_an_unmature_year_before_issuing_a_request(tmp_path: Path) -> None:
    preregistration_store = LockedTestPreregistrationStore(
        tmp_path / "preregistrations", signing_key=b"preregistration-key"
    )
    preregistration = _preregistration()
    issued = preregistration_store.issue(preregistration)
    _, maturity_calendar, watermark = _maturity_evidence(tmp_path)
    identities = _operator_identities(tmp_path)
    authority = LockedTestAuthority(
        tmp_path / "locked-test-authority",
        preregistration_store=preregistration_store,
        registry=_shadow_registry(preregistration, tmp_path),
        maturity_calendar=maturity_calendar,
        maturity_evidence_root=tmp_path / "maturity",
        operator_identity_root=tmp_path / "operator-identities",
        signing_key=b"locked-test-authority-key",
        access_signing_key=b"locked-test-access-key",
        clock=lambda: datetime(2027, 1, 11, tzinfo=UTC),
    )

    with pytest.raises(ValueError, match="not mature"):
        authority.request_unlock(
            issued,
            applicant=identities.issue(actor_id="research-principal", role="research-owner"),
            current_date=date(2027, 1, 11),
            watermark=watermark,
        )

    assert authority.audit_events()[-1].outcome == "DENIED"


def test_expired_capability_cannot_mark_lineage_exposed_or_unlock_labels(tmp_path: Path) -> None:
    now = [datetime(2027, 1, 12, 9, tzinfo=UTC)]
    authority, _, capability, issued, _ = _approved_capability(tmp_path, clock=lambda: now[0])
    now[0] += timedelta(hours=1)

    with pytest.raises(ValueError, match="expired"):
        authority.consume(capability, issued, consumer="locked-test-evaluator")

    assert not authority.is_lineage_exposed(_preregistration().lineage_id)
    assert authority.audit_events()[-1].outcome == "DENIED"


def test_changed_preregistration_hash_cannot_consume_an_approved_capability(tmp_path: Path) -> None:
    authority, _, capability, _, preregistration_store = _approved_capability(tmp_path)
    changed = replace(_preregistration(), evaluation_code_hash="9" * 64)
    changed_issued = preregistration_store.issue(changed)

    with pytest.raises(ValueError, match="preregistration hash"):
        authority.consume(capability, changed_issued, consumer="locked-test-evaluator")

    assert not authority.is_lineage_exposed(changed.lineage_id)


def test_threshold_failure_reaches_locked_test_failed_without_a_second_attempt(
    tmp_path: Path,
) -> None:
    authority, registry, capability, issued, _ = _approved_capability(tmp_path)
    evaluator = LockedTestEvaluator(
        tmp_path / "locked-test-results",
        authority=authority,
        registry=registry,
        signing_key=b"locked-test-results-key",
        artifact_bundle_hash="a" * 64,
    )

    result = evaluator.evaluate(
        capability,
        issued,
        actor="locked-test-evaluator",
        metrics_reader=lambda _access: _locked_test_metrics(rank_ic=0.029),
    )

    assert not result.passed
    assert registry.get(_MODEL_ID).status is ModelStatus.LOCKED_TEST_FAILED
    with pytest.raises(ValueError, match="consumed"):
        authority.consume(capability, issued, consumer="locked-test-evaluator")
    approvals = ProductionDeploymentApprovalStore(
        tmp_path / "production-approvals",
        registry=registry,
        result_root=tmp_path / "locked-test-results",
        result_verification_key=b"locked-test-results-key",
        signing_key=b"production-approval-key",
    )
    with pytest.raises(ValueError, match="passed"):
        approvals.issue(
            result,
            actor="production-risk-approver",
        )


def test_exposed_lineage_cannot_reenter_a_2026_locked_test_as_a_descendant(tmp_path: Path) -> None:
    authority, registry, capability, issued, preregistration_store = _approved_capability(tmp_path)
    authority.consume(capability, issued, consumer="locked-test-evaluator")
    descendant = _preregistration(model_id="sha256:" + "b" * 64, lineage_id=_MODEL_ID)
    descendant_issued = preregistration_store.issue(descendant)
    _enter_shadow(
        registry,
        descendant,
        tmp_path / "descendant",
        parent_model_id=_MODEL_ID,
    )
    _, _, watermark = _maturity_evidence(tmp_path / "descendant-maturity")
    identities = _operator_identities(tmp_path)

    with pytest.raises(ValueError, match="LOCKED_EXPOSED"):
        authority.request_unlock(
            descendant_issued,
            applicant=identities.issue(actor_id="research-principal", role="research-owner"),
            current_date=date(2027, 1, 12),
            watermark=watermark,
        )


def test_independent_production_approval_requires_a_verified_locked_test_pass(
    tmp_path: Path,
) -> None:
    authority, registry, capability, issued, _ = _approved_capability(tmp_path)
    result_root = tmp_path / "locked-test-results"
    evaluator = LockedTestEvaluator(
        result_root,
        authority=authority,
        registry=registry,
        signing_key=b"locked-test-results-key",
        artifact_bundle_hash="a" * 64,
    )
    result = evaluator.evaluate(
        capability,
        issued,
        actor="locked-test-evaluator",
        metrics_reader=lambda _access: _locked_test_metrics(),
    )
    approvals = ProductionDeploymentApprovalStore(
        tmp_path / "production-approvals",
        registry=registry,
        result_root=result_root,
        result_verification_key=b"locked-test-results-key",
        signing_key=b"production-approval-key",
    )

    with pytest.raises(ValueError, match="independent"):
        approvals.issue(
            result,
            actor="locked-test-evaluator",
        )
    approval = approvals.issue(
        result,
        actor="production-risk-approver",
    )

    assert approval.model_id == _MODEL_ID
    assert registry.get(_MODEL_ID).status is ModelStatus.PRODUCTION


def test_production_pointer_accepts_only_the_signed_approval_for_its_exact_artifact(
    tmp_path: Path,
) -> None:
    authority, registry, capability, issued, _ = _approved_capability(tmp_path)
    result_root = tmp_path / "locked-test-results"
    evaluator = LockedTestEvaluator(
        result_root,
        authority=authority,
        registry=registry,
        signing_key=b"locked-test-results-key",
        artifact_bundle_hash="a" * 64,
    )
    result = evaluator.evaluate(
        capability,
        issued,
        actor="locked-test-evaluator",
        metrics_reader=lambda _access: _locked_test_metrics(),
    )
    approval_root = tmp_path / "production-approvals"
    approval = ProductionDeploymentApprovalStore(
        approval_root,
        registry=registry,
        result_root=result_root,
        result_verification_key=b"locked-test-results-key",
        signing_key=b"production-approval-key",
    ).issue(
        result,
        actor="production-risk-approver",
    )
    mismatched_store = ActiveModelPointerStore(
        eligibility=PointerEligibilityGuard(registry, bundle_digest=lambda _model_id: "b" * 64),
        clock=lambda: datetime(2027, 1, 12, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="artifact"):
        mismatched_store.compare_and_swap(
            consumer_id="paper-sim",
            environment="production",
            expected_current=None,
            model_id=_MODEL_ID,
            model_version=1,
            actor="production-operator",
            reason="mismatched artifact release",
            evidence=PointerSwitchEvidence(
                production_approval=approval,
                production_approval_root=approval_root,
            ),
            run_id="00000000-0000-0000-0000-000000000001",
        )
    store = ActiveModelPointerStore(
        eligibility=PointerEligibilityGuard(registry, bundle_digest=lambda _model_id: "a" * 64),
        clock=lambda: datetime(2027, 1, 12, tzinfo=UTC),
    )

    update = store.compare_and_swap(
        consumer_id="paper-sim",
        environment="production",
        expected_current=None,
        model_id=_MODEL_ID,
        model_version=1,
        actor="production-operator",
        reason="approved locked-test release",
        evidence=PointerSwitchEvidence(
            production_approval=approval,
            production_approval_root=approval_root,
        ),
        run_id="00000000-0000-0000-0000-000000000001",
    )

    assert update is not None
    assert update.event.production_approval_id == approval.approval_id


# --- the frozen inputs and the disclosed aggregate ---------------------------
#
# The workflow above is well tested: signing, one-shot consumption, maturity, the
# two-person unlock, crash handling, lineage exposure, production approval. What was
# not tested is the validation of the inputs those workflows carry, because every
# workflow test feeds well-formed data -- and deleting a validation clause cannot break
# a test that never exercised the clause. Eleven of twelve mutations in this slice
# survived. Each test below is one group of those clauses.


def test_preregistration_rejects_a_weaker_gate_list_and_the_other_input_faults() -> None:
    """The preregistration is the frozen definition of the attempt, so its inputs are checked.

    A shorter gate list defines a weaker locked test than the one that was approved; equal
    applicant and approver roles remove the two-person rule at the point where it is
    cheapest to enforce; fewer than three seed hashes, or three that are not distinct,
    unbinds the attempt from a specific three-seed artifact.
    """
    base = _preregistration()
    cases = (
        (lambda: replace(base, required_gates=("G1", "G2", "G3", "G4", "G5")), "exactly G1-G6"),
        (lambda: replace(base, approver_role="research-owner"), "roles must be distinct"),
        (lambda: replace(base, seed_artifact_hashes=("a" * 64, "b" * 64)), "seed_artifact_hashes"),
        (lambda: replace(base, seed_artifact_hashes=("a" * 64,) * 3), "seed_artifact_hashes"),
        (lambda: replace(base, model_id="model-1"), "model_id"),
        (lambda: replace(base, universe_id="   "), "universe_id"),
        (lambda: replace(base, command=()), "command"),
    )

    for build, match in cases:
        with pytest.raises(ValueError, match=match):
            build()


def test_the_preregistration_hash_binds_every_frozen_input() -> None:
    """Every frozen input is inside the hash, and each is distinguishable from the others.

    Measured before this test existed: dropping ``evaluation_code_hash`` from ``payload()``
    survived the whole workflow suite. The one test that did fail, failed with
    ``FileExistsError`` -- the preregistration store is addressed by this hash, so a
    preregistration differing only in its evaluation code became the SAME document and
    collided on write. That collision is the defect showing up, but it is not an assertion
    about it, and it would read as a test-fixture problem to whoever triaged it. This
    asserts the property directly for all fourteen fields that can vary while staying
    valid, and also that no two of them hash to the same document.
    """
    base = _preregistration()
    variants: dict[str, object] = {
        "model_id": "sha256:" + "b" * 64,
        "lineage_id": "sha256:" + "c" * 64,
        "seed_artifact_hashes": ("a" * 64, "b" * 64, "9" * 64),
        "artifact_bundle_hash": "9" * 64,
        "final_train_manifest_hash": "9" * 64,
        "final_validation_manifest_hash": "9" * 64,
        "factor_set_id": "factor-set-final-v2",
        "preprocessor_hash": "9" * 64,
        "universe_id": "cn-liquid-a-2026-v2",
        "label_definition_hash": "9" * 64,
        "evaluation_code_hash": "9" * 64,
        "command": ("qr-v2", "registry", "locked-test", "evaluate", "--strict"),
        "applicant_role": "research-owner-2",
        "approver_role": "risk-approver-2",
    }
    # required_gates is the fifteenth payload field and cannot vary: the clause above
    # requires it to be exactly G1-G6, so there is no valid alternative to hash.
    assert set(base.payload()) == set(variants) | {"required_gates"}

    hashes = {
        field: replace(base, **{field: value}).preregistration_hash
        for field, value in variants.items()
    }

    assert base.preregistration_hash not in hashes.values()
    assert len(set(hashes.values())) == len(variants), "two fields hash to the same document"


def test_locked_test_metrics_refuse_inputs_that_cannot_be_aggregated() -> None:
    """The metrics dataclass is the only checkpoint between the isolated evaluator and the verdict.

    ``aggregate()`` divides by ``valid_test_day_count``, so a zero there is a division by
    zero waiting to happen rather than a number; a score-health tuple whose length
    disagrees with the day count makes the disclosed ratio meaningless in the same way
    G6's numerator and denominator would; and a non-finite mean or t statistic would be
    carried into the signed result.
    """
    base = _locked_test_metrics()
    # Guard the guard: the helper's metrics are valid and aggregate to a perfect ratio.
    assert base.aggregate().score_health_pass_ratio == pytest.approx(1.0)

    cases = (
        (lambda: replace(base, mean_rank_ic=float("nan")), "mean_rank_ic"),
        (lambda: replace(base, hac_t_statistic=float("inf")), "hac_t_statistic"),
        (lambda: replace(base, candidate_delta_rank_ic=float("nan")), "candidate_delta_rank_ic"),
        (lambda: replace(base, valid_test_day_count=0), "valid_test_day_count"),
        (lambda: replace(base, valid_test_day_count=5), "score_health is incomplete"),
        (lambda: replace(base, score_health=base.score_health[:5]), "score_health is incomplete"),
    )

    for build, match in cases:
        with pytest.raises(ValueError, match=match):
            build()


def test_the_disclosed_score_health_ratio_is_computed_and_carried_into_the_payload() -> None:
    """``aggregate()`` yields the only score-health number allowed out of the isolation.

    Both mutations here survived the workflow suite. Hardcoding the ratio to 1.0 reports
    every locked test as perfectly score-healthy whatever its days looked like, and
    dropping the field from ``payload()`` removes it from the signed result entirely --
    in both cases the workflow still runs, still signs, still verifies, and still promotes.
    """
    healthy = ScoreHealthEvidence(coverage=0.95, unique_score_count=30, score_std=0.1)
    collapsed = ScoreHealthEvidence(coverage=0.95, unique_score_count=1, score_std=0.1)
    metrics = LockedTestMetrics(
        mean_rank_ic=0.04,
        hac_t_statistic=2.5,
        candidate_delta_rank_ic=0.006,
        valid_test_day_count=4,
        score_health=(healthy, healthy, collapsed, collapsed),
    )

    aggregate = metrics.aggregate()

    assert aggregate.score_health_pass_ratio == pytest.approx(0.5)
    assert aggregate.valid_test_day_count == 4
    assert aggregate.mean_rank_ic == pytest.approx(0.04)
    assert aggregate.payload()["score_health_pass_ratio"] == pytest.approx(0.5)
    assert set(aggregate.payload()) == {
        "candidate_delta_rank_ic",
        "hac_t_statistic",
        "mean_rank_ic",
        "score_health_pass_ratio",
        "valid_test_day_count",
    }


def test_canonical_json_sorts_keys_so_a_signature_does_not_depend_on_key_order() -> None:
    """Every signed document here is hashed through ``_canonical_json``, and sorting matters.

    Measured, and it is not the no-op it looks like: five of this module's six ``payload()``
    literals already list their keys alphabetically, but ``LockedTestResult.payload()`` does
    not -- it writes ``evaluator_actor`` before ``evaluated_at``, the reverse of sorted
    order, so the canonical form of the signed result depends on this call sorting. It
    survived anyway because a test that signs and verifies inside one process changes both
    sides together; the exposure is cross-build, where a result signed by one version stops
    verifying under another.
    """
    assert _canonical_json({"b": 1, "a": 2}) == '{"a":2,"b":1}'
    assert _canonical_json({"b": 1, "a": 2}) == _canonical_json({"a": 2, "b": 1})
    # The pair that is out of order in LockedTestResult.payload().
    assert _canonical_json({"evaluator_actor": "e", "evaluated_at": "t"}) == (
        '{"evaluated_at":"t","evaluator_actor":"e"}'
    )
