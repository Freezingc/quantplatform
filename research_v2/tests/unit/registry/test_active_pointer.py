from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest

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
from research_v2.registry.model_registry import ModelRecord, ModelRegistry, ModelStatus

_A = "sha256:" + "a" * 64
_B = "sha256:" + "b" * 64
_C = "sha256:" + "c" * 64
_APPROVAL_KEY = "pointer-approval-key"
_RUN_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture(autouse=True)
def _trusted_shadow_approval_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QRV2_SHADOW_APPROVAL_HMAC_KEY", _APPROVAL_KEY)
    monkeypatch.setenv("QRV2_SHADOW_EVIDENCE_HMAC_KEY", "pointer-evidence-key")
    monkeypatch.setenv("QRV2_SHADOW_DEPLOYER_ID", "shadow-controller")


def _evidence() -> GateEvidence:
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


def _registry() -> ModelRegistry:
    registry = ModelRegistry()
    for model_id in (_A, _B, _C):
        registry.register_draft(model_id, actor="trainer")
        registry.promote_walk_forward(model_id, actor="reviewer", evidence=_evidence())
    return registry


def _approval(
    registry: ModelRegistry,
    root: Path,
    model_id: str,
    artifact_digest: str,
) -> PointerSwitchEvidence:
    approval = ShadowDeploymentApprovalStore(root, signing_key=_APPROVAL_KEY.encode()).issue(
        model_id=model_id,
        actor="risk-approver",
        walk_forward_evidence_hash=registry.history(model_id)[-1].evidence_hashes[0],
        artifact_bundle_hash=artifact_digest,
        inference_contract_hash="d" * 64,
        v2test_drill_hash="e" * 64,
        rollback_plan_hash="f" * 64,
    )
    return PointerSwitchEvidence(shadow_approval=approval, shadow_approval_root=root)


def _store(registry: ModelRegistry, digests: dict[str, str]) -> ActiveModelPointerStore:
    return ActiveModelPointerStore(
        eligibility=PointerEligibilityGuard(registry, bundle_digest=digests.__getitem__),
        clock=lambda: datetime(2026, 8, 28, 9, 30, tzinfo=UTC),
    )


def _ready_shadow_evidence(root: Path, model_id: str, *, max_psi_errors: int) -> ShadowEvidence:
    dates = tuple(date(2026, 1, 2) + timedelta(days=index) for index in range(60))
    qualification = ShadowQualification(
        model_id=model_id,
        trading_calendar_id="a" * 64,
        valid_day_count=60,
        status=ShadowQualificationStatus.READY_FOR_REVIEW,
        max_consecutive_psi_errors=max_psi_errors,
        valid_signal_dates=dates,
        invalid_signal_dates=(),
        valid_report_ids=tuple(f"{index:064x}" for index in range(60)),
        valid_runtime_day_ids=tuple(f"{index + 100:064x}" for index in range(60)),
        invalid_runtime_day_ids=(),
    )
    return ShadowEvidenceStore(root, signing_key=b"pointer-evidence-key").issue(qualification)


def test_switch_uses_full_pointer_cas_and_records_an_append_only_audit_event(
    tmp_path: Path,
) -> None:
    registry = _registry()
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    store = _store(registry, digests)
    first = store.compare_and_swap(
        consumer_id="paper-sim",
        environment="shadow",
        expected_current=None,
        model_id=_A,
        model_version=1,
        actor="operator-a",
        reason="initial shadow deployment",
        evidence=_approval(registry, tmp_path / "approvals", _A, digests[_A]),
        run_id=_RUN_ID,
    )

    assert first is not None
    assert first.pointer.model_id == _A
    assert first.pointer.pointer_version == 1
    assert first.event.old_pointer is None
    assert first.event.new_pointer == first.pointer
    assert first.event.actor == "operator-a"
    assert first.event.reason == "initial shadow deployment"
    assert first.event.effective_at == datetime(2026, 8, 28, 9, 30, tzinfo=UTC)
    assert first.event.run_id == _RUN_ID
    assert first.event.evidence_ids == (first.event.shadow_approval_id,)

    stale = store.compare_and_swap(
        consumer_id="paper-sim",
        environment="shadow",
        expected_current=None,
        model_id=_B,
        model_version=1,
        actor="operator-b",
        reason="stale overwrite",
        evidence=_approval(registry, tmp_path / "approvals", _B, digests[_B]),
        run_id=_RUN_ID,
    )

    assert stale is None
    assert store.get("paper-sim", "shadow") == first.pointer
    assert store.history("paper-sim", "shadow") == (first.event,)


def test_concurrent_switches_allow_exactly_one_winner_and_one_complete_pointer(
    tmp_path: Path,
) -> None:
    registry = _registry()
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    store = _store(registry, digests)
    first = store.compare_and_swap(
        consumer_id="paper-sim",
        environment="shadow",
        expected_current=None,
        model_id=_A,
        model_version=1,
        actor="operator-a",
        reason="initial shadow deployment",
        evidence=_approval(registry, tmp_path / "approvals", _A, digests[_A]),
        run_id=_RUN_ID,
    )
    assert first is not None
    barrier = Barrier(2)

    def switch(model_id: str, actor: str) -> object:
        barrier.wait()
        return store.compare_and_swap(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=first.pointer,
            model_id=model_id,
            model_version=1,
            actor=actor,
            reason="concurrent change",
            evidence=_approval(registry, tmp_path / "approvals", model_id, digests[model_id]),
            run_id=_RUN_ID,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        candidates = ((_B, "operator-b"), (_C, "operator-c"))
        updates = list(executor.map(lambda item: switch(*item), candidates))

    winner = [update for update in updates if update is not None]
    assert len(winner) == 1
    current = store.get("paper-sim", "shadow")
    assert current is not None
    assert current.model_id in {_B, _C}
    assert current.pointer_version == 2
    assert len(store.history("paper-sim", "shadow")) == 2


def test_rollback_is_a_new_cas_event_and_rejects_a_damaged_previous_artifact(
    tmp_path: Path,
) -> None:
    registry = _registry()
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    store = _store(registry, digests)
    first = store.compare_and_swap(
        consumer_id="paper-sim",
        environment="shadow",
        expected_current=None,
        model_id=_A,
        model_version=1,
        actor="operator-a",
        reason="initial shadow deployment",
        evidence=_approval(registry, tmp_path / "approvals", _A, digests[_A]),
        run_id=_RUN_ID,
    )
    assert first is not None
    second = store.compare_and_swap(
        consumer_id="paper-sim",
        environment="shadow",
        expected_current=first.pointer,
        model_id=_B,
        model_version=1,
        actor="operator-b",
        reason="candidate B switch",
        evidence=_approval(registry, tmp_path / "approvals", _B, digests[_B]),
        run_id=_RUN_ID,
    )
    assert second is not None

    del digests[_A]
    with pytest.raises(ValueError, match="artifact"):
        store.rollback(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=second.pointer,
            event_id=second.event.event_id,
            actor="incident-operator",
            reason="B score-health failure",
            evidence=_approval(registry, tmp_path / "approvals", _A, "1" * 64),
            run_id=_RUN_ID,
        )

    assert store.get("paper-sim", "shadow") == second.pointer

    digests[_A] = "1" * 64
    rolled_back = store.rollback(
        consumer_id="paper-sim",
        environment="shadow",
        expected_current=second.pointer,
        event_id=second.event.event_id,
        actor="incident-operator",
        reason="B score-health failure",
        evidence=_approval(registry, tmp_path / "approvals", _A, digests[_A]),
        run_id=_RUN_ID,
    )

    assert rolled_back is not None
    assert rolled_back.pointer.model_id == _A
    assert rolled_back.pointer.pointer_version == 3
    assert rolled_back.event.kind == "ROLLBACK"
    assert rolled_back.event.reverts_event_id == second.event.event_id
    assert store.history("paper-sim", "shadow") == (first.event, second.event, rolled_back.event)


def test_shadow_switch_fails_closed_without_signed_approval_or_matching_artifact(
    tmp_path: Path,
) -> None:
    registry = _registry()
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    store = _store(registry, digests)

    with pytest.raises(ValueError, match="approval"):
        store.compare_and_swap(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=None,
            model_id=_A,
            model_version=1,
            actor="operator-a",
            reason="missing approval",
            evidence=PointerSwitchEvidence(),
            run_id=_RUN_ID,
        )
    with pytest.raises(ValueError, match="artifact"):
        store.compare_and_swap(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=None,
            model_id=_A,
            model_version=1,
            actor="operator-a",
            reason="mismatched artifact",
            evidence=_approval(registry, tmp_path / "approvals", _A, "f" * 64),
            run_id=_RUN_ID,
        )


def test_shadow_guard_accepts_ready_evidence_below_the_drift_review_threshold(
    tmp_path: Path,
) -> None:
    registry = _registry()
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    approval_root = tmp_path / "approvals"
    approval_evidence = _approval(registry, approval_root, _A, digests[_A])
    assert approval_evidence.shadow_approval is not None
    registry.enter_shadow(
        _A,
        approval=approval_evidence.shadow_approval,
        approval_root=approval_root,
    )
    shadow_root = tmp_path / "shadow-evidence"
    evidence = _ready_shadow_evidence(shadow_root, _A, max_psi_errors=1)
    store = _store(registry, digests)

    update = store.compare_and_swap(
        consumer_id="paper-sim",
        environment="shadow",
        expected_current=None,
        model_id=_A,
        model_version=1,
        actor="operator-a",
        reason="qualified shadow switch",
        evidence=PointerSwitchEvidence(
            shadow_approval=approval_evidence.shadow_approval,
            shadow_approval_root=approval_root,
            shadow_evidence=evidence,
            shadow_evidence_root=shadow_root,
        ),
        run_id=_RUN_ID,
    )

    assert update is not None
    assert update.event.shadow_evidence_id == evidence.evidence_id


def test_shadow_model_requires_signed_readiness_evidence_before_pointer_switch(
    tmp_path: Path,
) -> None:
    registry = _registry()
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    approval_root = tmp_path / "approvals"
    approval_evidence = _approval(registry, approval_root, _A, digests[_A])
    assert approval_evidence.shadow_approval is not None
    registry.enter_shadow(
        _A,
        approval=approval_evidence.shadow_approval,
        approval_root=approval_root,
    )
    store = _store(registry, digests)

    with pytest.raises(ValueError, match="readiness evidence"):
        store.compare_and_swap(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=None,
            model_id=_A,
            model_version=1,
            actor="operator-a",
            reason="shadow switch without readiness evidence",
            evidence=approval_evidence,
            run_id=_RUN_ID,
        )


def test_production_pointer_rejects_a_model_below_production_status(tmp_path: Path) -> None:
    registry = _registry()
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    store = _store(registry, digests)

    with pytest.raises(ValueError, match="PRODUCTION"):
        store.compare_and_swap(
            consumer_id="paper-sim",
            environment="production",
            expected_current=None,
            model_id=_A,
            model_version=1,
            actor="production-operator",
            reason="invalid production switch",
            evidence=PointerSwitchEvidence(evidence_ids=("d" * 64,)),
            run_id=_RUN_ID,
        )


def test_production_pointer_rejects_unverified_caller_supplied_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry()
    monkeypatch.setattr(
        registry,
        "get",
        lambda model_id: ModelRecord(model_id, ModelStatus.PRODUCTION),
    )
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    store = _store(registry, digests)

    with pytest.raises(ValueError, match="verified production approval"):
        store.compare_and_swap(
            consumer_id="paper-sim",
            environment="production",
            expected_current=None,
            model_id=_A,
            model_version=1,
            actor="production-operator",
            reason="unverified production switch",
            evidence=PointerSwitchEvidence(evidence_ids=("d" * 64,)),
            run_id=_RUN_ID,
        )


def test_pointer_rejects_an_unsigned_model_version_above_the_artifact_contract(
    tmp_path: Path,
) -> None:
    registry = _registry()
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    store = _store(registry, digests)

    with pytest.raises(ValueError, match="model_version"):
        store.compare_and_swap(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=None,
            model_id=_A,
            model_version=2,
            actor="operator-a",
            reason="unbound version switch",
            evidence=_approval(registry, tmp_path / "approvals", _A, digests[_A]),
            run_id=_RUN_ID,
        )


def test_unconfigured_v1_consumer_has_no_pointer_or_switch_side_effect(tmp_path: Path) -> None:
    registry = _registry()
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    store = _store(registry, digests)

    update = store.compare_and_swap(
        consumer_id="paper-sim",
        environment="shadow",
        expected_current=None,
        model_id=_A,
        model_version=1,
        actor="operator-a",
        reason="initial shadow deployment",
        evidence=_approval(registry, tmp_path / "approvals", _A, digests[_A]),
        run_id=_RUN_ID,
    )

    assert update is not None
    assert store.get("legacy-v1-consumer", "production") is None
    assert store.history("legacy-v1-consumer", "production") == ()
