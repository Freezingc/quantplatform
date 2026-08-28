from __future__ import annotations

import os
from datetime import UTC, datetime
from pathlib import Path

import pytest

from research_v2.evaluation.gates import GateEvidence, ScoreHealthEvidence
from research_v2.monitoring.shadow import ShadowDeploymentApprovalStore
from research_v2.registry.active_pointer import (
    ActiveModelPointerStore,
    PointerEligibilityGuard,
    PointerSwitchEvidence,
)
from research_v2.registry.model_registry import ModelRegistry

pytestmark = pytest.mark.integration

_A = "sha256:" + "a" * 64
_B = "sha256:" + "b" * 64
_C = "sha256:" + "c" * 64
_RUN_ID = "00000000-0000-0000-0000-000000000001"


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
    approval = ShadowDeploymentApprovalStore(root, signing_key=b"integration-approval-key").issue(
        model_id=model_id,
        actor="risk-approver",
        walk_forward_evidence_hash=registry.history(model_id)[-1].evidence_hashes[0],
        artifact_bundle_hash=artifact_digest,
        inference_contract_hash="d" * 64,
        v2test_drill_hash="e" * 64,
        rollback_plan_hash="f" * 64,
    )
    return PointerSwitchEvidence(shadow_approval=approval, shadow_approval_root=root)


@pytest.mark.skipif(
    os.environ.get("QRV2_INTEGRATION") != "1",
    reason="set QRV2_INTEGRATION=1 to run pointer rollback integration tests",
)
def test_pointer_switch_conflict_rollback_and_artifact_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QRV2_SHADOW_APPROVAL_HMAC_KEY", "integration-approval-key")
    registry = _registry()
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    store = ActiveModelPointerStore(
        eligibility=PointerEligibilityGuard(registry, bundle_digest=digests.__getitem__),
        clock=lambda: datetime(2026, 8, 28, 9, 30, tzinfo=UTC),
    )
    approvals = tmp_path / "approvals"
    first = store.compare_and_swap(
        consumer_id="paper-sim",
        environment="shadow",
        expected_current=None,
        model_id=_A,
        model_version=1,
        actor="operator-a",
        reason="initial shadow deployment",
        evidence=_approval(registry, approvals, _A, digests[_A]),
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
        evidence=_approval(registry, approvals, _B, digests[_B]),
        run_id=_RUN_ID,
    )
    assert second is not None
    assert (
        store.compare_and_swap(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=first.pointer,
            model_id=_C,
            model_version=1,
            actor="operator-c",
            reason="stale candidate C switch",
            evidence=_approval(registry, approvals, _C, digests[_C]),
            run_id=_RUN_ID,
        )
        is None
    )
    del digests[_A]
    with pytest.raises(ValueError, match="artifact"):
        store.rollback(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=second.pointer,
            event_id=second.event.event_id,
            actor="incident-operator",
            reason="B rollback with damaged A artifact",
            evidence=_approval(registry, approvals, _A, "1" * 64),
            run_id=_RUN_ID,
        )
    assert store.get("paper-sim", "shadow") == second.pointer
    digests[_A] = "1" * 64
    rollback = store.rollback(
        consumer_id="paper-sim",
        environment="shadow",
        expected_current=second.pointer,
        event_id=second.event.event_id,
        actor="incident-operator",
        reason="B score-health failure",
        evidence=_approval(registry, approvals, _A, digests[_A]),
        run_id=_RUN_ID,
    )
    assert rollback is not None
    assert rollback.pointer.model_id == _A
    assert rollback.event.reverts_event_id == second.event.event_id
    assert store.get("legacy-v1-consumer", "production") is None
