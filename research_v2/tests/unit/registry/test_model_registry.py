from __future__ import annotations

import pytest

from research_v2.evaluation.gates import GateEvidence, ScoreHealthEvidence
from research_v2.monitoring.shadow import ShadowDeploymentApprovalStore
from research_v2.registry.model_registry import ModelRegistry, ModelStatus

MODEL_ID = "sha256:" + "a" * 64
CHILD_MODEL_ID = "sha256:" + "b" * 64


def _evidence(*, valid_score_days: int = 6) -> GateEvidence:
    return GateEvidence(
        mean_rank_ic=0.04,
        hac_t_statistic=2.5,
        yearly_mean_rank_ic={year: 0.04 for year in range(2020, 2026)},
        candidate_delta_rank_ic=0.006,
        valid_test_day_count=6,
        score_health=tuple(
            ScoreHealthEvidence(coverage=0.95, unique_score_count=30, score_std=0.1)
            for _ in range(valid_score_days)
        ),
    )


def test_registry_requires_all_gates_before_walk_forward_promotion() -> None:
    registry = ModelRegistry()
    registry.register_draft(MODEL_ID, actor="trainer")

    promoted = registry.promote_walk_forward(MODEL_ID, actor="reviewer", evidence=_evidence())

    assert promoted.status is ModelStatus.WALK_FORWARD_PASSED
    history = registry.history(MODEL_ID)
    assert len(history) == 2
    assert history[-1].reason == "all G1-G6 walk-forward gates passed"
    assert len(history[-1].evidence_hashes) == 1
    with pytest.raises(ValueError, match="cannot transition"):
        registry.transition(
            MODEL_ID,
            ModelStatus.PRODUCTION,
            actor="reviewer",
            evidence_hashes=("b" * 64,),
            reason="skip shadow",
        )


def test_registry_rejects_failed_gate_evidence_and_allows_retirement() -> None:
    registry = ModelRegistry()
    registry.register_draft(MODEL_ID, actor="trainer")
    with pytest.raises(ValueError, match="G6"):
        registry.promote_walk_forward(
            MODEL_ID,
            actor="reviewer",
            evidence=_evidence(valid_score_days=5),
        )

    retired = registry.transition(
        MODEL_ID,
        ModelStatus.RETIRED,
        actor="reviewer",
        evidence_hashes=(),
        reason="candidate withdrawn",
    )
    assert retired.status is ModelStatus.RETIRED


def test_derived_models_inherit_the_immutable_registered_parent_lineage() -> None:
    registry = ModelRegistry()
    root = registry.register_draft(MODEL_ID, actor="trainer")
    child = registry.register_draft(CHILD_MODEL_ID, actor="trainer", parent_model_id=MODEL_ID)

    assert root.lineage_id == MODEL_ID
    assert child.lineage_id == MODEL_ID
    assert child.parent_model_id == MODEL_ID
    with pytest.raises(ValueError, match="parent"):
        registry.register_draft(
            "sha256:" + "c" * 64,
            actor="trainer",
            parent_model_id=MODEL_ID,
            lineage_id="fake-lineage",
        )


def test_registry_reserves_locked_test_lifecycle_for_its_dedicated_workflow(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QRV2_SHADOW_APPROVAL_HMAC_KEY", "test-shadow-approval-key")
    monkeypatch.setenv("QRV2_SHADOW_DEPLOYER_ID", "shadow-controller")
    registry = ModelRegistry()
    registry.register_draft(MODEL_ID, actor="trainer")
    registry.promote_walk_forward(MODEL_ID, actor="reviewer", evidence=_evidence())
    approvals = ShadowDeploymentApprovalStore(
        tmp_path / "approvals",  # type: ignore[operator]
        signing_key=b"test-shadow-approval-key",
    )
    approval = approvals.issue(
        model_id=MODEL_ID,
        actor="risk-approver",
        walk_forward_evidence_hash=registry.history(MODEL_ID)[-1].evidence_hashes[0],
        artifact_bundle_hash="b" * 64,
        inference_contract_hash="c" * 64,
        v2test_drill_hash="d" * 64,
        rollback_plan_hash="e" * 64,
    )
    shadow = registry.enter_shadow(
        MODEL_ID,
        approval=approval,
        approval_root=tmp_path / "approvals",  # type: ignore[operator]
    )
    assert shadow.status is ModelStatus.SHADOW
    with pytest.raises(ValueError, match="dedicated workflow"):
        registry.transition(
            MODEL_ID,
            ModelStatus.LOCKED_TEST_APPROVED,
            actor="attacker",
            evidence_hashes=("c" * 64,),
            reason="bypass locked-test evidence verification",
        )
    with pytest.raises(ValueError, match="verified workflow proof"):
        registry._approve_locked_test(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot transition"):
        registry.transition(
            MODEL_ID,
            ModelStatus.WALK_FORWARD_PASSED,
            actor="reviewer",
            evidence_hashes=("f" * 64,),
            reason="failed rollback",
        )
