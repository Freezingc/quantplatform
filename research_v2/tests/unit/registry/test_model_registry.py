from __future__ import annotations

import pytest

from research_v2.evaluation.gates import GateEvidence, ScoreHealthEvidence
from research_v2.registry.model_registry import ModelRegistry, ModelStatus

MODEL_ID = "sha256:" + "a" * 64


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


def test_registry_allows_only_the_documented_lifecycle_path() -> None:
    registry = ModelRegistry()
    registry.register_draft(MODEL_ID, actor="trainer")
    registry.promote_walk_forward(MODEL_ID, actor="reviewer", evidence=_evidence())
    shadow = registry.transition(
        MODEL_ID,
        ModelStatus.SHADOW,
        actor="shadow-controller",
        evidence_hashes=("b" * 64,),
        reason="shadow evidence accepted",
    )
    approved = registry.transition(
        MODEL_ID,
        ModelStatus.LOCKED_TEST_APPROVED,
        actor="locked-controller",
        evidence_hashes=("c" * 64,),
        reason="locked test approved",
    )
    passed = registry.transition(
        MODEL_ID,
        ModelStatus.LOCKED_TEST_PASSED,
        actor="locked-controller",
        evidence_hashes=("d" * 64,),
        reason="locked test passed",
    )
    production = registry.transition(
        MODEL_ID,
        ModelStatus.PRODUCTION,
        actor="release-controller",
        evidence_hashes=("e" * 64,),
        reason="production approval",
    )

    assert [shadow.status, approved.status, passed.status, production.status] == [
        ModelStatus.SHADOW,
        ModelStatus.LOCKED_TEST_APPROVED,
        ModelStatus.LOCKED_TEST_PASSED,
        ModelStatus.PRODUCTION,
    ]
    with pytest.raises(ValueError, match="cannot transition"):
        registry.transition(
            MODEL_ID,
            ModelStatus.WALK_FORWARD_PASSED,
            actor="reviewer",
            evidence_hashes=("f" * 64,),
            reason="failed rollback",
        )
