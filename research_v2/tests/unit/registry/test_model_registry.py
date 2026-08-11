from __future__ import annotations

import pytest

from research_v2.evaluation.gates import GateReport, GateResult, GateStatus
from research_v2.registry.model_registry import ModelRegistry, ModelStatus


def _report(passed: bool) -> GateReport:
    status = GateStatus.PASS if passed else GateStatus.FAIL
    return GateReport(
        {
            gate: GateResult(status, 1.0, 0.0)
            for gate in ("G1", "G2", "G3", "G4", "G5", "G6")
        }
    )


def test_registry_requires_all_gates_before_walk_forward_promotion() -> None:
    registry = ModelRegistry()
    registry.register_draft("model-1", actor="trainer")

    promoted = registry.promote_walk_forward("model-1", actor="reviewer", gates=_report(True))

    assert promoted.status is ModelStatus.WALK_FORWARD_PASSED
    assert len(registry.history("model-1")) == 2
    with pytest.raises(ValueError, match="cannot transition"):
        registry.transition("model-1", ModelStatus.PRODUCTION, actor="reviewer", evidence_hash="x")


def test_registry_rejects_failed_gate_evidence_and_allows_retirement() -> None:
    registry = ModelRegistry()
    registry.register_draft("model-2", actor="trainer")
    with pytest.raises(ValueError, match="G1"):
        registry.promote_walk_forward("model-2", actor="reviewer", gates=_report(False))

    retired = registry.transition(
        "model-2",
        ModelStatus.RETIRED,
        actor="reviewer",
        evidence_hash="x",
    )
    assert retired.status is ModelStatus.RETIRED
