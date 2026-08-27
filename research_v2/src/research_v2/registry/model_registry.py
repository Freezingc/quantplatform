"""Append-only model lifecycle registry with evidence-gated walk-forward promotion."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum
from numbers import Integral, Real
from pathlib import Path
from typing import cast

from research_v2.evaluation.gates import (
    GateEvidence,
    GateReport,
    GateStatus,
    ScoreHealthEvidence,
    evaluate_oos_gates,
)
from research_v2.monitoring.shadow import (
    ShadowDeploymentApproval,
    ShadowDeploymentApprovalVerifier,
)

_MODEL_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_EVIDENCE_HASH = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_GATES = ("G1", "G2", "G3", "G4", "G5", "G6")
_SHADOW_DEPLOYER_ENV = "QRV2_SHADOW_DEPLOYER_ID"


class ModelStatus(StrEnum):
    DRAFT = "DRAFT"
    WALK_FORWARD_PASSED = "WALK_FORWARD_PASSED"
    SHADOW = "SHADOW"
    LOCKED_TEST_APPROVED = "LOCKED_TEST_APPROVED"
    LOCKED_TEST_PASSED = "LOCKED_TEST_PASSED"
    LOCKED_TEST_FAILED = "LOCKED_TEST_FAILED"
    PRODUCTION = "PRODUCTION"
    RETIRED = "RETIRED"


_ALLOWED_TRANSITIONS: dict[ModelStatus, frozenset[ModelStatus]] = {
    ModelStatus.DRAFT: frozenset({ModelStatus.WALK_FORWARD_PASSED, ModelStatus.RETIRED}),
    ModelStatus.WALK_FORWARD_PASSED: frozenset({ModelStatus.SHADOW, ModelStatus.RETIRED}),
    ModelStatus.SHADOW: frozenset({ModelStatus.LOCKED_TEST_APPROVED, ModelStatus.RETIRED}),
    ModelStatus.LOCKED_TEST_APPROVED: frozenset(
        {ModelStatus.LOCKED_TEST_PASSED, ModelStatus.LOCKED_TEST_FAILED, ModelStatus.RETIRED}
    ),
    ModelStatus.LOCKED_TEST_PASSED: frozenset({ModelStatus.PRODUCTION, ModelStatus.RETIRED}),
    ModelStatus.LOCKED_TEST_FAILED: frozenset({ModelStatus.RETIRED}),
    ModelStatus.PRODUCTION: frozenset({ModelStatus.RETIRED}),
    ModelStatus.RETIRED: frozenset(),
}


@dataclass(frozen=True)
class ModelRecord:
    model_id: str
    status: ModelStatus


@dataclass(frozen=True)
class TransitionEvent:
    model_id: str
    previous_status: ModelStatus | None
    new_status: ModelStatus
    actor: str
    reason: str
    evidence_hashes: tuple[str, ...]
    transitioned_at: datetime


class ModelRegistry:
    """In-memory domain registry; persistence adapters must retain every transition event."""

    def __init__(self) -> None:
        self._records: dict[str, ModelRecord] = {}
        self._history: dict[str, list[TransitionEvent]] = {}

    def register_draft(self, model_id: str, *, actor: str) -> ModelRecord:
        validate_model_id(model_id)
        _validate_text(actor, "actor")
        if model_id in self._records:
            raise ValueError(f"model is already registered: {model_id}")
        record = ModelRecord(model_id=model_id, status=ModelStatus.DRAFT)
        self._records[model_id] = record
        self._append(record, None, actor, "draft registered", ())
        return record

    def promote_walk_forward(
        self,
        model_id: str,
        *,
        actor: str,
        evidence: GateEvidence,
    ) -> ModelRecord:
        """Promote only a DRAFT model whose raw ordinary-OOS G1-G6 evidence passes."""
        record = self._get(model_id)
        if record.status is not ModelStatus.DRAFT:
            raise ValueError("only DRAFT models can enter WALK_FORWARD_PASSED")
        gate_report = validate_walk_forward_evidence(evidence)
        return self._transition(
            record,
            ModelStatus.WALK_FORWARD_PASSED,
            actor,
            reason="all G1-G6 walk-forward gates passed",
            evidence_hashes=(_gate_evidence_hash(evidence, gate_report),),
        )

    def transition(
        self,
        model_id: str,
        status: ModelStatus,
        *,
        actor: str,
        evidence_hashes: tuple[str, ...],
        reason: str,
    ) -> ModelRecord:
        """Apply later lifecycle controls only when their append-only evidence is present."""
        record = self._get(model_id)
        if status not in _ALLOWED_TRANSITIONS[record.status]:
            raise ValueError(f"cannot transition {record.status} to {status}")
        if status is ModelStatus.WALK_FORWARD_PASSED:
            raise ValueError("WALK_FORWARD_PASSED requires raw G1-G6 evidence")
        if status is ModelStatus.SHADOW:
            raise ValueError("SHADOW requires a verified deployment approval")
        if status in {
            ModelStatus.LOCKED_TEST_APPROVED,
            ModelStatus.LOCKED_TEST_PASSED,
            ModelStatus.LOCKED_TEST_FAILED,
            ModelStatus.PRODUCTION,
        }:
            raise ValueError("locked-test and production states require their dedicated workflow")
        if status is not ModelStatus.RETIRED and not evidence_hashes:
            raise ValueError("lifecycle transition requires evidence hashes")
        return self._transition(record, status, actor, reason, evidence_hashes)

    def enter_shadow(
        self,
        model_id: str,
        *,
        approval: ShadowDeploymentApproval,
        approval_root: Path,
    ) -> ModelRecord:
        """Enter SHADOW only after an evaluator-owned deployment approval verifies all evidence."""
        record = self._get(model_id)
        if record.status is not ModelStatus.WALK_FORWARD_PASSED:
            raise ValueError("only WALK_FORWARD_PASSED models can enter SHADOW")
        verified = ShadowDeploymentApprovalVerifier.from_environment(approval_root).verify(approval)
        if verified.model_id != model_id:
            raise ValueError("shadow deployment approval does not match the registered model")
        walk_forward_hashes = self._history[model_id][-1].evidence_hashes
        if verified.walk_forward_evidence_hash not in walk_forward_hashes:
            raise ValueError("shadow deployment approval does not match walk-forward evidence")
        actor = os.environ.get(_SHADOW_DEPLOYER_ENV, "")
        if not actor.strip():
            raise ValueError(f"shadow deployment requires {_SHADOW_DEPLOYER_ENV}")
        if actor == verified.actor:
            raise ValueError("shadow deployment operator must be separate from the approver")
        return self._transition(
            record,
            ModelStatus.SHADOW,
            actor,
            reason="verified shadow deployment approval",
            evidence_hashes=(approval.approval_id,),
        )

    def history(self, model_id: str) -> tuple[TransitionEvent, ...]:
        self._get(model_id)
        return tuple(self._history[model_id])

    def _transition(
        self,
        record: ModelRecord,
        status: ModelStatus,
        actor: str,
        reason: str,
        evidence_hashes: tuple[str, ...],
    ) -> ModelRecord:
        _validate_text(actor, "actor")
        _validate_text(reason, "reason")
        _validate_evidence_hashes(evidence_hashes)
        updated = replace(record, status=status)
        self._records[record.model_id] = updated
        self._append(updated, record.status, actor, reason, evidence_hashes)
        return updated

    def _get(self, model_id: str) -> ModelRecord:
        try:
            return self._records[model_id]
        except KeyError as error:
            raise ValueError(f"model is not registered: {model_id}") from error

    def _append(
        self,
        record: ModelRecord,
        previous_status: ModelStatus | None,
        actor: str,
        reason: str,
        evidence_hashes: tuple[str, ...],
    ) -> None:
        self._history.setdefault(record.model_id, []).append(
            TransitionEvent(
                model_id=record.model_id,
                previous_status=previous_status,
                new_status=record.status,
                actor=actor,
                reason=reason,
                evidence_hashes=evidence_hashes,
                transitioned_at=datetime.now(UTC),
            )
        )


def load_walk_forward_evidence(path: Path) -> tuple[GateEvidence, str, GateReport]:
    """Load raw ordinary-OOS evidence and independently recalculate promotion gates."""
    try:
        raw = path.read_bytes()
        document = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"walk-forward evidence cannot be loaded: {path}") from error
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("walk-forward evidence schema is invalid")
    if document.get("role") != "ORDINARY_TEST":
        raise ValueError("walk-forward evidence role must be ORDINARY_TEST")
    payload = document.get("gate_evidence")
    if not isinstance(payload, dict):
        raise ValueError("walk-forward evidence gate_evidence is invalid")
    evidence = _parse_gate_evidence(cast(dict[str, object], payload))
    report = validate_walk_forward_evidence(evidence)
    return evidence, hashlib.sha256(raw).hexdigest(), report


def validate_walk_forward_evidence(evidence: GateEvidence) -> GateReport:
    """Recompute exactly G1-G6 from raw ordinary-OOS inputs, failing closed on every gap."""
    if not isinstance(evidence, GateEvidence):
        raise TypeError("walk-forward promotion requires GateEvidence")
    report = evaluate_oos_gates(evidence, is_baseline=False)
    if tuple(sorted(report.results)) != _REQUIRED_GATES:
        raise ValueError("walk-forward evidence must contain exactly G1-G6")
    failed = [
        gate for gate in _REQUIRED_GATES if report.results[gate].status is not GateStatus.PASS
    ]
    if failed:
        raise ValueError(f"walk-forward evidence did not pass: {', '.join(failed)}")
    return report


def _parse_gate_evidence(payload: dict[str, object]) -> GateEvidence:
    expected = {
        "mean_rank_ic",
        "hac_t_statistic",
        "yearly_mean_rank_ic",
        "candidate_delta_rank_ic",
        "valid_test_day_count",
        "score_health",
    }
    if set(payload) != expected:
        raise ValueError("walk-forward evidence gate_evidence fields are invalid")
    yearly_raw = payload["yearly_mean_rank_ic"]
    health_raw = payload["score_health"]
    if not isinstance(yearly_raw, dict) or not isinstance(health_raw, list):
        raise ValueError("walk-forward evidence yearly or score-health fields are invalid")
    yearly = {
        _parse_year(year): _finite_number(value, "yearly_mean_rank_ic")
        for year, value in yearly_raw.items()
    }
    if set(yearly) != set(range(2020, 2026)):
        raise ValueError("walk-forward evidence must contain yearly values for 2020-2025")
    health = tuple(_parse_score_health(item) for item in health_raw)
    valid_day_count = payload["valid_test_day_count"]
    if (
        not isinstance(valid_day_count, Integral)
        or isinstance(valid_day_count, bool)
        or valid_day_count <= 0
    ):
        raise ValueError("walk-forward evidence valid_test_day_count is invalid")
    return GateEvidence(
        mean_rank_ic=_finite_number(payload["mean_rank_ic"], "mean_rank_ic"),
        hac_t_statistic=_finite_number(payload["hac_t_statistic"], "hac_t_statistic"),
        yearly_mean_rank_ic=yearly,
        candidate_delta_rank_ic=_finite_number(
            payload["candidate_delta_rank_ic"],
            "candidate_delta_rank_ic",
        ),
        valid_test_day_count=int(valid_day_count),
        score_health=health,
    )


def _parse_score_health(value: object) -> ScoreHealthEvidence:
    if not isinstance(value, dict) or set(value) != {"coverage", "unique_score_count", "score_std"}:
        raise ValueError("walk-forward evidence score_health is invalid")
    count = value["unique_score_count"]
    if not isinstance(count, Integral) or isinstance(count, bool):
        raise ValueError("walk-forward evidence unique_score_count is invalid")
    return ScoreHealthEvidence(
        coverage=_finite_number(value["coverage"], "coverage"),
        unique_score_count=int(count),
        score_std=_finite_number(value["score_std"], "score_std"),
    )


def _parse_year(value: object) -> int:
    if not isinstance(value, str) or not value.isdecimal():
        raise ValueError("walk-forward evidence year is invalid")
    return int(value)


def _finite_number(value: object, name: str) -> float:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"walk-forward evidence {name} is invalid")
    return float(value)


def _gate_evidence_hash(evidence: GateEvidence, report: GateReport) -> str:
    payload = {
        "gate_evidence": asdict(evidence),
        "recomputed_gates": {
            name: result.status.value for name, result in sorted(report.results.items())
        },
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def validate_model_id(model_id: str) -> None:
    if not isinstance(model_id, str) or not _MODEL_ID.fullmatch(model_id):
        raise ValueError("model_id must be sha256:<64 lowercase hex characters>")


def _validate_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")


def _validate_evidence_hashes(values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple) or len(set(values)) != len(values):
        raise ValueError("evidence hashes must be a unique tuple")
    if not all(isinstance(value, str) and _EVIDENCE_HASH.fullmatch(value) for value in values):
        raise ValueError("evidence hashes must be SHA-256 hex digests")


__all__ = [
    "ModelRecord",
    "ModelRegistry",
    "ModelStatus",
    "TransitionEvent",
    "load_walk_forward_evidence",
    "validate_model_id",
    "validate_walk_forward_evidence",
]
