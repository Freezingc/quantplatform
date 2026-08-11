"""Append-only model lifecycle registry with evidence-gated walk-forward promotion."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from enum import StrEnum

from research_v2.evaluation.gates import GateReport, GateStatus


class ModelStatus(StrEnum):
    DRAFT = "DRAFT"
    WALK_FORWARD_PASSED = "WALK_FORWARD_PASSED"
    SHADOW = "SHADOW"
    LOCKED_TEST_APPROVED = "LOCKED_TEST_APPROVED"
    LOCKED_TEST_PASSED = "LOCKED_TEST_PASSED"
    PRODUCTION = "PRODUCTION"
    RETIRED = "RETIRED"


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
    evidence_hash: str
    transitioned_at: datetime


class ModelRegistry:
    """In-memory domain registry; persistence adapters must retain every transition event."""

    def __init__(self) -> None:
        self._records: dict[str, ModelRecord] = {}
        self._history: dict[str, list[TransitionEvent]] = {}

    def register_draft(self, model_id: str, *, actor: str) -> ModelRecord:
        if not model_id or not actor:
            raise ValueError("model_id and actor cannot be empty")
        if model_id in self._records:
            raise ValueError(f"model is already registered: {model_id}")
        record = ModelRecord(model_id=model_id, status=ModelStatus.DRAFT)
        self._records[model_id] = record
        self._append(record, None, actor, evidence_hash="")
        return record

    def promote_walk_forward(
        self,
        model_id: str,
        *,
        actor: str,
        gates: GateReport,
    ) -> ModelRecord:
        record = self._get(model_id)
        if record.status is not ModelStatus.DRAFT:
            raise ValueError("only DRAFT models can enter WALK_FORWARD_PASSED")
        failed = [
            name for name, result in gates.results.items() if result.status is not GateStatus.PASS
        ]
        if failed:
            raise ValueError(f"walk-forward evidence did not pass: {', '.join(sorted(failed))}")
        return self._transition(
            record,
            ModelStatus.WALK_FORWARD_PASSED,
            actor,
            evidence_hash=_gate_evidence_hash(gates),
        )

    def transition(
        self,
        model_id: str,
        status: ModelStatus,
        *,
        actor: str,
        evidence_hash: str,
    ) -> ModelRecord:
        """Permit only retirement here; later shadow/locked transitions belong to F controls."""
        record = self._get(model_id)
        if status is not ModelStatus.RETIRED:
            raise ValueError(
                f"cannot transition {record.status} to {status} through this registry path"
            )
        return self._transition(record, status, actor, evidence_hash=evidence_hash)

    def history(self, model_id: str) -> tuple[TransitionEvent, ...]:
        self._get(model_id)
        return tuple(self._history[model_id])

    def _transition(
        self,
        record: ModelRecord,
        status: ModelStatus,
        actor: str,
        *,
        evidence_hash: str,
    ) -> ModelRecord:
        if not actor:
            raise ValueError("actor cannot be empty")
        if not evidence_hash and status is not ModelStatus.RETIRED:
            raise ValueError("promotion requires an evidence hash")
        updated = replace(record, status=status)
        self._records[record.model_id] = updated
        self._append(updated, record.status, actor, evidence_hash)
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
        evidence_hash: str,
    ) -> None:
        self._history.setdefault(record.model_id, []).append(
            TransitionEvent(
                model_id=record.model_id,
                previous_status=previous_status,
                new_status=record.status,
                actor=actor,
                evidence_hash=evidence_hash,
                transitioned_at=datetime.now(UTC),
            )
        )


def _gate_evidence_hash(gates: GateReport) -> str:
    payload = {
        name: {
            "observed_value": result.observed_value,
            "status": result.status.value,
            "threshold": result.threshold,
        }
        for name, result in sorted(gates.results.items())
    }
    encoded = json.dumps(payload, separators=(",", ":"), sort_keys=True).encode()
    return hashlib.sha256(encoded).hexdigest()
