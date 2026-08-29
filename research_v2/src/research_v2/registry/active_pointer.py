"""Audited compare-and-swap active model pointers and rollback guards."""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from threading import RLock

from research_v2.monitoring.shadow import (
    ShadowDeploymentApproval,
    ShadowDeploymentApprovalVerifier,
    ShadowEvidence,
    ShadowEvidenceVerifier,
    ShadowQualificationStatus,
)
from research_v2.registry.locked_test import (
    ProductionDeploymentApproval,
    ProductionDeploymentApprovalVerifier,
)
from research_v2.registry.model_registry import ModelRegistry, ModelStatus, validate_model_id

_CONSUMER_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_EVIDENCE_ID = re.compile(r"^[0-9a-f]{64}$")
_ENVIRONMENTS = frozenset({"shadow", "production"})
_ARTIFACT_CONTRACT_MODEL_VERSION = 1


class PointerEventKind(StrEnum):
    SWITCH = "SWITCH"
    ROLLBACK = "ROLLBACK"


@dataclass(frozen=True)
class ActiveModelPointer:
    consumer_id: str
    environment: str
    model_id: str
    model_version: int
    pointer_version: int


@dataclass(frozen=True)
class PointerSwitchEvidence:
    """Verified inputs required to activate a target model for one environment."""

    evidence_ids: tuple[str, ...] = ()
    shadow_approval: ShadowDeploymentApproval | None = None
    shadow_approval_root: Path | None = None
    shadow_evidence: ShadowEvidence | None = None
    shadow_evidence_root: Path | None = None
    production_approval: ProductionDeploymentApproval | None = None
    production_approval_root: Path | None = None


@dataclass(frozen=True)
class PointerEvent:
    event_id: str
    kind: PointerEventKind
    consumer_id: str
    environment: str
    old_pointer: ActiveModelPointer | None
    new_pointer: ActiveModelPointer
    actor: str
    reason: str
    evidence_ids: tuple[str, ...]
    effective_at: datetime
    run_id: str
    shadow_approval_id: str | None
    shadow_evidence_id: str | None
    production_approval_id: str | None
    reverts_event_id: str | None


@dataclass(frozen=True)
class ActivePointerUpdate:
    pointer: ActiveModelPointer
    event: PointerEvent


@dataclass(frozen=True)
class _Eligibility:
    evidence_ids: tuple[str, ...]
    shadow_approval_id: str | None
    shadow_evidence_id: str | None
    production_approval_id: str | None


class PointerEligibilityGuard:
    """Verify registry, artifact, and signed shadow evidence before pointer activation."""

    def __init__(
        self,
        registry: ModelRegistry,
        *,
        bundle_digest: Callable[[str], str],
    ) -> None:
        self._registry = registry
        self._bundle_digest = bundle_digest

    def validate(
        self,
        environment: str,
        model_id: str,
        model_version: int,
        evidence: PointerSwitchEvidence,
    ) -> _Eligibility:
        _validate_evidence(evidence)
        if model_version != _ARTIFACT_CONTRACT_MODEL_VERSION:
            raise ValueError("active pointer model_version is not bound to the artifact contract")
        record = self._registry.get(model_id)
        artifact_digest = self._verified_bundle_digest(model_id)
        if environment == "production":
            if record.status is not ModelStatus.PRODUCTION:
                raise ValueError("production pointer requires a PRODUCTION model")
            production_approval = evidence.production_approval
            production_approval_root = evidence.production_approval_root
            if production_approval is None or not isinstance(production_approval_root, Path):
                raise ValueError("production pointer requires a verified production approval")
            verified = ProductionDeploymentApprovalVerifier.from_environment(
                production_approval_root
            ).verify(
                production_approval
            )
            if verified.model_id != model_id:
                raise ValueError("production approval model does not match pointer target")
            if verified.artifact_bundle_hash != artifact_digest:
                raise ValueError("production approval artifact does not match pointer target")
            evidence_ids = (*evidence.evidence_ids, verified.approval_id)
            _validate_evidence_ids(evidence_ids)
            return _Eligibility(evidence_ids, None, None, verified.approval_id)
        if environment != "shadow":
            raise ValueError("active pointer environment is invalid")
        if record.status not in {ModelStatus.WALK_FORWARD_PASSED, ModelStatus.SHADOW}:
            raise ValueError("shadow pointer requires at least a WALK_FORWARD_PASSED model")
        approval, approval_root = evidence.shadow_approval, evidence.shadow_approval_root
        if approval is None or not isinstance(approval_root, Path):
            raise ValueError("shadow pointer requires a verified deployment approval")
        approved = ShadowDeploymentApprovalVerifier.from_environment(approval_root).verify(approval)
        if approved.model_id != model_id:
            raise ValueError("shadow deployment approval model does not match pointer target")
        if approved.artifact_bundle_hash != artifact_digest:
            raise ValueError("shadow deployment approval artifact does not match pointer target")
        walk_forward_hashes = {
            value
            for event in self._registry.history(model_id)
            if event.new_status is ModelStatus.WALK_FORWARD_PASSED
            for value in event.evidence_hashes
        }
        if approved.walk_forward_evidence_hash not in walk_forward_hashes:
            raise ValueError("shadow deployment approval does not match walk-forward evidence")
        evidence_ids = (*evidence.evidence_ids, approval.approval_id)
        shadow_evidence_id: str | None = None
        if (
            record.status is ModelStatus.SHADOW
            or evidence.shadow_evidence is not None
            or evidence.shadow_evidence_root is not None
        ):
            shadow_evidence_id = self._verify_shadow_readiness(model_id, evidence)
            evidence_ids = (*evidence_ids, shadow_evidence_id)
        _validate_evidence_ids(evidence_ids)
        return _Eligibility(evidence_ids, approval.approval_id, shadow_evidence_id, None)

    def _verified_bundle_digest(self, model_id: str) -> str:
        try:
            value = self._bundle_digest(model_id)
        except (KeyError, OSError, ValueError) as error:
            raise ValueError("pointer target artifact integrity verification failed") from error
        if not isinstance(value, str) or not _EVIDENCE_ID.fullmatch(value):
            raise ValueError("pointer target artifact digest is invalid")
        return value

    @staticmethod
    def _verify_shadow_readiness(model_id: str, evidence: PointerSwitchEvidence) -> str:
        shadow_evidence = evidence.shadow_evidence
        evidence_root = evidence.shadow_evidence_root
        if shadow_evidence is None or not isinstance(evidence_root, Path):
            raise ValueError("SHADOW pointer requires verified readiness evidence")
        ShadowEvidenceVerifier.from_environment(evidence_root).verify(shadow_evidence)
        payload = shadow_evidence.payload
        max_psi_errors = payload.get("max_consecutive_psi_errors")
        if (
            payload.get("model_id") != model_id
            or payload.get("qualification_status") != ShadowQualificationStatus.READY_FOR_REVIEW
            or isinstance(max_psi_errors, bool)
            or not isinstance(max_psi_errors, int)
        ):
            raise ValueError("shadow readiness evidence has unresolved blocker or drift review")
        if max_psi_errors >= 3:
            raise ValueError("shadow readiness evidence has unresolved blocker or drift review")
        return shadow_evidence.evidence_id


class ActiveModelPointerStore:
    """In-memory CAS store with atomic pointer visibility and append-only audit history."""

    def __init__(
        self,
        *,
        eligibility: PointerEligibilityGuard,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        self._eligibility = eligibility
        self._clock = clock or (lambda: datetime.now(UTC))
        self._pointers: dict[tuple[str, str], ActiveModelPointer] = {}
        self._history: dict[tuple[str, str], list[PointerEvent]] = {}
        self._events: dict[str, PointerEvent] = {}
        self._lock = RLock()

    def get(self, consumer_id: str, environment: str) -> ActiveModelPointer | None:
        _validate_key(consumer_id, environment)
        with self._lock:
            return self._pointers.get((consumer_id, environment))

    def history(self, consumer_id: str, environment: str) -> tuple[PointerEvent, ...]:
        _validate_key(consumer_id, environment)
        with self._lock:
            return tuple(self._history.get((consumer_id, environment), ()))

    def compare_and_swap(
        self,
        *,
        consumer_id: str,
        environment: str,
        expected_current: ActiveModelPointer | None,
        model_id: str,
        model_version: int,
        actor: str,
        reason: str,
        evidence: PointerSwitchEvidence,
        run_id: str,
    ) -> ActivePointerUpdate | None:
        """Switch only when the full pointer observed by the caller still matches."""
        _validate_key(consumer_id, environment)
        _validate_expected(expected_current, consumer_id, environment)
        _validate_model_version(model_id, model_version)
        _validate_text(actor, "actor")
        _validate_text(reason, "reason")
        _validate_run_id(run_id)
        with self._lock:
            key = (consumer_id, environment)
            current = self._pointers.get(key)
            if current != expected_current:
                return None
            if current is not None and (current.model_id, current.model_version) == (
                model_id,
                model_version,
            ):
                raise ValueError("active pointer already targets this model version")
            eligibility = self._eligibility.validate(environment, model_id, model_version, evidence)
            pointer = ActiveModelPointer(
                consumer_id=consumer_id,
                environment=environment,
                model_id=model_id,
                model_version=model_version,
                pointer_version=(current.pointer_version if current is not None else 0) + 1,
            )
            event = self._event(
                PointerEventKind.SWITCH,
                pointer,
                old_pointer=current,
                actor=actor,
                reason=reason,
                run_id=run_id,
                eligibility=eligibility,
                reverts_event_id=None,
            )
            self._append(key, pointer, event)
            return ActivePointerUpdate(pointer, event)

    def rollback(
        self,
        *,
        consumer_id: str,
        environment: str,
        expected_current: ActiveModelPointer | None,
        event_id: str,
        actor: str,
        reason: str,
        evidence: PointerSwitchEvidence,
        run_id: str,
    ) -> ActivePointerUpdate | None:
        """Append a CAS rollback to the verified target of one current switch event."""
        _validate_key(consumer_id, environment)
        _validate_expected(expected_current, consumer_id, environment)
        _validate_text(actor, "actor")
        _validate_text(reason, "reason")
        _validate_run_id(run_id)
        if not isinstance(event_id, str) or not _is_uuid(event_id):
            raise ValueError("rollback event_id is invalid")
        with self._lock:
            key = (consumer_id, environment)
            current = self._pointers.get(key)
            if current != expected_current:
                return None
            source = self._events.get(event_id)
            if (
                source is None
                or source.kind is not PointerEventKind.SWITCH
                or source.consumer_id != consumer_id
                or source.environment != environment
                or source.new_pointer != current
                or source.old_pointer is None
            ):
                raise ValueError(
                    "rollback event does not identify the current reversible pointer switch"
                )
            target = source.old_pointer
            eligibility = self._eligibility.validate(
                environment, target.model_id, target.model_version, evidence
            )
            pointer = ActiveModelPointer(
                consumer_id=consumer_id,
                environment=environment,
                model_id=target.model_id,
                model_version=target.model_version,
                pointer_version=current.pointer_version + 1,
            )
            event = self._event(
                PointerEventKind.ROLLBACK,
                pointer,
                old_pointer=current,
                actor=actor,
                reason=reason,
                run_id=run_id,
                eligibility=eligibility,
                reverts_event_id=source.event_id,
            )
            self._append(key, pointer, event)
            return ActivePointerUpdate(pointer, event)

    def _event(
        self,
        kind: PointerEventKind,
        pointer: ActiveModelPointer,
        *,
        old_pointer: ActiveModelPointer | None,
        actor: str,
        reason: str,
        run_id: str,
        eligibility: _Eligibility,
        reverts_event_id: str | None,
    ) -> PointerEvent:
        effective_at = self._clock()
        if not isinstance(effective_at, datetime) or effective_at.tzinfo is None:
            raise ValueError("active pointer clock must return a timezone-aware datetime")
        return PointerEvent(
            event_id=str(uuid.uuid4()),
            kind=kind,
            consumer_id=pointer.consumer_id,
            environment=pointer.environment,
            old_pointer=old_pointer,
            new_pointer=pointer,
            actor=actor,
            reason=reason,
            evidence_ids=eligibility.evidence_ids,
            effective_at=effective_at.astimezone(UTC),
            run_id=run_id,
            shadow_approval_id=eligibility.shadow_approval_id,
            shadow_evidence_id=eligibility.shadow_evidence_id,
            production_approval_id=eligibility.production_approval_id,
            reverts_event_id=reverts_event_id,
        )

    def _append(
        self,
        key: tuple[str, str],
        pointer: ActiveModelPointer,
        event: PointerEvent,
    ) -> None:
        self._pointers[key] = pointer
        self._history.setdefault(key, []).append(event)
        self._events[event.event_id] = event


def _validate_key(consumer_id: str, environment: str) -> None:
    if not isinstance(consumer_id, str) or not _CONSUMER_ID.fullmatch(consumer_id):
        raise ValueError("active pointer consumer_id is invalid")
    if environment not in _ENVIRONMENTS:
        raise ValueError("active pointer environment is invalid")


def _validate_expected(
    expected: ActiveModelPointer | None,
    consumer_id: str,
    environment: str,
) -> None:
    if expected is None:
        return
    if not isinstance(expected, ActiveModelPointer):
        raise ValueError("active pointer expected_current is invalid")
    if expected.consumer_id != consumer_id or expected.environment != environment:
        raise ValueError("active pointer expected_current key is invalid")
    _validate_model_version(expected.model_id, expected.model_version)
    if expected.pointer_version < 1:
        raise ValueError("active pointer expected_current version is invalid")


def _validate_model_version(model_id: str, model_version: int) -> None:
    validate_model_id(model_id)
    if isinstance(model_version, bool) or not isinstance(model_version, int) or model_version < 1:
        raise ValueError("active pointer model_version is invalid")


def _validate_evidence(evidence: PointerSwitchEvidence) -> None:
    if not isinstance(evidence, PointerSwitchEvidence):
        raise ValueError("active pointer evidence is invalid")
    _validate_evidence_ids(evidence.evidence_ids)
    paired = (
        (evidence.shadow_approval, evidence.shadow_approval_root),
        (evidence.shadow_evidence, evidence.shadow_evidence_root),
        (evidence.production_approval, evidence.production_approval_root),
    )
    for signed, root in paired:
        if (signed is None) != (root is None):
            raise ValueError("active pointer signed evidence and root must be provided together")


def _validate_evidence_ids(values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple) or len(set(values)) != len(values):
        raise ValueError("active pointer evidence IDs must be unique")
    if not all(isinstance(value, str) and _EVIDENCE_ID.fullmatch(value) for value in values):
        raise ValueError("active pointer evidence IDs are invalid")


def _validate_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"active pointer {name} is invalid")


def _validate_run_id(run_id: str) -> None:
    if not isinstance(run_id, str) or not _is_uuid(run_id):
        raise ValueError("active pointer run_id is invalid")


def _is_uuid(value: str) -> bool:
    try:
        return str(uuid.UUID(value)) == value
    except ValueError:
        return False


__all__ = [
    "ActiveModelPointer",
    "ActiveModelPointerStore",
    "ActivePointerUpdate",
    "PointerEligibilityGuard",
    "PointerEvent",
    "PointerEventKind",
    "PointerSwitchEvidence",
]
