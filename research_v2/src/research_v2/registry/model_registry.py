"""Append-only model lifecycle registry with evidence-gated walk-forward promotion."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from collections.abc import Mapping
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
    _ordinary_test_years,
    evaluate_oos_gates,
)
from research_v2.evaluation.model_evaluator import FrozenOosEvaluation
from research_v2.monitoring.shadow import (
    ShadowDeploymentApproval,
    ShadowDeploymentApprovalVerifier,
)

_MODEL_ID = re.compile(r"^sha256:[0-9a-f]{64}$")
_EVIDENCE_HASH = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_GATES = ("G1", "G2", "G3", "G4", "G5", "G6")
_SHADOW_DEPLOYER_ENV = "QRV2_SHADOW_DEPLOYER_ID"
_WORKFLOW_PROOF_ISSUER = object()


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
    lineage_id: str = ""
    parent_model_id: str | None = None


@dataclass(frozen=True)
class TransitionEvent:
    model_id: str
    previous_status: ModelStatus | None
    new_status: ModelStatus
    actor: str
    reason: str
    evidence_hashes: tuple[str, ...]
    transitioned_at: datetime


class _RegistryWorkflowProof:
    """Unforgeable in-process proof emitted only after a signed workflow record verifies."""

    _issuer: object
    operation: str
    model_id: str
    actor: str
    evidence_hash: str
    passed: bool | None

    def __init__(self) -> None:
        raise TypeError("registry workflow proofs are authority-issued")

    @classmethod
    def _issue(
        cls,
        *,
        operation: str,
        model_id: str,
        actor: str,
        evidence_hash: str,
        passed: bool | None = None,
    ) -> _RegistryWorkflowProof:
        validate_model_id(model_id)
        _validate_text(actor, "actor")
        _validate_evidence_hashes((evidence_hash,))
        if operation not in {"APPROVE_LOCKED_TEST", "COMPLETE_LOCKED_TEST", "APPROVE_PRODUCTION"}:
            raise ValueError("registry workflow proof operation is invalid")
        if operation == "COMPLETE_LOCKED_TEST":
            if not isinstance(passed, bool):
                raise ValueError("registry workflow proof passed flag is invalid")
        elif passed is not None:
            raise ValueError("registry workflow proof passed flag is invalid")
        proof = object.__new__(cls)
        proof._issuer = _WORKFLOW_PROOF_ISSUER
        proof.operation = operation
        proof.model_id = model_id
        proof.actor = actor
        proof.evidence_hash = evidence_hash
        proof.passed = passed
        return proof


def _issue_registry_workflow_proof(
    *,
    operation: str,
    model_id: str,
    actor: str,
    evidence_hash: str,
    passed: bool | None = None,
) -> _RegistryWorkflowProof:
    """Internal authority hook for evidence-backed lifecycle transitions."""
    return _RegistryWorkflowProof._issue(
        operation=operation,
        model_id=model_id,
        actor=actor,
        evidence_hash=evidence_hash,
        passed=passed,
    )


class ModelRegistry:
    """In-memory domain registry; persistence adapters must retain every transition event."""

    def __init__(self) -> None:
        self._records: dict[str, ModelRecord] = {}
        self._history: dict[str, list[TransitionEvent]] = {}

    @classmethod
    def restore(
        cls,
        records: Mapping[str, ModelRecord],
        histories: Mapping[str, tuple[TransitionEvent, ...]],
    ) -> ModelRegistry:
        """Rebuild a registry from persisted state whose signatures were already verified.

        Transitions are deliberately NOT re-validated. Each was validated against its evidence
        when it was written, and the later lifecycle states require ``_RegistryWorkflowProof``
        objects that are unforgeable in-process tokens -- they cannot be reconstructed from disk
        even in principle, so replaying history through the transition methods is impossible
        rather than merely inconvenient. What this does validate is everything about the
        persisted state itself: identity format, status membership, lineage agreement with the
        registered parent, and an unbroken event chain whose last status is the record's status.
        A store that hands over state failing any of those is corrupt and must not become live.
        """
        if set(records) != set(histories):
            raise ValueError("registry restore requires one event history per record")
        restored = cls()
        for model_id, record in records.items():
            validate_model_id(model_id)
            if record.model_id != model_id:
                raise ValueError(f"persisted record names a different model: {model_id}")
            if not isinstance(record.status, ModelStatus):
                raise ValueError(f"persisted status is not a ModelStatus: {model_id}")
            if record.parent_model_id is None:
                if record.lineage_id != model_id:
                    raise ValueError(f"root model lineage_id must equal its model_id: {model_id}")
            else:
                validate_model_id(record.parent_model_id)
                parent = records.get(record.parent_model_id)
                if parent is None:
                    raise ValueError(f"persisted parent model is absent: {record.parent_model_id}")
                if record.lineage_id != parent.lineage_id:
                    raise ValueError(f"derived model lineage disagrees with its parent: {model_id}")
            events = histories[model_id]
            if not events:
                raise ValueError(f"persisted model has no transition events: {model_id}")
            previous: ModelStatus | None = None
            for position, event in enumerate(events):
                if event.model_id != model_id:
                    raise ValueError(f"transition event names a different model: {model_id}")
                if event.previous_status is not previous:
                    raise ValueError(
                        f"transition event chain is broken at position {position}: {model_id}"
                    )
                if not isinstance(event.new_status, ModelStatus):
                    raise ValueError(f"transition event status is invalid: {model_id}")
                _validate_text(event.actor, "actor")
                _validate_text(event.reason, "reason")
                _validate_evidence_hashes(event.evidence_hashes)
                previous = event.new_status
            if previous is not record.status:
                raise ValueError(
                    f"persisted status disagrees with the last transition event: {model_id}"
                )
            first = events[0]
            if first.previous_status is not None or first.new_status is not ModelStatus.DRAFT:
                raise ValueError(f"persisted history must begin at DRAFT: {model_id}")
            restored._records[model_id] = record
            restored._history[model_id] = list(events)
        return restored

    def model_ids(self) -> tuple[str, ...]:
        """Every registered model id, sorted, so a store can enumerate what it persisted."""
        return tuple(sorted(self._records))

    def register_draft(
        self,
        model_id: str,
        *,
        actor: str,
        lineage_id: str | None = None,
        parent_model_id: str | None = None,
    ) -> ModelRecord:
        validate_model_id(model_id)
        _validate_text(actor, "actor")
        if model_id in self._records:
            raise ValueError(f"model is already registered: {model_id}")
        if parent_model_id is None:
            if lineage_id is not None and lineage_id != model_id:
                raise ValueError("root model lineage_id must equal its immutable model_id")
            resolved_lineage_id = model_id
        else:
            validate_model_id(parent_model_id)
            parent = self._get(parent_model_id)
            if lineage_id is not None and lineage_id != parent.lineage_id:
                raise ValueError("derived model lineage_id must match its registered parent")
            resolved_lineage_id = parent.lineage_id
        record = ModelRecord(
            model_id=model_id,
            status=ModelStatus.DRAFT,
            lineage_id=resolved_lineage_id,
            parent_model_id=parent_model_id,
        )
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

    def _approve_locked_test(self, proof: _RegistryWorkflowProof) -> ModelRecord:
        """Enter LOCKED_TEST_APPROVED only through the one-shot authority."""
        record = self._verified_workflow_proof(proof, "APPROVE_LOCKED_TEST")
        if record.status is not ModelStatus.SHADOW:
            raise ValueError("only SHADOW models can enter LOCKED_TEST_APPROVED")
        return self._transition(
            record,
            ModelStatus.LOCKED_TEST_APPROVED,
            proof.actor,
            reason="verified two-person locked-test approval",
            evidence_hashes=(proof.evidence_hash,),
        )

    def _complete_locked_test(self, proof: _RegistryWorkflowProof) -> ModelRecord:
        """Record the terminal locked-test decision made by the dedicated evaluator."""
        record = self._verified_workflow_proof(proof, "COMPLETE_LOCKED_TEST")
        if record.status is not ModelStatus.LOCKED_TEST_APPROVED:
            raise ValueError("only LOCKED_TEST_APPROVED models can complete a locked test")
        assert isinstance(proof.passed, bool)
        status = ModelStatus.LOCKED_TEST_PASSED if proof.passed else ModelStatus.LOCKED_TEST_FAILED
        return self._transition(
            record,
            status,
            proof.actor,
            reason="locked-test aggregate evaluation completed",
            evidence_hashes=(proof.evidence_hash,),
        )

    def _approve_production(self, proof: _RegistryWorkflowProof) -> ModelRecord:
        """Enter PRODUCTION only after the independent deployment approval verifies a pass."""
        record = self._verified_workflow_proof(proof, "APPROVE_PRODUCTION")
        if record.status is not ModelStatus.LOCKED_TEST_PASSED:
            raise ValueError("only LOCKED_TEST_PASSED models can enter PRODUCTION")
        return self._transition(
            record,
            ModelStatus.PRODUCTION,
            proof.actor,
            reason="verified independent production approval",
            evidence_hashes=(proof.evidence_hash,),
        )

    def history(self, model_id: str) -> tuple[TransitionEvent, ...]:
        self._get(model_id)
        return tuple(self._history[model_id])

    def get(self, model_id: str) -> ModelRecord:
        """Return the immutable registered state for downstream deployment guards."""
        return self._get(model_id)

    def _verified_workflow_proof(
        self,
        proof: _RegistryWorkflowProof,
        operation: str,
    ) -> ModelRecord:
        if (
            not isinstance(proof, _RegistryWorkflowProof)
            or getattr(proof, "_issuer", None) is not _WORKFLOW_PROOF_ISSUER
            or proof.operation != operation
        ):
            raise ValueError("registry transition requires a verified workflow proof")
        return self._get(proof.model_id)

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


def build_walk_forward_evidence_document(evaluation: FrozenOosEvaluation) -> dict[str, object]:
    """Render a completed ordinary-OOS evaluation as the registry's promotion evidence.

    ``load_walk_forward_evidence`` is the only route into the lifecycle registry, and nothing
    in the codebase produced the document it consumes: the candidate driver writes a
    human-facing report whose shape the loader rejects on the first field. This is that
    missing producer, and it is the last gap between "passed all six gates" and "promotable".

    The output is safe to treat as untrusted input, because the registry does not trust it.
    ``load_walk_forward_evidence`` re-derives G1-G6 from these raw numbers through
    ``validate_walk_forward_evidence`` and never reads a reported gate status, so emitting a
    document cannot promote a model that does not actually pass.
    """
    evidence = evaluation.gate_evidence
    if evidence.candidate_delta_rank_ic is None:
        raise ValueError(
            "promotion evidence requires a candidate delta against a baseline; a baseline's "
            "own walk-forward run is not promotable evidence"
        )
    return {
        "schema_version": 1,
        "role": "ORDINARY_TEST",
        "gate_evidence": {
            "mean_rank_ic": evidence.mean_rank_ic,
            "hac_t_statistic": evidence.hac_t_statistic,
            "yearly_mean_rank_ic": {
                str(year): value for year, value in evidence.yearly_mean_rank_ic.items()
            },
            "candidate_delta_rank_ic": evidence.candidate_delta_rank_ic,
            "valid_test_day_count": evidence.valid_test_day_count,
            "score_health": [asdict(item) for item in evidence.score_health],
        },
    }


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
    # Derived from the fold contract, never hardcoded. This demanded 2020-2025 -- the test
    # years of the retired F01-F06 grid -- so evidence from the current F1-F4 contract (test
    # years 2022-2025) was rejected outright and NO v2 candidate could be promoted, however
    # it scored. That is the same stale-range bug commit fd55d2d fixed in evaluate_oos_gates,
    # in the one place that fix did not reach: the gates themselves were already deriving
    # their required years from the contract, while the loader feeding them was not.
    required_years = set(_ordinary_test_years())
    if set(yearly) != required_years:
        raise ValueError(
            "walk-forward evidence must contain yearly values for "
            f"{min(required_years)}-{max(required_years)}"
        )
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


def derive_walk_forward_model_id(bundle_digests: Mapping[str, str]) -> str:
    """Mint the registry model_id for a walk-forward candidate from its verified bundles.

    A walk-forward run trains one bundle per ordinary fold, but the registry -- and the shadow
    and pointer subsystems that consume a model_id -- identify a model by a single
    ``sha256:<64 hex>`` value. Nothing derived that value, so a candidate that passed all six
    gates had no name to be registered under and ``register_draft`` could not be called.

    The id is a deterministic function of the verified per-fold bundle digests and of nothing
    else, which is what makes it safe to mint outside a trusted boundary. Each bundle digest
    already covers every file in its bundle -- config, feature order, the model card carrying
    family and model_code_version, the signature, all three seed checkpoints and the validation
    predictions -- so two runs yield the same id only if they produced the same models, and a
    caller cannot shift the id with metadata the artifacts do not already pin.
    """
    if not isinstance(bundle_digests, Mapping) or not bundle_digests:
        raise ValueError("a walk-forward model id requires at least one fold bundle digest")
    normalized: dict[str, str] = {}
    for fold_id, digest in bundle_digests.items():
        _validate_text(fold_id, "fold_id")
        if not isinstance(digest, str) or not _EVIDENCE_HASH.fullmatch(digest):
            raise ValueError(f"bundle digest for {fold_id} must be a SHA-256 hex digest")
        normalized[fold_id] = digest
    payload = json.dumps(
        {"bundle_digests": normalized}, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    model_id = "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()
    validate_model_id(model_id)
    return model_id


def _validate_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")


def _validate_evidence_hashes(values: tuple[str, ...]) -> None:
    if not isinstance(values, tuple) or len(set(values)) != len(values):
        raise ValueError("evidence hashes must be a unique tuple")
    if not all(isinstance(value, str) and _EVIDENCE_HASH.fullmatch(value) for value in values):
        raise ValueError("evidence hashes must be SHA-256 hex digests")


__all__ = [
    "derive_walk_forward_model_id",
    "ModelRecord",
    "ModelRegistry",
    "ModelStatus",
    "TransitionEvent",
    "load_walk_forward_evidence",
    "validate_model_id",
    "validate_walk_forward_evidence",
]
