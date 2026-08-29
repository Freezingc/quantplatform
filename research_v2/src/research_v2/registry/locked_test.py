"""Fail-closed contracts for the one-shot 2026 locked-test workflow."""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import io
import json
import math
import os
import re
import secrets
import uuid
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from numbers import Integral, Real
from pathlib import Path

from research_v2.datasets.builder import LockedTestAccess
from research_v2.evaluation.gates import ScoreHealthEvidence
from research_v2.monitoring.shadow import (
    ShadowEvidence,
    ShadowEvidenceVerifier,
    ShadowQualificationStatus,
)
from research_v2.registry.model_registry import (
    ModelRegistry,
    ModelStatus,
    _issue_registry_workflow_proof,
    validate_model_id,
)

_HASH = re.compile(r"^[0-9a-f]{64}$")
_REQUIRED_GATES = ("G1", "G2", "G3", "G4", "G5", "G6")
_DEFAULT_CAPABILITY_TTL = timedelta(hours=1)
_MATURITY_EVIDENCE_KEY_ENV = "QRV2_LOCKED_TEST_MATURITY_HMAC_KEY"
_OPERATOR_IDENTITY_KEY_ENV = "QRV2_LOCKED_TEST_IDENTITY_HMAC_KEY"


@dataclass(frozen=True)
class LockedTestPreregistration:
    """Frozen inputs for the sole authorized 2026 label evaluation."""

    model_id: str
    lineage_id: str
    seed_artifact_hashes: tuple[str, ...]
    artifact_bundle_hash: str
    final_train_manifest_hash: str
    final_validation_manifest_hash: str
    factor_set_id: str
    preprocessor_hash: str
    universe_id: str
    label_definition_hash: str
    evaluation_code_hash: str
    required_gates: tuple[str, ...]
    command: tuple[str, ...]
    applicant_role: str
    approver_role: str

    def __post_init__(self) -> None:
        validate_model_id(self.model_id)
        _validate_text(self.lineage_id, "lineage_id")
        _validate_hashes(self.seed_artifact_hashes, "seed_artifact_hashes", exact_count=3)
        _validate_hash(self.artifact_bundle_hash, "artifact_bundle_hash")
        _validate_hash(self.final_train_manifest_hash, "final_train_manifest_hash")
        _validate_hash(self.final_validation_manifest_hash, "final_validation_manifest_hash")
        _validate_text(self.factor_set_id, "factor_set_id")
        _validate_hash(self.preprocessor_hash, "preprocessor_hash")
        _validate_text(self.universe_id, "universe_id")
        _validate_hash(self.label_definition_hash, "label_definition_hash")
        _validate_hash(self.evaluation_code_hash, "evaluation_code_hash")
        if self.required_gates != _REQUIRED_GATES:
            raise ValueError("locked-test preregistration requires exactly G1-G6")
        if not self.command or not all(isinstance(value, str) and value for value in self.command):
            raise ValueError("locked-test preregistration command is invalid")
        _validate_text(self.applicant_role, "applicant_role")
        _validate_text(self.approver_role, "approver_role")
        if self.applicant_role == self.approver_role:
            raise ValueError("locked-test preregistration roles must be distinct")

    @property
    def preregistration_hash(self) -> str:
        return hashlib.sha256(_canonical_json(self.payload()).encode("utf-8")).hexdigest()

    def payload(self) -> dict[str, object]:
        return {
            "applicant_role": self.applicant_role,
            "approver_role": self.approver_role,
            "artifact_bundle_hash": self.artifact_bundle_hash,
            "command": list(self.command),
            "evaluation_code_hash": self.evaluation_code_hash,
            "factor_set_id": self.factor_set_id,
            "final_train_manifest_hash": self.final_train_manifest_hash,
            "final_validation_manifest_hash": self.final_validation_manifest_hash,
            "label_definition_hash": self.label_definition_hash,
            "lineage_id": self.lineage_id,
            "model_id": self.model_id,
            "preprocessor_hash": self.preprocessor_hash,
            "required_gates": list(self.required_gates),
            "seed_artifact_hashes": list(self.seed_artifact_hashes),
            "universe_id": self.universe_id,
        }


@dataclass(frozen=True)
class IssuedPreregistration:
    preregistration: LockedTestPreregistration
    preregistration_hash: str
    signature: str

    def to_json(self) -> str:
        return _canonical_json(
            {
                "payload": self.preregistration.payload(),
                "preregistration_hash": self.preregistration_hash,
                "signature": self.signature,
            }
        ) + "\n"


@dataclass(frozen=True)
class LockedTestOperatorIdentity:
    """Signed operator identity issued by the workflow identity service."""

    identity_id: str
    actor_id: str
    role: str
    signature: str

    def payload(self) -> dict[str, object]:
        return {"actor_id": self.actor_id, "role": self.role}

    def to_json(self) -> str:
        return _canonical_json(
            {
                "identity_id": self.identity_id,
                "payload": self.payload(),
                "signature": self.signature,
            }
        ) + "\n"


class LockedTestOperatorIdentityStore:
    """Issue and verify immutable authenticated identities for separated workflow roles."""

    def __init__(self, root: Path, *, signing_key: bytes) -> None:
        if not isinstance(root, Path):
            raise ValueError("locked-test operator identity root is invalid")
        if not isinstance(signing_key, bytes) or not signing_key:
            raise ValueError("locked-test operator identity signing_key must be non-empty bytes")
        self._root = root
        self._signing_key = signing_key

    @classmethod
    def from_environment(cls, root: Path) -> LockedTestOperatorIdentityStore:
        """Open the trusted identity store using the service-injected verification key."""
        secret = os.environ.get(_OPERATOR_IDENTITY_KEY_ENV, "")
        if not secret:
            raise ValueError(
                f"locked-test operator identity requires {_OPERATOR_IDENTITY_KEY_ENV}"
            )
        return cls(root, signing_key=secret.encode("utf-8"))

    def issue(self, *, actor_id: str, role: str) -> LockedTestOperatorIdentity:
        _validate_text(actor_id, "operator actor_id")
        _validate_text(role, "operator role")
        payload = {"actor_id": actor_id, "role": role}
        identity_id = _content_hash(payload)
        identity = LockedTestOperatorIdentity(
            identity_id=identity_id,
            actor_id=actor_id,
            role=role,
            signature=_signature(self._signing_key, identity_id, payload),
        )
        destination = self._root / ".issued" / f"{identity_id}.json"
        try:
            _write_exclusive(destination, identity.to_json())
        except FileExistsError as error:
            try:
                existing = destination.read_text(encoding="utf-8")
            except OSError:
                raise FileExistsError(
                    f"locked-test operator identity is immutable: {identity_id}"
                ) from error
            if existing != identity.to_json():
                raise FileExistsError(
                    f"locked-test operator identity is immutable: {identity_id}"
                ) from error
        return identity

    def verify(self, identity: LockedTestOperatorIdentity) -> LockedTestOperatorIdentity:
        if not isinstance(identity, LockedTestOperatorIdentity):
            raise ValueError("locked-test operator identity is invalid")
        _validate_text(identity.actor_id, "operator actor_id")
        _validate_text(identity.role, "operator role")
        payload = identity.payload()
        if identity.identity_id != _content_hash(payload):
            raise ValueError("locked-test operator identity content does not match its identity_id")
        expected = _signature(self._signing_key, identity.identity_id, payload)
        if not hmac.compare_digest(identity.signature, expected):
            raise ValueError("locked-test operator identity signature verification failed")
        try:
            stored = (self._root / ".issued" / f"{identity.identity_id}.json").read_text(
                encoding="utf-8"
            )
        except OSError as error:
            raise ValueError("locked-test operator identity has not been issued") from error
        if stored != identity.to_json():
            raise ValueError("locked-test operator identity does not match its immutable record")
        return identity


@dataclass(frozen=True)
class LockedTestUnlockRequest:
    """A signed request made after the locked year and label watermark have matured."""

    request_id: str
    preregistration: IssuedPreregistration
    applicant: str
    label_watermark_id: str
    requested_at: datetime
    signature: str

    def payload(self) -> dict[str, object]:
        preregistration = self.preregistration.preregistration
        return {
            "applicant": self.applicant,
            "label_watermark_id": self.label_watermark_id,
            "lineage_id": preregistration.lineage_id,
            "model_id": preregistration.model_id,
            "preregistration_hash": self.preregistration.preregistration_hash,
            "requested_at": _timestamp(self.requested_at),
        }

    def to_json(self) -> str:
        return _canonical_json(
            {
                "payload": self.payload(),
                "request_id": self.request_id,
                "signature": self.signature,
            }
        ) + "\n"


@dataclass(frozen=True)
class LockedTestCapability:
    """Bearer capability which can unlock the 2026 labels exactly once before expiry."""

    capability_id: str
    model_id: str
    lineage_id: str
    preregistration_hash: str
    label_watermark_id: str
    expires_at: datetime
    authorization_token: str
    signature: str

    def payload(self) -> dict[str, object]:
        return {
            "expires_at": _timestamp(self.expires_at),
            "label_watermark_id": self.label_watermark_id,
            "lineage_id": self.lineage_id,
            "model_id": self.model_id,
            "preregistration_hash": self.preregistration_hash,
            "token_hash": hashlib.sha256(self.authorization_token.encode("utf-8")).hexdigest(),
        }

    def to_json(self) -> str:
        """Serialize only the token hash so audit storage never contains the bearer secret."""
        return _canonical_json(
            {
                "capability_id": self.capability_id,
                "payload": self.payload(),
                "signature": self.signature,
            }
        ) + "\n"


@dataclass(frozen=True)
class LockedTestAuditEvent:
    """Append-only audit fact; it deliberately excludes labels, metrics, and bearer tokens."""

    audit_id: str
    action: str
    outcome: str
    model_id: str | None
    capability_id: str | None
    recorded_at: datetime


@dataclass(frozen=True)
class LockedTestMetrics:
    """Aggregate-only 2026 measurements returned by the dedicated evaluator adapter."""

    mean_rank_ic: float
    hac_t_statistic: float
    candidate_delta_rank_ic: float
    valid_test_day_count: int
    score_health: tuple[ScoreHealthEvidence, ...]

    def __post_init__(self) -> None:
        _validate_finite_number(self.mean_rank_ic, "mean_rank_ic")
        _validate_finite_number(self.hac_t_statistic, "hac_t_statistic")
        _validate_finite_number(self.candidate_delta_rank_ic, "candidate_delta_rank_ic")
        if (
            not isinstance(self.valid_test_day_count, Integral)
            or isinstance(self.valid_test_day_count, bool)
            or self.valid_test_day_count <= 0
        ):
            raise ValueError("locked-test valid_test_day_count is invalid")
        if (
            not isinstance(self.score_health, tuple)
            or len(self.score_health) != self.valid_test_day_count
            or not all(isinstance(value, ScoreHealthEvidence) for value in self.score_health)
        ):
            raise ValueError("locked-test score_health is incomplete")

    def aggregate(self) -> LockedTestAggregate:
        """Calculate the only score-health value disclosed outside the isolated evaluator."""
        return LockedTestAggregate(
            mean_rank_ic=float(self.mean_rank_ic),
            hac_t_statistic=float(self.hac_t_statistic),
            candidate_delta_rank_ic=float(self.candidate_delta_rank_ic),
            valid_test_day_count=int(self.valid_test_day_count),
            score_health_pass_ratio=sum(item.passes() for item in self.score_health)
            / self.valid_test_day_count,
        )


@dataclass(frozen=True)
class LockedTestAggregate:
    """Result-safe metrics with no labels, per-security scores, or daily performance series."""

    mean_rank_ic: float
    hac_t_statistic: float
    candidate_delta_rank_ic: float
    valid_test_day_count: int
    score_health_pass_ratio: float

    def payload(self) -> dict[str, object]:
        return {
            "candidate_delta_rank_ic": self.candidate_delta_rank_ic,
            "hac_t_statistic": self.hac_t_statistic,
            "mean_rank_ic": self.mean_rank_ic,
            "score_health_pass_ratio": self.score_health_pass_ratio,
            "valid_test_day_count": self.valid_test_day_count,
        }


@dataclass(frozen=True)
class LockedTestResult:
    """Signed final verdict from the only allowed 2026 evaluation attempt."""

    result_id: str
    model_id: str
    capability_id: str
    preregistration_hash: str
    artifact_bundle_hash: str
    passed: bool
    aggregate: LockedTestAggregate | None
    evaluator_actor: str
    evaluated_at: datetime
    signature: str

    def payload(self) -> dict[str, object]:
        return {
            "aggregate": self.aggregate.payload() if self.aggregate is not None else None,
            "artifact_bundle_hash": self.artifact_bundle_hash,
            "capability_id": self.capability_id,
            "evaluator_actor": self.evaluator_actor,
            "evaluated_at": _timestamp(self.evaluated_at),
            "model_id": self.model_id,
            "passed": self.passed,
            "preregistration_hash": self.preregistration_hash,
        }

    def to_json(self) -> str:
        return _canonical_json(
            {
                "payload": self.payload(),
                "result_id": self.result_id,
                "signature": self.signature,
            }
        ) + "\n"


class LockedTestResultVerifier:
    """Verify the immutable aggregate result before a deployment approval can use it."""

    def __init__(self, root: Path, *, verification_key: bytes) -> None:
        if not isinstance(root, Path):
            raise ValueError("locked-test result root is invalid")
        if not isinstance(verification_key, bytes) or not verification_key:
            raise ValueError("locked-test result verification_key must be non-empty bytes")
        self._root = root
        self._verification_key = verification_key

    def verify(self, result: LockedTestResult) -> LockedTestResult:
        if not isinstance(result, LockedTestResult):
            raise ValueError("locked-test result is invalid")
        validate_model_id(result.model_id)
        _validate_hash(result.capability_id, "result capability_id")
        _validate_hash(result.preregistration_hash, "result preregistration_hash")
        _validate_hash(result.artifact_bundle_hash, "result artifact_bundle_hash")
        _validate_text(result.evaluator_actor, "result evaluator_actor")
        _validate_datetime(result.evaluated_at, "result evaluated_at")
        if not isinstance(result.passed, bool):
            raise ValueError("locked-test result passed flag is invalid")
        if result.passed and result.aggregate is None:
            raise ValueError("passed locked-test result must contain aggregate metrics")
        payload = result.payload()
        if result.result_id != _content_hash(payload):
            raise ValueError("locked-test result content does not match its result_id")
        expected = _signature(self._verification_key, result.result_id, payload)
        if not hmac.compare_digest(result.signature, expected):
            raise ValueError("locked-test result signature verification failed")
        try:
            stored = (self._root / "results" / f"{result.result_id}.json").read_text(
                encoding="utf-8"
            )
        except OSError as error:
            raise ValueError("locked-test result has not been issued") from error
        if stored != result.to_json():
            raise ValueError("locked-test result does not match its immutable record")
        return result


@dataclass(frozen=True)
class ProductionDeploymentApproval:
    """Signed independent authorization to move one passed model into production."""

    approval_id: str
    model_id: str
    locked_test_result_id: str
    artifact_bundle_hash: str
    actor: str
    signature: str

    def payload(self) -> dict[str, object]:
        return {
            "actor": self.actor,
            "artifact_bundle_hash": self.artifact_bundle_hash,
            "locked_test_result_id": self.locked_test_result_id,
            "model_id": self.model_id,
        }

    def to_json(self) -> str:
        return _canonical_json(
            {
                "approval_id": self.approval_id,
                "payload": self.payload(),
                "signature": self.signature,
            }
        ) + "\n"


class ProductionDeploymentApprovalStore:
    """Issue the single independent approval required after a passed locked evaluation."""

    def __init__(
        self,
        root: Path,
        *,
        registry: ModelRegistry,
        result_root: Path,
        result_verification_key: bytes,
        signing_key: bytes,
    ) -> None:
        if not isinstance(root, Path):
            raise ValueError("production approval root is invalid")
        if not isinstance(registry, ModelRegistry):
            raise ValueError("production approval registry is invalid")
        if not isinstance(signing_key, bytes) or not signing_key:
            raise ValueError("production approval signing_key must be non-empty bytes")
        self._root = root
        self._registry = registry
        self._result_verifier = LockedTestResultVerifier(
            result_root,
            verification_key=result_verification_key,
        )
        self._signing_key = signing_key

    def issue(
        self,
        result: LockedTestResult,
        *,
        actor: str,
    ) -> ProductionDeploymentApproval:
        """Verify a passing final result and then atomically preserve an approval record."""
        verified = self._result_verifier.verify(result)
        _validate_text(actor, "production approver")
        if not verified.passed:
            raise ValueError("production approval requires a passed locked-test result")
        if actor == verified.evaluator_actor:
            raise ValueError("production approval must be independent of the locked-test evaluator")
        record = self._registry.get(verified.model_id)
        if record.status is not ModelStatus.LOCKED_TEST_PASSED:
            raise ValueError("production approval requires LOCKED_TEST_PASSED status")
        payload = {
            "actor": actor,
            "artifact_bundle_hash": verified.artifact_bundle_hash,
            "locked_test_result_id": verified.result_id,
            "model_id": verified.model_id,
        }
        approval_id = _content_hash(payload)
        approval = ProductionDeploymentApproval(
            approval_id=approval_id,
            model_id=verified.model_id,
            locked_test_result_id=verified.result_id,
            artifact_bundle_hash=verified.artifact_bundle_hash,
            actor=actor,
            signature=_signature(self._signing_key, approval_id, payload),
        )
        try:
            _write_exclusive(self._root / ".issued" / f"{approval_id}.json", approval.to_json())
        except FileExistsError as error:
            raise FileExistsError(
                f"production deployment approval is immutable: {approval_id}"
            ) from error
        self._registry._approve_production(
            _issue_registry_workflow_proof(
                operation="APPROVE_PRODUCTION",
                model_id=verified.model_id,
                actor=actor,
                evidence_hash=approval.approval_id,
            )
        )
        return approval


class ProductionDeploymentApprovalVerifier:
    """Verify a production approval before an active pointer can deploy its model artifact."""

    def __init__(self, root: Path, *, verification_key: bytes) -> None:
        if not isinstance(root, Path):
            raise ValueError("production approval root is invalid")
        if not isinstance(verification_key, bytes) or not verification_key:
            raise ValueError("production approval verification_key must be non-empty bytes")
        self._root = root
        self._verification_key = verification_key

    @classmethod
    def from_environment(cls, root: Path) -> ProductionDeploymentApprovalVerifier:
        secret = os.environ.get("QRV2_PRODUCTION_APPROVAL_HMAC_KEY", "")
        if not secret:
            raise ValueError("production approval requires QRV2_PRODUCTION_APPROVAL_HMAC_KEY")
        return cls(root, verification_key=secret.encode("utf-8"))

    def verify(self, approval: ProductionDeploymentApproval) -> ProductionDeploymentApproval:
        if not isinstance(approval, ProductionDeploymentApproval):
            raise ValueError("production deployment approval is invalid")
        validate_model_id(approval.model_id)
        _validate_hash(approval.locked_test_result_id, "production locked_test_result_id")
        _validate_hash(approval.artifact_bundle_hash, "production artifact_bundle_hash")
        _validate_text(approval.actor, "production actor")
        payload = approval.payload()
        if approval.approval_id != _content_hash(payload):
            raise ValueError(
                "production deployment approval content does not match its approval_id"
            )
        expected = _signature(self._verification_key, approval.approval_id, payload)
        if not hmac.compare_digest(approval.signature, expected):
            raise ValueError("production deployment approval signature verification failed")
        try:
            stored = (self._root / ".issued" / f"{approval.approval_id}.json").read_text(
                encoding="utf-8"
            )
        except OSError as error:
            raise ValueError("production deployment approval has not been issued") from error
        if stored != approval.to_json():
            raise ValueError("production deployment approval does not match its immutable record")
        return approval


class LockedTestPreregistrationStore:
    """Trusted issuer for immutable, result-free locked-test preregistrations."""

    def __init__(self, root: Path, *, signing_key: bytes) -> None:
        if not isinstance(signing_key, bytes) or not signing_key:
            raise ValueError("locked-test preregistration signing_key must be non-empty bytes")
        self._root = root
        self._signing_key = signing_key

    def issue(self, preregistration: LockedTestPreregistration) -> IssuedPreregistration:
        if not isinstance(preregistration, LockedTestPreregistration):
            raise ValueError("locked-test preregistration is invalid")
        preregistration_hash = preregistration.preregistration_hash
        issued = IssuedPreregistration(
            preregistration=preregistration,
            preregistration_hash=preregistration_hash,
            signature=_signature(
                self._signing_key,
                preregistration_hash,
                preregistration.payload(),
            ),
        )
        destination = self._root / ".issued" / f"{preregistration_hash}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            _write_exclusive(destination, issued.to_json())
        except FileExistsError as error:
            raise FileExistsError(
                f"locked-test preregistration is immutable: {preregistration_hash}"
            ) from error
        return issued

    def verify(self, issued: IssuedPreregistration) -> LockedTestPreregistration:
        if not isinstance(issued, IssuedPreregistration):
            raise ValueError("locked-test preregistration is invalid")
        preregistration = issued.preregistration
        if issued.preregistration_hash != preregistration.preregistration_hash:
            raise ValueError("locked-test preregistration hash is invalid")
        expected = _signature(
            self._signing_key,
            issued.preregistration_hash,
            preregistration.payload(),
        )
        if not hmac.compare_digest(issued.signature, expected):
            raise ValueError("locked-test preregistration signature verification failed")
        path = self._root / ".issued" / f"{issued.preregistration_hash}.json"
        try:
            stored = path.read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError("locked-test preregistration has not been issued") from error
        if stored != issued.to_json():
            raise ValueError("locked-test preregistration does not match its immutable record")
        return preregistration


@dataclass(frozen=True)
class LabelMaturityWatermark:
    watermark_id: str
    exit_date: date
    adjusted_open_stable: bool
    reconciled: bool
    dq_passed: bool
    published: bool

    def __post_init__(self) -> None:
        _validate_hash(self.watermark_id, "label watermark_id")
        if not isinstance(self.exit_date, date):
            raise ValueError("label watermark exit_date is invalid")
        for name in ("adjusted_open_stable", "reconciled", "dq_passed", "published"):
            if not isinstance(getattr(self, name), bool):
                raise ValueError(f"label watermark {name} is invalid")


@dataclass(frozen=True)
class LockedTestTradingCalendar:
    """Signed market-calendar snapshot used to calculate the final 2026 exit date."""

    calendar_id: str
    trading_dates: tuple[date, ...]
    signature: str

    def payload(self) -> dict[str, object]:
        return {"trading_dates": [value.isoformat() for value in self.trading_dates]}

    def to_json(self) -> str:
        return _canonical_json(
            {
                "calendar_id": self.calendar_id,
                "payload": self.payload(),
                "signature": self.signature,
            }
        ) + "\n"


@dataclass(frozen=True)
class IssuedLabelMaturityWatermark:
    """Signed data-quality watermark supplied by the label service."""

    watermark: LabelMaturityWatermark
    signature: str

    def payload(self) -> dict[str, object]:
        return {
            "adjusted_open_stable": self.watermark.adjusted_open_stable,
            "dq_passed": self.watermark.dq_passed,
            "exit_date": self.watermark.exit_date.isoformat(),
            "published": self.watermark.published,
            "reconciled": self.watermark.reconciled,
            "watermark_id": self.watermark.watermark_id,
        }

    def to_json(self) -> str:
        return _canonical_json(
            {
                "payload": self.payload(),
                "signature": self.signature,
                "watermark_id": self.watermark.watermark_id,
            }
        ) + "\n"


class LockedTestMaturityEvidenceStore:
    """Issue and verify immutable calendar and label-watermark facts for unlocking."""

    def __init__(self, root: Path, *, signing_key: bytes) -> None:
        if not isinstance(root, Path):
            raise ValueError("locked-test maturity evidence root is invalid")
        if not isinstance(signing_key, bytes) or not signing_key:
            raise ValueError("locked-test maturity evidence signing_key must be non-empty bytes")
        self._root = root
        self._signing_key = signing_key

    @classmethod
    def from_environment(cls, root: Path) -> LockedTestMaturityEvidenceStore:
        secret = os.environ.get(_MATURITY_EVIDENCE_KEY_ENV, "")
        if not secret:
            raise ValueError(f"locked-test maturity evidence requires {_MATURITY_EVIDENCE_KEY_ENV}")
        return cls(root, signing_key=secret.encode("utf-8"))

    def issue_calendar(self, trading_dates: tuple[date, ...]) -> LockedTestTradingCalendar:
        calendar = _validated_trading_dates(trading_dates)
        payload = {"trading_dates": [value.isoformat() for value in calendar]}
        calendar_id = _content_hash(payload)
        issued = LockedTestTradingCalendar(
            calendar_id=calendar_id,
            trading_dates=calendar,
            signature=_signature(self._signing_key, calendar_id, payload),
        )
        try:
            _write_exclusive(self._calendar_path(calendar_id), issued.to_json())
        except FileExistsError as error:
            raise FileExistsError(f"locked-test calendar is immutable: {calendar_id}") from error
        return issued

    def issue_watermark(self, watermark: LabelMaturityWatermark) -> IssuedLabelMaturityWatermark:
        if not isinstance(watermark, LabelMaturityWatermark):
            raise ValueError("locked-test label watermark is invalid")
        payload = {
            "adjusted_open_stable": watermark.adjusted_open_stable,
            "dq_passed": watermark.dq_passed,
            "exit_date": watermark.exit_date.isoformat(),
            "published": watermark.published,
            "reconciled": watermark.reconciled,
            "watermark_id": watermark.watermark_id,
        }
        issued = IssuedLabelMaturityWatermark(
            watermark=watermark,
            signature=_signature(self._signing_key, watermark.watermark_id, payload),
        )
        try:
            _write_exclusive(self._watermark_path(watermark.watermark_id), issued.to_json())
        except FileExistsError as error:
            raise FileExistsError(
                f"locked-test label watermark is immutable: {watermark.watermark_id}"
            ) from error
        return issued

    def verify_calendar(self, calendar: LockedTestTradingCalendar) -> tuple[date, ...]:
        if not isinstance(calendar, LockedTestTradingCalendar):
            raise ValueError("locked-test calendar is invalid")
        dates = _validated_trading_dates(calendar.trading_dates)
        payload = {"trading_dates": [value.isoformat() for value in dates]}
        if calendar.calendar_id != _content_hash(payload):
            raise ValueError("locked-test calendar content does not match its calendar_id")
        expected = _signature(self._signing_key, calendar.calendar_id, payload)
        if not hmac.compare_digest(calendar.signature, expected):
            raise ValueError("locked-test calendar signature verification failed")
        try:
            stored = self._calendar_path(calendar.calendar_id).read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError("locked-test calendar has not been issued") from error
        if stored != calendar.to_json():
            raise ValueError("locked-test calendar does not match its immutable record")
        return dates

    def verify_watermark(self, issued: IssuedLabelMaturityWatermark) -> LabelMaturityWatermark:
        if not isinstance(issued, IssuedLabelMaturityWatermark):
            raise ValueError("locked-test label watermark is invalid")
        watermark = issued.watermark
        payload = issued.payload()
        expected = _signature(self._signing_key, watermark.watermark_id, payload)
        if not hmac.compare_digest(issued.signature, expected):
            raise ValueError("locked-test label watermark signature verification failed")
        try:
            stored = self._watermark_path(watermark.watermark_id).read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError("locked-test label watermark has not been issued") from error
        if stored != issued.to_json():
            raise ValueError("locked-test label watermark does not match its immutable record")
        return watermark

    def _calendar_path(self, calendar_id: str) -> Path:
        return self._root / "calendars" / f"{calendar_id}.json"

    def _watermark_path(self, watermark_id: str) -> Path:
        return self._root / "watermarks" / f"{watermark_id}.json"


@dataclass(frozen=True)
class LockedTestMaturity:
    last_2026_trading_date: date
    required_exit_date: date
    label_watermark_id: str


class LockedTestMaturityGuard:
    """Calculate the required maturity date from the trading calendar, never a day constant."""

    def __init__(
        self,
        calendar: LockedTestTradingCalendar,
        *,
        evidence_store: LockedTestMaturityEvidenceStore,
    ) -> None:
        if not isinstance(evidence_store, LockedTestMaturityEvidenceStore):
            raise ValueError("locked-test maturity evidence store is invalid")
        self._trading_dates = evidence_store.verify_calendar(calendar)
        self._evidence_store = evidence_store

    def validate(
        self,
        current_date: date,
        issued_watermark: IssuedLabelMaturityWatermark,
    ) -> LockedTestMaturity:
        if not isinstance(current_date, date):
            raise ValueError("locked-test maturity inputs are invalid")
        watermark = self._evidence_store.verify_watermark(issued_watermark)
        positions = [index for index, value in enumerate(self._trading_dates) if value.year == 2026]
        if not positions or positions[-1] + 6 >= len(self._trading_dates):
            raise ValueError("locked-test calendar does not contain six subsequent trading days")
        last_2026 = self._trading_dates[positions[-1]]
        required_exit = self._trading_dates[positions[-1] + 6]
        if current_date <= required_exit:
            raise ValueError("2026 locked test is not mature")
        if watermark.exit_date != required_exit:
            raise ValueError("label watermark does not match the required locked-test exit date")
        if not all(
            (
                watermark.adjusted_open_stable,
                watermark.reconciled,
                watermark.dq_passed,
                watermark.published,
            )
        ):
            raise ValueError("label watermark is not clean, reconciled, and published")
        return LockedTestMaturity(last_2026, required_exit, watermark.watermark_id)


class LockedTestAuthority:
    """Issue, approve, consume, and audit the one permitted 2026 label capability."""

    def __init__(
        self,
        root: Path,
        *,
        preregistration_store: LockedTestPreregistrationStore,
        registry: ModelRegistry,
        maturity_calendar: LockedTestTradingCalendar,
        maturity_evidence_root: Path,
        operator_identity_root: Path,
        signing_key: bytes,
        access_signing_key: bytes,
        clock: Callable[[], datetime] | None = None,
        capability_ttl: timedelta = _DEFAULT_CAPABILITY_TTL,
    ) -> None:
        if not isinstance(root, Path):
            raise ValueError("locked-test authority root is invalid")
        if not isinstance(preregistration_store, LockedTestPreregistrationStore):
            raise ValueError("locked-test preregistration store is invalid")
        if not isinstance(registry, ModelRegistry):
            raise ValueError("locked-test model registry is invalid")
        if not isinstance(signing_key, bytes) or not signing_key:
            raise ValueError("locked-test authority signing_key must be non-empty bytes")
        if not isinstance(access_signing_key, bytes) or not access_signing_key:
            raise ValueError("locked-test access_signing_key must be non-empty bytes")
        if not isinstance(capability_ttl, timedelta) or capability_ttl <= timedelta():
            raise ValueError("locked-test capability_ttl must be positive")
        self._root = root
        self._preregistration_store = preregistration_store
        self._registry = registry
        self._maturity_guard = LockedTestMaturityGuard(
            maturity_calendar,
            evidence_store=LockedTestMaturityEvidenceStore.from_environment(maturity_evidence_root),
        )
        self._operator_identities = LockedTestOperatorIdentityStore.from_environment(
            operator_identity_root
        )
        self._signing_key = signing_key
        self._access_signing_key = access_signing_key
        self._clock = clock or (lambda: datetime.now(UTC))
        self._capability_ttl = capability_ttl
        self._audit_events: list[LockedTestAuditEvent] = []

    def request_unlock(
        self,
        preregistration: IssuedPreregistration,
        *,
        applicant: LockedTestOperatorIdentity,
        current_date: date,
        watermark: IssuedLabelMaturityWatermark,
    ) -> LockedTestUnlockRequest:
        """Record an immutable unlock request after all non-label preconditions are met."""
        model_id = _issued_model_id(preregistration)
        try:
            frozen = self._preregistration_store.verify(preregistration)
            verified_applicant = self._operator_identities.verify(applicant)
            if verified_applicant.role != frozen.applicant_role:
                raise ValueError("locked-test applicant does not match the preregistered role")
            record = self._registry.get(frozen.model_id)
            if record.status is not ModelStatus.SHADOW:
                raise ValueError("locked-test unlock requires a SHADOW model")
            if record.lineage_id != frozen.lineage_id:
                raise ValueError(
                    "locked-test preregistration lineage does not match the registered model"
                )
            if self.is_lineage_exposed(frozen.lineage_id):
                raise ValueError("locked-test model lineage is already LOCKED_EXPOSED")
            maturity = self._maturity_guard.validate(current_date, watermark)
            requested_at = self._now()
            payload = {
                "applicant": verified_applicant.actor_id,
                "label_watermark_id": maturity.label_watermark_id,
                "lineage_id": frozen.lineage_id,
                "model_id": frozen.model_id,
                "preregistration_hash": preregistration.preregistration_hash,
                "requested_at": _timestamp(requested_at),
            }
            request_id = _content_hash(payload)
            request = LockedTestUnlockRequest(
                request_id=request_id,
                preregistration=preregistration,
                applicant=verified_applicant.actor_id,
                label_watermark_id=maturity.label_watermark_id,
                requested_at=requested_at,
                signature=_signature(self._signing_key, request_id, payload),
            )
            _write_exclusive(self._request_path(request_id), request.to_json())
        except (FileExistsError, ValueError) as error:
            self._audit("UNLOCK_REQUEST", "DENIED", model_id=model_id, capability_id=None)
            raise error
        self._audit("UNLOCK_REQUEST", "GRANTED", model_id=model_id, capability_id=None)
        return request

    def approve(
        self,
        request: LockedTestUnlockRequest,
        *,
        approver: LockedTestOperatorIdentity,
        shadow_evidence: ShadowEvidence,
        shadow_evidence_root: Path,
    ) -> LockedTestCapability:
        """Approve one request only after a separate operator verifies signed shadow evidence."""
        model_id = _request_model_id(request)
        try:
            frozen = self._verify_request(request)
            verified_approver = self._operator_identities.verify(approver)
            if request.applicant == verified_approver.actor_id:
                raise ValueError("locked-test requester and approver must be different")
            if verified_approver.role != frozen.approver_role:
                raise ValueError("locked-test approver does not match the preregistered role")
            record = self._registry.get(frozen.model_id)
            if record.status is not ModelStatus.SHADOW:
                raise ValueError("locked-test approval requires a SHADOW model")
            if record.lineage_id != frozen.lineage_id:
                raise ValueError(
                    "locked-test preregistration lineage does not match the registered model"
                )
            if self.is_lineage_exposed(frozen.lineage_id):
                raise ValueError("locked-test model lineage is already LOCKED_EXPOSED")
            self._verify_shadow_evidence(frozen.model_id, shadow_evidence, shadow_evidence_root)
            approval_hash = _content_hash(
                {
                    "approver": verified_approver.actor_id,
                    "preregistration_hash": request.preregistration.preregistration_hash,
                    "request_id": request.request_id,
                    "shadow_evidence_id": shadow_evidence.evidence_id,
                }
            )
            capability = self._new_capability(
                model_id=frozen.model_id,
                lineage_id=frozen.lineage_id,
                preregistration_hash=request.preregistration.preregistration_hash,
                label_watermark_id=request.label_watermark_id,
            )
            _write_exclusive(self._capability_path(capability.capability_id), capability.to_json())
            self._registry._approve_locked_test(
                _issue_registry_workflow_proof(
                    operation="APPROVE_LOCKED_TEST",
                    model_id=frozen.model_id,
                    actor=verified_approver.actor_id,
                    evidence_hash=approval_hash,
                )
            )
        except (FileExistsError, ValueError) as error:
            self._audit("UNLOCK_APPROVAL", "DENIED", model_id=model_id, capability_id=None)
            raise error
        self._audit(
            "UNLOCK_APPROVAL",
            "GRANTED",
            model_id=model_id,
            capability_id=capability.capability_id,
        )
        return capability

    def consume(
        self,
        capability: LockedTestCapability,
        preregistration: IssuedPreregistration,
        *,
        consumer: str,
        result_root: Path | None = None,
    ) -> LockedTestAccess:
        """Atomically consume before any label reader can receive the authorization identifier."""
        model_id = capability.model_id if isinstance(capability, LockedTestCapability) else None
        capability_id = (
            capability.capability_id if isinstance(capability, LockedTestCapability) else None
        )
        try:
            self._verify_capability(capability)
            frozen = self._preregistration_store.verify(preregistration)
            _validate_text(consumer, "locked-test consumer")
            if result_root is not None and not isinstance(result_root, Path):
                raise ValueError("locked-test result_root is invalid")
            if capability.preregistration_hash != preregistration.preregistration_hash:
                raise ValueError("locked-test capability does not match the preregistration hash")
            if capability.model_id != frozen.model_id or capability.lineage_id != frozen.lineage_id:
                raise ValueError("locked-test capability does not match the preregistration")
            if self._now() >= capability.expires_at:
                raise ValueError("locked-test capability has expired")
            consumed_path = self._consumed_path(capability.capability_id)
            if consumed_path.exists():
                raise ValueError("locked-test capability is already consumed")
            record = self._registry.get(capability.model_id)
            if record.status is not ModelStatus.LOCKED_TEST_APPROVED:
                raise ValueError("locked-test capability requires LOCKED_TEST_APPROVED status")
            if record.lineage_id != capability.lineage_id:
                raise ValueError(
                    "locked-test capability lineage does not match the registered model"
                )
            with _exclusive_lock(consumed_path.with_suffix(".lock")):
                if consumed_path.exists():
                    raise ValueError("locked-test capability is already consumed")
                if self.is_lineage_exposed(capability.lineage_id):
                    raise ValueError("locked-test model lineage is already LOCKED_EXPOSED")
                # Persist recovery before any irreversible exposure so a process death fails closed.
                if result_root is not None:
                    self._write_pending_evaluation(capability, consumer, result_root)
                self._mark_lineage_exposed(capability)
                consumption = {
                    "capability_id": capability.capability_id,
                    "consumer": consumer,
                    "consumed_at": _timestamp(self._now()),
                    "model_id": capability.model_id,
                }
                _write_exclusive(consumed_path, _canonical_json(consumption) + "\n")
        except (FileExistsError, ValueError) as error:
            self._audit(
                "CAPABILITY_CONSUME",
                "DENIED",
                model_id=model_id,
                capability_id=capability_id,
            )
            raise error
        self._audit(
            "CAPABILITY_CONSUME",
            "CONSUMED",
            model_id=model_id,
            capability_id=capability_id,
        )
        return LockedTestAccess.issue(
            locked_year=2026,
            authorization_id=capability.capability_id,
            signing_key=self._access_signing_key,
        )

    def is_lineage_exposed(self, lineage_id: str) -> bool:
        """Return whether any candidate in this lineage has already read 2026 labels."""
        _validate_text(lineage_id, "lineage_id")
        path = self._exposure_path(lineage_id)
        if not path.exists():
            return False
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("locked-test exposure record cannot be verified") from error
        if not isinstance(document, dict):
            raise ValueError("locked-test exposure record is invalid")
        payload = document.get("payload")
        exposure_id = document.get("exposure_id")
        signature = document.get("signature")
        if (
            not isinstance(payload, dict)
            or not isinstance(exposure_id, str)
            or not isinstance(signature, str)
            or payload.get("lineage_id") != lineage_id
            or exposure_id != _content_hash(payload)
        ):
            raise ValueError("locked-test exposure record is invalid")
        expected = _signature(self._signing_key, exposure_id, payload)
        if not hmac.compare_digest(signature, expected):
            raise ValueError("locked-test exposure signature verification failed")
        return True

    def audit_events(self) -> tuple[LockedTestAuditEvent, ...]:
        """Return the current process's append-only, non-sensitive authorization audit trail."""
        return tuple(self._audit_events)

    def verify_preregistration(
        self,
        preregistration: IssuedPreregistration,
    ) -> LockedTestPreregistration:
        """Return only the signature-verified frozen inputs required by the evaluator."""
        return self._preregistration_store.verify(preregistration)

    def recover_incomplete_evaluations(self, *, result_root: Path, actor: str) -> None:
        """Fail-close initiated evaluations that did not persist a terminal workflow marker."""
        if not isinstance(result_root, Path):
            raise ValueError("locked-test result_root is invalid")
        _validate_text(actor, "locked-test recovery actor")
        pending_root = self._root / "pending"
        if not pending_root.exists():
            return
        for path in sorted(pending_root.glob("*.json")):
            pending = self._load_pending_evaluation(path)
            capability_id = pending["capability_id"]
            if (
                pending["result_root"] != str(result_root)
                or self._completed_path(capability_id).exists()
            ):
                continue
            model_id = pending["model_id"]
            if self._registry.get(model_id).status is not ModelStatus.LOCKED_TEST_APPROVED:
                continue
            evidence_hash = _content_hash(
                {
                    "capability_id": capability_id,
                    "model_id": model_id,
                    "outcome": "RECOVERED_CRASH",
                    "preregistration_hash": pending["preregistration_hash"],
                }
            )
            self._registry._complete_locked_test(
                _issue_registry_workflow_proof(
                    operation="COMPLETE_LOCKED_TEST",
                    model_id=model_id,
                    actor=actor,
                    evidence_hash=evidence_hash,
                    passed=False,
                )
            )
            _write_exclusive(
                self._completed_path(capability_id),
                _canonical_json({"outcome": "RECOVERED_CRASH"}) + "\n",
            )
            self._audit(
                "EVALUATION_RECOVERY",
                "FAILED_CLOSED",
                model_id=model_id,
                capability_id=capability_id,
            )

    def finalize_evaluation(self, capability: LockedTestCapability) -> None:
        """Persist that the consumed evaluator invocation reached a terminal state."""
        if not isinstance(capability, LockedTestCapability):
            raise ValueError("locked-test capability is invalid")
        if not self._pending_path(capability.capability_id).exists():
            return
        _write_exclusive(
            self._completed_path(capability.capability_id),
            _canonical_json({"outcome": "TERMINAL"}) + "\n",
        )

    def _verify_request(self, request: LockedTestUnlockRequest) -> LockedTestPreregistration:
        if not isinstance(request, LockedTestUnlockRequest):
            raise ValueError("locked-test unlock request is invalid")
        payload = request.payload()
        if request.request_id != _content_hash(payload):
            raise ValueError("locked-test unlock request content does not match its request_id")
        expected = _signature(self._signing_key, request.request_id, payload)
        if not hmac.compare_digest(request.signature, expected):
            raise ValueError("locked-test unlock request signature verification failed")
        try:
            stored = self._request_path(request.request_id).read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError("locked-test unlock request has not been issued") from error
        if stored != request.to_json():
            raise ValueError("locked-test unlock request does not match its immutable record")
        frozen = self._preregistration_store.verify(request.preregistration)
        if (
            request.label_watermark_id == ""
            or request.preregistration.preregistration_hash != frozen.preregistration_hash
        ):
            raise ValueError("locked-test unlock request does not match preregistration")
        return frozen

    def _verify_capability(self, capability: LockedTestCapability) -> None:
        if not isinstance(capability, LockedTestCapability):
            raise ValueError("locked-test capability is invalid")
        validate_model_id(capability.model_id)
        _validate_text(capability.lineage_id, "lineage_id")
        _validate_hash(capability.preregistration_hash, "capability preregistration_hash")
        _validate_hash(capability.label_watermark_id, "capability label_watermark_id")
        _validate_datetime(capability.expires_at, "capability expires_at")
        _validate_text(capability.authorization_token, "capability authorization_token")
        payload = capability.payload()
        if capability.capability_id != _content_hash(payload):
            raise ValueError("locked-test capability content does not match its capability_id")
        expected = _signature(self._signing_key, capability.capability_id, payload)
        if not hmac.compare_digest(capability.signature, expected):
            raise ValueError("locked-test capability signature verification failed")
        try:
            stored = self._capability_path(capability.capability_id).read_text(encoding="utf-8")
        except OSError as error:
            raise ValueError("locked-test capability has not been issued") from error
        if stored != capability.to_json():
            raise ValueError("locked-test capability does not match its immutable record")

    def _verify_shadow_evidence(
        self,
        model_id: str,
        shadow_evidence: ShadowEvidence,
        shadow_evidence_root: Path,
    ) -> None:
        if not isinstance(shadow_evidence_root, Path):
            raise ValueError("locked-test shadow evidence root is invalid")
        ShadowEvidenceVerifier.from_environment(shadow_evidence_root).verify(shadow_evidence)
        payload = shadow_evidence.payload
        consecutive_errors = payload.get("max_consecutive_psi_errors")
        if (
            payload.get("model_id") != model_id
            or payload.get("qualification_status")
            != ShadowQualificationStatus.READY_FOR_REVIEW.value
            or isinstance(consecutive_errors, bool)
            or not isinstance(consecutive_errors, int)
            or consecutive_errors >= 3
        ):
            raise ValueError("locked-test shadow evidence is not ready for approval")

    def _new_capability(
        self,
        *,
        model_id: str,
        lineage_id: str,
        preregistration_hash: str,
        label_watermark_id: str,
    ) -> LockedTestCapability:
        expires_at = self._now() + self._capability_ttl
        authorization_token = secrets.token_urlsafe(32)
        payload = {
            "expires_at": _timestamp(expires_at),
            "label_watermark_id": label_watermark_id,
            "lineage_id": lineage_id,
            "model_id": model_id,
            "preregistration_hash": preregistration_hash,
            "token_hash": hashlib.sha256(authorization_token.encode("utf-8")).hexdigest(),
        }
        capability_id = _content_hash(payload)
        return LockedTestCapability(
            capability_id=capability_id,
            model_id=model_id,
            lineage_id=lineage_id,
            preregistration_hash=preregistration_hash,
            label_watermark_id=label_watermark_id,
            expires_at=expires_at,
            authorization_token=authorization_token,
            signature=_signature(self._signing_key, capability_id, payload),
        )

    def _mark_lineage_exposed(self, capability: LockedTestCapability) -> None:
        payload = {
            "capability_id": capability.capability_id,
            "exposed_at": _timestamp(self._now()),
            "lineage_id": capability.lineage_id,
            "model_id": capability.model_id,
            "preregistration_hash": capability.preregistration_hash,
        }
        exposure_id = _content_hash(payload)
        document = {
            "exposure_id": exposure_id,
            "payload": payload,
            "signature": _signature(self._signing_key, exposure_id, payload),
        }
        try:
            _write_exclusive(
                self._exposure_path(capability.lineage_id),
                _canonical_json(document) + "\n",
            )
        except FileExistsError as error:
            raise ValueError("locked-test model lineage is already LOCKED_EXPOSED") from error

    def _write_pending_evaluation(
        self,
        capability: LockedTestCapability,
        consumer: str,
        result_root: Path,
    ) -> None:
        payload = {
            "capability_id": capability.capability_id,
            "consumer": consumer,
            "model_id": capability.model_id,
            "preregistration_hash": capability.preregistration_hash,
            "result_root": str(result_root),
        }
        pending_id = _content_hash(payload)
        document = {
            "payload": payload,
            "pending_id": pending_id,
            "signature": _signature(self._signing_key, pending_id, payload),
        }
        _write_exclusive(
            self._pending_path(capability.capability_id),
            _canonical_json(document) + "\n",
        )

    def _load_pending_evaluation(self, path: Path) -> dict[str, str]:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("locked-test pending evaluation cannot be verified") from error
        if not isinstance(document, dict):
            raise ValueError("locked-test pending evaluation is invalid")
        payload = document.get("payload")
        pending_id = document.get("pending_id")
        signature = document.get("signature")
        required = {
            "capability_id",
            "consumer",
            "model_id",
            "preregistration_hash",
            "result_root",
        }
        if (
            not isinstance(payload, dict)
            or set(payload) != required
            or not all(isinstance(value, str) and value for value in payload.values())
            or not isinstance(pending_id, str)
            or not isinstance(signature, str)
            or pending_id != _content_hash(payload)
        ):
            raise ValueError("locked-test pending evaluation is invalid")
        expected = _signature(self._signing_key, pending_id, payload)
        if not hmac.compare_digest(signature, expected):
            raise ValueError("locked-test pending evaluation signature verification failed")
        return {key: value for key, value in payload.items() if isinstance(value, str)}

    def _audit(
        self,
        action: str,
        outcome: str,
        *,
        model_id: str | None,
        capability_id: str | None,
    ) -> None:
        recorded_at = self._now()
        event = LockedTestAuditEvent(
            audit_id=uuid.uuid4().hex,
            action=action,
            outcome=outcome,
            model_id=model_id,
            capability_id=capability_id,
            recorded_at=recorded_at,
        )
        payload = {
            "action": event.action,
            "capability_id": event.capability_id,
            "model_id": event.model_id,
            "outcome": event.outcome,
            "recorded_at": _timestamp(event.recorded_at),
        }
        document = {
            "audit_id": event.audit_id,
            "payload": payload,
            "signature": _signature(self._signing_key, event.audit_id, payload),
        }
        _write_exclusive(self._audit_path(event.audit_id), _canonical_json(document) + "\n")
        self._audit_events.append(event)

    def _now(self) -> datetime:
        now = self._clock()
        _validate_datetime(now, "authority clock")
        return now.astimezone(UTC)

    def _request_path(self, request_id: str) -> Path:
        return self._root / "requests" / f"{request_id}.json"

    def _capability_path(self, capability_id: str) -> Path:
        return self._root / "capabilities" / f"{capability_id}.json"

    def _consumed_path(self, capability_id: str) -> Path:
        return self._root / "consumed" / f"{capability_id}.json"

    def _pending_path(self, capability_id: str) -> Path:
        return self._root / "pending" / f"{capability_id}.json"

    def _completed_path(self, capability_id: str) -> Path:
        return self._root / "completed" / f"{capability_id}.json"

    def _exposure_path(self, lineage_id: str) -> Path:
        lineage_hash = hashlib.sha256(lineage_id.encode("utf-8")).hexdigest()
        return self._root / "exposure" / f"{lineage_hash}.json"

    def _audit_path(self, audit_id: str) -> Path:
        return self._root / "audit" / f"{audit_id}.json"


class LockedTestEvaluator:
    """Run the pre-registered FINAL/TEST once without exposing intermediate label data."""

    def __init__(
        self,
        root: Path,
        *,
        authority: LockedTestAuthority,
        registry: ModelRegistry,
        signing_key: bytes,
        artifact_bundle_hash: str,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        if not isinstance(root, Path):
            raise ValueError("locked-test evaluator root is invalid")
        if not isinstance(authority, LockedTestAuthority):
            raise ValueError("locked-test authority is invalid")
        if not isinstance(registry, ModelRegistry):
            raise ValueError("locked-test model registry is invalid")
        if not isinstance(signing_key, bytes) or not signing_key:
            raise ValueError("locked-test evaluator signing_key must be non-empty bytes")
        _validate_hash(artifact_bundle_hash, "evaluator artifact_bundle_hash")
        self._root = root
        self._authority = authority
        self._registry = registry
        self._signing_key = signing_key
        self._artifact_bundle_hash = artifact_bundle_hash
        self._clock = clock or (lambda: datetime.now(UTC))
        self._authority.recover_incomplete_evaluations(
            result_root=self._root,
            actor="locked-test-recovery",
        )

    def evaluate(
        self,
        capability: LockedTestCapability,
        preregistration: IssuedPreregistration,
        *,
        actor: str,
        metrics_reader: Callable[[LockedTestAccess], LockedTestMetrics],
    ) -> LockedTestResult:
        """Consume first, then return a single signed aggregate verdict or record a failure."""
        _validate_text(actor, "locked-test evaluator actor")
        if not callable(metrics_reader):
            raise ValueError("locked-test metrics_reader is invalid")
        frozen = self._authority.verify_preregistration(preregistration)
        if (
            frozen.model_id != capability.model_id
            or frozen.lineage_id != capability.lineage_id
            or frozen.artifact_bundle_hash != self._artifact_bundle_hash
        ):
            raise ValueError(
                "locked-test evaluator artifact does not match the frozen preregistration"
            )
        access = self._authority.consume(
            capability,
            preregistration,
            consumer=actor,
            result_root=self._root,
        )
        try:
            with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
                metrics = metrics_reader(access)
            if not isinstance(metrics, LockedTestMetrics):
                raise ValueError("locked-test metrics_reader returned invalid aggregate metrics")
            aggregate = metrics.aggregate()
            passed = _passes_locked_test_thresholds(aggregate)
            result = self._write_result(capability, aggregate=aggregate, passed=passed, actor=actor)
            self._registry._complete_locked_test(
                _issue_registry_workflow_proof(
                    operation="COMPLETE_LOCKED_TEST",
                    model_id=capability.model_id,
                    actor=actor,
                    evidence_hash=result.result_id,
                    passed=passed,
                )
            )
            self._authority.finalize_evaluation(capability)
            return result
        except BaseException:
            self._record_crashed_evaluation(capability, actor)
            raise

    def _record_crashed_evaluation(self, capability: LockedTestCapability, actor: str) -> None:
        result_hash = _content_hash(
            {
                "capability_id": capability.capability_id,
                "model_id": capability.model_id,
                "outcome": "CRASHED",
                "preregistration_hash": capability.preregistration_hash,
            }
        )
        try:
            result = self._write_result(capability, aggregate=None, passed=False, actor=actor)
            result_hash = result.result_id
        except (FileExistsError, OSError, ValueError):
            pass
        if self._registry.get(capability.model_id).status is ModelStatus.LOCKED_TEST_APPROVED:
            self._registry._complete_locked_test(
                _issue_registry_workflow_proof(
                    operation="COMPLETE_LOCKED_TEST",
                    model_id=capability.model_id,
                    actor=actor,
                    evidence_hash=result_hash,
                    passed=False,
                )
            )
            self._authority.finalize_evaluation(capability)

    def _write_result(
        self,
        capability: LockedTestCapability,
        *,
        aggregate: LockedTestAggregate | None,
        passed: bool,
        actor: str,
    ) -> LockedTestResult:
        evaluated_at = self._now()
        result = LockedTestResult(
            result_id="",
            model_id=capability.model_id,
            capability_id=capability.capability_id,
            preregistration_hash=capability.preregistration_hash,
            artifact_bundle_hash=self._artifact_bundle_hash,
            passed=passed,
            aggregate=aggregate,
            evaluator_actor=actor,
            evaluated_at=evaluated_at,
            signature="",
        )
        payload = result.payload()
        result_id = _content_hash(payload)
        result = LockedTestResult(
            result_id=result_id,
            model_id=result.model_id,
            capability_id=result.capability_id,
            preregistration_hash=result.preregistration_hash,
            artifact_bundle_hash=result.artifact_bundle_hash,
            passed=result.passed,
            aggregate=result.aggregate,
            evaluator_actor=result.evaluator_actor,
            evaluated_at=result.evaluated_at,
            signature=_signature(self._signing_key, result_id, payload),
        )
        _write_exclusive(self._root / "results" / f"{result_id}.json", result.to_json())
        return result

    def _now(self) -> datetime:
        now = self._clock()
        _validate_datetime(now, "evaluator clock")
        return now.astimezone(UTC)


def _issued_model_id(preregistration: object) -> str | None:
    if isinstance(preregistration, IssuedPreregistration):
        return preregistration.preregistration.model_id
    return None


def _request_model_id(request: object) -> str | None:
    if isinstance(request, LockedTestUnlockRequest):
        return request.preregistration.preregistration.model_id
    return None


def _content_hash(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _timestamp(value: datetime) -> str:
    _validate_datetime(value, "timestamp")
    return value.astimezone(UTC).isoformat()


def _validate_datetime(value: object, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"locked-test {name} must be timezone-aware")


def _validate_finite_number(value: object, name: str) -> None:
    if not isinstance(value, Real) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise ValueError(f"locked-test {name} is invalid")


def _validated_trading_dates(values: tuple[date, ...]) -> tuple[date, ...]:
    calendar = tuple(values)
    if (
        not calendar
        or any(not isinstance(value, date) for value in calendar)
        or calendar != tuple(sorted(calendar))
        or len(set(calendar)) != len(calendar)
    ):
        raise ValueError("locked-test trading calendar is invalid")
    return calendar


def _passes_locked_test_thresholds(aggregate: LockedTestAggregate) -> bool:
    return (
        aggregate.mean_rank_ic >= 0.030
        and aggregate.hac_t_statistic >= 2.000
        and aggregate.candidate_delta_rank_ic >= 0.005
        and aggregate.score_health_pass_ratio >= 0.95
    )


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        os.close(descriptor)


def _validate_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"locked-test {name} is invalid")


def _validate_hash(value: object, name: str) -> None:
    if not isinstance(value, str) or not _HASH.fullmatch(value):
        raise ValueError(f"locked-test {name} is invalid")


def _validate_hashes(values: object, name: str, *, exact_count: int) -> None:
    if (
        not isinstance(values, tuple)
        or len(values) != exact_count
        or len(set(values)) != exact_count
    ):
        raise ValueError(f"locked-test {name} is invalid")
    for value in values:
        _validate_hash(value, name)


def _signature(signing_key: bytes, document_id: str, payload: Mapping[str, object]) -> str:
    signed = {"document_id": document_id, "payload": payload}
    return hmac.new(
        signing_key,
        _canonical_json(signed).encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _write_exclusive(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(value)
            output.flush()
            os.fsync(output.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


__all__ = [
    "IssuedPreregistration",
    "IssuedLabelMaturityWatermark",
    "LabelMaturityWatermark",
    "LockedTestAggregate",
    "LockedTestAuditEvent",
    "LockedTestAuthority",
    "LockedTestCapability",
    "LockedTestEvaluator",
    "LockedTestMaturity",
    "LockedTestMaturityEvidenceStore",
    "LockedTestMaturityGuard",
    "LockedTestMetrics",
    "LockedTestPreregistration",
    "LockedTestPreregistrationStore",
    "LockedTestResult",
    "LockedTestResultVerifier",
    "LockedTestTradingCalendar",
    "LockedTestUnlockRequest",
    "ProductionDeploymentApproval",
    "ProductionDeploymentApprovalStore",
    "ProductionDeploymentApprovalVerifier",
]
