"""Pre-registered, fold-bound trial scheduling and validation-only model selection."""

from __future__ import annotations

import fcntl
import hashlib
import hmac
import itertools
import json
import math
import os
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager, suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from numbers import Integral, Real
from pathlib import Path
from threading import Lock
from types import MappingProxyType
from typing import Literal

import pandas as pd

from research_v2.datasets.splits import annual_folds
from research_v2.datasets.storage import DatasetArtifactStore

# Two inner folds under the v2 four-year outer train window (was three under 5y).
_INNER_FOLDS = ("I01", "I02")
_INNER_SEED = 20260801
_VALIDATION_PROVENANCE_KEY_ENV = "QRV2_VALIDATION_PROVENANCE_HMAC_KEY"
# Derived from the single fold-contract source rather than restated, so the map
# cannot drift the way the hardcoded F01-F06 tuple did.
_OUTER_VALIDATION_YEAR_BY_FOLD = {
    fold.fold_id: fold.validation_years[0] for fold in annual_folds() if not fold.locked
}
_HOST_GPU_LOCK_ROOT = Path(tempfile.gettempdir()) / "quant-research-v2-gpu-leases"
_DEFAULT_LEASE_DURATION = timedelta(hours=4)


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class SearchTrial:
    model_family: str
    fold_id: str
    config: Mapping[str, object]
    config_hash: str

    def __post_init__(self) -> None:
        _require_text(self.model_family, "model_family")
        _require_text(self.fold_id, "fold_id")
        object.__setattr__(self, "config", _freeze_mapping(self.config))


@dataclass(frozen=True)
class ValidationProvenance:
    provenance_id: str
    dataset_id: str
    outer_fold_id: str
    split_name: Literal["validation"] = "validation"

    def __post_init__(self) -> None:
        if not _is_sha256(self.provenance_id):
            raise ValueError("validation provenance_id must be a SHA-256 hex digest")
        _require_text(self.dataset_id, "validation dataset_id")
        _require_text(self.outer_fold_id, "validation outer_fold_id")
        if self.split_name != "validation":
            raise ValueError("validation provenance must reference the validation split")

    def verify(self, store: DatasetArtifactStore) -> None:
        expected_year = _OUTER_VALIDATION_YEAR_BY_FOLD.get(self.outer_fold_id)
        if expected_year is None:
            raise ValueError("validation provenance requires an ordinary F1-F4 fold")
        metadata = store.load_metadata(self.dataset_id)
        if metadata.get("dataset_role") != "VALIDATION":
            raise ValueError("validation provenance requires a verified VALIDATION artifact")
        frame = store.load_split(
            self.dataset_id,
            self.split_name,
            expected_fold_id=self.outer_fold_id,
        )
        if "signal_date" not in frame.columns:
            raise ValueError("validation artifact is missing signal_date")
        years = pd.to_datetime(frame["signal_date"], errors="raise").dt.year
        if not years.eq(expected_year).all():
            raise ValueError(
                "validation artifact dates do not match the expected validation year: "
                f"{expected_year}"
            )


class ValidationProvenanceVerifier:
    """Verify immutable validation provenance using the evaluator-owned HMAC key."""

    def __init__(
        self,
        root: Path,
        dataset_store: DatasetArtifactStore,
        *,
        verification_key: bytes,
    ) -> None:
        if not isinstance(verification_key, bytes) or not verification_key:
            raise ValueError("validation provenance verification_key must be non-empty bytes")
        self._root = root
        self._dataset_store = dataset_store
        self._verification_key = verification_key

    @classmethod
    def from_environment(
        cls,
        root: Path,
        dataset_store: DatasetArtifactStore,
    ) -> ValidationProvenanceVerifier:
        secret = os.environ.get(_VALIDATION_PROVENANCE_KEY_ENV, "")
        if not secret:
            raise ValueError(
                "validation provenance verification requires "
                f"{_VALIDATION_PROVENANCE_KEY_ENV}"
            )
        return cls(root, dataset_store, verification_key=secret.encode("utf-8"))

    def verify(self, provenance: ValidationProvenance) -> None:
        path = self._root / f"{provenance.provenance_id}.json"
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError("validation provenance cannot be loaded") from error
        payload = document.get("payload") if isinstance(document, dict) else None
        signature = document.get("signature") if isinstance(document, dict) else None
        if (
            not isinstance(payload, dict)
            or document.get("provenance_id") != provenance.provenance_id
            or not isinstance(signature, str)
            or not hmac.compare_digest(
                signature,
                _provenance_signature(
                    self._verification_key,
                    provenance.provenance_id,
                    payload,
                ),
            )
            or _provenance_id(payload) != provenance.provenance_id
        ):
            raise ValueError("validation provenance signature verification failed")
        recorded = _provenance_from_payload(provenance.provenance_id, payload)
        if recorded != provenance:
            raise ValueError("validation provenance does not match its immutable record")
        recorded.verify(self._dataset_store)


class ValidationProvenanceStore(ValidationProvenanceVerifier):
    """Issue signed validation provenance before any ordinary TEST evaluator can run."""

    def __init__(
        self,
        root: Path,
        dataset_store: DatasetArtifactStore,
        *,
        signing_key: bytes,
    ) -> None:
        super().__init__(root, dataset_store, verification_key=signing_key)
        self._signing_key = signing_key

    def issue(self, *, dataset_id: str, outer_fold_id: str) -> ValidationProvenance:
        payload: dict[str, object] = {
            "dataset_id": dataset_id,
            "outer_fold_id": outer_fold_id,
            "split_name": "validation",
        }
        provenance_id = _provenance_id(payload)
        provenance = _provenance_from_payload(provenance_id, payload)
        provenance.verify(self._dataset_store)
        self._root.mkdir(parents=True, exist_ok=True)
        path = self._root / f"{provenance_id}.json"
        document = {
            "payload": payload,
            "provenance_id": provenance_id,
            "signature": _provenance_signature(self._signing_key, provenance_id, payload),
        }
        try:
            _write_exclusive(path, _canonical_json(document) + "\n")
        except FileExistsError as error:
            raise FileExistsError(f"validation provenance is immutable: {provenance_id}") from error
        return provenance


class TrialState(StrEnum):
    PENDING = "PENDING"
    LEASED = "LEASED"
    COMPLETED = "COMPLETED"


@dataclass(frozen=True)
class TrialLease:
    trial: SearchTrial
    lease_id: str
    worker_id: str
    gpu_id: str
    expires_at: datetime


@dataclass(frozen=True)
class TrialAuditEvent:
    config_hash: str
    event: str
    worker_id: str
    gpu_id: str
    lease_id: str
    occurred_at: str


@dataclass(frozen=True)
class _LeaseClaim:
    config_hash: str
    lease_id: str
    worker_id: str
    gpu_id: str
    expires_at: datetime


@dataclass(frozen=True)
class _CompletionRecord:
    config_hash: str
    lease_id: str
    worker_id: str
    gpu_id: str


@dataclass(frozen=True)
class InnerFoldValidation:
    config_hash: str
    inner_fold_id: str
    seed: int
    rank_ic: float
    provenance: ValidationProvenance


@dataclass(frozen=True)
class OuterValidationScore:
    mean_rank_ic: float
    hac_t_statistic: float
    parameter_count: int
    provenance: ValidationProvenance


class TrialScheduler:
    """Lease pre-registered trials with expiring claims and host-global GPU locks."""

    def __init__(
        self,
        trials: Sequence[SearchTrial],
        *,
        lease_root: Path,
        lease_duration: timedelta = _DEFAULT_LEASE_DURATION,
        clock: Callable[[], datetime] = _utc_now,
    ) -> None:
        if not trials or len({trial.config_hash for trial in trials}) != len(trials):
            raise ValueError("trial scheduler requires unique declared trials")
        if len(trials) > 30:
            raise ValueError("trial scheduler cannot accept more than 30 trials")
        if lease_duration <= timedelta(0):
            raise ValueError("lease_duration must be positive")
        _require_uniform_trial_scope(trials)
        _validate_trial_identities(trials)
        self._trials = tuple(sorted(trials, key=lambda trial: trial.config_hash))
        self._states = {trial.config_hash: TrialState.PENDING for trial in self._trials}
        self._leases: dict[str, TrialLease] = {}
        self._leased_gpu_ids: set[str] = set()
        self._gpu_lock_descriptors: dict[str, int] = {}
        self._lock = Lock()
        self._lease_root = lease_root
        self._lease_duration = lease_duration
        self._clock = clock
        self._lease_root.mkdir(parents=True, exist_ok=True)
        self._event_root.mkdir(parents=True, exist_ok=True)

    def lease_next(self, *, worker_id: str, gpu_id: str) -> TrialLease | None:
        _require_text(worker_id, "worker_id")
        _require_text(gpu_id, "gpu_id")
        with self._lock:
            if gpu_id in self._leased_gpu_ids:
                raise ValueError(f"GPU already holds an active trial lease: {gpu_id}")
            for trial in self._trials:
                if self._states[trial.config_hash] is not TrialState.PENDING:
                    continue
                trial_claim = self._trial_claim_path(trial.config_hash)
                with _exclusive_lock(self._claim_lock_path(trial_claim)):
                    if self._reconcile_completed_trial(trial.config_hash):
                        continue
                    existing = _load_lease_claim(trial_claim)
                    now = _as_utc(self._clock())
                    if existing is not None:
                        if existing.expires_at > now:
                            continue
                        self._write_audit_event(
                            TrialAuditEvent(
                                trial.config_hash,
                                "EXPIRED",
                                existing.worker_id,
                                existing.gpu_id,
                                existing.lease_id,
                                _timestamp(now),
                            )
                        )
                        trial_claim.unlink()
                    lease = TrialLease(
                        trial=trial,
                        lease_id=self._next_lease_id(trial, worker_id, gpu_id),
                        worker_id=worker_id,
                        gpu_id=gpu_id,
                        expires_at=now + self._lease_duration,
                    )
                    _write_exclusive(trial_claim, _canonical_json(_lease_payload(lease)) + "\n")
                    descriptor = _acquire_host_gpu_lock(gpu_id)
                    if descriptor is None:
                        trial_claim.unlink(missing_ok=True)
                        raise ValueError(f"GPU already holds an active trial lease: {gpu_id}")
                    event = TrialAuditEvent(
                        trial.config_hash,
                        "LEASED",
                        worker_id,
                        gpu_id,
                        lease.lease_id,
                        _timestamp(now),
                    )
                    try:
                        self._write_audit_event(event)
                    except Exception:
                        _release_host_gpu_lock(descriptor)
                        trial_claim.unlink(missing_ok=True)
                        raise
                    self._states[trial.config_hash] = TrialState.LEASED
                    self._leases[trial.config_hash] = lease
                    self._leased_gpu_ids.add(gpu_id)
                    self._gpu_lock_descriptors[trial.config_hash] = descriptor
                    return lease
            return None

    def complete(self, config_hash: str, *, lease_id: str) -> None:
        with self._lock:
            lease = self._leases.get(config_hash)
            if lease is None or lease.lease_id != lease_id:
                raise ValueError("trial completion requires its active lease")
            trial_claim = self._trial_claim_path(config_hash)
            with _exclusive_lock(self._claim_lock_path(trial_claim)):
                completion = _load_completion_record(self._completion_path(config_hash))
                if completion is None:
                    claim = _load_lease_claim(trial_claim)
                    if claim is None or claim.lease_id != lease.lease_id:
                        raise ValueError("trial completion requires its active lease")
                    completion = _CompletionRecord(
                        config_hash,
                        lease.lease_id,
                        lease.worker_id,
                        lease.gpu_id,
                    )
                    _write_exclusive(
                        self._completion_path(config_hash),
                        _canonical_json(_completion_payload(completion)) + "\n",
                    )
                elif completion.lease_id != lease.lease_id:
                    raise ValueError("trial has already been completed by another worker")
                self._write_audit_event(
                    TrialAuditEvent(
                        config_hash,
                        "COMPLETED",
                        lease.worker_id,
                        lease.gpu_id,
                        lease.lease_id,
                        _timestamp(_as_utc(self._clock())),
                    )
                )
                _remove_claim_if_owned(trial_claim, lease.lease_id)
                self._release_gpu_lock(config_hash)
                self._states[config_hash] = TrialState.COMPLETED
                del self._leases[config_hash]
                self._leased_gpu_ids.remove(lease.gpu_id)

    def audit(self) -> tuple[TrialAuditEvent, ...]:
        with self._lock:
            return self._read_audit_events()

    def write_audit(self, root: Path) -> Path:
        """Publish one immutable content-addressed audit without any test-period inputs."""
        events = self.audit()
        with self._lock:
            payload = {
                "events": [_audit_summary(event) for event in events],
                "fold_id": self._trials[0].fold_id,
                "model_family": self._trials[0].model_family,
                "schema_version": 1,
                "trials": [
                    {"config": _json_compatible(trial.config), "config_hash": trial.config_hash}
                    for trial in self._trials
                ],
            }
        audit_id = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
        destination = root / f"{audit_id}.json"
        root.mkdir(parents=True, exist_ok=True)
        try:
            _write_exclusive(destination, _canonical_json({"audit_id": audit_id, **payload}) + "\n")
        except FileExistsError as error:
            raise FileExistsError(f"trial audit is immutable: {audit_id}") from error
        return destination

    def _trial_claim_path(self, config_hash: str) -> Path:
        return self._lease_root / f"trial-{config_hash}.lease"

    def _completion_path(self, config_hash: str) -> Path:
        return self._lease_root / f"trial-{config_hash}.completed"

    @property
    def _event_root(self) -> Path:
        return self._lease_root / "events"

    @staticmethod
    def _claim_lock_path(claim_path: Path) -> Path:
        return claim_path.with_suffix(claim_path.suffix + ".lock")

    def _reconcile_completed_trial(self, config_hash: str) -> bool:
        completion = _load_completion_record(self._completion_path(config_hash))
        if completion is None:
            return False
        _remove_claim_if_owned(self._trial_claim_path(config_hash), completion.lease_id)
        self._states[config_hash] = TrialState.COMPLETED
        return True

    def _write_audit_event(self, event: TrialAuditEvent) -> None:
        payload = asdict(event)
        event_id = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
        try:
            _write_exclusive(self._event_root / f"{event_id}.json", _canonical_json(payload) + "\n")
        except FileExistsError:
            return

    def _next_lease_id(self, trial: SearchTrial, worker_id: str, gpu_id: str) -> str:
        expiration_count = sum(
            event.event == "EXPIRED" and event.config_hash == trial.config_hash
            for event in self._read_audit_events()
        )
        return _lease_id(trial.config_hash, worker_id, gpu_id, expiration_count)

    def _read_audit_events(self) -> tuple[TrialAuditEvent, ...]:
        events = [_load_audit_event(path) for path in sorted(self._event_root.glob("*.json"))]
        return tuple(
            sorted(
                events,
                key=lambda event: (
                    event.occurred_at,
                    {"LEASED": 0, "EXPIRED": 1, "COMPLETED": 2}[event.event],
                    event.config_hash,
                    event.lease_id,
                ),
            )
        )

    def _release_gpu_lock(self, config_hash: str) -> None:
        descriptor = self._gpu_lock_descriptors.pop(config_hash, None)
        if descriptor is not None:
            _release_host_gpu_lock(descriptor)

    def close(self) -> None:
        """Release local GPU descriptors; persistent trial claims expire if not completed."""
        with self._lock:
            for descriptor in tuple(self._gpu_lock_descriptors.values()):
                _release_host_gpu_lock(descriptor)
            self._gpu_lock_descriptors.clear()
            self._leased_gpu_ids.clear()

    def __del__(self) -> None:
        with suppress(Exception):
            self.close()


def build_pre_registered_trials(
    model_family: str,
    fold_id: str,
    search_space: Mapping[str, Sequence[object]],
    *,
    maximum_trials: int = 30,
) -> tuple[SearchTrial, ...]:
    """Enumerate a declared grid, sort by fold-bound config hash, and enforce the hard cap."""
    _require_text(model_family, "model_family")
    _require_text(fold_id, "fold_id")
    if not search_space or not 1 <= maximum_trials <= 30:
        raise ValueError("search_space or maximum_trials is invalid")
    names = sorted(search_space)
    values_per_name: list[list[object]] = []
    for name in names:
        values = list(search_space[name])
        if not name or not values:
            raise ValueError("search dimensions must have names and values")
        canonical_values = sorted(values, key=_canonical_json)
        if len({_canonical_json(value) for value in canonical_values}) != len(canonical_values):
            raise ValueError(f"search dimension contains duplicate values: {name}")
        values_per_name.append(canonical_values)
    trials: list[SearchTrial] = []
    for combination in itertools.product(*values_per_name):
        config = dict(zip(names, combination, strict=True))
        config_hash = _trial_config_hash(model_family, fold_id, config)
        trials.append(SearchTrial(model_family, fold_id, config, config_hash))
    return tuple(sorted(trials, key=lambda trial: trial.config_hash)[:maximum_trials])


def select_model_config(
    trials: Sequence[SearchTrial],
    inner_validation: Sequence[InnerFoldValidation],
    outer_validation: Mapping[str, OuterValidationScore],
    *,
    validation_store: DatasetArtifactStore,
    validation_provenance_root: Path,
) -> SearchTrial:
    """Choose top-three inner candidates, then use outer validation without TEST access."""
    verifier = ValidationProvenanceVerifier.from_environment(
        validation_provenance_root,
        validation_store,
    )
    ranked_inner, _ = _prepare_selection_inputs(
        trials,
        inner_validation,
        outer_validation,
        verifier,
    )
    return min(
        ranked_inner,
        key=lambda trial: (
            -outer_validation[trial.config_hash].mean_rank_ic,
            -outer_validation[trial.config_hash].hac_t_statistic,
            outer_validation[trial.config_hash].parameter_count,
            trial.config_hash,
        ),
    )


def selection_evidence_hash(
    trials: Sequence[SearchTrial],
    inner_validation: Sequence[InnerFoldValidation],
    outer_validation: Mapping[str, OuterValidationScore],
    *,
    validation_store: DatasetArtifactStore,
    validation_provenance_root: Path,
) -> str:
    """Hash only pre-declared trial and VALIDATION evidence, never test labels or metrics."""
    verifier = ValidationProvenanceVerifier.from_environment(
        validation_provenance_root,
        validation_store,
    )
    _prepare_selection_inputs(trials, inner_validation, outer_validation, verifier)
    payload = {
        "inner_validation": [
            asdict(value)
            for value in sorted(
                inner_validation,
                key=lambda value: (
                    value.config_hash,
                    value.inner_fold_id,
                    value.provenance.dataset_id,
                ),
            )
        ],
        "outer_validation": {
            config_hash: asdict(value) for config_hash, value in sorted(outer_validation.items())
        },
        "trials": [
            {
                "config": _json_compatible(trial.config),
                "config_hash": trial.config_hash,
                "fold_id": trial.fold_id,
                "model_family": trial.model_family,
            }
            for trial in sorted(trials, key=lambda trial: trial.config_hash)
        ],
    }
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _prepare_selection_inputs(
    trials: Sequence[SearchTrial],
    inner_validation: Sequence[InnerFoldValidation],
    outer_validation: Mapping[str, OuterValidationScore],
    verifier: ValidationProvenanceVerifier,
) -> tuple[tuple[SearchTrial, ...], dict[str, float]]:
    if not trials:
        raise ValueError("at least one trial is required")
    _require_uniform_trial_scope(trials)
    _validate_trial_identities(trials)
    if len(trials) < 3:
        raise ValueError("selection requires at least three trials for the top three stage")
    trial_by_hash = {trial.config_hash: trial for trial in trials}
    if len(trial_by_hash) != len(trials):
        raise ValueError("trials must have unique config hashes")
    expected_fold_id = trials[0].fold_id
    inner_means = _inner_validation_means(
        inner_validation,
        set(trial_by_hash),
        verifier=verifier,
        expected_fold_id=expected_fold_id,
    )
    ranked_inner = tuple(
        sorted(
            trials,
            key=lambda trial: (-inner_means[trial.config_hash], trial.config_hash),
        )[:3]
    )
    selected_hashes = {trial.config_hash for trial in ranked_inner}
    if set(outer_validation) != selected_hashes:
        raise ValueError("outer validation must cover exactly the top three inner trials")
    _validate_outer_scores(
        outer_validation,
        verifier=verifier,
        expected_fold_id=expected_fold_id,
    )
    return ranked_inner, inner_means


def _inner_validation_means(
    values: Sequence[InnerFoldValidation],
    expected_hashes: set[str],
    *,
    verifier: ValidationProvenanceVerifier,
    expected_fold_id: str,
) -> dict[str, float]:
    grouped: dict[str, dict[str, float]] = {config_hash: {} for config_hash in expected_hashes}
    verified: set[ValidationProvenance] = set()
    for value in values:
        if (
            value.config_hash not in grouped
            or value.inner_fold_id not in _INNER_FOLDS
            or value.seed != _INNER_SEED
            or not _finite(value.rank_ic)
            or value.inner_fold_id in grouped[value.config_hash]
        ):
            raise ValueError("inner validation evidence is invalid")
        _verify_validation_provenance(
            value.provenance,
            verifier,
            expected_fold_id=expected_fold_id,
            verified=verified,
        )
        grouped[value.config_hash][value.inner_fold_id] = value.rank_ic
    if any(set(folds) != set(_INNER_FOLDS) for folds in grouped.values()):
        raise ValueError("each trial requires two inner folds for selection")
    return {
        config_hash: sum(scores.values()) / len(scores) for config_hash, scores in grouped.items()
    }


def _validate_outer_scores(
    values: Mapping[str, OuterValidationScore],
    *,
    verifier: ValidationProvenanceVerifier,
    expected_fold_id: str,
) -> None:
    verified: set[ValidationProvenance] = set()
    for score in values.values():
        if (
            not isinstance(score, OuterValidationScore)
            or not _finite(score.mean_rank_ic)
            or not _finite(score.hac_t_statistic)
            or not isinstance(score.parameter_count, Integral)
            or isinstance(score.parameter_count, bool)
            or score.parameter_count < 1
        ):
            raise ValueError("outer validation score is invalid")
        _verify_validation_provenance(
            score.provenance,
            verifier,
            expected_fold_id=expected_fold_id,
            verified=verified,
        )


def _verify_validation_provenance(
    provenance: ValidationProvenance,
    verifier: ValidationProvenanceVerifier,
    *,
    expected_fold_id: str,
    verified: set[ValidationProvenance],
) -> None:
    if not isinstance(provenance, ValidationProvenance):
        raise ValueError("validation evidence requires verified validation provenance")
    if provenance.outer_fold_id != expected_fold_id:
        raise ValueError("validation provenance does not match the trial fold")
    if provenance not in verified:
        verifier.verify(provenance)
        verified.add(provenance)


def _lease_id(config_hash: str, worker_id: str, gpu_id: str, attempt: int) -> str:
    return hashlib.sha256(f"{config_hash}|{worker_id}|{gpu_id}|{attempt}".encode()).hexdigest()


def _audit_summary(event: TrialAuditEvent) -> dict[str, str]:
    """Keep the published experiment audit reproducible while the event log retains time."""
    return {
        "config_hash": event.config_hash,
        "event": event.event,
        "gpu_id": event.gpu_id,
        "lease_id": event.lease_id,
        "worker_id": event.worker_id,
    }


def _lease_payload(lease: TrialLease) -> dict[str, str]:
    return {
        "config_hash": lease.trial.config_hash,
        "expires_at": _as_utc(lease.expires_at).isoformat(),
        "gpu_id": lease.gpu_id,
        "lease_id": lease.lease_id,
        "worker_id": lease.worker_id,
    }


def _completion_payload(completion: _CompletionRecord) -> dict[str, str]:
    return {
        "config_hash": completion.config_hash,
        "gpu_id": completion.gpu_id,
        "lease_id": completion.lease_id,
        "worker_id": completion.worker_id,
    }


def _load_lease_claim(path: Path) -> _LeaseClaim | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("trial lease claim is malformed") from error
    if not isinstance(payload, dict) or set(payload) != {
        "config_hash",
        "expires_at",
        "gpu_id",
        "lease_id",
        "worker_id",
    }:
        raise ValueError("trial lease claim is malformed")
    config_hash = payload.get("config_hash")
    lease_id = payload.get("lease_id")
    worker_id = payload.get("worker_id")
    gpu_id = payload.get("gpu_id")
    expires_at = payload.get("expires_at")
    if (
        not isinstance(config_hash, str)
        or not _is_sha256(config_hash)
        or not isinstance(lease_id, str)
        or not isinstance(worker_id, str)
        or not isinstance(gpu_id, str)
        or not isinstance(expires_at, str)
    ):
        raise ValueError("trial lease claim is malformed")
    _require_text(lease_id, "lease_id")
    _require_text(worker_id, "worker_id")
    _require_text(gpu_id, "gpu_id")
    try:
        return _LeaseClaim(
            config_hash,
            lease_id,
            worker_id,
            gpu_id,
            _as_utc(datetime.fromisoformat(expires_at)),
        )
    except ValueError as error:
        raise ValueError("trial lease claim is malformed") from error


def _load_completion_record(path: Path) -> _CompletionRecord | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("trial completion record is malformed") from error
    if not isinstance(payload, dict) or set(payload) != {
        "config_hash",
        "gpu_id",
        "lease_id",
        "worker_id",
    }:
        raise ValueError("trial completion record is malformed")
    config_hash = payload.get("config_hash")
    lease_id = payload.get("lease_id")
    worker_id = payload.get("worker_id")
    gpu_id = payload.get("gpu_id")
    if (
        not isinstance(config_hash, str)
        or not _is_sha256(config_hash)
        or not isinstance(lease_id, str)
        or not isinstance(worker_id, str)
        or not isinstance(gpu_id, str)
    ):
        raise ValueError("trial completion record is malformed")
    _require_text(lease_id, "lease_id")
    _require_text(worker_id, "worker_id")
    _require_text(gpu_id, "gpu_id")
    return _CompletionRecord(config_hash, lease_id, worker_id, gpu_id)


def _load_audit_event(path: Path) -> TrialAuditEvent:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("trial audit event is malformed") from error
    if not isinstance(payload, dict) or set(payload) != {
        "config_hash",
        "event",
        "gpu_id",
        "lease_id",
        "occurred_at",
        "worker_id",
    }:
        raise ValueError("trial audit event is malformed")
    config_hash = payload.get("config_hash")
    event = payload.get("event")
    worker_id = payload.get("worker_id")
    gpu_id = payload.get("gpu_id")
    lease_id = payload.get("lease_id")
    occurred_at = payload.get("occurred_at")
    if (
        not isinstance(config_hash, str)
        or not _is_sha256(config_hash)
        or event not in {"LEASED", "EXPIRED", "COMPLETED"}
        or not isinstance(worker_id, str)
        or not isinstance(gpu_id, str)
        or not isinstance(lease_id, str)
        or not isinstance(occurred_at, str)
    ):
        raise ValueError("trial audit event is malformed")
    _require_text(worker_id, "worker_id")
    _require_text(gpu_id, "gpu_id")
    _require_text(lease_id, "lease_id")
    try:
        _as_utc(datetime.fromisoformat(occurred_at))
    except ValueError as error:
        raise ValueError("trial audit event is malformed") from error
    return TrialAuditEvent(config_hash, event, worker_id, gpu_id, lease_id, occurred_at)


def _as_utc(value: datetime) -> datetime:
    if not isinstance(value, datetime) or value.tzinfo is None:
        raise ValueError("lease timestamps must be timezone-aware datetimes")
    return value.astimezone(UTC)


def _timestamp(value: datetime) -> str:
    return _as_utc(value).isoformat()


def _provenance_id(payload: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _provenance_signature(
    signing_key: bytes,
    provenance_id: str,
    payload: Mapping[str, object],
) -> str:
    signed = {"payload": payload, "provenance_id": provenance_id}
    return hmac.new(
        signing_key,
        _canonical_json(signed).encode(),
        hashlib.sha256,
    ).hexdigest()


def _provenance_from_payload(
    provenance_id: str,
    payload: Mapping[str, object],
) -> ValidationProvenance:
    expected = {"dataset_id", "outer_fold_id", "split_name"}
    if set(payload) != expected:
        raise ValueError("validation provenance record is malformed")
    dataset_id = payload.get("dataset_id")
    outer_fold_id = payload.get("outer_fold_id")
    split_name = payload.get("split_name")
    if not isinstance(dataset_id, str) or not isinstance(outer_fold_id, str):
        raise ValueError("validation provenance record is malformed")
    if split_name != "validation":
        raise ValueError("validation provenance record is malformed")
    return ValidationProvenance(provenance_id, dataset_id, outer_fold_id, split_name)


def _is_sha256(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )


def _write_exclusive(path: Path, content: str) -> None:
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
    descriptor = os.open(path, flags, 0o644)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as file:
            file.write(content)
            file.flush()
            os.fsync(file.fileno())
    except Exception:
        path.unlink(missing_ok=True)
        raise


def _remove_claim(path: Path, lease_id: str) -> None:
    claim = _load_lease_claim(path)
    if claim is None:
        raise ValueError("active lease claim is missing")
    if claim.lease_id != lease_id:
        raise ValueError("active lease claim does not match the completing lease")
    path.unlink()


def _remove_claim_if_owned(path: Path, lease_id: str) -> None:
    try:
        _remove_claim(path, lease_id)
    except ValueError as error:
        if str(error) != "active lease claim is missing":
            raise


@contextmanager
def _exclusive_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _acquire_host_gpu_lock(gpu_id: str) -> int | None:
    _HOST_GPU_LOCK_ROOT.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(gpu_id.encode("utf-8")).hexdigest()
    descriptor = os.open(_HOST_GPU_LOCK_ROOT / f"{digest}.lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        os.close(descriptor)
        return None
    return descriptor


def _release_host_gpu_lock(descriptor: int) -> None:
    try:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _require_uniform_trial_scope(trials: Sequence[SearchTrial]) -> None:
    families = {trial.model_family for trial in trials}
    folds = {trial.fold_id for trial in trials}
    if len(families) != 1 or len(folds) != 1:
        raise ValueError("trials must use one model family and one fold")


def _validate_trial_identities(trials: Sequence[SearchTrial]) -> None:
    for trial in trials:
        expected = _trial_config_hash(trial.model_family, trial.fold_id, trial.config)
        if trial.config_hash != expected:
            raise ValueError("trial config_hash does not match its immutable identity")


def _trial_config_hash(model_family: str, fold_id: str, config: Mapping[str, object]) -> str:
    payload = {"model_family": model_family, "fold_id": fold_id, "config": _json_compatible(config)}
    return hashlib.sha256(_canonical_json(payload).encode()).hexdigest()


def _freeze_mapping(value: Mapping[str, object]) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ValueError("trial config must be a string-keyed mapping")
    return MappingProxyType({key: _freeze_value(item) for key, item in value.items()})


def _freeze_value(value: object) -> object:
    if isinstance(value, Mapping):
        return _freeze_mapping(value)
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_value(item) for item in value)
    return value


def _json_compatible(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    return value


def _finite(value: object) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(float(value))


def _require_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} cannot be empty")


def _canonical_json(value: object) -> str:
    return json.dumps(
        _json_compatible(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


__all__ = [
    "InnerFoldValidation",
    "OuterValidationScore",
    "SearchTrial",
    "TrialAuditEvent",
    "TrialLease",
    "TrialScheduler",
    "TrialState",
    "ValidationProvenance",
    "ValidationProvenanceStore",
    "ValidationProvenanceVerifier",
    "build_pre_registered_trials",
    "selection_evidence_hash",
    "select_model_config",
]
