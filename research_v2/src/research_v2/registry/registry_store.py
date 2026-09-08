"""Append-only signed filesystem persistence for the model lifecycle registry.

``ModelRegistry`` holds the domain rules and is deliberately in-memory; its docstring has always
said persistence adapters must retain every transition event, and until now no adapter existed,
so ``qr-v2 registry promote`` could only ever refuse. This is that adapter.

The design follows the other evidence stores in this codebase: content-addressed, HMAC-signed,
atomically written, and immutable once written. Two files per model --

    <root>/models/<hex>/record.json    current state, signed, atomically replaced
    <root>/models/<hex>/events.jsonl   every transition event, signed, append-only

The event log is the authority. ``record.json`` is a convenience snapshot, and opening a store
verifies it against the last event rather than trusting it, so a truncated or hand-edited
snapshot is rejected instead of silently becoming the model's state.

Restoration does not re-validate transitions, and that is a deliberate choice rather than a
shortcut: each transition was validated against its evidence when it was written, and the later
lifecycle states require workflow proofs that are unforgeable objects living only in the process
that issued them, so they cannot be replayed from disk at all. What restoration does validate is
everything about the persisted state itself -- identity format, status membership, lineage
consistency, event ordering, hash format, and every signature.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import shutil
import tempfile
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path

from research_v2.registry.model_registry import (
    ModelRecord,
    ModelRegistry,
    ModelStatus,
    TransitionEvent,
    load_walk_forward_evidence,
    validate_model_id,
)

REGISTRY_ROOT_ENV = "QRV2_MODEL_REGISTRY_ROOT"
_RECORD_FILENAME = "record.json"
_EVENTS_FILENAME = "events.jsonl"
_EVENT_FIELDS = (
    "model_id",
    "previous_status",
    "new_status",
    "actor",
    "reason",
    "evidence_hashes",
    "transitioned_at",
)
_RECORD_FIELDS = ("model_id", "status", "lineage_id", "parent_model_id")


class RegistryStoreError(ValueError):
    """The registry store cannot be opened, read, or appended to, and why must be explicit."""


def _canonical(payload: Mapping[str, object]) -> str:
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def _sign(key: bytes, payload: Mapping[str, object]) -> str:
    return hmac.new(key, _canonical(payload).encode("utf-8"), hashlib.sha256).hexdigest()


def _hex_of(model_id: str) -> str:
    return model_id.removeprefix("sha256:")


def _event_payload(event: TransitionEvent) -> dict[str, object]:
    return {
        "model_id": event.model_id,
        "previous_status": None if event.previous_status is None else event.previous_status.value,
        "new_status": event.new_status.value,
        "actor": event.actor,
        "reason": event.reason,
        "evidence_hashes": list(event.evidence_hashes),
        "transitioned_at": event.transitioned_at.isoformat(),
    }


def _record_payload(record: ModelRecord) -> dict[str, object]:
    return {
        "model_id": record.model_id,
        "status": record.status.value,
        "lineage_id": record.lineage_id,
        "parent_model_id": record.parent_model_id,
    }


def _parse_status(value: object, field: str) -> ModelStatus | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise RegistryStoreError(f"{field} must be a status name or null")
    try:
        return ModelStatus(value)
    except ValueError as error:
        raise RegistryStoreError(f"{field} is not a known model status: {value!r}") from error


def _parse_event(payload: Mapping[str, object], model_id: str) -> TransitionEvent:
    if set(payload) != set(_EVENT_FIELDS):
        raise RegistryStoreError(f"transition event fields are invalid for {model_id}")
    if payload["model_id"] != model_id:
        raise RegistryStoreError(f"transition event names a different model: {model_id}")
    evidence = payload["evidence_hashes"]
    if not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
        raise RegistryStoreError(f"transition event evidence hashes are invalid for {model_id}")
    actor = payload["actor"]
    reason = payload["reason"]
    if not isinstance(actor, str) or not actor.strip():
        raise RegistryStoreError(f"transition event actor is invalid for {model_id}")
    if not isinstance(reason, str) or not reason.strip():
        raise RegistryStoreError(f"transition event reason is invalid for {model_id}")
    stamp = payload["transitioned_at"]
    if not isinstance(stamp, str):
        raise RegistryStoreError(f"transition event timestamp is invalid for {model_id}")
    try:
        transitioned_at = datetime.fromisoformat(stamp)
    except ValueError as error:
        raise RegistryStoreError(
            f"transition event timestamp is not ISO-8601 for {model_id}: {stamp!r}"
        ) from error
    if transitioned_at.tzinfo is None:
        raise RegistryStoreError(f"transition event timestamp must carry a timezone: {model_id}")
    new_status = _parse_status(payload["new_status"], "new_status")
    if new_status is None:
        raise RegistryStoreError(f"transition event new_status cannot be null for {model_id}")
    return TransitionEvent(
        model_id=model_id,
        previous_status=_parse_status(payload["previous_status"], "previous_status"),
        new_status=new_status,
        actor=actor,
        reason=reason,
        evidence_hashes=tuple(evidence),
        transitioned_at=transitioned_at.astimezone(UTC),
    )


def _parse_record(payload: Mapping[str, object], model_id: str) -> ModelRecord:
    if set(payload) != set(_RECORD_FIELDS):
        raise RegistryStoreError(f"model record fields are invalid for {model_id}")
    if payload["model_id"] != model_id:
        raise RegistryStoreError(f"model record names a different model: {model_id}")
    status = _parse_status(payload["status"], "status")
    if status is None:
        raise RegistryStoreError(f"model record status cannot be null for {model_id}")
    lineage_id = payload["lineage_id"]
    if not isinstance(lineage_id, str):
        raise RegistryStoreError(f"model record lineage_id is invalid for {model_id}")
    parent = payload["parent_model_id"]
    if parent is not None and not isinstance(parent, str):
        raise RegistryStoreError(f"model record parent_model_id is invalid for {model_id}")
    return ModelRecord(
        model_id=model_id,
        status=status,
        lineage_id=lineage_id,
        parent_model_id=parent,
    )


class ModelRegistryStore:
    """Persist a ``ModelRegistry`` as signed, append-only artifacts under one root."""

    def __init__(self, root: Path, *, signing_key: bytes) -> None:
        if not isinstance(signing_key, bytes) or not signing_key:
            raise RegistryStoreError("registry store requires a non-empty signing key")
        self._root = Path(root)
        self._key = signing_key
        self._models = self._root / "models"
        self._registry = self._restore()

    @classmethod
    def from_environment(cls, root: Path | None = None) -> ModelRegistryStore:
        """Open the store configured by QRV2_MODEL_REGISTRY_ROOT and the trust root.

        Absent configuration this raises rather than falling back to a default path: a registry
        that silently materialises somewhere nobody asked for is how two operators end up with
        two disagreeing sources of truth about what is in production.
        """
        resolved = root if root is not None else os.environ.get(REGISTRY_ROOT_ENV, "")
        if not isinstance(resolved, str) or not resolved.strip():
            raise RegistryStoreError(f"{REGISTRY_ROOT_ENV} is not configured")
        secret = os.environ.get("QRV2_SELECTION_EVIDENCE_HMAC_KEY", "")
        if not secret:
            raise RegistryStoreError(
                "QRV2_SELECTION_EVIDENCE_HMAC_KEY is not configured; the registry is signed with "
                "the same single trust root as the selection and prediction evidence"
            )
        return cls(Path(resolved), signing_key=secret.encode("utf-8"))

    @property
    def registry(self) -> ModelRegistry:
        """The restored in-memory registry; every mutation must go through this store."""
        return self._registry

    # ---------- reads ----------

    def _verified_payload(self, path: Path, identity: str, what: str) -> dict[str, object]:
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise RegistryStoreError(f"{what} cannot be read for {identity}: {error}") from error
        if not isinstance(document, dict):
            raise RegistryStoreError(f"{what} is not an object for {identity}")
        payload = document.get("payload")
        signature = document.get("signature")
        if not isinstance(payload, dict) or not isinstance(signature, str):
            raise RegistryStoreError(f"{what} is malformed for {identity}")
        typed = payload
        if not hmac.compare_digest(signature, _sign(self._key, typed)):
            raise RegistryStoreError(f"{what} signature verification failed for {identity}")
        return typed

    def _read_events(self, model_dir: Path, model_id: str) -> tuple[TransitionEvent, ...]:
        path = model_dir / _EVENTS_FILENAME
        if not path.is_file():
            raise RegistryStoreError(f"transition event log is missing for {model_id}")
        events: list[TransitionEvent] = []
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError) as error:
            raise RegistryStoreError(f"transition event log cannot be read: {model_id}") from error
        for number, line in enumerate(lines, start=1):
            if not line.strip():
                raise RegistryStoreError(
                    f"transition event log has an empty line {number}: {model_id}"
                )
            try:
                document = json.loads(line)
            except json.JSONDecodeError as error:
                raise RegistryStoreError(
                    f"transition event log line {number} is not JSON: {model_id}"
                ) from error
            if not isinstance(document, dict):
                raise RegistryStoreError(
                    f"transition event log line {number} is not an object: {model_id}"
                )
            payload = document.get("payload")
            signature = document.get("signature")
            if not isinstance(payload, dict) or not isinstance(signature, str):
                raise RegistryStoreError(
                    f"transition event log line {number} is malformed: {model_id}"
                )
            if not hmac.compare_digest(signature, _sign(self._key, payload)):
                raise RegistryStoreError(
                    f"transition event log line {number} failed signature verification: {model_id}"
                )
            events.append(_parse_event(payload, model_id))
        if not events:
            raise RegistryStoreError(f"transition event log is empty: {model_id}")
        return tuple(events)

    def _restore(self) -> ModelRegistry:
        records: dict[str, ModelRecord] = {}
        histories: dict[str, tuple[TransitionEvent, ...]] = {}
        if not self._models.is_dir():
            return ModelRegistry.restore(records, histories)
        for model_dir in sorted(self._models.iterdir()):
            if not model_dir.is_dir():
                continue
            model_id = f"sha256:{model_dir.name}"
            try:
                validate_model_id(model_id)
            except ValueError as error:
                raise RegistryStoreError(
                    f"registry contains a directory that is not a model id: {model_dir.name}"
                ) from error
            events = self._read_events(model_dir, model_id)
            record = _parse_record(
                self._verified_payload(model_dir / _RECORD_FILENAME, model_id, "model record"),
                model_id,
            )
            # The event log is the authority; the snapshot must agree with it or it is stale,
            # truncated, or edited, and none of those may become the model's state.
            if events[-1].new_status is not record.status:
                raise RegistryStoreError(
                    f"model record status {record.status.value} disagrees with the last "
                    f"transition event {events[-1].new_status.value}: {model_id}"
                )
            previous: ModelStatus | None = None
            for event in events:
                if event.previous_status is not previous:
                    raise RegistryStoreError(
                        f"transition event chain is broken at {event.new_status.value}: {model_id}"
                    )
                previous = event.new_status
            records[model_id] = record
            histories[model_id] = events
        return ModelRegistry.restore(records, histories)

    def known_model_ids(self) -> tuple[str, ...]:
        """Every model id the store restored from disk."""
        return self._registry.model_ids()

    # ---------- writes ----------

    def register_draft(
        self, model_id: str, *, actor: str, lineage_id: str | None = None,
        parent_model_id: str | None = None,
    ) -> ModelRecord:
        if self._model_dir(model_id).exists():
            raise RegistryStoreError(f"model is already persisted: {model_id}")
        record = self._registry.register_draft(
            model_id, actor=actor, lineage_id=lineage_id, parent_model_id=parent_model_id
        )
        self._persist(model_id, record)
        return record

    def promote_walk_forward(
        self, model_id: str, *, actor: str, evidence_path: Path
    ) -> ModelRecord:
        """Promote a persisted DRAFT on evidence the registry re-derives G1-G6 from."""
        evidence, _, _ = load_walk_forward_evidence(Path(evidence_path))
        record = self._registry.promote_walk_forward(model_id, actor=actor, evidence=evidence)
        self._append_event(model_id, record)
        return record

    def _model_dir(self, model_id: str) -> Path:
        validate_model_id(model_id)
        return self._models / _hex_of(model_id)

    def _persist(self, model_id: str, record: ModelRecord) -> None:
        """Create a model directory holding the registration event and the first snapshot."""
        destination = self._model_dir(model_id)
        if destination.exists():
            raise RegistryStoreError(f"model is already persisted: {model_id}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".registry-", dir=destination.parent))
        try:
            events = self._registry.history(model_id)
            (temporary / _EVENTS_FILENAME).write_text(
                "".join(self._signed_event_line(event) for event in events), encoding="utf-8"
            )
            self._write_snapshot(temporary / _RECORD_FILENAME, record)
            temporary.rename(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise

    def _append_event(self, model_id: str, record: ModelRecord) -> None:
        """Append the newest transition event, then refresh the snapshot."""
        destination = self._model_dir(model_id)
        if not destination.is_dir():
            raise RegistryStoreError(f"model is not persisted: {model_id}")
        events = self._registry.history(model_id)
        path = destination / _EVENTS_FILENAME
        if not path.is_file():
            raise RegistryStoreError(f"transition event log is missing for {model_id}")
        existing = path.read_text(encoding="utf-8").splitlines()
        if len(events) != len(existing) + 1:
            raise RegistryStoreError(
                f"transition event log for {model_id} holds {len(existing)} events but the "
                f"registry reports {len(events)}; refusing to append to a diverged log"
            )
        # The log is append-only: one signed line, flushed and synced before the snapshot moves,
        # so a crash between the two leaves the log authoritative and the snapshot merely stale.
        with path.open("a", encoding="utf-8") as handle:
            handle.write(self._signed_event_line(events[-1]))
            handle.flush()
            os.fsync(handle.fileno())
        self._write_snapshot(destination / _RECORD_FILENAME, record)

    def _signed_event_line(self, event: TransitionEvent) -> str:
        payload = _event_payload(event)
        document = {"payload": payload, "signature": _sign(self._key, payload)}
        return _canonical(document) + "\n"

    def _write_snapshot(self, path: Path, record: ModelRecord) -> None:
        payload = _record_payload(record)
        document = {"payload": payload, "signature": _sign(self._key, payload)}
        temporary = path.with_name(path.name + ".tmp")
        temporary.write_text(_canonical(document) + "\n", encoding="utf-8")
        os.replace(temporary, path)


__all__ = [
    "REGISTRY_ROOT_ENV",
    "ModelRegistryStore",
    "RegistryStoreError",
]
