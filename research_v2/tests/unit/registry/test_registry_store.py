"""Tests for the append-only signed registry persistence adapter.

The store is what turns "passed all six gates" into a recorded lifecycle state, so the
properties that matter are: every transition event is retained, nothing can be edited without
detection, the event log outranks the snapshot, and a refused promotion leaves no trace.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from research_v2.evaluation.gates import GateEvidence, ScoreHealthEvidence
from research_v2.registry.model_registry import (
    ModelRecord,
    ModelStatus,
    TransitionEvent,
    build_walk_forward_evidence_document,
    derive_walk_forward_model_id,
    validate_walk_forward_evidence,
)
from research_v2.registry.registry_store import (
    REGISTRY_ROOT_ENV,
    ModelRegistryStore,
    RegistryStoreError,
)

MODEL_ID = "sha256:" + "a1" * 32
OTHER_ID = "sha256:" + "b2" * 32
KEY = b"registry-trust-root"
HEX_DIR = MODEL_ID.removeprefix("sha256:")
LABEL_DAYS = 969
SCORING_DAYS = 945


def _passing_document() -> dict[str, object]:
    """A gate document with the real M0R numbers, so promotion genuinely succeeds."""
    health = tuple(
        [ScoreHealthEvidence(coverage=0.98, unique_score_count=1800, score_std=0.29)]
        * SCORING_DAYS
        + [ScoreHealthEvidence(coverage=0.50, unique_score_count=1800, score_std=0.29)]
        * (LABEL_DAYS - SCORING_DAYS)
    )
    evidence = GateEvidence(
        mean_rank_ic=0.094376,
        hac_t_statistic=12.905,
        yearly_mean_rank_ic={year: 0.09 for year in (2022, 2023, 2024, 2025)},
        candidate_delta_rank_ic=0.009149,
        valid_test_day_count=LABEL_DAYS,
        score_health=health,
    )
    assert all(
        result.status.value == "PASS"
        for result in validate_walk_forward_evidence(evidence).results.values()
    )

    class _Evaluation:
        gate_evidence = evidence

    return build_walk_forward_evidence_document(_Evaluation())  # type: ignore[arg-type]


def _failing_document() -> dict[str, object]:
    payload = _passing_document()
    gate = payload["gate_evidence"]
    assert isinstance(gate, dict)
    gate["candidate_delta_rank_ic"] = 0.001  # below the G4 bar of 0.005
    return payload


@pytest.fixture
def evidence_path(tmp_path: Path) -> Path:
    path = tmp_path / "promotion_evidence.json"
    path.write_text(json.dumps(_passing_document(), sort_keys=True), encoding="utf-8")
    return path


@pytest.fixture
def failing_evidence_path(tmp_path: Path) -> Path:
    path = tmp_path / "failing_evidence.json"
    path.write_text(json.dumps(_failing_document(), sort_keys=True), encoding="utf-8")
    return path


def _open(root: Path) -> ModelRegistryStore:
    return ModelRegistryStore(root, signing_key=KEY)


def test_register_draft_round_trips_through_a_reopen(tmp_path: Path) -> None:
    store = _open(tmp_path)
    record = store.register_draft(MODEL_ID, actor="researcher")
    assert record.status is ModelStatus.DRAFT
    assert record.lineage_id == MODEL_ID

    reopened = _open(tmp_path)
    assert reopened.known_model_ids() == (MODEL_ID,)
    restored = reopened.registry.get(MODEL_ID)
    assert restored == record
    history = reopened.registry.history(MODEL_ID)
    assert len(history) == 1
    assert history[0].previous_status is None
    assert history[0].new_status is ModelStatus.DRAFT
    assert history[0].actor == "researcher"


def test_promotion_is_persisted_with_both_events(tmp_path: Path, evidence_path: Path) -> None:
    store = _open(tmp_path)
    store.register_draft(MODEL_ID, actor="researcher")
    promoted = store.promote_walk_forward(MODEL_ID, actor="researcher", evidence_path=evidence_path)
    assert promoted.status is ModelStatus.WALK_FORWARD_PASSED

    reopened = _open(tmp_path)
    assert reopened.registry.get(MODEL_ID).status is ModelStatus.WALK_FORWARD_PASSED
    history = reopened.registry.history(MODEL_ID)
    assert [event.new_status for event in history] == [
        ModelStatus.DRAFT,
        ModelStatus.WALK_FORWARD_PASSED,
    ]
    assert history[1].previous_status is ModelStatus.DRAFT
    # the promotion event carries the gate-evidence hash it was decided on
    assert len(history[1].evidence_hashes) == 1
    assert len(history[1].evidence_hashes[0]) == 64


def test_event_log_is_append_only(tmp_path: Path, evidence_path: Path) -> None:
    store = _open(tmp_path)
    store.register_draft(MODEL_ID, actor="researcher")
    log = tmp_path / "models" / HEX_DIR / "events.jsonl"
    after_registration = log.read_text(encoding="utf-8")
    assert after_registration.count("\n") == 1

    store.promote_walk_forward(MODEL_ID, actor="researcher", evidence_path=evidence_path)
    after_promotion = log.read_text(encoding="utf-8")
    assert after_promotion.startswith(after_registration), "an existing event line was rewritten"
    assert after_promotion.count("\n") == 2


def test_a_refused_promotion_leaves_no_trace(
    tmp_path: Path, failing_evidence_path: Path
) -> None:
    """Gates are recomputed from raw numbers, so a failing candidate must not be recorded."""
    store = _open(tmp_path)
    store.register_draft(MODEL_ID, actor="researcher")
    log = tmp_path / "models" / HEX_DIR / "events.jsonl"
    before = log.read_bytes()

    with pytest.raises(ValueError, match="did not pass"):
        store.promote_walk_forward(
            MODEL_ID, actor="researcher", evidence_path=failing_evidence_path
        )
    assert log.read_bytes() == before, "a refused promotion appended an event"
    assert _open(tmp_path).registry.get(MODEL_ID).status is ModelStatus.DRAFT


def test_registering_twice_is_refused(tmp_path: Path) -> None:
    store = _open(tmp_path)
    store.register_draft(MODEL_ID, actor="researcher")
    with pytest.raises(RegistryStoreError, match="already persisted"):
        _open(tmp_path).register_draft(MODEL_ID, actor="researcher")


def test_promoting_an_unregistered_model_is_refused(tmp_path: Path, evidence_path: Path) -> None:
    with pytest.raises(ValueError, match="not registered"):
        _open(tmp_path).promote_walk_forward(
            MODEL_ID, actor="researcher", evidence_path=evidence_path
        )


def test_tampering_with_the_snapshot_is_detected(tmp_path: Path) -> None:
    store = _open(tmp_path)
    store.register_draft(MODEL_ID, actor="researcher")
    record_path = tmp_path / "models" / HEX_DIR / "record.json"
    document = json.loads(record_path.read_text(encoding="utf-8"))
    document["payload"]["status"] = ModelStatus.PRODUCTION.value
    record_path.write_text(json.dumps(document, sort_keys=True), encoding="utf-8")
    with pytest.raises(RegistryStoreError, match="signature verification failed"):
        _open(tmp_path)


def test_tampering_with_an_event_line_is_detected(tmp_path: Path) -> None:
    store = _open(tmp_path)
    store.register_draft(MODEL_ID, actor="researcher")
    log = tmp_path / "models" / HEX_DIR / "events.jsonl"
    line = json.loads(log.read_text(encoding="utf-8").strip())
    line["payload"]["actor"] = "someone-else"
    log.write_text(json.dumps(line, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RegistryStoreError, match="failed signature verification"):
        _open(tmp_path)


def test_a_resigned_snapshot_that_disagrees_with_the_log_is_refused(tmp_path: Path) -> None:
    """The log is the authority: a validly signed but stale snapshot must not win.

    Editing the snapshot and re-signing it with the trust root defeats signature checking, so
    the store also requires the snapshot's status to equal the last event's. Without this an
    operator holding the key could mark a DRAFT as PRODUCTION without any transition event.
    """
    store = _open(tmp_path)
    store.register_draft(MODEL_ID, actor="researcher")
    record_path = tmp_path / "models" / HEX_DIR / "record.json"
    document = json.loads(record_path.read_text(encoding="utf-8"))
    document["payload"]["status"] = ModelStatus.PRODUCTION.value
    canonical = json.dumps(
        document["payload"], ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )
    document["signature"] = hmac.new(KEY, canonical.encode("utf-8"), hashlib.sha256).hexdigest()
    record_path.write_text(json.dumps(document, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(RegistryStoreError, match="disagrees with the last transition event"):
        _open(tmp_path)


def test_a_broken_event_chain_is_refused(tmp_path: Path, evidence_path: Path) -> None:
    store = _open(tmp_path)
    store.register_draft(MODEL_ID, actor="researcher")
    store.promote_walk_forward(MODEL_ID, actor="researcher", evidence_path=evidence_path)
    log = tmp_path / "models" / HEX_DIR / "events.jsonl"
    lines = log.read_text(encoding="utf-8").splitlines()
    # drop the registration event, leaving a promotion whose previous_status has no predecessor
    log.write_text(lines[1] + "\n", encoding="utf-8")
    with pytest.raises(RegistryStoreError, match="chain is broken|must begin at DRAFT"):
        _open(tmp_path)


def test_an_empty_event_line_is_refused(tmp_path: Path) -> None:
    store = _open(tmp_path)
    store.register_draft(MODEL_ID, actor="researcher")
    log = tmp_path / "models" / HEX_DIR / "events.jsonl"
    log.write_text(log.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(RegistryStoreError, match="empty line"):
        _open(tmp_path)


def test_a_directory_that_is_not_a_model_id_is_refused(tmp_path: Path) -> None:
    _open(tmp_path)
    (tmp_path / "models" / "not-a-model-id").mkdir(parents=True)
    with pytest.raises(RegistryStoreError, match="not a model id"):
        _open(tmp_path)


def test_from_environment_refuses_to_guess_a_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No default path: two operators must not end up with two sources of truth."""
    monkeypatch.delenv(REGISTRY_ROOT_ENV, raising=False)
    monkeypatch.setenv("QRV2_SELECTION_EVIDENCE_HMAC_KEY", "trust-root")
    with pytest.raises(RegistryStoreError, match=REGISTRY_ROOT_ENV):
        ModelRegistryStore.from_environment()

    monkeypatch.setenv(REGISTRY_ROOT_ENV, str(tmp_path))
    monkeypatch.delenv("QRV2_SELECTION_EVIDENCE_HMAC_KEY", raising=False)
    with pytest.raises(RegistryStoreError, match="QRV2_SELECTION_EVIDENCE_HMAC_KEY"):
        ModelRegistryStore.from_environment()

    monkeypatch.setenv("QRV2_SELECTION_EVIDENCE_HMAC_KEY", "trust-root")
    store = ModelRegistryStore.from_environment()
    assert store.known_model_ids() == ()


def test_derived_model_id_can_be_registered_and_promoted(
    tmp_path: Path, evidence_path: Path
) -> None:
    """The id minted from a walk-forward workspace is a valid registry identity."""
    model_id = derive_walk_forward_model_id({fold: f"{i:064x}" for i, fold in enumerate(
        ("F1", "F2", "F3", "F4"), start=1)})
    validate_model_id_shape(model_id)
    store = _open(tmp_path)
    store.register_draft(model_id, actor="researcher")
    assert store.promote_walk_forward(
        model_id, actor="researcher", evidence_path=evidence_path
    ).status is ModelStatus.WALK_FORWARD_PASSED
    assert _open(tmp_path).known_model_ids() == (model_id,)


def validate_model_id_shape(model_id: str) -> None:
    assert model_id.startswith("sha256:") and len(model_id) == len("sha256:") + 64


def test_restore_rejects_inconsistent_persisted_state() -> None:
    """The restore seam validates state, since it cannot re-validate transitions."""
    from research_v2.registry.model_registry import ModelRegistry

    event = TransitionEvent(
        model_id=MODEL_ID,
        previous_status=None,
        new_status=ModelStatus.DRAFT,
        actor="researcher",
        reason="draft registered",
        evidence_hashes=(),
        transitioned_at=datetime(2026, 9, 9, tzinfo=UTC),
    )
    record = ModelRecord(model_id=MODEL_ID, status=ModelStatus.DRAFT, lineage_id=MODEL_ID)
    ModelRegistry.restore({MODEL_ID: record}, {MODEL_ID: (event,)})  # must not raise

    # a record whose status is not its last event's status
    stale = ModelRecord(model_id=MODEL_ID, status=ModelStatus.PRODUCTION, lineage_id=MODEL_ID)
    with pytest.raises(ValueError, match="disagrees with the last transition event"):
        ModelRegistry.restore({MODEL_ID: stale}, {MODEL_ID: (event,)})

    # a history that does not begin at DRAFT
    promoted = TransitionEvent(
        model_id=MODEL_ID,
        previous_status=None,
        new_status=ModelStatus.WALK_FORWARD_PASSED,
        actor="researcher",
        reason="gates passed",
        evidence_hashes=("c" * 64,),
        transitioned_at=datetime(2026, 9, 9, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="must begin at DRAFT"):
        ModelRegistry.restore(
            {MODEL_ID: ModelRecord(
                model_id=MODEL_ID,
                status=ModelStatus.WALK_FORWARD_PASSED,
                lineage_id=MODEL_ID,
            )},
            {MODEL_ID: (promoted,)},
        )

    # records and histories must cover the same models
    with pytest.raises(ValueError, match="one event history per record"):
        ModelRegistry.restore({MODEL_ID: record, OTHER_ID: record}, {MODEL_ID: (event,)})

    # a derived model whose parent is absent
    child = ModelRecord(
        model_id=OTHER_ID, status=ModelStatus.DRAFT, lineage_id=MODEL_ID, parent_model_id=MODEL_ID
    )
    child_event = TransitionEvent(
        model_id=OTHER_ID,
        previous_status=None,
        new_status=ModelStatus.DRAFT,
        actor="researcher",
        reason="draft registered",
        evidence_hashes=(),
        transitioned_at=datetime(2026, 9, 9, tzinfo=UTC),
    )
    with pytest.raises(ValueError, match="parent model is absent"):
        ModelRegistry.restore({OTHER_ID: child}, {OTHER_ID: (child_event,)})
