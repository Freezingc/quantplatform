from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from threading import Barrier

import pytest

from research_v2.evaluation.gates import GateEvidence, ScoreHealthEvidence
from research_v2.monitoring.shadow import (
    ShadowDeploymentApprovalStore,
    ShadowEvidence,
    ShadowEvidenceStore,
    ShadowQualification,
    ShadowQualificationStatus,
)
from research_v2.registry.active_pointer import (
    ActiveModelPointerStore,
    PointerEligibilityGuard,
    PointerSwitchEvidence,
)
from research_v2.registry.locked_test import (
    ProductionDeploymentApproval,
    _content_hash,
    _signature,
)
from research_v2.registry.model_registry import ModelRecord, ModelRegistry, ModelStatus

_A = "sha256:" + "a" * 64
_B = "sha256:" + "b" * 64
_C = "sha256:" + "c" * 64
_APPROVAL_KEY = "pointer-approval-key"
_RUN_ID = "00000000-0000-0000-0000-000000000001"


@pytest.fixture(autouse=True)
def _trusted_shadow_approval_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QRV2_SHADOW_APPROVAL_HMAC_KEY", _APPROVAL_KEY)
    monkeypatch.setenv("QRV2_SHADOW_EVIDENCE_HMAC_KEY", "pointer-evidence-key")
    monkeypatch.setenv("QRV2_SHADOW_DEPLOYER_ID", "shadow-controller")


def _evidence() -> GateEvidence:
    return GateEvidence(
        mean_rank_ic=0.04,
        hac_t_statistic=2.5,
        yearly_mean_rank_ic={year: 0.04 for year in range(2020, 2026)},
        candidate_delta_rank_ic=0.006,
        valid_test_day_count=6,
        score_health=tuple(
            ScoreHealthEvidence(coverage=0.95, unique_score_count=30, score_std=0.1)
            for _ in range(6)
        ),
    )


def _registry() -> ModelRegistry:
    registry = ModelRegistry()
    for model_id in (_A, _B, _C):
        registry.register_draft(model_id, actor="trainer")
        registry.promote_walk_forward(model_id, actor="reviewer", evidence=_evidence())
    return registry


def _approval(
    registry: ModelRegistry,
    root: Path,
    model_id: str,
    artifact_digest: str,
) -> PointerSwitchEvidence:
    approval = ShadowDeploymentApprovalStore(root, signing_key=_APPROVAL_KEY.encode()).issue(
        model_id=model_id,
        actor="risk-approver",
        walk_forward_evidence_hash=registry.history(model_id)[-1].evidence_hashes[0],
        artifact_bundle_hash=artifact_digest,
        inference_contract_hash="d" * 64,
        v2test_drill_hash="e" * 64,
        rollback_plan_hash="f" * 64,
    )
    return PointerSwitchEvidence(shadow_approval=approval, shadow_approval_root=root)


def _store(registry: ModelRegistry, digests: dict[str, str]) -> ActiveModelPointerStore:
    return ActiveModelPointerStore(
        eligibility=PointerEligibilityGuard(registry, bundle_digest=digests.__getitem__),
        clock=lambda: datetime(2026, 8, 28, 9, 30, tzinfo=UTC),
    )


def _ready_shadow_evidence(root: Path, model_id: str, *, max_psi_errors: int) -> ShadowEvidence:
    dates = tuple(date(2026, 1, 2) + timedelta(days=index) for index in range(60))
    qualification = ShadowQualification(
        model_id=model_id,
        trading_calendar_id="a" * 64,
        valid_day_count=60,
        status=ShadowQualificationStatus.READY_FOR_REVIEW,
        max_consecutive_psi_errors=max_psi_errors,
        valid_signal_dates=dates,
        invalid_signal_dates=(),
        valid_report_ids=tuple(f"{index:064x}" for index in range(60)),
        valid_runtime_day_ids=tuple(f"{index + 100:064x}" for index in range(60)),
        invalid_runtime_day_ids=(),
    )
    return ShadowEvidenceStore(root, signing_key=b"pointer-evidence-key").issue(qualification)


def test_switch_uses_full_pointer_cas_and_records_an_append_only_audit_event(
    tmp_path: Path,
) -> None:
    registry = _registry()
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    store = _store(registry, digests)
    first = store.compare_and_swap(
        consumer_id="paper-sim",
        environment="shadow",
        expected_current=None,
        model_id=_A,
        model_version=1,
        actor="operator-a",
        reason="initial shadow deployment",
        evidence=_approval(registry, tmp_path / "approvals", _A, digests[_A]),
        run_id=_RUN_ID,
    )

    assert first is not None
    assert first.pointer.model_id == _A
    assert first.pointer.pointer_version == 1
    assert first.event.old_pointer is None
    assert first.event.new_pointer == first.pointer
    assert first.event.actor == "operator-a"
    assert first.event.reason == "initial shadow deployment"
    assert first.event.effective_at == datetime(2026, 8, 28, 9, 30, tzinfo=UTC)
    assert first.event.run_id == _RUN_ID
    assert first.event.evidence_ids == (first.event.shadow_approval_id,)

    stale = store.compare_and_swap(
        consumer_id="paper-sim",
        environment="shadow",
        expected_current=None,
        model_id=_B,
        model_version=1,
        actor="operator-b",
        reason="stale overwrite",
        evidence=_approval(registry, tmp_path / "approvals", _B, digests[_B]),
        run_id=_RUN_ID,
    )

    assert stale is None
    assert store.get("paper-sim", "shadow") == first.pointer
    assert store.history("paper-sim", "shadow") == (first.event,)


def test_concurrent_switches_allow_exactly_one_winner_and_one_complete_pointer(
    tmp_path: Path,
) -> None:
    registry = _registry()
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    store = _store(registry, digests)
    first = store.compare_and_swap(
        consumer_id="paper-sim",
        environment="shadow",
        expected_current=None,
        model_id=_A,
        model_version=1,
        actor="operator-a",
        reason="initial shadow deployment",
        evidence=_approval(registry, tmp_path / "approvals", _A, digests[_A]),
        run_id=_RUN_ID,
    )
    assert first is not None
    barrier = Barrier(2)

    def switch(model_id: str, actor: str) -> object:
        barrier.wait()
        return store.compare_and_swap(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=first.pointer,
            model_id=model_id,
            model_version=1,
            actor=actor,
            reason="concurrent change",
            evidence=_approval(registry, tmp_path / "approvals", model_id, digests[model_id]),
            run_id=_RUN_ID,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        candidates = ((_B, "operator-b"), (_C, "operator-c"))
        updates = list(executor.map(lambda item: switch(*item), candidates))

    winner = [update for update in updates if update is not None]
    assert len(winner) == 1
    current = store.get("paper-sim", "shadow")
    assert current is not None
    assert current.model_id in {_B, _C}
    assert current.pointer_version == 2
    assert len(store.history("paper-sim", "shadow")) == 2


def test_rollback_is_a_new_cas_event_and_rejects_a_damaged_previous_artifact(
    tmp_path: Path,
) -> None:
    registry = _registry()
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    store = _store(registry, digests)
    first = store.compare_and_swap(
        consumer_id="paper-sim",
        environment="shadow",
        expected_current=None,
        model_id=_A,
        model_version=1,
        actor="operator-a",
        reason="initial shadow deployment",
        evidence=_approval(registry, tmp_path / "approvals", _A, digests[_A]),
        run_id=_RUN_ID,
    )
    assert first is not None
    second = store.compare_and_swap(
        consumer_id="paper-sim",
        environment="shadow",
        expected_current=first.pointer,
        model_id=_B,
        model_version=1,
        actor="operator-b",
        reason="candidate B switch",
        evidence=_approval(registry, tmp_path / "approvals", _B, digests[_B]),
        run_id=_RUN_ID,
    )
    assert second is not None

    del digests[_A]
    with pytest.raises(ValueError, match="artifact"):
        store.rollback(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=second.pointer,
            event_id=second.event.event_id,
            actor="incident-operator",
            reason="B score-health failure",
            evidence=_approval(registry, tmp_path / "approvals", _A, "1" * 64),
            run_id=_RUN_ID,
        )

    assert store.get("paper-sim", "shadow") == second.pointer

    digests[_A] = "1" * 64
    rolled_back = store.rollback(
        consumer_id="paper-sim",
        environment="shadow",
        expected_current=second.pointer,
        event_id=second.event.event_id,
        actor="incident-operator",
        reason="B score-health failure",
        evidence=_approval(registry, tmp_path / "approvals", _A, digests[_A]),
        run_id=_RUN_ID,
    )

    assert rolled_back is not None
    assert rolled_back.pointer.model_id == _A
    assert rolled_back.pointer.pointer_version == 3
    assert rolled_back.event.kind == "ROLLBACK"
    assert rolled_back.event.reverts_event_id == second.event.event_id
    assert store.history("paper-sim", "shadow") == (first.event, second.event, rolled_back.event)


def test_shadow_switch_fails_closed_without_signed_approval_or_matching_artifact(
    tmp_path: Path,
) -> None:
    registry = _registry()
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    store = _store(registry, digests)

    with pytest.raises(ValueError, match="approval"):
        store.compare_and_swap(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=None,
            model_id=_A,
            model_version=1,
            actor="operator-a",
            reason="missing approval",
            evidence=PointerSwitchEvidence(),
            run_id=_RUN_ID,
        )
    with pytest.raises(ValueError, match="artifact"):
        store.compare_and_swap(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=None,
            model_id=_A,
            model_version=1,
            actor="operator-a",
            reason="mismatched artifact",
            evidence=_approval(registry, tmp_path / "approvals", _A, "f" * 64),
            run_id=_RUN_ID,
        )


def test_shadow_guard_accepts_ready_evidence_below_the_drift_review_threshold(
    tmp_path: Path,
) -> None:
    registry = _registry()
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    approval_root = tmp_path / "approvals"
    approval_evidence = _approval(registry, approval_root, _A, digests[_A])
    assert approval_evidence.shadow_approval is not None
    registry.enter_shadow(
        _A,
        approval=approval_evidence.shadow_approval,
        approval_root=approval_root,
    )
    shadow_root = tmp_path / "shadow-evidence"
    evidence = _ready_shadow_evidence(shadow_root, _A, max_psi_errors=1)
    store = _store(registry, digests)

    update = store.compare_and_swap(
        consumer_id="paper-sim",
        environment="shadow",
        expected_current=None,
        model_id=_A,
        model_version=1,
        actor="operator-a",
        reason="qualified shadow switch",
        evidence=PointerSwitchEvidence(
            shadow_approval=approval_evidence.shadow_approval,
            shadow_approval_root=approval_root,
            shadow_evidence=evidence,
            shadow_evidence_root=shadow_root,
        ),
        run_id=_RUN_ID,
    )

    assert update is not None
    assert update.event.shadow_evidence_id == evidence.evidence_id


def test_shadow_model_requires_signed_readiness_evidence_before_pointer_switch(
    tmp_path: Path,
) -> None:
    registry = _registry()
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    approval_root = tmp_path / "approvals"
    approval_evidence = _approval(registry, approval_root, _A, digests[_A])
    assert approval_evidence.shadow_approval is not None
    registry.enter_shadow(
        _A,
        approval=approval_evidence.shadow_approval,
        approval_root=approval_root,
    )
    store = _store(registry, digests)

    with pytest.raises(ValueError, match="readiness evidence"):
        store.compare_and_swap(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=None,
            model_id=_A,
            model_version=1,
            actor="operator-a",
            reason="shadow switch without readiness evidence",
            evidence=approval_evidence,
            run_id=_RUN_ID,
        )


def test_production_pointer_rejects_a_model_below_production_status(tmp_path: Path) -> None:
    registry = _registry()
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    store = _store(registry, digests)

    with pytest.raises(ValueError, match="PRODUCTION"):
        store.compare_and_swap(
            consumer_id="paper-sim",
            environment="production",
            expected_current=None,
            model_id=_A,
            model_version=1,
            actor="production-operator",
            reason="invalid production switch",
            evidence=PointerSwitchEvidence(evidence_ids=("d" * 64,)),
            run_id=_RUN_ID,
        )


def test_production_pointer_rejects_unverified_caller_supplied_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = _registry()
    monkeypatch.setattr(
        registry,
        "get",
        lambda model_id: ModelRecord(model_id, ModelStatus.PRODUCTION),
    )
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    store = _store(registry, digests)

    with pytest.raises(ValueError, match="verified production approval"):
        store.compare_and_swap(
            consumer_id="paper-sim",
            environment="production",
            expected_current=None,
            model_id=_A,
            model_version=1,
            actor="production-operator",
            reason="unverified production switch",
            evidence=PointerSwitchEvidence(evidence_ids=("d" * 64,)),
            run_id=_RUN_ID,
        )


def test_pointer_rejects_an_unsigned_model_version_above_the_artifact_contract(
    tmp_path: Path,
) -> None:
    registry = _registry()
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    store = _store(registry, digests)

    with pytest.raises(ValueError, match="model_version"):
        store.compare_and_swap(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=None,
            model_id=_A,
            model_version=2,
            actor="operator-a",
            reason="unbound version switch",
            evidence=_approval(registry, tmp_path / "approvals", _A, digests[_A]),
            run_id=_RUN_ID,
        )


def test_unconfigured_v1_consumer_has_no_pointer_or_switch_side_effect(tmp_path: Path) -> None:
    registry = _registry()
    digests = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}
    store = _store(registry, digests)

    update = store.compare_and_swap(
        consumer_id="paper-sim",
        environment="shadow",
        expected_current=None,
        model_id=_A,
        model_version=1,
        actor="operator-a",
        reason="initial shadow deployment",
        evidence=_approval(registry, tmp_path / "approvals", _A, digests[_A]),
        run_id=_RUN_ID,
    )

    assert update is not None
    assert store.get("legacy-v1-consumer", "production") is None
    assert store.history("legacy-v1-consumer", "production") == ()


# --- who is allowed to be pointed at -----------------------------------------
#
# The store's atomicity was already pinned: dropping the compare in compare_and_swap
# and freezing the pointer version both die on the concurrency and CAS tests. What was
# not pinned is almost the whole eligibility guard -- ten of this round's sixteen
# mutations survived, and eight of the ten are in PointerEligibilityGuard or in the
# rollback's own compare. Each test below isolates one refusal, and each is built so
# that the clause under test is the ONLY thing standing between the input and a switch.

_DIGESTS = {_A: "1" * 64, _B: "2" * 64, _C: "3" * 64}


def _shadow_stage(tmp_path: Path):
    """Put _A into SHADOW and return everything a qualified shadow switch needs."""
    registry = _registry()
    approval_root = tmp_path / "approvals"
    approval = _approval(registry, approval_root, _A, _DIGESTS[_A])
    assert approval.shadow_approval is not None
    registry.enter_shadow(_A, approval=approval.shadow_approval, approval_root=approval_root)
    shadow_root = tmp_path / "shadow-evidence"
    evidence = _ready_shadow_evidence(shadow_root, _A, max_psi_errors=1)
    bundle = PointerSwitchEvidence(
        shadow_approval=approval.shadow_approval,
        shadow_approval_root=approval_root,
        shadow_evidence=evidence,
        shadow_evidence_root=shadow_root,
    )
    return registry, bundle


def _switch(store: ActiveModelPointerStore, bundle: PointerSwitchEvidence, *, reason: str):
    return store.compare_and_swap(
        consumer_id="paper-sim",
        environment="shadow",
        expected_current=None,
        model_id=_A,
        model_version=1,
        actor="operator-a",
        reason=reason,
        evidence=bundle,
        run_id=_RUN_ID,
    )


def test_shadow_pointer_refuses_an_approval_signed_for_a_different_model(
    tmp_path: Path,
) -> None:
    """An approval naming another model must not authorise this one's pointer.

    Built so the model match is the only thing that can refuse it: the approval is issued
    for _B but carries _A's artifact digest AND _A's walk-forward evidence hash, so both
    of the other two matches pass. Measured -- with the model match removed the switch is
    authorised and a live consumer starts reading _A under an approval signed for _B.
    The production path has the same clause (P-C) and is filed rather than pinned here,
    because reaching it needs a signed locked-test result.
    """
    registry = _registry()
    root = tmp_path / "approvals"
    cross_model = ShadowDeploymentApprovalStore(root, signing_key=_APPROVAL_KEY.encode()).issue(
        model_id=_B,
        actor="risk-approver",
        walk_forward_evidence_hash=registry.history(_A)[-1].evidence_hashes[0],
        artifact_bundle_hash=_DIGESTS[_A],
        inference_contract_hash="d" * 64,
        v2test_drill_hash="e" * 64,
        rollback_plan_hash="f" * 64,
    )
    store = _store(registry, _DIGESTS)

    with pytest.raises(ValueError, match="approval model does not match pointer target"):
        store.compare_and_swap(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=None,
            model_id=_A,
            model_version=1,
            actor="operator-a",
            reason="cross-model approval",
            evidence=PointerSwitchEvidence(
                shadow_approval=cross_model, shadow_approval_root=root
            ),
            run_id=_RUN_ID,
        )


def test_shadow_pointer_refuses_an_approval_bound_to_another_models_evidence(
    tmp_path: Path,
) -> None:
    """The approval must cite the walk-forward evidence actually on record for THIS model.

    Issued for _A with _A's artifact digest but citing a walk-forward evidence hash that is
    on nobody's record, so the model and artifact matches both pass and only this clause
    can refuse. Note that _registry() promotes all three models from one identical
    GateEvidence, so their recorded hashes are equal and "another model's hash" would not
    be foreign at all -- the hash has to be one no promotion ever produced. This clause is
    also what makes a DRAFT unreachable: a draft has no WALK_FORWARD_PASSED event, so the
    set of hashes on record for it is empty.
    """
    registry = _registry()
    root = tmp_path / "approvals"
    wrong_evidence = ShadowDeploymentApprovalStore(
        root, signing_key=_APPROVAL_KEY.encode()
    ).issue(
        model_id=_A,
        actor="risk-approver",
        walk_forward_evidence_hash="9" * 64,
        artifact_bundle_hash=_DIGESTS[_A],
        inference_contract_hash="d" * 64,
        v2test_drill_hash="e" * 64,
        rollback_plan_hash="f" * 64,
    )
    store = _store(registry, _DIGESTS)

    with pytest.raises(ValueError, match="does not match walk-forward evidence"):
        store.compare_and_swap(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=None,
            model_id=_A,
            model_version=1,
            actor="operator-a",
            reason="foreign walk-forward evidence",
            evidence=PointerSwitchEvidence(
                shadow_approval=wrong_evidence, shadow_approval_root=root
            ),
            run_id=_RUN_ID,
        )


def test_a_draft_model_is_refused_by_the_status_clause_not_by_a_later_one(
    tmp_path: Path,
) -> None:
    """Which clause refuses matters: "not eligible yet" and "approval does not match" differ.

    Adding DRAFT to the eligible statuses cannot actually let a draft through, because a
    draft has no walk-forward evidence hash on record and the next clause refuses it --
    measured, not assumed. But it changes WHICH error an operator gets, and those two
    errors send them in opposite directions, so the status clause is pinned by its message.
    """
    registry = ModelRegistry()
    registry.register_draft(_A, actor="trainer")
    root = tmp_path / "approvals"
    approval = ShadowDeploymentApprovalStore(root, signing_key=_APPROVAL_KEY.encode()).issue(
        model_id=_A,
        actor="risk-approver",
        walk_forward_evidence_hash="9" * 64,
        artifact_bundle_hash=_DIGESTS[_A],
        inference_contract_hash="d" * 64,
        v2test_drill_hash="e" * 64,
        rollback_plan_hash="f" * 64,
    )
    store = _store(registry, _DIGESTS)

    with pytest.raises(ValueError, match="requires at least a WALK_FORWARD_PASSED model"):
        store.compare_and_swap(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=None,
            model_id=_A,
            model_version=1,
            actor="operator-a",
            reason="draft is not eligible",
            evidence=PointerSwitchEvidence(shadow_approval=approval, shadow_approval_root=root),
            run_id=_RUN_ID,
        )


def test_shadow_readiness_evidence_must_carry_a_valid_signature(tmp_path: Path) -> None:
    """Readiness evidence is verified, not merely well shaped.

    The forged copy keeps a correct evidence_id and a correct payload -- only the HMAC is
    wrong -- so every structural check passes and the signature is the only refusal. With
    the verification call removed this switch is authorised on evidence nobody signed.
    """
    registry, genuine = _shadow_stage(tmp_path / "genuine")
    assert _switch(_store(registry, _DIGESTS), genuine, reason="genuine evidence") is not None

    registry, bundle = _shadow_stage(tmp_path / "forged")
    assert bundle.shadow_evidence is not None
    forged = PointerSwitchEvidence(
        shadow_approval=bundle.shadow_approval,
        shadow_approval_root=bundle.shadow_approval_root,
        shadow_evidence=replace(bundle.shadow_evidence, signature="0" * 64),
        shadow_evidence_root=bundle.shadow_evidence_root,
    )

    with pytest.raises(ValueError, match="signature verification failed"):
        _switch(_store(registry, _DIGESTS), forged, reason="forged evidence")


def test_the_pointer_target_digest_must_be_a_well_formed_evidence_id(tmp_path: Path) -> None:
    """The bundle digest is re-validated here even though the store produced it."""
    registry = _registry()
    root = tmp_path / "approvals"
    approval = _approval(registry, root, _A, _DIGESTS[_A])
    store = _store(registry, {_A: "not-a-digest", **{_B: _DIGESTS[_B], _C: _DIGESTS[_C]}})

    with pytest.raises(ValueError, match="artifact digest is invalid"):
        store.compare_and_swap(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=None,
            model_id=_A,
            model_version=1,
            actor="operator-a",
            reason="malformed digest",
            evidence=approval,
            run_id=_RUN_ID,
        )


def test_a_repeated_switch_is_refused_and_a_stale_rollback_changes_nothing(
    tmp_path: Path,
) -> None:
    """Two store-level refusals that the switch path's tests do not reach.

    Re-switching to the model version already targeted must raise rather than mint a new
    pointer version for the same target. And a rollback whose ``expected_current`` does
    not match what is live must decline by returning None -- measured: with the rollback's
    compare removed it proceeds to the event check and raises instead, so a caller that
    treats None as "someone else already moved the pointer" gets an exception instead.
    """
    registry, bundle = _shadow_stage(tmp_path)
    store = _store(registry, _DIGESTS)
    update = _switch(store, bundle, reason="initial switch")
    assert update is not None

    with pytest.raises(ValueError, match="already targets this model version"):
        store.compare_and_swap(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=update.pointer,
            model_id=_A,
            model_version=1,
            actor="operator-a",
            reason="duplicate switch",
            evidence=bundle,
            run_id=_RUN_ID,
        )

    assert (
        store.rollback(
            consumer_id="paper-sim",
            environment="shadow",
            expected_current=None,
            event_id=update.event.event_id,
            actor="operator-b",
            reason="stale rollback",
            evidence=bundle,
            run_id=_RUN_ID,
        )
        is None
    )
    assert store.get("paper-sim", "shadow") == update.pointer


# --- the production branch's two matches, pinned where the clauses live -------
#
# Both were already killed at full scope before this test existed -- but by
# test_locked_test.py's end-to-end chain test, whose name describes a pointer
# property while living in the locked-test suite because it needs that suite's
# machinery to produce one approval. That is the fifth case in the attribution
# matrix and the most name-mismatched one. These are the pointer guard's own
# clauses, so they are pinned here against a directly signed approval; the
# locked-test suite keeps its chain test, because a unit pin on the clause and an
# end-to-end pin on the pipeline are different assertions, not duplicated ones.

_PRODUCTION_KEY = "production-approval-key"


def _production_approval(
    root: Path, *, model_id: str, artifact_bundle_hash: str
) -> ProductionDeploymentApproval:
    """A correctly signed production approval, without running a locked test to get one.

    The verifier checks the fields, the content hash, the HMAC under
    QRV2_PRODUCTION_APPROVAL_HMAC_KEY and a stored record, so a test that holds the key
    can produce one directly -- which is what makes it possible to name a DIFFERENT model
    or cite a DIFFERENT artifact while everything else stays valid.
    """
    provisional = ProductionDeploymentApproval(
        approval_id="0" * 64,
        model_id=model_id,
        locked_test_result_id="7" * 64,
        artifact_bundle_hash=artifact_bundle_hash,
        actor="production-risk-approver",
        signature="0" * 64,
    )
    payload = provisional.payload()
    approval_id = _content_hash(payload)
    approval = replace(
        provisional,
        approval_id=approval_id,
        signature=_signature(_PRODUCTION_KEY.encode(), approval_id, payload),
    )
    # Guard the guard: the payload signed here is the one the verifier rebuilds.
    assert approval.payload() == payload
    assert approval.approval_id == _content_hash(approval.payload())
    record = root / ".issued" / f"{approval_id}.json"
    record.parent.mkdir(parents=True, exist_ok=True)
    record.write_text(approval.to_json(), encoding="utf-8")
    return approval


def _production_switch(
    registry: ModelRegistry, approval: ProductionDeploymentApproval, root: Path, *, reason: str
):
    """One production switch attempt against _A on a fresh store, so attempts stay independent."""
    return _store(registry, _DIGESTS).compare_and_swap(
        consumer_id="paper-sim",
        environment="production",
        expected_current=None,
        model_id=_A,
        model_version=1,
        actor="production-operator",
        reason=reason,
        evidence=PointerSwitchEvidence(
            production_approval=approval, production_approval_root=root
        ),
        run_id=_RUN_ID,
    )


def test_production_pointer_refuses_an_approval_for_another_model_or_artifact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A production approval authorises ONE model's ONE artifact, and both halves are checked.

    Each refusal is built so only the clause under test can produce it: the cross-model
    approval cites _A's artifact digest, and the cross-artifact approval names _A, so in
    each case the other match passes. Measured death causes: with the model match removed
    an approval signed for _B authorises a switch to _A; with the artifact match removed
    the switch is authorised on an approval for a different bundle -- ``DID NOT RAISE`` in
    both cases, i.e. a live consumer starts serving an artifact nobody approved.
    """
    monkeypatch.setenv("QRV2_PRODUCTION_APPROVAL_HMAC_KEY", _PRODUCTION_KEY)
    registry = _registry()
    # The guard reads the target's status; the other production tests in this file put the
    # registry into PRODUCTION the same way.
    monkeypatch.setattr(
        registry, "get", lambda model_id: ModelRecord(model_id, ModelStatus.PRODUCTION)
    )

    genuine_root = tmp_path / "genuine"
    genuine = _production_approval(
        genuine_root, model_id=_A, artifact_bundle_hash=_DIGESTS[_A]
    )
    # Control: the matching approval really does switch the pointer.
    assert _production_switch(registry, genuine, genuine_root, reason="genuine") is not None

    cross_model_root = tmp_path / "cross-model"
    cross_model = _production_approval(
        cross_model_root, model_id=_B, artifact_bundle_hash=_DIGESTS[_A]
    )
    with pytest.raises(ValueError, match="approval model does not match pointer target"):
        _production_switch(registry, cross_model, cross_model_root, reason="cross-model")

    cross_artifact_root = tmp_path / "cross-artifact"
    cross_artifact = _production_approval(
        cross_artifact_root, model_id=_A, artifact_bundle_hash=_DIGESTS[_B]
    )
    with pytest.raises(ValueError, match="approval artifact does not match pointer target"):
        _production_switch(registry, cross_artifact, cross_artifact_root, reason="cross-artifact")
