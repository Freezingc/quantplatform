from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import pytest

from research_v2.evaluation.gates import (
    GateEvidence,
    GateStatus,
    ScoreHealthEvidence,
    _ordinary_test_years,
    evaluate_oos_gates,
)
from research_v2.monitoring.shadow import ShadowDeploymentApprovalStore
from research_v2.registry.model_registry import (
    _ALLOWED_TRANSITIONS,
    ModelRecord,
    ModelRegistry,
    ModelStatus,
    _gate_evidence_hash,
    _issue_registry_workflow_proof,
    load_walk_forward_evidence,
    validate_model_id,
)

# The fold contract's ordinary test years (F1-F4 -> 2022-2025). Fixtures derive from it so
# they cannot silently pin the retired F01-F06 range again.
CONTRACT_YEARS = _ordinary_test_years()

MODEL_ID = "sha256:" + "a" * 64
CHILD_MODEL_ID = "sha256:" + "b" * 64


def _evidence(*, valid_score_days: int = 6) -> GateEvidence:
    return GateEvidence(
        mean_rank_ic=0.04,
        hac_t_statistic=2.5,
        yearly_mean_rank_ic={year: 0.04 for year in CONTRACT_YEARS},
        candidate_delta_rank_ic=0.006,
        valid_test_day_count=6,
        score_health=tuple(
            ScoreHealthEvidence(coverage=0.95, unique_score_count=30, score_std=0.1)
            for _ in range(valid_score_days)
        ),
    )


def test_registry_requires_all_gates_before_walk_forward_promotion() -> None:
    registry = ModelRegistry()
    registry.register_draft(MODEL_ID, actor="trainer")

    promoted = registry.promote_walk_forward(MODEL_ID, actor="reviewer", evidence=_evidence())

    assert promoted.status is ModelStatus.WALK_FORWARD_PASSED
    history = registry.history(MODEL_ID)
    assert len(history) == 2
    assert history[-1].reason == "all G1-G6 walk-forward gates passed"
    assert len(history[-1].evidence_hashes) == 1
    with pytest.raises(ValueError, match="cannot transition"):
        registry.transition(
            MODEL_ID,
            ModelStatus.PRODUCTION,
            actor="reviewer",
            evidence_hashes=("b" * 64,),
            reason="skip shadow",
        )


def test_registry_rejects_failed_gate_evidence_and_allows_retirement() -> None:
    registry = ModelRegistry()
    registry.register_draft(MODEL_ID, actor="trainer")
    with pytest.raises(ValueError, match="G6"):
        registry.promote_walk_forward(
            MODEL_ID,
            actor="reviewer",
            evidence=_evidence(valid_score_days=5),
        )

    retired = registry.transition(
        MODEL_ID,
        ModelStatus.RETIRED,
        actor="reviewer",
        evidence_hashes=(),
        reason="candidate withdrawn",
    )
    assert retired.status is ModelStatus.RETIRED


def test_derived_models_inherit_the_immutable_registered_parent_lineage() -> None:
    registry = ModelRegistry()
    root = registry.register_draft(MODEL_ID, actor="trainer")
    child = registry.register_draft(CHILD_MODEL_ID, actor="trainer", parent_model_id=MODEL_ID)

    assert root.lineage_id == MODEL_ID
    assert child.lineage_id == MODEL_ID
    assert child.parent_model_id == MODEL_ID
    with pytest.raises(ValueError, match="parent"):
        registry.register_draft(
            "sha256:" + "c" * 64,
            actor="trainer",
            parent_model_id=MODEL_ID,
            lineage_id="fake-lineage",
        )


def test_registry_reserves_locked_test_lifecycle_for_its_dedicated_workflow(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("QRV2_SHADOW_APPROVAL_HMAC_KEY", "test-shadow-approval-key")
    monkeypatch.setenv("QRV2_SHADOW_DEPLOYER_ID", "shadow-controller")
    registry = ModelRegistry()
    registry.register_draft(MODEL_ID, actor="trainer")
    registry.promote_walk_forward(MODEL_ID, actor="reviewer", evidence=_evidence())
    approvals = ShadowDeploymentApprovalStore(
        tmp_path / "approvals",  # type: ignore[operator]
        signing_key=b"test-shadow-approval-key",
    )
    approval = approvals.issue(
        model_id=MODEL_ID,
        actor="risk-approver",
        walk_forward_evidence_hash=registry.history(MODEL_ID)[-1].evidence_hashes[0],
        artifact_bundle_hash="b" * 64,
        inference_contract_hash="c" * 64,
        v2test_drill_hash="d" * 64,
        rollback_plan_hash="e" * 64,
    )
    shadow = registry.enter_shadow(
        MODEL_ID,
        approval=approval,
        approval_root=tmp_path / "approvals",  # type: ignore[operator]
    )
    assert shadow.status is ModelStatus.SHADOW
    with pytest.raises(ValueError, match="dedicated workflow"):
        registry.transition(
            MODEL_ID,
            ModelStatus.LOCKED_TEST_APPROVED,
            actor="attacker",
            evidence_hashes=("c" * 64,),
            reason="bypass locked-test evidence verification",
        )
    with pytest.raises(ValueError, match="verified workflow proof"):
        registry._approve_locked_test(object())  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="cannot transition"):
        registry.transition(
            MODEL_ID,
            ModelStatus.WALK_FORWARD_PASSED,
            actor="reviewer",
            evidence_hashes=("f" * 64,),
            reason="failed rollback",
        )


# --- the registry's own copy of the promotion wall ---------------------------
#
# evaluate_frozen_oos refuses a non-TEST role and model_evaluator refuses locked
# dates, but what a reviewer actually hands the registry is an evidence FILE, so the
# registry re-implements those refusals on the parsed document. Six of the sixteen
# mutations measured against this file are refused somewhere else or cannot be
# reached at all; these six tests are for the ones nothing else catches. Every
# docstring below states a probed behaviour, not a suspected one.


def _write_evidence(tmp_path: Path, name: str = "walk_forward.json", **overrides: object) -> Path:
    """A valid ORDINARY_TEST evidence file, with any top-level field overridden."""
    payload: dict[str, object] = {
        "schema_version": 1,
        "role": "ORDINARY_TEST",
        "gate_evidence": {
            "mean_rank_ic": 0.04,
            "hac_t_statistic": 2.5,
            "yearly_mean_rank_ic": {str(year): 0.04 for year in CONTRACT_YEARS},
            "candidate_delta_rank_ic": 0.006,
            "valid_test_day_count": 6,
            "score_health": [
                {"coverage": 0.95, "unique_score_count": 30, "score_std": 0.1}
                for _ in range(6)
            ],
        },
    }
    payload.update(overrides)
    path = tmp_path / name
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_evidence_role_must_be_ordinary_test(tmp_path: Path) -> None:
    """Validation-split evidence must not promote a model, whatever the gates say.

    Probed: with the role check removed, both ``role="VALIDATION"`` and a missing
    role load cleanly and recompute to six PASSes -- validation evidence would walk a
    model straight into WALK_FORWARD_PASSED. The evaluator refuses the same thing
    (``role != "TEST"`` raises PermissionError) and that refusal is pinned there;
    this is the registry's copy of the same wall, on the file a reviewer hands it.
    """
    control = _write_evidence(tmp_path, name="control.json")
    _, _, report = load_walk_forward_evidence(control)
    # Guard the guard: the untouched file really does load and really does pass.
    assert all(result.status is GateStatus.PASS for result in report.results.values())

    for name, overrides in (
        ("validation.json", {"role": "VALIDATION"}),
        ("locked.json", {"role": "LOCKED_TEST"}),
        ("missing.json", {"role": None}),
    ):
        path = _write_evidence(tmp_path, name=name, **overrides)
        with pytest.raises(ValueError, match="ORDINARY_TEST"):
            load_walk_forward_evidence(path)


def test_evidence_file_carrying_unknown_fields_is_refused(tmp_path: Path) -> None:
    """Unknown fields are refused rather than ignored, and they change the hash.

    Probed: relaxing the exact-field check to a subset check lets an evidence file
    carrying an extra field load and recompute to six PASSes -- and because the
    returned hash is over the raw bytes, that file hashes DIFFERENTLY from the same
    evidence without the field (6b539dab... versus 16a2a3d2... in the probe). So an
    approval bound to one would not match the other. Accepting unknown fields would
    also mean a future schema field is silently ignored by this version.
    """
    clean = _write_evidence(tmp_path, name="clean.json")
    _, clean_hash, _ = load_walk_forward_evidence(clean)
    assert len(clean_hash) == 64

    document = json.loads(clean.read_text(encoding="utf-8"))
    document["gate_evidence"]["surprise"] = 1
    tainted = tmp_path / "tainted.json"
    tainted.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="gate_evidence fields are invalid"):
        load_walk_forward_evidence(tainted)


def test_only_a_draft_can_enter_walk_forward_passed() -> None:
    """The DRAFT-only rule is the ONLY thing stopping a backwards lifecycle jump.

    Probed, and the probe contradicted the obvious guess. ``promote_walk_forward``
    calls the private ``_transition`` directly, and ``_transition`` does not re-check
    ``_ALLOWED_TRANSITIONS`` -- that check lives in the public ``transition()``. So
    with the DRAFT-only guard removed, a second promotion of an already-promoted
    model SUCCEEDS and appends another WALK_FORWARD_PASSED event; the same holds for
    a model that has reached SHADOW, PRODUCTION or RETIRED. Nothing else catches it.
    """
    registry = ModelRegistry()
    registry.register_draft(MODEL_ID, actor="trainer")
    registry.promote_walk_forward(MODEL_ID, actor="reviewer", evidence=_evidence())
    assert len(registry.history(MODEL_ID)) == 2

    with pytest.raises(ValueError, match="only DRAFT"):
        registry.promote_walk_forward(MODEL_ID, actor="reviewer", evidence=_evidence())
    # And the failed attempt appended nothing.
    assert len(registry.history(MODEL_ID)) == 2

    # A retired model cannot be resurrected through the same door.
    registry.register_draft(CHILD_MODEL_ID, actor="trainer")
    registry.transition(
        CHILD_MODEL_ID,
        ModelStatus.RETIRED,
        actor="reviewer",
        evidence_hashes=(),
        reason="withdrawn before evaluation",
    )
    with pytest.raises(ValueError, match="only DRAFT"):
        registry.promote_walk_forward(CHILD_MODEL_ID, actor="reviewer", evidence=_evidence())


def test_evidence_hash_binds_the_recomputed_verdict() -> None:
    """The recorded hash commits to the verdict, not only to the inputs.

    Probed: one GateEvidence hashed against two different recomputed verdicts gives
    two different digests today, and one identical digest once "recomputed_gates" is
    dropped from the hashed payload. That binding is what makes an approval expire
    when the gate code changes -- without it, an approval granted against a passing
    verdict would still match after the same evidence started failing.
    """
    evidence = _evidence()
    passing = evaluate_oos_gates(evidence, is_baseline=False)
    failing = evaluate_oos_gates(_evidence(valid_score_days=5), is_baseline=False)
    # Guard the guard: the two verdicts really do differ.
    assert passing.results["G6"].status is GateStatus.PASS
    assert failing.results["G6"].status is GateStatus.FAIL

    assert _gate_evidence_hash(evidence, passing) != _gate_evidence_hash(evidence, failing)


def test_evidence_hash_is_independent_of_yearly_insertion_order() -> None:
    """The digest is canonical: same content, different dict order, same hash.

    Probed: yearly values built in forward and in reversed insertion order hash
    identically today, and differently once ``sort_keys=True`` is dropped. Evidence
    is written by one process and verified by another, and ``json.dumps`` preserves
    insertion order, so without sorting a harmless reordering would invalidate every
    approval hash recorded against the same evidence.
    """
    forward = _evidence()
    reversed_years = replace(
        forward,
        yearly_mean_rank_ic={year: 0.04 for year in reversed(CONTRACT_YEARS)},
    )
    # Guard the guard: equal content, genuinely different insertion order.
    assert forward.yearly_mean_rank_ic == reversed_years.yearly_mean_rank_ic
    assert list(forward.yearly_mean_rank_ic) != list(reversed_years.yearly_mean_rank_ic)

    report = evaluate_oos_gates(forward, is_baseline=False)
    assert _gate_evidence_hash(forward, report) == _gate_evidence_hash(reversed_years, report)


def test_model_id_must_match_exactly_not_only_at_the_front() -> None:
    """A trailing newline must not turn an invalid model_id into a valid one.

    Probed: ``re.match`` accepts ``"sha256:<64 hex>\\n"`` because ``$`` also matches
    immediately before a trailing newline, while ``fullmatch`` refuses it. Model ids
    reach this function from files and command lines, where a stray newline is
    ordinary, and a content-addressed id that is one character too long is not the
    same object.
    """
    validate_model_id(MODEL_ID)  # control: the well-formed id passes

    for candidate in (MODEL_ID + "\n", MODEL_ID + " ", MODEL_ID + "0", "model-1"):
        with pytest.raises(ValueError, match="model_id"):
            validate_model_id(candidate)


def test_the_evidence_hash_binds_the_file_bytes_not_its_parsed_content(tmp_path: Path) -> None:
    """The digest is over the raw bytes, so two files that parse alike hash differently.

    Probed, and before this test the property was pinned in exactly one place: the CLI
    dry-run test's ``payload["evidence_hash"] == hashlib.sha256(raw).hexdigest()``. A pin
    carried by a single foreign test disappears when that test is rewritten. Hashing the
    parsed payload instead of the bytes would let the digest survive a re-serialisation of
    the evidence file, which is precisely what a content-addressed record must not do: an
    approval binds the artifact a reviewer actually looked at, and the same content written
    out differently is a different artifact.
    """
    original = _write_evidence(tmp_path, name="original.json")
    document = json.loads(original.read_text(encoding="utf-8"))
    reformatted = tmp_path / "reformatted.json"
    reformatted.write_text(json.dumps(document, indent=2), encoding="utf-8")
    # Guard the guard: same parsed content, genuinely different bytes.
    assert json.loads(reformatted.read_text(encoding="utf-8")) == document
    assert reformatted.read_bytes() != original.read_bytes()

    _, original_hash, _ = load_walk_forward_evidence(original)
    _, reformatted_hash, _ = load_walk_forward_evidence(reformatted)

    assert original_hash == hashlib.sha256(original.read_bytes()).hexdigest()
    assert reformatted_hash == hashlib.sha256(reformatted.read_bytes()).hexdigest()
    assert reformatted_hash != original_hash


# --- the other four hand-written lifecycle guards ----------------------------
#
# 续五十七's W-M probe found the structure rather than a single hole: the lifecycle
# table _ALLOWED_TRANSITIONS is consulted only by the public transition(), while five
# entry points call the private _transition -- which validates actor, reason and
# evidence hashes and never looks at the table -- each carrying its own hand-written
# "only X models can enter Y" guard. The DRAFT guard is pinned. These four were not:
# `git grep` over the whole test tree at that commit found no test asserting any of
# their messages. Each guard is the ONLY thing stopping the jump it names.


def _proof(operation: str, model_id: str = MODEL_ID, *, passed: bool | None = None):
    """Mint a valid one-shot proof through the registry's own internal authority hook."""
    return _issue_registry_workflow_proof(
        operation=operation,
        model_id=model_id,
        actor="evaluator",
        evidence_hash="e" * 64,
        passed=passed,
    )


# (entry-point name, the status it moves a model to) -- the five private paths.
_ENTRY_POINTS = (
    ("promote_walk_forward", ModelStatus.WALK_FORWARD_PASSED),
    ("enter_shadow", ModelStatus.SHADOW),
    ("_approve_locked_test", ModelStatus.LOCKED_TEST_APPROVED),
    ("_complete_locked_test", ModelStatus.LOCKED_TEST_PASSED),
    ("_approve_production", ModelStatus.PRODUCTION),
)

# The statuses reachable without the shadow-approval and locked-test machinery.
_REACHABLE = (ModelStatus.DRAFT, ModelStatus.WALK_FORWARD_PASSED, ModelStatus.RETIRED)


def _drive_to(registry: ModelRegistry, model_id: str, status: ModelStatus) -> None:
    """Park a freshly registered draft in ``status`` by the only route the registry allows."""
    if status is ModelStatus.DRAFT:
        return
    if status is ModelStatus.RETIRED:
        registry.transition(
            model_id,
            ModelStatus.RETIRED,
            actor="reviewer",
            evidence_hashes=(),
            reason="withdrawn before evaluation",
        )
        return
    registry.promote_walk_forward(model_id, actor="reviewer", evidence=_evidence())


def _drive_to_shadow(
    registry: ModelRegistry,
    model_id: str,
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reach SHADOW by the registry's own route: a signed approval that really verifies.

    _REACHABLE stops at WALK_FORWARD_PASSED because the forbidden-jump grid drives thirteen
    combinations and each needs a fresh registry, but SHADOW is reachable in this file too.
    That matters: 续七十 measured mutation X-F -- the production guard rewritten to require
    SHADOW instead of LOCKED_TEST_PASSED -- surviving here, and the reason is that nothing in
    this file ever calls _approve_production from SHADOW, which is the one source status that
    tells the correct guard from the mutated one.
    """
    monkeypatch.setenv("QRV2_SHADOW_APPROVAL_HMAC_KEY", "test-shadow-approval-key")
    monkeypatch.setenv("QRV2_SHADOW_DEPLOYER_ID", "shadow-controller")
    registry.promote_walk_forward(model_id, actor="reviewer", evidence=_evidence())
    approvals = ShadowDeploymentApprovalStore(
        tmp_path / "approvals",  # type: ignore[operator]
        signing_key=b"test-shadow-approval-key",
    )
    approval = approvals.issue(
        model_id=model_id,
        actor="risk-approver",
        walk_forward_evidence_hash=registry.history(model_id)[-1].evidence_hashes[0],
        artifact_bundle_hash="b" * 64,
        inference_contract_hash="c" * 64,
        v2test_drill_hash="d" * 64,
        rollback_plan_hash="e" * 64,
    )
    shadow = registry.enter_shadow(
        model_id,
        approval=approval,
        approval_root=tmp_path / "approvals",  # type: ignore[operator]
    )
    # Guard the guard: the fixture really is in SHADOW, so any refusal below is the guard's own.
    assert shadow.status is ModelStatus.SHADOW


def _invoke(registry: ModelRegistry, entry_point: str) -> ModelRecord:
    """Call one private entry point with inputs valid enough to reach its status guard."""
    if entry_point == "promote_walk_forward":
        return registry.promote_walk_forward(
            MODEL_ID, actor="reviewer", evidence=_evidence()
        )
    if entry_point == "enter_shadow":
        # The status guard runs before approval verification, so placeholders suffice.
        return registry.enter_shadow(
            MODEL_ID,
            approval=object(),  # type: ignore[arg-type]
            approval_root=Path("/nonexistent-approval-root"),
        )
    if entry_point == "_approve_locked_test":
        return registry._approve_locked_test(_proof("APPROVE_LOCKED_TEST"))
    if entry_point == "_complete_locked_test":
        return registry._complete_locked_test(_proof("COMPLETE_LOCKED_TEST", passed=True))
    return registry._approve_production(_proof("APPROVE_PRODUCTION"))


def test_enter_shadow_refuses_a_model_that_has_not_passed_walk_forward() -> None:
    """DRAFT -> SHADOW is not in the table, and this guard is the only thing saying so."""
    registry = ModelRegistry()
    registry.register_draft(MODEL_ID, actor="trainer")

    with pytest.raises(ValueError, match="only WALK_FORWARD_PASSED"):
        _invoke(registry, "enter_shadow")

    assert registry.get(MODEL_ID).status is ModelStatus.DRAFT
    assert len(registry.history(MODEL_ID)) == 1


def test_locked_test_approval_refuses_a_model_that_is_not_in_shadow() -> None:
    """WALK_FORWARD_PASSED -> LOCKED_TEST_APPROVED skips the entire shadow stage."""
    registry = ModelRegistry()
    registry.register_draft(MODEL_ID, actor="trainer")
    registry.promote_walk_forward(MODEL_ID, actor="reviewer", evidence=_evidence())

    with pytest.raises(ValueError, match="only SHADOW"):
        _invoke(registry, "_approve_locked_test")

    assert registry.get(MODEL_ID).status is ModelStatus.WALK_FORWARD_PASSED
    assert len(registry.history(MODEL_ID)) == 2


def test_completing_a_locked_test_refuses_a_model_never_approved_for_one() -> None:
    """DRAFT -> LOCKED_TEST_PASSED would record a locked-test verdict for a test never run."""
    registry = ModelRegistry()
    registry.register_draft(MODEL_ID, actor="trainer")

    with pytest.raises(ValueError, match="only LOCKED_TEST_APPROVED"):
        _invoke(registry, "_complete_locked_test")

    assert registry.get(MODEL_ID).status is ModelStatus.DRAFT
    assert len(registry.history(MODEL_ID)) == 1


def test_production_approval_refuses_a_model_that_has_not_passed_the_locked_test() -> None:
    """The shortest possible path to PRODUCTION is exactly the one this guard blocks.

    Probed: with the guard removed a model that is still DRAFT -- never evaluated, never
    shadowed, never locked-tested -- is recorded as PRODUCTION, because _transition does
    not consult the lifecycle table and nothing else stands between the two.
    """
    registry = ModelRegistry()
    registry.register_draft(MODEL_ID, actor="trainer")

    with pytest.raises(ValueError, match="only LOCKED_TEST_PASSED"):
        _invoke(registry, "_approve_production")

    assert registry.get(MODEL_ID).status is ModelStatus.DRAFT
    assert len(registry.history(MODEL_ID)) == 1


def test_production_approval_refuses_a_model_that_is_still_in_shadow(
    tmp_path: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The production guard names exactly one acceptable source status; this pins both sides.

    The test above drives from DRAFT, and a DRAFT is refused whether the guard requires
    LOCKED_TEST_PASSED, SHADOW, or nothing at all, so it cannot tell the correct guard from a
    wrong one. 续七十 measured the consequence for X-F: this file stayed green, and the only two
    failures in the whole chain-6 scope were
    test_independent_production_approval_requires_a_verified_locked_test_pass and
    test_production_pointer_accepts_only_the_signed_approval_for_its_exact_artifact -- each of
    which needs a legal production approval to succeed in order to test something else, and each
    of which died on its own happy path carrying the guard's message while asserting nothing
    about the guard. Their kill signal also proved to be a strict subset of X-K's, so a single
    refactor of that shared setup would have dropped both pins at once.

    Driving from SHADOW separates the two readings: the mutated guard waves it through, the real
    one refuses it. The second half walks the same model on to LOCKED_TEST_PASSED through the
    registry's own private entries and requires the approval to succeed, so the guard is pinned
    from both sides and "requires exactly LOCKED_TEST_PASSED" is the only reading left standing.
    """
    registry = ModelRegistry()
    registry.register_draft(MODEL_ID, actor="trainer")
    _drive_to_shadow(registry, MODEL_ID, tmp_path, monkeypatch)
    events_before = len(registry.history(MODEL_ID))

    with pytest.raises(ValueError, match="only LOCKED_TEST_PASSED"):
        _invoke(registry, "_approve_production")

    # A refused attempt leaves no trace, exactly as the four guard tests above require.
    assert registry.get(MODEL_ID).status is ModelStatus.SHADOW
    assert len(registry.history(MODEL_ID)) == events_before

    _invoke(registry, "_approve_locked_test")
    _invoke(registry, "_complete_locked_test")
    # Guard the guard: the ladder really did reach a passed locked test, so the approval below
    # exercises the guard's accepting side instead of failing for some earlier reason.
    assert registry.get(MODEL_ID).status is ModelStatus.LOCKED_TEST_PASSED

    approved = _invoke(registry, "_approve_production")

    assert approved.status is ModelStatus.PRODUCTION
    assert registry.get(MODEL_ID).status is ModelStatus.PRODUCTION


def test_no_private_entry_point_permits_a_jump_the_lifecycle_table_forbids() -> None:
    """The table is the policy; the five hand-written guards must not contradict it.

    Derived from _ALLOWED_TRANSITIONS rather than from a second copy of it, so the test
    follows the policy instead of restating it: for every reachable status and every
    private entry point whose target the table forbids, the entry point must refuse and
    must leave both the record and the append-only history untouched.
    """
    checked = 0
    for source in _REACHABLE:
        for entry_point, target in _ENTRY_POINTS:
            if target in _ALLOWED_TRANSITIONS[source]:
                continue  # permitted by policy; the happy paths are tested elsewhere
            registry = ModelRegistry()
            registry.register_draft(MODEL_ID, actor="trainer")
            _drive_to(registry, MODEL_ID, source)
            events_before = len(registry.history(MODEL_ID))

            with pytest.raises(ValueError):
                _invoke(registry, entry_point)

            assert registry.get(MODEL_ID).status is source, (source, entry_point)
            assert len(registry.history(MODEL_ID)) == events_before, (source, entry_point)
            checked += 1

    # Guard the guard: this really did walk the whole forbidden grid, not an empty loop.
    # 4 forbidden jumps from DRAFT, 4 from WALK_FORWARD_PASSED, 5 from RETIRED.
    assert checked == 13


def test_public_transition_refuses_every_pair_the_lifecycle_table_forbids() -> None:
    """The one entry point that does consult the table must consult it for every pair.

    Existing tests check two forbidden pairs. This checks all of them from the three
    reachable statuses, and insists the refusal is the table's own message rather than
    one of the later, target-specific refusals -- so removing the table check is caught
    even where a downstream rule would have refused the same jump for another reason.
    """
    checked = 0
    for source in _REACHABLE:
        registry = ModelRegistry()
        registry.register_draft(MODEL_ID, actor="trainer")
        _drive_to(registry, MODEL_ID, source)
        for target in ModelStatus:
            if target in _ALLOWED_TRANSITIONS[source]:
                continue
            with pytest.raises(ValueError, match="cannot transition"):
                registry.transition(
                    MODEL_ID,
                    target,
                    actor="reviewer",
                    evidence_hashes=("e" * 64,),
                    reason="probe",
                )
            assert registry.get(MODEL_ID).status is source
            checked += 1

    # Guard the guard: 6 forbidden targets from DRAFT, 6 from WALK_FORWARD_PASSED,
    # 8 from RETIRED (whose allowed set is empty).
    assert checked == 20


def test_the_lifecycle_table_is_the_reviewed_policy() -> None:
    """A deliberate change-detector, because nothing else would notice a loosened row.

    The table is duplicated by five hand-written guards and the two encodings are never
    cross-checked. A loosened row is invisible to every behaviour test: for the pairs
    probed here the public transition() refuses the target for its own reason before the
    table matters (a DRAFT allowed to reach PRODUCTION still hits "locked-test and
    production states require their dedicated workflow"), and the private entry points
    never read the table at all. So this test copies the policy on purpose -- it is the
    only thing that makes drift between the two writings visible.
    """
    assert {
        ModelStatus.DRAFT: frozenset({ModelStatus.WALK_FORWARD_PASSED, ModelStatus.RETIRED}),
        ModelStatus.WALK_FORWARD_PASSED: frozenset({ModelStatus.SHADOW, ModelStatus.RETIRED}),
        ModelStatus.SHADOW: frozenset(
            {ModelStatus.LOCKED_TEST_APPROVED, ModelStatus.RETIRED}
        ),
        ModelStatus.LOCKED_TEST_APPROVED: frozenset(
            {ModelStatus.LOCKED_TEST_PASSED, ModelStatus.LOCKED_TEST_FAILED, ModelStatus.RETIRED}
        ),
        ModelStatus.LOCKED_TEST_PASSED: frozenset({ModelStatus.PRODUCTION, ModelStatus.RETIRED}),
        ModelStatus.LOCKED_TEST_FAILED: frozenset({ModelStatus.RETIRED}),
        ModelStatus.PRODUCTION: frozenset({ModelStatus.RETIRED}),
        ModelStatus.RETIRED: frozenset(),
    } == _ALLOWED_TRANSITIONS


def _gate_evidence_payload(yearly: dict[str, float]) -> dict[str, object]:
    return {
        "mean_rank_ic": 0.04,
        "hac_t_statistic": 2.5,
        "yearly_mean_rank_ic": yearly,
        "candidate_delta_rank_ic": 0.006,
        "valid_test_day_count": 6,
        "score_health": [
            {"coverage": 0.95, "unique_score_count": 30, "score_std": 0.1} for _ in range(6)
        ],
    }


def test_walk_forward_evidence_accepts_the_current_fold_contract_years(tmp_path: Path) -> None:
    """The F1-F4 contract's ordinary test years are 2022-2025; promotion accepts exactly those.

    A v2 candidate that passes all six gates must actually be promotable. Before the loader
    derived its required years from the fold contract it demanded 2020-2025, so this document
    -- the only shape the current contract can produce -- was rejected and nothing trained
    under F1-F4 could ever reach the registry.
    """
    assert CONTRACT_YEARS == (2022, 2023, 2024, 2025)
    path = _write_evidence(tmp_path)
    evidence, digest, report = load_walk_forward_evidence(path)
    assert set(evidence.yearly_mean_rank_ic) == set(CONTRACT_YEARS)
    assert len(digest) == 64
    assert all(result.status is GateStatus.PASS for result in report.results.values())


def test_walk_forward_evidence_rejects_the_retired_fold_grid_years(tmp_path: Path) -> None:
    """2020-2025 belongs to the retired F01-F06 grid and must not be promotable."""
    stale = _gate_evidence_payload({str(year): 0.04 for year in range(2020, 2026)})
    path = _write_evidence(tmp_path, gate_evidence=stale)
    with pytest.raises(ValueError, match="yearly values for 2022-2025"):
        load_walk_forward_evidence(path)


def test_walk_forward_evidence_rejects_a_missing_contract_year(tmp_path: Path) -> None:
    """Fail closed on partial coverage: a candidate cannot drop its worst test year."""
    partial = _gate_evidence_payload({str(year): 0.04 for year in CONTRACT_YEARS[:-1]})
    path = _write_evidence(tmp_path, gate_evidence=partial)
    with pytest.raises(ValueError, match="yearly values for 2022-2025"):
        load_walk_forward_evidence(path)
