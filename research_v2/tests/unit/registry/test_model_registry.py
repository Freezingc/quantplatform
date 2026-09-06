from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from research_v2.evaluation.gates import (
    GateEvidence,
    GateStatus,
    ScoreHealthEvidence,
    evaluate_oos_gates,
)
from research_v2.monitoring.shadow import ShadowDeploymentApprovalStore
from research_v2.registry.model_registry import (
    ModelRegistry,
    ModelStatus,
    _gate_evidence_hash,
    load_walk_forward_evidence,
    validate_model_id,
)

MODEL_ID = "sha256:" + "a" * 64
CHILD_MODEL_ID = "sha256:" + "b" * 64


def _evidence(*, valid_score_days: int = 6) -> GateEvidence:
    return GateEvidence(
        mean_rank_ic=0.04,
        hac_t_statistic=2.5,
        yearly_mean_rank_ic={year: 0.04 for year in range(2020, 2026)},
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
            "yearly_mean_rank_ic": {str(year): 0.04 for year in range(2020, 2026)},
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
        yearly_mean_rank_ic={year: 0.04 for year in reversed(range(2020, 2026))},
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
