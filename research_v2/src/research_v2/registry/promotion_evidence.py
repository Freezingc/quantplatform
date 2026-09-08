"""Emit the lifecycle registry's promotion-evidence document from a completed run workspace.

``load_walk_forward_evidence`` is the only route from WALK_FORWARD_PASSED onward, and nothing
in the codebase produced the document it consumes: the candidate driver writes a human-facing
report whose shape the loader rejects on its first field. That left every candidate unable to
reach the registry, including one that passed all six gates.

This is that producer, and it re-verifies rather than trusts. The signed selection, model and
prediction artifacts are re-loaded through ``evaluate_model_walk_forward`` -- which checks
every bundle digest and HMAC signature -- the gates are recomputed from raw numbers, and the
result is cross-checked against the driver's own report before anything is written. It reads
only the workspace, never the database, so it can be pointed at any completed run, including
one that predates this module.

Emission is not a trust decision. The registry re-derives G1-G6 from the raw numbers in the
document and never reads a reported gate status, so writing this file cannot promote a model
that does not actually pass; a failing candidate raises here instead of producing a document.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, cast

from research_v2.datasets.storage import DatasetArtifactStore
from research_v2.evaluation.model_walk_forward import (
    ModelOosFold,
    evaluate_model_walk_forward,
)
from research_v2.evaluation.walk_forward import ORDINARY_TEST_FOLDS
from research_v2.registry.model_registry import (
    ModelStatus,
    build_walk_forward_evidence_document,
    derive_walk_forward_model_id,
    load_walk_forward_evidence,
)
from research_v2.training.model_bundle import ModelArtifactBundleStore

if TYPE_CHECKING:
    from research_v2.registry.registry_store import ModelRegistryStore

# Lifecycle states that mean "this run earned the right to be served". DRAFT has not been
# evaluated, RETIRED has been withdrawn, and LOCKED_TEST_FAILED is a terminal refusal.
SERVABLE_STATUSES = frozenset(
    {
        ModelStatus.WALK_FORWARD_PASSED,
        ModelStatus.SHADOW,
        ModelStatus.LOCKED_TEST_APPROVED,
        ModelStatus.LOCKED_TEST_PASSED,
        ModelStatus.PRODUCTION,
    }
)

EVIDENCE_SCHEMA_VERSION = 1
EVIDENCE_ROLE = "ORDINARY_TEST"
PROMOTION_EVIDENCE_FILENAME = "promotion_evidence.json"
# evaluate_model_walk_forward builds its selection and prediction stores from the environment,
# so the trust root has to be configured before this module can re-verify anything. Both names
# are checked up front because the design is one trust root: the driver signs predictions with
# the selection key, and a run that exports only one of them fails deep inside the evaluator
# with a message that does not say which artifact it was reading.
REQUIRED_TRUST_ROOT_ENV = (
    "QRV2_SELECTION_EVIDENCE_HMAC_KEY",
    "QRV2_PREDICTION_EVIDENCE_HMAC_KEY",
)


def _require_trust_root() -> None:
    missing = [name for name in REQUIRED_TRUST_ROOT_ENV if not os.environ.get(name, "")]
    if missing:
        raise PromotionEvidenceError(
            "promotion evidence re-verification requires the trust root: "
            + ", ".join(missing)
            + " must be set (both hold the same key by design)"
        )


class PromotionEvidenceError(ValueError):
    """The workspace cannot yield promotable evidence, and the reason must be explicit."""


def _prediction_ids_by_fold_and_kind(workspace: Path) -> dict[tuple[str, str], str]:
    """Index the workspace's frozen prediction artifacts by (fold, kind)."""
    root = workspace / "predictions"
    if not root.is_dir():
        raise PromotionEvidenceError(f"no frozen prediction artifacts under {root}")
    index: dict[tuple[str, str], str] = {}
    for manifest in sorted(root.glob("*/manifest.json")):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        metadata = payload.get("metadata")
        prediction_id = payload.get("prediction_id")
        if not isinstance(metadata, dict) or not isinstance(prediction_id, str):
            raise PromotionEvidenceError(f"unreadable prediction manifest: {manifest}")
        fold_id = metadata.get("fold_id")
        kind = metadata.get("kind")
        if not isinstance(fold_id, str) or not isinstance(kind, str):
            raise PromotionEvidenceError(f"prediction manifest lacks fold/kind: {manifest}")
        key = (fold_id, kind)
        if key in index and index[key] != prediction_id:
            raise PromotionEvidenceError(
                f"workspace holds two {kind} predictions for {fold_id}: "
                f"{index[key]} and {prediction_id}"
            )
        index[key] = prediction_id
    return index


def _load_full_report(report_path: Path) -> dict[str, object]:
    if not report_path.exists():
        raise PromotionEvidenceError(f"run report does not exist: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(report, dict):
        raise PromotionEvidenceError(f"run report is not an object: {report_path}")
    if report.get("mode") != "full":
        raise PromotionEvidenceError(
            f"promotion evidence requires a full run report, got mode={report.get('mode')!r}"
        )
    return report


def _cross_check_against_report(report: dict[str, object], out_path: Path) -> None:
    """Refuse to emit evidence whose recomputed numbers disagree with the driver's report.

    The comparison is deliberately on quantities both sides define identically: the six gate
    statuses and observed values, the pooled mean RankIC, the HAC t statistic, the candidate
    delta, and the per-year means. It does NOT compare the report's ``oos.valid_day_count``
    against the evidence's ``valid_test_day_count``, because those are different measures and
    equating them would be a false alarm on every run: the report's is the number of days with
    a usable RankIC (945 on the v2 folds) while the evidence's is the number of label days wide
    enough to score (969), and their ratio is exactly what G6 reports.
    """
    oos = report.get("oos")
    if not isinstance(oos, dict):
        raise PromotionEvidenceError("run report has no oos block")
    reloaded, _, recomputed = load_walk_forward_evidence(out_path)

    reported_gates = oos.get("gates")
    if not isinstance(reported_gates, dict):
        raise PromotionEvidenceError("run report has no oos.gates block")
    for gate, recomputed_result in recomputed.results.items():
        reported = reported_gates.get(gate)
        if not isinstance(reported, dict):
            raise PromotionEvidenceError(f"run report oos.gates.{gate} is not an object")
        if reported.get("status") != recomputed_result.status.value:
            raise PromotionEvidenceError(
                f"recomputed {gate}={recomputed_result.status.value} disagrees with the run "
                f"report's {gate}={reported.get('status')!r}"
            )
        reported_observed = reported.get("observed_value")
        recomputed_observed = recomputed_result.observed_value
        if (reported_observed is None) != (recomputed_observed is None):
            raise PromotionEvidenceError(f"recomputed {gate} observed value presence differs")
        if (
            isinstance(reported_observed, (int, float))
            and not isinstance(reported_observed, bool)
            and recomputed_observed is not None
            and abs(float(reported_observed) - float(recomputed_observed)) > 1e-9
        ):
            raise PromotionEvidenceError(
                f"recomputed {gate} observed={recomputed_observed!r} disagrees with the run "
                f"report's {reported_observed!r}"
            )

    scalars: tuple[tuple[str, float | None], ...] = (
        ("mean_rank_ic", reloaded.mean_rank_ic),
        ("hac_t_statistic", reloaded.hac_t_statistic),
        ("delta_rank_ic", reloaded.candidate_delta_rank_ic),
    )
    for field, recomputed_value in scalars:
        if recomputed_value is None:
            raise PromotionEvidenceError(f"recomputed oos.{field} is missing")
        reported_value = oos.get(field)
        if not isinstance(reported_value, (int, float)) or isinstance(reported_value, bool):
            raise PromotionEvidenceError(f"run report oos.{field} is not a number")
        if abs(float(reported_value) - float(recomputed_value)) > 1e-12:
            raise PromotionEvidenceError(
                f"recomputed oos.{field}={recomputed_value!r} disagrees with the run report's "
                f"{reported_value!r}"
            )

    reported_yearly = oos.get("yearly_mean_rank_ic")
    if not isinstance(reported_yearly, dict):
        raise PromotionEvidenceError("run report has no oos.yearly_mean_rank_ic block")
    if set(reported_yearly) != {str(year) for year in reloaded.yearly_mean_rank_ic}:
        raise PromotionEvidenceError(
            f"recomputed yearly years {sorted(reloaded.yearly_mean_rank_ic)} disagree with the "
            f"run report's {sorted(reported_yearly)}"
        )
    for year, recomputed_value in reloaded.yearly_mean_rank_ic.items():
        reported_value = reported_yearly[str(year)]
        if not isinstance(reported_value, (int, float)) or isinstance(reported_value, bool):
            raise PromotionEvidenceError(f"run report yearly {year} is not a number")
        if abs(float(reported_value) - float(recomputed_value)) > 1e-12:
            raise PromotionEvidenceError(
                f"recomputed yearly {year}={recomputed_value!r} disagrees with the run "
                f"report's {reported_value!r}"
            )


def walk_forward_model_id(workspace: Path, report_path: Path) -> str:
    """Derive the registry model_id for a completed walk-forward run.

    The registry names a model by one ``sha256:<64 hex>`` id while a walk-forward run trains
    one bundle per ordinary fold, so the id has to be minted from the folds. It is derived from
    the *verified* bundle digests -- ``bundle_digest`` re-hashes and re-checks every file -- so
    asking for the id is itself an integrity check on the four models, and a tampered or
    truncated bundle raises here rather than producing a plausible-looking identity.
    """
    report = _load_full_report(report_path)
    folds_in_report = report.get("folds")
    if not isinstance(folds_in_report, list):
        raise PromotionEvidenceError("run report has no folds list")
    store = ModelArtifactBundleStore(workspace / "models")
    digests: dict[str, str] = {}
    for entry in folds_in_report:
        if not isinstance(entry, dict):
            raise PromotionEvidenceError("run report fold entry is not an object")
        fold_id = entry.get("fold_id")
        model_artifact = entry.get("model_id")
        if not isinstance(fold_id, str) or not isinstance(model_artifact, str):
            raise PromotionEvidenceError("run report fold entry lacks fold_id/model_id")
        digests[fold_id] = store.bundle_digest(model_artifact)
    if set(digests) != set(ORDINARY_TEST_FOLDS):
        raise PromotionEvidenceError(
            f"workspace holds bundles for {sorted(digests)}, expected "
            f"{list(ORDINARY_TEST_FOLDS)}"
        )
    return derive_walk_forward_model_id(digests)


@dataclass(frozen=True)
class PromotedRun:
    """A walk-forward run whose lifecycle status was verified against the registry.

    ``model_id`` is the registry identity derived from the run's four verified bundle digests;
    ``fold_artifact_ids`` maps each ordinary fold to the bundle that run trained, so a caller can
    check that the specific bundle it is about to load belongs to this promoted run rather than to
    some other run sharing the workspace.
    """

    model_id: str
    status: ModelStatus
    fold_artifact_ids: Mapping[str, str]


def verify_run_promotion(
    workspace: Path,
    report_path: Path,
    registry: ModelRegistryStore,
    *,
    servable_statuses: frozenset[ModelStatus] = SERVABLE_STATUSES,
) -> PromotedRun:
    """Prove a run is promoted, using the registry rather than anything the run said about itself.

    This exists because the model card's ``model_status`` field cannot serve as that proof. The
    candidate driver writes ``"model_status": "WALK_FORWARD_PASSED"`` into every bundle at five
    sites, and it must write the bundle before the OOS evaluation runs, because that evaluation
    consumes the frozen predictions the bundle produces. So the field is true by construction for
    every bundle of every run, including runs that failed G4: the m0_v3 workspace, whose report
    says ``all_passed: false`` at delta +0.004202, carries cards reading WALK_FORWARD_PASSED on all
    four folds, and an inference loader that gates on that field gates on nothing.

    The registry is the only authority that can answer the question, because it is the only place
    the status was derived from recomputed gates instead of asserted by the artifact's producer.
    """
    report = _load_full_report(report_path)
    folds_in_report = report.get("folds")
    if not isinstance(folds_in_report, list):
        raise PromotionEvidenceError("run report has no folds list")
    fold_artifact_ids: dict[str, str] = {}
    for entry in folds_in_report:
        if not isinstance(entry, dict):
            raise PromotionEvidenceError("run report fold entry is not an object")
        fold_id = entry.get("fold_id")
        model_artifact = entry.get("model_id")
        if not isinstance(fold_id, str) or not isinstance(model_artifact, str):
            raise PromotionEvidenceError("run report fold entry lacks fold_id/model_id")
        fold_artifact_ids[fold_id] = model_artifact
    if set(fold_artifact_ids) != set(ORDINARY_TEST_FOLDS):
        raise PromotionEvidenceError(
            f"run report covers {sorted(fold_artifact_ids)}, expected "
            f"{list(ORDINARY_TEST_FOLDS)}"
        )
    # Deriving the id re-hashes and re-verifies all four bundles, so a tampered bundle fails here.
    model_id = walk_forward_model_id(workspace, report_path)
    try:
        record = registry.registry.get(model_id)
    except ValueError as error:
        raise PromotionEvidenceError(
            f"run is not registered in the model registry, so it cannot be served: {error}"
        ) from error
    if record.status not in servable_statuses:
        raise PromotionEvidenceError(
            f"model {model_id} is {record.status.value}, which is not a servable status "
            f"(one of {sorted(status.value for status in servable_statuses)})"
        )
    return PromotedRun(
        model_id=model_id,
        status=record.status,
        fold_artifact_ids=MappingProxyType(fold_artifact_ids),
    )


def emit_promotion_evidence(
    workspace: Path,
    report_path: Path,
    out_path: Path,
    *,
    minimum_cross_section: int = 30,
) -> dict[str, object]:
    """Re-verify a completed run and write its registry promotion-evidence document.

    Requires the trust root in the environment (see ``REQUIRED_TRUST_ROOT_ENV``); the stores
    that verify signatures are built from it inside ``evaluate_model_walk_forward``.

    Raises ``PromotionEvidenceError`` if the trust root is missing or the workspace is
    incomplete or inconsistent, and whatever ``load_walk_forward_evidence`` raises -- a
    ``ValueError`` naming the failed gates -- if the candidate does not actually pass, so a
    failing run never yields a document.
    """
    _require_trust_root()
    report = _load_full_report(report_path)
    folds_in_report = report.get("folds")
    if not isinstance(folds_in_report, list):
        raise PromotionEvidenceError("run report has no folds list")
    selection_by_fold: dict[str, str] = {}
    for entry in folds_in_report:
        if not isinstance(entry, dict):
            raise PromotionEvidenceError("run report fold entry is not an object")
        fold_id = entry.get("fold_id")
        selection_id = entry.get("model_selection_id")
        if not isinstance(fold_id, str) or not isinstance(selection_id, str):
            raise PromotionEvidenceError("run report fold entry lacks fold_id/model_selection_id")
        selection_by_fold[fold_id] = selection_id
    if set(selection_by_fold) != set(ORDINARY_TEST_FOLDS):
        raise PromotionEvidenceError(
            f"run report covers {sorted(selection_by_fold)}, expected "
            f"{list(ORDINARY_TEST_FOLDS)}"
        )

    predictions = _prediction_ids_by_fold_and_kind(workspace)
    oos_folds: dict[str, ModelOosFold] = {}
    for fold_id in ORDINARY_TEST_FOLDS:
        try:
            candidate_id = predictions[(fold_id, "candidate")]
            baseline_id = predictions[(fold_id, "best_simple")]
        except KeyError as error:
            raise PromotionEvidenceError(
                f"workspace is missing a frozen {error} prediction for {fold_id}"
            ) from error
        oos_folds[fold_id] = ModelOosFold(
            model_selection_id=selection_by_fold[fold_id],
            candidate_prediction_id=candidate_id,
            best_simple_prediction_id=baseline_id,
        )

    result = evaluate_model_walk_forward(
        oos_folds,
        dataset_store=DatasetArtifactStore(workspace / "datasets"),
        model_store=ModelArtifactBundleStore(workspace / "models"),
        selection_root=workspace / "selections",
        prediction_root=workspace / "predictions",
        minimum_cross_section=minimum_cross_section,
    )
    document = build_walk_forward_evidence_document(result.evaluation)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )
    # Loading it back is the point: this is where a failing candidate is refused, and where
    # the recomputed numbers are checked against the report the run itself wrote.
    _cross_check_against_report(report, out_path)
    evidence, digest, gate_report = load_walk_forward_evidence(out_path)
    return {
        "model_id": walk_forward_model_id(workspace, report_path),
        "model_family": result.model_family,
        "evidence_path": str(out_path),
        "evidence_sha256": digest,
        "mean_rank_ic": evidence.mean_rank_ic,
        "candidate_delta_rank_ic": evidence.candidate_delta_rank_ic,
        "valid_test_day_count": evidence.valid_test_day_count,
        "gates": {
            gate: gate_result.status.value for gate, gate_result in gate_report.results.items()
        },
        "score_health_days": len(evidence.score_health),
        "reported_family": cast(str, report.get("family")),
    }
