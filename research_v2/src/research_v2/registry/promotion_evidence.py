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
from pathlib import Path
from typing import cast

from research_v2.datasets.storage import DatasetArtifactStore
from research_v2.evaluation.model_walk_forward import (
    ModelOosFold,
    evaluate_model_walk_forward,
)
from research_v2.evaluation.walk_forward import ORDINARY_TEST_FOLDS
from research_v2.registry.model_registry import (
    build_walk_forward_evidence_document,
    load_walk_forward_evidence,
)
from research_v2.training.model_bundle import ModelArtifactBundleStore

EVIDENCE_SCHEMA_VERSION = 1
EVIDENCE_ROLE = "ORDINARY_TEST"
PROMOTION_EVIDENCE_FILENAME = "promotion_evidence.json"


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


def emit_promotion_evidence(
    workspace: Path,
    report_path: Path,
    out_path: Path,
    *,
    signing_key: bytes,
    minimum_cross_section: int = 30,
) -> dict[str, object]:
    """Re-verify a completed run and write its registry promotion-evidence document.

    Raises ``PromotionEvidenceError`` if the workspace is incomplete or inconsistent, and
    whatever ``load_walk_forward_evidence`` raises -- a ``ValueError`` naming the failed gates
    -- if the candidate does not actually pass, so a failing run never yields a document.
    """
    if not isinstance(signing_key, bytes) or not signing_key:
        raise PromotionEvidenceError("promotion evidence requires a non-empty signing key")
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
