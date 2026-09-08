"""Verify that a run's signed candidate predictions are what its own bundles actually produce.

The frozen evidence chain already binds a prediction to a model: the prediction artifact is signed
over its score bytes and its metadata, ``_require_candidate_evidence`` matches its selection id,
model id, fold and dataset against the frozen selection, and ``_require_verified_model_bundle``
matches that selection's bundle digest against a re-hashed bundle. What none of that establishes is
that the scores are the ones the bundle computes. The chain proves attribution -- these scores are
recorded against this bundle -- not reproduction.

That gap is not hypothetical. Whoever holds the trust root can sign arbitrary score bytes under a
correct metadata payload, and every existing check would pass. Recomputation is the only thing that
closes it, and it is cheap: the bundle is immutable, the TEST frame is an immutable verified
artifact, and the scoring rule is deterministic.

This module does the recomputation. It re-derives the run's registry identity (which re-hashes and
re-verifies all four bundles), requires that identity to be promoted, loads each fold's bundle
through the serving path under that authorisation, scores the fold's frozen TEST split, and compares
value for value against the signed candidate prediction artifact. Any mismatch is a refusal.

Scope: the LightGBM families (M0, M0R), whose scoring is a pure function of the bundle and the
frame. The torch families additionally need the run's FoldPreprocessor state and batch construction
to reproduce a score, which lives in the candidate driver rather than in a library, so they are
refused with an explicit reason instead of being silently skipped.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from research_v2.datasets.storage import DatasetArtifactStore
from research_v2.evaluation.selection_evidence import (
    FrozenPredictionStore,
    SelectionEvidenceVerifier,
)
from research_v2.inference.lgbm_artifact_loader import (
    LGBM_MODEL_CODE_VERSIONS,
    load_lgbm_inference_artifact,
    score_lgbm_artifact,
)
from research_v2.registry.promotion_evidence import (
    _load_full_report,
    _prediction_ids_by_fold_and_kind,
    verify_run_promotion,
)
from research_v2.training.model_bundle import ModelArtifactBundleStore

if TYPE_CHECKING:
    from research_v2.registry.registry_store import ModelRegistryStore

_BUSINESS_KEYS = ("signal_date", "ts_code")


class ReproductionError(ValueError):
    """A run's stored predictions are not what its bundles produce, and how they differ."""


@dataclass(frozen=True)
class FoldReproduction:
    fold_id: str
    model_artifact_id: str
    family: str
    row_count: int
    max_absolute_difference: float
    identical: bool


def verify_candidate_predictions_reproduce(
    workspace: Path,
    report_path: Path,
    registry: ModelRegistryStore,
) -> dict[str, object]:
    """Recompute every fold's candidate scores from its bundle and compare to the signed artifact.

    Raises ``ReproductionError`` on any difference, on a row-set difference, on a run that is not
    promoted, and for families this module cannot yet reproduce.
    """
    promoted = verify_run_promotion(workspace, report_path, registry)
    report = _load_full_report(report_path)
    folds_in_report = report.get("folds")
    if not isinstance(folds_in_report, list):
        raise ReproductionError("run report has no folds list")
    fold_meta: dict[str, dict[str, str]] = {}
    for entry in folds_in_report:
        if not isinstance(entry, dict):
            raise ReproductionError("run report fold entry is not an object")
        fold_id = entry.get("fold_id")
        model_artifact = entry.get("model_id")
        dataset_id = entry.get("dataset_id")
        if (
            not isinstance(fold_id, str)
            or not isinstance(model_artifact, str)
            or not isinstance(dataset_id, str)
        ):
            raise ReproductionError("run report fold entry lacks fold_id/model_id/dataset_id")
        fold_meta[fold_id] = {"model_id": model_artifact, "dataset_id": dataset_id}
    if set(fold_meta) != set(promoted.fold_artifact_ids):
        raise ReproductionError("run report folds disagree with the promoted run's folds")

    prediction_ids = _prediction_ids_by_fold_and_kind(workspace)
    dataset_store = DatasetArtifactStore(workspace / "datasets")
    model_store = ModelArtifactBundleStore(workspace / "models")
    selections = SelectionEvidenceVerifier.from_environment(workspace / "selections")
    predictions = FrozenPredictionStore.from_environment(workspace / "predictions")

    results: list[FoldReproduction] = []
    for fold_id in sorted(fold_meta):
        meta = fold_meta[fold_id]
        model_artifact = meta["model_id"]
        if promoted.fold_artifact_ids[fold_id] != model_artifact:
            raise ReproductionError(
                f"{fold_id} report names {model_artifact} but the promoted run names "
                f"{promoted.fold_artifact_ids[fold_id]}"
            )
        card_path = model_store.bundle_path(model_artifact) / "model_card.json"
        card = json.loads(card_path.read_text(encoding="utf-8"))
        family = card.get("family")
        if family not in LGBM_MODEL_CODE_VERSIONS:
            raise ReproductionError(
                f"{fold_id} bundle family {family!r} cannot be reproduced by this verifier: the "
                "torch families need the run's FoldPreprocessor state and batch construction, "
                "which live in the candidate driver rather than in a library"
            )
        test = dataset_store.load_split(
            meta["dataset_id"], "test", expected_fold_id=fold_id
        )
        loaded = load_lgbm_inference_artifact(
            model_store,
            model_artifact,
            expected_dataset_id=f"{meta['dataset_id']}:train",
            expected_feature_order=_feature_order(model_store, model_artifact),
            promoted_run=promoted,
        )
        recomputed = score_lgbm_artifact(loaded, test)
        try:
            candidate_id = prediction_ids[(fold_id, "candidate")]
        except KeyError as error:
            raise ReproductionError(
                f"{fold_id} has no signed candidate prediction artifact"
            ) from error
        stored = predictions.load(candidate_id, selections=selections).scores

        results.append(
            _compare_scores(recomputed, stored, fold_id=fold_id, model_artifact_id=model_artifact,
                            family=str(family))
        )
    if not results:
        raise ReproductionError("no folds were reproduced")
    return {
        "model_id": promoted.model_id,
        "status": promoted.status.value,
        "folds": [
            {
                "fold_id": item.fold_id,
                "model_artifact_id": item.model_artifact_id,
                "family": item.family,
                "row_count": item.row_count,
                "max_absolute_difference": item.max_absolute_difference,
                "identical": item.identical,
            }
            for item in results
        ],
        "total_rows": sum(item.row_count for item in results),
        "reproduced": True,
    }


def _compare_scores(
    recomputed: pd.DataFrame,
    stored: pd.DataFrame,
    *,
    fold_id: str,
    model_artifact_id: str,
    family: str,
) -> FoldReproduction:
    """Compare recomputed against stored scores exactly, refusing any difference at all.

    Extracted from the fold loop so its non-vacuity is testable without a signed workspace. That
    matters more here than usual: a verifier that always passes is worse than no verifier, because
    it turns "the scores were never checked" into "the scores were checked and matched". The
    comparison is exact equality on an outer join of business keys, so a differing value, a missing
    row and an extra row each fail distinctly and say which.
    """
    left = _normalized(recomputed, "recomputed")
    right = _normalized(stored, "stored")
    merged = left.merge(
        right,
        on=list(_BUSINESS_KEYS),
        how="outer",
        validate="one_to_one",
        suffixes=("_recomputed", "_stored"),
        indicator=True,
    )
    unmatched = merged.loc[merged["_merge"] != "both"]
    if not unmatched.empty:
        only_recomputed = int((unmatched["_merge"] == "left_only").sum())
        only_stored = int((unmatched["_merge"] == "right_only").sum())
        raise ReproductionError(
            f"{fold_id} score row sets differ: {only_recomputed} keys only in the recomputed "
            f"scores, {only_stored} only in the stored artifact"
        )
    difference = (merged["score_recomputed"] - merged["score_stored"]).abs()
    if not np.isfinite(difference.to_numpy()).all():
        raise ReproductionError(f"{fold_id} produced non-finite score differences")
    max_difference = float(difference.max()) if len(merged) else 0.0
    if max_difference != 0.0:
        raise ReproductionError(
            f"{fold_id} stored candidate scores differ from what bundle {model_artifact_id} "
            f"computes: max absolute difference {max_difference!r} over {len(merged)} rows"
        )
    return FoldReproduction(
        fold_id=fold_id,
        model_artifact_id=model_artifact_id,
        family=family,
        row_count=int(len(merged)),
        max_absolute_difference=max_difference,
        identical=True,
    )


def _feature_order(store: ModelArtifactBundleStore, model_artifact_id: str) -> tuple[str, ...]:
    document = json.loads(
        (store.bundle_path(model_artifact_id) / "feature_order.json").read_text(encoding="utf-8")
    )
    order = document.get("feature_order")
    if not isinstance(order, list) or not order:
        raise ReproductionError(f"{model_artifact_id} has no usable feature_order")
    return tuple(str(name) for name in order)


def _normalized(frame: pd.DataFrame, label: str) -> pd.DataFrame:
    required = set(_BUSINESS_KEYS) | {"score"}
    if missing := sorted(required.difference(frame.columns)):
        raise ReproductionError(f"{label} scores are missing columns: {missing}")
    out = frame.loc[:, list(_BUSINESS_KEYS) + ["score"]].copy()
    out["signal_date"] = pd.to_datetime(out["signal_date"], errors="raise").dt.date
    out["score"] = pd.to_numeric(out["score"], errors="coerce")
    if out["score"].isna().any():
        raise ReproductionError(f"{label} scores contain non-numeric values")
    if out.duplicated(list(_BUSINESS_KEYS)).any():
        raise ReproductionError(f"{label} scores contain duplicate business keys")
    return out


__all__ = [
    "FoldReproduction",
    "ReproductionError",
    "verify_candidate_predictions_reproduce",
]
