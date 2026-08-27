"""Verified F01-F06 candidate-model evaluation against frozen best-simple evidence."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from research_v2.datasets.storage import DatasetArtifactStore
from research_v2.evaluation.model_evaluator import FrozenOosEvaluation
from research_v2.evaluation.selection_evidence import (
    FrozenPredictionArtifact,
    FrozenPredictionStore,
    ModelSelectionEvidence,
    SelectionEvidenceVerifier,
)
from research_v2.evaluation.walk_forward import (
    ORDINARY_TEST_FOLDS,
    OosFold,
    evaluate_walk_forward_oos,
)
from research_v2.training.model_bundle import ModelArtifactBundleStore

MODEL_FAMILIES = ("M1", "M2", "M3")
_MODEL_CARD_FAMILIES = {
    "M1": {"M1", "RESIDUAL_MLP"},
    "M2": {"M2", "TEMPORAL"},
    "M3": {"M3", "SET_ATTENTION"},
}


@dataclass(frozen=True)
class ModelOosFold:
    """References to selection and prediction evidence, never caller-supplied TEST rows."""

    model_selection_id: str
    candidate_prediction_id: str
    best_simple_prediction_id: str


@dataclass(frozen=True)
class ModelWalkForwardResult:
    model_family: str
    selections: tuple[ModelSelectionEvidence, ...]
    evaluation: FrozenOosEvaluation


def evaluate_model_walk_forward(
    folds: Mapping[str, ModelOosFold],
    *,
    dataset_store: DatasetArtifactStore,
    model_store: ModelArtifactBundleStore,
    selection_root: Path,
    prediction_root: Path,
    minimum_cross_section: int = 30,
) -> ModelWalkForwardResult:
    """Independently load only verified frozen artifact, TEST manifest, and selections."""
    _require_complete_fold_set(folds)
    selections_store = SelectionEvidenceVerifier.from_environment(selection_root)
    prediction_store = FrozenPredictionStore.from_environment(prediction_root)
    selections = tuple(
        selections_store.load_model_selection(folds[fold_id].model_selection_id)
        for fold_id in ORDINARY_TEST_FOLDS
    )
    model_family = _require_one_model_family(selections)
    _require_distinct(selections, "model_artifact_id", "artifact")
    _require_distinct(selections, "evidence_id", "selection")
    oos_folds: dict[str, OosFold] = {}
    for fold_id, selection in zip(ORDINARY_TEST_FOLDS, selections, strict=True):
        request = folds[fold_id]
        if selection.fold_id != fold_id:
            raise ValueError(f"{fold_id} model selection evidence is bound to {selection.fold_id}")
        _require_verified_model_bundle(model_store, selection)
        test_frame = dataset_store.load_split(
            selection.test_dataset_id,
            "test",
            expected_fold_id=fold_id,
        )
        candidate = prediction_store.load(
            request.candidate_prediction_id,
            selections=selections_store,
        )
        baseline = prediction_store.load(
            request.best_simple_prediction_id,
            selections=selections_store,
        )
        _require_candidate_evidence(candidate, selection, fold_id)
        baseline_selection = selections_store.load_baseline_selection(
            selection.baseline_selection_id
        )
        _require_best_simple_evidence(
            baseline,
            baseline_selection.evidence_id,
            baseline_selection.selected_baseline_id,
            fold_id,
            selection.test_dataset_id,
        )
        labels = _labels_with_lineage(test_frame, fold_id, selection.test_dataset_id)
        oos_folds[fold_id] = OosFold(
            candidate_predictions=_prediction_with_lineage(candidate.scores, candidate, fold_id),
            labels=labels,
            baseline_predictions=_prediction_with_lineage(baseline.scores, baseline, fold_id),
        )
    return ModelWalkForwardResult(
        model_family=model_family,
        selections=selections,
        evaluation=evaluate_walk_forward_oos(
            oos_folds,
            minimum_cross_section=minimum_cross_section,
            is_baseline=False,
        ),
    )


def _require_complete_fold_set(folds: Mapping[str, ModelOosFold]) -> None:
    expected = set(ORDINARY_TEST_FOLDS)
    actual = set(folds)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        unexpected = sorted(actual.difference(expected))
        details: list[str] = []
        if missing:
            details.append(f"missing folds: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected folds: {', '.join(unexpected)}")
        raise ValueError("model walk-forward folds are incomplete: " + "; ".join(details))


def _require_one_model_family(selections: tuple[ModelSelectionEvidence, ...]) -> str:
    families = {selection.model_family for selection in selections}
    if len(families) != 1 or next(iter(families)) not in MODEL_FAMILIES:
        raise ValueError("model walk-forward evaluation requires one supported model family")
    return selections[0].model_family


def _require_distinct(
    selections: tuple[ModelSelectionEvidence, ...],
    attribute: str,
    description: str,
) -> None:
    values = [str(getattr(selection, attribute)) for selection in selections]
    if len(set(values)) != len(values):
        raise ValueError(f"each model walk-forward fold requires a distinct frozen {description}")


def _require_verified_model_bundle(
    store: ModelArtifactBundleStore,
    selection: ModelSelectionEvidence,
) -> None:
    card = store.load_model_card(
        selection.model_artifact_id,
        expected_bundle_digest=selection.model_bundle_digest,
    )
    family = card.get("family")
    if family not in _MODEL_CARD_FAMILIES[selection.model_family]:
        raise ValueError("model bundle family does not match frozen model selection")
    if card.get("dataset_id") != selection.training_dataset_id:
        raise ValueError("model bundle training dataset does not match frozen model selection")


def _require_candidate_evidence(
    prediction: FrozenPredictionArtifact,
    selection: ModelSelectionEvidence,
    fold_id: str,
) -> None:
    if (
        prediction.kind != "candidate"
        or prediction.selection_id != selection.evidence_id
        or prediction.model_id != selection.model_artifact_id
        or prediction.fold_id != fold_id
        or prediction.dataset_id != selection.test_dataset_id
    ):
        raise ValueError("candidate prediction does not match frozen model selection evidence")


def _require_best_simple_evidence(
    prediction: FrozenPredictionArtifact,
    baseline_selection_id: str,
    selected_baseline_id: str,
    fold_id: str,
    test_dataset_id: str,
) -> None:
    if (
        prediction.kind != "best_simple"
        or prediction.selection_id != baseline_selection_id
        or prediction.model_id != selected_baseline_id
        or prediction.fold_id != fold_id
        or prediction.dataset_id != test_dataset_id
    ):
        raise ValueError("best-simple prediction does not match the selected baseline evidence")


def _labels_with_lineage(frame: pd.DataFrame, fold_id: str, dataset_id: str) -> pd.DataFrame:
    required = {"signal_date", "ts_code", "raw_return_5d", "label_valid"}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"verified TEST dataset is missing label columns: {missing}")
    labels = frame.loc[:, ["signal_date", "ts_code", "raw_return_5d", "label_valid"]].copy()
    labels["fold_id"] = fold_id
    labels["dataset_id"] = dataset_id
    return labels


def _prediction_with_lineage(
    scores: pd.DataFrame,
    evidence: FrozenPredictionArtifact,
    fold_id: str,
) -> pd.DataFrame:
    frame = scores.copy(deep=True)
    frame["model_id"] = evidence.model_id
    frame["fold_id"] = fold_id
    frame["dataset_id"] = evidence.dataset_id
    return frame


__all__ = [
    "MODEL_FAMILIES",
    "ModelOosFold",
    "ModelWalkForwardResult",
    "evaluate_model_walk_forward",
]
