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
re-verifies all four bundles), loads each fold's bundle through the EVALUATION path, scores the
fold's frozen TEST split the way the candidate driver scored it, and compares value for value
against the signed candidate prediction artifact. Any mismatch is a refusal.

This is an audit, not a serving authorisation, so it deliberately does not require the run to be
promoted. Requiring promotion would exclude the runs most worth auditing: a run that failed a gate
still wrote signed candidate predictions out of real bundles, and "did those scores come from those
bundles" is precisely the question a failed run leaves open. The registry status is therefore looked
up and reported, never enforced, and the promotion-gated serving path (``load_inference_artifact`` /
``load_lgbm_inference_artifact``) is untouched by this module.

Scope: every family the chain can produce. The LightGBM carriers (M0, M0R) and the static torch
families (M1 RESIDUAL_MLP, M3 SET_ATTENTION) score a frozen frame directly; the temporal families
(M2, whose card reads TEMPORAL_GRU or TEMPORAL_TCN after the winning config) need the 20-day window
panel, so they rebuild it from the same published splits through the one library function the driver
itself calls. An audit that rebuilt the panel its own way would only ever be agreeing with itself.

Exact equality requires the audit to score under the thread count the run was TRAINED under, and
that was learned the hard way. An untrained model's forward pass looked bitwise identical at
1, 4, 16 and 52 intra-op threads, which suggested thread count did not matter here; a real trained
bundle says otherwise -- auditing a 52-thread fold at 1 thread differs by 4.9e-05 across 803,572
rows, which is rank inversions between near-tied stocks rather than rounding. So
``recorded_run_threads`` reads the count the run's own checkpoint manifests record and the audit
process pins itself to it before scoring. An audit that ran at the host default would accuse an
honest run, and on a host whose core count differs from the producer's it would accuse every run.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import torch

from research_v2.datasets.storage import DatasetArtifactStore
from research_v2.evaluation.selection_evidence import (
    FrozenPredictionStore,
    ModelSelectionEvidence,
    SelectionEvidenceVerifier,
)
from research_v2.inference.artifact_loader import load_bundle_for_evaluation
from research_v2.inference.lgbm_artifact_loader import (
    LGBM_MODEL_CODE_VERSIONS,
    load_lgbm_bundle_for_evaluation,
    score_lgbm_artifact,
)
from research_v2.inference.torch_scoring import concat_date_scores, prepare_frame
from research_v2.models.families import STATIC_TORCH_FAMILIES, TEMPORAL_TORCH_FAMILIES
from research_v2.registry.promotion_evidence import (
    _load_full_report,
    _prediction_ids_by_fold_and_kind,
    walk_forward_model_id,
)
from research_v2.training.model_bundle import ModelArtifactBundleStore
from research_v2.training.seeds import describe_runtime_environment
from research_v2.training.static_batches import build_static_training_batches
from research_v2.training.temporal_search import TemporalFoldSpec, temporal_test_batches

if TYPE_CHECKING:
    from research_v2.registry.registry_store import ModelRegistryStore

_BUSINESS_KEYS = ("signal_date", "ts_code")
# Families whose score is a pure function of (bundle, frozen frame): one row per
# (signal_date, ts_code), no history window, so the frozen TEST split is a sufficient input.
# Both sets are derived from the model classes rather than written out here; see models/families.py
# for the three-way drift that motivated it. This module was the one consumer that already had both
# temporal spellings right, and it says below why both have to be named.
_STATIC_TORCH_FAMILIES = STATIC_TORCH_FAMILIES
# M2 scores a 20-day history window per row, which the frozen TEST split does not contain. The
# family is named by the WINNING config (the grid searches GRU and TCN alike, and the card reads
# TEMPORAL_<choice>), so both spellings have to be refused for the same stated reason -- otherwise
# a TCN run would fall through to "not allowlisted", which reads like an omission rather than a
# contract difference.
_TEMPORAL_TORCH_FAMILIES = TEMPORAL_TORCH_FAMILIES
# Reported when the derived identity is absent from the registry, which is the normal state of a
# run that failed a gate and was never promoted. Informational: it authorises serving nothing.
_UNREGISTERED = "UNREGISTERED"


class ReproductionError(ValueError):
    """A run's stored predictions are not what its bundles produce, and how they differ."""


@dataclass(frozen=True)
class RunIdentity:
    """The run's registry identity plus the lifecycle status the registry actually holds for it.

    ``status`` is reported and never enforced. ``UNREGISTERED`` means the id derived from the four
    verified bundle digests is not in the registry at all. Neither value authorises serving.
    """

    model_id: str
    status: str


@dataclass(frozen=True)
class _FoldBundle:
    """Everything one fold's recomputation needs, read from the bundle and the run report."""

    fold_id: str
    model_artifact_id: str
    family: str
    dataset_id: str
    report_dataset_id: str
    factor_order: tuple[str, ...]
    store: ModelArtifactBundleStore
    workspace: Path
    test: pd.DataFrame


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

    Raises ``ReproductionError`` on any score difference, on a row-set difference, and for families
    this module cannot reproduce. Raises ``PromotionEvidenceError`` when the report does not cover
    exactly the four ordinary folds or a bundle will not re-hash. It does NOT raise on a run that is
    unpromoted, unregistered or rejected: the registry status is reported in the summary instead.
    """
    identity = _run_identity(workspace, report_path, registry)
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
    # Four-fold coverage is already enforced: deriving the identity above read this same report's
    # folds and refuses unless they are exactly the four ordinary ones.

    prediction_ids = _prediction_ids_by_fold_and_kind(workspace)
    dataset_store = DatasetArtifactStore(workspace / "datasets")
    model_store = ModelArtifactBundleStore(workspace / "models")
    selections = SelectionEvidenceVerifier.from_environment(workspace / "selections")
    predictions = FrozenPredictionStore.from_environment(workspace / "predictions")

    results: list[FoldReproduction] = []
    for fold_id in sorted(fold_meta):
        meta = fold_meta[fold_id]
        model_artifact = meta["model_id"]
        card = _model_card(model_store, model_artifact)
        bundle = _FoldBundle(
            fold_id=fold_id,
            model_artifact_id=model_artifact,
            family=_family_of(card, fold_id),
            dataset_id=_checked_card_dataset_id(card, meta["dataset_id"], fold_id=fold_id),
            report_dataset_id=meta["dataset_id"],
            factor_order=_feature_order(model_store, model_artifact),
            store=model_store,
            workspace=workspace,
            test=dataset_store.load_split(
                meta["dataset_id"], "test", expected_fold_id=fold_id
            ),
        )
        recomputed = _recompute_fold_scores(bundle)
        try:
            candidate_id = prediction_ids[(fold_id, "candidate")]
        except KeyError as error:
            raise ReproductionError(
                f"{fold_id} has no signed candidate prediction artifact"
            ) from error
        stored = predictions.load(candidate_id, selections=selections).scores

        results.append(
            _compare_scores(
                recomputed,
                stored,
                fold_id=fold_id,
                model_artifact_id=model_artifact,
                family=bundle.family,
            )
        )
    if not results:
        raise ReproductionError("no folds were reproduced")
    return {
        "model_id": identity.model_id,
        "registry_status": identity.status,
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


def _run_identity(workspace: Path, report_path: Path, registry: ModelRegistryStore) -> RunIdentity:
    """Derive the run's registry identity and read its lifecycle status WITHOUT enforcing it.

    Deriving the id re-hashes and re-verifies all four bundles, so the audit keeps that integrity
    check whether or not the run was ever promoted. The registry lookup is the part that must not
    raise: a run that failed a gate is normally absent from the registry, and refusing to audit it
    would exempt exactly the predictions nobody has checked. A ``RegistryStoreError`` from opening a
    corrupt registry is deliberately not swallowed -- only "this id is not registered" is.
    """
    model_id = walk_forward_model_id(workspace, report_path)
    model_registry = registry.registry
    try:
        record = model_registry.get(model_id)
    except ValueError:
        return RunIdentity(model_id=model_id, status=_UNREGISTERED)
    return RunIdentity(model_id=model_id, status=record.status.value)


def verify_fold_predictions_reproduce(workspace: Path, fold_id: str) -> dict[str, object]:
    """Audit ONE fold, without the run report and without the other three bundles.

    The whole-run verifier needs ``report.json`` and all four bundles, so a run that is still
    training its last fold -- or that was interrupted in it -- cannot be audited at all. Its
    completed folds are exactly the ones worth checking early: a divergence found at hour four costs
    a restart, and the same divergence found at hour forty costs the verdict.

    Everything the report would have supplied is taken from signed evidence instead. The model
    selection for this fold names the bundle and the TEST dataset, and its ``model_bundle_digest``
    is compared against a re-hash of that bundle, so the fold being audited is the fold the chain
    says was frozen -- not merely whichever directory happens to be named ``m1-f1``.
    """
    selections = SelectionEvidenceVerifier.from_environment(workspace / "selections")
    evidence_id, evidence = _signed_model_evidence(workspace, selections, fold_id)
    model_store = ModelArtifactBundleStore(workspace / "models")
    model_artifact = evidence.model_artifact_id
    digest = model_store.bundle_digest(model_artifact)
    if evidence.model_bundle_digest != digest:
        raise ReproductionError(
            f"{fold_id} signed evidence binds bundle digest {evidence.model_bundle_digest}, but "
            f"bundle {model_artifact} re-hashes to {digest}"
        )
    card = _model_card(model_store, model_artifact)
    family = _family_of(card, fold_id)
    dataset_id = _checked_card_dataset_id(card, evidence.test_dataset_id, fold_id=fold_id)
    dataset_store = DatasetArtifactStore(workspace / "datasets")
    bundle = _FoldBundle(
        fold_id=fold_id,
        model_artifact_id=model_artifact,
        family=family,
        dataset_id=dataset_id,
        report_dataset_id=evidence.test_dataset_id,
        factor_order=_feature_order(model_store, model_artifact),
        store=model_store,
        workspace=workspace,
        test=dataset_store.load_split(
            evidence.test_dataset_id, "test", expected_fold_id=fold_id
        ),
    )
    recomputed = _recompute_fold_scores(bundle)
    prediction_ids = _prediction_ids_by_fold_and_kind(workspace)
    try:
        candidate_id = prediction_ids[(fold_id, "candidate")]
    except KeyError as error:
        raise ReproductionError(
            f"{fold_id} has no signed candidate prediction artifact"
        ) from error
    predictions = FrozenPredictionStore.from_environment(workspace / "predictions")
    stored = predictions.load(candidate_id, selections=selections).scores
    result = _compare_scores(
        recomputed,
        stored,
        fold_id=fold_id,
        model_artifact_id=model_artifact,
        family=family,
    )
    return {
        "fold_id": fold_id,
        "model_artifact_id": model_artifact,
        "model_selection_id": evidence_id,
        "candidate_prediction_id": candidate_id,
        "bundle_digest": digest,
        "family": family,
        "row_count": result.row_count,
        "max_absolute_difference": result.max_absolute_difference,
        "identical": result.identical,
        "reproduced": True,
    }


def _signed_model_evidence(
    workspace: Path, selections: SelectionEvidenceVerifier, fold_id: str
) -> tuple[str, ModelSelectionEvidence]:
    """The one signed model selection for this fold, or a refusal that says what was found.

    Two selections for one fold is not a case to pick between: it means the workspace holds two
    runs' worth of evidence for the same fold, and auditing either one would be auditing a guess.
    """
    directory = workspace / "selections" / "model"
    found: list[tuple[str, ModelSelectionEvidence]] = []
    if directory.exists():
        for path in sorted(directory.glob("*.json")):
            try:
                evidence = selections.load_model_selection(path.stem)
            except ValueError:
                continue
            if evidence.fold_id == fold_id:
                found.append((path.stem, evidence))
    if not found:
        raise ReproductionError(f"{fold_id} has no signed model selection evidence")
    if len(found) > 1:
        identities = ", ".join(identity for identity, _ in found)
        raise ReproductionError(
            f"{fold_id} has {len(found)} signed model selections ({identities}); this auditor "
            "will not choose between them"
        )
    return found[0]


def _model_card(store: ModelArtifactBundleStore, model_artifact_id: str) -> Mapping[str, object]:
    path = store.bundle_path(model_artifact_id) / "model_card.json"
    try:
        card = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ReproductionError(
            f"{model_artifact_id} has no readable model card: {error}"
        ) from error
    if not isinstance(card, dict):
        raise ReproductionError(f"{model_artifact_id} model card is not an object")
    return card


def recorded_run_threads(workspace: Path) -> int | None:
    """The intra-op thread count this run's fits recorded, or None if it recorded none.

    Read from the run's own checkpoint manifests, which the training runner writes for every
    checkpoint. Two refusals rather than a guess: a workspace whose checkpoints record more than one
    count mixes lineages and cannot be audited as one run, and a manifest that records the field as
    something other than a positive integer is skipped rather than trusted.

    Callers that audit through this module directly must pin the process to the returned value
    before scoring -- ``scripts/verify_prediction_reproduction.py`` does, at startup, so that both
    thread pools are set before any parallel work has begun.
    """
    root = workspace / "checkpoints"
    counts: set[int] = set()
    if not root.exists():
        return None
    for manifest in sorted(root.glob("*/manifest.json")):
        try:
            document = json.loads(manifest.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        metadata = document.get("metadata") if isinstance(document, dict) else None
        threads = metadata.get("torch_intraop_threads") if isinstance(metadata, dict) else None
        if isinstance(threads, int) and not isinstance(threads, bool) and threads > 0:
            counts.add(threads)
    if not counts:
        return None
    if len(counts) > 1:
        raise ReproductionError(
            f"this workspace's checkpoints record {len(counts)} different torch thread counts "
            f"({sorted(counts)}), so it mixes execution lineages and cannot be audited as one run"
        )
    return next(iter(counts))


def audit_runtime_threads() -> int:
    """The intra-op thread count this process is currently scoring under, for the audit record."""
    return int(describe_runtime_environment().torch_intraop_threads)


def _family_of(card: Mapping[str, object], fold_id: str) -> str:
    family = card.get("family")
    if not isinstance(family, str) or not family:
        raise ReproductionError(f"{fold_id} bundle model card records no usable family")
    return family


def _checked_card_dataset_id(
    card: Mapping[str, object], report_dataset_id: str, *, fold_id: str
) -> str:
    """Return the bundle's own dataset id after proving it is derived from the report's.

    The two writers disagree by design: the LightGBM path records the TRAIN split it fitted on
    (``<dataset>:train``) while the torch path records the bare TEST dataset id. Both are
    legitimate, so the check is membership rather than equality -- but membership of a two-element
    set built from this fold's report entry, which still rejects a bundle dropped in from another
    run or another fold. The accepted value is then what the loader is asked to match, so the
    loader's own dataset check compares the card against something the report vouches for.
    """
    recorded = card.get("dataset_id")
    train_variant = f"{report_dataset_id}:train"
    if not isinstance(recorded, str) or recorded not in (report_dataset_id, train_variant):
        raise ReproductionError(
            f"{fold_id} bundle records dataset_id {recorded!r}, which is neither the report's "
            f"{report_dataset_id!r} nor its train split {train_variant!r}"
        )
    return recorded


def _recorded_preprocessing(bundle: _FoldBundle) -> Mapping[str, object]:
    """The bundle's own preprocessing document: the state hash with the fitted state beside it."""
    path = bundle.store.bundle_path(bundle.model_artifact_id) / "preprocessing.json"
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ReproductionError(
            f"{bundle.model_artifact_id} has no readable preprocessing artifact: {error}"
        ) from error
    if not isinstance(document, dict):
        raise ReproductionError(
            f"{bundle.model_artifact_id} preprocessing artifact is not an object"
        )
    return document


def _recorded_preprocessor_hash(bundle: _FoldBundle) -> str:
    """Read the preprocessor state hash the bundle itself records.

    Passing a bundle's own recorded value as the expectation looks circular and is not: the loader
    restores the fitted state from the persisted payload, re-derives the hash from what it rebuilt,
    and compares the two, so a payload edited after the bundle was written is refused there. What
    this cannot catch is a hash and payload replaced together, and nothing short of the run's own
    training record can -- the guarantee here is that the stored scores are what THIS bundle
    computes, which is the guarantee the attribution chain does not provide.
    """
    state_hash = _recorded_preprocessing(bundle).get("state_hash")
    if not isinstance(state_hash, str) or not state_hash:
        raise ReproductionError(f"{bundle.model_artifact_id} records no preprocessor state hash")
    return state_hash


def _recompute_fold_scores(bundle: _FoldBundle) -> pd.DataFrame:
    """Route one fold to the scorer that owns its family, refusing the rest by name."""
    family = bundle.family
    if family in LGBM_MODEL_CODE_VERSIONS:
        return _recompute_lgbm(bundle)
    if family in _STATIC_TORCH_FAMILIES:
        return _recompute_static_torch(bundle)
    if family in _TEMPORAL_TORCH_FAMILIES:
        return _recompute_temporal_torch(bundle)
    raise ReproductionError(
        f"{bundle.fold_id} bundle family {family!r} is not allowlisted for reproduction"
    )


def _recompute_lgbm(bundle: _FoldBundle) -> pd.DataFrame:
    loaded = load_lgbm_bundle_for_evaluation(
        bundle.store,
        bundle.model_artifact_id,
        expected_dataset_id=bundle.dataset_id,
        expected_feature_order=bundle.factor_order,
    )
    return score_lgbm_artifact(loaded, bundle.test)


def _recompute_static_torch(bundle: _FoldBundle) -> pd.DataFrame:
    """Recompute a static torch fold's scores by repeating the driver's own scoring path.

    Every step mirrors ``scripts/run_real_candidate_training.py``: load the bundle for evaluation
    against the preprocessor hash the bundle records, put the frozen TEST frame through the
    train-fitted preprocessing with the label column zeroed (the driver does the same, because the
    batch builder requires a target column and scoring must not be able to read it), build one
    date-isolated batch per signal date, then average the three seeds' within-date percentile ranks.
    Substituting any step with an equivalent-looking one yields a different scorer, and the
    comparison below is exact, so that would be caught rather than tolerated.
    """
    loaded = load_bundle_for_evaluation(
        bundle.store,
        bundle.model_artifact_id,
        expected_dataset_id=bundle.dataset_id,
        expected_feature_order=bundle.factor_order,
        expected_preprocessor_hash=_recorded_preprocessor_hash(bundle),
        device=torch.device("cpu"),
    )
    preprocessor = loaded.preprocessor
    if preprocessor is None:
        raise ReproductionError(
            f"{bundle.fold_id} torch bundle restored no preprocessing state, so the frozen TEST "
            "frame cannot be put into the space its weights were trained in"
        )
    prepared = prepare_frame(
        bundle.test.assign(raw_return_5d=0.0), preprocessor, bundle.factor_order
    )
    batches = build_static_training_batches(prepared, factor_order=bundle.factor_order)
    return concat_date_scores(loaded.models, batches)


def _recompute_temporal_torch(bundle: _FoldBundle) -> pd.DataFrame:
    """Recompute a temporal fold's frozen TEST scores by repeating the driver's own path.

    The difference from the static families is the input, not the scorer: a TEST row is scored from
    a 20-day window, so the panel is train+validation+test concatenated and then filtered to the
    TEST dates. That composition lives in one library function the driver also calls, which is the
    only reason this recomputation is evidence about the run rather than about this module.

    The label column is NOT zeroed here, because the driver does not zero it for M2: a temporal TEST
    row without a usable label never entered the panel, so the stored scores cover the labelled
    subset. That row set is a property of the run being audited, and normalising it away here would
    make the audit compare against something the chain never produced.
    """
    loaded = load_bundle_for_evaluation(
        bundle.store,
        bundle.model_artifact_id,
        expected_dataset_id=bundle.dataset_id,
        expected_feature_order=bundle.factor_order,
        expected_preprocessor_hash=_recorded_preprocessor_hash(bundle),
        device=torch.device("cpu"),
    )
    spec = TemporalFoldSpec(
        workspace=bundle.workspace,
        fold_id=bundle.fold_id,
        dataset_id=bundle.report_dataset_id,
        factor_order=bundle.factor_order,
        preprocessing=_recorded_preprocessing(bundle),
        torch_threads=torch.get_num_threads(),
    )
    return concat_date_scores(loaded.models, temporal_test_batches(spec))


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
    "RunIdentity",
    "audit_runtime_threads",
    "recorded_run_threads",
    "verify_candidate_predictions_reproduce",
    "verify_fold_predictions_reproduce",
]
