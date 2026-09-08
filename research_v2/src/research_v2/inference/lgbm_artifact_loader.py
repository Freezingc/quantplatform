"""Allowlisted, hash-verified loaders for immutable LightGBM inference artifacts.

This is a sibling of ``artifact_loader`` rather than a branch inside it, and the reason is the
artifact shape, not convenience. ``load_inference_artifact`` is torch-shaped end to end: it
returns ``nn.Module`` instances, reads each ``seed-<seed>.pt`` with ``torch.load(...,
weights_only=True)``, and verifies a bundle by reconstructing a prototype module and comparing
its ``state_dict``. A LightGBM bundle stores those same filenames as **booster text**, and its
signature is a ``ModelSignature`` over the config hash rather than a module's
``artifact_signature()``. Generalising the torch loader to also return non-modules would change
a contract every M1/M2/M3 consumer depends on, so the LightGBM families get their own loader
with the same discipline: verified bundle, allowlisted family, code-version match, metadata
cross-checks, all three seeds, and an audit record.

It also fixes a placement problem. The scoring semantics -- per-day percentile rank per booster,
then the mean across the three seeds, merged on business keys and never positionally -- lived
only inside the training driver, a script. Inference could not reach them without importing a
script, so ``score_lgbm_artifact`` is the library home for them and a test pins that it agrees
with the driver's own functions on real bundles.

Both LightGBM families are allowlisted, not just the new one: M0 (``LGBM_RANKER``) had the same
gap, and a loader that served M0R while leaving M0 unservable would make the two carriers
impossible to compare in production.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import Literal

import lightgbm as lgb
import numpy as np
import pandas as pd

from research_v2.inference.artifact_loader import (
    InferenceContract,
    InferenceLoadAudit,
    _model_card_code_version,
    _validate_inference_contract,
    unverified_bindings,
)
from research_v2.models.base import ModelSignature
from research_v2.registry.promotion_evidence import PromotedRun
from research_v2.training.ensemble import REQUIRED_SEEDS
from research_v2.training.model_bundle import ModelArtifactBundleStore

LGBMModelFamily = Literal["LGBM_RANKER", "LGBM_RANKREG"]

LGBM_MODEL_CODE_VERSIONS: dict[str, str] = {
    "LGBM_RANKER": "lgbm-ranker-1",
    "LGBM_RANKREG": "lgbm-rankreg-1",
}
# The booster text records its own objective, so the family can be checked against the artifact
# rather than trusted from the model card alone.
LGBM_FAMILY_OBJECTIVES: dict[str, str] = {
    "LGBM_RANKER": "lambdarank",
    "LGBM_RANKREG": "regression",
}
_BUSINESS_KEYS = ("signal_date", "ts_code")
_PREPROCESSING_IDENTITY = "identity"


@dataclass(frozen=True)
class LoadedLGBMInferenceArtifact:
    """Three verified seed boosters plus everything needed to score a frame reproducibly.

    ``promotion_model_id`` and ``promotion_status`` record which registry entry authorised this
    load, so a loaded artifact carries its own authority rather than leaving the caller to
    remember whether it checked.
    """

    model_id: str
    family: str
    factor_order: tuple[str, ...]
    boosters: Mapping[int, lgb.Booster]
    preprocessing: Mapping[str, object]
    promotion_model_id: str
    promotion_status: str
    audit: InferenceLoadAudit | None = None
    # Bindings this load could not verify; see artifact_loader.unverified_bindings.
    unverified_bindings: tuple[str, ...] = ()


def _config_hash(config_payload: Mapping[str, object]) -> str:
    return hashlib.sha256(
        json.dumps(dict(config_payload), separators=(",", ":"), sort_keys=True).encode()
    ).hexdigest()


def load_lgbm_inference_artifact(
    store: ModelArtifactBundleStore,
    model_id: str,
    *,
    expected_dataset_id: str,
    expected_feature_order: tuple[str, ...],
    expected_preprocessor_hash: str = _PREPROCESSING_IDENTITY,
    expected_inference_contract: InferenceContract | None = None,
    run_id: str | None = None,
    promoted_run: PromotedRun | None = None,
) -> LoadedLGBMInferenceArtifact:
    """Load an M0 or M0R bundle only after full integrity, allowlist and contract checks.

    ``promoted_run`` is what authorises the load, and omitting it is refused. The bundle's own
    ``model_status`` card field cannot play that role: the driver writes
    ``WALK_FORWARD_PASSED`` into every bundle before the OOS evaluation exists, so it is true by
    construction even for runs that failed G4 -- verified on the real m0_v3 workspace, whose
    report says ``all_passed: false`` at delta +0.004202 while all four of its cards read
    WALK_FORWARD_PASSED. Authority therefore comes from ``verify_run_promotion``, which asks the
    registry, and the requested bundle must be one of that promoted run's four fold artifacts.
    """
    if promoted_run is None:
        raise ValueError(
            "loading a LightGBM bundle for inference requires a PromotedRun verified against the "
            "model registry; the bundle's own model_status card field is not promotion evidence"
        )
    if model_id not in set(promoted_run.fold_artifact_ids.values()):
        raise ValueError(
            f"bundle {model_id} does not belong to promoted run {promoted_run.model_id}, whose "
            f"fold artifacts are {sorted(promoted_run.fold_artifact_ids.values())}"
        )
    bundle = store.load_verified_bundle(model_id)
    metadata = store.load_metadata_from_bundle(
        bundle,
        expected_dataset_id=expected_dataset_id,
        expected_feature_order=expected_feature_order,
        expected_signature=None,
        expected_preprocessor_hash=expected_preprocessor_hash,
    )
    model_card = metadata["model_card"]
    _validate_inference_contract(model_card, expected_inference_contract)
    # A bundle-shape check, NOT promotion proof -- see the docstring. Kept because a bundle that
    # never went through the walk-forward writer is not a candidate artifact at all.
    if model_card.get("model_status") != "WALK_FORWARD_PASSED":
        raise ValueError("model bundle was not produced by the walk-forward candidate writer")
    factor_set_status = model_card.get("factor_set_status")
    if factor_set_status is not None and factor_set_status != "FROZEN":
        raise ValueError("inference requires a FROZEN factor set")

    family = model_card.get("family")
    if not isinstance(family, str) or family not in LGBM_MODEL_CODE_VERSIONS:
        raise ValueError("model family is not allowlisted for LightGBM inference")
    code_version = _model_card_code_version(model_card)
    if code_version != LGBM_MODEL_CODE_VERSIONS[family]:
        raise ValueError("model_code_version is not allowlisted for model family")

    config = dict(metadata["config"])
    raw_factor_count = config.pop("raw_factor_count", None)
    feature_document = metadata["feature_order"]
    declared_order = feature_document.get("feature_order")
    if not isinstance(declared_order, list):
        raise ValueError("model bundle feature_order document is malformed")
    factor_order = tuple(str(name) for name in declared_order)
    if factor_order != tuple(expected_feature_order):
        raise ValueError("model bundle feature_order does not match the requested feature order")
    if raw_factor_count != len(factor_order):
        raise ValueError("model bundle raw_factor_count disagrees with its feature order")
    # The signature covers the search config WITHOUT raw_factor_count, which the bundle stores
    # alongside it; hashing the stored config as-is would never match.
    expected_signature = dict(
        ModelSignature(
            model_family=family,
            raw_factor_count=len(factor_order),
            config_hash=_config_hash(config),
        ).__dict__
    )
    if dict(metadata["signature"]) != expected_signature:
        raise ValueError("model bundle signature does not match its reconstructed config")

    boosters: dict[int, lgb.Booster] = {}
    for seed in REQUIRED_SEEDS:
        name = f"seed-{seed}.pt"
        if name not in bundle.files:
            raise ValueError(f"model bundle is missing the booster for seed {seed}")
        booster = lgb.Booster(model_str=bundle.files[name].decode("utf-8"))
        if booster.num_feature() != len(factor_order):
            raise ValueError(
                f"booster for seed {seed} expects {booster.num_feature()} features, "
                f"the bundle declares {len(factor_order)}"
            )
        recorded = _booster_objective(booster)
        if recorded != LGBM_FAMILY_OBJECTIVES[family]:
            raise ValueError(
                f"booster objective {recorded!r} does not match family {family}; the artifact "
                "and its model card describe different models"
            )
        if booster.num_trees() < 1:
            raise ValueError(f"booster for seed {seed} has no trees")
        boosters[seed] = booster

    return LoadedLGBMInferenceArtifact(
        model_id=model_id,
        family=family,
        factor_order=factor_order,
        boosters=MappingProxyType(boosters),
        preprocessing=metadata["preprocessing"],
        promotion_model_id=promoted_run.model_id,
        promotion_status=promoted_run.status.value,
        unverified_bindings=unverified_bindings(model_card, expected_inference_contract),
        audit=InferenceLoadAudit(
            model_id=model_id,
            bundle_digest=bundle.digest,
            artifact_hashes=MappingProxyType(dict(bundle.artifact_hashes)),
            # LightGBM boosters are CPU artifacts; the field is kept so one audit shape covers
            # both loaders and a consumer cannot mistake one for the other.
            device="cpu:lightgbm",
            code_version=code_version,
            loaded_at=datetime.now(UTC),
            run_id=run_id,
        ),
    )


def _booster_objective(booster: lgb.Booster) -> str:
    """Read the objective the booster was actually trained with, from its own dump."""
    dump = booster.dump_model()
    objective = dump.get("objective") if isinstance(dump, dict) else None
    if not isinstance(objective, str):
        raise ValueError("booster dump does not record its objective")
    # LightGBM writes e.g. "regression" or "lambdarank" possibly with parameters appended
    return objective.split(" ")[0].strip()


def score_lgbm_artifact(
    loaded: LoadedLGBMInferenceArtifact, frame: pd.DataFrame
) -> pd.DataFrame:
    """Score a frame exactly as the walk-forward chain scored its frozen TEST frames.

    Each booster's raw scores become per-day percentile ranks, and the three seed ranks are
    averaged after merging on ``(signal_date, ts_code)`` -- never positionally, so a frame whose
    row order differs from training still scores correctly. Percentile-ranking before averaging
    is what makes the seeds commensurable: raw booster outputs are on arbitrary scales, and
    averaging them would let whichever seed happened to produce the largest magnitudes dominate.
    """
    keys = list(_BUSINESS_KEYS)
    required = set(keys) | set(loaded.factor_order)
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"frame is missing columns required for scoring: {missing}")
    if frame.duplicated(keys).any():
        raise ValueError("frame contains duplicate business keys")
    merged: pd.DataFrame | None = None
    rank_columns: list[str] = []
    for seed in REQUIRED_SEEDS:
        booster = loaded.boosters[seed]
        values = frame.loc[:, list(loaded.factor_order)].apply(pd.to_numeric, errors="coerce")
        matrix = values.to_numpy(dtype=float)
        # NaN is expected and handled natively; infinities are not a missing value and would
        # silently dominate a percentile rank, so they are refused.
        if np.isinf(matrix).any():
            raise ValueError("frame feature values must not be infinite")
        raw = booster.predict(matrix)
        if not np.isfinite(raw).all():
            raise ValueError("booster emitted non-finite scores")
        percentile = frame.loc[:, keys].copy()
        column = f"seed_rank_{seed}"
        percentile[column] = (
            pd.Series(raw).groupby(frame["signal_date"].to_numpy()).rank(pct=True).to_numpy()
        )
        merged = (
            percentile
            if merged is None
            else merged.merge(percentile, on=keys, validate="one_to_one")
        )
        rank_columns.append(column)
    if merged is None:
        raise ValueError("at least one seed booster is required")
    merged["score"] = merged.loc[:, rank_columns].mean(axis=1)
    return merged.loc[:, [*keys, "score"]]


__all__ = [
    "LGBM_FAMILY_OBJECTIVES",
    "LGBM_MODEL_CODE_VERSIONS",
    "LoadedLGBMInferenceArtifact",
    "load_lgbm_inference_artifact",
    "score_lgbm_artifact",
]
