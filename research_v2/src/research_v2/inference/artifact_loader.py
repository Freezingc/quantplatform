"""Allowlisted, hash-verified CPU-first loaders for immutable inference artifacts."""

from __future__ import annotations

import io
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TypeAlias, cast

import torch
from torch import nn

from research_v2.datasets.preprocessing import FoldPreprocessor
from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
from research_v2.models.set_attention import SetAttention, SetAttentionConfig
from research_v2.models.temporal import TemporalGRU, TemporalGRUConfig
from research_v2.registry.promotion_evidence import PromotedRun
from research_v2.training.ensemble import REQUIRED_SEEDS
from research_v2.training.model_bundle import ModelArtifactBundleStore

InferenceContract: TypeAlias = Mapping[str, object]
_INFERENCE_CONTRACT_FIELDS = frozenset(
    {
        "factor_set_id",
        "factor_set_status",
        "universe_id",
        "universe_version",
        "label_id",
        "label_version",
        "split_id",
        "code_version",
        "model_code_version",
    }
)
_MODEL_CODE_VERSIONS = {
    "RESIDUAL_MLP": "residual-mlp-1",
    "TEMPORAL_GRU": "temporal-gru-1",
    "SET_ATTENTION": "set-attention-1",
}


@dataclass(frozen=True)
class InferenceLoadAudit:
    """Non-sensitive evidence describing one verified artifact load."""

    model_id: str
    bundle_digest: str
    artifact_hashes: Mapping[str, str]
    device: str
    code_version: str | None
    loaded_at: datetime
    run_id: str | None


def unverified_bindings(
    model_card: Mapping[str, object],
    expected_inference_contract: InferenceContract | None,
) -> tuple[str, ...]:
    """Name the load-time bindings that could NOT be checked, instead of skipping them silently.

    Two guards in this module are conditional, and for every bundle the chain has ever written
    both conditions are false, so neither has ever fired:

      ``factor_set_status``   no model card contains the field. The driver writes dataset_id,
                              family, model_status and model_code_version and nothing else, so
                              "inference requires a FROZEN factor set" has never been enforced
                              against a real artifact -- verified across the m0_v3, m0r_v2 and
                              m0r_v3 workspaces.
      ``inference_contract``  no production caller passes one; only tests do. So the binding to a
                              factor set, universe, label version, split and code version is never
                              checked on the real path either. Two of its eight fields
                              (``split_id``, ``code_version``) have no producer anywhere in the
                              repository, so populating them is an ownership decision rather than
                              a wiring task.

    Neither can simply be made strict today: requiring the field would refuse every existing
    bundle, including the promoted M0R, and inventing values for two undefined fields would bake
    guesses into immutable content-addressed artifacts. What this does instead is make the gap
    impossible to miss -- a loaded artifact states which of its bindings are unverified, so no
    caller can mistake an unchecked binding for a checked one.
    """
    unverified: list[str] = []
    if model_card.get("factor_set_status") is None:
        unverified.append("factor_set_status")
    # Empty counts as unverified: a contract naming no fields establishes nothing. The loader
    # refuses one outright, so this is defence in depth for any path that bypasses it.
    if not expected_inference_contract:
        unverified.append("inference_contract")
    return tuple(unverified)


@dataclass(frozen=True)
class LoadedInferenceArtifact:
    model_id: str
    models: tuple[nn.Module, ...]
    preprocessing: Mapping[str, object]
    audit: InferenceLoadAudit | None = None
    # Which registry entry authorised this load, and its status. Empty for an evaluation load,
    # which is authorised by nothing beyond the bundle's own integrity -- it is not served.
    promotion_model_id: str = ""
    promotion_status: str = ""
    # The fitted preprocessing restored from the bundle, so a caller can put a new frame into the
    # space these weights were trained in. Weights plus a state hash are not a servable model.
    preprocessor: FoldPreprocessor | None = None
    # Bindings this load could not verify; see unverified_bindings. Empty means every guard fired.
    unverified_bindings: tuple[str, ...] = ()


def load_inference_artifact(
    store: ModelArtifactBundleStore,
    model_id: str,
    *,
    expected_dataset_id: str,
    expected_feature_order: tuple[str, ...],
    expected_preprocessor_hash: str,
    device: torch.device,
    expected_inference_contract: InferenceContract | None = None,
    run_id: str | None = None,
    promoted_run: PromotedRun | None = None,
) -> LoadedInferenceArtifact:
    """Load a bundle FOR SERVING, which requires registry-verified promotion.

    ``promoted_run`` is mandatory and the bundle must be one of that run's four fold artifacts.
    The model card's own ``model_status`` cannot substitute: the candidate driver writes
    ``WALK_FORWARD_PASSED`` into every bundle before the OOS evaluation exists, so it is true by
    construction even for runs that failed G4 -- the m0_v3 workspace, whose report says
    ``all_passed: false``, carries that value on all four folds. Authority comes from the registry,
    which is the only place the status was derived from recomputed gates rather than asserted by
    the artifact's producer. See
    docs/handoffs/2026-09-09-model-status-fail-open-finding.md.

    To load a bundle in order to EVALUATE it -- which is what the candidate driver does when it
    re-reads its own frozen bundle to produce the TEST predictions the gates then score -- use
    ``load_bundle_for_evaluation``. That is a different operation with different authority, and
    conflating the two is what made the check vacuous.
    """
    if promoted_run is None:
        raise ValueError(
            "serving a model requires a PromotedRun verified against the model registry; use "
            "load_bundle_for_evaluation to load a bundle for evaluation instead"
        )
    if model_id not in set(promoted_run.fold_artifact_ids.values()):
        raise ValueError(
            f"bundle {model_id} does not belong to promoted run {promoted_run.model_id}, whose "
            f"fold artifacts are {sorted(promoted_run.fold_artifact_ids.values())}"
        )
    loaded = load_bundle_for_evaluation(
        store,
        model_id,
        expected_dataset_id=expected_dataset_id,
        expected_feature_order=expected_feature_order,
        expected_preprocessor_hash=expected_preprocessor_hash,
        device=device,
        expected_inference_contract=expected_inference_contract,
        run_id=run_id,
    )
    return replace(
        loaded,
        promotion_model_id=promoted_run.model_id,
        promotion_status=promoted_run.status.value,
    )


def load_bundle_for_evaluation(
    store: ModelArtifactBundleStore,
    model_id: str,
    *,
    expected_dataset_id: str,
    expected_feature_order: tuple[str, ...],
    expected_preprocessor_hash: str,
    device: torch.device,
    expected_inference_contract: InferenceContract | None = None,
    run_id: str | None = None,
) -> LoadedInferenceArtifact:
    """Load known model classes only after full bundle and reconstruction-contract checks.

    This is the load-to-evaluate path: the candidate driver uses it to re-read a bundle it just
    froze, so that the TEST predictions the gates score provably come from the persisted artifact
    rather than from training memory. At that moment no gate verdict exists yet, so nothing here
    claims or checks promotion. It is NOT a serving path -- see ``load_inference_artifact``.
    """
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
    # A bundle-shape marker, NOT promotion proof: the driver writes this literal before the OOS
    # evaluation exists, so it is true for every bundle of every run, including runs that failed
    # G4. It is kept because a bundle the walk-forward writer never produced is not a candidate
    # artifact at all. Promotion is checked in load_inference_artifact, against the registry.
    if model_card.get("model_status") != "WALK_FORWARD_PASSED":
        raise ValueError("model bundle was not produced by the walk-forward candidate writer")
    factor_set_status = model_card.get("factor_set_status")
    if factor_set_status is not None and factor_set_status != "FROZEN":
        raise ValueError("inference requires a FROZEN factor set")
    factory = _allowlisted_factory(model_card, metadata["config"])
    prototype = cast(ResidualMLP | TemporalGRU | SetAttention, factory())
    expected_signature = asdict(prototype.artifact_signature())
    if dict(metadata["signature"]) != expected_signature:
        raise ValueError("model bundle signature does not match reconstructed model contract")
    preprocessor = _restored_preprocessor(metadata["preprocessing"])
    # The state's own factor order must be the bundle's feature order. Both are checked against
    # the caller's expectation separately, but comparing them to each other is what catches a
    # bundle assembled from one run's weights and another run's preprocessing -- each half would
    # pass its own check while transform() silently scored the wrong columns.
    if preprocessor.fitted_factor_order != tuple(expected_feature_order):
        raise ValueError(
            "model bundle preprocessor state was fitted on a different factor order than the "
            "bundle's feature order"
        )
    reference_state = cast(Mapping[str, torch.Tensor], prototype.state_dict())
    seed_states: dict[int, Mapping[str, torch.Tensor]] = {}
    for seed in REQUIRED_SEEDS:
        payload = torch.load(
            io.BytesIO(bundle.files[f"seed-{seed}.pt"]),
            map_location="cpu",
            weights_only=True,
        )
        state = _state_dict(payload)
        _validate_state_dict(state, reference_state)
        seed_states[seed] = state
    models: list[nn.Module] = []
    for seed in REQUIRED_SEEDS:
        model = factory()
        model.load_state_dict(seed_states[seed], strict=True)
        model.to(device)
        model.eval()
        models.append(model)
    return LoadedInferenceArtifact(
        model_id=model_id,
        models=tuple(models),
        preprocessing=metadata["preprocessing"],
        preprocessor=preprocessor,
        unverified_bindings=unverified_bindings(model_card, expected_inference_contract),
        audit=InferenceLoadAudit(
            model_id=model_id,
            bundle_digest=bundle.digest,
            artifact_hashes=MappingProxyType(dict(bundle.artifact_hashes)),
            device=str(device),
            code_version=_model_card_code_version(model_card),
            loaded_at=datetime.now(UTC),
            run_id=run_id,
        ),
    )


def _restored_preprocessor(preprocessing: Mapping[str, object]) -> FoldPreprocessor:
    """Restore the fitted preprocessing state a torch bundle must carry to be servable.

    Refusing a bundle that records only the hash is the point: such a bundle loads its weights
    fine and then cannot transform anything, which is a model that looks deployable and is not.
    The restore recomputes the hash from what it rebuilt and compares it to the recorded one, so
    a payload edited after the bundle was written is rejected here rather than silently shifting
    every score downstream.
    """
    state_hash = preprocessing.get("state_hash")
    if not isinstance(state_hash, str) or not state_hash:
        raise ValueError("model bundle records no preprocessor state hash")
    state_payload = preprocessing.get("state_payload")
    if not isinstance(state_payload, Mapping):
        raise ValueError(
            "model bundle records a preprocessor state hash but not the state itself, so it "
            "cannot transform a frame and cannot be served"
        )
    return FoldPreprocessor.restore(state_payload, expected_state_hash=state_hash)


def _validate_inference_contract(
    model_card: Mapping[str, object],
    expected: InferenceContract | None,
) -> None:
    if expected is None:
        return
    if not expected:
        # An empty contract checks nothing, and accepting it would let a caller turn the
        # "inference_contract" entry off in unverified_bindings without verifying anything --
        # the loader would then actively misreport an unchecked binding as checked. Verified
        # before this guard existed: passing {} loaded a card stripped of all eight contract
        # fields and reported unverified_bindings == ("factor_set_status",) alone.
        raise ValueError("inference contract must name at least one field to verify")
    unsupported = [
        key for key in expected if not isinstance(key, str) or key not in _INFERENCE_CONTRACT_FIELDS
    ]
    if unsupported:
        raise ValueError(f"inference contract contains unsupported fields: {unsupported}")
    for field, expected_value in expected.items():
        if field not in model_card:
            raise ValueError(f"model card inference contract field is missing: {field}")
        if model_card[field] != expected_value:
            raise ValueError(f"model card inference contract mismatch: {field}")


def _model_card_code_version(model_card: Mapping[str, object]) -> str | None:
    for field in ("code_version", "model_code_version"):
        value = model_card.get(field)
        if isinstance(value, str):
            return value
    return None


def _allowlisted_factory(
    model_card: Mapping[str, object],
    config: Mapping[str, object],
) -> Callable[[], nn.Module]:
    family = model_card.get("family")
    # Both checks are mandatory, not conditional on the field being present. The code-version
    # check used to read `if model_code_version is not None and ...`, so a card that simply
    # omitted the field skipped it entirely and loaded -- verified: such a bundle loads, and its
    # audit then reports the inference contract's research `code_version` as if it were the model
    # code version, because _model_card_code_version falls back across the two fields. An
    # artifact of unstated provenance must be refused, not loaded with a borrowed identity.
    if not isinstance(family, str) or family not in _MODEL_CODE_VERSIONS:
        raise ValueError("model family is not allowlisted for inference")
    expected_code_version = _MODEL_CODE_VERSIONS[family]
    model_code_version = model_card.get("model_code_version")
    if not isinstance(model_code_version, str) or model_code_version != expected_code_version:
        raise ValueError("model_code_version is not allowlisted for model family")
    if family == "RESIDUAL_MLP":
        residual = _residual_config(config)
        return lambda: ResidualMLP(residual)
    if family == "TEMPORAL_GRU":
        temporal_gru = _temporal_gru_config(config)
        return lambda: TemporalGRU(temporal_gru)
    if family == "SET_ATTENTION":
        set_attention = _set_attention_config(config)
        return lambda: SetAttention(set_attention)
    raise ValueError("model family is not allowlisted for inference")


def _residual_config(config: Mapping[str, object]) -> ResidualMLPConfig:
    _require_fields(config, {"raw_factor_count", "hidden_dim", "num_blocks", "dropout"})
    return ResidualMLPConfig(
        raw_factor_count=_int_config(config["raw_factor_count"], "raw_factor_count"),
        hidden_dim=_int_config(config["hidden_dim"], "hidden_dim"),
        num_blocks=_int_config(config["num_blocks"], "num_blocks"),
        dropout=_float_config(config["dropout"], "dropout"),
    )


def _temporal_gru_config(config: Mapping[str, object]) -> TemporalGRUConfig:
    _require_fields(config, {"raw_factor_count", "hidden_dim", "dropout"})
    return TemporalGRUConfig(
        raw_factor_count=_int_config(config["raw_factor_count"], "raw_factor_count"),
        hidden_dim=_int_config(config["hidden_dim"], "hidden_dim"),
        dropout=_float_config(config["dropout"], "dropout"),
    )


def _set_attention_config(config: Mapping[str, object]) -> SetAttentionConfig:
    _require_fields(
        config,
        {
            "raw_factor_count",
            "hidden_dim",
            "num_heads",
            "inducing_points",
            "num_blocks",
            "dropout",
        },
    )
    return SetAttentionConfig(
        raw_factor_count=_int_config(config["raw_factor_count"], "raw_factor_count"),
        hidden_dim=_int_config(config["hidden_dim"], "hidden_dim"),
        num_heads=_int_config(config["num_heads"], "num_heads"),
        inducing_points=_int_config(config["inducing_points"], "inducing_points"),
        num_blocks=_int_config(config["num_blocks"], "num_blocks"),
        dropout=_float_config(config["dropout"], "dropout"),
    )


def _require_fields(config: Mapping[str, object], expected: set[str]) -> None:
    if set(config) != expected:
        raise ValueError("model config has unsupported or missing fields")


def _int_config(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"model config {name} must be an integer")
    return value


def _float_config(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"model config {name} must be numeric")
    return float(value)


def _state_dict(payload: object) -> Mapping[str, torch.Tensor]:
    if not isinstance(payload, dict):
        raise ValueError("seed artifact payload must be a mapping")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not all(
        isinstance(key, str) and isinstance(value, torch.Tensor) for key, value in state.items()
    ):
        raise ValueError("seed artifact is not a tensor-only model_state_dict")
    return cast(Mapping[str, torch.Tensor], state)


def _validate_state_dict(
    state: Mapping[str, torch.Tensor],
    reference: Mapping[str, torch.Tensor],
) -> None:
    if set(state) != set(reference):
        raise ValueError("seed artifact state_dict keys do not match model contract")
    for key, tensor in state.items():
        expected = reference[key]
        if tensor.device.type != "cpu":
            raise ValueError("seed artifact state_dict must be loaded on CPU")
        if tensor.shape != expected.shape:
            raise ValueError(f"seed artifact state_dict shape mismatch: {key}")
        if tensor.dtype != expected.dtype:
            raise ValueError(f"seed artifact state_dict dtype mismatch: {key}")
