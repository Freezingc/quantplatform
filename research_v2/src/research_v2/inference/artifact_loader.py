"""Allowlisted, hash-verified CPU-first loaders for immutable inference artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from types import MappingProxyType
from typing import TypeAlias, cast

import torch
from torch import nn

from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
from research_v2.models.set_attention import SetAttention, SetAttentionConfig
from research_v2.models.temporal import TemporalGRU, TemporalGRUConfig
from research_v2.training.ensemble import REQUIRED_SEEDS
from research_v2.training.model_bundle import ModelArtifactBundleStore

InferenceContract: TypeAlias = Mapping[str, object]
_INFERENCE_CONTRACT_FIELDS = frozenset(
    {
        "factor_set_id",
        "universe_id",
        "universe_version",
        "label_id",
        "label_version",
        "split_id",
        "code_version",
        "model_code_version",
    }
)


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


@dataclass(frozen=True)
class LoadedInferenceArtifact:
    model_id: str
    models: tuple[nn.Module, ...]
    preprocessing: Mapping[str, object]
    audit: InferenceLoadAudit | None = None


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
) -> LoadedInferenceArtifact:
    """Load known model classes only after full bundle and reconstruction-contract checks."""
    metadata = store.load_metadata(
        model_id,
        expected_dataset_id=expected_dataset_id,
        expected_feature_order=expected_feature_order,
        expected_signature=None,
        expected_preprocessor_hash=expected_preprocessor_hash,
    )
    model_card = metadata["model_card"]
    _validate_inference_contract(model_card, expected_inference_contract)
    if model_card.get("model_status") != "WALK_FORWARD_PASSED":
        raise ValueError("inference requires a WALK_FORWARD_PASSED model")
    factor_set_status = model_card.get("factor_set_status")
    if factor_set_status is not None and factor_set_status != "FROZEN":
        raise ValueError("inference requires a FROZEN factor set")
    bundle_digest = store.bundle_digest(model_id)
    artifact_hashes = store.artifact_hashes(model_id)
    factory = _allowlisted_factory(model_card, metadata["config"])
    prototype = cast(ResidualMLP | TemporalGRU | SetAttention, factory())
    expected_signature = asdict(prototype.artifact_signature())
    if dict(metadata["signature"]) != expected_signature:
        raise ValueError("model bundle signature does not match reconstructed model contract")
    reference_state = cast(Mapping[str, torch.Tensor], prototype.state_dict())
    seed_states: dict[int, Mapping[str, torch.Tensor]] = {}
    for seed in REQUIRED_SEEDS:
        payload = torch.load(
            store.bundle_path(model_id) / f"seed-{seed}.pt",
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
        audit=InferenceLoadAudit(
            model_id=model_id,
            bundle_digest=bundle_digest,
            artifact_hashes=MappingProxyType(dict(artifact_hashes)),
            device=str(device),
            code_version=_model_card_code_version(model_card),
            loaded_at=datetime.now(UTC),
            run_id=run_id,
        ),
    )


def _validate_inference_contract(
    model_card: Mapping[str, object],
    expected: InferenceContract | None,
) -> None:
    if expected is None:
        return
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
