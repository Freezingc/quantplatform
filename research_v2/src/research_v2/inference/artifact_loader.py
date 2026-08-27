"""Allowlisted, hash-verified CPU-first loaders for immutable inference artifacts."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from typing import cast

import torch
from torch import nn

from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
from research_v2.models.set_attention import SetAttention, SetAttentionConfig
from research_v2.models.temporal import TemporalGRU, TemporalGRUConfig
from research_v2.training.ensemble import REQUIRED_SEEDS
from research_v2.training.model_bundle import ModelArtifactBundleStore


@dataclass(frozen=True)
class LoadedInferenceArtifact:
    model_id: str
    models: tuple[nn.Module, ...]
    preprocessing: Mapping[str, object]


def load_inference_artifact(
    store: ModelArtifactBundleStore,
    model_id: str,
    *,
    expected_dataset_id: str,
    expected_feature_order: tuple[str, ...],
    expected_preprocessor_hash: str,
    device: torch.device,
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
    if model_card.get("model_status") != "WALK_FORWARD_PASSED":
        raise ValueError("inference requires a WALK_FORWARD_PASSED model")
    factory = _allowlisted_factory(model_card, metadata["config"])
    prototype = cast(ResidualMLP | TemporalGRU | SetAttention, factory())
    expected_signature = asdict(prototype.artifact_signature())
    if dict(metadata["signature"]) != expected_signature:
        raise ValueError("model bundle signature does not match reconstructed model contract")
    models: list[nn.Module] = []
    for seed in REQUIRED_SEEDS:
        model = factory()
        payload = torch.load(
            store.bundle_path(model_id) / f"seed-{seed}.pt",
            map_location="cpu",
            weights_only=True,
        )
        model.load_state_dict(_state_dict(payload))
        model.to(device)
        model.eval()
        models.append(model)
    return LoadedInferenceArtifact(
        model_id=model_id,
        models=tuple(models),
        preprocessing=metadata["preprocessing"],
    )


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
