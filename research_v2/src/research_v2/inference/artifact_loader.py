"""Allowlisted, hash-verified CPU-first loaders for immutable inference artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
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
    """Load only known model classes and tensors after full bundle-contract verification."""
    metadata = store.load_metadata(
        model_id,
        expected_dataset_id=expected_dataset_id,
        expected_feature_order=expected_feature_order,
        expected_signature={"raw_factor_count": len(expected_feature_order)},
        expected_preprocessor_hash=expected_preprocessor_hash,
    )
    model_card = metadata["model_card"]
    if model_card.get("model_status") != "WALK_FORWARD_PASSED":
        raise ValueError("inference requires a WALK_FORWARD_PASSED model")
    if model_card.get("family") != "RESIDUAL_MLP":
        raise ValueError("model family is not allowlisted for inference")
    config = _residual_config(metadata["config"])
    models: list[nn.Module] = []
    for seed in REQUIRED_SEEDS:
        model = ResidualMLP(config)
        payload = torch.load(
            store.bundle_path(model_id) / f"seed-{seed}.pt",
            map_location="cpu",
            weights_only=True,
        )
        state = _state_dict(payload)
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        models.append(model)
    return LoadedInferenceArtifact(
        model_id=model_id,
        models=tuple(models),
        preprocessing=metadata["preprocessing"],
    )


def _residual_config(config: Mapping[str, object]) -> ResidualMLPConfig:
    expected = {"raw_factor_count", "hidden_dim", "num_blocks", "dropout"}
    if set(config) != expected:
        raise ValueError("residual model config has unsupported or missing fields")
    raw_factor_count = _int_config(config["raw_factor_count"], "raw_factor_count")
    hidden_dim = _int_config(config["hidden_dim"], "hidden_dim")
    num_blocks = _int_config(config["num_blocks"], "num_blocks")
    dropout = _float_config(config["dropout"], "dropout")
    return ResidualMLPConfig(
        raw_factor_count=raw_factor_count,
        hidden_dim=hidden_dim,
        num_blocks=num_blocks,
        dropout=dropout,
    )


def _int_config(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"residual config {name} must be an integer")
    return value


def _float_config(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"residual config {name} must be numeric")
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
