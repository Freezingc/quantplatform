from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import cast

import pandas as pd
import pytest
import torch

from research_v2.inference.artifact_loader import load_inference_artifact
from research_v2.models.base import FeatureBatch, TemporalFeatureBatch
from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
from research_v2.models.set_attention import SetAttention, SetAttentionConfig
from research_v2.models.temporal import TemporalGRU, TemporalGRUConfig
from research_v2.training.ensemble import REQUIRED_SEEDS
from research_v2.training.model_bundle import ModelArtifactBundleStore

_GOLDEN_ROOT = Path(__file__).with_name("data")
_BATCH_PATH = _GOLDEN_ROOT / "model_batch.pt"
_EXPECTED_PATH = _GOLDEN_ROOT / "expected_model_outputs.json"


@pytest.mark.parametrize("case", ("M1", "M2_GRU", "M3"))
def test_cpu_golden_outputs_and_artifact_reconstruction(tmp_path: Path, case: str) -> None:
    fixture = cast(
        dict[str, object],
        torch.load(_BATCH_PATH, map_location="cpu", weights_only=True),
    )
    expected = cast(dict[str, object], json.loads(_EXPECTED_PATH.read_text(encoding="utf-8")))
    model, batch, family, config = _build_case(case, fixture, expected)
    state = cast(dict[str, torch.Tensor], fixture[f"{case.lower()}_state_dict"])
    model.load_state_dict(state)
    model.cpu().eval()
    with torch.no_grad():
        scores = model(batch)
    case_expected = cast(dict[str, object], cast(dict[str, object], expected["cases"])[case])
    assert torch.allclose(
        scores,
        torch.tensor(case_expected["scores"], dtype=torch.float32),
        rtol=0.0,
        atol=1e-6,
    )

    checkpoints = _write_seed_checkpoints(tmp_path, case, state)
    store = ModelArtifactBundleStore(tmp_path / "models")
    model_id = f"golden-{case.lower()}"
    feature_order = tuple(cast(list[str], expected["feature_order"]))
    store.write(
        model_id,
        model_card={
            "dataset_id": f"{model_id}-dataset",
            "family": family,
            "model_status": "WALK_FORWARD_PASSED",
        },
        config=config,
        signature=asdict(model.artifact_signature()),
        feature_order=feature_order,
        preprocessing={"state_hash": f"{model_id}-preprocessor"},
        seed_checkpoints=checkpoints,
        validation_predictions=pd.DataFrame(
            {"signal_date": ["2025-01-02"], "ts_code": ["a"], "percentile_rank": [0.5]}
        ),
    )
    loaded = load_inference_artifact(
        store,
        model_id,
        expected_dataset_id=f"{model_id}-dataset",
        expected_feature_order=feature_order,
        expected_preprocessor_hash=f"{model_id}-preprocessor",
        device=torch.device("cpu"),
    )
    with torch.no_grad():
        reconstructed = loaded.models[0](batch)
    assert torch.allclose(reconstructed, scores, rtol=0.0, atol=1e-6)
    with pytest.raises(ValueError, match="feature_order"):
        load_inference_artifact(
            store,
            model_id,
            expected_dataset_id=f"{model_id}-dataset",
            expected_feature_order=tuple(reversed(feature_order)),
            expected_preprocessor_hash=f"{model_id}-preprocessor",
            device=torch.device("cpu"),
        )


def _build_case(
    case: str,
    fixture: dict[str, object],
    expected: dict[str, object],
) -> tuple[torch.nn.Module, FeatureBatch | TemporalFeatureBatch, str, dict[str, object]]:
    case_config = cast(dict[str, object], cast(dict[str, object], expected["cases"])[case])
    config = cast(dict[str, object], case_config["config"])
    if case == "M1":
        return (
            ResidualMLP(ResidualMLPConfig(**cast(dict[str, int | float], config))),
            _feature_batch(fixture),
            "RESIDUAL_MLP",
            config,
        )
    if case == "M2_GRU":
        return (
            TemporalGRU(TemporalGRUConfig(**cast(dict[str, int | float], config))),
            _temporal_batch(fixture),
            "TEMPORAL_GRU",
            config,
        )
    if case == "M3":
        return (
            SetAttention(SetAttentionConfig(**cast(dict[str, int | float], config))),
            _feature_batch(fixture),
            "SET_ATTENTION",
            config,
        )
    raise AssertionError(f"unknown golden case: {case}")


def _feature_batch(fixture: dict[str, object]) -> FeatureBatch:
    return FeatureBatch(
        signal_dates=(date(2025, 1, 2),),
        ts_codes=(("a", "b", ""),),
        features=cast(torch.Tensor, fixture["cross_section_features"]),
        feature_missing=cast(torch.Tensor, fixture["cross_section_missing"]),
        stock_mask=cast(torch.Tensor, fixture["cross_section_stock_mask"]),
    )


def _temporal_batch(fixture: dict[str, object]) -> TemporalFeatureBatch:
    return TemporalFeatureBatch(
        signal_dates=(date(2025, 1, 2),),
        ts_codes=(("a", "b", ""),),
        features=cast(torch.Tensor, fixture["temporal_features"]),
        feature_missing=cast(torch.Tensor, fixture["temporal_missing"]),
        stock_mask=cast(torch.Tensor, fixture["temporal_stock_mask"]),
        history_mask=cast(torch.Tensor, fixture["temporal_history_mask"]),
    )


def _write_seed_checkpoints(
    root: Path,
    case: str,
    state: dict[str, torch.Tensor],
) -> dict[int, Path]:
    checkpoints: dict[int, Path] = {}
    for seed in REQUIRED_SEEDS:
        path = root / f"{case}-{seed}.pt"
        torch.save({"model_state_dict": state}, path)
        checkpoints[seed] = path
    return checkpoints
