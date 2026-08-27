from __future__ import annotations

from dataclasses import asdict

import pandas as pd
import torch

from research_v2.inference.artifact_loader import load_inference_artifact
from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
from research_v2.models.set_attention import SetAttention, SetAttentionConfig
from research_v2.models.temporal import TemporalGRU, TemporalGRUConfig
from research_v2.training.ensemble import REQUIRED_SEEDS
from research_v2.training.model_bundle import ModelArtifactBundleStore


def test_allowlisted_loader_reads_only_verified_three_seed_state_dicts(tmp_path: object) -> None:
    model = ResidualMLP(ResidualMLPConfig(raw_factor_count=15, hidden_dim=16, num_blocks=1))
    seeds = {}
    for seed in REQUIRED_SEEDS:
        path = tmp_path / f"seed-{seed}.pt"  # type: ignore[operator]
        torch.save({"model_state_dict": model.state_dict()}, path)
        seeds[seed] = path
    store = ModelArtifactBundleStore(tmp_path / "models")  # type: ignore[operator]
    store.write(
        "model-1",
        model_card={
            "dataset_id": "dataset-1",
            "family": "RESIDUAL_MLP",
            "model_status": "WALK_FORWARD_PASSED",
        },
        config={"raw_factor_count": 15, "hidden_dim": 16, "num_blocks": 1, "dropout": 0.1},
        signature=asdict(model.artifact_signature()),
        feature_order=tuple(f"f{index}" for index in range(15)),
        preprocessing={"state_hash": "pre-1"},
        seed_checkpoints=seeds,
        validation_predictions=pd.DataFrame(
            {"signal_date": ["2025-01-02"], "ts_code": ["a"], "percentile_rank": [0.5]}
        ),
    )
    loaded = load_inference_artifact(
        store,
        "model-1",
        expected_dataset_id="dataset-1",
        expected_feature_order=tuple(f"f{index}" for index in range(15)),
        expected_preprocessor_hash="pre-1",
        device=torch.device("cpu"),
    )

    assert len(loaded.models) == 3
    assert all(not candidate.training for candidate in loaded.models)


def test_allowlisted_loader_reconstructs_gru_and_set_attention_bundles(tmp_path: object) -> None:
    cases = (
        (
            "gru-model",
            "TEMPORAL_GRU",
            TemporalGRU(TemporalGRUConfig(raw_factor_count=2, hidden_dim=4, dropout=0.0)),
            {"raw_factor_count": 2, "hidden_dim": 4, "dropout": 0.0},
            TemporalGRU,
        ),
        (
            "set-model",
            "SET_ATTENTION",
            SetAttention(
                SetAttentionConfig(
                    raw_factor_count=2,
                    hidden_dim=4,
                    num_heads=2,
                    inducing_points=2,
                    num_blocks=1,
                    dropout=0.0,
                )
            ),
            {
                "raw_factor_count": 2,
                "hidden_dim": 4,
                "num_heads": 2,
                "inducing_points": 2,
                "num_blocks": 1,
                "dropout": 0.0,
            },
            SetAttention,
        ),
    )
    store = ModelArtifactBundleStore(tmp_path / "models")  # type: ignore[operator]
    for model_id, family, model, config, expected_type in cases:
        seeds = {}
        for seed in REQUIRED_SEEDS:
            path = tmp_path / f"{model_id}-{seed}.pt"  # type: ignore[operator]
            torch.save({"model_state_dict": model.state_dict()}, path)
            seeds[seed] = path
        store.write(
            model_id,
            model_card={
                "dataset_id": f"{model_id}-dataset",
                "family": family,
                "model_status": "WALK_FORWARD_PASSED",
            },
            config=config,
            signature=asdict(model.artifact_signature()),
            feature_order=("factor_a", "factor_b"),
            preprocessing={"state_hash": f"{model_id}-preprocessor"},
            seed_checkpoints=seeds,
            validation_predictions=pd.DataFrame(
                {"signal_date": ["2025-01-02"], "ts_code": ["a"], "percentile_rank": [0.5]}
            ),
        )

        loaded = load_inference_artifact(
            store,
            model_id,
            expected_dataset_id=f"{model_id}-dataset",
            expected_feature_order=("factor_a", "factor_b"),
            expected_preprocessor_hash=f"{model_id}-preprocessor",
            device=torch.device("cpu"),
        )

        assert len(loaded.models) == 3
        assert all(isinstance(candidate, expected_type) for candidate in loaded.models)
