from __future__ import annotations

import pandas as pd
import torch

from research_v2.inference.artifact_loader import load_inference_artifact
from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
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
        signature={"raw_factor_count": 15},
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
