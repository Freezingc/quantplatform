from __future__ import annotations

import pandas as pd

from research_v2.training.ensemble import REQUIRED_SEEDS
from research_v2.training.model_bundle import ModelArtifactBundleStore


def test_model_bundle_is_atomic_complete_and_tamper_detected(tmp_path: object) -> None:
    root = tmp_path / "models"  # type: ignore[operator]
    source_dir = tmp_path / "seeds"  # type: ignore[operator]
    source_dir.mkdir()  # type: ignore[attr-defined]
    seed_files = {}
    for seed in REQUIRED_SEEDS:
        path = source_dir / f"{seed}.pt"  # type: ignore[operator]
        path.write_bytes(f"state-{seed}".encode())  # type: ignore[attr-defined]
        seed_files[seed] = path
    predictions = pd.DataFrame(
        {
            "signal_date": ["2025-01-02"],
            "ts_code": ["a"],
            "percentile_rank": [0.5],
        }
    )
    store = ModelArtifactBundleStore(root)

    bundle = store.write(
        "model-abc",
        model_card={"family": "RESIDUAL_MLP", "dataset_id": "dataset-v1"},
        config={"hidden_dim": 16},
        signature={"raw_factor_count": 15},
        feature_order=("f1",),
        preprocessing={"state_hash": "state-v1"},
        seed_checkpoints=seed_files,
        validation_predictions=predictions,
    )

    assert (bundle.path / "seed-20260801.pt").is_file()
    assert (bundle.path / "validation_predictions.parquet").is_file()
    assert store.verify("model-abc")
    metadata = store.load_metadata(
        "model-abc",
        expected_dataset_id="dataset-v1",
        expected_feature_order=("f1",),
        expected_signature={"raw_factor_count": 15},
        expected_preprocessor_hash="state-v1",
    )
    assert metadata["config"]["hidden_dim"] == 16
    (bundle.path / "config.json").write_text("tampered", encoding="utf-8")
    assert not store.verify("model-abc")
