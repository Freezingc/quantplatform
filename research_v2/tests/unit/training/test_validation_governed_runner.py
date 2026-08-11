from __future__ import annotations

from datetime import date

import torch

from research_v2.models.base import FeatureBatch
from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
from research_v2.training.artifacts import CheckpointStore
from research_v2.training.batches import TrainingBatch
from research_v2.training.runner import run_validation_governed_training
from research_v2.training.trainer import DatasetRole, DatasetView, TrainerConfig


def _batch(signal_date: date) -> TrainingBatch:
    features = torch.randn((1, 30, 15), generator=torch.Generator().manual_seed(signal_date.day))
    inputs = FeatureBatch(
        signal_dates=(signal_date,),
        ts_codes=(tuple(f"S{index:03d}" for index in range(30)),),
        features=features,
        feature_missing=torch.zeros_like(features, dtype=torch.bool),
        stock_mask=torch.ones((1, 30), dtype=torch.bool),
    )
    target = features[:, :, 0]
    return TrainingBatch(inputs=inputs, target_raw=target, target_rank=target)


def test_runner_connects_train_validation_and_checkpoint_artifacts(tmp_path: object) -> None:
    train_date = date(2020, 1, 2)
    validation_date = date(2021, 1, 4)
    model = ResidualMLP(ResidualMLPConfig(raw_factor_count=15, hidden_dim=16, num_blocks=1))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    store = CheckpointStore(tmp_path / "checkpoints")  # type: ignore[operator]

    result = run_validation_governed_training(
        model,
        optimizer,
        train_batches={train_date: _batch(train_date)},
        validation_batches={validation_date: _batch(validation_date)},
        train_view=DatasetView(DatasetRole.TRAIN, "train-manifest"),
        validation_view=DatasetView(DatasetRole.VALIDATION, "validation-manifest"),
        trainer_config=TrainerConfig(max_epochs=1, early_stopping_patience=1),
        seed=20260801,
        checkpoint_store=store,
        checkpoint_prefix="F01-M1",
    )

    assert result.training.best_epoch == 1
    assert len(result.checkpoint_ids) == 1
    assert store.verify(result.checkpoint_ids[0])
