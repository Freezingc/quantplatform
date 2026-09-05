"""End-to-end train/validation orchestration with immutable best-checkpoint artifacts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import torch
from torch import nn

from research_v2.training.artifacts import CheckpointStore
from research_v2.training.batches import SupervisedBatch
from research_v2.training.seeds import configure_reproducibility
from research_v2.training.torch_loop import TorchDateTrainingLoop
from research_v2.training.trainer import DatasetView, Trainer, TrainerConfig, TrainingResult
from research_v2.training.validation import evaluate_validation_batches


@dataclass(frozen=True)
class ValidationGovernedRun:
    training: TrainingResult
    checkpoint_ids: tuple[str, ...]


def run_validation_governed_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    train_batches: Mapping[date, SupervisedBatch],
    validation_batches: Mapping[date, SupervisedBatch],
    train_view: DatasetView,
    validation_view: DatasetView,
    trainer_config: TrainerConfig,
    seed: int,
    checkpoint_store: CheckpointStore,
    checkpoint_prefix: str,
) -> ValidationGovernedRun:
    """Train on TRAIN, select only on validation RankIC, and atomically save improvements."""
    if not checkpoint_prefix:
        raise ValueError("checkpoint_prefix cannot be empty")
    environment = configure_reproducibility(seed)
    loop = TorchDateTrainingLoop(
        model,
        optimizer,
        train_batches,
        seed=seed,
        gradient_clip_norm=1.0,
    )
    trainer = Trainer(train_view, validation_view, trainer_config)
    checkpoint_ids: list[str] = []
    # Device is numerics-affecting (measured: cpu and cuda digests differ for the same
    # seed/config/data), and the loop already derives it from the model's parameters,
    # so record the same value the loop will actually train on.
    first_parameter = next(model.parameters(), None)
    model_device = str(first_parameter.device) if first_parameter is not None else "cpu"

    def validation_epoch(_: int) -> float:
        return evaluate_validation_batches(model, validation_batches).summary.mean_rank_ic

    def checkpoint_epoch(epoch: int, validation_rank_ic: float) -> None:
        checkpoint_id = f"{checkpoint_prefix}-seed-{seed}-epoch-{epoch}"
        checkpoint_store.save(
            checkpoint_id,
            model,
            optimizer,
            metadata={
                "epoch": epoch,
                "seed": seed,
                "torch_version": environment.torch_version,
                "torch_intraop_threads": environment.torch_intraop_threads,
                "torch_interop_threads": environment.torch_interop_threads,
                "model_device": model_device,
                "train_dataset_id": train_view.dataset_id,
                "validation_dataset_id": validation_view.dataset_id,
                "validation_rank_ic": validation_rank_ic,
            },
        )
        checkpoint_ids.append(checkpoint_id)

    training = trainer.fit(
        train_epoch=loop.train_epoch,
        validation_epoch=validation_epoch,
        checkpoint_epoch=checkpoint_epoch,
    )
    return ValidationGovernedRun(training=training, checkpoint_ids=tuple(checkpoint_ids))
