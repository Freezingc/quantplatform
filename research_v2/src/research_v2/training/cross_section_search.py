"""Spawn-safe M1/M3 three-seed validation configuration trials."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import torch

from research_v2.inference.torch_scoring import percentile_frame
from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
from research_v2.models.set_attention import SetAttention, SetAttentionConfig
from research_v2.training.artifacts import CheckpointStore
from research_v2.training.batches import TrainingBatch
from research_v2.training.ensemble import REQUIRED_SEEDS, build_seed_ensemble
from research_v2.training.runner import run_validation_governed_training
from research_v2.training.search_metrics import validation_rank_ic
from research_v2.training.seeds import configure_reproducibility, configure_torch_threads
from research_v2.training.trainer import DatasetRole, DatasetView, TrainerConfig

CrossSectionFamily = Literal["M1", "M3"]
_worker_torch_threads: int | None = None


@dataclass(frozen=True)
class CrossSectionConfigTask:
    """All read-only inputs for one independently executable config trial."""

    family: CrossSectionFamily
    config_index: int
    config_payload: Mapping[str, Any]
    factor_order: tuple[str, ...]
    train_batches: Mapping[date, TrainingBatch]
    validation_batches: Mapping[date, TrainingBatch]
    validation_labels: pd.DataFrame
    checkpoint_root: Path
    checkpoint_prefix: str
    train_dataset_id: str
    validation_dataset_id: str
    max_epochs: int
    early_stopping_patience: int
    torch_threads: int


def run_cross_section_config_task(payload: Mapping[str, object]) -> dict[str, object]:
    """Train all required seeds and return the serial loop's validation result shape."""
    task = payload.get("task")
    if not isinstance(task, CrossSectionConfigTask):
        raise ValueError("parallel cross-section search task is malformed")
    _configure_worker_threads(task.torch_threads)
    checkpoint_store = CheckpointStore(task.checkpoint_root)
    seed_validation_predictions: dict[int, pd.DataFrame] = {}
    seed_rank_ic: dict[int, float] = {}
    for seed in REQUIRED_SEEDS:
        configure_reproducibility(seed)
        model = _model(task.family, task.config_payload, task.factor_order)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(task.config_payload.get("learning_rate", 0.003)),
        )
        run_validation_governed_training(
            model,
            optimizer,
            train_batches=task.train_batches,
            validation_batches=task.validation_batches,
            train_view=DatasetView(role=DatasetRole.TRAIN, dataset_id=task.train_dataset_id),
            validation_view=DatasetView(
                role=DatasetRole.VALIDATION,
                dataset_id=task.validation_dataset_id,
            ),
            trainer_config=TrainerConfig(
                max_epochs=task.max_epochs,
                early_stopping_patience=task.early_stopping_patience,
            ),
            seed=seed,
            checkpoint_store=checkpoint_store,
            checkpoint_prefix=f"{task.checkpoint_prefix}-{seed}",
        )
        predictions = percentile_frame(model, task.validation_batches)
        seed_validation_predictions[seed] = predictions
        seed_rank_ic[seed] = validation_rank_ic(predictions, task.validation_labels)
    ensemble = build_seed_ensemble(
        seed_validation_predictions,
        validation_rank_ic=seed_rank_ic,
    )
    ensemble_predictions = ensemble.predictions.rename(
        columns={"ensemble_score": "percentile_rank"}
    )
    return {
        "config": dict(task.config_payload),
        "ensemble_validation_rank_ic": validation_rank_ic(
            ensemble_predictions,
            task.validation_labels,
        ),
        "seed_validation_rank_ic": seed_rank_ic,
        "worst_seed": ensemble.worst_seed,
    }


def _configure_worker_threads(thread_count: int) -> None:
    """Set inter-op threads once per spawned worker before it builds a model."""
    global _worker_torch_threads
    if _worker_torch_threads is None:
        configure_torch_threads(thread_count)
        _worker_torch_threads = thread_count
    elif _worker_torch_threads != thread_count:
        raise ValueError("one spawned worker cannot mix Torch thread counts")


def _model(
    family: CrossSectionFamily,
    config_payload: Mapping[str, Any],
    factor_order: tuple[str, ...],
) -> torch.nn.Module:
    if family == "M1":
        return ResidualMLP(
            ResidualMLPConfig(
                raw_factor_count=len(factor_order),
                hidden_dim=int(config_payload["hidden_dim"]),
                num_blocks=int(config_payload["num_blocks"]),
                dropout=float(config_payload["dropout"]),
            )
        )
    if family == "M3":
        return SetAttention(
            SetAttentionConfig(
                raw_factor_count=len(factor_order),
                hidden_dim=int(config_payload["hidden_dim"]),
                num_heads=int(config_payload["num_heads"]),
                inducing_points=int(config_payload["inducing_points"]),
                num_blocks=int(config_payload["num_blocks"]),
                dropout=float(config_payload["dropout"]),
            )
        )
    raise ValueError(f"unsupported parallel cross-section family: {family}")


__all__ = ["CrossSectionConfigTask", "CrossSectionFamily", "run_cross_section_config_task"]
