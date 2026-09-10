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
    seed_validation_predictions: dict[int, pd.DataFrame] = {}
    seed_rank_ic: dict[int, float] = {}
    for seed in REQUIRED_SEEDS:
        fitted = _fit_one_seed(
            family=task.family,
            config_payload=task.config_payload,
            factor_order=task.factor_order,
            seed=seed,
            train_batches=task.train_batches,
            validation_batches=task.validation_batches,
            validation_labels=task.validation_labels,
            checkpoint_root=task.checkpoint_root,
            checkpoint_prefix=f"{task.checkpoint_prefix}-{seed}",
            train_dataset_id=task.train_dataset_id,
            validation_dataset_id=task.validation_dataset_id,
            max_epochs=task.max_epochs,
            early_stopping_patience=task.early_stopping_patience,
        )
        seed_validation_predictions[seed] = fitted.predictions
        seed_rank_ic[seed] = fitted.validation_rank_ic
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


@dataclass(frozen=True)
class SeedFit:
    """One seeded fit's durable output: where its weights landed and how it scored validation."""

    seed: int
    checkpoint_path: Path
    predictions: pd.DataFrame
    validation_rank_ic: float


@dataclass(frozen=True)
class CrossSectionSeedTask:
    """One seed of one configuration -- the unit the FINAL three-seed fit parallelises over.

    The same inputs as ``CrossSectionConfigTask`` with a seed in place of a config index, because a
    final fit is one seed of the already-selected configuration and nothing more.
    """

    family: CrossSectionFamily
    config_payload: Mapping[str, Any]
    factor_order: tuple[str, ...]
    seed: int
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


def run_cross_section_seed_task(payload: Mapping[str, object]) -> dict[str, object]:
    """Train ONE seed in a spawned worker and hand the parent everything the bundle write needs."""
    task = payload.get("task")
    if not isinstance(task, CrossSectionSeedTask):
        raise ValueError("parallel cross-section seed task is malformed")
    _configure_worker_threads(task.torch_threads)
    fitted = _fit_one_seed(
        family=task.family,
        config_payload=task.config_payload,
        factor_order=task.factor_order,
        seed=task.seed,
        train_batches=task.train_batches,
        validation_batches=task.validation_batches,
        validation_labels=task.validation_labels,
        checkpoint_root=task.checkpoint_root,
        checkpoint_prefix=task.checkpoint_prefix,
        train_dataset_id=task.train_dataset_id,
        validation_dataset_id=task.validation_dataset_id,
        max_epochs=task.max_epochs,
        early_stopping_patience=task.early_stopping_patience,
    )
    return {
        "seed": fitted.seed,
        "checkpoint_path": str(fitted.checkpoint_path),
        "predictions": fitted.predictions,
        "validation_rank_ic": fitted.validation_rank_ic,
    }


def _fit_one_seed(
    *,
    family: CrossSectionFamily,
    config_payload: Mapping[str, Any],
    factor_order: tuple[str, ...],
    seed: int,
    train_batches: Mapping[date, TrainingBatch],
    validation_batches: Mapping[date, TrainingBatch],
    validation_labels: pd.DataFrame,
    checkpoint_root: Path,
    checkpoint_prefix: str,
    train_dataset_id: str,
    validation_dataset_id: str,
    max_epochs: int,
    early_stopping_patience: int,
) -> SeedFit:
    """One seeded fit, defined once for both the config-level and the seed-level worker.

    Returning the checkpoint path is what lets the parent freeze a bundle from a fit it did not run
    itself: the bundle is written from these paths, so a worker that trained the weights and threw
    the path away would leave the parent with an ensemble it could score but not persist.
    """
    configure_reproducibility(seed)
    model = _model(family, config_payload, factor_order)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config_payload.get("learning_rate", 0.003)),
    )
    checkpoint_store = CheckpointStore(checkpoint_root)
    run = run_validation_governed_training(
        model,
        optimizer,
        train_batches=train_batches,
        validation_batches=validation_batches,
        train_view=DatasetView(role=DatasetRole.TRAIN, dataset_id=train_dataset_id),
        validation_view=DatasetView(
            role=DatasetRole.VALIDATION,
            dataset_id=validation_dataset_id,
        ),
        trainer_config=TrainerConfig(
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
        ),
        seed=seed,
        checkpoint_store=checkpoint_store,
        checkpoint_prefix=checkpoint_prefix,
    )
    predictions = percentile_frame(model, validation_batches)
    return SeedFit(
        seed=seed,
        checkpoint_path=checkpoint_store._root / run.checkpoint_ids[-1] / "checkpoint.pt",
        predictions=predictions,
        validation_rank_ic=validation_rank_ic(predictions, validation_labels),
    )


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


__all__ = [
    "CrossSectionConfigTask",
    "CrossSectionFamily",
    "CrossSectionSeedTask",
    "SeedFit",
    "run_cross_section_config_task",
    "run_cross_section_seed_task",
]
