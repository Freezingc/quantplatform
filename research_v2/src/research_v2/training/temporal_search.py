"""Spawn-safe M2 three-seed validation configuration trials.

The sibling of ``cross_section_search``, and the reason it cannot simply reuse it is the payload.
The static worker carries the fold's batches inside each task, which for M1/M3 costs a few minutes
of pickling per fold against ~57 hours of fitting -- noise. An M2 batch holds a 20-day window per
row and its validation panel is train+validation concatenated, so the same design would pickle
gigabytes per task and multiply that by the grid. It is not slow, it is not viable.

So a task here carries identifiers only -- workspace, fold, published dataset id, factor order and
the fitted preprocessing state -- and each worker rebuilds the fold's panels and batches ONCE and
caches them for every configuration it is later handed. The rebuild is the same bytes the parent
would produce, and that is a tested property rather than an assumption: the split is an immutable
verified dataset artifact, ``FoldPreprocessor.restore`` re-derives the state hash it was given, and
the window contract (20 days, 15 minimum) is stated once here. A worker that reconstructed a
slightly different panel would report a validation RankIC for a different model than the one the
parent finally fits and freezes, and the search would select on it.

Thread pinning stays the runner's decision: ``run_temporal_config_task`` pins the spawned child
before it builds a model, exactly as the static worker does, while batch reconstruction itself
pins nothing -- it is pandas work, and pinning in the parent would perturb a serial run.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import torch

from research_v2.datasets.preprocessing import FoldPreprocessor
from research_v2.datasets.storage import DatasetArtifactStore
from research_v2.inference.torch_scoring import percentile_frame, prepare_frame
from research_v2.models.temporal import (
    TemporalGRU,
    TemporalGRUConfig,
    TemporalTCN,
    TemporalTCNConfig,
)
from research_v2.training.artifacts import CheckpointStore
from research_v2.training.batches import TemporalTrainingBatch
from research_v2.training.ensemble import REQUIRED_SEEDS, build_seed_ensemble
from research_v2.training.runner import run_validation_governed_training
from research_v2.training.search_metrics import validation_rank_ic
from research_v2.training.seeds import configure_reproducibility, configure_torch_threads
from research_v2.training.temporal_batches import build_temporal_training_batches
from research_v2.training.trainer import DatasetRole, DatasetView, TrainerConfig

HISTORY_LENGTH = 20
"""Days of history each M2 observation is scored from. The driver's serial loop uses 20."""

MINIMUM_VALID_HISTORY_LENGTH = 15
"""How much of that window may be missing before the observation is dropped rather than padded."""

_FOLD_CACHE: dict[tuple[Any, ...], TemporalFoldBatches] = {}
_worker_torch_threads: int | None = None


@dataclass(frozen=True)
class TemporalFoldSpec:
    """Identifiers sufficient to rebuild one fold's temporal batches inside a spawned worker."""

    workspace: Path
    fold_id: str
    dataset_id: str
    factor_order: tuple[str, ...]
    preprocessing: Mapping[str, object]
    torch_threads: int

    def cache_key(self) -> tuple[Any, ...]:
        """Identity of the reconstruction, so two specs that would build different batches differ.

        The preprocessing hash is in the key rather than the payload: a worker handed the same fold
        under a different fitted state must not reuse the first one's batches.
        """
        return (
            str(self.workspace),
            self.fold_id,
            self.dataset_id,
            tuple(self.factor_order),
            str(self.preprocessing.get("state_hash")),
            self.torch_threads,
        )


@dataclass(frozen=True)
class TemporalFoldBatches:
    """One fold's rebuilt training inputs, plus the labels the selection metric scores against."""

    train_batches: Mapping[date, TemporalTrainingBatch]
    validation_batches: Mapping[date, TemporalTrainingBatch]
    validation_labels: pd.DataFrame


@dataclass(frozen=True)
class TemporalConfigTask:
    """One independently executable M2 configuration trial: three seeds, no durable writes."""

    config_index: int
    config_payload: Mapping[str, Any]
    fold: TemporalFoldSpec
    train_dataset_id: str
    validation_dataset_id: str
    checkpoint_root: Path
    checkpoint_prefix: str
    max_epochs: int
    early_stopping_patience: int


def temporal_panel_batches(
    panel: pd.DataFrame, factor_order: tuple[str, ...], allowed_dates: set[date]
) -> dict[date, TemporalTrainingBatch]:
    """Windowed batches for the allowed signal dates only.

    The window builder sees the whole panel -- which is why the validation panel is
    train+validation concatenated -- and the filter afterwards is what keeps a validation batch from
    being emitted for a training date.
    """
    batches = build_temporal_training_batches(
        panel,
        factor_order=factor_order,
        history_length=HISTORY_LENGTH,
        minimum_valid_history_length=MINIMUM_VALID_HISTORY_LENGTH,
    )
    return {day: batch for day, batch in batches.items() if day in allowed_dates}


def temporal_model(
    config_payload: Mapping[str, Any], factor_order: tuple[str, ...]
) -> torch.nn.Module:
    """Build the architecture the configuration names; the M2 grid searches GRU and TCN alike."""
    family = config_payload.get("family")
    hidden_dim = int(config_payload["hidden_dim"])
    dropout = float(config_payload["dropout"])
    if family == "GRU":
        return TemporalGRU(
            TemporalGRUConfig(
                raw_factor_count=len(factor_order), hidden_dim=hidden_dim, dropout=dropout
            )
        )
    if family == "TCN":
        return TemporalTCN(
            TemporalTCNConfig(
                raw_factor_count=len(factor_order), hidden_dim=hidden_dim, dropout=dropout
            )
        )
    raise ValueError(
        f"unsupported temporal family {family!r}; the M2 grid searches GRU and TCN"
    )


def temporal_test_panel_batches(
    prepared_train: pd.DataFrame,
    prepared_validation: pd.DataFrame,
    prepared_test: pd.DataFrame,
    factor_order: tuple[str, ...],
    test_dates: set[date],
) -> dict[date, TemporalTrainingBatch]:
    """Windowed TEST batches over the train+validation+test panel.

    A TEST row is scored from a window that reaches back through validation into train, so the
    panel has to be all three concatenated; filtering to ``test_dates`` afterwards is what keeps a
    batch from being emitted for a history-only date. One definition, shared by the driver that
    produces the frozen TEST predictions and the audit that recomputes them -- an audit that
    rebuilt the panel its own way would only ever be agreeing with itself.
    """
    panel = pd.concat(
        [prepared_train, prepared_validation, prepared_test], ignore_index=True
    )
    return temporal_panel_batches(panel, factor_order, test_dates)


def temporal_test_batches(spec: TemporalFoldSpec) -> Mapping[date, TemporalTrainingBatch]:
    """Rebuild one fold's TEST-side batches from published artifacts alone.

    This is the reproduction audit's entry point: it reads the three splits back out of the
    immutable dataset artifact, restores the fitted preprocessing state the bundle carries, and
    composes the panel exactly as the driver did when it scored the frozen TEST split.
    """
    store = DatasetArtifactStore(spec.workspace / "datasets")
    preprocessor = _restored_preprocessor(spec)
    train = store.load_split(spec.dataset_id, "train", expected_fold_id=spec.fold_id)
    validation = store.load_split(spec.dataset_id, "validation", expected_fold_id=spec.fold_id)
    test = store.load_split(spec.dataset_id, "test", expected_fold_id=spec.fold_id)
    test_dates = set(pd.to_datetime(test["signal_date"]).dt.date)
    return temporal_test_panel_batches(
        prepare_frame(train, preprocessor, spec.factor_order),
        prepare_frame(validation, preprocessor, spec.factor_order),
        prepare_frame(test, preprocessor, spec.factor_order),
        spec.factor_order,
        test_dates,
    )


def temporal_fold_batches(spec: TemporalFoldSpec) -> TemporalFoldBatches:
    """Rebuild one fold's temporal batches, once per worker, from published artifacts only."""
    key = spec.cache_key()
    cached = _FOLD_CACHE.get(key)
    if cached is not None:
        return cached
    store = DatasetArtifactStore(spec.workspace / "datasets")
    train = store.load_split(spec.dataset_id, "train", expected_fold_id=spec.fold_id)
    validation = store.load_split(spec.dataset_id, "validation", expected_fold_id=spec.fold_id)
    preprocessor = _restored_preprocessor(spec)
    prepared_train = prepare_frame(train, preprocessor, spec.factor_order)
    prepared_validation = prepare_frame(validation, preprocessor, spec.factor_order)
    train_dates = set(pd.to_datetime(train["signal_date"]).dt.date)
    validation_dates = set(pd.to_datetime(validation["signal_date"]).dt.date)
    panel = pd.concat([prepared_train, prepared_validation], ignore_index=True)
    built = TemporalFoldBatches(
        train_batches=temporal_panel_batches(prepared_train, spec.factor_order, train_dates),
        validation_batches=temporal_panel_batches(panel, spec.factor_order, validation_dates),
        validation_labels=validation,
    )
    _FOLD_CACHE[key] = built
    return built


def run_temporal_config_task(payload: Mapping[str, object]) -> dict[str, object]:
    """Train all required seeds for one configuration and return the serial loop's result shape."""
    task = payload.get("task")
    if not isinstance(task, TemporalConfigTask):
        raise ValueError("parallel temporal search task is malformed")
    _configure_worker_threads(task.fold.torch_threads)
    # Cached, so this costs nothing per seed below; it is here for the labels the ensemble is
    # scored against, which stay in the parent's hands rather than travelling per seed.
    fold = temporal_fold_batches(task.fold)
    seed_validation_predictions: dict[int, pd.DataFrame] = {}
    seed_rank_ic: dict[int, float] = {}
    for seed in REQUIRED_SEEDS:
        fitted = _fit_one_temporal_seed(
            config_payload=task.config_payload,
            seed=seed,
            fold=task.fold,
            train_dataset_id=task.train_dataset_id,
            validation_dataset_id=task.validation_dataset_id,
            checkpoint_root=task.checkpoint_root,
            checkpoint_prefix=f"{task.checkpoint_prefix}-{seed}",
            max_epochs=task.max_epochs,
            early_stopping_patience=task.early_stopping_patience,
        )
        seed_validation_predictions[seed] = fitted.predictions
        seed_rank_ic[seed] = fitted.validation_rank_ic
    ensemble = build_seed_ensemble(
        seed_validation_predictions, validation_rank_ic=seed_rank_ic
    )
    ensemble_predictions = ensemble.predictions.rename(
        columns={"ensemble_score": "percentile_rank"}
    )
    return {
        "config": dict(task.config_payload),
        "ensemble_validation_rank_ic": validation_rank_ic(
            ensemble_predictions, fold.validation_labels
        ),
        "seed_validation_rank_ic": seed_rank_ic,
        "worst_seed": ensemble.worst_seed,
    }


@dataclass(frozen=True)
class TemporalSeedFit:
    """One seeded temporal fit's durable output: its checkpoint and its validation scoring."""

    seed: int
    checkpoint_path: Path
    predictions: pd.DataFrame
    validation_rank_ic: float


@dataclass(frozen=True)
class TemporalSeedTask:
    """One seed of the selected configuration -- the unit the M2 final fit parallelises over.

    Carries the fold spec rather than batches, so a worker spawned for a final fit rebuilds the
    panels exactly as one spawned for a search trial does, and the cache means a worker that runs
    several of them pays for the reconstruction once.
    """

    config_payload: Mapping[str, Any]
    seed: int
    fold: TemporalFoldSpec
    train_dataset_id: str
    validation_dataset_id: str
    checkpoint_root: Path
    checkpoint_prefix: str
    max_epochs: int
    early_stopping_patience: int


def run_temporal_seed_task(payload: Mapping[str, object]) -> dict[str, object]:
    """Train ONE temporal seed in a spawned worker, returning what the bundle write consumes."""
    task = payload.get("task")
    if not isinstance(task, TemporalSeedTask):
        raise ValueError("parallel temporal seed task is malformed")
    _configure_worker_threads(task.fold.torch_threads)
    fitted = _fit_one_temporal_seed(
        config_payload=task.config_payload,
        seed=task.seed,
        fold=task.fold,
        train_dataset_id=task.train_dataset_id,
        validation_dataset_id=task.validation_dataset_id,
        checkpoint_root=task.checkpoint_root,
        checkpoint_prefix=task.checkpoint_prefix,
        max_epochs=task.max_epochs,
        early_stopping_patience=task.early_stopping_patience,
    )
    return {
        "seed": fitted.seed,
        "checkpoint_path": str(fitted.checkpoint_path),
        "predictions": fitted.predictions,
        "validation_rank_ic": fitted.validation_rank_ic,
    }


def _fit_one_temporal_seed(
    *,
    config_payload: Mapping[str, Any],
    seed: int,
    fold: TemporalFoldSpec,
    train_dataset_id: str,
    validation_dataset_id: str,
    checkpoint_root: Path,
    checkpoint_prefix: str,
    max_epochs: int,
    early_stopping_patience: int,
) -> TemporalSeedFit:
    """One seeded temporal fit, defined once for the config-level and the seed-level worker."""
    batches = temporal_fold_batches(fold)
    configure_reproducibility(seed)
    model = temporal_model(config_payload, fold.factor_order)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config_payload["learning_rate"])
    )
    checkpoint_store = CheckpointStore(checkpoint_root)
    run = run_validation_governed_training(
        model,
        optimizer,
        train_batches=batches.train_batches,
        validation_batches=batches.validation_batches,
        train_view=DatasetView(role=DatasetRole.TRAIN, dataset_id=train_dataset_id),
        validation_view=DatasetView(
            role=DatasetRole.VALIDATION, dataset_id=validation_dataset_id
        ),
        trainer_config=TrainerConfig(
            max_epochs=max_epochs,
            early_stopping_patience=early_stopping_patience,
        ),
        seed=seed,
        checkpoint_store=checkpoint_store,
        checkpoint_prefix=checkpoint_prefix,
    )
    predictions = percentile_frame(model, batches.validation_batches)
    return TemporalSeedFit(
        seed=seed,
        checkpoint_path=checkpoint_store._root / run.checkpoint_ids[-1] / "checkpoint.pt",
        predictions=predictions,
        validation_rank_ic=validation_rank_ic(predictions, batches.validation_labels),
    )


def _restored_preprocessor(spec: TemporalFoldSpec) -> FoldPreprocessor:
    """Restore the fitted state the parent used, refusing a spec that carries only a hash."""
    state_hash = spec.preprocessing.get("state_hash")
    if not isinstance(state_hash, str) or not state_hash:
        raise ValueError("temporal fold spec carries no preprocessor state hash")
    state_payload = spec.preprocessing.get("state_payload")
    if not isinstance(state_payload, Mapping):
        raise ValueError(
            "temporal fold spec carries a preprocessor hash but not the fitted state, so it "
            "cannot put a frame into the space the search is selecting in"
        )
    return FoldPreprocessor.restore(state_payload, expected_state_hash=state_hash)


def _configure_worker_threads(thread_count: int) -> None:
    """Pin both Torch pools once per spawned worker, before it constructs a model."""
    global _worker_torch_threads
    if _worker_torch_threads is None:
        configure_torch_threads(thread_count)
        _worker_torch_threads = thread_count
    elif _worker_torch_threads != thread_count:
        raise ValueError("one spawned worker cannot mix Torch thread counts")


__all__ = [
    "HISTORY_LENGTH",
    "MINIMUM_VALID_HISTORY_LENGTH",
    "TemporalConfigTask",
    "TemporalFoldBatches",
    "TemporalFoldSpec",
    "TemporalSeedFit",
    "TemporalSeedTask",
    "run_temporal_config_task",
    "run_temporal_seed_task",
    "temporal_fold_batches",
    "temporal_model",
    "temporal_panel_batches",
    "temporal_test_batches",
    "temporal_test_panel_batches",
]
