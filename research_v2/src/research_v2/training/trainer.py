"""Validation-governed trainer state machine with strict dataset role boundaries."""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from enum import StrEnum


class DatasetRole(StrEnum):
    TRAIN = "TRAIN"
    VALIDATION = "VALIDATION"
    TEST = "TEST"
    LOCKED = "LOCKED"


class TrainerState(StrEnum):
    INITIALIZED = "INITIALIZED"
    TRAINING = "TRAINING"
    VALIDATING = "VALIDATING"
    CHECKPOINTED = "CHECKPOINTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


@dataclass(frozen=True)
class DatasetView:
    role: DatasetRole
    dataset_id: str

    def __post_init__(self) -> None:
        if not self.dataset_id:
            raise ValueError("dataset_id cannot be empty")


@dataclass(frozen=True)
class TrainerConfig:
    max_epochs: int = 100
    early_stopping_patience: int = 10

    def __post_init__(self) -> None:
        if self.max_epochs < 1 or self.early_stopping_patience < 1:
            raise ValueError("epoch count and early-stopping patience must be positive")


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    validation_rank_ic: float


@dataclass(frozen=True)
class TrainingResult:
    best_epoch: int
    best_validation_rank_ic: float
    history: tuple[EpochRecord, ...]


class Trainer:
    """A callback-driven trainer shell that can never consult TEST or LOCKED data."""

    def __init__(
        self,
        train_view: DatasetView,
        validation_view: DatasetView,
        config: TrainerConfig,
    ) -> None:
        valid_roles = (
            train_view.role is DatasetRole.TRAIN
            and validation_view.role is DatasetRole.VALIDATION
        )
        if not valid_roles:
            raise PermissionError("Trainer accepts only TRAIN and VALIDATION dataset views")
        self._train_view = train_view
        self._validation_view = validation_view
        self._config = config
        self._state = TrainerState.INITIALIZED

    @property
    def state(self) -> TrainerState:
        return self._state

    def fit(
        self,
        *,
        train_epoch: Callable[[int], float],
        validation_epoch: Callable[[int], float],
        checkpoint_epoch: Callable[[int, float], None] | None = None,
    ) -> TrainingResult:
        """Train and checkpoint only on validation RankIC; callback exceptions mark failure."""
        if self._state is not TrainerState.INITIALIZED:
            raise RuntimeError("Trainer.fit may be called once from INITIALIZED state")
        history: list[EpochRecord] = []
        best_epoch = 0
        best_value = float("-inf")
        stale_epochs = 0
        try:
            for epoch in range(1, self._config.max_epochs + 1):
                self._state = TrainerState.TRAINING
                train_loss = float(train_epoch(epoch))
                if not math.isfinite(train_loss):
                    raise ValueError("train_epoch returned a non-finite loss")
                self._state = TrainerState.VALIDATING
                validation_rank_ic = float(validation_epoch(epoch))
                if not math.isfinite(validation_rank_ic):
                    raise ValueError("validation_epoch returned a non-finite RankIC")
                history.append(EpochRecord(epoch, train_loss, validation_rank_ic))
                if validation_rank_ic > best_value:
                    best_epoch = epoch
                    best_value = validation_rank_ic
                    stale_epochs = 0
                    self._state = TrainerState.CHECKPOINTED
                    if checkpoint_epoch is not None:
                        checkpoint_epoch(epoch, validation_rank_ic)
                else:
                    stale_epochs += 1
                if stale_epochs >= self._config.early_stopping_patience:
                    break
        except Exception:
            self._state = TrainerState.FAILED
            raise
        self._state = TrainerState.COMPLETED
        return TrainingResult(best_epoch, best_value, tuple(history))
