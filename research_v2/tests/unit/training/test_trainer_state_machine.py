from __future__ import annotations

import pytest

from research_v2.training.trainer import (
    DatasetRole,
    DatasetView,
    Trainer,
    TrainerConfig,
    TrainerState,
)


def test_trainer_selects_best_validation_epoch_and_early_stops() -> None:
    trainer = Trainer(
        DatasetView(DatasetRole.TRAIN, "train-id"),
        DatasetView(DatasetRole.VALIDATION, "validation-id"),
        TrainerConfig(max_epochs=10, early_stopping_patience=2),
    )
    validation_values = iter([0.01, 0.03, 0.02, 0.01, 0.00])

    result = trainer.fit(
        train_epoch=lambda epoch: 1.0 / (epoch + 1),
        validation_epoch=lambda epoch: next(validation_values),
    )

    assert result.best_epoch == 2
    assert result.best_validation_rank_ic == pytest.approx(0.03)
    assert len(result.history) == 4
    assert trainer.state is TrainerState.COMPLETED


def test_trainer_checkpoints_only_validation_improvements() -> None:
    trainer = Trainer(
        DatasetView(DatasetRole.TRAIN, "train-id"),
        DatasetView(DatasetRole.VALIDATION, "validation-id"),
        TrainerConfig(max_epochs=4, early_stopping_patience=3),
    )
    saved: list[tuple[int, float]] = []

    trainer.fit(
        train_epoch=lambda epoch: 1.0,
        validation_epoch=lambda epoch: (0.01, 0.03, 0.02, 0.04)[epoch - 1],
        checkpoint_epoch=lambda epoch, value: saved.append((epoch, value)),
    )

    assert saved == [(1, 0.01), (2, 0.03), (4, 0.04)]


def test_trainer_rejects_test_or_locked_views_before_training() -> None:
    validation = DatasetView(DatasetRole.VALIDATION, "validation-id")
    for forbidden_role in (DatasetRole.TEST, DatasetRole.LOCKED):
        with pytest.raises(PermissionError, match="TRAIN and VALIDATION"):
            Trainer(
                DatasetView(forbidden_role, "forbidden"),
                validation,
                TrainerConfig(max_epochs=1, early_stopping_patience=1),
            )
