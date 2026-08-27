"""Deterministic expanding time folds contained inside one outer Train dataset."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from research_v2.baselines.base import BaselineDataset


@dataclass(frozen=True)
class InnerBaselineFold:
    inner_fold_id: str
    train: BaselineDataset
    validation: BaselineDataset


def expanding_inner_folds(
    dataset: BaselineDataset,
    *,
    fold_count: int = 3,
) -> tuple[InnerBaselineFold, ...]:
    """Create three expanding folds from five chronological date blocks.

    The first two blocks form the warm-up period. Each following block is evaluated
    once, so every returned validation date is later than every date used to fit it.
    """
    if dataset.role != "TRAIN":
        raise PermissionError("inner folds can only be created from TRAIN data")
    if fold_count != 3:
        raise ValueError("the baseline protocol requires exactly three inner folds")
    dates = sorted(pd.to_datetime(dataset.features["signal_date"], errors="raise").dt.date.unique())
    if len(dates) < 5:
        raise ValueError("three expanding inner folds require at least five training dates")
    blocks = [
        dates[(len(dates) * index) // 5 : (len(dates) * (index + 1)) // 5]
        for index in range(5)
    ]
    folds: list[InnerBaselineFold] = []
    for validation_index in (2, 3, 4):
        validation_dates = set(blocks[validation_index])
        train_dates = set(date for block in blocks[:validation_index] for date in block)
        train_features = _date_slice(dataset.features, train_dates)
        train_labels = _date_slice(dataset.labels, train_dates)
        validation_features = _date_slice(dataset.features, validation_dates)
        validation_labels = _date_slice(dataset.labels, validation_dates)
        folds.append(
            InnerBaselineFold(
                inner_fold_id=f"{dataset.fold_id}.I{validation_index - 1}",
                train=BaselineDataset(
                    role="TRAIN",
                    fold_id=dataset.fold_id,
                    dataset_id=f"{dataset.dataset_id}:inner-train-{validation_index - 1}",
                    features=train_features,
                    labels=train_labels,
                ),
                validation=BaselineDataset(
                    role="VALIDATION",
                    fold_id=dataset.fold_id,
                    dataset_id=f"{dataset.dataset_id}:inner-validation-{validation_index - 1}",
                    features=validation_features,
                    labels=validation_labels,
                ),
            )
        )
    return tuple(folds)


def _date_slice(frame: pd.DataFrame, dates: set[object]) -> pd.DataFrame:
    normalized_dates = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
    return frame.loc[normalized_dates.isin(dates)].copy()
