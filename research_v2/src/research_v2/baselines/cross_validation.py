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
    fold_count: int = 2,
    purge_trading_days: int = 6,
) -> tuple[InnerBaselineFold, ...]:
    """Create two purged expanding folds from four outer-TRAIN calendar years.

    The first two years form the warm-up period. Each following year is evaluated
    once, so every returned validation date is later than every date used to fit it.
    Both inner-TRAIN and inner-VALIDATION segments remove their final six trading
    dates, matching the frozen annual split contract for overlapping five-day labels.
    The v2 outer folds use a four-year train window, so there are exactly two inner
    folds (validation on the third and fourth training years).
    """
    if dataset.role != "TRAIN":
        raise PermissionError("inner folds can only be created from TRAIN data")
    if fold_count != 2:
        raise ValueError("the baseline protocol requires exactly two inner folds")
    if purge_trading_days < 0:
        raise ValueError("purge_trading_days cannot be negative")
    dates = sorted(pd.to_datetime(dataset.features["signal_date"], errors="raise").dt.date.unique())
    years = tuple(sorted({day.year for day in dates}))
    if len(years) != 4:
        raise ValueError("two expanding inner folds require exactly four training years")
    dates_by_year = {year: tuple(day for day in dates if day.year == year) for year in years}
    for year, year_dates in dates_by_year.items():
        if len(year_dates) <= purge_trading_days:
            raise ValueError(
                f"inner folds require more than {purge_trading_days} dates in training year {year}"
            )
    folds: list[InnerBaselineFold] = []
    for validation_index in (2, 3):
        train_dates = _purge_tail(
            tuple(day for year in years[:validation_index] for day in dates_by_year[year]),
            purge_trading_days,
        )
        validation_dates = _purge_tail(
            dates_by_year[years[validation_index]],
            purge_trading_days,
        )
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


def _purge_tail(dates: tuple[object, ...], purge_trading_days: int) -> set[object]:
    if len(dates) <= purge_trading_days:
        raise ValueError(f"insufficient dates to purge {purge_trading_days} trading days")
    return set(dates[:-purge_trading_days]) if purge_trading_days else set(dates)


def _date_slice(frame: pd.DataFrame, dates: set[object]) -> pd.DataFrame:
    normalized_dates = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
    return frame.loc[normalized_dates.isin(dates)].copy()
