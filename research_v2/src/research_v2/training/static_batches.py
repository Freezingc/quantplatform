"""Construction of date-isolated static training batches.

The dataset reader and :class:`~research_v2.datasets.preprocessing.FoldPreprocessor`
own point-in-time joins and train-only fitting respectively.  This module deliberately
does neither: it only turns an already assembled (and, where desired, preprocessed)
PIT frame into the keyed tensors consumed by ``TorchDateTrainingLoop``.

One batch is emitted for each signal date.  Cross-sections are padded to a common width
within this call and all padding is represented by ``stock_mask=False`` and an empty
security key.  Consequently a training update can never accidentally mix observations
from two dates, while the loss can still give every date equal weight.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date

import numpy as np
import pandas as pd
import torch

from research_v2.datasets.builder import LockedDatasetError
from research_v2.models.base import FeatureBatch
from research_v2.training.batches import TrainingBatch


def build_static_training_batches(
    frame: pd.DataFrame,
    *,
    factor_order: tuple[str, ...],
    minimum_valid_stocks: int = 30,
    pad_to: int | None = None,
) -> Mapping[date, TrainingBatch]:
    """Build one :class:`TrainingBatch` per signal date from a static PIT frame.

    ``frame`` must already be scoped to one train or validation segment.  No fitting,
    winsorisation, imputation, or database access occurs here; numeric feature values
    are merely converted to ``float32`` and non-finite values are represented as zero
    plus a ``feature_missing`` indicator.  This lets callers pass the output of a
    train-fitted :class:`FoldPreprocessor` without leaking validation information.

    Rows with an invalid/non-finite target are excluded.  Dates with fewer than
    ``minimum_valid_stocks`` remaining observations are excluded as well, matching the
    cross-sectional loss/evaluation minimum.  A frame containing the locked 2026
    segment (or any future date) is rejected before any tensors are constructed.
    """

    _validate_arguments(frame, factor_order, minimum_valid_stocks, pad_to)
    required = {"signal_date", "ts_code", "raw_return_5d", "label_valid", *factor_order}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"static PIT frame is missing columns: {missing}")

    if frame.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("static PIT frame contains duplicate signal-date/security keys")

    columns = ["signal_date", "ts_code", "raw_return_5d", "label_valid", *factor_order]
    normalized = frame.loc[:, columns].copy()
    normalized["signal_date"] = pd.to_datetime(normalized["signal_date"], errors="raise").dt.date
    if normalized["signal_date"].isna().any():
        raise ValueError("signal_date cannot be null")
    if any(signal_date.year >= 2026 for signal_date in normalized["signal_date"]):
        raise LockedDatasetError("static training batches cannot consume locked/future labels")
    codes = normalized["ts_code"].astype(str)
    if normalized["ts_code"].isna().any() or codes.str.strip().eq("").any():
        raise ValueError("ts_code cannot be null or empty")

    # Keep valid labels only.  ``label_valid`` is intentionally interpreted as a
    # boolean flag, while target conversion rejects values that are not numeric.
    target = pd.to_numeric(normalized["raw_return_5d"], errors="coerce")
    valid_labels = normalized["label_valid"].fillna(False).astype(bool) & np.isfinite(target)
    normalized = normalized.loc[valid_labels].copy()
    normalized["_target"] = target.loc[normalized.index].astype(float)
    if normalized.empty:
        raise ValueError("static PIT frame has no valid target observations")

    grouped: list[tuple[date, pd.DataFrame]] = []
    for signal_date, group in normalized.groupby("signal_date", sort=True):
        ordered = group.sort_values("ts_code", kind="mergesort").reset_index(drop=True)
        if len(ordered) >= minimum_valid_stocks:
            grouped.append((signal_date, ordered))
    if not grouped:
        raise ValueError("no signal date meets minimum_valid_stocks")

    width = max(len(group) for _, group in grouped)
    if pad_to is not None:
        width = max(width, pad_to)
    batches: dict[date, TrainingBatch] = {}
    for signal_date, group in grouped:
        batches[signal_date] = _build_one_date_batch(
            signal_date,
            group,
            factor_order=factor_order,
            width=width,
        )
    return batches


def _build_one_date_batch(
    signal_date: date,
    group: pd.DataFrame,
    *,
    factor_order: tuple[str, ...],
    width: int,
) -> TrainingBatch:
    count = len(group)
    raw_features = group.loc[:, list(factor_order)].apply(pd.to_numeric, errors="coerce")
    raw_values = raw_features.replace([np.inf, -np.inf], np.nan).to_numpy(dtype=np.float32)
    feature_missing = np.isnan(raw_values)
    values = np.nan_to_num(raw_values, nan=0.0, posinf=0.0, neginf=0.0)

    features = np.zeros((1, width, len(factor_order)), dtype=np.float32)
    missing = np.zeros((1, width, len(factor_order)), dtype=bool)
    stock_mask = np.zeros((1, width), dtype=bool)
    target_raw = np.full((1, width), np.nan, dtype=np.float32)
    target_rank = np.full((1, width), np.nan, dtype=np.float32)
    features[0, :count] = values
    missing[0, :count] = feature_missing
    stock_mask[0, :count] = True
    target_values = group["_target"].to_numpy(dtype=np.float32)
    target_raw[0, :count] = target_values
    # Percentile ranks are computed within each cross-section only.  Keeping ties at
    # their average rank is deterministic and agrees with evaluation's Spearman ranks.
    target_rank[0, :count] = (
        pd.Series(target_values).rank(method="average", pct=True).to_numpy(dtype=np.float32)
    )
    codes = tuple(group["ts_code"].astype(str).tolist()) + ("",) * (width - count)
    inputs = FeatureBatch(
        signal_dates=(signal_date,),
        ts_codes=(codes,),
        features=torch.from_numpy(features),
        feature_missing=torch.from_numpy(missing),
        stock_mask=torch.from_numpy(stock_mask),
    )
    return TrainingBatch(
        inputs=inputs,
        target_raw=torch.from_numpy(target_raw),
        target_rank=torch.from_numpy(target_rank),
    )


def _validate_arguments(
    frame: pd.DataFrame,
    factor_order: tuple[str, ...],
    minimum_valid_stocks: int,
    pad_to: int | None,
) -> None:
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not factor_order or len(set(factor_order)) != len(factor_order):
        raise ValueError("factor_order must be non-empty and unique")
    reserved = {"signal_date", "ts_code", "raw_return_5d", "label_valid"}
    if reserved.intersection(factor_order):
        raise ValueError("factor_order contains a reserved PIT column")
    if minimum_valid_stocks < 2:
        raise ValueError("minimum_valid_stocks must be at least two")
    if pad_to is not None and pad_to < minimum_valid_stocks:
        raise ValueError("pad_to must be at least minimum_valid_stocks")
