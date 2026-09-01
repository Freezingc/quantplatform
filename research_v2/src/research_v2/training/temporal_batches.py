"""Construction of date-isolated temporal training batches with causal history.

Mirrors :mod:`research_v2.training.static_batches`: the caller owns point-in-time
joins and train-only fitting; this module only turns an assembled PIT frame into
keyed ``TemporalTrainingBatch`` tensors for ``TorchDateTrainingLoop``.

For every signal date ``T`` each security's history window is the most recent
``history_length`` of its own observed rows at or before ``T`` (oldest first,
left-aligned).  Securities with fewer than ``minimum_valid_history_length``
observed steps cannot take part in the loss and are excluded, matching the frozen
``minimum_valid_history_length`` contract in ``configs/models/temporal.yaml``.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import cast

import numpy as np
import pandas as pd
import torch

from research_v2.datasets.builder import LockedDatasetError
from research_v2.models.base import TemporalFeatureBatch
from research_v2.training.batches import TemporalTrainingBatch


def build_temporal_training_batches(
    frame: pd.DataFrame,
    *,
    factor_order: tuple[str, ...],
    history_length: int = 20,
    minimum_valid_history_length: int = 15,
    minimum_valid_stocks: int = 30,
    pad_to: int | None = None,
) -> Mapping[date, TemporalTrainingBatch]:
    """Build one :class:`TemporalTrainingBatch` per signal date from a PIT frame.

    Rows with an invalid/non-finite target are excluded before any window is
    assembled.  Dates with fewer than ``minimum_valid_stocks`` eligible
    securities are excluded entirely.  Any 2026 or later signal date is rejected
    before tensor construction.
    """
    _validate_arguments(
        frame, factor_order, history_length, minimum_valid_history_length,
        minimum_valid_stocks, pad_to,
    )
    required = {"signal_date", "ts_code", "raw_return_5d", "label_valid", *factor_order}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"temporal PIT frame is missing columns: {missing}")
    if frame.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("temporal PIT frame contains duplicate signal-date/security keys")

    columns = ["signal_date", "ts_code", "raw_return_5d", "label_valid", *factor_order]
    normalized = frame.loc[:, columns].copy()
    normalized["signal_date"] = pd.to_datetime(normalized["signal_date"], errors="raise").dt.date
    if normalized["signal_date"].isna().any():
        raise ValueError("signal_date cannot be null")
    if any(signal_date.year >= 2026 for signal_date in normalized["signal_date"]):
        raise LockedDatasetError("temporal training batches cannot consume locked/future labels")
    codes = normalized["ts_code"].astype(str)
    if normalized["ts_code"].isna().any() or codes.str.strip().eq("").any():
        raise ValueError("ts_code cannot be null or empty")
    normalized["ts_code"] = codes

    normalized = normalized.sort_values(["ts_code", "signal_date"], kind="mergesort")

    target = pd.to_numeric(normalized["raw_return_5d"], errors="coerce")
    valid_labels = normalized["label_valid"].fillna(False).astype(bool) & np.isfinite(target)
    valid_rows = normalized.loc[valid_labels]
    valid_targets: dict[tuple[str, date], float] = {
        (str(code), day): float(value)
        for code, day, value in zip(
            valid_rows["ts_code"], valid_rows["signal_date"], target.loc[valid_labels],
            strict=True,
        )
    }
    if not valid_targets:
        raise ValueError("temporal PIT frame has no valid target observations")

    history_by_code: dict[str, list[tuple[date, dict[str, float]]]] = {}
    for row in normalized.itertuples(index=False):
        values = {}
        for factor_name in factor_order:
            value = pd.to_numeric(getattr(row, factor_name), errors="coerce")
            values[factor_name] = float(value) if pd.notna(value) else float("nan")
        history_by_code.setdefault(str(row.ts_code), []).append(
            (cast(date, row.signal_date), values)
        )

    all_dates = sorted(normalized["signal_date"].unique())
    batches: dict[date, TemporalTrainingBatch] = {}
    for signal_date in all_dates:
        eligible: list[tuple[str, list[tuple[date, dict[str, float]]], float]] = []
        for code, observations in sorted(history_by_code.items()):
            current_target = valid_targets.get((code, signal_date))
            if current_target is None:
                continue
            window: list[tuple[date, dict[str, float]]] = []
            for observation_date, values in observations:
                if observation_date > signal_date:
                    break
                window.append((observation_date, values))
            history = window[-history_length:]
            if len(history) < minimum_valid_history_length:
                continue
            eligible.append((code, history, current_target))
        if len(eligible) < minimum_valid_stocks:
            continue
        batches[signal_date] = _build_one_date_batch(
            signal_date,
            eligible,
            factor_order=factor_order,
            history_length=history_length,
            pad_to=pad_to,
        )
    if not batches:
        raise ValueError("no signal date meets the temporal eligibility requirements")
    return batches


def _build_one_date_batch(
    signal_date: date,
    eligible: list[tuple[str, list[tuple[date, dict[str, float]]], float]],
    *,
    factor_order: tuple[str, ...],
    history_length: int,
    pad_to: int | None,
) -> TemporalTrainingBatch:
    count = len(eligible)
    width = max(count, pad_to) if pad_to is not None else count
    factor_count = len(factor_order)
    features = np.zeros((1, width, history_length, factor_count), dtype=np.float32)
    missing = np.zeros((1, width, history_length, factor_count), dtype=bool)
    history_mask = np.zeros((1, width, history_length), dtype=bool)
    stock_mask = np.zeros((1, width), dtype=bool)
    target_raw = np.full((1, width), np.nan, dtype=np.float32)
    target_rank = np.full((1, width), np.nan, dtype=np.float32)

    target_values = np.asarray([item[2] for item in eligible], dtype=np.float32)
    ranks = pd.Series(target_values).rank(method="average", pct=True).to_numpy(dtype=np.float32)
    for position, (_code, history, current_target) in enumerate(eligible):
        stock_mask[0, position] = True
        target_raw[0, position] = current_target
        target_rank[0, position] = ranks[position]
        for step, (_, values) in enumerate(history):
            history_mask[0, position, step] = True
            for factor_index, factor_name in enumerate(factor_order):
                raw_value = values[factor_name]
                if np.isfinite(raw_value):
                    features[0, position, step, factor_index] = raw_value
                else:
                    missing[0, position, step, factor_index] = True

    codes = tuple(code for code, _, _ in eligible) + ("",) * (width - count)
    inputs = TemporalFeatureBatch(
        signal_dates=(signal_date,),
        ts_codes=(codes,),
        features=torch.from_numpy(features),
        feature_missing=torch.from_numpy(missing),
        stock_mask=torch.from_numpy(stock_mask),
        history_mask=torch.from_numpy(history_mask),
    )
    return TemporalTrainingBatch(
        inputs=inputs,
        target_raw=torch.from_numpy(target_raw),
        target_rank=torch.from_numpy(target_rank),
    )


def _validate_arguments(
    frame: pd.DataFrame,
    factor_order: tuple[str, ...],
    history_length: int,
    minimum_valid_history_length: int,
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
    if history_length < 1:
        raise ValueError("history_length must be at least one")
    if not 1 <= minimum_valid_history_length <= history_length:
        raise ValueError("minimum_valid_history_length must be within [1, history_length]")
    if minimum_valid_stocks < 2:
        raise ValueError("minimum_valid_stocks must be at least two")
    if pad_to is not None and pad_to < minimum_valid_stocks:
        raise ValueError("pad_to must be at least minimum_valid_stocks")
