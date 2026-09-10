"""Construction of date-isolated temporal training batches with causal history.

Mirrors :mod:`research_v2.training.static_batches`: the caller owns point-in-time
joins and train-only fitting; this module only turns an assembled PIT frame into
keyed ``TemporalTrainingBatch`` tensors for ``TorchDateTrainingLoop``.

For every signal date ``T`` each security's history window is the most recent
``history_length`` of its own observed rows at or before ``T`` (oldest first,
left-aligned).  Securities with fewer than ``minimum_valid_history_length``
observed steps cannot take part in the loss and are excluded, matching the frozen
``minimum_valid_history_length`` contract in ``configs/models/temporal.yaml``.

The construction is array-based end to end, and the reason is measured rather than aesthetic. The
previous version held one Python ``dict`` of factor values per input row and called
``pd.to_numeric`` once per cell; on a 1,200-row frame that traced 5.0x its own output volume in
peak heap, and on the real F1 fold (2,296,573 rows) it put one worker at 16.8 GiB and 40.2 minutes
of one-off rebuild, which is what capped M2 at 19 workers. Holding the factors in a single
``(rows, factors)`` array and gathering each window by index removes both costs and leaves the
output bitwise unchanged: every value reaching a tensor is still a float64 taken from the same
source cell and cast to float32 exactly once, which is what the scalar assignment did.
``tests/unit/training/test_temporal_batches_footprint.py`` pins that equivalence against a frozen
copy of the previous algorithm, so a later speed-up cannot quietly open a new numerical lineage --
the measured lineage gap is already a third of the G4 gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

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

    panel = _Panel.from_frame(normalized, factor_order)
    if not panel.valid_row.any():
        raise ValueError("temporal PIT frame has no valid target observations")

    batches: dict[date, TemporalTrainingBatch] = {}
    for date_id, signal_date in enumerate(panel.dates):
        batch = _batch_for_date(
            panel,
            date_id,
            signal_date,
            factor_count=len(factor_order),
            history_length=history_length,
            minimum_valid_history_length=minimum_valid_history_length,
            minimum_valid_stocks=minimum_valid_stocks,
            pad_to=pad_to,
        )
        if batch is not None:
            batches[signal_date] = batch
    if not batches:
        raise ValueError("no signal date meets the temporal eligibility requirements")
    return batches


@dataclass(frozen=True, eq=False)
class _Panel:
    """The sorted frame as index arrays, so no window is assembled out of Python objects.

    ``row_of[code_id, date_id]`` is the row index of that security's observation on that date, or
    ``-1`` when it was not observed then.  That one integer matrix is what replaces the previous
    per-code list of per-row dicts, and it costs ``codes x dates`` integers instead of
    ``rows x factors`` Python floats.
    """

    codes: tuple[str, ...]
    dates: tuple[date, ...]
    code_start: np.ndarray
    row_of: np.ndarray
    factors: np.ndarray
    target: np.ndarray
    valid_row: np.ndarray

    @classmethod
    def from_frame(cls, normalized: pd.DataFrame, factor_order: tuple[str, ...]) -> _Panel:
        row_count = len(normalized)
        factors = np.empty((row_count, len(factor_order)), dtype=np.float64)
        for column_index, factor_name in enumerate(factor_order):
            factors[:, column_index] = pd.to_numeric(
                normalized[factor_name], errors="coerce"
            ).to_numpy(dtype=np.float64)
        # +inf and NaN take the same branch downstream (feature 0.0, missing True), so they are
        # folded together here once instead of once per cell.
        factors[~np.isfinite(factors)] = np.nan

        target = pd.to_numeric(normalized["raw_return_5d"], errors="coerce").to_numpy(
            dtype=np.float64
        )
        label_valid = normalized["label_valid"].fillna(False).astype(bool).to_numpy()
        valid_row = label_valid & np.isfinite(target)

        unique_codes, first_rows = np.unique(normalized["ts_code"].to_numpy(), return_index=True)
        code_start = np.asarray(first_rows, dtype=np.int64)
        # The frame is sorted by (ts_code, signal_date), so every code owns exactly one contiguous
        # run of rows and that run's offset is enough to recover the security from a row index.
        run_lengths = np.append(code_start[1:], row_count) - code_start
        code_ids = np.repeat(np.arange(len(unique_codes), dtype=np.int64), run_lengths)

        appearances, first_seen = pd.factorize(normalized["signal_date"], sort=False)
        unique_dates = np.asarray(first_seen, dtype=object)
        ascending = np.argsort(unique_dates, kind="stable")
        date_ids = np.empty(len(unique_dates), dtype=np.int64)
        date_ids[ascending] = np.arange(len(unique_dates), dtype=np.int64)

        row_of = np.full((len(unique_codes), len(unique_dates)), -1, dtype=np.int64)
        row_of[code_ids, date_ids[np.asarray(appearances, dtype=np.int64)]] = np.arange(
            row_count, dtype=np.int64
        )
        return cls(
            codes=tuple(str(code) for code in np.asarray(unique_codes).tolist()),
            dates=tuple(unique_dates[ascending].tolist()),
            code_start=code_start,
            row_of=row_of,
            factors=factors,
            target=target,
            valid_row=valid_row,
        )


def _batch_for_date(
    panel: _Panel,
    date_id: int,
    signal_date: date,
    *,
    factor_count: int,
    history_length: int,
    minimum_valid_history_length: int,
    minimum_valid_stocks: int,
    pad_to: int | None,
) -> TemporalTrainingBatch | None:
    """The batch for one signal date, or None when too few securities are eligible.

    None rather than an empty batch because the caller must not insert a key for a date the previous
    implementation skipped: which dates are present is part of what has to stay identical.
    """
    position = panel.row_of[:, date_id]
    observed = position >= 0
    eligible_mask = observed & panel.valid_row[np.where(observed, position, 0)]
    candidate = np.flatnonzero(eligible_mask)
    # Observations of a security at or before the signal date are exactly the rows from the start of
    # its run up to and including this one, because the frame is sorted by (ts_code, signal_date).
    observed_count = position[candidate] - panel.code_start[candidate] + 1
    length = np.minimum(history_length, observed_count)
    kept = length >= minimum_valid_history_length
    candidate = candidate[kept]
    if len(candidate) < minimum_valid_stocks:
        return None
    row = position[candidate]
    length = length[kept]

    count = int(len(candidate))
    width = max(count, pad_to) if pad_to is not None else count

    steps = np.arange(history_length, dtype=np.int64)
    live = steps[None, :] < length[:, None]
    # Oldest kept observation first, so step s is row - length + 1 + s. A lane past a security's own
    # history points at row 0 and is masked out below; it must not contribute a value, because the
    # previous builder never wrote to those cells and they stayed at their initial 0.0/False.
    source = (row - length + 1)[:, None] + steps[None, :]
    gathered = panel.factors[np.where(live, source, 0)]
    usable = np.isfinite(gathered) & live[:, :, None]

    features = np.zeros((1, width, history_length, factor_count), dtype=np.float32)
    features[0, :count] = np.where(usable, gathered, 0.0)
    feature_missing = np.zeros((1, width, history_length, factor_count), dtype=bool)
    feature_missing[0, :count] = live[:, :, None] & ~np.isfinite(gathered)
    history_mask = np.zeros((1, width, history_length), dtype=bool)
    history_mask[0, :count] = live
    stock_mask = np.zeros((1, width), dtype=bool)
    stock_mask[0, :count] = True

    target_raw = np.full((1, width), np.nan, dtype=np.float32)
    target_rank = np.full((1, width), np.nan, dtype=np.float32)
    as_float32 = np.asarray(panel.target[row], dtype=np.float32)
    target_raw[0, :count] = as_float32
    target_rank[0, :count] = pd.Series(as_float32).rank(
        method="average", pct=True
    ).to_numpy(dtype=np.float32)

    codes = tuple(panel.codes[int(code_id)] for code_id in candidate) + ("",) * (width - count)
    inputs = TemporalFeatureBatch(
        signal_dates=(signal_date,),
        ts_codes=(codes,),
        features=torch.from_numpy(features),
        feature_missing=torch.from_numpy(feature_missing),
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
