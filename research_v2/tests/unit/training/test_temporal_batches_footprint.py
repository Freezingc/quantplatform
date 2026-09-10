"""Footprint and frozen-reference equivalence for the temporal window builder.

Two different jobs live here, and they fail in opposite directions:

``test_production_builder_matches_the_frozen_reference_bitwise`` is a CHARACTERISATION guard. It
passes before the optimisation and must keep passing after it, because the reference below is a
verbatim copy of the builder as it stood at 9e75e67 -- the version that produced every temporal
artifact so far. A faster builder that moved a single bit would silently start a third thread
lineage, and the lineage gap is measured at a third of the G4 gate, so "same numbers" is not a
nicety here.

``test_builder_holds_no_python_object_per_cell`` is the RED test for the property being added. The
frozen reference materialises one Python dict of ``len(factor_order)`` floats per input row and
calls ``pd.to_numeric`` once per cell; measured on a 1,200-row frame that is ~2.7 kB of traced heap
per row against ~0.4 MB of output tensors, i.e. roughly 7x the volume it produces. On the real F1
fold (2,296,573 rows) the same shape is what puts a worker at 16.8 GiB and caps M2 at 19 of them.
The bound below is 4x the builder's own output volume: it fails at ~7x for the reference and is met
by a builder that keeps the factors in one array.
"""

from __future__ import annotations

import gc
import tracemalloc
from collections.abc import Mapping, Sequence
from datetime import date
from typing import cast

import numpy as np
import pandas as pd
import torch

from research_v2.training.batches import TemporalTrainingBatch
from research_v2.training.temporal_batches import build_temporal_training_batches


def _reference_build(
    frame: pd.DataFrame,
    *,
    factor_order: tuple[str, ...],
    history_length: int,
    minimum_valid_history_length: int,
    minimum_valid_stocks: int,
    pad_to: int | None = None,
) -> Mapping[date, TemporalTrainingBatch]:
    """Verbatim copy of ``build_temporal_training_batches`` as of 9e75e67. DO NOT EDIT.

    Argument validation is omitted because every caller here passes valid arguments; the algorithm
    that decides WHICH rows become WHICH cells is copied whole.
    """
    required = {"signal_date", "ts_code", "raw_return_5d", "label_valid", *factor_order}
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"temporal PIT frame is missing columns: {missing}")
    if frame.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("temporal PIT frame contains duplicate signal-date/security keys")

    columns = ["signal_date", "ts_code", "raw_return_5d", "label_valid", *factor_order]
    normalized = frame.loc[:, columns].copy()
    normalized["signal_date"] = pd.to_datetime(normalized["signal_date"], errors="raise").dt.date
    codes = normalized["ts_code"].astype(str)
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

        code_tuple = tuple(code for code, _, _ in eligible) + ("",) * (width - count)
        from research_v2.models.base import TemporalFeatureBatch

        batches[signal_date] = TemporalTrainingBatch(
            inputs=TemporalFeatureBatch(
                signal_dates=(signal_date,),
                ts_codes=(code_tuple,),
                features=torch.from_numpy(features),
                feature_missing=torch.from_numpy(missing),
                stock_mask=torch.from_numpy(stock_mask),
                history_mask=torch.from_numpy(history_mask),
            ),
            target_raw=torch.from_numpy(target_raw),
            target_rank=torch.from_numpy(target_rank),
        )
    return batches


def _panel(n_codes: int, n_dates: int, factor_order: Sequence[str], seed: int) -> pd.DataFrame:
    """A panel that exercises every branch the window builder has.

    Uneven history length per code, invalid labels, non-finite factors of both kinds (NaN and inf,
    which the builder treats alike but reaches by different routes), dates too thin to qualify, and
    tied targets so the average-rank path is not merely exercised but exercised with ties.
    """
    rng = np.random.default_rng(seed)
    days = [date(2024, 1, 1).replace(month=1 + index // 28, day=1 + index % 28)
            for index in range(n_dates)]
    rows: list[dict[str, object]] = []
    for code_index in range(n_codes):
        code = f"{code_index:06d}.SZ"
        # Codes start at staggered offsets, so history length differs per code and per date.
        first = code_index % 7
        for day_index in range(first, n_dates):
            row: dict[str, object] = {
                "signal_date": days[day_index],
                "ts_code": code,
                "raw_return_5d": float(rng.standard_normal()) / 100.0,
                "label_valid": bool(rng.random() > 0.05),
            }
            for factor_name in factor_order:
                row[factor_name] = float(rng.standard_normal())
            rows.append(row)
    frame = pd.DataFrame(rows)
    # Ties in the target, so average ranking has something to average.
    frame.loc[frame.index[: max(2, len(frame) // 20)], "raw_return_5d"] = 0.0125
    # Non-finite factors: NaN and +-inf must both land in feature_missing with a 0.0 feature.
    hit_nan = rng.choice(len(frame), size=max(1, len(frame) // 37), replace=False)
    frame.loc[frame.index[hit_nan], factor_order[0]] = np.nan
    hit_inf = rng.choice(len(frame), size=max(1, len(frame) // 53), replace=False)
    frame.loc[frame.index[hit_inf], factor_order[-1]] = np.inf
    # Some targets non-finite, which is a different route to exclusion than label_valid=False.
    hit_target = rng.choice(len(frame), size=max(1, len(frame) // 41), replace=False)
    frame.loc[frame.index[hit_target], "raw_return_5d"] = np.nan
    return frame


_CASES = (
    {"history_length": 20, "minimum_valid_history_length": 15, "minimum_valid_stocks": 30,
     "pad_to": None},
    {"history_length": 5, "minimum_valid_history_length": 1, "minimum_valid_stocks": 2,
     "pad_to": None},
    {"history_length": 5, "minimum_valid_history_length": 5, "minimum_valid_stocks": 2,
     "pad_to": 64},
    {"history_length": 31, "minimum_valid_history_length": 8, "minimum_valid_stocks": 5,
     "pad_to": None},
)


def _bitwise(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Compare raw bytes: NaN payloads and signed zeros included, which ``torch.equal`` skips."""
    return left.numpy().tobytes() == right.numpy().tobytes()


def test_production_builder_matches_the_frozen_reference_bitwise() -> None:
    factor_order = tuple(f"f{index:02d}" for index in range(4))
    frame = _panel(38, 34, factor_order, seed=20260911)

    for case in _CASES:
        produced = build_temporal_training_batches(
            frame, factor_order=factor_order, **cast(dict, case)
        )
        expected = _reference_build(frame, factor_order=factor_order, **cast(dict, case))

        assert list(produced) == list(expected), case
        for day, want in expected.items():
            got = produced[day]
            assert got.inputs.signal_dates == want.inputs.signal_dates, (case, day)
            assert got.inputs.ts_codes == want.inputs.ts_codes, (case, day)
            for name in ("features", "feature_missing", "stock_mask", "history_mask"):
                assert _bitwise(getattr(got.inputs, name), getattr(want.inputs, name)), (
                    case, day, name
                )
            for name in ("target_raw", "target_rank"):
                assert _bitwise(getattr(got, name), getattr(want, name)), (case, day, name)


def test_builder_holds_no_python_object_per_cell() -> None:
    """The builder's traced peak must stay within a small multiple of what it produces.

    This is the property the frozen reference does not have: it holds one Python dict per row while
    building, so its peak is roughly 7x its own output on the frame below and scales with the fold
    rather than with the batches. Everything the builder returns has to exist, so the honest bound
    is expressed against the output volume instead of an absolute byte count.
    """
    factor_order = tuple(f"f{index:02d}" for index in range(4))
    frame = _panel(40, 30, factor_order, seed=99)
    gc.collect()

    tracemalloc.start()
    try:
        batches = build_temporal_training_batches(
            frame,
            factor_order=factor_order,
            history_length=20,
            minimum_valid_history_length=15,
            minimum_valid_stocks=10,
        )
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    produced = sum(
        tensor.numpy().nbytes
        for batch in batches.values()
        for tensor in (
            batch.inputs.features, batch.inputs.feature_missing, batch.inputs.stock_mask,
            batch.inputs.history_mask, batch.target_raw, batch.target_rank,
        )
    )
    assert batches, "the frame must produce batches for the bound to mean anything"
    assert peak < 4 * produced, (
        f"traced peak {peak:,} bytes is {peak / produced:.1f}x the {produced:,} bytes of batches "
        "produced; the builder is still holding per-row Python objects"
    )
