from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from research_v2.datasets.builder import LockedDatasetError
from research_v2.training.temporal_batches import build_temporal_training_batches

_DAYS = [date(2024, 1, d) for d in (2, 3, 4, 5, 8, 9, 10)]


def _rows(days, codes, *, value=1.0, target=0.01, valid=True):
    rows = []
    for day in days:
        for code in codes:
            rows.append(
                {
                    "signal_date": day,
                    "ts_code": code,
                    "f1": value,
                    "raw_return_5d": target,
                    "label_valid": valid,
                }
            )
    return rows


def test_temporal_batches_use_causal_left_aligned_history() -> None:
    frame = pd.DataFrame(_rows(_DAYS[:5], ["a", "b"]))
    batches = build_temporal_training_batches(
        frame,
        factor_order=("f1",),
        history_length=3,
        minimum_valid_history_length=3,
        minimum_valid_stocks=2,
    )

    assert set(batches) == {_DAYS[2], _DAYS[3], _DAYS[4]}
    batch = batches[_DAYS[4]]
    assert batch.inputs.history_mask[0, 0].tolist() == [True, True, True]
    # Left-aligned oldest-first history values, marked by per-day feature values.
    marked = pd.DataFrame(
        [
            {
                "signal_date": day,
                "ts_code": code,
                "f1": float(index),
                "raw_return_5d": 0.01,
                "label_valid": True,
            }
            for index, day in enumerate(_DAYS[:5])
            for code in ("a", "b")
        ]
    )
    rebuilt = build_temporal_training_batches(
        marked,
        factor_order=("f1",),
        history_length=3,
        minimum_valid_history_length=3,
        minimum_valid_stocks=2,
    )
    window = rebuilt[_DAYS[4]].inputs.features[0, 0, :, 0].tolist()
    assert window == [2.0, 3.0, 4.0]

    future_perturbed = pd.concat(
        [frame, pd.DataFrame(_rows([_DAYS[5], _DAYS[6]], ["a", "b"], value=999.0))],
        ignore_index=True,
    )
    perturbed_batches = build_temporal_training_batches(
        future_perturbed,
        factor_order=("f1",),
        history_length=3,
        minimum_valid_history_length=3,
        minimum_valid_stocks=2,
    )
    assert torch_equal_batch(batches[_DAYS[4]], perturbed_batches[_DAYS[4]])


def torch_equal_batch(left, right) -> bool:
    import torch

    return (
        torch.equal(left.inputs.features, right.inputs.features)
        and torch.equal(left.inputs.history_mask, right.inputs.history_mask)
        and torch.equal(left.target_raw, right.target_raw)
    )


def test_temporal_batches_enforce_minimum_history_length() -> None:
    frame = pd.DataFrame(_rows(_DAYS[:5], ["a", "c"]) + _rows(_DAYS[3:5], ["b"]))
    batches = build_temporal_training_batches(
        frame,
        factor_order=("f1",),
        history_length=5,
        minimum_valid_history_length=3,
        minimum_valid_stocks=2,
    )

    batch = batches[_DAYS[4]]
    assert batch.inputs.ts_codes[0] == ("a", "c")
    assert batch.inputs.stock_mask[0].tolist() == [True, True]


def test_temporal_batches_skip_missing_dates_and_keep_observed_order() -> None:
    frame = pd.DataFrame(
        _rows([_DAYS[0], _DAYS[2], _DAYS[4]], ["a", "b"], value=1.0)
    )
    batches = build_temporal_training_batches(
        frame,
        factor_order=("f1",),
        history_length=5,
        minimum_valid_history_length=3,
        minimum_valid_stocks=2,
    )
    batch = batches[_DAYS[4]]
    assert batch.inputs.history_mask[0, 0].tolist() == [True, True, True, False, False]


def test_temporal_batches_reject_locked_year() -> None:
    frame = pd.DataFrame(_rows([date(2026, 1, 2)], ["a", "b"]))
    with pytest.raises(LockedDatasetError):
        build_temporal_training_batches(
            frame,
            factor_order=("f1",),
            history_length=3,
            minimum_valid_history_length=1,
            minimum_valid_stocks=2,
        )


def test_temporal_batches_exclude_invalid_targets_and_small_dates() -> None:
    frame = pd.DataFrame(
        _rows(_DAYS[:4], ["a", "b"])
        + [
            {
                "signal_date": _DAYS[3],
                "ts_code": "c",
                "f1": 1.0,
                "raw_return_5d": 0.02,
                "label_valid": False,
            }
        ]
    )
    batches = build_temporal_training_batches(
        frame,
        factor_order=("f1",),
        history_length=4,
        minimum_valid_history_length=4,
        minimum_valid_stocks=2,
    )
    batch = batches[_DAYS[3]]
    assert batch.inputs.ts_codes[0] == ("a", "b")
    assert not np.isnan(batch.target_raw.numpy()).any()


def test_temporal_batches_reject_invalid_configuration() -> None:
    frame = pd.DataFrame(_rows(_DAYS[:4], ["a"]))
    with pytest.raises(ValueError):
        build_temporal_training_batches(
            frame,
            factor_order=("f1",),
            history_length=3,
            minimum_valid_history_length=4,
            minimum_valid_stocks=1,
        )
    with pytest.raises(ValueError):
        build_temporal_training_batches(
            frame,
            factor_order=(),
            history_length=3,
            minimum_valid_history_length=1,
            minimum_valid_stocks=1,
        )
    with pytest.raises(ValueError):
        build_temporal_training_batches(
            frame,
            factor_order=("f1",),
            history_length=3,
            minimum_valid_history_length=3,
            minimum_valid_stocks=1,
        )
