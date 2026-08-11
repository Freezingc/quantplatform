from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
import torch

from research_v2.datasets.builder import LockedDatasetError
from research_v2.training.static_batches import build_static_training_batches

FACTORS = ("f1", "f2", "f3", "f4", "f5", "f6")


def _frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for signal_date, count in ((date(2025, 1, 2), 3), (date(2025, 1, 3), 3)):
        for index in range(count):
            rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": f"S{index:03d}",
                    "raw_return_5d": float(index - 1),
                    "label_valid": True,
                    "f1": float(index),
                    "f2": float(index + 1),
                    "f3": float(index + 2),
                    "f4": float(index + 3),
                    "f5": float(index + 4),
                    "f6": float(index + 5),
                }
            )
    rows[1]["f2"] = float("nan")
    rows[-1]["label_valid"] = False
    return pd.DataFrame(rows)


def test_build_static_batches_is_date_isolated_and_padded() -> None:
    batches = build_static_training_batches(
        _frame(),
        factor_order=FACTORS,
        minimum_valid_stocks=2,
    )

    assert tuple(batches) == (date(2025, 1, 2), date(2025, 1, 3))
    first = batches[date(2025, 1, 2)]
    second = batches[date(2025, 1, 3)]
    assert first.inputs.features.shape == (1, 3, 6)
    assert second.inputs.features.shape == (1, 3, 6)
    assert first.inputs.ts_codes == (("S000", "S001", "S002"),)
    assert second.inputs.ts_codes == (("S000", "S001", ""),)
    assert second.inputs.stock_mask.tolist() == [[True, True, False]]
    assert bool(first.inputs.feature_missing[0, 1, 1])
    assert first.inputs.features[0, 1, 1].item() == 0.0
    assert torch.isfinite(first.target_raw[first.inputs.stock_mask]).all()
    assert torch.isnan(second.target_raw[0, 2])
    assert torch.isnan(second.target_rank[0, 2])


def test_build_static_batches_rejects_duplicate_keys() -> None:
    frame = _frame()
    frame = pd.concat([frame, frame.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="duplicate"):
        build_static_training_batches(frame, factor_order=FACTORS, minimum_valid_stocks=2)


def test_build_static_batches_rejects_locked_year_before_tensor_access() -> None:
    frame = _frame()
    frame.loc[0, "signal_date"] = date(2026, 1, 2)
    with pytest.raises(LockedDatasetError, match="locked"):
        build_static_training_batches(frame, factor_order=FACTORS, minimum_valid_stocks=2)


def test_build_static_batches_requires_enough_cross_section() -> None:
    with pytest.raises(ValueError, match="no signal date"):
        build_static_training_batches(_frame(), factor_order=FACTORS, minimum_valid_stocks=4)
