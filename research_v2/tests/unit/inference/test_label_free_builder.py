from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.datasets.builder import FactorInput
from research_v2.datasets.preprocessing import FoldPreprocessor
from research_v2.inference.builder import (
    build_label_free_inference_batch,
    build_label_free_inference_frame,
)


def test_inference_builder_anchors_universe_and_never_creates_labels() -> None:
    signal_date = date(2025, 1, 2)
    universe = pd.DataFrame(
        {
            "signal_date": [signal_date, signal_date],
            "ts_code": ["a", "b"],
            "is_member": [True, False],
        }
    )
    factors = pd.DataFrame(
        {
            "signal_date": [signal_date],
            "ts_code": ["a"],
            "factor_id": ["f1"],
            "factor_version": [1],
            "value": [1.0],
            "available_date": [signal_date],
        }
    )

    frame = build_label_free_inference_frame(
        universe,
        factors,
        factor_members=(FactorInput("f1", 1, "f1"),),
        signal_date=signal_date,
    )

    assert frame.columns.tolist() == ["signal_date", "ts_code", "f1"]
    assert frame["ts_code"].tolist() == ["a"]
    future = factors.assign(available_date=date(2025, 1, 3))
    with pytest.raises(ValueError, match="available"):
        build_label_free_inference_frame(
            universe,
            future,
            factor_members=(FactorInput("f1", 1, "f1"),),
            signal_date=signal_date,
        )


def test_label_free_builder_rejects_null_availability() -> None:
    signal_date = date(2025, 1, 2)
    universe = pd.DataFrame(
        {"signal_date": [signal_date, signal_date], "ts_code": ["a", "b"]}
    )
    factors = pd.DataFrame(
        {
            "signal_date": [signal_date, signal_date],
            "ts_code": ["a", "b"],
            "factor_id": ["f1", "f1"],
            "factor_version": [1, 1],
            "value": [1.0, 2.0],
            "available_date": [pd.NaT, signal_date],
        }
    )
    with pytest.raises(ValueError, match="available_date"):
        build_label_free_inference_frame(
            universe,
            factors,
            factor_members=(FactorInput("f1", 1, "f1"),),
            signal_date=signal_date,
        )


def test_label_free_builder_transforms_into_model_feature_batch() -> None:
    signal_date = date(2025, 1, 2)
    preprocessor = FoldPreprocessor(("f1",))
    preprocessor.fit(pd.DataFrame({"f1": [0.0, 1.0, 2.0]}), role="TRAIN")
    universe = pd.DataFrame({"signal_date": [signal_date], "ts_code": ["a"]})
    factors = pd.DataFrame(
        {
            "signal_date": [signal_date],
            "ts_code": ["a"],
            "factor_id": ["f1"],
            "factor_version": [1],
            "value": [1.0],
            "available_date": [signal_date],
        }
    )

    batch = build_label_free_inference_batch(
        universe,
        factors,
        factor_members=(FactorInput("f1", 1, "f1"),),
        signal_date=signal_date,
        preprocessor=preprocessor,
    )

    assert batch.features.shape == (1, 1, 1)
    assert batch.ts_codes == (("a",),)
