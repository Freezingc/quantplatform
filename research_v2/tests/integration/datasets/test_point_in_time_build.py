"""Point-in-time dataset build integration evidence (D9.4)."""

from __future__ import annotations

import os
from datetime import date

import pandas as pd
import pytest

from research_v2.datasets.builder import (
    DatasetBuildRequest,
    FactorInput,
    LockedDatasetError,
    build_point_in_time_dataset,
)
from research_v2.datasets.manifest import build_dataset_manifest

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("QRV2_INTEGRATION") != "1",
        reason="set QRV2_INTEGRATION=1 to run dataset integration tests",
    ),
]

_DAYS = [day.date() for day in pd.bdate_range("2024-01-02", periods=20)]
_CODES = tuple(f"s{index:02d}" for index in range(40))
_SEGMENT_END = date(2024, 3, 31)
_FACTOR_MEMBERS = (FactorInput(factor_id="alpha_demo", factor_version=1, feature_name="f01"),)


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        [{"signal_date": day, "ts_code": code} for day in _DAYS for code in _CODES]
    )


def _factors() -> pd.DataFrame:
    rows = []
    for day_index, day in enumerate(_DAYS):
        for code_index, code in enumerate(_CODES):
            rows.append(
                {
                    "signal_date": day,
                    "ts_code": code,
                    "factor_id": "alpha_demo",
                    "factor_version": 1,
                    "value": float(day_index * 100 + code_index),
                    "available_date": day,
                }
            )
    return pd.DataFrame(rows)


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "signal_date": day,
                "ts_code": code,
                "raw_return_5d": 0.01,
                "label_valid": True,
                "exit_date": day,
            }
            for day in _DAYS
            for code in _CODES
        ]
    )


def _build_train() -> pd.DataFrame:
    return build_point_in_time_dataset(
        _universe(),
        _factors(),
        _labels(),
        factor_members=_FACTOR_MEMBERS,
        segment_end=_SEGMENT_END,
        request=DatasetBuildRequest(fold_id="F01", split_role="TRAIN"),
    )


def test_synthetic_train_build_is_shaped_sorted_and_deterministic() -> None:
    dataset = _build_train()

    assert dataset.shape == (len(_DAYS) * len(_CODES), 6)
    assert dataset.columns.tolist() == [
        "signal_date",
        "ts_code",
        "f01",
        "raw_return_5d",
        "label_valid",
        "exit_date",
    ]
    sorted_view = dataset.sort_values(
        ["signal_date", "ts_code"], kind="mergesort"
    ).reset_index(drop=True)
    pd.testing.assert_frame_equal(dataset.reset_index(drop=True), sorted_view)

    manifest = build_dataset_manifest(
        dataset,
        business_keys=("signal_date", "ts_code"),
        metadata={"fold_id": "F01", "split_role": "TRAIN"},
    )
    rebuilt = _build_train()
    rebuilt_manifest = build_dataset_manifest(
        rebuilt,
        business_keys=("signal_date", "ts_code"),
        metadata={"fold_id": "F01", "split_role": "TRAIN"},
    )
    assert manifest.dataset_id == rebuilt_manifest.dataset_id


def test_locked_year_inputs_are_rejected_on_the_ordinary_path() -> None:
    future_universe = pd.DataFrame(
        [{"signal_date": date(2026, 1, 6), "ts_code": "s00"}]
    )
    with pytest.raises(LockedDatasetError):
        build_point_in_time_dataset(
            future_universe,
            _factors(),
            _labels(),
            factor_members=_FACTOR_MEMBERS,
            segment_end=date(2026, 3, 31),
            request=DatasetBuildRequest(fold_id="F01", split_role="TRAIN"),
        )


def test_final_test_requires_locked_access() -> None:
    with pytest.raises(LockedDatasetError):
        DatasetBuildRequest(fold_id="FINAL", split_role="TEST")
    with pytest.raises(LockedDatasetError):
        build_point_in_time_dataset(
            _universe(),
            _factors(),
            _labels(),
            factor_members=_FACTOR_MEMBERS,
            segment_end=_SEGMENT_END,
            request=DatasetBuildRequest(fold_id="FINAL", split_role="TEST", locked_access=None),
        )


def test_point_in_time_availability_is_enforced() -> None:
    leaked_factors = _factors()
    leaked_factors["available_date"] = leaked_factors["signal_date"] + pd.Timedelta(days=1)
    with pytest.raises(ValueError, match="not available by signal date"):
        build_point_in_time_dataset(
            _universe(),
            leaked_factors,
            _labels(),
            factor_members=_FACTOR_MEMBERS,
            segment_end=_SEGMENT_END,
            request=DatasetBuildRequest(fold_id="F01", split_role="TRAIN"),
        )


def test_labels_exiting_after_segment_end_are_rejected() -> None:
    late_labels = _labels()
    late_labels.loc[0, "exit_date"] = date(2024, 4, 30)
    with pytest.raises(ValueError, match="exit_date after the segment end"):
        build_point_in_time_dataset(
            _universe(),
            _factors(),
            late_labels,
            factor_members=_FACTOR_MEMBERS,
            segment_end=_SEGMENT_END,
            request=DatasetBuildRequest(fold_id="F01", split_role="TRAIN"),
        )
