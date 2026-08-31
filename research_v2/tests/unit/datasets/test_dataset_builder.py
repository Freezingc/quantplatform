from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.datasets.builder import (
    DatasetBuildRequest,
    FactorInput,
    LockedDatasetError,
    LockedTestAccess,
    build_point_in_time_dataset,
)
from research_v2.factors.admission import FactorAdmissionContract, FactorAdmissionError
from research_v2.factors.factor_sets import (
    FactorSetStatus,
    create_factor_set,
    transition_factor_set,
)


@pytest.fixture(autouse=True)
def _locked_access_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QRV2_LOCKED_TEST_ACCESS_HMAC_KEY", "dataset-test-access-key")


def _locked_access() -> LockedTestAccess:
    return LockedTestAccess._issue_authorized(
        locked_year=2026,
        authorization_id="user-approved-final-evaluation",
        signing_key=b"dataset-test-access-key",
    )


def test_builder_anchors_universe_and_enforces_pit_availability() -> None:
    universe = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "ts_code": ["a", "b"],
        }
    )
    factors = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "ts_code": ["a", "b"],
            "factor_id": ["f1", "f1"],
            "factor_version": [1, 1],
            "value": [1.0, 2.0],
            "available_date": [date(2025, 1, 2), date(2025, 1, 2)],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "ts_code": ["a", "b"],
            "raw_return_5d": [0.01, 0.02],
            "label_valid": [True, True],
            "exit_date": [date(2025, 1, 10), date(2025, 1, 10)],
        }
    )

    built = build_point_in_time_dataset(
        universe,
        factors,
        labels,
        factor_members=(FactorInput("f1", 1, "f1"),),
        segment_end=date(2025, 1, 10),
        request=DatasetBuildRequest(fold_id="F01", split_role="TRAIN"),
    )

    assert built.columns.tolist() == [
        "signal_date",
        "ts_code",
        "f1",
        "raw_return_5d",
        "label_valid",
        "exit_date",
    ]
    assert len(built) == 2

    factors.loc[0, "available_date"] = date(2025, 1, 3)
    with pytest.raises(ValueError, match="available"):
        build_point_in_time_dataset(
            universe,
            factors,
            labels,
            factor_members=(FactorInput("f1", 1, "f1"),),
            segment_end=date(2025, 1, 10),
            request=DatasetBuildRequest(fold_id="F01", split_role="TRAIN"),
        )


def test_builder_rejects_locked_final_test_before_reading_inputs() -> None:
    with pytest.raises(LockedDatasetError):
        build_point_in_time_dataset(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            factor_members=(FactorInput("f1", 1, "f1"),),
            segment_end=date(2026, 12, 31),
            request=DatasetBuildRequest(fold_id="FINAL", split_role="TEST"),
        )


def test_build_request_rejects_runtime_split_role_before_lock_evaluation() -> None:
    with pytest.raises(ValueError, match="split_role"):
        DatasetBuildRequest(fold_id="FINAL", split_role="test")  # type: ignore[arg-type]


def test_builder_rejects_locked_year_for_ordinary_dataset_requests() -> None:
    with pytest.raises(LockedDatasetError, match="2026|locked"):
        build_point_in_time_dataset(
            pd.DataFrame({"signal_date": [date(2026, 1, 2)]}),
            pd.DataFrame(),
            pd.DataFrame(),
            factor_members=(FactorInput("f1", 1, "f1"),),
            segment_end=date(2026, 12, 31),
            request=DatasetBuildRequest(fold_id="F01", split_role="TEST"),
        )


def test_builder_accepts_locked_final_test_only_with_explicit_access() -> None:
    access = _locked_access()

    built = build_point_in_time_dataset(
        pd.DataFrame({"signal_date": [date(2026, 1, 2)], "ts_code": ["a"]}),
        pd.DataFrame(
            {
                "signal_date": [date(2026, 1, 2)],
                "ts_code": ["a"],
                "factor_id": ["f1"],
                "factor_version": [1],
                "value": [1.0],
                "available_date": [date(2026, 1, 2)],
            }
        ),
        pd.DataFrame(
            {
                "signal_date": [date(2026, 1, 2)],
                "ts_code": ["a"],
                "raw_return_5d": [0.01],
                "label_valid": [True],
                "exit_date": [date(2026, 1, 12)],
            }
        ),
        factor_members=(FactorInput("f1", 1, "f1"),),
        segment_end=date(2026, 12, 31),
        request=DatasetBuildRequest(fold_id="FINAL", split_role="TEST", locked_access=access),
    )

    assert len(built) == 1


def test_builder_rejects_a_fabricated_unsigned_locked_access() -> None:
    forged = LockedTestAccess(locked_year=2026, authorization_id="fabricated")

    with pytest.raises(LockedDatasetError, match="signature"):
        DatasetBuildRequest(fold_id="FINAL", split_role="TEST", locked_access=forged)


def test_builder_accepts_v2_label_daily_raw_return_column() -> None:
    universe = pd.DataFrame({"signal_date": [date(2025, 1, 2)], "ts_code": ["a"]})
    factors = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)],
            "ts_code": ["a"],
            "factor_id": ["f1"],
            "factor_version": [1],
            "value": [1.0],
            "available_date": [date(2025, 1, 2)],
        }
    )
    v2_labels = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)],
            "ts_code": ["a"],
            "raw_return": [0.01],
            "label_valid": [True],
            "exit_date": [date(2025, 1, 10)],
        }
    )

    built = build_point_in_time_dataset(
        universe,
        factors,
        v2_labels,
        factor_members=(FactorInput("f1", 1, "f1"),),
        segment_end=date(2025, 1, 10),
        request=DatasetBuildRequest(fold_id="F01", split_role="TRAIN"),
    )

    assert built.loc[0, "raw_return_5d"] == 0.01


def test_builder_applies_frozen_factor_admission_before_training() -> None:
    universe = pd.DataFrame(
        {"signal_date": [date(2025, 1, 2), date(2025, 1, 2)], "ts_code": ["a", "b"]}
    )
    factors = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "ts_code": ["a", "b"],
            "factor_id": ["f1", "f1"],
            "factor_version": [1, 1],
            "value": [1.0, 2.0],
            "available_date": [date(2025, 1, 2), date(2025, 1, 2)],
        }
    )
    labels = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "ts_code": ["a", "b"],
            "raw_return_5d": [0.01, 0.02],
            "label_valid": [True, True],
            "exit_date": [date(2025, 1, 10), date(2025, 1, 10)],
        }
    )
    factor_set = create_factor_set(
        [("f1", 1, 1, "definition")], selection_scope_id="scope", rules_version=1
    )
    factor_set = transition_factor_set(factor_set, FactorSetStatus.VALIDATED)
    factor_set = transition_factor_set(factor_set, FactorSetStatus.FROZEN)
    contract = FactorAdmissionContract.from_factor_set(
        factor_set,
        feature_names_by_factor_id={"f1": "f1"},
        rules_hash="a" * 64,
        preprocessor_hash="b" * 64,
    )

    built = build_point_in_time_dataset(
        universe,
        factors,
        labels,
        factor_members=(FactorInput("f1", 1, "f1"),),
        segment_end=date(2025, 1, 10),
        request=DatasetBuildRequest(fold_id="F01", split_role="TRAIN"),
        admission_contract=contract,
    )

    assert len(built) == 2

    with pytest.raises(FactorAdmissionError, match="IDs, versions, and order"):
        build_point_in_time_dataset(
            universe,
            factors,
            labels,
            factor_members=(FactorInput("f1", 2, "f1"),),
            segment_end=date(2025, 1, 10),
            request=DatasetBuildRequest(fold_id="F01", split_role="TRAIN"),
            admission_contract=contract,
        )

    factors.loc[1, "value"] = float("nan")
    with pytest.raises(FactorAdmissionError, match="finite ratio"):
        build_point_in_time_dataset(
            universe,
            factors,
            labels,
            factor_members=(FactorInput("f1", 1, "f1"),),
            segment_end=date(2025, 1, 10),
            request=DatasetBuildRequest(fold_id="F01", split_role="TRAIN"),
            admission_contract=contract,
            minimum_finite_ratio=0.75,
        )
