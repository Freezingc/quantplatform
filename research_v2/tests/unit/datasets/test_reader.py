from __future__ import annotations

import re
from datetime import UTC, date, datetime

import pandas as pd
import pytest

from research_v2.datasets.builder import DatasetBuildRequest, FactorInput, LockedTestAccess
from research_v2.datasets.reader import CompactDatasetReader, CompactDataUnavailable
from research_v2.factors.admission import FactorAdmissionContract, FactorAdmissionError
from research_v2.factors.factor_sets import (
    FactorSetStatus,
    create_factor_set,
    transition_factor_set,
)


class ReaderSession:
    target_database = "dfs://QuantResearchV2Test"

    def __init__(self, *, publish_month: bool = True) -> None:
        self.publish_month = publish_month
        self.queries: list[str] = []
        self.frames = {
            "universe_daily_latest": pd.DataFrame(
                {
                    "batch_id": [11, 11],
                    "signal_date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
                    "ts_code": ["a", "b"],
                    "is_member": [True, True],
                }
            ),
            "label_daily_latest": pd.DataFrame(
                {
                    "batch_id": [12, 12],
                    "signal_date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
                    "ts_code": ["a", "b"],
                    "raw_return": [0.01, 0.02],
                    "label_valid": [True, True],
                    "exit_date": pd.to_datetime(["2025-01-10", "2025-01-10"]),
                }
            ),
            "factor_observation_clean_latest": pd.DataFrame(
                {
                    "batch_id": [13, 13],
                    "signal_date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
                    "ts_code": ["a", "b"],
                    "factor_id": ["f1", "f1"],
                    "factor_version": [1, 1],
                    "value": [1.0, 2.0],
                    "available_date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
                }
            ),
        }

    def upload_frame(self, name: str, frame: pd.DataFrame) -> None:
        raise AssertionError("reader must not upload")

    def execute(self, script: str) -> object:
        raise AssertionError("reader must not execute")

    def query(self, script: str) -> object:
        self.queries.append(script)
        if "latest_manifest" in script:
            object_match = re.search(r"object_name=`([A-Za-z0-9_]+)", script)
            partition_match = re.search(r"logical_partition='([^']+)'", script)
            object_name = object_match.group(1) if object_match else ""
            partition = partition_match.group(1) if partition_match else ""
            if not self.publish_month or partition != "2025-01":
                return pd.DataFrame()
            source_object = object_name.removesuffix("_latest")
            batch_id = {"universe_daily_latest": 11, "label_daily_latest": 12}.get(
                object_name, 13
            )
            return pd.DataFrame(
                {
                    "object_name": [object_name],
                    "logical_partition": [partition],
                    "batch_id": [batch_id],
                    "source_object": [source_object],
                    "source_data_version": [f"{source_object}-v1"],
                    "published_at": [datetime(2026, 8, 2, tzinfo=UTC)],
                    "row_count": [2],
                    "business_hash": ["a" * 64],
                    "status": ["PUBLISHED"],
                }
            )
        for object_name, frame in self.frames.items():
            if object_name in script:
                return frame.copy(deep=True)
        raise AssertionError(f"unexpected query: {script}")


def _contract() -> FactorAdmissionContract:
    factor_set = create_factor_set(
        [("f1", 1, 1, "definition")], selection_scope_id="scope", rules_version=1
    )
    factor_set = transition_factor_set(factor_set, FactorSetStatus.VALIDATED)
    factor_set = transition_factor_set(factor_set, FactorSetStatus.FROZEN)
    return FactorAdmissionContract.from_factor_set(
        factor_set,
        feature_names_by_factor_id={"f1": "f1"},
        rules_hash="a" * 64,
        preprocessor_hash="b" * 64,
    )


def test_reader_fails_closed_when_a_requested_month_has_no_manifest() -> None:
    reader = CompactDatasetReader(ReaderSession(publish_month=False))

    with pytest.raises(CompactDataUnavailable, match="no published compact batch"):
        reader.read_range(
            object_name="label_daily_latest",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
            columns=("batch_id", "signal_date", "ts_code", "raw_return"),
        )


def test_reader_builds_pit_dataset_from_latest_tables_and_requires_admission() -> None:
    session = ReaderSession()
    reader = CompactDatasetReader(session)

    result = reader.build_point_in_time_dataset(
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
        factor_members=(FactorInput("f1", 1, "f1"),),
        segment_end=date(2025, 1, 10),
        request=DatasetBuildRequest(fold_id="F01", split_role="TRAIN"),
        admission_contract=_contract(),
    )

    assert result.columns.tolist() == [
        "signal_date",
        "ts_code",
        "f1",
        "raw_return_5d",
        "label_valid",
        "exit_date",
    ]
    assert len(result) == 2
    assert any("factor_id='f1' and factor_version=1" in query for query in session.queries)


def test_reader_preflights_factor_coverage_before_locked_label_access() -> None:
    session = ReaderSession()
    session.frames["factor_observation_clean_latest"].loc[1, "value"] = float("nan")
    reader = CompactDatasetReader(session)

    with pytest.raises(FactorAdmissionError, match="finite ratio"):
        reader.build_point_in_time_dataset(
            start_date=date(2025, 1, 1),
            end_date=date(2025, 1, 31),
            factor_members=(FactorInput("f1", 1, "f1"),),
            segment_end=date(2025, 1, 10),
            request=DatasetBuildRequest(
                fold_id="FINAL",
                split_role="TEST",
                locked_access=LockedTestAccess(2026, "approved"),
            ),
            admission_contract=_contract(),
        )

    assert not any(
        "latest_manifest" in query and "label_daily_latest" in query
        for query in session.queries
    )
