"""Batch publish integration evidence against the in-memory Test DB (B7.4/D9.4)."""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from research_v2.db.migrations import MigrationRunner
from research_v2.db.publish import (
    PublishState,
    PublishValidationError,
    StagedBatch,
    V2BatchPublisher,
)
from research_v2.quality.engine import CheckResult, CheckStatus, Severity

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("QRV2_INTEGRATION") != "1",
        reason="set QRV2_INTEGRATION=1 to run database integration tests",
    ),
]

_MIGRATION_DIRECTORY = Path(__file__).resolve().parents[3] / "migrations"
_CLEAN = [CheckResult("row_count", Severity.INFO, CheckStatus.PASS)]
_BLOCKED = [CheckResult("duplicate_keys", Severity.BLOCKER, CheckStatus.FAIL)]


def _bootstrapped_database(fake_v2_test_database):
    runner = MigrationRunner(fake_v2_test_database, migration_directory=_MIGRATION_DIRECTORY)
    runner.apply()
    return fake_v2_test_database


def _staged_frame(day_count: int, *, version: str, start_day: int = 2) -> pd.DataFrame:
    rows = []
    for offset in range(day_count):
        day = date(2024, 1, start_day + offset)
        for index in range(3):
            rows.append(
                {
                    "trade_date": day,
                    "ts_code": f"s{index:02d}",
                    "adjusted_open": 10.0 + offset + index,
                    "data_version": version,
                }
            )
    return pd.DataFrame(rows)


def _batch(version: str, *, day_count: int = 2, start_day: int = 2) -> StagedBatch:
    return StagedBatch.create(
        object_name="market_daily_clean",
        logical_partition="2024-01",
        data_version=version,
        frame=_staged_frame(day_count, version=version, start_day=start_day),
        business_keys=("trade_date", "ts_code"),
        partition_date_column="trade_date",
    )


def test_publish_appends_rows_and_advances_pointer_last(fake_v2_test_database) -> None:
    database = _bootstrapped_database(fake_v2_test_database)
    publisher = V2BatchPublisher(database)
    batch = _batch("sha256:" + "a" * 64)

    publisher.publish(batch, _CLEAN)

    table = database.tables["market_daily_clean"]
    assert len(table) == 6
    assert set(table["data_version"]) == {"sha256:" + "a" * 64}
    pointers = database.tables["published_version"]
    assert len(pointers) == 1
    pointer = pointers.iloc[0]
    assert pointer["object_name"] == "market_daily_clean"
    assert pointer["logical_partition"] == "2024-01"
    assert pointer["data_version"] == "sha256:" + "a" * 64
    assert pointer["previous_data_version"] == ""
    assert batch.state is PublishState.PUBLISHED
    # The visibility pointer must be the final write of the transaction.
    assert "published_version" in database.executed_scripts[-1]


def test_republish_of_current_version_is_idempotent(fake_v2_test_database) -> None:
    database = _bootstrapped_database(fake_v2_test_database)
    publisher = V2BatchPublisher(database)
    version = "sha256:" + "b" * 64
    publisher.publish(_batch(version), _CLEAN)

    publisher.publish(_batch(version), _CLEAN)

    assert len(database.tables["market_daily_clean"]) == 6
    assert len(database.tables["published_version"]) == 1


def test_blocking_dq_rejects_without_visibility(fake_v2_test_database) -> None:
    database = _bootstrapped_database(fake_v2_test_database)
    publisher = V2BatchPublisher(database)
    batch = _batch("sha256:" + "c" * 64)

    with pytest.raises(PublishValidationError, match="data-quality gate"):
        publisher.publish(batch, _BLOCKED)

    assert len(database.tables["market_daily_clean"]) == 0
    assert "published_version" not in database.tables or database.tables[
        "published_version"
    ].empty
    assert batch.state is PublishState.REJECTED


def test_publish_chunks_share_one_version_and_pointer(fake_v2_test_database) -> None:
    database = _bootstrapped_database(fake_v2_test_database)
    publisher = V2BatchPublisher(database)
    version = "sha256:" + "d" * 64
    first_chunk = StagedBatch.create(
        object_name="market_daily_clean",
        logical_partition="2024-01",
        data_version=version,
        frame=_staged_frame(2, version=version, start_day=2),
        business_keys=("trade_date", "ts_code"),
        partition_date_column="trade_date",
    )
    second_chunk = StagedBatch.create(
        object_name="market_daily_clean",
        logical_partition="2024-01",
        data_version=version,
        frame=_staged_frame(2, version=version, start_day=4),
        business_keys=("trade_date", "ts_code"),
        partition_date_column="trade_date",
    )
    chunks = [first_chunk, second_chunk]

    publisher.publish_chunks(chunks, _CLEAN)

    table = database.tables["market_daily_clean"]
    assert len(table) == 12
    assert set(table["data_version"]) == {version}
    pointers = database.tables["published_version"]
    assert len(pointers) == 1
    assert all(chunk.state is PublishState.PUBLISHED for chunk in chunks)
