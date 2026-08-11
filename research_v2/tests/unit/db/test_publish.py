from uuid import uuid4

import pandas as pd
import pytest

from research_v2.db.publish import (
    PublishState,
    PublishValidationError,
    StagedBatch,
    V2BatchPublisher,
)
from research_v2.quality.engine import CheckResult, CheckStatus, Severity


class RecordingWriteSession:
    target_database = "dfs://QuantResearchV2Test"

    def __init__(self) -> None:
        self.uploads: list[tuple[str, pd.DataFrame]] = []
        self.scripts: list[str] = []

    def upload_frame(self, name: str, frame: pd.DataFrame) -> None:
        self.uploads.append((name, frame.copy()))

    def execute(self, script: str) -> object:
        self.scripts.append(script)
        return None

    def query(self, script: str) -> object:
        if script.startswith("schema("):
            return pd.DataFrame(
                {
                    "name": ["delist_date", "run_id"],
                    "typeString": ["DATE", "UUID"],
                }
            )
        if "count(*) as row_count" in script:
            return pd.DataFrame({"row_count": [0]})
        return pd.DataFrame(columns=["data_version"])


def test_staged_batch_publishes_after_business_key_validation() -> None:
    batch = StagedBatch.create(
        object_name="market_daily_clean",
        logical_partition="2025-01",
        data_version="version-a",
        frame=pd.DataFrame(
            {
                "trade_date": ["2025-01-02", "2025-01-02"],
                "ts_code": ["000001.SZ", "000002.SZ"],
                "value": [1.0, 2.0],
            }
        ),
        business_keys=("trade_date", "ts_code"),
    )

    batch.validate()
    batch.publish()

    assert batch.state is PublishState.PUBLISHED


def test_staged_batch_rejects_duplicate_business_keys_before_publish() -> None:
    batch = StagedBatch.create(
        object_name="market_daily_clean",
        logical_partition="2025-01",
        data_version="version-a",
        frame=pd.DataFrame(
            {
                "trade_date": ["2025-01-02", "2025-01-02"],
                "ts_code": ["000001.SZ", "000001.SZ"],
            }
        ),
        business_keys=("trade_date", "ts_code"),
    )

    with pytest.raises(PublishValidationError, match="duplicate"):
        batch.validate()

    assert batch.state is PublishState.REJECTED


def test_staged_batch_cannot_publish_without_validation() -> None:
    batch = StagedBatch.create(
        object_name="market_daily_clean",
        logical_partition="2025-01",
        data_version="version-a",
        frame=pd.DataFrame({"trade_date": ["2025-01-02"], "ts_code": ["000001.SZ"]}),
        business_keys=("trade_date", "ts_code"),
    )

    with pytest.raises(PublishValidationError, match="validated"):
        batch.publish()


def test_split_staged_batch_by_month_creates_one_valid_monthly_publish_batch() -> None:
    batch = StagedBatch.create(
        object_name="market_daily_clean",
        logical_partition="2024-12",
        data_version="version-a",
        frame=pd.DataFrame(
            {
                "trade_date": ["2024-12-31", "2025-01-02"],
                "ts_code": ["000001.SZ", "000001.SZ"],
            }
        ),
        business_keys=("trade_date", "ts_code"),
        partition_date_column="trade_date",
    )

    split = batch.split_by_month()

    assert [item.logical_partition for item in split] == ["2024-12", "2025-01"]
    assert [len(item.frame) for item in split] == [1, 1]
    for item in split:
        item.validate()


def test_staged_batch_rejects_dates_outside_its_declared_month_partition() -> None:
    batch = StagedBatch.create(
        object_name="market_daily_clean",
        logical_partition="2024-12",
        data_version="version-a",
        frame=pd.DataFrame(
            {
                "trade_date": ["2024-12-31", "2025-01-02"],
                "ts_code": ["000001.SZ", "000001.SZ"],
            }
        ),
        business_keys=("trade_date", "ts_code"),
        partition_date_column="trade_date",
    )

    with pytest.raises(PublishValidationError, match="outside logical partition"):
        batch.validate()


def test_database_publisher_updates_pointer_only_after_validated_write() -> None:
    batch = StagedBatch.create(
        object_name="market_daily_clean",
        logical_partition="2025-01",
        data_version="version-a",
        frame=pd.DataFrame({"trade_date": ["2025-01-02"], "ts_code": ["000001.SZ"]}),
        business_keys=("trade_date", "ts_code"),
    )
    session = RecordingWriteSession()

    V2BatchPublisher(session).publish(
        batch,
        [CheckResult("keys_unique", Severity.BLOCKER, CheckStatus.PASS)],
    )

    assert batch.state is PublishState.PUBLISHED
    assert len(session.uploads) == 1
    assert "tableInsert" in session.scripts[0]
    assert "published_version" in session.scripts[1]


def test_database_publisher_does_not_write_when_dq_rejects_batch() -> None:
    batch = StagedBatch.create(
        object_name="market_daily_clean",
        logical_partition="2025-01",
        data_version="version-a",
        frame=pd.DataFrame({"trade_date": ["2025-01-02"], "ts_code": ["000001.SZ"]}),
        business_keys=("trade_date", "ts_code"),
    )
    session = RecordingWriteSession()

    with pytest.raises(PublishValidationError, match="data-quality"):
        V2BatchPublisher(session).publish(
            batch,
            [CheckResult("keys_unique", Severity.BLOCKER, CheckStatus.FAIL)],
        )

    assert batch.state is PublishState.REJECTED
    assert session.uploads == []
    assert session.scripts == []


def test_database_publisher_rejects_republishing_the_current_data_version() -> None:
    batch = StagedBatch.create(
        object_name="market_daily_clean",
        logical_partition="2025-01",
        data_version="version-a",
        frame=pd.DataFrame({"trade_date": ["2025-01-02"], "ts_code": ["000001.SZ"]}),
        business_keys=("trade_date", "ts_code"),
    )

    class CurrentVersionSession(RecordingWriteSession):
        def query(self, script: str) -> object:
            if script.startswith("schema("):
                return super().query(script)
            if "count(*) as row_count" in script:
                return pd.DataFrame({"row_count": [0]})
            return pd.DataFrame({"data_version": ["version-a"]})

    session = CurrentVersionSession()

    with pytest.raises(PublishValidationError, match="already current"):
        V2BatchPublisher(session).publish(
            batch,
            [CheckResult("keys_unique", Severity.BLOCKER, CheckStatus.PASS)],
        )

    assert session.uploads == []
    assert session.scripts == []


def test_database_publisher_converts_uploaded_uuid_columns_on_the_server() -> None:
    identifier = uuid4()
    batch = StagedBatch.create(
        object_name="source_reconciliation",
        logical_partition="2025-01",
        data_version="version-a",
        frame=pd.DataFrame(
            {
                "trade_date": ["2025-01-02"],
                "ts_code": ["000001.SZ"],
                "field_name": ["open"],
                "check_version": [1],
                "run_id": [identifier],
            }
        ),
        business_keys=("trade_date", "ts_code", "field_name", "check_version"),
    )
    session = RecordingWriteSession()

    V2BatchPublisher(session).publish(
        batch,
        [CheckResult("keys_unique", Severity.BLOCKER, CheckStatus.PASS)],
    )

    assert session.uploads[0][1].loc[0, "run_id"] == str(identifier)
    assert "uuid(run_id) as run_id" in session.scripts[0]


def test_database_publisher_casts_all_null_target_date_columns_on_the_server() -> None:
    batch = StagedBatch.create(
        object_name="security_master_scd",
        logical_partition="2025-01-03",
        data_version="version-a",
        frame=pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "valid_from": ["2025-01-03"],
                "delist_date": [pd.NaT],
                "run_id": ["00000000-0000-0000-0000-000000000001"],
            }
        ),
        business_keys=("ts_code", "valid_from"),
    )
    session = RecordingWriteSession()

    V2BatchPublisher(session).publish(
        batch,
        [CheckResult("keys_unique", Severity.BLOCKER, CheckStatus.PASS)],
    )

    assert "date(delist_date) as delist_date" in session.scripts[0]


def test_database_publisher_publishes_chunks_before_advancing_one_pointer() -> None:
    batches = [
        StagedBatch.create(
            object_name="factor_observation_clean",
            logical_partition="2025-01",
            data_version="version-a",
            frame=pd.DataFrame(
                {
                    "signal_date": ["2025-01-02"],
                    "ts_code": [f"00000{index}.SZ"],
                    "factor_id": [f"f{index}"],
                    "factor_version": [1],
                }
            ),
            business_keys=("signal_date", "ts_code", "factor_id", "factor_version"),
        )
        for index in range(2)
    ]
    session = RecordingWriteSession()

    V2BatchPublisher(session).publish_chunks(
        batches,
        [CheckResult("keys_unique", Severity.BLOCKER, CheckStatus.PASS)],
    )

    assert len(session.uploads) == 2
    assert len([script for script in session.scripts if "published_version" in script]) == 1


def test_database_publisher_streams_chunks_without_materializing_one_combined_frame() -> None:
    batches = (
        StagedBatch.create(
            object_name="factor_observation_clean",
            logical_partition="2025-01",
            data_version="stream-version",
            frame=pd.DataFrame(
                {
                    "signal_date": ["2025-01-02"],
                    "ts_code": [f"00000{index}.SZ"],
                    "factor_id": [f"f{index}"],
                    "factor_version": [1],
                }
            ),
            business_keys=("signal_date", "ts_code", "factor_id", "factor_version"),
        )
        for index in range(2)
    )
    session = RecordingWriteSession()

    V2BatchPublisher(session).publish_chunk_stream(
        (batch for batch in batches),
        [CheckResult("keys_unique", Severity.BLOCKER, CheckStatus.PASS)],
    )

    assert len(session.uploads) == 2
    assert len([script for script in session.scripts if "published_version" in script]) == 1


def test_database_publisher_stream_cleans_each_temporary_stage() -> None:
    batch = StagedBatch.create(
        object_name="factor_observation_clean",
        logical_partition="2025-01",
        data_version="stream-version",
        frame=pd.DataFrame(
            {
                "signal_date": ["2025-01-02"],
                "ts_code": ["000001.SZ"],
                "factor_id": ["f1"],
                "factor_version": [1],
            }
        ),
        business_keys=("signal_date", "ts_code", "factor_id", "factor_version"),
    )
    session = RecordingWriteSession()

    V2BatchPublisher(session).publish_chunk_stream(
        [batch],
        [CheckResult("keys_unique", Severity.BLOCKER, CheckStatus.PASS)],
    )

    assert any(script.startswith("undef(`stage_") for script in session.scripts)


def test_database_publisher_scopes_orphan_check_to_the_batch_month() -> None:
    batch = StagedBatch.create(
        object_name="factor_observation_clean",
        logical_partition="2023-07",
        data_version="stream-version",
        frame=pd.DataFrame(
            {
                "signal_date": ["2023-07-03"],
                "ts_code": ["000001.SZ"],
                "factor_id": ["f1"],
                "factor_version": [1],
            }
        ),
        business_keys=("signal_date", "ts_code", "factor_id", "factor_version"),
        partition_date_column="signal_date",
    )
    class QueryRecordingSession(RecordingWriteSession):
        def __init__(self) -> None:
            super().__init__()
            self.queries: list[str] = []

        def query(self, script: str) -> object:
            self.queries.append(script)
            return super().query(script)

    session = QueryRecordingSession()
    publisher = V2BatchPublisher(session)

    publisher._current_version_row_count(batch)

    count_query = next(script for script in session.queries if "count(*) as row_count" in script)
    assert "signal_date between date('2023.07.01') and date('2023.07.31')" in count_query
