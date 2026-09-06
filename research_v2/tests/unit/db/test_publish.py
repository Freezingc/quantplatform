from datetime import date
from uuid import uuid4

import pandas as pd
import pytest

from research_v2.db.publish import (
    PublishState,
    PublishValidationError,
    StagedBatch,
    V2BatchPublisher,
)
from research_v2.materializers.oos import build_model_oos_daily_batch
from research_v2.quality.engine import CheckResult, CheckStatus, Severity
from research_v2.runs.context import RunContext


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


class OosSchemaSession(RecordingWriteSession):
    def query(self, script: str) -> object:
        if script.startswith("schema("):
            return pd.DataFrame(
                {
                    "name": [
                        "signal_date",
                        "evaluation_id",
                        "model_id",
                        "fold_id",
                        "role",
                        "rank_ic",
                        "sample_count",
                        "run_id",
                        "data_version",
                        "created_at",
                        "source_watermark_id",
                        "record_hash",
                        "eligible_count",
                        "valid_label_count",
                        "coverage",
                        "prediction_unique_count",
                        "prediction_std",
                        "rank_ic_valid",
                    ],
                    "typeString": [
                        "DATE",
                        "STRING",
                        "SYMBOL",
                        "SYMBOL",
                        "SYMBOL",
                        "DOUBLE",
                        "LONG",
                        "UUID",
                        "STRING",
                        "TIMESTAMP",
                        "STRING",
                        "STRING",
                        "LONG",
                        "LONG",
                        "DOUBLE",
                        "LONG",
                        "DOUBLE",
                        "BOOL",
                    ],
                }
            )
        return super().query(script)


def test_oos_materialized_batch_matches_migrated_target_column_order() -> None:
    daily = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 2)],
            "rank_ic": [0.1],
            "sample_count": [100],
            "eligible_count": [110],
            "valid_label_count": [105],
            "coverage": [0.95],
            "prediction_unique_count": [100],
            "prediction_std": [0.25],
            "rank_ic_valid": [True],
        }
    )
    batch = build_model_oos_daily_batch(
        daily,
        RunContext(
            run_id="00000000-0000-0000-0000-000000000001",
            data_version="v1",
        ),
        evaluation_id="eval",
        model_id="model",
        fold_id="F06",
        source_watermark_id="source-v1",
    )
    session = OosSchemaSession()

    V2BatchPublisher(session).publish(
        batch,
        [CheckResult("keys_unique", Severity.BLOCKER, CheckStatus.PASS)],
    )

    assert session.uploads[0][1].columns.tolist() == [
        "signal_date",
        "evaluation_id",
        "model_id",
        "fold_id",
        "role",
        "rank_ic",
        "sample_count",
        "run_id",
        "data_version",
        "created_at",
        "source_watermark_id",
        "record_hash",
        "eligible_count",
        "valid_label_count",
        "coverage",
        "prediction_unique_count",
        "prediction_std",
        "rank_ic_valid",
    ]


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
            idempotent=False,
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


# --- C5.5/C5.6: the same-version retry path, which is what "正常发布幂等" actually means ----------


class RetrySession(RecordingWriteSession):
    """A session whose target already holds ``current_version`` with ``current_rows`` rows."""

    def __init__(self, current_version: str, current_rows: int) -> None:
        super().__init__()
        self._current_version = current_version
        self._current_rows = current_rows

    def query(self, script: str) -> object:
        if script.startswith("schema("):
            return super().query(script)
        if "count(*) as row_count" in script:
            # The real query is scoped by data_version, so rows exist only for the version
            # already published. Answering every count with the same number would make a
            # genuinely new version look like it had orphan rows, and the retry branch would
            # never be distinguishable from a first publish.
            if f"data_version='{self._current_version}'" not in script:
                return pd.DataFrame({"row_count": [0]})
            return pd.DataFrame({"row_count": [self._current_rows]})
        return pd.DataFrame({"data_version": [self._current_version]})


def _retry_batch(rows: int = 2, data_version: str = "version-a") -> StagedBatch:
    return StagedBatch.create(
        object_name="market_daily_clean",
        logical_partition="2025-01",
        data_version=data_version,
        frame=pd.DataFrame(
            {
                "trade_date": ["2025-01-02"] * rows,
                "ts_code": [f"{index:06d}.SZ" for index in range(rows)],
                "value": [float(index) for index in range(rows)],
            }
        ),
        business_keys=("trade_date", "ts_code"),
    )


_PASSING = [CheckResult("keys_unique", Severity.BLOCKER, CheckStatus.PASS)]
_FAILING = [CheckResult("keys_unique", Severity.BLOCKER, CheckStatus.FAIL)]


def test_a_same_version_retry_writes_nothing_at_all() -> None:
    """C5.6's 正常发布幂等: a retry is a no-op against the target, not a second append.

    This is the property that makes an interrupted run safe to restart. If a retry re-uploaded and
    re-inserted, the partition would hold the same rows twice and the business-key uniqueness the
    publisher enforces on the way in would already have been violated in the table.
    """
    batch = _retry_batch(rows=2)
    batch.validate()
    session = RetrySession(current_version="version-a", current_rows=2)

    V2BatchPublisher(session).publish(batch, list(_PASSING))

    assert session.uploads == []
    assert session.scripts == []
    assert batch.state is PublishState.PUBLISHED


def test_a_retry_carrying_different_data_under_the_same_version_is_refused() -> None:
    """The sharpest edge of idempotence: same version, different content, must not ride through.

    ``data_version`` is content-addressed, so a retry whose row count differs from what the target
    already holds means the caller is reusing an existing version identifier for different data.
    Accepting that would let a version pointer keep its meaning while the rows behind it changed --
    which is precisely the invariant the pointer exists to protect, and it would be invisible
    afterwards because nothing about the version string would look wrong.
    """
    batch = _retry_batch(rows=3)
    batch.validate()
    session = RetrySession(current_version="version-a", current_rows=2)

    with pytest.raises(PublishValidationError, match="row count does not match"):
        V2BatchPublisher(session).publish(batch, list(_PASSING))

    assert session.uploads == []
    assert session.scripts == []


def test_a_retry_still_passes_through_the_quality_gate() -> None:
    """Idempotence shortcuts the write, not the checks."""
    batch = _retry_batch(rows=2)
    batch.validate()
    session = RetrySession(current_version="version-a", current_rows=2)

    with pytest.raises(PublishValidationError, match="data-quality gate"):
        V2BatchPublisher(session).publish(batch, list(_FAILING))

    assert batch.state is PublishState.REJECTED
    assert session.uploads == []
    assert session.scripts == []


def test_an_empty_quality_run_never_satisfies_a_retry() -> None:
    """No checks run is not the same as checks passed."""
    batch = _retry_batch(rows=2)
    batch.validate()
    session = RetrySession(current_version="version-a", current_rows=2)

    with pytest.raises(PublishValidationError, match="data-quality gate"):
        V2BatchPublisher(session).publish(batch, [])


def test_a_rejected_batch_cannot_be_retried_into_existence() -> None:
    """Retry is for restarting an interrupted publish, not for resurrecting a refused one."""
    batch = _retry_batch(rows=2)
    batch.validate()
    batch.state = PublishState.REJECTED
    session = RetrySession(current_version="version-a", current_rows=2)

    with pytest.raises(PublishValidationError, match="newly created or validated"):
        V2BatchPublisher(session).publish(batch, list(_PASSING))

    assert session.uploads == []


def test_a_batch_that_was_never_validated_is_validated_by_the_retry() -> None:
    """A CREATED batch is the normal state after a crash between staging and validating."""
    batch = _retry_batch(rows=2)
    assert batch.state is PublishState.CREATED
    session = RetrySession(current_version="version-a", current_rows=2)

    V2BatchPublisher(session).publish(batch, list(_PASSING))

    assert batch.state is PublishState.PUBLISHED
    assert session.uploads == []


def test_an_already_published_batch_may_be_retried_without_writing_again() -> None:
    batch = _retry_batch(rows=2)
    batch.validate()
    batch.publish()
    assert batch.state is PublishState.PUBLISHED
    session = RetrySession(current_version="version-a", current_rows=2)

    V2BatchPublisher(session).publish(batch, list(_PASSING))

    assert batch.state is PublishState.PUBLISHED
    assert session.uploads == []
    assert session.scripts == []


def test_a_different_version_is_a_real_publish_not_a_retry() -> None:
    """The retry branch must not swallow a genuine new version."""
    batch = _retry_batch(rows=2, data_version="version-b")
    session = RetrySession(current_version="version-a", current_rows=2)

    V2BatchPublisher(session).publish(batch, list(_PASSING))

    assert batch.state is PublishState.PUBLISHED
    assert session.uploads, "a new version must actually be written"
    assert session.scripts, "and the pointer must be advanced"
