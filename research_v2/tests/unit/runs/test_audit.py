from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from research_v2.runs.audit import (
    PipelineRunAuditStore,
    PipelineRunRecord,
    PipelineRunStatus,
    pipeline_run,
    scrub_message,
)
from research_v2.runs.context import RunContext

_SENTINEL = "sentinel-secret-9f8e7d6c"


def _context() -> RunContext:
    return RunContext.build(
        config={"stage": "unit"},
        watermarks=("wm-1",),
        code_commit="sha256:" + "a" * 64,
        schema_version="013",
    )


def _clock(start: datetime):
    state = {"now": start}

    def clock() -> datetime:
        current = state["now"]
        state["now"] = current + timedelta(seconds=1)
        return current

    return clock


class RecordingSink:
    def __init__(self) -> None:
        self.records: list[PipelineRunRecord] = []

    def write(self, record: PipelineRunRecord) -> None:
        self.records.append(record)


def test_pipeline_run_writes_running_then_success() -> None:
    sink = RecordingSink()
    started = datetime(2025, 3, 1, tzinfo=UTC)
    with pipeline_run(
        sink,
        name="db.migrate",
        context=_context(),
        config_hash="sha256:" + "b" * 64,
        code_commit="sha256:" + "a" * 64,
        clock=_clock(started),
    ):
        pass

    assert [record.status for record in sink.records] == [
        PipelineRunStatus.RUNNING,
        PipelineRunStatus.SUCCESS,
    ]
    running, success = sink.records
    assert running.run_id == success.run_id
    assert running.finished_at is None and running.error_message is None
    assert success.finished_at is not None and success.finished_at > running.started_at
    assert success.pipeline_name == "db.migrate"


def test_pipeline_run_writes_failed_status_and_reraises() -> None:
    sink = RecordingSink()
    with pytest.raises(ValueError, match="boom"), pipeline_run(
        sink,
        name="db.migrate",
        context=_context(),
        config_hash="sha256:" + "b" * 64,
        code_commit="sha256:" + "a" * 64,
        clock=_clock(datetime(2025, 3, 1, tzinfo=UTC)),
    ):
        raise ValueError("boom")

    assert [record.status for record in sink.records] == [
        PipelineRunStatus.RUNNING,
        PipelineRunStatus.FAILED,
    ]
    failed = sink.records[-1]
    assert failed.error_class == "ValueError"
    assert failed.error_message == "boom"


def test_failed_run_scrubs_registered_secrets_from_audit() -> None:
    sink = RecordingSink()
    with pytest.raises(RuntimeError), pipeline_run(
        sink,
        name="db.migrate",
        context=_context(),
        config_hash="sha256:" + "b" * 64,
        code_commit="sha256:" + "a" * 64,
        secrets=(_SENTINEL,),
        clock=_clock(datetime(2025, 3, 1, tzinfo=UTC)),
    ):
        raise RuntimeError(f"connection failed for user with {_SENTINEL}")

    failed = sink.records[-1]
    serialized = failed.to_json()
    assert _SENTINEL not in serialized
    assert "[REDACTED]" in (failed.error_message or "")


def test_scrubbing_catches_unregistered_credential_patterns() -> None:
    message = (
        f"login failed: password={_SENTINEL}; host=db "
        f"token='{_SENTINEL}', secret: {_SENTINEL} apiKey=\"{_SENTINEL}\""
    )
    scrubbed = scrub_message(message)

    assert _SENTINEL not in scrubbed
    assert "host=db" in scrubbed


def test_scrub_message_rejects_non_string_input() -> None:
    with pytest.raises(ValueError):
        scrub_message(None)  # type: ignore[arg-type]


def test_audit_store_persists_scrubbed_records_without_secrets(tmp_path) -> None:
    store = PipelineRunAuditStore(tmp_path)
    sink = RecordingSink()
    started = datetime(2025, 3, 1, tzinfo=UTC)
    with pytest.raises(RuntimeError), pipeline_run(
        store,
        name="db.migrate",
        context=_context(),
        config_hash="sha256:" + "b" * 64,
        code_commit="sha256:" + "a" * 64,
        secrets=(_SENTINEL,),
        clock=_clock(started),
    ):
        raise RuntimeError(_SENTINEL)

    content = "\n".join(record.to_json() for record in store.records())
    assert _SENTINEL not in content
    statuses = [record.status for record in store.records()]
    assert statuses == [PipelineRunStatus.RUNNING, PipelineRunStatus.FAILED]
    assert sink.records == []
