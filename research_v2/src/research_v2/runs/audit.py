"""Append-only pipeline run audit with secret-scrubbed failure evidence."""

from __future__ import annotations

import enum
import json
import re
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol

from research_v2.runs.context import RunContext

_REDACTED = "[REDACTED]"
_MAXIMUM_ERROR_MESSAGE_LENGTH = 1000
_SECRET_PATTERN = re.compile(
    r"(?P<key>password|passwd|token|secret|api_?key|access_?key)"
    r"(?P<separator>\s*[=:]\s*)"
    r"(?P<value>'[^']*'|\"[^\"]*\"|[^\s;,\"']+)",
    re.IGNORECASE,
)


class PipelineRunStatus(enum.StrEnum):
    RUNNING = "RUNNING"
    SUCCESS = "SUCCESS"
    FAILED = "FAILED"


def scrub_message(message: object, *, secrets: Iterable[str] = ()) -> str:
    """Redact registered secret values and credential-looking key/value pairs."""
    if not isinstance(message, str):
        raise ValueError("audit messages must be strings")
    scrubbed = message
    for secret in secrets:
        if not isinstance(secret, str):
            raise ValueError("secrets must be strings")
        if secret:
            scrubbed = scrubbed.replace(secret, _REDACTED)
    return _SECRET_PATTERN.sub(
        lambda match: f"{match['key']}{match['separator']}{_REDACTED}",
        scrubbed,
    )


@dataclass(frozen=True)
class PipelineRunRecord:
    """One immutable audit state for a pipeline run, mirroring the control table."""

    run_id: str
    pipeline_name: str
    status: PipelineRunStatus
    data_version: str
    config_hash: str
    code_commit: str
    started_at: datetime
    finished_at: datetime | None = None
    error_class: str | None = None
    error_message: str | None = None

    def to_json(self) -> str:
        payload: dict[str, Any] = {
            "run_id": self.run_id,
            "pipeline_name": self.pipeline_name,
            "status": self.status.value,
            "data_version": self.data_version,
            "config_hash": self.config_hash,
            "code_commit": self.code_commit,
            "started_at": self.started_at.isoformat(),
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "error_class": self.error_class,
            "error_message": self.error_message,
        }
        return json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


class PipelineRunSink(Protocol):
    """Destination capable of persisting pipeline run audit records."""

    def write(self, record: PipelineRunRecord) -> None: ...


class PipelineRunAuditStore:
    """Append-only JSON-lines audit store for pipeline run records."""

    def __init__(self, root: Path) -> None:
        self._path = root / "pipeline_runs.jsonl"
        self._path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, record: PipelineRunRecord) -> None:
        with self._path.open("a", encoding="utf-8") as handle:
            handle.write(record.to_json() + "\n")

    def records(self) -> tuple[PipelineRunRecord, ...]:
        if not self._path.exists():
            return ()
        records: list[PipelineRunRecord] = []
        for line in self._path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            payload = json.loads(line)
            finished = payload.get("finished_at")
            records.append(
                PipelineRunRecord(
                    run_id=str(payload["run_id"]),
                    pipeline_name=str(payload["pipeline_name"]),
                    status=PipelineRunStatus(payload["status"]),
                    data_version=str(payload["data_version"]),
                    config_hash=str(payload["config_hash"]),
                    code_commit=str(payload["code_commit"]),
                    started_at=datetime.fromisoformat(payload["started_at"]),
                    finished_at=datetime.fromisoformat(finished) if finished else None,
                    error_class=payload.get("error_class"),
                    error_message=payload.get("error_message"),
                )
            )
        return tuple(records)


@contextmanager
def pipeline_run(
    sink: PipelineRunSink,
    *,
    name: str,
    context: RunContext,
    config_hash: str,
    code_commit: str,
    secrets: Iterable[str] = (),
    clock: Callable[[], datetime] | None = None,
) -> Iterator[PipelineRunRecord]:
    """Audit RUNNING/SUCCESS/FAILED transitions without leaking secrets."""
    if not name or not name.strip():
        raise ValueError("pipeline name cannot be empty")
    if not config_hash or not config_hash.strip():
        raise ValueError("config_hash cannot be empty")
    if not code_commit or not code_commit.strip():
        raise ValueError("code_commit cannot be empty")
    registered_secrets = tuple(secrets)
    now = clock or (lambda: datetime.now(UTC))
    running = PipelineRunRecord(
        run_id=context.run_id,
        pipeline_name=name,
        status=PipelineRunStatus.RUNNING,
        data_version=context.data_version,
        config_hash=config_hash,
        code_commit=code_commit,
        started_at=now(),
    )
    sink.write(running)
    try:
        yield running
    except Exception as error:
        scrubbed = scrub_message(str(error), secrets=registered_secrets)
        failed = replace(
            running,
            status=PipelineRunStatus.FAILED,
            finished_at=now(),
            error_class=scrub_message(type(error).__name__, secrets=registered_secrets),
            error_message=scrubbed[:_MAXIMUM_ERROR_MESSAGE_LENGTH],
        )
        sink.write(failed)
        raise
    sink.write(replace(running, status=PipelineRunStatus.SUCCESS, finished_at=now()))
