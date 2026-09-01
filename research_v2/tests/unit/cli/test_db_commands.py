from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from typer.testing import CliRunner

import research_v2.cli.db as db_commands
from research_v2.cli.app import app


class FakeScriptClient:
    def __init__(self, query_results: dict[str, object] | None = None) -> None:
        self.executed: list[str] = []
        self.query_results = query_results or {}

    def run(self, script: str) -> object:
        self.executed.append(script)
        for key, result in self.query_results.items():
            if key in script:
                return result
        return None


@pytest.fixture(autouse=True)
def _database_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QRV2_DDB_HOST", "127.0.0.1")
    monkeypatch.setenv("QRV2_DDB_USERNAME", "unit")
    monkeypatch.setenv("QRV2_DDB_PASSWORD", "sentinel-database-password-123")


def test_db_migrate_dry_run_prints_plan_and_boundaries() -> None:
    result = CliRunner().invoke(app, ["db", "migrate", "--env", "test", "--dry-run"])

    assert result.exit_code == 0
    plan = json.loads(result.stdout)
    assert plan["environment"] == "test"
    assert plan["target_database"] == "dfs://QuantResearchV2Test"
    assert plan["dry_run"] is True
    versions = [item["version"] for item in plan["planned"]]
    assert versions[0] == 1 and len(versions) >= 13


def test_production_migrate_requires_confirm_target() -> None:
    result = CliRunner().invoke(app, ["db", "migrate", "--env", "production", "--dry-run"])

    assert result.exit_code != 0
    assert "--confirm-target" in result.stdout


def test_production_dry_run_succeeds_with_confirm_target() -> None:
    result = CliRunner().invoke(
        app,
        [
            "db",
            "migrate",
            "--env",
            "production",
            "--dry-run",
            "--confirm-target",
            "QuantResearchV2",
        ],
    )

    assert result.exit_code == 0
    plan = json.loads(result.stdout)
    assert plan["target_database"] == "dfs://QuantResearchV2"


def test_migrate_without_backend_fails_closed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    result = CliRunner().invoke(app, ["db", "migrate", "--env", "test"])

    assert result.exit_code != 0
    assert "runtime backend" in result.stdout


def test_migrate_executes_through_session_and_audits_without_secrets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    empty_schema = pd.DataFrame(
        {"schema_version": pd.Series(dtype="int64"), "checksum": pd.Series(dtype="object")}
    )
    client = FakeScriptClient({"schema_migration": empty_schema})
    monkeypatch.setattr(db_commands, "client_factory", lambda settings: client)

    result = CliRunner().invoke(app, ["db", "migrate", "--env", "test"])

    assert result.exit_code == 0, result.stdout
    summary = json.loads(result.stdout)
    assert summary["executed"][0] == 1
    assert len(summary["executed"]) >= 13
    assert "sentinel-database-password-123" not in result.stdout

    audit_path = tmp_path / "artifacts" / "audit" / "pipeline_runs.jsonl"
    audit_content = audit_path.read_text(encoding="utf-8")
    assert "RUNNING" in audit_content and "SUCCESS" in audit_content
    assert "sentinel-database-password-123" not in audit_content


def test_migrate_failure_is_audited_without_leaking_credentials(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)

    class ExplodingClient:
        def run(self, script: str) -> object:
            raise ConnectionError("login failed password=sentinel-database-password-123")

    monkeypatch.setattr(db_commands, "client_factory", lambda settings: ExplodingClient())

    result = CliRunner().invoke(app, ["db", "migrate", "--env", "test"])

    assert result.exit_code != 0
    audit_content = (tmp_path / "artifacts" / "audit" / "pipeline_runs.jsonl").read_text(
        encoding="utf-8"
    )
    assert "FAILED" in audit_content
    assert "sentinel-database-password-123" not in audit_content
    assert "sentinel-database-password-123" not in result.stdout


def test_db_status_reports_applied_migrations(monkeypatch: pytest.MonkeyPatch) -> None:
    applied = pd.DataFrame(
        {
            "schema_version": [1, 2],
            "name": ["control_tables", "pit_tables"],
            "checksum": ["a" * 64, "b" * 64],
            "applied_at": ["2025-01-01T00:00:00", "2025-01-01T00:01:00"],
        }
    )
    client = FakeScriptClient({"schema_migration": applied})
    monkeypatch.setattr(db_commands, "client_factory", lambda settings: client)

    result = CliRunner().invoke(app, ["db", "status", "--env", "test"])

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["target_database"] == "dfs://QuantResearchV2Test"
    assert [row["schema_version"] for row in payload["schema_migration"]] == [1, 2]


def test_source_inspect_queries_watermarks_as_of(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    client = FakeScriptClient({"source_watermark": pd.DataFrame({"table_name": ["t"]})})
    monkeypatch.setattr(db_commands, "client_factory", lambda settings: client)

    result = CliRunner().invoke(
        app, ["source", "inspect", "--env", "test", "--as-of", "2025-06-30"]
    )

    assert result.exit_code == 0, result.stdout
    payload = json.loads(result.stdout)
    assert payload["as_of"] == "2025-06-30"
    assert any("2025-06-30" in script for script in client.executed)
