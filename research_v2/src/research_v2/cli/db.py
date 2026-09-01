"""Database migration, status, and source-inspection commands with fail-closed guards."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from datetime import date
from pathlib import Path

import pandas as pd
import typer

from research_v2.config.loader import load_settings
from research_v2.config.models import DatabaseSettings, EnvironmentName
from research_v2.db.migrations import MigrationRunner, discover_migrations
from research_v2.db.session import ScriptClient, V2WriteSession
from research_v2.runs.audit import PipelineRunAuditStore, pipeline_run
from research_v2.runs.context import RunContext

MIGRATION_DIRECTORY = Path(__file__).resolve().parents[3] / "migrations"
_CONFIG_ROOT = Path(__file__).resolve().parents[3] / "configs"
PRODUCTION_CONFIRM_TARGET = "QuantResearchV2"


class BackendNotConfigured(RuntimeError):
    """Raised when a live DolphinDB client is required but not wired."""


def _default_client(settings: DatabaseSettings) -> ScriptClient:
    raise BackendNotConfigured(
        "live DolphinDB runtime backend is not configured for this environment"
    )


client_factory: Callable[[DatabaseSettings], ScriptClient] = _default_client


_ENVIRONMENTS = frozenset({"dev", "test", "production"})


def _validated_environment(environment: str) -> EnvironmentName:
    if environment == "dev":
        return "dev"
    if environment == "test":
        return "test"
    if environment == "production":
        return "production"
    raise ValueError("environment must be one of dev, test, production")


def _require_production_confirmation(environment: str, confirm_target: str | None) -> None:
    if environment == "production" and confirm_target != PRODUCTION_CONFIRM_TARGET:
        raise PermissionError(
            "production migration requires --confirm-target QuantResearchV2"
        )


def _config_hash(environment: str, target_database: str) -> str:
    payload = json.dumps(
        {"environment": environment, "target_database": target_database},
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return "sha256:" + hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _records(result: object) -> object:
    if isinstance(result, pd.DataFrame):
        return result.to_dict("records")
    return result


def migrate(environment: str, dry_run: bool, confirm_target: str | None) -> None:
    """Apply V2 schema migrations through the allowlisted write session."""
    resolved = _validated_environment(environment)
    settings = load_settings(resolved, config_root=_CONFIG_ROOT)
    target = settings.dolphindb.target_database
    _require_production_confirmation(environment, confirm_target)
    migrations = discover_migrations(MIGRATION_DIRECTORY)
    plan_summary = {
        "command": "db.migrate",
        "environment": environment,
        "target_database": target,
        "dry_run": dry_run,
        "planned": [
            {"version": migration.version, "name": migration.name}
            for migration in migrations
        ],
    }
    if dry_run:
        typer.echo(json.dumps(plan_summary, ensure_ascii=True, indent=2, sort_keys=True))
        return
    client = client_factory(settings.dolphindb)
    session = V2WriteSession(client, target_database=target)
    runner = MigrationRunner(session, migration_directory=MIGRATION_DIRECTORY)
    audit_store = PipelineRunAuditStore(settings.artifact_root / "audit")
    context = RunContext.build(
        config={"environment": environment, "target_database": target},
        watermarks=(),
        code_commit=f"cli:db.migrate:{environment}",
        schema_version=str(migrations[-1].version).zfill(3),
    )
    with pipeline_run(
        audit_store,
        name="db.migrate",
        context=context,
        config_hash=_config_hash(environment, target),
        code_commit=f"cli:db.migrate:{environment}",
        secrets=(settings.dolphindb.password.get_secret_value(),),
    ):
        result = runner.apply()
    typer.echo(
        json.dumps(
            {
                **plan_summary,
                "executed": [migration.version for migration in result.executed],
                "skipped": [migration.version for migration in result.skipped],
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
        )
    )


def status(environment: str) -> None:
    """Report applied schema migration versions for one V2 environment."""
    resolved = _validated_environment(environment)
    settings = load_settings(resolved, config_root=_CONFIG_ROOT)
    client = client_factory(settings.dolphindb)
    session = V2WriteSession(client, target_database=settings.dolphindb.target_database)
    result = session.query(
        "select schema_version, name, checksum, applied_at from loadTable("
        f"'{settings.dolphindb.target_database}', 'schema_migration')"
    )
    typer.echo(
        json.dumps(
            {
                "command": "db.status",
                "environment": environment,
                "target_database": settings.dolphindb.target_database,
                "schema_migration": _records(result),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )


def source_inspect(environment: str, as_of: date) -> None:
    """Report captured source watermarks up to one observation date."""
    resolved = _validated_environment(environment)
    settings = load_settings(resolved, config_root=_CONFIG_ROOT)
    client = client_factory(settings.dolphindb)
    session = V2WriteSession(client, target_database=settings.dolphindb.target_database)
    result = session.query(
        "select * from loadTable("
        f"'{settings.dolphindb.target_database}', 'source_watermark') "
        f"where captured_at <= {as_of.isoformat()}"
    )
    typer.echo(
        json.dumps(
            {
                "command": "source.inspect",
                "environment": environment,
                "as_of": as_of.isoformat(),
                "source_watermark": _records(result),
            },
            ensure_ascii=True,
            indent=2,
            sort_keys=True,
            default=str,
        )
    )
