"""Versioned DolphinDB migration discovery."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from numbers import Integral
from pathlib import Path
from typing import Protocol, cast

import pandas as pd


class MigrationError(ValueError):
    """Raised when migration files are not safe to apply."""


@dataclass(frozen=True)
class Migration:
    version: int
    name: str
    checksum: str
    script: str


class MigrationExecutionSession(Protocol):
    """Restricted session interface required by the migration runner."""

    @property
    def target_database(self) -> str: ...

    def execute(self, script: str) -> object: ...

    def query(self, script: str) -> object: ...


@dataclass(frozen=True)
class MigrationApplyResult:
    planned: tuple[Migration, ...]
    executed: tuple[Migration, ...]
    skipped: tuple[Migration, ...] = ()


_MIGRATION_FILENAME = re.compile(r"(?P<version>[0-9]{3})_(?P<name>[a-z][a-z0-9_]*).dos")


def discover_migrations(directory: Path) -> tuple[Migration, ...]:
    """Load all migration scripts in their application order."""
    if not directory.is_dir():
        raise MigrationError(f"migration directory does not exist: {directory}")

    migrations: list[Migration] = []
    versions: set[int] = set()
    for path in sorted(directory.glob("*.dos")):
        match = _MIGRATION_FILENAME.fullmatch(path.name)
        if match is None:
            raise MigrationError(f"migration filename must use NNN_name.dos: {path.name}")
        version = int(match.group("version"))
        if version in versions:
            raise MigrationError(f"duplicate migration version: {version:03d}")
        script = path.read_text(encoding="utf-8")
        if not script.strip():
            raise MigrationError(f"migration script cannot be empty: {path.name}")
        migrations.append(
            Migration(
                version=version,
                name=match.group("name"),
                checksum=hashlib.sha256(script.encode("utf-8")).hexdigest(),
                script=script,
            )
        )
        versions.add(version)
    return tuple(sorted(migrations, key=lambda migration: migration.version))


class MigrationRunner:
    """Execute discovered scripts only through an allowlisted V2 session."""

    def __init__(
        self, session: MigrationExecutionSession, *, migration_directory: Path
    ) -> None:
        self._session = session
        self._migration_directory = migration_directory

    def apply(self, *, dry_run: bool = False) -> MigrationApplyResult:
        """Apply all discovered migrations or return their dry-run plan."""
        planned = discover_migrations(self._migration_directory)
        if dry_run:
            return MigrationApplyResult(planned=planned, executed=())

        applied_checksums = self._applied_checksums()
        executed: list[Migration] = []
        skipped: list[Migration] = []
        for migration in planned:
            prior_checksum = applied_checksums.get(migration.version)
            if prior_checksum is not None:
                if prior_checksum != migration.checksum:
                    raise MigrationError(
                        f"migration checksum changed for version {migration.version:03d}"
                    )
                skipped.append(migration)
                continue
            script = f"targetDatabase = '{self._session.target_database}'\n{migration.script}"
            self._session.execute(script)
            self._session.execute(self._migration_audit_script(migration))
            executed.append(migration)
        return MigrationApplyResult(
            planned=planned, executed=tuple(executed), skipped=tuple(skipped)
        )

    def _applied_checksums(self) -> dict[int, str]:
        result = self._session.query(
            "select schema_version, checksum from loadTable("
            f"'{self._session.target_database}', 'schema_migration')"
        )
        if not isinstance(result, pd.DataFrame):
            raise MigrationError("schema_migration query did not return a DataFrame")
        required_columns = {"schema_version", "checksum"}
        if not required_columns.issubset(result.columns):
            raise MigrationError("schema_migration query returned an invalid schema")

        applied: dict[int, str] = {}
        for row in result.loc[:, ["schema_version", "checksum"]].to_dict("records"):
            values = cast(dict[str, object], row)
            version = values["schema_version"]
            checksum = values["checksum"]
            if not isinstance(version, Integral) or not isinstance(checksum, str):
                raise MigrationError("schema_migration contains invalid values")
            normalized_version = int(version)
            if normalized_version in applied and applied[normalized_version] != checksum:
                raise MigrationError(
                    f"schema_migration has conflicting version {normalized_version:03d}"
                )
            applied[normalized_version] = checksum
        return applied

    def _migration_audit_script(self, migration: Migration) -> str:
        return (
            "migrationAudit = table("
            f"[{migration.version}] as schema_version, "
            f"['{migration.name}'] as name, "
            f"['{migration.checksum}'] as checksum, "
            "[now()] as applied_at, [`qr_v2] as applied_by, [uuid()] as run_id, "
            f"['{migration.checksum}'] as data_version, [now()] as created_at, "
            "['INTERNAL'] as source_watermark_id, "
            f"['{migration.checksum}'] as record_hash)\n"
            "tableInsert{loadTable("
            f"'{self._session.target_database}', 'schema_migration')"
            "}(migrationAudit)"
        )
