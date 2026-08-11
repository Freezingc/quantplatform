"""Database session boundaries."""

from __future__ import annotations

import re
from typing import Protocol, cast

import pandas as pd


class ReadOnlyViolation(ValueError):
    """Raised when a source-database session receives a mutation."""


class TargetDatabaseViolation(ValueError):
    """Raised when a session targets a database outside its allowlist."""


class ScriptClient(Protocol):
    """Minimal interface implemented by the DolphinDB client."""

    def run(self, script: str) -> object: ...


class UploadClient(ScriptClient, Protocol):
    """DolphinDB client capability used for DataFrame-backed V2 staging."""

    def upload(self, values: dict[str, pd.DataFrame]) -> None: ...


_YESLAB_DATABASE = "dfs://YesLab"
_V2_DATABASES = frozenset({"dfs://QuantResearchV2", "dfs://QuantResearchV2Test"})
_DATABASE_URI_PATTERN = re.compile(r"dfs://[A-Za-z0-9_]+")
_VARIABLE_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
_LITERAL_OR_COMMENT_PATTERN = re.compile(
    r"//[^\n]*|/\*.*?\*/|'(?:''|[^'])*'|\"(?:\\.|[^\"])*\"", re.DOTALL
)
_MUTATING_COMMAND_PATTERN = re.compile(
    r"\bappend!|\btableinsert\b|\bdroptable\b|\bdropdatabase\b|"
    r"\bcreatepartitionedtable\b|\bsavetable\b|\brename!|\bundef\b|"
    r"\bupdate\b|\bdelete\b|\binsert\b",
    re.IGNORECASE,
)


def _database_uris(script: str) -> set[str]:
    return set(_DATABASE_URI_PATTERN.findall(script))


def _executable_script(script: str) -> str:
    return _LITERAL_OR_COMMENT_PATTERN.sub("", script)


def _reject_empty_script(script: str) -> None:
    if not script.strip():
        raise ValueError("database script cannot be empty")


class YesLabReadOnlySession:
    """Read-only source database session."""

    def __init__(self, client: ScriptClient) -> None:
        self._client = client

    def query(self, script: str) -> object:
        _reject_empty_script(script)
        uris = _database_uris(script)
        if any(uri != _YESLAB_DATABASE for uri in uris):
            raise ReadOnlyViolation("YesLab read-only session only permits dfs://YesLab")
        if _MUTATING_COMMAND_PATTERN.search(_executable_script(script)):
            raise ReadOnlyViolation("YesLab read-only session rejects mutating scripts")
        return self._client.run(script)


class V2WriteSession:
    """Allowlisted V2 database session."""

    def __init__(self, client: ScriptClient, *, target_database: str) -> None:
        if target_database not in _V2_DATABASES:
            raise TargetDatabaseViolation(
                "V2 write session requires an approved V2 target database"
            )
        self._client = client
        self._target_database = target_database

    def query(self, script: str) -> object:
        self._validate_target_references(script)
        return self._client.run(script)

    @property
    def target_database(self) -> str:
        """Return the single V2 database this session may mutate."""
        return self._target_database

    def execute(self, script: str) -> object:
        self._validate_target_references(script)
        return self._client.run(script)

    def upload_frame(self, name: str, frame: pd.DataFrame) -> None:
        """Upload a temporary DataFrame variable for a validated V2 write only."""
        if not _VARIABLE_NAME_PATTERN.fullmatch(name):
            raise TargetDatabaseViolation("uploaded variable name is unsafe")
        if not hasattr(self._client, "upload"):
            raise TargetDatabaseViolation("underlying client does not support DataFrame upload")
        upload_client = cast(UploadClient, self._client)
        upload_client.upload({name: frame.copy(deep=True)})

    def _validate_target_references(self, script: str) -> None:
        _reject_empty_script(script)
        uris = _database_uris(script)
        if _YESLAB_DATABASE in uris:
            raise TargetDatabaseViolation("V2 write session cannot reference YesLab")
        if any(uri != self._target_database for uri in uris):
            raise TargetDatabaseViolation(
                "V2 write session script references an unapproved database"
            )
