"""Validated two-phase publishing primitives."""

from __future__ import annotations

import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum
from hashlib import sha256
from typing import Protocol
from uuid import UUID

import pandas as pd

from research_v2.quality.engine import CheckResult, can_publish


class PublishValidationError(ValueError):
    """Raised when staged data cannot be made visible."""


class PublishState(StrEnum):
    CREATED = "CREATED"
    VALIDATED = "VALIDATED"
    PUBLISHED = "PUBLISHED"
    REJECTED = "REJECTED"


class V2PublishSession(Protocol):
    """Minimal V2 write surface required for staged-version publishing."""

    @property
    def target_database(self) -> str: ...

    def upload_frame(self, name: str, frame: pd.DataFrame) -> None: ...

    def execute(self, script: str) -> object: ...

    def query(self, script: str) -> object: ...


_TABLE_NAME_PATTERN = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


def _quoted(value: str) -> str:
    return value.replace("'", "''")


@dataclass
class StagedBatch:
    object_name: str
    logical_partition: str
    data_version: str
    frame: pd.DataFrame
    business_keys: tuple[str, ...]
    partition_date_column: str | None = None
    state: PublishState = PublishState.CREATED

    @classmethod
    def create(
        cls,
        *,
        object_name: str,
        logical_partition: str,
        data_version: str,
        frame: pd.DataFrame,
        business_keys: tuple[str, ...],
        partition_date_column: str | None = None,
    ) -> StagedBatch:
        if not object_name or not logical_partition or not data_version:
            raise PublishValidationError("publish identity fields cannot be empty")
        if not business_keys:
            raise PublishValidationError("at least one business key is required")
        return cls(
            object_name=object_name,
            logical_partition=logical_partition,
            data_version=data_version,
            frame=frame.copy(deep=True),
            business_keys=business_keys,
            partition_date_column=partition_date_column,
        )

    def split_by_month(self) -> list[StagedBatch]:
        """Return independently publishable calendar-month batches."""
        if not self.partition_date_column:
            raise PublishValidationError("partition_date_column is required for monthly splitting")
        if self.partition_date_column not in self.frame.columns:
            raise PublishValidationError(
                f"partition date column is missing: {self.partition_date_column}"
            )
        dates = pd.to_datetime(self.frame[self.partition_date_column], errors="raise")
        if dates.isna().any():
            raise PublishValidationError("partition date column cannot contain nulls")
        monthly_batches: list[StagedBatch] = []
        groups = dates.groupby(dates.dt.strftime("%Y-%m"), sort=True).groups
        for partition, indexes in groups.items():
            monthly_batches.append(
                StagedBatch.create(
                    object_name=self.object_name,
                    logical_partition=str(partition),
                    data_version=self.data_version,
                    frame=self.frame.loc[indexes].reset_index(drop=True),
                    business_keys=self.business_keys,
                    partition_date_column=self.partition_date_column,
                )
            )
        return monthly_batches

    def validate(self) -> None:
        if self.state is not PublishState.CREATED:
            raise PublishValidationError("only newly created batches can be validated")
        if self.partition_date_column:
            if self.partition_date_column not in self.frame.columns:
                self.state = PublishState.REJECTED
                raise PublishValidationError(
                    f"partition date column is missing: {self.partition_date_column}"
                )
            partition_dates = pd.to_datetime(
                self.frame[self.partition_date_column], errors="raise"
            )
            if partition_dates.isna().any():
                self.state = PublishState.REJECTED
                raise PublishValidationError("partition date column cannot contain nulls")
            partitions = set(partition_dates.dt.strftime("%Y-%m"))
            if partitions != {self.logical_partition}:
                self.state = PublishState.REJECTED
                raise PublishValidationError("batch dates fall outside logical partition")
        missing_columns = set(self.business_keys).difference(self.frame.columns)
        if missing_columns:
            self.state = PublishState.REJECTED
            raise PublishValidationError(f"missing business keys: {sorted(missing_columns)}")
        if self.frame.empty:
            self.state = PublishState.REJECTED
            raise PublishValidationError("staged batch cannot be empty")
        if self.frame.loc[:, list(self.business_keys)].isna().any().any():
            self.state = PublishState.REJECTED
            raise PublishValidationError("business keys cannot be null")
        if self.frame.duplicated(subset=list(self.business_keys), keep=False).any():
            self.state = PublishState.REJECTED
            raise PublishValidationError("duplicate business keys in staged batch")
        self.state = PublishState.VALIDATED

    def publish(self) -> None:
        if self.state is not PublishState.VALIDATED:
            raise PublishValidationError("batch must be validated before publish")
        self.state = PublishState.PUBLISHED


class V2BatchPublisher:
    """Append a validated version and atomically advance its visibility pointer last."""

    def __init__(self, session: V2PublishSession) -> None:
        self._session = session

    def publish(
        self,
        batch: StagedBatch,
        quality_results: list[CheckResult],
        *,
        required_blocking_checks: frozenset[str] = frozenset(),
    ) -> None:
        """Publish one prevalidated, DQ-approved batch to its approved V2 target."""
        if not _TABLE_NAME_PATTERN.fullmatch(batch.object_name):
            raise PublishValidationError("object_name must be a simple table identifier")
        previous_version = self._previous_version(batch)
        if previous_version == batch.data_version:
            raise PublishValidationError("data_version is already current for this partition")
        batch.validate()
        if not can_publish(quality_results, required_blocking_checks=required_blocking_checks):
            batch.state = PublishState.REJECTED
            raise PublishValidationError("data-quality gate rejected staged batch")
        stage_name = self._stage_name(batch)
        target_types = self._target_types(batch)
        self._session.upload_frame(stage_name, self._uploadable_frame(batch.frame))
        self._session.execute(self._insert_script(batch, stage_name, target_types))
        self._session.execute(self._pointer_script(batch, previous_version))
        batch.publish()

    def publish_chunks(
        self,
        batches: list[StagedBatch],
        quality_results: list[CheckResult],
        *,
        required_blocking_checks: frozenset[str] = frozenset(),
    ) -> None:
        """Append validated chunks under one version and advance its pointer once."""
        if not batches:
            raise PublishValidationError("at least one chunk is required")
        first = batches[0]
        identity = (
            first.object_name,
            first.logical_partition,
            first.data_version,
            first.business_keys,
        )
        if any(
            (batch.object_name, batch.logical_partition, batch.data_version, batch.business_keys)
            != identity
            for batch in batches[1:]
        ):
            raise PublishValidationError("all chunks must share one publish identity")
        previous_version = self._previous_version(first)
        if previous_version == first.data_version:
            raise PublishValidationError("data_version is already current for this partition")
        if not can_publish(quality_results, required_blocking_checks=required_blocking_checks):
            raise PublishValidationError("data-quality gate rejected staged batch")
        seen_keys: set[tuple[object, ...]] = set()
        target_types = self._target_types(first)
        for index, batch in enumerate(batches):
            batch.validate()
            keys = [
                tuple(row)
                for row in batch.frame.loc[:, list(batch.business_keys)].itertuples(
                    index=False,
                    name=None,
                )
            ]
            if any(key in seen_keys for key in keys):
                raise PublishValidationError("duplicate business keys across chunks")
            seen_keys.update(keys)
            stage_name = f"{self._stage_name(batch)}_{index}"
            self._session.upload_frame(stage_name, self._uploadable_frame(batch.frame))
            self._session.execute(self._insert_script(batch, stage_name, target_types))
            self._session.execute(self._cleanup_stage_script(batch, stage_name, target_types))
        self._session.execute(self._pointer_script(first, previous_version))
        for batch in batches:
            batch.publish()

    def publish_chunk_stream(
        self,
        batches: Iterable[StagedBatch],
        quality_results: list[CheckResult],
        *,
        required_blocking_checks: frozenset[str] = frozenset(),
        allow_current_repair: bool = False,
    ) -> None:
        """Publish many chunks without retaining the complete frame in memory.

        Chunks are inserted before the single visibility-pointer advance. If a
        later chunk fails, the already inserted rows remain invisible because the
        pointer is not written; a retry can safely use the same ``data_version``.
        """
        iterator = iter(batches)
        try:
            first = next(iterator)
        except StopIteration as error:
            raise PublishValidationError("at least one chunk is required") from error
        identity = (
            first.object_name,
            first.logical_partition,
            first.data_version,
            first.business_keys,
        )
        previous_version = self._previous_version(first)
        existing_rows = self._current_version_row_count(first)
        advance_pointer = True
        if previous_version == first.data_version:
            if not allow_current_repair:
                raise PublishValidationError("data_version is already current for this partition")
            if existing_rows != 0:
                raise PublishValidationError(
                    "current-version repair requires an empty target data_version"
                )
            advance_pointer = False
        elif existing_rows != 0:
            raise PublishValidationError(
                "target contains orphan rows for data_version; use a fresh retry nonce"
            )
        if not can_publish(quality_results, required_blocking_checks=required_blocking_checks):
            raise PublishValidationError("data-quality gate rejected staged batch")
        target_types = self._target_types(first)
        seen_keys: set[tuple[object, ...]] = set()
        for index, batch in enumerate(_prepend(first, iterator)):
            if (
                batch.object_name,
                batch.logical_partition,
                batch.data_version,
                batch.business_keys,
            ) != identity:
                raise PublishValidationError("all chunks must share one publish identity")
            batch.validate()
            keys = [
                tuple(row)
                for row in batch.frame.loc[:, list(batch.business_keys)].itertuples(
                    index=False,
                    name=None,
                )
            ]
            if any(key in seen_keys for key in keys):
                raise PublishValidationError("duplicate business keys across chunks")
            seen_keys.update(keys)
            stage_name = f"{self._stage_name(batch)}_{index}"
            self._session.upload_frame(stage_name, self._uploadable_frame(batch.frame))
            self._session.execute(self._insert_script(batch, stage_name, target_types))
            self._session.execute(self._cleanup_stage_script(batch, stage_name, target_types))
        if advance_pointer:
            self._session.execute(self._pointer_script(first, previous_version))

    @staticmethod
    def _stream_identity(batch: StagedBatch) -> tuple[str, str, str, tuple[str, ...]]:
        return (
            batch.object_name,
            batch.logical_partition,
            batch.data_version,
            batch.business_keys,
        )

    @staticmethod
    def _uploadable_frame(frame: pd.DataFrame) -> pd.DataFrame:
        staged = frame.copy(deep=True)
        if "run_id" in staged.columns:
            staged["run_id"] = staged["run_id"].map(
                lambda value: str(value) if isinstance(value, UUID) else value
            )
        return staged

    def _insert_script(
        self,
        batch: StagedBatch,
        stage_name: str,
        target_types: dict[str, str],
    ) -> str:
        castable_types = {
            "BOOL": "bool",
            "BYTE": "byte",
            "SHORT": "short",
            "INT": "int",
            "LONG": "long",
            "FLOAT": "float",
            "DOUBLE": "double",
            "UUID": "uuid",
            "DATE": "date",
            "TIMESTAMP": "timestamp",
        }
        needs_typed_stage = any(
            target_types.get(column) in castable_types for column in batch.frame.columns
        )
        if not needs_typed_stage:
            return (
                "tableInsert{loadTable("
                f"'{self._session.target_database}', '{batch.object_name}')"
                f"}}({stage_name})"
            )
        selected_columns = [
            f"{castable_types[target_types[column]]}({column}) as {column}"
            if target_types.get(column) in castable_types
            else column
            for column in batch.frame.columns
        ]
        typed_stage = f"{stage_name}_typed"
        return (
            f"{typed_stage} = select {', '.join(selected_columns)} from {stage_name}\n"
            "tableInsert{loadTable("
            f"'{self._session.target_database}', '{batch.object_name}')"
            f"}}({typed_stage})"
        )

    def _cleanup_stage_script(
        self,
        batch: StagedBatch,
        stage_name: str,
        target_types: dict[str, str],
    ) -> str:
        castable_types = {
            "BOOL",
            "BYTE",
            "SHORT",
            "INT",
            "LONG",
            "FLOAT",
            "DOUBLE",
            "UUID",
            "DATE",
            "TIMESTAMP",
        }
        has_typed_stage = any(
            target_types.get(column) in castable_types for column in batch.frame.columns
        )
        cleanup = [f"undef(`{stage_name}, VAR)"]
        if has_typed_stage:
            cleanup.append(f"undef(`{stage_name}_typed, VAR)")
        return "\n".join(cleanup)

    def _current_version_row_count(self, batch: StagedBatch) -> int:
        date_predicate = _logical_partition_predicate(batch)
        result = self._session.query(
            "select count(*) as row_count from loadTable("
            f"'{self._session.target_database}', '{batch.object_name}') "
            f"where data_version='{_quoted(batch.data_version)}'{date_predicate}"
        )
        if not isinstance(result, pd.DataFrame) or "row_count" not in result.columns:
            raise PublishValidationError(
                "current-version row-count query returned an invalid result"
            )
        if result.empty:
            return 0
        return int(result.iloc[0]["row_count"])

    def _target_types(self, batch: StagedBatch) -> dict[str, str]:
        result = self._session.query(
            "schema(loadTable("
            f"'{self._session.target_database}', '{batch.object_name}')).colDefs"
        )
        if not isinstance(result, pd.DataFrame) or not {"name", "typeString"}.issubset(
            result.columns
        ):
            raise PublishValidationError("target table schema query returned an invalid result")
        return {
            str(row["name"]): str(row["typeString"])
            for row in result.loc[:, ["name", "typeString"]].to_dict("records")
        }

    def _previous_version(self, batch: StagedBatch) -> str:
        result = self._session.query(
            "select top 1 data_version from loadTable("
            f"'{self._session.target_database}', 'published_version') "
            f"where object_name=`{batch.object_name}, "
            f"logical_partition='{_quoted(batch.logical_partition)}' "
            "order by published_at desc"
        )
        if not isinstance(result, pd.DataFrame):
            raise PublishValidationError("published-version query did not return a DataFrame")
        if result.empty:
            return ""
        value = result.iloc[0]["data_version"]
        return str(value)

    def _pointer_script(self, batch: StagedBatch, previous_version: str) -> str:
        return (
            "publishPointer = table("
            f"[`{batch.object_name}] as object_name, "
            f"['{_quoted(batch.logical_partition)}'] as logical_partition, "
            f"['{_quoted(batch.data_version)}'] as data_version, "
            f"['{_quoted(previous_version)}'] as previous_data_version, "
            "[now()] as published_at, [uuid()] as run_id)\n"
            "tableInsert{loadTable("
            f"'{self._session.target_database}', 'published_version')"
            "}(publishPointer)"
        )

    @staticmethod
    def _stage_name(batch: StagedBatch) -> str:
        identity = f"{batch.object_name}|{batch.logical_partition}|{batch.data_version}"
        return f"stage_{sha256(identity.encode()).hexdigest()[:16]}"


def _prepend(first: StagedBatch, rest: Iterator[StagedBatch]) -> Iterator[StagedBatch]:
    """Yield one already-read chunk followed by a lazy iterator."""
    yield first
    yield from rest


def _logical_partition_predicate(batch: StagedBatch) -> str:
    """Bound orphan-row checks to a calendar month when the batch declares a date column."""
    column = batch.partition_date_column
    if not column or _TABLE_NAME_PATTERN.fullmatch(column) is None:
        return ""
    try:
        period = pd.Period(batch.logical_partition, freq="M")
    except (TypeError, ValueError):
        return ""
    if str(period) != batch.logical_partition:
        return ""
    start = period.start_time.strftime("%Y.%m.%d")
    end = period.end_time.strftime("%Y.%m.%d")
    return f" and {column} between date('{start}') and date('{end}')"
