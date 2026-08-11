"""Post-migration and post-backfill invariants for compact latest tables."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import pandas as pd

from research_v2.compact.contracts import compute_batch_id
from research_v2.compact.publisher import (
    DEFAULT_COMPACT_OBJECTS,
    CompactObjectSpec,
    CompactPublishError,
    CompactPublishSession,
    list_current_source_pointers,
)

_LINEAGE_COLUMNS: Final[frozenset[str]] = frozenset(
    {"run_id", "data_version", "created_at", "source_watermark_id", "record_hash"}
)


@dataclass(frozen=True)
class CompactObjectVerification:
    """Verification evidence for one source/compact table pair."""

    source_object: str
    target_object: str
    pointer_count: int
    current_manifest_count: int
    current_batch_count: int
    current_row_count: int
    target_has_lineage_columns: bool
    source_lineage_intact: bool
    errors: tuple[str, ...] = ()

    @property
    def ok(self) -> bool:
        return not self.errors


@dataclass(frozen=True)
class CompactVerificationReport:
    """Complete compact-storage verification report."""

    objects: tuple[CompactObjectVerification, ...]

    @property
    def ok(self) -> bool:
        return all(item.ok for item in self.objects)

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(
            f"{item.source_object}: {error}"
            for item in self.objects
            for error in item.errors
        )

    def raise_if_invalid(self) -> None:
        if not self.ok:
            raise CompactPublishError("compact verification failed: " + "; ".join(self.errors))


def verify_compact_backfill(
    session: CompactPublishSession,
    *,
    objects: tuple[CompactObjectSpec, ...] = DEFAULT_COMPACT_OBJECTS,
) -> CompactVerificationReport:
    """Check pointer coverage, batch row counts, and schema invariants."""
    reports: list[CompactObjectVerification] = []
    for spec in objects:
        pointers = list_current_source_pointers(session, spec.source_object)
        current_manifest = _current_manifests(session, spec.target_object)
        expected = {
            (pointer.logical_partition, compute_batch_id(
                spec.source_object, pointer.logical_partition, pointer.data_version
            ))
            for pointer in pointers
        }
        actual = {
            (str(row.logical_partition), int(str(row.batch_id)))
            for row in current_manifest.itertuples(index=False)
        }
        errors: list[str] = []
        missing = expected.difference(actual)
        unexpected = actual.difference(expected)
        if missing:
            errors.append(f"missing current manifests: {len(missing)}")
        if unexpected:
            errors.append(f"unexpected current manifests: {len(unexpected)}")

        target_counts = _batch_counts(session, spec.target_object)
        current_rows = current_manifest.loc[:, ["batch_id", "row_count"]].copy()
        if not current_rows.empty:
            current_rows["batch_id"] = current_rows["batch_id"].astype("int64")
            current_rows["row_count"] = current_rows["row_count"].astype("int64")
            merged = current_rows.merge(target_counts, on="batch_id", how="left")
            mismatched = merged[
                merged["target_row_count"].isna()
                | (merged["row_count"] != merged["target_row_count"])
            ]
            if not mismatched.empty:
                errors.append(f"row-count mismatches: {len(mismatched)}")

        target_schema = _schema_names(session, spec.target_object)
        source_schema = _schema_names(session, spec.source_object)
        target_has_lineage = bool(target_schema.intersection(_LINEAGE_COLUMNS))
        source_lineage_intact = _LINEAGE_COLUMNS.issubset(source_schema)
        if target_has_lineage:
            errors.append("compact target contains row-level lineage columns")
        if not source_lineage_intact:
            errors.append("existing source table lost a lineage column")
        reports.append(
            CompactObjectVerification(
                source_object=spec.source_object,
                target_object=spec.target_object,
                pointer_count=len(pointers),
                current_manifest_count=len(current_manifest),
                current_batch_count=len(
                    set(current_manifest["batch_id"].astype("int64"))
                ),
                current_row_count=int(current_manifest["row_count"].sum())
                if not current_manifest.empty
                else 0,
                target_has_lineage_columns=target_has_lineage,
                source_lineage_intact=source_lineage_intact,
                errors=tuple(errors),
            )
        )
    return CompactVerificationReport(objects=tuple(reports))


def _current_manifests(session: CompactPublishSession, target_object: str) -> pd.DataFrame:
    result = session.query(
        "select object_name, logical_partition, batch_id, source_object, "
        "source_data_version, published_at, row_count, business_hash, status "
        "from loadTable("
        f"'{session.target_database}', 'latest_manifest') "
        f"where object_name=`{target_object} and status=`PUBLISHED "
        "order by published_at desc"
    )
    if not isinstance(result, pd.DataFrame):
        raise CompactPublishError("latest manifest verification query did not return a DataFrame")
    if result.empty:
        return pd.DataFrame(columns=["logical_partition", "batch_id", "row_count"])
    required = {"logical_partition", "batch_id", "row_count", "published_at"}
    if not required.issubset(result.columns):
        raise CompactPublishError("latest manifest verification schema is invalid")
    ordered = result.copy(deep=True)
    ordered["published_at"] = pd.to_datetime(ordered["published_at"], errors="raise")
    ordered = ordered.sort_values("published_at", ascending=False, kind="mergesort")
    return ordered.drop_duplicates("logical_partition", keep="first").reset_index(drop=True)


def _batch_counts(session: CompactPublishSession, target_object: str) -> pd.DataFrame:
    result = session.query(
        "select batch_id, count(*) as target_row_count from loadTable("
        f"'{session.target_database}', '{target_object}') group by batch_id"
    )
    if not isinstance(result, pd.DataFrame) or not {"batch_id", "target_row_count"}.issubset(
        result.columns
    ):
        raise CompactPublishError("compact batch-count query returned an invalid result")
    frame = result.loc[:, ["batch_id", "target_row_count"]].copy()
    if frame.empty:
        return frame.astype({"batch_id": "int64", "target_row_count": "int64"})
    frame["batch_id"] = frame["batch_id"].astype("int64")
    frame["target_row_count"] = frame["target_row_count"].astype("int64")
    return frame


def _schema_names(session: CompactPublishSession, object_name: str) -> set[str]:
    result = session.query(
        "schema(loadTable("
        f"'{session.target_database}', '{object_name}')).colDefs"
    )
    if not isinstance(result, pd.DataFrame) or "name" not in result.columns:
        raise CompactPublishError("compact schema verification query returned an invalid result")
    return {str(value) for value in result["name"].tolist()}
