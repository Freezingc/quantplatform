from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd

from research_v2.compact.contracts import CompactManifestStatus
from research_v2.compact.materializer import build_compact_materialization, compact_business_columns
from research_v2.compact.publisher import (
    CompactObjectSpec,
    SourcePointer,
    list_current_source_pointers,
    publish_compact_materialization,
    read_latest_partition,
)


class RecordingCompactSession:
    target_database = "dfs://QuantResearchV2Test"

    def __init__(self, *, current_manifest: pd.DataFrame | None = None) -> None:
        self.current_manifest = current_manifest
        self.uploads: list[tuple[str, pd.DataFrame]] = []
        self.scripts: list[str] = []
        self.queries: list[str] = []

    def upload_frame(self, name: str, frame: pd.DataFrame) -> None:
        self.uploads.append((name, frame.copy(deep=True)))

    def execute(self, script: str) -> object:
        self.scripts.append(script)
        return None

    def query(self, script: str) -> object:
        self.queries.append(script)
        if "published_version" in script:
            return pd.DataFrame(
                {
                    "object_name": ["label_daily", "label_daily"],
                    "logical_partition": ["2025-01", "2025-01"],
                    "data_version": ["newer", "older"],
                    "published_at": pd.to_datetime(
                        ["2026-08-02 10:00:00", "2026-08-02 09:00:00"], utc=True
                    ),
                }
            )
        if "latest_manifest" in script:
            return self.current_manifest if self.current_manifest is not None else pd.DataFrame()
        if script.startswith("schema("):
            return pd.DataFrame(
                {
                    "name": ["batch_id", "signal_date", "ts_code", "label_id", "label_version"],
                    "typeString": ["LONG", "DATE", "SYMBOL", "SYMBOL", "INT"],
                }
            )
        return pd.DataFrame(
            {
                "batch_id": [123],
                "signal_date": pd.to_datetime(["2025-01-02"]),
                "ts_code": ["000001.SZ"],
            }
        )


def _materialization():
    source = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2025-01-02"]),
            "ts_code": ["000001.SZ"],
            "label_id": ["forward_open_5d"],
            "label_version": [1],
            "entry_date": pd.to_datetime(["2025-01-03"]),
            "exit_date": pd.to_datetime(["2025-01-10"]),
            "label_available_date": pd.to_datetime(["2025-01-10"]),
            "raw_return": [0.1],
            "rank_pct": [0.5],
            "winsorized_return": [0.1],
            "zscore_return": [0.0],
            "neutral_residual": [0.0],
            "label_valid": [True],
            "invalid_reason": [""],
            "entry_realized_tradable": [True],
            "data_version": ["newer"],
        }
    )
    return build_compact_materialization(
        source,
        object_name="label_daily",
        source_data_version="newer",
        logical_partition="2025-01",
        business_columns=compact_business_columns("label_daily"),
        business_keys=("signal_date", "ts_code", "label_id", "label_version"),
        partition_date_column="signal_date",
        published_at=datetime(2026, 8, 2, tzinfo=UTC),
    )


def test_current_source_pointers_keep_only_newest_version_per_partition() -> None:
    session = RecordingCompactSession()

    pointers = list_current_source_pointers(session, "label_daily")

    assert pointers == (
        SourcePointer(
            source_object="label_daily",
            logical_partition="2025-01",
            data_version="newer",
            published_at=pd.Timestamp("2026-08-02 10:00:00", tz="UTC"),
        ),
    )


def test_compact_publisher_writes_rows_before_manifest_and_never_published_version() -> None:
    session = RecordingCompactSession()

    result = publish_compact_materialization(session, _materialization())

    assert result.status == "published"
    assert len(session.uploads) == 1
    assert len(session.scripts) == 3
    assert "label_daily_latest" in session.scripts[0]
    assert "int(label_version) as label_version" in session.scripts[0]
    assert "undef(" in session.scripts[1]
    assert "latest_manifest" in session.scripts[2]
    assert "published_version" not in "\n".join(session.scripts)


def test_compact_publisher_is_idempotent_for_matching_published_manifest() -> None:
    materialization = _materialization()
    manifest = materialization.manifest
    current = pd.DataFrame(
        {
            "object_name": [manifest.object_name],
            "logical_partition": [manifest.logical_partition],
            "batch_id": [manifest.batch_id],
            "source_object": [manifest.source_object],
            "source_data_version": [manifest.source_data_version],
            "published_at": [manifest.published_at],
            "row_count": [manifest.row_count],
            "business_hash": [manifest.business_hash],
            "status": [CompactManifestStatus.PUBLISHED.value],
        }
    )
    session = RecordingCompactSession(current_manifest=current)

    result = publish_compact_materialization(session, materialization)

    assert result.status == "skipped"
    assert session.uploads == []
    assert session.scripts == []


def test_read_latest_partition_uses_manifest_batch_id() -> None:
    materialization = _materialization()
    manifest = materialization.manifest
    current = pd.DataFrame(
        {
            "object_name": [manifest.object_name],
            "logical_partition": [manifest.logical_partition],
            "batch_id": [manifest.batch_id],
            "source_object": [manifest.source_object],
            "source_data_version": [manifest.source_data_version],
            "published_at": [manifest.published_at],
            "row_count": [manifest.row_count],
            "business_hash": [manifest.business_hash],
            "status": [CompactManifestStatus.PUBLISHED.value],
        }
    )
    session = RecordingCompactSession(current_manifest=current)

    frame = read_latest_partition(
        session,
        object_name="label_daily_latest",
        logical_partition="2025-01",
        columns=("batch_id", "ts_code"),
    )

    assert list(frame.columns) == ["batch_id", "signal_date", "ts_code"]
    assert f"where batch_id={manifest.batch_id}" in session.queries[-1]


def test_read_latest_partition_can_filter_factor_identity_without_selecting_all_factors() -> None:
    materialization = _materialization()
    manifest = materialization.manifest
    current = pd.DataFrame(
        {
            "object_name": ["factor_observation_clean_latest"],
            "logical_partition": [manifest.logical_partition],
            "batch_id": [manifest.batch_id],
            "source_object": ["factor_observation_clean"],
            "source_data_version": [manifest.source_data_version],
            "published_at": [manifest.published_at],
            "row_count": [manifest.row_count],
            "business_hash": [manifest.business_hash],
            "status": [CompactManifestStatus.PUBLISHED.value],
        }
    )
    session = RecordingCompactSession(current_manifest=current)

    read_latest_partition(
        session,
        object_name="factor_observation_clean_latest",
        logical_partition="2025-01",
        columns=("batch_id", "signal_date", "ts_code", "factor_id", "value"),
        factor_keys=(("alpha", 2), ("beta", 1)),
    )

    query = session.queries[-1]
    assert "factor_id='alpha' and factor_version=2" in query
    assert "factor_id='beta' and factor_version=1" in query


def test_object_spec_is_strictly_bound_to_allowlisted_compact_table() -> None:
    spec = CompactObjectSpec(
        source_object="label_daily",
        target_object="label_daily_latest",
        business_keys=("signal_date", "ts_code", "label_id", "label_version"),
        partition_date_column="signal_date",
    )

    assert spec.target_object == "label_daily_latest"
