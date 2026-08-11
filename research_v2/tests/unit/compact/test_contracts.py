from __future__ import annotations

from datetime import UTC, datetime

import pytest

from research_v2.compact.contracts import (
    CompactManifestStatus,
    LatestManifestEntry,
    compute_batch_id,
)


def test_batch_id_is_deterministic_and_positive() -> None:
    first = compute_batch_id("label_daily_latest", "2025-01", "source-version-a")
    second = compute_batch_id("label_daily_latest", "2025-01", "source-version-a")

    assert first == second
    assert first > 0
    assert first != compute_batch_id("label_daily_latest", "2025-02", "source-version-a")


def test_manifest_entry_requires_complete_published_identity() -> None:
    entry = LatestManifestEntry(
        object_name="label_daily_latest",
        logical_partition="2025-01",
        batch_id=123,
        source_object="label_daily",
        source_data_version="source-version-a",
        published_at=datetime(2026, 8, 2, tzinfo=UTC),
        row_count=10,
        business_hash="a" * 64,
        status=CompactManifestStatus.PUBLISHED,
    )

    assert entry.status is CompactManifestStatus.PUBLISHED
    assert entry.row_count == 10


def test_manifest_entry_rejects_invalid_hash_or_row_count() -> None:
    with pytest.raises(ValueError, match="business_hash"):
        LatestManifestEntry(
            object_name="label_daily_latest",
            logical_partition="2025-01",
            batch_id=123,
            source_object="label_daily",
            source_data_version="source-version-a",
            published_at=datetime.now(UTC),
            row_count=10,
            business_hash="not-a-hash",
            status=CompactManifestStatus.STAGED,
        )

    with pytest.raises(ValueError, match="row_count"):
        LatestManifestEntry(
            object_name="label_daily_latest",
            logical_partition="2025-01",
            batch_id=123,
            source_object="label_daily",
            source_data_version="source-version-a",
            published_at=datetime.now(UTC),
            row_count=0,
            business_hash="a" * 64,
            status=CompactManifestStatus.STAGED,
        )
