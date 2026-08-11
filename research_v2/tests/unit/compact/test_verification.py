from __future__ import annotations

import pandas as pd

from research_v2.compact.contracts import compute_batch_id
from research_v2.compact.publisher import CompactObjectSpec
from research_v2.compact.verification import verify_compact_backfill


class VerificationSession:
    target_database = "dfs://QuantResearchV2Test"

    def __init__(self, *, bad_row_count: bool = False) -> None:
        self.bad_row_count = bad_row_count

    def upload_frame(self, name: str, frame: pd.DataFrame) -> None:
        raise AssertionError("verification must not upload")

    def execute(self, script: str) -> object:
        raise AssertionError("verification must not execute")

    def query(self, script: str) -> object:
        if "published_version" in script:
            return pd.DataFrame(
                {
                    "object_name": ["label_daily"],
                    "logical_partition": ["2025-01"],
                    "data_version": ["source-v1"],
                    "published_at": pd.to_datetime(["2026-08-02 10:00:00"], utc=True),
                }
            )
        if "latest_manifest" in script:
            batch_id = compute_batch_id("label_daily", "2025-01", "source-v1")
            return pd.DataFrame(
                {
                    "object_name": ["label_daily_latest"],
                    "logical_partition": ["2025-01"],
                    "batch_id": [batch_id],
                    "source_object": ["label_daily"],
                    "source_data_version": ["source-v1"],
                    "published_at": pd.to_datetime(["2026-08-02 10:01:00"], utc=True),
                    "row_count": [2],
                    "business_hash": ["a" * 64],
                    "status": ["PUBLISHED"],
                }
            )
        if "group by batch_id" in script:
            batch_id = compute_batch_id("label_daily", "2025-01", "source-v1")
            return pd.DataFrame(
                {
                    "batch_id": [batch_id],
                    "target_row_count": [1 if self.bad_row_count else 2],
                }
            )
        if "label_daily_latest" in script:
            return pd.DataFrame(
                {
                    "name": [
                        "batch_id",
                        "signal_date",
                        "ts_code",
                        "label_id",
                        "label_version",
                    ]
                }
            )
        return pd.DataFrame(
            {
                "name": [
                    "signal_date",
                    "ts_code",
                    "label_id",
                    "label_version",
                    "run_id",
                    "data_version",
                    "created_at",
                    "source_watermark_id",
                    "record_hash",
                ]
            }
        )


def _spec() -> tuple[CompactObjectSpec, ...]:
    return (
        CompactObjectSpec(
            source_object="label_daily",
            target_object="label_daily_latest",
            business_keys=("signal_date", "ts_code", "label_id", "label_version"),
            partition_date_column="signal_date",
        ),
    )


def test_verification_checks_pointer_coverage_row_counts_and_lineage_boundaries() -> None:
    report = verify_compact_backfill(VerificationSession(), objects=_spec())

    assert report.ok
    assert report.objects[0].pointer_count == 1
    assert report.objects[0].current_row_count == 2
    assert report.objects[0].source_lineage_intact
    assert not report.objects[0].target_has_lineage_columns


def test_verification_reports_target_row_count_mismatch() -> None:
    report = verify_compact_backfill(VerificationSession(bad_row_count=True), objects=_spec())

    assert not report.ok
    assert "row-count mismatches" in report.errors[0]
