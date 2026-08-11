from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest

from research_v2.compact.materializer import (
    build_compact_materialization,
    compact_business_columns,
)


def _source() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2025-01-02", "2025-01-02"]),
            "ts_code": ["000002.SZ", "000001.SZ"],
            "label_id": ["forward_open_5d", "forward_open_5d"],
            "label_version": [1, 1],
            "entry_date": pd.to_datetime(["2025-01-03", "2025-01-03"]),
            "exit_date": pd.to_datetime(["2025-01-10", "2025-01-10"]),
            "label_available_date": pd.to_datetime(["2025-01-10", "2025-01-10"]),
            "raw_return": [0.02, -0.01],
            "rank_pct": [float("nan"), float("nan")],
            "winsorized_return": [float("nan"), float("nan")],
            "zscore_return": [float("nan"), float("nan")],
            "neutral_residual": [float("nan"), float("nan")],
            "label_valid": [True, True],
            "invalid_reason": ["", ""],
            "entry_realized_tradable": [True, True],
            "data_version": ["source-v1", "source-v1"],
            "run_id": ["run-a", "run-a"],
            "record_hash": ["old-a", "old-b"],
        }
    )


def test_compact_materialization_filters_version_and_projects_business_columns() -> None:
    business = compact_business_columns("label_daily")
    result = build_compact_materialization(
        _source(),
        object_name="label_daily",
        source_data_version="source-v1",
        logical_partition="2025-01",
        business_columns=business,
        business_keys=("signal_date", "ts_code", "label_id", "label_version"),
        partition_date_column="signal_date",
        published_at=datetime(2026, 8, 2, tzinfo=UTC),
    )

    assert tuple(result.frame.columns) == ("batch_id", *business)
    assert len(result.frame) == 2
    assert "record_hash" not in result.frame.columns
    assert result.manifest.row_count == 2
    assert result.manifest.source_data_version == "source-v1"


def test_compact_hash_is_independent_of_source_row_order() -> None:
    kwargs = {
        "object_name": "label_daily",
        "source_data_version": "source-v1",
        "logical_partition": "2025-01",
        "business_columns": compact_business_columns("label_daily"),
        "business_keys": ("signal_date", "ts_code", "label_id", "label_version"),
        "partition_date_column": "signal_date",
    }

    first = build_compact_materialization(_source(), **kwargs)
    second = build_compact_materialization(_source().iloc[::-1], **kwargs)

    assert first.manifest.business_hash == second.manifest.business_hash
    assert first.manifest.batch_id == second.manifest.batch_id


def test_compact_materialization_rejects_duplicate_keys_and_mixed_partitions() -> None:
    duplicate = pd.concat([_source(), _source().iloc[[0]]], ignore_index=True)
    kwargs = {
        "object_name": "label_daily",
        "source_data_version": "source-v1",
        "logical_partition": "2025-01",
        "business_columns": compact_business_columns("label_daily"),
        "business_keys": ("signal_date", "ts_code", "label_id", "label_version"),
        "partition_date_column": "signal_date",
    }
    with pytest.raises(ValueError, match="duplicate"):
        build_compact_materialization(duplicate, **kwargs)

    mixed = _source().copy()
    mixed.loc[1, "signal_date"] = pd.Timestamp("2025-02-03")
    with pytest.raises(ValueError, match="logical partition"):
        build_compact_materialization(mixed, **kwargs)


def test_compact_materialization_rejects_unknown_object() -> None:
    with pytest.raises(ValueError, match="unsupported compact object"):
        compact_business_columns("unknown_daily")
