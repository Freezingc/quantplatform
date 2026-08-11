from __future__ import annotations

from datetime import date

import pandas as pd

from research_v2.contracts.factor import AvailabilityPolicy, FactorDefinition, FactorType
from research_v2.factors.materializer import (
    build_technical_factor_batches,
    combine_factor_materialization_batches,
)
from research_v2.runs.context import RunContext


def _definition() -> FactorDefinition:
    return FactorDefinition(
        factor_id="yeslab_1",
        factor_version=1,
        display_name="alpha",
        source_database="dfs://YesLab",
        source_table="factor_data1",
        source_value_column="value",
        source_date_column="trade_date",
        source_code_column="ts_code",
        factor_type=FactorType.TECHNICAL,
        frequency="DAILY",
        expected_direction=0,
        availability_policy=AvailabilityPolicy.SAME_DAY_CLOSE,
        max_staleness_calendar_days=0,
        unit="UNKNOWN",
        owner="research",
    )


def test_materializer_builds_clean_and_conflict_quarantine_batches() -> None:
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000002.SZ", "000003.SZ"],
            "trade_date": [date(2025, 1, 2)] * 4,
            "factor_name": ["alpha"] * 4,
            "value": [1.0, 2.0, 3.0, None],
        }
    )
    context = RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="version-a")

    batches = build_technical_factor_batches(
        source,
        _definition(),
        source_factor_name="alpha",
        source_watermark_id="watermark",
        context=context,
    )

    assert len(batches.clean_batch.frame) == 1
    assert batches.clean_batch.frame.loc[0, "signal_date"] == date(2025, 1, 2)
    assert batches.clean_batch.frame.loc[0, "value"] == 1.0
    assert batches.clean_batch.partition_date_column == "signal_date"
    assert batches.quarantine_batch is not None
    assert batches.quarantine_batch.frame.loc[0, "reason"] == "CONFLICT_QUARANTINE"
    assert batches.quarantine_batch.frame.loc[0, "distinct_values_json"] == "[2.0, 3.0]"


def test_factor_materialization_batches_combine_before_partition_publish() -> None:
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": [date(2025, 1, 2)],
            "factor_name": ["alpha"],
            "value": [1.0],
        }
    )
    context = RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="v1")
    batches = build_technical_factor_batches(
        source, _definition(), source_factor_name="alpha", source_watermark_id="w", context=context
    )

    combined = combine_factor_materialization_batches([batches])

    assert combined.clean_batch.data_version == "v1"
    assert len(combined.clean_batch.frame) == 1


def test_factor_materializer_fast_record_hashes_are_stable_sha256_strings() -> None:
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": [date(2025, 1, 2)],
            "factor_name": ["alpha"],
            "value": [1.0],
        }
    )
    context = RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="v1")

    batches = build_technical_factor_batches(
        source,
        _definition(),
        source_factor_name="alpha",
        source_watermark_id="w",
        context=context,
        record_hash_strategy="fast",
    )

    assert len(batches.clean_batch.frame.loc[0, "record_hash"]) == 64


def test_factor_materializer_can_skip_lineage_for_write_free_preflight() -> None:
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": [date(2025, 1, 2)],
            "factor_name": ["alpha"],
            "value": [1.0],
        }
    )
    context = RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="v1")

    batches = build_technical_factor_batches(
        source,
        _definition(),
        source_factor_name="alpha",
        source_watermark_id="w",
        context=context,
        include_lineage=False,
    )

    assert "record_hash" not in batches.clean_batch.frame.columns
    assert "data_version" not in batches.clean_batch.frame.columns
