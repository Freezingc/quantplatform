from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest

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


def _reject_constant(constant: str) -> float:
    raise ValueError(f"non-standard JSON constant: {constant}")


def test_a_quarantined_infinity_is_published_as_non_standard_json() -> None:
    """Defect, pinned deliberately: the quarantine's evidence column can hold invalid JSON.

    C2.3 requires +/-inf to be carried as an invalid flag rather than turned into NaN, so
    ``raw_value`` really can be infinite. A key whose observations are all non-finite is
    quarantined as ``NO_FINITE_VALUE``, and the materializer aggregates its evidence into
    ``distinct_values_json`` with ``json.dumps`` -- which emits ``Infinity``, a Python extension
    RFC 8259 does not allow. ``pd.notna`` keeps the infinity (only NaN is dropped), so the value
    survives into the string.

    The consequence lands where it hurts most. The quarantine *is* the audit trail for
    un-publishable observations, and a strict parser -- a database JSON function, JS
    ``JSON.parse``, serde_json -- rejects precisely the record describing the worst-behaved data
    in the source. The same aggregation also drops the reader's ``invalid_reason``, so the
    published row cannot distinguish "the source said infinity" from "the source said nothing".
    The clean batch is unaffected throughout; this is evidence format only.

    Not fixed here. How to represent an infinity is a format decision (string sentinel, null plus
    a flag, or carrying ``invalid_reason`` through) that changes a persisted column's contents and
    therefore its ``record_hash``, and the migrations already define the column. Recorded so the
    fix is a deliberate act with a known blast radius.
    """
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000002.SZ"],
            "trade_date": [date(2025, 1, 2)] * 3,
            "factor_name": ["alpha"] * 3,
            "value": [1.0, float("inf"), None],
        }
    )
    context = RunContext(
        run_id="00000000-0000-0000-0000-000000000001", data_version="version-a"
    )

    batches = build_technical_factor_batches(
        source,
        _definition(),
        source_factor_name="alpha",
        source_watermark_id="watermark",
        context=context,
    )

    assert batches.clean_batch.frame["ts_code"].tolist() == ["000001.SZ"], (
        "the infinite observation must not reach the clean batch"
    )
    assert batches.quarantine_batch is not None
    quarantine = batches.quarantine_batch.frame
    assert quarantine.loc[0, "reason"] == "NO_FINITE_VALUE"

    published = quarantine.loc[0, "distinct_values_json"]
    assert published == "[Infinity]"
    with pytest.raises(ValueError, match="non-standard JSON constant"):
        json.loads(published, parse_constant=_reject_constant)

    assert "invalid_reason" not in quarantine.columns, (
        "the reader's distinction between a missing and a non-finite value is not carried through"
    )


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
