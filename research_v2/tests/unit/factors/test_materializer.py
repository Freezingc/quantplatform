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


# --- C5.2: the published column contract, and its agreement with the table definition ------------

import re  # noqa: E402
from pathlib import Path  # noqa: E402

_MIGRATIONS = Path(__file__).resolve().parents[3] / "migrations"

#: C5.2's minimum field list for ``factor_observation_clean``, in the order the plan gives it.
_C52_MINIMUM_FIELDS = (
    "signal_date",
    "ts_code",
    "factor_id",
    "factor_version",
    "value",
    "observation_date",
    "available_date",
    "report_period",
    "stale_days",
    "dedup_resolution",
    "source_revision_id",
    "run_id",
    "data_version",
    "created_at",
    "source_watermark_id",
    "record_hash",
)

_TABLE_DECLARATION = re.compile(
    r"existsTable\(targetDatabase,\s*`(?P<table>\w+)\)\)\s*\{\s*"
    r"\w+\s*=\s*table\(1:0,\s*"
    r"`(?P<names>[^,\n]+),\s*"
    r"\[(?P<types>[^\]]+)\]",
    re.MULTILINE,
)


def _declared_columns(table_name: str, migration: str = "003_factor_label_tables.dos") -> tuple[
    list[str], list[str]
]:
    """Parse a table's column names and types out of its migration script."""
    script = (_MIGRATIONS / migration).read_text(encoding="utf-8")
    for match in _TABLE_DECLARATION.finditer(script):
        if match.group("table") == table_name:
            names = [name for name in match.group("names").split("`") if name]
            types = [kind.strip() for kind in match.group("types").split(",")]
            assert len(names) == len(types), "migration names and types disagree in length"
            return names, types
    raise AssertionError(f"no declaration found for {table_name} in {migration}")


def _published_clean_frame() -> pd.DataFrame:
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000002.SZ", "000003.SZ"],
            "trade_date": [date(2025, 1, 2)] * 4,
            "factor_name": ["alpha"] * 4,
            "value": [1.0, 2.0, 3.0, None],
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
    return batches.clean_batch.frame


def test_the_published_columns_are_exactly_the_c52_minimum_field_list() -> None:
    """C5.2's field list is the contract every downstream consumer reads against."""
    clean = _published_clean_frame()

    assert tuple(clean.columns) == _C52_MINIMUM_FIELDS


def test_the_materializer_output_matches_the_table_the_migration_creates() -> None:
    """The edge nothing else watches: producer columns against the declared table schema.

    Migration tests in this repository assert that a table *name* appears in the script text, and
    the golden sample compares a six-column subset, so a column added, dropped, renamed or
    reordered on either side would pass both. That is not a hypothetical drift -- the materializer
    assembles its frame by naming columns in a literal, and the migration declares them in a
    backtick-delimited list, with no shared source of truth between the two.

    Order is asserted as well as membership because a positional write would otherwise publish
    one column's values under another's name, which is silent corruption rather than an error.
    """
    declared_names, declared_types = _declared_columns("factor_observation_clean")
    clean = _published_clean_frame()

    assert list(clean.columns) == declared_names
    assert declared_types[declared_names.index("value")] == "DOUBLE"
    assert declared_types[declared_names.index("signal_date")] == "DATE"
    assert declared_types[declared_names.index("record_hash")] == "STRING"
    assert declared_types[declared_names.index("run_id")] == "UUID"


def test_the_quarantine_frame_matches_its_declared_table_too() -> None:
    """Same edge for the audit table, whose columns are assembled in a separate literal."""
    declared_names, _ = _declared_columns("factor_quarantine")
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000002.SZ"],
            "trade_date": [date(2025, 1, 2)] * 3,
            "factor_name": ["alpha"] * 3,
            "value": [1.0, 2.0, 3.0],
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

    assert batches.quarantine_batch is not None
    assert list(batches.quarantine_batch.frame.columns) == declared_names


def test_a_row_with_no_value_never_reaches_the_fact_table() -> None:
    """C5.2: 值为空的行默认不写事实表.

    Missingness is still represented -- it goes to the quarantine with a reason -- so this is not
    a silent drop. What must not happen is a null ``value`` sitting in the fact table, where every
    consumer would have to re-decide what it means.
    """
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000003.SZ", "000004.SZ"],
            "trade_date": [date(2025, 1, 2)] * 3,
            "factor_name": ["alpha"] * 3,
            "value": [1.0, None, float("nan")],
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

    clean = batches.clean_batch.frame
    assert clean["ts_code"].tolist() == ["000001.SZ"]
    assert clean["value"].notna().all(), "a null value reached the fact table"
    assert batches.quarantine_batch is not None
    assert set(batches.quarantine_batch.frame["ts_code"]) == {"000003.SZ", "000004.SZ"}
    assert set(batches.quarantine_batch.frame["reason"]) == {"NO_FINITE_VALUE"}


def test_a_technical_factor_publishes_a_null_report_period_and_a_real_revision_id() -> None:
    """C5.2 allows both to be null for technical factors; the implementation fills one of them.

    ``report_period`` is genuinely absent -- a daily technical observation belongs to no reporting
    period, so NaT is the honest value and a substituted date would imply an accounting period
    that does not exist. ``source_revision_id`` is populated with the source row id, which is
    stronger than the clause requires and worth pinning, because it is the only link from a
    published row back to the exact source observation it came from.
    """
    clean = _published_clean_frame()

    assert clean["report_period"].isna().all()
    assert clean["source_revision_id"].notna().all()
    assert clean["source_revision_id"].nunique() == len(clean)
