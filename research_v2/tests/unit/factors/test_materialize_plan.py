"""C5.3's dry-run half: a factor materialization plan built from metadata alone.

The clause this satisfies is the dry-run's contract -- factor count, source tables and columns,
date chunks, estimated rows, target partitions -- together with its two prohibitions: read no
values, write nothing. Both prohibitions are structural rather than behavioural, and are tested
that way: the planner accepts no session of any kind, so there is nothing through which a query
could be issued.

The registry-facing commands in C5.3 (``factor registry validate`` / ``import``) are deliberately
not built here: they are CLI over ``FactorRegistry``, which has no production caller, and which
representation is authoritative is an open decision. This module needs no registry -- it plans from
the catalog, the schema dry-run and the trading calendar, all of which already exist.
"""

from __future__ import annotations

import ast
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pytest
import research_v2.factors.materialize_plan as planner_module
from research_v2.factors.materialize_plan import (
    FactorPlanError,
    plan_factor_materialization,
)

from research_v2.backfill.windows import resolve_monthly_backfill_windows
from research_v2.factors.catalog_dry_run import SourceColumn
from research_v2.pit.calendar import TradingCalendar

_LIVE_TYPES = {
    "ts_code": "SYMBOL",
    "trade_date": "DATE",
    "factor_name": "STRING",
    "value": "DOUBLE",
}

# Enough history before 2014-01-02 for the default 20-trading-day lookback to resolve, plus two
# target months. December 2013 weekdays give 22 prior trading days.
_CALENDAR_DATES = tuple(
    
        date(2013, 12, 1) + timedelta(days=offset)
        for offset in range(31)
        if (date(2013, 12, 1) + timedelta(days=offset)).weekday() < 5
    
) + (
    date(2014, 1, 2),
    date(2014, 1, 3),
    date(2014, 1, 6),
    date(2014, 1, 7),
    date(2014, 1, 8),
    date(2014, 2, 3),
    date(2014, 2, 4),
    date(2014, 2, 5),
)


def _calendar() -> TradingCalendar:
    return TradingCalendar.from_dates(_CALENDAR_DATES)


def _entry(factor_id: str, table: str = "factor_data1", **overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "factor_id": factor_id,
        "factor_version": 1,
        "display_name": factor_id.upper(),
        "source_database": "dfs://YesLab",
        "source_table": table,
        "source_factor_name": factor_id.upper(),
        "source_value_column": "value",
        "source_date_column": "trade_date",
        "source_code_column": "ts_code",
    }
    entry.update(overrides)
    return entry


def _entries(count: int = 3) -> list[dict[str, Any]]:
    return [_entry(f"yeslab_{index}") for index in range(1, count + 1)]


def _schemas(*tables: str) -> dict[str, tuple[SourceColumn, ...]]:
    names = tables or ("factor_data1",)
    return {
        table: tuple(
            SourceColumn(name=column, type_string=kind) for column, kind in _LIVE_TYPES.items()
        )
        for table in names
    }


def test_the_planner_has_no_channel_through_which_it_could_read_or_write() -> None:
    """C5.3's prohibitions are structural, so they are checked structurally.

    A dry-run that accepts a session can always be talked into using it. This one accepts none:
    its inputs are the catalog, an optional already-collected schema map, and a trading calendar.
    Checked over the AST rather than by text so the docstrings that mention queries and writes --
    to say they do not happen -- do not produce a false failure.
    """
    path = Path(planner_module.__file__).resolve()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names.update(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not any(module.startswith("research_v2.db") for module in imported)
    assert not any("dolphindb" in module for module in imported)
    assert "query" not in names
    assert "V2WriteSession" not in names
    assert "YesLabReadOnlySession" not in names

    parameters = set()
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "plan_factor_materialization":
            parameters = {argument.arg for argument in node.args.args + node.args.kwonlyargs}
    assert "session" not in parameters and "client" not in parameters


def test_the_chunks_are_the_real_trading_months_with_their_lookback() -> None:
    plan = plan_factor_materialization(
        _entries(), calendar=_calendar(), start_date=date(2014, 1, 1), end_date=date(2014, 2, 28)
    )

    expected = resolve_monthly_backfill_windows(
        _calendar(), start_date=date(2014, 1, 1), end_date=date(2014, 2, 28)
    )
    assert [chunk.logical_partition for chunk in plan.chunks] == [
        window.month for window in expected
    ]
    assert [(chunk.target_start, chunk.target_end) for chunk in plan.chunks] == [
        (window.target_start, window.target_end) for window in expected
    ]
    assert [chunk.lookback_start for chunk in plan.chunks] == [
        window.lookback_start for window in expected
    ]


def test_the_estimate_counts_target_days_only_and_never_the_lookback() -> None:
    """The lookback feeds features; it is not published, so it must not inflate the estimate.

    Including it would overstate the write volume by the lookback fraction and, worse, would make
    the estimate disagree with the number of rows the materializer actually publishes for the
    month. January has five trading days in the fixture calendar and the default lookback reaches
    into December, so the two are visibly different here.
    """
    plan = plan_factor_materialization(
        _entries(2),
        calendar=_calendar(),
        start_date=date(2014, 1, 1),
        end_date=date(2014, 1, 31),
        cross_section_size=1000,
        lookback_trading_days=3,
    )

    chunk = plan.chunks[0]
    assert chunk.trading_days == 5
    assert chunk.lookback_start < chunk.target_start, "fixture premise: lookback reaches back"
    lookback_days = len(_calendar().window(chunk.lookback_start, chunk.target_end))
    assert lookback_days > chunk.trading_days

    assert chunk.estimated_rows == 2 * 5 * 1000
    assert chunk.estimated_rows != 2 * lookback_days * 1000


def test_the_row_estimate_is_absent_rather_than_invented_without_a_cross_section() -> None:
    """There is no way to know the cross-section width without reading data, so it says so.

    A dry-run that guesses a width produces a number that looks measured. ``None`` propagates to the
    total, so a caller cannot accidentally sum a mixture of estimates and guesses.
    """
    plan = plan_factor_materialization(
        _entries(), calendar=_calendar(), start_date=date(2014, 1, 1), end_date=date(2014, 2, 28)
    )

    assert plan.cross_section_size is None
    assert all(chunk.estimated_rows is None for chunk in plan.chunks)
    assert plan.estimated_rows_total is None


def test_a_blocked_factor_stops_the_plan_by_default() -> None:
    """C5.1's principle applied to planning: a factor that cannot be read must not be planned in."""
    schemas = _schemas()
    entries = [_entry("yeslab_1"), _entry("yeslab_2", source_value_column="absent_column")]

    with pytest.raises(FactorPlanError, match="yeslab_2") as error:
        plan_factor_materialization(
            entries,
            calendar=_calendar(),
            start_date=date(2014, 1, 1),
            end_date=date(2014, 1, 31),
            schemas=schemas,
        )

    assert "absent_column" in str(error.value)


def test_a_non_strict_plan_reports_blocked_factors_instead_of_raising() -> None:
    schemas = _schemas()
    entries = [_entry("yeslab_1"), _entry("yeslab_2", source_value_column="absent_column")]

    plan = plan_factor_materialization(
        entries,
        calendar=_calendar(),
        start_date=date(2014, 1, 1),
        end_date=date(2014, 1, 31),
        schemas=schemas,
        strict=False,
    )

    assert plan.blocked_factors == (("yeslab_2", 1),)
    assert plan.schema_verified is True
    assert plan.materializable is False


def test_a_plan_verified_against_a_clean_schema_is_materializable() -> None:
    plan = plan_factor_materialization(
        _entries(),
        calendar=_calendar(),
        start_date=date(2014, 1, 1),
        end_date=date(2014, 1, 31),
        schemas=_schemas(),
    )

    assert plan.schema_verified is True
    assert plan.blocked_factors == ()
    assert plan.materializable is True


def test_a_plan_built_without_schemas_says_it_was_not_verified() -> None:
    """An unverified plan must not look like a verified one."""
    plan = plan_factor_materialization(
        _entries(), calendar=_calendar(), start_date=date(2014, 1, 1), end_date=date(2014, 1, 31)
    )

    assert plan.schema_verified is False
    assert plan.materializable is False, (
        "not verified against the source is not the same as verified clean"
    )


def test_unmapped_source_columns_carry_through_from_the_schema_dry_run() -> None:
    drifted = dict(_LIVE_TYPES)
    drifted["newly_added"] = "DOUBLE"

    plan = plan_factor_materialization(
        _entries(),
        calendar=_calendar(),
        start_date=date(2014, 1, 1),
        end_date=date(2014, 1, 31),
        schemas={"factor_data1": tuple(
            SourceColumn(name=name, type_string=kind) for name, kind in drifted.items()
        )},
    )

    assert dict(plan.unmapped_source_columns) == {"factor_data1": ("newly_added",)}


def test_retired_factors_leave_the_count_the_estimate_and_the_columns() -> None:
    entries = [_entry("yeslab_1"), _entry("yeslab_2", source_value_column="legacy_value")]
    widened = dict(_LIVE_TYPES)
    widened["legacy_value"] = "DOUBLE"
    schemas = {
        "factor_data1": tuple(
            SourceColumn(name=name, type_string=kind) for name, kind in widened.items()
        )
    }

    full = plan_factor_materialization(
        entries,
        calendar=_calendar(),
        start_date=date(2014, 1, 1),
        end_date=date(2014, 1, 31),
        schemas=schemas,
        cross_section_size=100,
    )
    after = plan_factor_materialization(
        entries,
        calendar=_calendar(),
        start_date=date(2014, 1, 1),
        end_date=date(2014, 1, 31),
        schemas=schemas,
        cross_section_size=100,
        retired={("yeslab_2", 1)},
    )

    assert full.factor_count == 2 and after.factor_count == 1
    assert after.chunks[0].estimated_rows == 1 * 5 * 100
    assert dict(after.unmapped_source_columns) == {"factor_data1": ("legacy_value",)}
    assert after.source_columns["factor_data1"] == ("factor_name", "trade_date", "ts_code", "value")


def test_source_columns_come_from_the_catalog_registration_not_a_hardcoded_list() -> None:
    entries = [_entry("yeslab_1", source_value_column="val", source_date_column="dt")]
    schemas = {
        "factor_data1": (
            SourceColumn("ts_code", "SYMBOL"),
            SourceColumn("dt", "DATE"),
            SourceColumn("factor_name", "STRING"),
            SourceColumn("val", "DOUBLE"),
        )
    }

    plan = plan_factor_materialization(
        entries,
        calendar=_calendar(),
        start_date=date(2014, 1, 1),
        end_date=date(2014, 1, 31),
        schemas=schemas,
    )

    assert plan.source_columns["factor_data1"] == ("dt", "factor_name", "ts_code", "val")


def test_the_plan_names_the_target_table_and_partition_column_the_migration_declares() -> None:
    plan = plan_factor_materialization(
        _entries(), calendar=_calendar(), start_date=date(2014, 1, 1), end_date=date(2014, 1, 31)
    )

    assert plan.target_table == "factor_observation_clean"
    assert plan.target_partition_column == "signal_date"
    assert plan.chunks[0].logical_partition == "2014-01"


def test_an_inverted_or_empty_range_is_refused() -> None:
    with pytest.raises(FactorPlanError, match="start_date"):
        plan_factor_materialization(
            _entries(),
            calendar=_calendar(),
            start_date=date(2014, 3, 1),
            end_date=date(2014, 1, 1),
        )

    with pytest.raises(FactorPlanError, match="no trading days"):
        plan_factor_materialization(
            _entries(),
            calendar=_calendar(),
            start_date=date(2015, 1, 1),
            end_date=date(2015, 1, 31),
        )


def test_an_empty_catalog_is_refused() -> None:
    with pytest.raises(FactorPlanError, match="no factor entries"):
        plan_factor_materialization(
            [], calendar=_calendar(), start_date=date(2014, 1, 1), end_date=date(2014, 1, 31)
        )


def test_a_non_positive_cross_section_is_refused() -> None:
    with pytest.raises(FactorPlanError, match="cross_section_size"):
        plan_factor_materialization(
            _entries(),
            calendar=_calendar(),
            start_date=date(2014, 1, 1),
            end_date=date(2014, 1, 31),
            cross_section_size=0,
        )


def test_the_plan_is_independent_of_catalog_ordering() -> None:
    entries = _entries(4)

    forward = plan_factor_materialization(
        entries,
        calendar=_calendar(),
        start_date=date(2014, 1, 1),
        end_date=date(2014, 2, 28),
        cross_section_size=10,
    )
    backward = plan_factor_materialization(
        list(reversed(entries)),
        calendar=_calendar(),
        start_date=date(2014, 1, 1),
        end_date=date(2014, 2, 28),
        cross_section_size=10,
    )

    assert forward.chunks == backward.chunks
    assert forward.factor_count == backward.factor_count
    assert forward.estimated_rows_total == backward.estimated_rows_total


def test_factors_spanning_several_tables_are_grouped_per_chunk_not_flattened() -> None:
    """The read path queries one table per month, so the plan shows which tables each chunk hits."""
    entries = [
        _entry("yeslab_1", table="factor_data1"),
        _entry("yeslab_2", table="factor_data3"),
        _entry("yeslab_3", table="factor_data3"),
    ]

    plan = plan_factor_materialization(
        entries, calendar=_calendar(), start_date=date(2014, 1, 1), end_date=date(2014, 1, 31)
    )

    assert plan.source_tables == ("factor_data1", "factor_data3")
    assert plan.chunks[0].source_tables == ("factor_data1", "factor_data3")
    assert sorted(plan.source_columns) == ["factor_data1", "factor_data3"]
