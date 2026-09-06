"""Tests for the month-scoped YesLab factor query (C plan C2.2).

C2.2 requires that the reader ask the source for only the registered columns and only the
dates in scope, rather than loading a 135-column full history onto one machine, and that it
batch by month and factor group. The implementation does all of that, but nothing pinned it:
the projection, the two predicates and the table-name allow-list are exactly the parts that
a later edit could widen without any test noticing -- and widening the projection is how a
"temporary" full-table read becomes permanent.

The table-name allow-list is also the injection guard, so it is tested to reject before any
query is issued rather than merely to raise eventually.
"""

from __future__ import annotations

import pandas as pd
import pytest

from research_v2.contracts.factor import (
    AvailabilityPolicy,
    FactorDefinition,
    FactorType,
)
from research_v2.factors.source_backfill import (
    FactorBackfillError,
    FactorMonth,
    _query_source_month,
)

_COLUMNS = ["ts_code", "trade_date", "factor_name", "value"]


def _definition(factor_id: str, display_name: str, source_table: str) -> FactorDefinition:
    return FactorDefinition(
        factor_id=factor_id,
        factor_version=1,
        display_name=display_name,
        source_database="dfs://YesLab",
        source_table=source_table,
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


class _RecordingSession:
    def __init__(self, frame: pd.DataFrame | None = None) -> None:
        self.queries: list[str] = []
        self.frame = (
            frame
            if frame is not None
            else pd.DataFrame({name: [] for name in _COLUMNS})
        )

    def query(self, script: str) -> pd.DataFrame:
        self.queries.append(script)
        return self.frame


def test_the_query_projects_only_the_registered_columns_and_the_month_in_scope() -> None:
    session = _RecordingSession()
    month = FactorMonth.from_partition("2025-03")
    definitions = (
        _definition("yeslab_1", "Alpha1", "factor_data1"),
        _definition("yeslab_2", "Alpha2", "factor_data1"),
    )

    _query_source_month(session, "factor_data1", definitions, month)

    assert len(session.queries) == 1
    sql = session.queries[0]
    assert sql.startswith("select ts_code, trade_date, factor_name, value from")
    assert "loadTable('dfs://YesLab', 'factor_data1')" in sql
    # factor group predicate
    assert "factor_name in ['Alpha1', 'Alpha2']" in sql
    # month chunk predicate: the closed calendar month, not the whole history
    assert "trade_date between date('2025.03.01') and date('2025.03.31')" in sql


@pytest.mark.parametrize(
    "unsafe",
    [
        "factor_data12",
        "factor_data0",
        "factor_data",
        "stock_factor_daily",
        "factor_data1') ; drop table x --",
        "factor_data1,factor_data2",
    ],
)
def test_an_unsafe_table_name_is_rejected_before_any_query_is_issued(unsafe: str) -> None:
    """The allow-list is the injection guard, so it must fire before the session is touched."""
    session = _RecordingSession()

    with pytest.raises(FactorBackfillError, match="unsafe YesLab factor table"):
        _query_source_month(
            session,
            unsafe,
            (_definition("yeslab_1", "Alpha1", unsafe),),
            FactorMonth.from_partition("2025-03"),
        )

    assert session.queries == [], "a query was issued for a rejected table name"


def test_a_quote_in_a_factor_name_is_escaped_rather_than_terminating_the_literal() -> None:
    session = _RecordingSession()

    _query_source_month(
        session,
        "factor_data1",
        (_definition("yeslab_1", "Alpha'1", "factor_data1"),),
        FactorMonth.from_partition("2025-03"),
    )

    assert "factor_name in ['Alpha''1']" in session.queries[0]


def test_a_source_that_returns_an_unexpected_schema_is_rejected() -> None:
    session = _RecordingSession(pd.DataFrame({"ts_code": [], "trade_date": []}))

    with pytest.raises(FactorBackfillError, match="invalid schema"):
        _query_source_month(
            session,
            "factor_data1",
            (_definition("yeslab_1", "Alpha1", "factor_data1"),),
            FactorMonth.from_partition("2025-03"),
        )


def test_the_returned_frame_is_reprojected_to_exactly_the_registered_columns() -> None:
    """An extra column in the response must not ride along into the pipeline."""
    delivered = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": [pd.Timestamp("2025-03-03").date()],
            "factor_name": ["Alpha1"],
            "value": [1.5],
            "unexpected_extra": [99.0],
        }
    )
    session = _RecordingSession(delivered)

    result = _query_source_month(
        session,
        "factor_data1",
        (_definition("yeslab_1", "Alpha1", "factor_data1"),),
        FactorMonth.from_partition("2025-03"),
    )

    assert list(result.columns) == _COLUMNS
