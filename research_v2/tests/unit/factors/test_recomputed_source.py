"""Unit evidence for the recomputed factor source session."""

from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.factors.recomputed_source import (
    RecomputedFactorSourceSession,
    RecomputeRoute,
    parse_recomputation_query,
)


class FakeCalculator:
    """Deterministic stand-in: value = day-of-month + factor ordinal."""

    def __init__(self, client, lookback_days, *, ordinal: float = 0.0):
        self.lookback_days = lookback_days
        self.ordinal = ordinal

    def calculate(self, start_time: str, end_time: str) -> pd.DataFrame:
        days = pd.date_range(start_time, end_time, freq="B")
        columns = {
            f"s{index:02d}": [float(day.day) + self.ordinal for day in days]
            for index in range(3)
        }
        return pd.DataFrame(columns, index=days)


def _route(ordinal: float, source_table: str = "factor_data8") -> RecomputeRoute:
    return RecomputeRoute(
        source_table=source_table,
        lookback_days=10,
        factory=lambda client, lookback_days: FakeCalculator(
            client, lookback_days, ordinal=ordinal
        ),
    )


def test_parse_recomputation_query_extracts_table_names_and_window() -> None:
    script = (
        "select ts_code, trade_date, factor_name, value from "
        "loadTable('dfs://YesLab', 'factor_data8') "
        "where factor_name in ['GP_007', 'GP_008'] "
        "and trade_date between date('2014.01.01') and date('2014.01.31')"
    )
    parsed = parse_recomputation_query(script)
    assert parsed.source_table == "factor_data8"
    assert parsed.factor_names == ("GP_007", "GP_008")
    assert parsed.start_date == date(2014, 1, 1)
    assert parsed.end_date == date(2014, 1, 31)


def test_parse_rejects_unsafe_tables() -> None:
    script = (
        "select ts_code from loadTable('dfs://YesLab', 'evil_table') "
        "where factor_name in ['X'] and trade_date between date('2014.01.01') "
        "and date('2014.01.31')"
    )
    with pytest.raises(ValueError):
        parse_recomputation_query(script)


def test_session_serves_recomputed_long_frames_within_window() -> None:
    session = RecomputedFactorSourceSession(
        market_client=object(),
        factor_routes={"GP_007": _route(1.0), "GP_008": _route(2.0)},
    )
    frame = session.query(
        "select ts_code, trade_date, factor_name, value from "
        "loadTable('dfs://YesLab', 'factor_data8') "
        "where factor_name in ['GP_007', 'GP_008'] "
        "and trade_date between date('2014.01.01') and date('2014.01.10')"
    )

    assert list(frame.columns) == ["ts_code", "trade_date", "factor_name", "value"]
    assert set(frame["factor_name"]) == {"GP_007", "GP_008"}
    assert frame["trade_date"].min() >= date(2014, 1, 1)
    assert frame["trade_date"].max() <= date(2014, 1, 10)
    gp007 = frame.loc[
        (frame["factor_name"] == "GP_007") & (frame["ts_code"] == "s00"), "value"
    ].to_numpy()
    gp008 = frame.loc[
        (frame["factor_name"] == "GP_008") & (frame["ts_code"] == "s00"), "value"
    ].to_numpy()
    assert len(gp007) == len(gp008) > 0
    assert (gp007 == gp008 - 1.0).all()


def test_session_rejects_unknown_factor_or_table() -> None:
    session = RecomputedFactorSourceSession(
        market_client=object(), factor_routes={"GP_007": _route(0.0)}
    )
    with pytest.raises(ValueError):
        session.query(
            "select ts_code, trade_date, factor_name, value from "
            "loadTable('dfs://YesLab', 'factor_data8') "
            "where factor_name in ['GP_999'] "
            "and trade_date between date('2014.01.01') and date('2014.01.10')"
        )
    with pytest.raises(ValueError):
        session.query(
            "select ts_code, trade_date, factor_name, value from "
            "loadTable('dfs://YesLab', 'factor_data9') "
            "where factor_name in ['GP_007'] "
            "and trade_date between date('2014.01.01') and date('2014.01.10')"
        )
