from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.sources.yeslab import SourceSchemaDrift, YesLabAdapter


class FakeReadSession:
    def __init__(self, schema_names: list[str], market: pd.DataFrame) -> None:
        self.schema_names = schema_names
        self.market = market
        self.scripts: list[str] = []

    def query(self, script: str) -> object:
        self.scripts.append(script)
        if script.startswith("schema("):
            return pd.DataFrame({"name": self.schema_names, "typeString": "DOUBLE"})
        return self.market.copy()


def test_market_adapter_returns_a_versioned_batch_from_explicit_columns() -> None:
    market = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": [date(2025, 1, 2)],
            "open": [10.0],
            "high": [10.5],
            "low": [9.8],
            "close": [10.2],
            "vol": [1000.0],
            "amount": [10000.0],
            "adj_factor": [1.2],
        }
    )
    adapter = YesLabAdapter(FakeReadSession(list(market.columns), market))

    batch = adapter.fetch_market(date(2025, 1, 2), date(2025, 1, 2))

    assert batch.source_name == "YesLab"
    assert batch.table_name == "stock_factor_daily"
    assert batch.max_observation_date == date(2025, 1, 2)
    assert batch.source_row_count == 1
    assert "select ts_code, trade_date, open" in adapter.last_query


def test_market_adapter_fails_closed_on_source_schema_drift() -> None:
    adapter = YesLabAdapter(
        FakeReadSession(
            ["ts_code", "trade_date", "high", "low", "close", "vol", "amount", "adj_factor"],
            pd.DataFrame(),
        )
    )

    with pytest.raises(SourceSchemaDrift, match="open"):
        adapter.fetch_market(date(2025, 1, 2), date(2025, 1, 2))


def test_market_adapter_reads_only_requested_security_observations() -> None:
    observations = pd.DataFrame(
        {
            "ts_code": ["300114.SZ"],
            "trade_date": [date(2013, 1, 4)],
        }
    )
    adapter = YesLabAdapter(FakeReadSession(list(observations.columns), observations))

    batch = adapter.fetch_market_observations(
        {"300114.SZ"},
        date(2013, 1, 1),
        date(2025, 1, 2),
    )

    assert batch.table_name == "stock_factor_daily"
    assert batch.frame.equals(observations)
    assert "ts_code in [`300114.SZ]" in adapter.last_query


def test_calendar_adapter_selects_open_days_with_an_explicit_exchange_contract() -> None:
    calendar = pd.DataFrame(
        {
            "exchange": ["SSE"],
            "cal_date": [date(2025, 1, 2)],
            "is_open": [1],
            "pretrade_date": [date(2024, 12, 31)],
        }
    )

    class CalendarSession:
        def query(self, script: str) -> object:
            if script.startswith("schema("):
                return pd.DataFrame(
                    {
                        "name": list(calendar.columns),
                        "typeString": ["SYMBOL", "DATE", "INT", "DATE"],
                    }
                )
            return calendar.copy()

    adapter = YesLabAdapter(CalendarSession())

    batch = adapter.fetch_trading_calendar("SSE", date(2025, 1, 2), date(2025, 1, 2))

    assert batch.table_name == "trade_calendar"
    assert batch.source_row_count == 1
    assert "exchange = `SSE" in adapter.last_query
    assert "is_open = 1" in adapter.last_query


def test_security_adapters_use_explicit_snapshot_history_and_suspension_columns() -> None:
    tables = {
        "stock_info_table": pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "exchange": ["SZSE"],
                "market": ["主板"],
                "list_date": [date(2020, 1, 2)],
                "delist_date": [pd.NaT],
                "name": ["Alpha"],
                "list_status": ["L"],
            }
        ),
        "stock_used_name": pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "name": ["*ST Alpha"],
                "start_date": [date(2020, 2, 3)],
                "end_date": [date(2020, 2, 7)],
                "ann_date": [date(2020, 2, 3)],
                "change_reason": ["ST"],
            }
        ),
        "stock_trading_suspension_info": pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": [date(2025, 1, 2)],
                "suspend_timing": [""],
                "suspend_type": ["S"],
            }
        ),
    }

    class SecuritySession:
        def query(self, script: str) -> object:
            table_name = next(name for name in tables if name in script)
            frame = tables[table_name]
            if script.startswith("schema("):
                return pd.DataFrame({"name": list(frame.columns), "typeString": "STRING"})
            return frame.copy()

    adapter = YesLabAdapter(SecuritySession())

    snapshot = adapter.fetch_security_snapshot(date(2025, 1, 3))
    history = adapter.fetch_name_history(date(2025, 1, 2), date(2025, 1, 3))
    suspensions = adapter.fetch_suspensions(date(2025, 1, 2), date(2025, 1, 3))

    assert snapshot.table_name == "stock_info_table"
    assert history.table_name == "stock_used_name"
    assert suspensions.table_name == "stock_trading_suspension_info"
    assert suspensions.source_row_count == 1
