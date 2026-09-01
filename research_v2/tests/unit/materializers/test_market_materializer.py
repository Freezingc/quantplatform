from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

import research_v2.materializers.market as market_materializer
from research_v2.contracts.source import SourceBatch
from research_v2.materializers.market import (
    active_listing_dates,
    build_market_batch,
    build_market_quarantine_batch,
)
from research_v2.pit.calendar import TradingCalendar
from research_v2.runs.context import RunContext


def test_market_batch_maps_clean_fields_and_computes_listing_age() -> None:
    source = SourceBatch(
        frame=pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000001.SZ"],
                "trade_date": [date(2025, 1, 2), date(2025, 1, 3)],
                "open": [10.0, 11.0],
                "high": [10.5, 11.5],
                "low": [9.8, 10.8],
                "close": [10.2, 11.2],
                "vol": [1000.0, 1000.0],
                "amount": [10000.0, 11000.0],
                "adj_factor": [1.2, 1.2],
            }
        ),
        source_name="YesLab",
        table_name="stock_factor_daily",
        watermark_id="market-watermark",
        max_observation_date=date(2025, 1, 3),
        source_row_count=2,
        schema_fingerprint="schema",
        extracted_at=datetime(2025, 1, 4, tzinfo=UTC),
    )
    security_scd = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "list_date": [pd.Timestamp("2025-01-02")],
            "valid_from": [pd.Timestamp("2025-01-02")],
            "valid_to": [pd.NaT],
        }
    )
    context = RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="version-a")

    batch = build_market_batch(
        source,
        security_scd,
        TradingCalendar.from_dates([date(2025, 1, 2), date(2025, 1, 3)]),
        context,
    )

    assert batch.object_name == "market_daily_clean"
    assert batch.business_keys == ("trade_date", "ts_code")
    assert batch.partition_date_column == "trade_date"
    assert batch.frame["raw_open"].tolist() == [10.0, 11.0]
    assert batch.frame["volume"].tolist() == [100000, 100000]
    assert batch.frame["listing_age_trading_days"].tolist() == [1, 2]
    assert batch.frame["source_watermark_id"].tolist() == ["market-watermark"] * 2


def test_market_batch_excludes_rows_with_missing_adjustment_factors() -> None:
    source = SourceBatch(
        frame=pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "trade_date": [date(2025, 1, 2), date(2025, 1, 2)],
                "open": [10.0, 11.0],
                "high": [10.5, 11.5],
                "low": [9.8, 10.8],
                "close": [10.2, 11.2],
                "vol": [1000.0, 1000.0],
                "amount": [10000.0, 11000.0],
                "adj_factor": [1.2, float("nan")],
            }
        ),
        source_name="YesLab",
        table_name="stock_factor_daily",
        watermark_id="market-watermark",
        max_observation_date=date(2025, 1, 2),
        source_row_count=2,
        schema_fingerprint="schema",
        extracted_at=datetime(2025, 1, 4, tzinfo=UTC),
    )
    security_scd = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "list_date": [date(2020, 1, 1), date(2020, 1, 1)],
            "valid_from": [date(2020, 1, 1), date(2020, 1, 1)],
            "valid_to": [pd.NaT, pd.NaT],
        }
    )
    context = RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="version-a")

    batch = build_market_batch(
        source,
        security_scd,
        TradingCalendar.from_dates([date(2025, 1, 2)]),
        context,
    )

    assert batch.frame["ts_code"].tolist() == ["000001.SZ"]


def test_market_batch_uses_lookback_rows_without_publishing_or_validating_them() -> None:
    source = SourceBatch(
        frame=pd.DataFrame(
            {
                "ts_code": ["832317.BJ", "000001.SZ"],
                "trade_date": [date(2019, 12, 31), date(2020, 1, 2)],
                "open": [10.0, 11.0],
                "high": [10.5, 11.5],
                "low": [9.8, 10.8],
                "close": [10.2, 11.2],
                "vol": [1000.0, 1000.0],
                "amount": [10000.0, 11000.0],
                "adj_factor": [1.0, 1.0],
            }
        ),
        source_name="YesLab",
        table_name="stock_factor_daily",
        watermark_id="market-watermark",
        max_observation_date=date(2020, 1, 2),
        source_row_count=2,
        schema_fingerprint="schema",
        extracted_at=datetime(2020, 1, 3, tzinfo=UTC),
    )
    security_scd = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "list_date": [date(2015, 1, 1)],
            "valid_from": [date(2015, 1, 1)],
            "valid_to": [pd.NaT],
        }
    )

    batch = build_market_batch(
        source,
        security_scd,
        TradingCalendar.from_dates([date(2019, 12, 31), date(2020, 1, 2)]),
        RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="v1"),
        publish_start=date(2020, 1, 2),
        publish_end=date(2020, 1, 2),
    )

    assert batch.frame["trade_date"].tolist() == [date(2020, 1, 2)]
    assert batch.frame["ts_code"].tolist() == ["000001.SZ"]


def test_history_mode_quarantines_target_rows_without_an_active_security_scd() -> None:
    source = SourceBatch(
        frame=pd.DataFrame(
            {
                "ts_code": ["832317.BJ", "000001.SZ"],
                "trade_date": [date(2020, 1, 2), date(2020, 1, 2)],
                "open": [10.0, 11.0],
                "high": [10.5, 11.5],
                "low": [9.8, 10.8],
                "close": [10.2, 11.2],
                "vol": [1000.0, 1000.0],
                "amount": [10000.0, 11000.0],
                "adj_factor": [1.0, 1.0],
            }
        ),
        source_name="YesLab",
        table_name="stock_factor_daily",
        watermark_id="market-watermark",
        max_observation_date=date(2020, 1, 2),
        source_row_count=2,
        schema_fingerprint="schema",
        extracted_at=datetime(2020, 1, 3, tzinfo=UTC),
    )
    security_scd = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "list_date": [date(2015, 1, 1)],
            "valid_from": [date(2015, 1, 1)],
            "valid_to": [pd.NaT],
        }
    )
    context = RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="v1")
    calendar = TradingCalendar.from_dates([date(2020, 1, 2)])

    clean = build_market_batch(
        source,
        security_scd,
        calendar,
        context,
        quarantine_inactive_security=True,
    )
    quarantine = market_materializer.build_market_security_quarantine_batch(
        source,
        security_scd,
        context,
    )

    assert clean.frame["ts_code"].tolist() == ["000001.SZ"]
    assert quarantine.frame.loc[0, "ts_code"] == "832317.BJ"
    assert quarantine.frame.loc[0, "reason"] == "NO_ACTIVE_SECURITY_SCD"


def test_market_quarantine_batch_preserves_missing_adjustment_factor_evidence() -> None:
    source = SourceBatch(
        frame=pd.DataFrame(
            {
                "ts_code": ["000002.SZ"],
                "trade_date": [date(2025, 1, 2)],
                "adj_factor": [float("nan")],
            }
        ),
        source_name="YesLab",
        table_name="stock_factor_daily",
        watermark_id="market-watermark",
        max_observation_date=date(2025, 1, 2),
        source_row_count=1,
        schema_fingerprint="schema",
        extracted_at=datetime(2025, 1, 4, tzinfo=UTC),
    )
    context = RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="version-a")

    batch = build_market_quarantine_batch(source, context)

    assert batch.object_name == "market_quarantine"
    assert batch.business_keys == ("trade_date", "ts_code", "field_name")
    assert batch.partition_date_column == "trade_date"
    assert batch.frame.loc[0, "reason"] == "INVALID_ADJ_FACTOR"
    assert batch.frame.loc[0, "raw_value"] == ""


def test_active_listing_dates_indexes_only_the_scd_interval_active_on_each_date() -> None:
    security_scd = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ"],
            "list_date": [date(2020, 1, 1), date(2020, 1, 1), date(2021, 1, 1)],
            "valid_from": [date(2020, 1, 1), date(2025, 1, 2), date(2021, 1, 1)],
            "valid_to": [date(2025, 1, 2), pd.NaT, pd.NaT],
        }
    )

    indexed = active_listing_dates(security_scd, date(2025, 1, 2))

    assert indexed == {"000001.SZ": date(2020, 1, 1), "000002.SZ": date(2021, 1, 1)}


def test_market_batch_record_hashes_are_wall_clock_independent() -> None:
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": [date(2025, 1, 2), date(2025, 1, 3)],
            "open": [10.0, 11.0],
            "high": [10.5, 11.5],
            "low": [9.8, 10.8],
            "close": [10.2, 11.2],
            "vol": [1000.0, 1000.0],
            "amount": [10000.0, 11000.0],
            "adj_factor": [1.2, 1.2],
        }
    )
    scd = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "list_date": [pd.Timestamp("2025-01-02")],
            "valid_from": [pd.Timestamp("2025-01-02")],
            "valid_to": [pd.NaT],
        }
    )
    calendar = TradingCalendar.from_dates([date(2025, 1, 2), date(2025, 1, 3)])

    def build_once() -> list[str]:
        source = SourceBatch(
            frame=frame.copy(),
            source_name="YesLab",
            table_name="stock_factor_daily",
            watermark_id="market-watermark",
            max_observation_date=date(2025, 1, 3),
            source_row_count=2,
            schema_fingerprint="schema",
            extracted_at=datetime(2025, 1, 4, tzinfo=UTC),
        )
        context = RunContext(
            run_id="00000000-0000-0000-0000-000000000001", data_version="version-a"
        )
        return build_market_batch(source, scd, calendar, context).frame[
            "record_hash"
        ].tolist()

    first = build_once()
    second = build_once()
    assert first == second
