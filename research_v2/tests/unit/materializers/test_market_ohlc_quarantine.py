"""Unit evidence for legacy OHLC-violation quarantine."""

from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from research_v2.contracts.source import SourceBatch
from research_v2.materializers.market import (
    MarketDataQualityError,
    build_market_ohlc_quarantine_batch,
    partition_invalid_ohlc,
)
from research_v2.runs.context import RunContext


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "A.SZ",
                "trade_date": date(2014, 6, 18),
                "open": 10.0,
                "high": 10.5,
                "low": 9.8,
                "close": 10.2,
                "adj_factor": 1.0,
            },
            {
                "ts_code": "B.BJ",
                "trade_date": date(2014, 6, 18),
                "open": 10.88,
                "high": 10.88,
                "low": 10.88,
                "close": 10.81,
                "adj_factor": 0.994,
            },
            {
                "ts_code": "C.SH",
                "trade_date": date(2014, 6, 19),
                "open": 5.0,
                "high": 5.2,
                "low": 4.9,
                "close": 5.1,
                "adj_factor": 1.0,
            },
        ]
    )


def test_partition_separates_only_ohlc_violations() -> None:
    clean, invalid = partition_invalid_ohlc(_frame())
    assert clean["ts_code"].tolist() == ["A.SZ", "C.SH"]
    assert invalid["ts_code"].tolist() == ["B.BJ"]


def test_partition_rejects_missing_columns() -> None:
    with pytest.raises(MarketDataQualityError):
        partition_invalid_ohlc(pd.DataFrame({"ts_code": ["A.SZ"]}))


def test_quarantine_batch_records_ohlc_violation() -> None:
    _, invalid = partition_invalid_ohlc(_frame())
    source = SourceBatch(
        frame=invalid,
        source_name="yeslab",
        table_name="stock_factor_daily",
        watermark_id="w",
        max_observation_date=date(2014, 6, 18),
        source_row_count=1,
        schema_fingerprint="x",
        extracted_at=datetime(2026, 9, 2, tzinfo=UTC),
    )
    context = RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="v")
    batch = build_market_ohlc_quarantine_batch(invalid, source, context)
    assert batch.object_name == "market_quarantine"
    assert batch.frame["reason"].tolist() == ["INVALID_OHLC_RELATIONSHIP"]
    assert batch.frame["field_name"].tolist() == ["ohlc"]
    assert "c=10.81" in batch.frame["raw_value"].iloc[0]


def test_quarantine_batch_requires_rows() -> None:
    clean, _ = partition_invalid_ohlc(_frame())
    source = SourceBatch(
        frame=clean,
        source_name="yeslab",
        table_name="stock_factor_daily",
        watermark_id="w",
        max_observation_date=date(2014, 6, 19),
        source_row_count=2,
        schema_fingerprint="x",
        extracted_at=datetime(2026, 9, 2, tzinfo=UTC),
    )
    context = RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="v")
    with pytest.raises(MarketDataQualityError):
        build_market_ohlc_quarantine_batch(clean.iloc[0:0], source, context)
