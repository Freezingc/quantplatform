from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from research_v2.contracts.source import SourceBatch
from research_v2.materializers.tradability import build_tradability_batch
from research_v2.pit.calendar import TradingCalendar
from research_v2.runs.context import RunContext


def _batch(name: str, frame: pd.DataFrame, watermark: str) -> SourceBatch:
    return SourceBatch(
        frame=frame,
        source_name="test",
        table_name=name,
        watermark_id=watermark,
        max_observation_date=date(2025, 1, 3),
        source_row_count=len(frame),
        schema_fingerprint="schema",
        extracted_at=datetime(2025, 1, 4, tzinfo=UTC),
    )


def test_tradability_materializer_maps_pit_state_to_v2_schema() -> None:
    market = _batch(
        "market_daily_clean",
        pd.DataFrame(
            {
                "trade_date": [date(2025, 1, 3)],
                "ts_code": ["000001.SZ"],
                "valid_days_20": [15],
                "adv20": [100.0],
            }
        ),
        "market-watermark",
    )
    suspensions = _batch(
        "stock_trading_suspension_info",
        pd.DataFrame(columns=["trade_date", "ts_code", "suspend_type"]),
        "suspension-watermark",
    )
    security = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "list_date": [date(2020, 1, 2)],
            "delist_date": [pd.NaT],
            "is_st": [False],
            "valid_from": [date(2020, 1, 2)],
            "valid_to": [pd.NaT],
        }
    )

    batch = build_tradability_batch(
        market,
        security,
        suspensions,
        TradingCalendar.from_dates([date(2025, 1, 3)]),
        RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="v1"),
        minimum_listing_trading_days=1,
        minimum_valid_days_20=15,
    )

    assert batch.object_name == "tradability_daily"
    assert batch.business_keys == ("trade_date", "ts_code", "rule_version")
    assert batch.partition_date_column == "trade_date"
    assert batch.frame.loc[0, "is_eligible_pre_liquidity"]
    assert batch.frame.loc[0, "source_watermark_id"] != "market-watermark"
