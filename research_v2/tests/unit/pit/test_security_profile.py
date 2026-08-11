from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from research_v2.contracts.source import SourceBatch
from research_v2.pit.security_profile import build_market_derived_security_profile


def _market_batch() -> SourceBatch:
    return SourceBatch(
        frame=pd.DataFrame(
            {
                "ts_code": ["300114.SZ", "300114.SZ"],
                "trade_date": [date(2013, 1, 4), date(2025, 1, 2)],
            }
        ),
        source_name="YesLab",
        table_name="stock_factor_daily",
        watermark_id="market-watermark",
        max_observation_date=date(2025, 1, 2),
        source_row_count=2,
        schema_fingerprint="schema",
        extracted_at=datetime(2025, 1, 3, tzinfo=UTC),
    )


def test_market_derived_profile_uses_earliest_trade_as_conservative_listing_date() -> None:
    batch = build_market_derived_security_profile(
        _market_batch(),
        as_of_date=date(2025, 1, 2),
        ts_codes={"300114.SZ"},
    )

    assert batch.table_name == "derived_market_security_profile"
    assert batch.frame.to_dict(orient="records") == [
        {
            "ts_code": "300114.SZ",
            "exchange": "SZSE",
            "market": "创业板",
            "list_date": date(2013, 1, 4),
            "delist_date": None,
            "name": None,
            "list_status": "L",
        }
    ]
    assert batch.watermark_id != "market-watermark"


def test_market_derived_profile_preserves_a_beijing_exchange_code() -> None:
    market = _market_batch()
    market = SourceBatch(
        frame=pd.DataFrame({"ts_code": ["430017.BJ"], "trade_date": [date(2021, 11, 15)]}),
        source_name=market.source_name,
        table_name=market.table_name,
        watermark_id=market.watermark_id,
        max_observation_date=market.max_observation_date,
        source_row_count=1,
        schema_fingerprint=market.schema_fingerprint,
        extracted_at=market.extracted_at,
    )

    batch = build_market_derived_security_profile(
        market,
        as_of_date=date(2025, 1, 2),
        ts_codes={"430017.BJ"},
    )

    assert batch.frame.loc[0, ["exchange", "market"]].tolist() == ["BSE", "北交所"]
