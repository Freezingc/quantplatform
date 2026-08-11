from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd
import pytest

from research_v2.contracts.source import SourceBatch
from research_v2.labels.forward_return import LockedLabelAccess
from research_v2.materializers.labels import build_label_batch
from research_v2.pit.calendar import TradingCalendar
from research_v2.runs.context import RunContext


def test_label_materializer_maps_raw_five_day_label_to_v2_schema() -> None:
    market = SourceBatch(
        frame=pd.DataFrame(
            {
                "trade_date": [date(2025, 1, 6), date(2025, 1, 13)],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "adjusted_open": [10.0, 11.0],
            }
        ),
        source_name="QuantResearchV2Test",
        table_name="market_daily_clean",
        watermark_id="market-v1",
        max_observation_date=date(2025, 1, 13),
        source_row_count=2,
        schema_fingerprint="schema",
        extracted_at=datetime(2025, 1, 14, tzinfo=UTC),
    )
    calendar = TradingCalendar.from_dates(
        [
            date(2025, 1, 3),
            date(2025, 1, 6),
            date(2025, 1, 7),
            date(2025, 1, 8),
            date(2025, 1, 9),
            date(2025, 1, 10),
            date(2025, 1, 13),
        ]
    )

    batch = build_label_batch(
        pd.DataFrame({"signal_date": [date(2025, 1, 3)], "ts_code": ["000001.SZ"]}),
        market,
        calendar,
        RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="v1"),
    )

    assert batch.object_name == "label_daily"
    assert batch.business_keys == ("signal_date", "ts_code", "label_id", "label_version")
    assert batch.partition_date_column == "signal_date"
    assert batch.frame.loc[0, "raw_return"] == pytest.approx(0.1)
    assert batch.frame.loc[0, "label_id"] == "forward_open_5d"
    assert set({"run_id", "data_version", "record_hash"}).issubset(batch.frame.columns)


def test_label_materializer_binds_membership_watermark_to_label_lineage() -> None:
    market = SourceBatch(
        frame=pd.DataFrame(
            {
                "trade_date": [date(2025, 1, 6), date(2025, 1, 13)],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "adjusted_open": [10.0, 11.0],
            }
        ),
        source_name="QuantResearchV2Test",
        table_name="market_daily_clean",
        watermark_id="market-v1",
        max_observation_date=date(2025, 1, 13),
        source_row_count=2,
        schema_fingerprint="schema",
        extracted_at=datetime(2025, 1, 14, tzinfo=UTC),
    )
    calendar = TradingCalendar.from_dates(
        [
            date(2025, 1, 3),
            date(2025, 1, 6),
            date(2025, 1, 7),
            date(2025, 1, 8),
            date(2025, 1, 9),
            date(2025, 1, 10),
            date(2025, 1, 13),
        ]
    )

    batch = build_label_batch(
        pd.DataFrame({"signal_date": [date(2025, 1, 3)], "ts_code": ["000001.SZ"]}),
        market,
        calendar,
        RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="v1"),
        members_watermark_id="universe-v1",
    )

    assert batch.frame.loc[0, "source_watermark_id"] != "market-v1"


def test_label_materializer_requires_explicit_access_for_locked_year() -> None:
    market = SourceBatch(
        frame=pd.DataFrame(
            {
                "trade_date": [date(2026, 1, 5), date(2026, 1, 12)],
                "ts_code": ["000001.SZ", "000001.SZ"],
                "adjusted_open": [10.0, 11.0],
            }
        ),
        source_name="QuantResearchV2Test",
        table_name="market_daily_clean",
        watermark_id="market-v1",
        max_observation_date=date(2026, 1, 12),
        source_row_count=2,
        schema_fingerprint="schema",
        extracted_at=datetime(2026, 1, 13, tzinfo=UTC),
    )
    calendar = TradingCalendar.from_dates(
        [
            date(2026, 1, 2),
            date(2026, 1, 5),
            date(2026, 1, 6),
            date(2026, 1, 7),
            date(2026, 1, 8),
            date(2026, 1, 9),
            date(2026, 1, 12),
        ]
    )

    batch = build_label_batch(
        pd.DataFrame({"signal_date": [date(2026, 1, 2)], "ts_code": ["000001.SZ"]}),
        market,
        calendar,
        RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="v1"),
        locked_access=LockedLabelAccess(2026, "user-approved-final-evaluation"),
    )

    assert batch.frame.loc[0, "label_valid"]
