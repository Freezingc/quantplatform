from __future__ import annotations

from datetime import UTC, date, datetime

import pandas as pd

from research_v2.contracts.source import SourceBatch
from research_v2.materializers.universe import build_universe_batch
from research_v2.runs.context import RunContext
from research_v2.universe.builder import UniverseDefinition
from research_v2.universe.rules import UniverseRules


def test_universe_materializer_publishes_versioned_membership_with_pit_board() -> None:
    tradability = SourceBatch(
        frame=pd.DataFrame(
            {
                "trade_date": [date(2025, 1, 3), date(2025, 1, 3)],
                "ts_code": ["000001.SZ", "430017.BJ"],
                "is_listed": [True, True],
                "is_st": [False, False],
                "is_suspended": [False, False],
                "listing_age_trading_days": [300, 300],
                "valid_days_20": [20, 20],
                "adv20": [100.0, 200.0],
                "reason_bitmap": [0, 0],
            }
        ),
        source_name="QuantResearchV2Test",
        table_name="tradability_daily",
        watermark_id="tradability-watermark",
        max_observation_date=date(2025, 1, 3),
        source_row_count=2,
        schema_fingerprint="schema",
        extracted_at=datetime(2025, 1, 4, tzinfo=UTC),
    )
    security_scd = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "430017.BJ"],
            "board": ["SZ_MAIN", "BEIJING"],
            "valid_from": [date(2020, 1, 1), date(2021, 11, 15)],
            "valid_to": [pd.NaT, pd.NaT],
        }
    )
    definition = UniverseDefinition(
        universe_id="cn_liquid_a",
        version=1,
        rules=UniverseRules(retained_adv20_fraction=1.0),
    )

    batch = build_universe_batch(
        tradability,
        security_scd,
        definition,
        RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="v1"),
    )

    assert batch.object_name == "universe_daily"
    assert batch.business_keys == ("signal_date", "universe_id", "universe_version", "ts_code")
    assert batch.partition_date_column == "signal_date"
    indexed = batch.frame.set_index("ts_code")
    assert indexed.loc["000001.SZ", "is_member"]
    assert not indexed.loc["430017.BJ", "is_member"]
    assert indexed.loc["430017.BJ", "exclusion_reason"] == "OUTSIDE_BOARD"


def test_universe_materializer_normalizes_dolphindb_scd_timestamps() -> None:
    tradability = SourceBatch(
        frame=pd.DataFrame(
            {
                "trade_date": [date(2025, 1, 3)],
                "ts_code": ["000001.SZ"],
                "is_listed": [True],
                "is_st": [False],
                "is_suspended": [False],
                "listing_age_trading_days": [300],
                "valid_days_20": [20],
                "adv20": [100.0],
                "reason_bitmap": [0],
            }
        ),
        source_name="QuantResearchV2Test",
        table_name="tradability_daily",
        watermark_id="tradability-watermark",
        max_observation_date=date(2025, 1, 3),
        source_row_count=1,
        schema_fingerprint="schema",
        extracted_at=datetime(2025, 1, 4, tzinfo=UTC),
    )

    batch = build_universe_batch(
        tradability,
        pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "board": ["SZ_MAIN"],
                "valid_from": [pd.Timestamp("2020-01-01")],
                "valid_to": [pd.NaT],
            }
        ),
        UniverseDefinition("cn_liquid_a", 1, UniverseRules(retained_adv20_fraction=1.0)),
        RunContext(run_id="00000000-0000-0000-0000-000000000001", data_version="v1"),
    )

    assert batch.frame.loc[0, "is_member"]
