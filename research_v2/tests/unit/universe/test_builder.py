from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.universe.builder import UniverseDefinition, build_universe_snapshot
from research_v2.universe.rules import UniverseRules


def _candidate(trade_date: date, ts_code: str, adv20: float) -> dict[str, object]:
    return {
        "trade_date": trade_date,
        "ts_code": ts_code,
        "board": "SZ_MAIN",
        "is_listed": True,
        "is_st": False,
        "is_suspended": False,
        "listing_age_trading_days": 300,
        "valid_days_20": 20,
        "adv20": adv20,
    }


def test_universe_snapshot_is_versioned_and_ranks_members_stably() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(date(2025, 1, 3), "000002.SZ", 100.0),
            _candidate(date(2025, 1, 3), "000001.SZ", 100.0),
            _candidate(date(2025, 1, 3), "000003.SZ", 80.0),
        ]
    )
    definition = UniverseDefinition(
        universe_id="cn_liquid_a",
        version=1,
        rules=UniverseRules(retained_adv20_fraction=2 / 3),
    )

    snapshot = build_universe_snapshot(candidates, definition)

    assert snapshot["signal_date"].tolist() == [date(2025, 1, 3)] * 3
    assert snapshot["universe_id"].tolist() == ["cn_liquid_a"] * 3
    assert snapshot["universe_version"].tolist() == [1] * 3
    assert snapshot.loc[snapshot["is_member"], "ts_code"].tolist() == ["000001.SZ", "000002.SZ"]
    assert snapshot.set_index("ts_code").loc["000001.SZ", "adv20_rank"] == 1
    assert snapshot.set_index("ts_code").loc["000002.SZ", "membership_rank"] == 2
    assert snapshot.set_index("ts_code").loc["000003.SZ", "exclusion_reason"] == "LOW_LIQUIDITY"


def test_universe_snapshot_rejects_multiple_signal_dates() -> None:
    candidates = pd.DataFrame(
        [
            _candidate(date(2025, 1, 3), "000001.SZ", 100.0),
            _candidate(date(2025, 1, 6), "000002.SZ", 100.0),
        ]
    )

    with pytest.raises(ValueError, match="exactly one signal date"):
        build_universe_snapshot(
            candidates,
            UniverseDefinition(universe_id="cn_liquid_a", version=1, rules=UniverseRules()),
        )
