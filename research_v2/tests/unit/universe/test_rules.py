from __future__ import annotations

import pandas as pd

from research_v2.universe.rules import UniverseRules, build_universe


def test_universe_excludes_ineligible_securities_before_adv20_cutoff() -> None:
    candidates = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.BJ", "000004.SZ"],
            "board": ["SZ_MAIN", "SZ_MAIN", "BEIJING", "SZ_MAIN"],
            "is_listed": [True, True, True, True],
            "is_st": [False, True, False, False],
            "is_suspended": [False, False, False, True],
            "listing_age_trading_days": [300, 300, 300, 300],
            "valid_days_20": [20, 20, 20, 20],
            "adv20": [100.0, 90.0, 80.0, 70.0],
        }
    )

    universe = build_universe(candidates, UniverseRules())

    assert universe.set_index("ts_code").loc["000001.SZ", "is_member"]
    assert universe.set_index("ts_code").loc["000002.SZ", "exclusion_reason"] == "ST"
    assert universe.set_index("ts_code").loc["000003.BJ", "exclusion_reason"] == "OUTSIDE_BOARD"
    assert universe.set_index("ts_code").loc["000004.SZ", "exclusion_reason"] == "SUSPENDED"


def test_universe_drops_bottom_twenty_percent_by_stable_adv20_order() -> None:
    candidates = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ", "000005.SZ"],
            "board": ["SZ_MAIN"] * 5,
            "is_listed": [True] * 5,
            "is_st": [False] * 5,
            "is_suspended": [False] * 5,
            "listing_age_trading_days": [300] * 5,
            "valid_days_20": [20] * 5,
            "adv20": [100.0, 100.0, 90.0, 80.0, 70.0],
        }
    )

    universe = build_universe(candidates, UniverseRules())

    members = universe.loc[universe["is_member"], "ts_code"].tolist()
    assert members == ["000001.SZ", "000002.SZ", "000003.SZ", "000004.SZ"]
    assert universe.set_index("ts_code").loc["000005.SZ", "exclusion_reason"] == "LOW_LIQUIDITY"
