from __future__ import annotations

from datetime import date

import pandas as pd

from research_v2.pit.reconciliation import reconcile_market, should_sample_history


def test_reconciliation_accepts_values_within_absolute_or_relative_tolerance() -> None:
    primary = pd.DataFrame(
        [
            {
                "trade_date": date(2025, 1, 3),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "volume": 1000.0,
                "amount": 10000.0,
                "listing_status": "L",
            }
        ]
    )
    secondary = primary.copy()
    secondary.loc[0, "open"] = 10.005
    secondary.loc[0, "volume"] = 1000.9

    assert reconcile_market(primary, secondary).empty


def test_reconciliation_reports_price_and_status_mismatches_without_mutating_primary() -> None:
    primary = pd.DataFrame(
        [
            {
                "trade_date": date(2025, 1, 3),
                "ts_code": "000001.SZ",
                "open": 10.0,
                "volume": 1000.0,
                "amount": 10000.0,
                "listing_status": "L",
            }
        ]
    )
    secondary = primary.copy()
    secondary.loc[0, "open"] = 10.02
    secondary.loc[0, "listing_status"] = "D"

    result = reconcile_market(primary, secondary)

    assert set(result["field"]) == {"open", "listing_status"}
    assert primary.loc[0, "open"] == 10.0
    assert primary.loc[0, "listing_status"] == "L"


def test_historical_sampling_is_deterministic() -> None:
    keys = [(date(2025, 1, 3), f"000{i:03d}.SZ") for i in range(100)]
    first = [key for key in keys if should_sample_history(*key)]
    second = [key for key in reversed(keys) if should_sample_history(*key)]

    assert sorted(first) == sorted(second)
    assert 0 < len(first) < len(keys)
