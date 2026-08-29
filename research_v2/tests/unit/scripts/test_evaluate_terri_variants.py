from __future__ import annotations

import evaluate_terri_variants as evaluation
import numpy as np
import pandas as pd
import pytest


def test_market_benchmarks_are_cross_sectional_and_deterministic() -> None:
    close = pd.DataFrame(
        [[10.0, 20.0, 100.0], [11.0, 18.0, 101.0]],
        index=pd.date_range("2025-01-01", periods=2),
        columns=["a", "b", "c"],
    )
    pct = pd.DataFrame(
        [[1.0, 2.0, 100.0], [3.0, 5.0, 7.0]],
        index=close.index,
        columns=close.columns,
    )

    result = evaluation.market_benchmarks(close, pct)

    assert result["equal_weight_return"].iloc[1] == 5.0
    assert result["median_return"].iloc[1] == 5.0
    assert result["trimmed_return_10pct"].iloc[1] == 5.0
    assert np.isnan(result["price_mean_change"].iloc[0])


def test_compute_variant_panel_requires_twenty_observations() -> None:
    dates = pd.date_range("2025-01-01", periods=21)
    market = pd.DataFrame(
        {
            "trade_date": dates.repeat(2),
            "ts_code": ["000001.SZ", "600000.SH"] * 21,
            "close": [10.0 + i for i in range(21)] * 2,
            "pct_chg": [1.0, -1.0] * 21,
        }
    )

    factors = evaluation.compute_variant_panel(market)

    assert set(factors) == set(evaluation.VARIANT_NAMES)
    assert all(frame.index.equals(dates) for frame in factors.values())
    assert all(frame.iloc[:19].isna().all().all() for frame in factors.values())


def test_restrict_market_to_panel_codes_excludes_non_panel_sh_sz_rows() -> None:
    market = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2025-01-02"] * 3),
            "ts_code": ["000001.SZ", "600000.SH", "000002.SZ"],
            "close": [10.0, 20.0, 30.0],
            "pct_chg": [1.0, 2.0, 3.0],
        }
    )

    result = evaluation.restrict_market_to_panel_codes(market, ["000001.SZ", "600000.SH"])

    assert result.ts_code.tolist() == ["000001.SZ", "600000.SH"]


def test_summarize_months_uses_frozen_direction() -> None:
    daily = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(["2025-01-02", "2025-01-03", "2025-02-03"]),
            "rank_ic": [-0.1, -0.2, 0.1],
            "sample_count": [100, 100, 100],
        }
    )

    result = evaluation.summarize_months(daily)

    january = result.loc[result["month"] == "2025-01"].iloc[0]
    assert january["signed_mean_rank_ic"] == pytest.approx(0.15)
    assert january["directional_day_share"] == 1.0
