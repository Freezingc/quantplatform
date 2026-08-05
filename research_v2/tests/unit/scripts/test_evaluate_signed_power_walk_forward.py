from __future__ import annotations

import evaluate_signed_power_walk_forward as audit
import numpy as np
import pandas as pd


def test_merge_turnover_overlay_prefers_tushare_value() -> None:
    market = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2020-09-18"), pd.Timestamp("2020-09-18")],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "vol": [100.0, 200.0],
            "pct_chg": [1.0, -1.0],
            "turnover_rate": [np.nan, 2.0],
        }
    )
    overlay = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2020-09-18")],
            "ts_code": ["000001.SZ"],
            "turnover_rate": [0.75],
        }
    )

    result = audit.merge_turnover_overlay(market, overlay)

    assert result.loc[result["ts_code"] == "000001.SZ", "turnover_rate"].item() == 0.75
    assert result.loc[result["ts_code"] == "000002.SZ", "turnover_rate"].item() == 2.0


def test_summarize_walk_forward_metrics_freezes_train_direction() -> None:
    daily = pd.DataFrame(
        {
            "factor_id": ["yeslab_371"] * 4,
            "signal_date": pd.date_range("2024-01-01", periods=4, freq="D"),
            "rank_ic": [-0.04, -0.02, -0.03, -0.01],
            "sample_count": [100] * 4,
        }
    )

    result = audit.summarize_walk_forward_metrics(daily, train_direction=-1)

    assert result.loc[0, "direction"] == -1
    assert np.isclose(result.loc[0, "signed_mean_rank_ic"], 0.025)
    assert result.loc[0, "directional_day_share"] == 1.0
