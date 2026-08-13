from __future__ import annotations

from pathlib import Path

import audit_formal_monitor_regimes as audit
import pandas as pd


def test_load_rank_ic_filters_locked_test_and_deduplicates(tmp_path: Path) -> None:
    path = tmp_path / "rank_ic.csv"
    pd.DataFrame(
        {
            "factor_id": ["yeslab_380", "yeslab_380", "yeslab_431"],
            "signal_date": ["2025-12-31", "2026-01-02", "2025-12-31"],
            "rank_ic": [-0.1, -0.2, -0.3],
        }
    ).to_csv(path, index=False)

    result = audit.load_rank_ic([path], ["yeslab_380", "yeslab_431"])

    assert len(result) == 2
    assert result["signal_date"].max() < audit.LOCKED_TEST_START


def test_load_market_states_excludes_bj(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "600000.SH", "830001.BJ"],
            "trade_date": ["20200101"] * 3,
            "pct_chg": [1.0, 3.0, 100.0],
        }
    )
    frame.to_json(tmp_path / "daily_20200101.json", orient="table")

    result = audit.load_market_states(
        tmp_path, pd.Timestamp("2020-01-01"), pd.Timestamp("2020-01-01")
    )

    assert result["market_return"].tolist() == [0.02]


def test_audit_marks_regime_below_threshold() -> None:
    dates = pd.date_range("2020-01-01", periods=2)
    rank_ic = pd.DataFrame(
        {
            "factor_id": ["yeslab_380"] * 2,
            "signal_date": dates,
            "rank_ic": [0.01, 0.01],
        }
    )
    states = pd.DataFrame(
        {
            "signal_date": dates,
            "trend_regime": ["positive", "positive"],
            "volatility_regime": ["low", "low"],
        }
    )
    train = pd.concat(
        [rank_ic] + [rank_ic.assign(factor_id=factor) for factor in audit.FORMAL_MONITORED[1:]]
    )
    validation = train.copy()

    result = audit.audit(train, validation, states)

    assert "trend:positive" in result["yeslab_380"]["train_2020_2024"]["failed_regimes"]
    assert result["yeslab_431"]["train_2020_2024"]["failed_regimes"]