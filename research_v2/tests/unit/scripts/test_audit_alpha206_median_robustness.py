from __future__ import annotations

from pathlib import Path

import audit_alpha206_median_robustness as audit
import numpy as np
import pandas as pd
import pytest


def test_read_rank_ic_excludes_locked_test(tmp_path: Path) -> None:
    path = tmp_path / "rank_ic.csv"
    pd.DataFrame({"signal_date": ["2025-12-31", "2026-01-02"], "rank_ic": [-0.1, -0.2]}).to_csv(
        path, index=False
    )
    result = audit._read_rank_ic(path)
    assert result["signal_date"].dt.year.tolist() == [2025]


def test_leave_one_month_preserves_signed_direction() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": pd.date_range("2025-01-01", periods=6, freq="MS"),
            "rank_ic": [-0.1, -0.2, -0.3, -0.4, -0.5, -0.6],
        }
    )
    result = audit.leave_one_month(frame)
    assert result["minimum_leave_one_month_signed_mean_rank_ic"] > 0.02
    assert result["worst_signed_mean_exclusion"] == "2025-06"


def test_month_block_bootstrap_is_reproducible() -> None:
    frame = pd.DataFrame(
        {
            "signal_date": pd.date_range("2025-01-01", periods=12, freq="MS"),
            "rank_ic": np.full(12, -0.1),
        }
    )
    left = audit.month_block_bootstrap(frame, 200, np.random.default_rng(audit.SEED))
    right = audit.month_block_bootstrap(frame, 200, np.random.default_rng(audit.SEED))
    assert left == right
    assert left["ci95"] == pytest.approx([0.1, 0.1])


def test_market_state_loader_excludes_bj(tmp_path: Path) -> None:
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