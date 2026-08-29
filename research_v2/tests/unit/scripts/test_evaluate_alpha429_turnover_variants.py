from __future__ import annotations

import evaluate_alpha429_turnover_variants as evaluation
import numpy as np
import pandas as pd
import pytest


def test_compute_variants_preserves_formula_windows() -> None:
    dates = pd.date_range("2025-01-01", periods=25)
    market = pd.DataFrame(
        {
            "trade_date": dates.repeat(2),
            "ts_code": ["000001.SZ", "600000.SH"] * len(dates),
            "pct_chg": np.tile(np.arange(1.0, 26.0), 2),
            "turnover_rate": np.tile(np.linspace(1.0, 5.0, len(dates)), 2),
        }
    )
    result = evaluation.compute_variants(market)
    assert tuple(result) == evaluation.VARIANTS
    assert all(frame.shape == (25, 2) for frame in result.values())
    # MAX_10 followed by STD_10 becomes valid on the 19th observation.
    assert result["raw_turnover"].iloc[:18].isna().all().all()
    assert result["raw_turnover"].iloc[18:].notna().all().all()


def test_parser_rejects_locked_test() -> None:
    with pytest.raises(SystemExit):
        evaluation.parse_args(
            ["--start", "2026-01-01", "--end", "2026-01-02", "--output", "/tmp/x"]
        )


def test_daily_basic_filename_date_is_supported(tmp_path) -> None:
    daily = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20250102"],
            "open": [1.0],
            "high": [1.0],
            "low": [1.0],
            "close": [1.0],
            "pre_close": [1.0],
            "vol": [1.0],
            "amount": [1.0],
            "pct_chg": [0.0],
        }
    )
    basic = pd.DataFrame({"ts_code": ["000001.SZ"], "turnover_rate": [1.0]})
    daily.to_json(tmp_path / "daily_20250102.json", orient="table")
    basic.to_json(tmp_path / "daily_basic_20250102.json", orient="table")
    result = evaluation.load_cached_market(
        tmp_path,
        pd.Timestamp("2025-01-02").date(),
        pd.Timestamp("2025-01-02").date(),
        ["000001.SZ"],
    )
    assert result.loc[0, "turnover_rate"] == 1.0
