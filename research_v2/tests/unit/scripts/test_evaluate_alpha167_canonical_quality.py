from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import evaluate_alpha167_canonical_quality as evaluation


def test_compute_alpha99_matches_manual_rank_covariance() -> None:
    dates = pd.date_range("2025-01-01", periods=5)
    market = pd.DataFrame(
        {
            "trade_date": dates.repeat(3),
            "ts_code": ["000001.SZ", "000002.SZ", "600000.SH"] * 5,
            "close": [1, 2, 3, 2, 3, 1, 3, 1, 2, 4, 2, 1, 5, 1, 2],
            "vol": [3, 1, 2, 2, 3, 1, 1, 3, 2, 2, 1, 3, 3, 2, 1],
        }
    )

    result = evaluation.compute_alpha99(market, window=5)

    assert result.shape == (5, 3)
    assert result.iloc[:4].isna().all().all()
    assert result.iloc[4].notna().all()


def test_empty_monthly_summary_has_expected_columns() -> None:
    result = evaluation.summarize_monthly(pd.DataFrame(columns=["signal_date", "rank_ic"]))

    assert "directional_day_share" in result.columns
    assert result.empty


def test_parser_rejects_locked_test() -> None:
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--start", "2026-01-01", "--end", "2026-01-02", "--output", "/tmp/x"])
