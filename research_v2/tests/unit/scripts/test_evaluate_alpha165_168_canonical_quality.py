from __future__ import annotations

import evaluate_alpha165_168_canonical_quality as evaluation
import numpy as np
import pandas as pd
import pytest


def test_compute_volume_std_uses_window_and_sample_std() -> None:
    dates = pd.date_range("2025-01-01", periods=3)
    market = pd.DataFrame(
        {
            "trade_date": dates.repeat(2),
            "ts_code": ["000001.SZ", "600000.SH"] * 3,
            "vol": [1.0, 2.0, 2.0, 4.0, 3.0, 6.0],
        }
    )
    result = evaluation.compute_volume_std(market, window=3)
    assert result.iloc[:2].isna().all().all()
    np.testing.assert_allclose(result.iloc[2].to_numpy(), [1.0, 2.0])


def test_empty_monthly_summary_has_expected_columns() -> None:
    result = evaluation.summarize_monthly(pd.DataFrame(columns=["signal_date", "rank_ic"]))
    assert result.empty
    assert "signed_mean_rank_ic" in result.columns


def test_parser_rejects_locked_test() -> None:
    with pytest.raises(SystemExit):
        evaluation.parse_args(
            ["--start", "2026-01-01", "--end", "2026-01-02", "--output", "/tmp/x"]
        )
