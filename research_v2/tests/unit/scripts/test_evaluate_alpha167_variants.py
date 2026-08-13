from __future__ import annotations

import evaluate_alpha167_variants as evaluation
import pandas as pd
import pytest


def test_compute_variants_keeps_five_day_warmup() -> None:
    dates = pd.date_range("2025-01-01", periods=5)
    market = pd.DataFrame(
        {
            "trade_date": dates.repeat(3),
            "ts_code": ["000001.SZ", "000002.SZ", "600000.SH"] * 5,
            "close": [1, 2, 3, 2, 3, 1, 3, 1, 2, 4, 2, 1, 5, 1, 2],
            "vol": [3, 1, 2, 2, 3, 1, 1, 3, 2, 2, 1, 3, 3, 2, 1],
        }
    )
    result = evaluation.compute_variants(market)
    assert tuple(result) == evaluation.VARIANTS
    for panel in result.values():
        assert panel.iloc[:4].isna().all().all()
        assert panel.iloc[4].notna().all()


def test_variants_have_the_same_contract_and_shape() -> None:
    dates = pd.date_range("2025-01-01", periods=6)
    market = pd.DataFrame(
        {
            "trade_date": dates.repeat(4),
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ", "600000.SH"] * 6,
            "close": [1, 2, 3, 4, 4, 2, 1, 3, 3, 4, 2, 1, 1, 3, 4, 2, 2, 1, 4, 3, 4, 3, 1, 2],
            "vol": [4, 1, 3, 2, 1, 4, 2, 3, 3, 2, 4, 1, 2, 3, 1, 4, 4, 2, 3, 1, 1, 3, 2, 4],
        }
    )
    result = evaluation.compute_variants(market)
    assert set(result) == set(evaluation.VARIANTS)
    assert result["raw_covariance"].shape == result["rolling_correlation"].shape
    assert result["raw_covariance"].iloc[4].notna().all()
    assert result["rolling_correlation"].iloc[4].notna().all()


def test_parser_rejects_locked_test() -> None:
    with pytest.raises(SystemExit):
        evaluation.parse_args(
            ["--start", "2026-01-01", "--end", "2026-01-02", "--output", "/tmp/x"]
        )
