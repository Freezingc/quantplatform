from __future__ import annotations

import evaluate_alpha95_relative_amount_volatility as evaluation
import numpy as np
import pandas as pd


def make_market(values: list[float]) -> pd.DataFrame:
    dates = pd.date_range("2025-01-01", periods=len(values))
    return pd.DataFrame(
        {
            "trade_date": dates.repeat(2),
            "ts_code": ["000001.SZ", "600000.SH"] * len(values),
            "amount": np.repeat(values, 2),
        }
    )


def test_relative_amount_volatility_uses_complete_trailing_window() -> None:
    market = pd.DataFrame(
        {
            "trade_date": pd.date_range("2025-01-01", periods=4).repeat(2),
            "ts_code": ["000001.SZ", "600000.SH"] * 4,
            "amount": [1.0, 2.0, 2.0, 4.0, 3.0, 6.0, 4.0, 8.0],
        }
    )

    result = evaluation.compute_relative_amount_volatility(market, window=3)

    assert result.iloc[:2].isna().all().all()
    np.testing.assert_allclose(result.iloc[2].to_numpy(), [0.5, 0.5])


def test_relative_amount_volatility_replaces_zero_baseline_with_missing() -> None:
    result = evaluation.compute_relative_amount_volatility(make_market([0.0, 0.0, 0.0]), 3)

    assert result.iloc[2].isna().all()


def test_relative_amount_volatility_does_not_use_future_amount() -> None:
    base = make_market([1.0, 2.0, 3.0, 4.0, 5.0])
    changed = base.copy()
    changed.loc[changed["trade_date"] == pd.Timestamp("2025-01-05"), "amount"] = 999.0

    before = evaluation.compute_relative_amount_volatility(base, 3)
    after = evaluation.compute_relative_amount_volatility(changed, 3)

    assert before.iloc[2].equals(after.iloc[2])
