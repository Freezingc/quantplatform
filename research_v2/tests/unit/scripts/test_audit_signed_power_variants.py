from __future__ import annotations

import audit_signed_power_variants as audit
import numpy as np
import pandas as pd


def test_signed_power_preserves_sign_for_fractional_exponents() -> None:
    base = pd.Series([-4.0, 4.0, np.nan])
    exponent = pd.Series([0.5, 0.5, 0.5])

    result = audit.signed_power(base, exponent)

    assert result.iloc[:2].tolist() == [-2.0, 2.0]
    assert np.isnan(result.iloc[2])


def test_compute_variants_improves_domain_coverage() -> None:
    dates = pd.date_range("2020-01-01", periods=65, freq="D")
    rows = []
    for code in ("000001.SZ", "000002.SZ"):
        for index, day in enumerate(dates):
            rows.append(
                {
                    "trade_date": day,
                    "ts_code": code,
                    "vol": 100.0 + index,
                    "amount": 1000.0 + 5.0 * index,
                    "pct_chg": -1.0 if index % 2 else 1.0,
                    "turnover_rate": 0.5,
                }
            )
    market = pd.DataFrame(rows)

    variants = audit.compute_signed_power_variants(market)

    assert set(variants) == {"yeslab_370", "yeslab_371", "yeslab_372", "yeslab_389", "yeslab_390"}
    assert variants["yeslab_371"]["value"].notna().sum() > 0
    assert variants["yeslab_389"]["value"].notna().sum() > 0
    assert variants["yeslab_370"]["value"].notna().sum() > 0
    assert variants["yeslab_372"]["value"].notna().sum() > 0


def test_gp021_signed_power_variant_avoids_negative_fractional_power_nan() -> None:
    dates = pd.date_range("2020-01-01", periods=65, freq="D")
    rows = []
    for code in ("000001.SZ", "000002.SZ"):
        for index, day in enumerate(dates):
            rows.append(
                {
                    "trade_date": day,
                    "ts_code": code,
                    "vol": 100.0 + index,
                    "amount": 1000.0 + 5.0 * index,
                    "pct_chg": -1.0 if index % 2 else 1.0,
                    "turnover_rate": 0.5,
                }
            )

    variants = audit.compute_signed_power_variants(pd.DataFrame(rows))

    assert variants["yeslab_390"]["value"].notna().sum() > 0


def test_vectorized_slope_matches_linear_series() -> None:
    values = pd.DataFrame({"000001.SZ": np.arange(1.0, 7.0)})

    result = audit._rolling_slope_matrix(values, 3)

    assert np.allclose(result.iloc[2:, 0].to_numpy(), 1.0)
