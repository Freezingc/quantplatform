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
                    "pct_chg": -1.0 if index % 2 else 1.0,
                    "turnover_rate": 0.5,
                }
            )
    market = pd.DataFrame(rows)

    variants = audit.compute_signed_power_variants(market)

    assert set(variants) == {"yeslab_371", "yeslab_389"}
    assert variants["yeslab_371"]["value"].notna().sum() > 0
    assert variants["yeslab_389"]["value"].notna().sum() > 0
