from __future__ import annotations

import pandas as pd
from audit_factor_market_coverage import compare_factor_market_coverage


def test_compare_factor_market_coverage_counts_finite_orphans() -> None:
    source = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-02", "2020-01-03", "2020-01-03"]),
            "ts_code": ["000001.SZ", "000001.SZ", "000002.SZ"],
            "value": [1.0, 2.0, float("nan")],
        }
    )
    raw = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-02"]),
            "ts_code": ["000001.SZ"],
        }
    )

    result = compare_factor_market_coverage(source, raw)

    assert result["source_rows"] == 3
    assert result["source_finite_rows"] == 2
    assert result["finite_orphan_rows"] == 1
    assert result["finite_orphan_codes"] == 1
    assert result["finite_orphan_dates"] == 1


def test_compare_factor_market_coverage_passes_when_keys_are_present() -> None:
    source = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-02"]),
            "ts_code": ["000001.SZ"],
            "value": [1.0],
        }
    )
    raw = source.loc[:, ["trade_date", "ts_code"]]

    result = compare_factor_market_coverage(source, raw)

    assert result["finite_orphan_rows"] == 0
    assert result["decision_status"] == "MARKET_COVERAGE_PASS"
