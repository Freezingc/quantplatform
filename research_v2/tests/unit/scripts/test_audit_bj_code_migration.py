from __future__ import annotations

from datetime import date

import audit_bj_code_migration as audit
import pandas as pd


def _frame(codes: list[str], values: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "trade_date": [date(2025, 1, 2), date(2025, 1, 3)] * len(codes),
            "ts_code": [code for code in codes for _ in range(2)],
            "open": [value for value in values for _ in range(2)],
            "high": [value + 1 for value in values for _ in range(2)],
            "low": [value - 1 for value in values for _ in range(2)],
            "close": [value + 0.5 for value in values for _ in range(2)],
            "pct_chg": [1.0 for _ in range(2 * len(codes))],
            "vol": [100.0 for _ in range(2 * len(codes))],
        }
    )


def test_infer_aliases_matches_unique_sequences() -> None:
    source = _frame(["830001.BJ", "830002.BJ"], [10.0, 20.0])
    tushare = _frame(["920101.BJ", "920102.BJ"], [10.0, 20.0])

    result = audit.infer_aliases(source, tushare)

    assert result.mapping == {"830001.BJ": "920101.BJ", "830002.BJ": "920102.BJ"}
    assert result.unmatched_source == ()
    assert result.ambiguous_source == ()


def test_infer_aliases_does_not_guess_ambiguous_sequences() -> None:
    source = _frame(["830001.BJ"], [10.0])
    tushare = _frame(["920101.BJ", "920102.BJ"], [10.0, 10.0])

    result = audit.infer_aliases(source, tushare)

    assert result.mapping == {}
    assert result.ambiguous_source == ("830001.BJ",)


def test_market_fingerprint_is_stable_under_row_order() -> None:
    frame = _frame(["830001.BJ"], [10.0])

    assert audit.market_fingerprint(frame.iloc[::-1]) == audit.market_fingerprint(frame)


def test_infer_aliases_allows_extra_tushare_dates() -> None:
    source = _frame(["920116.BJ"], [10.0])
    extra = source.iloc[[0]].copy()
    extra["trade_date"] = date(2025, 1, 1)
    tushare = pd.concat([source, extra], ignore_index=True)

    result = audit.infer_aliases(source, tushare)

    assert result.mapping == {"920116.BJ": "920116.BJ"}
