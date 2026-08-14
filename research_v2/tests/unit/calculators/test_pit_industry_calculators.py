from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

DATABASE_PATH = Path(__file__).resolve().parents[4] / "数据库"
if str(DATABASE_PATH) not in sys.path:
    sys.path.insert(0, str(DATABASE_PATH))

from factor_calculators.AttentionSpilloverCalculator import (  # noqa: E402
    AttentionSpilloverCalculator,
)
from factor_calculators.IndustryStandAbsCalculator import (  # noqa: E402
    IndustryStandAbsCalculator,
)
from factor_calculators.IndustryStandAbsDivSkewAbsCalculator import (  # noqa: E402
    IndustryStandAbsDivSkewAbsCalculator,
)


class LabelProvider:
    def __init__(self, labels: pd.DataFrame) -> None:
        self.labels = labels
        self.calls: list[tuple[tuple[pd.Timestamp, ...], tuple[str, ...]]] = []

    def get_industry_labels(
        self, dates: list[pd.Timestamp], codes: list[str]
    ) -> pd.DataFrame:
        date_values = tuple(pd.Timestamp(value) for value in dates)
        code_values = tuple(map(str, codes))
        self.calls.append((date_values, code_values))
        return self.labels.reindex(index=date_values, columns=code_values)


def _market_frame(fields: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for field, frame in fields.items():
        copy = frame.copy()
        copy.index = pd.MultiIndex.from_product([[field], copy.index])
        rows.append(copy)
    return pd.concat(rows)


def test_industry_standardization_uses_date_specific_labels() -> None:
    dates = pd.date_range("2020-01-01", periods=4, freq="D")
    codes = ["000001.SZ", "000002.SZ", "600000.SH"]
    close = pd.DataFrame(
        {
            codes[0]: [100.0, 110.0, 110.0, 121.0],
            codes[1]: [100.0, 100.0, 110.0, 110.0],
            codes[2]: [100.0, 100.0, 100.0, 100.0],
        },
        index=dates,
    )
    labels = pd.DataFrame(
        [
            ["g", "g", "g"],
            ["g", "g", "g"],
            ["g", "g", "h"],
            ["g", "g", "h"],
        ],
        index=dates,
        columns=codes,
    )
    provider = LabelProvider(labels)
    calculator = IndustryStandAbsCalculator(None, 1)
    calculator.industry_provider = provider
    calculator.fetch_data = lambda start, end: _market_frame({"close": close})

    result = calculator._calculate("2020.01.01", "2020.01.04")

    assert provider.calls == [(tuple(dates), tuple(codes))]
    assert np.isfinite(result.loc[dates[2], codes[0]])
    assert pd.isna(result.loc[dates[2], codes[2]])


def test_attention_spillover_does_not_query_static_industry_when_provider_is_set() -> None:
    dates = pd.date_range("2020-01-01", periods=22, freq="D")
    codes = [f"00000{i}.SZ" for i in range(1, 7)]
    pct_chg = pd.DataFrame(
        np.tile(np.arange(1.0, 7.0), (len(dates), 1)), index=dates, columns=codes
    )
    total_mv = pd.DataFrame(
        np.tile(np.arange(1.0, 7.0), (len(dates), 1)), index=dates, columns=codes
    )
    labels = pd.DataFrame("g", index=dates, columns=codes, dtype=object)
    labels.loc[dates[-1], codes[-1]] = "h"
    provider = LabelProvider(labels)

    class Session:
        def run(self, script: str) -> object:
            raise AssertionError("static industry query must not be used")

    calculator = AttentionSpilloverCalculator(Session(), 21)
    calculator.industry_provider = provider
    calculator.fetch_data = lambda start, end: _market_frame(
        {"pct_chg": pct_chg, "total_mv": total_mv}
    )

    result = calculator._calculate("2020.01.01", "2020.01.22")

    assert provider.calls == [(tuple(dates), tuple(codes))]
    assert result.loc[dates[-1]].notna().any()


def test_industry_skew_abs_variant_takes_abs_after_rolling_mean(monkeypatch) -> None:
    dates = pd.date_range("2020-01-01", periods=5, freq="D")
    codes = ["000001.SZ"]
    close = pd.DataFrame({codes[0]: [100.0] * len(dates)}, index=dates)
    zscore = pd.DataFrame(1.0, index=dates, columns=codes)
    daily_skew = pd.DataFrame(
        [-1.0, -1.0, 1.0, 1.0, 1.0], index=dates, columns=codes
    )
    calculator = IndustryStandAbsDivSkewAbsCalculator(None, 5)
    calculator.industry_provider = object()
    calculator.fetch_data = lambda start, end: _market_frame({"close": close})
    monkeypatch.setattr(
        calculator,
        "get_industry_labels",
        lambda dates, codes: pd.DataFrame("g", index=dates, columns=codes),
    )
    monkeypatch.setattr(
        calculator,
        "industry_zscore_and_skew",
        lambda stock_ret, labels: (zscore, daily_skew),
    )

    result = calculator._calculate("2020.01.01", "2020.01.05")

    expected = 1.0 / daily_skew.rolling(5).mean().abs()
    assert result.loc[dates[-1], codes[0]] == expected.loc[dates[-1], codes[0]]
