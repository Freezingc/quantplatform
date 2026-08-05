from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from research_v2.factors.turnover_repair import compute_turnover_gap_candidate


def _market(days: int = 70) -> pd.DataFrame:
    dates = [date(2020, 1, 1) + timedelta(days=index) for index in range(days)]
    rows: list[dict[str, object]] = []
    for day_index, trade_date in enumerate(dates):
        for symbol_index, ts_code in enumerate(("000001.SZ", "000002.SZ", "000003.SZ")):
            base = float(day_index * 10 + symbol_index + 1)
            rows.append(
                {
                    "trade_date": trade_date,
                    "ts_code": ts_code,
                    "open": base + 0.1,
                    "close": base + 0.2,
                    "high": base + 0.3,
                    "vol": base + 1.0,
                    "amount": base + 2.0,
                    "pct_chg": float(symbol_index - 1),
                    "turnover_rate": base / 100.0,
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "factor_name",
    (
        "GP_008",
        "GP_011",
        "GP_013",
        "GP_016",
        "GP_018",
        "GP_019",
        "GP_026",
        "GP_031",
        "GP_070",
        "GP_087",
    ),
)
def test_turnover_gap_candidates_have_canonical_long_shape(factor_name: str) -> None:
    result = compute_turnover_gap_candidate(factor_name, _market())

    assert list(result.columns) == ["trade_date", "ts_code", "candidate_value"]
    assert len(result) == 70 * 3
    assert np.isfinite(result["candidate_value"].to_numpy(dtype=float)).any()


def test_turnover_gap_candidate_rejects_unknown_formula() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        compute_turnover_gap_candidate("GP_999", _market())
