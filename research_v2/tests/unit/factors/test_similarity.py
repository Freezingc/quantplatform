from __future__ import annotations

from datetime import date

import pandas as pd

from research_v2.factors.similarity import compute_factor_similarity


def test_similarity_is_scope_bounded_and_uses_absolute_daily_spearman() -> None:
    rows: list[dict[str, object]] = []
    for signal_date in (date(2025, 1, 2), date(2025, 1, 3)):
        for ts_code, left, right in zip(
            ["000001.SZ", "000002.SZ", "000003.SZ"],
            [1.0, 2.0, 3.0],
            [3.0, 2.0, 1.0],
            strict=True,
        ):
            rows.extend(
                [
                    {
                        "signal_date": signal_date,
                        "ts_code": ts_code,
                        "factor_id": "a",
                        "factor_version": 1,
                        "value": left,
                    },
                    {
                        "signal_date": signal_date,
                        "ts_code": ts_code,
                        "factor_id": "b",
                        "factor_version": 1,
                        "value": right,
                    },
                ]
            )
    rows.extend(
        [
            {
                "signal_date": date(2025, 1, 6),
                "ts_code": "000001.SZ",
                "factor_id": "a",
                "factor_version": 1,
                "value": 1.0,
            },
            {
                "signal_date": date(2025, 1, 6),
                "ts_code": "000001.SZ",
                "factor_id": "b",
                "factor_version": 1,
                "value": 1.0,
            },
        ]
    )

    result = compute_factor_similarity(
        pd.DataFrame(rows),
        start_date=date(2025, 1, 2),
        end_date=date(2025, 1, 3),
        minimum_common_members=3,
        minimum_valid_days=2,
    )

    assert len(result) == 1
    assert result.loc[0, "factor_id_left"] == "a"
    assert result.loc[0, "factor_id_right"] == "b"
    assert result.loc[0, "median_daily_abs_spearman"] == 1.0
    assert result.loc[0, "valid_days"] == 2
    assert result.loc[0, "status"] == "PASS"
