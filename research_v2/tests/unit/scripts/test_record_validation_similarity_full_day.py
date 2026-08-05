from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
from record_validation_similarity_full_day import build_month_dates


def test_build_month_dates_deduplicates_dates_and_keeps_validation_boundary() -> None:
    daily = pd.DataFrame(
        {
            "signal_date": [
                date(2025, 2, 3),
                date(2025, 1, 3),
                date(2025, 1, 3),
                date(2025, 2, 4),
            ]
        }
    )

    result = build_month_dates(daily, start=date(2025, 1, 1), end=date(2025, 12, 31))

    assert result == [
        ("2025-01", (date(2025, 1, 3),)),
        ("2025-02", (date(2025, 2, 3), date(2025, 2, 4))),
    ]


def test_build_month_dates_rejects_locked_test_dates() -> None:
    daily = pd.DataFrame({"signal_date": [date(2026, 1, 2)]})

    with pytest.raises(ValueError, match="outside validation window"):
        build_month_dates(daily, start=date(2025, 1, 1), end=date(2025, 12, 31))
