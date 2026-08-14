"""Tests for the isolated canonical Alpha484 admission evaluator."""

from __future__ import annotations

from datetime import date

import evaluate_alpha484_canonical as evaluation
import pandas as pd
import pytest


def test_candidate_identity_is_isolated_from_historical_factor() -> None:
    assert evaluation.CANDIDATE_ID == "yeslab_484_tushare_shsz_v1"
    assert evaluation.SOURCE_FACTOR_ID == "yeslab_484"
    assert evaluation.CANDIDATE_ID != evaluation.SOURCE_FACTOR_ID


def test_parse_args_rejects_the_locked_test_boundary() -> None:
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--end", "2026-01-01", "--output", "/tmp/out"])


def test_validate_full_market_dates_rejects_a_missing_open_date() -> None:
    market = pd.DataFrame({"trade_date": [pd.Timestamp("2025-01-02")]})

    with pytest.raises(ValueError, match="full SH/SZ"):
        evaluation.validate_full_market_dates(
            market, (date(2025, 1, 2), date(2025, 1, 3))
        )
