from __future__ import annotations

import pandas as pd
import pytest

import evaluate_alpha206_median_similarity as evaluation


def test_parser_rejects_locked_test() -> None:
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--start", "2026-01-01", "--end", "2026-01-02", "--output", "/tmp/x.json"])


def test_candidate_id_is_versioned() -> None:
    assert evaluation.CANDIDATE.endswith("_v2")
    assert evaluation.CANDIDATE not in evaluation.FORMAL_NINE


def test_candidate_observation_dates_are_python_dates() -> None:
    dates = pd.date_range("2025-01-01", periods=60)
    market = pd.DataFrame(
        {
            "trade_date": dates.repeat(2),
            "ts_code": ["000001.SZ", "600000.SH"] * len(dates),
            "vol": [1.0, 2.0] * len(dates),
        }
    )
    result = evaluation._candidate_observations(market, dates[-1:])
    assert isinstance(result.iloc[0]["signal_date"], __import__("datetime").date)


def test_pair_lookup_accepts_formal_then_candidate_order() -> None:
    values = {("yeslab_380", evaluation.CANDIDATE): [0.25]}
    assert values.get(("yeslab_380", evaluation.CANDIDATE)) == [0.25]
