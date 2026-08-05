from __future__ import annotations

from datetime import date

import audit_formula_replay_tushare as audit
import pandas as pd
import pytest
from audit_formula_replay_consistency import REPLAY_DEFINITIONS


def test_parser_accepts_bounded_tushare_replay() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_398",
            "--max-ts-codes",
            "64",
            "--start",
            "2020-09-14",
            "--end",
            "2020-09-25",
            "--output",
            "/tmp/tushare-replay.json",
        ]
    )

    assert args.factor_id == ("yeslab_398",)
    assert args.max_ts_codes == 64
    assert args.start == date(2020, 9, 14)


def test_parser_rejects_locked_test_and_unsupported_factor() -> None:
    with pytest.raises(ValueError, match="2026"):
        audit.parse_args(
            [
                "--factor-id",
                "yeslab_398",
                "--start",
                "2026-01-01",
                "--end",
                "2026-01-02",
                "--output",
                "/tmp/tushare-replay.json",
            ]
        )
    with pytest.raises(ValueError, match="unsupported"):
        audit.parse_args(
            [
                "--factor-id",
                "yeslab_443",
                "--output",
                "/tmp/tushare-replay.json",
            ]
        )


def test_market_frame_is_normalized_and_restricted_to_codes() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000002.SZ", "000001.SZ"],
            "trade_date": ["20200915", "20200915"],
            "open": [2.0, 1.0],
            "low": [1.5, 0.5],
            "amount": [20.123, 10.456],
            "pct_chg": [1.2, -0.4],
        }
    )

    normalized = audit.normalize_tushare_market(raw, codes=("000001.SZ",))

    assert normalized.to_dict("records") == [
        {
            "trade_date": pd.Timestamp("2020-09-15"),
            "ts_code": "000001.SZ",
            "open": 1.0,
            "low": 0.5,
            "amount": 10.456,
            "pct_chg": -0.4,
        }
    ]


def test_query_codes_deduplicates_top_factor_rows() -> None:
    class Client:
        def run(self, script: str) -> pd.DataFrame:
            assert "select distinct ts_code" in script
            return pd.DataFrame({"ts_code": ["000002.SZ", "000001.SZ", "000001.SZ"]})

    codes = audit._query_codes(
        Client(), REPLAY_DEFINITIONS["yeslab_398"], date(2020, 9, 14), date(2020, 9, 25), 2
    )

    assert codes == ("000001.SZ", "000002.SZ")
