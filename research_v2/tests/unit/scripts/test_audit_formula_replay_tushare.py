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


def test_parser_accepts_full_market_batches() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_438",
            "--full-market",
            "--restrict-to-v2-universe",
            "--include-daily-basic",
            "--stock-batch-size",
            "256",
            "--start",
            "2020-09-14",
            "--end",
            "2020-09-25",
            "--output",
            "/tmp/tushare-replay.json",
        ]
    )

    assert args.full_market is True
    assert args.restrict_to_v2_universe is True
    assert args.include_daily_basic is True
    assert args.stock_batch_size == 256


def test_parser_accepts_isolated_cross_sectional_blocked_factors() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_395",
            "--full-market",
            "--restrict-to-v2-universe",
            "--stock-batch-size",
            "5000",
            "--start",
            "2020-09-14",
            "--end",
            "2020-09-25",
            "--output",
            "/tmp/tushare-gp026.json",
        ]
    )

    assert args.factor_id == ("yeslab_395",)
    assert args.stock_batch_size == 5000


def test_parser_accepts_low_coverage_gp_replay_factors() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_390",
            "--factor-id",
            "yeslab_404",
            "--full-market",
            "--stock-batch-size",
            "5000",
            "--start",
            "2020-09-14",
            "--end",
            "2020-09-25",
            "--output",
            "/tmp/tushare-low-coverage-gp.json",
        ]
    )

    assert args.factor_id == ("yeslab_390", "yeslab_404")


def test_default_factor_registry_has_unique_ids() -> None:
    args = audit.parse_args(["--output", "/tmp/tushare-default.json"])

    assert len(args.factor_id) == len(set(args.factor_id))


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
                "yeslab_999",
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


def test_normalize_tushare_market_preserves_daily_basic_turnover() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20200918"],
            "open": [1.0],
            "low": [0.5],
            "amount": [10.456],
            "pct_chg": [-0.4],
            "turnover_rate": [0.7076],
        }
    )

    normalized = audit.normalize_tushare_market(raw, codes=("000001.SZ",))

    assert normalized.loc[0, "turnover_rate"] == pytest.approx(0.7076)


def test_query_codes_deduplicates_top_factor_rows() -> None:
    class Client:
        def run(self, script: str) -> pd.DataFrame:
            assert "select distinct ts_code" in script
            assert "isValid(value)" in script
            return pd.DataFrame({"ts_code": ["000002.SZ", "000001.SZ", "000001.SZ"]})

    codes = audit._query_codes(
        Client(), REPLAY_DEFINITIONS["yeslab_398"], date(2020, 9, 14), date(2020, 9, 25), 2
    )

    assert codes == ("000001.SZ", "000002.SZ")


def test_aggregate_comparisons_preserves_mismatch_gate() -> None:
    result = audit.aggregate_comparisons(
        [
            {
                "source_rows": 10,
                "replay_rows": 10,
                "overlap_rows": 10,
                "overlap_finite_rows": 9,
                "mismatch_rows": 0,
                "mismatch_date_count": 0,
                "mismatch_start": None,
                "mismatch_end": None,
                "max_abs_diff": 0.1,
                "max_rel_diff": 0.2,
                "decision_status": "FORMULA_REPLAY_PASS",
                "reason_codes": [],
            },
            {
                "source_rows": 10,
                "replay_rows": 10,
                "overlap_rows": 10,
                "overlap_finite_rows": 8,
                "mismatch_rows": 2,
                "mismatch_date_count": 1,
                "mismatch_start": "2020-09-18",
                "mismatch_end": "2020-09-18",
                "max_abs_diff": 0.3,
                "max_rel_diff": 0.4,
                "decision_status": "FORMULA_REPLAY_BLOCKED",
                "reason_codes": ["FINITE_OVERLAP_MISMATCH"],
            },
        ]
    )

    assert result["source_rows"] == 20
    assert result["overlap_finite_rows"] == 17
    assert result["mismatch_rows"] == 2
    assert result["decision_status"] == "FORMULA_REPLAY_BLOCKED"
    assert result["reason_codes"] == ["FINITE_OVERLAP_MISMATCH"]
