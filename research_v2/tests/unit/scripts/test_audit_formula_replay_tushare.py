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


def test_parser_accepts_alpha5_tushare_replay() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_5",
            "--exclude-bj",
            "--start",
            "2025-01-02",
            "--end",
            "2025-01-31",
            "--output",
            "/tmp/tushare-alpha5.json",
        ]
    )

    assert args.factor_id == ("yeslab_5",)
    assert not audit.requires_full_cross_section(args.factor_id)
    assert audit.required_market_fields((REPLAY_DEFINITIONS["yeslab_5"],)) == (
        "high",
        "vol",
    )


def test_normalize_tushare_market_preserves_total_market_value() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20250102"],
            "open": [10.0],
            "low": [9.8],
            "amount": [1_000_000.0],
            "pct_chg": [1.2],
            "total_mv": [123_456.78],
        }
    )

    result = audit.normalize_tushare_market(raw, codes=("000001.SZ",))

    assert list(result["total_mv"]) == [123_456.78]


def test_parser_accepts_alpha31_full_cross_section_replay() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_31",
            "--full-market",
            "--exclude-bj",
            "--stock-batch-size",
            "6000",
            "--start",
            "2025-01-02",
            "--end",
            "2025-01-31",
            "--output",
            "/tmp/tushare-alpha31.json",
        ]
    )

    assert args.factor_id == ("yeslab_31",)
    assert audit.requires_full_cross_section(args.factor_id)
    assert audit.required_market_fields((REPLAY_DEFINITIONS["yeslab_31"],)) == (
        "high",
        "vol",
    )


def test_parser_accepts_alpha62_full_cross_section_replay() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_71",
            "--full-market",
            "--restrict-to-v2-universe",
            "--exclude-bj",
            "--start",
            "2025-01-02",
            "--end",
            "2025-01-31",
            "--output",
            "/tmp/alpha62-replay.json",
        ]
    )

    assert args.factor_id == ("yeslab_71",)
    assert audit.requires_full_cross_section(args.factor_id)
    assert audit.required_market_fields((REPLAY_DEFINITIONS["yeslab_71"],)) == (
        "high",
        "vol",
    )


def test_parser_accepts_alpha83_full_cross_section_replay() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_105",
            "--full-market",
            "--exclude-bj",
            "--start",
            "2025-01-02",
            "--end",
            "2025-01-31",
            "--output",
            "/tmp/alpha83-replay.json",
        ]
    )

    assert args.factor_id == ("yeslab_105",)
    assert audit.requires_full_cross_section(args.factor_id)
    assert audit.required_market_fields((REPLAY_DEFINITIONS["yeslab_105"],)) == (
        "high",
        "vol",
    )


def test_parser_accepts_full_market_batches() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_438",
            "--full-market",
            "--restrict-to-v2-universe",
            "--exclude-bj",
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
    assert args.exclude_bj is True
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


def test_parser_accepts_industry_normalized_factor_for_full_market_replay() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_357",
            "--full-market",
            "--exclude-bj",
            "--stock-batch-size",
            "5000",
            "--start",
            "2020-09-14",
            "--end",
            "2020-09-25",
            "--output",
            "/tmp/tushare-industry-357.json",
        ]
    )

    assert args.factor_id == ("yeslab_357",)
    assert audit.requires_full_cross_section(args.factor_id)


def test_parser_accepts_industry_skew_factors_for_full_market_replay() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_362",
            "--factor-id",
            "yeslab_363",
            "--full-market",
            "--exclude-bj",
            "--stock-batch-size",
            "5000",
            "--start",
            "2025-01-02",
            "--end",
            "2025-01-31",
            "--output",
            "/tmp/tushare-industry-skew.json",
        ]
    )

    assert args.factor_id == ("yeslab_362", "yeslab_363")
    assert audit.requires_full_cross_section(args.factor_id)


def test_parser_accepts_adjusted_csad_and_corwin_schultz_factors() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_481",
            "--factor-id",
            "yeslab_487",
            "--full-market",
            "--exclude-bj",
            "--start",
            "2025-01-02",
            "--end",
            "2025-01-31",
            "--output",
            "/tmp/tushare-adjusted-factors.json",
        ]
    )

    assert args.factor_id == ("yeslab_481", "yeslab_487")
    assert audit.required_market_fields(
        (REPLAY_DEFINITIONS["yeslab_481"], REPLAY_DEFINITIONS["yeslab_487"])
    ) == ("close_hfq", "high_hfq", "low_hfq")


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


def test_parser_accepts_registered_gp_quality_replay_factors() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_377",
            "--factor-id",
            "yeslab_387",
            "--factor-id",
            "yeslab_388",
            "--factor-id",
            "yeslab_391",
            "--factor-id",
            "yeslab_432",
            "--factor-id",
            "yeslab_433",
            "--factor-id",
            "yeslab_443",
            "--full-market",
            "--restrict-to-v2-universe",
            "--exclude-bj",
            "--start",
            "2025-01-01",
            "--end",
            "2025-01-31",
            "--output",
            "/tmp/registered-gp-replay.json",
        ]
    )

    assert args.factor_id == (
        "yeslab_377",
        "yeslab_387",
        "yeslab_388",
        "yeslab_391",
        "yeslab_432",
        "yeslab_433",
        "yeslab_443",
    )


def test_parser_accepts_annual_stable_candidates_for_tushare_replay() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_165",
            "--factor-id",
            "yeslab_167",
            "--factor-id",
            "yeslab_168",
            "--factor-id",
            "yeslab_206",
            "--factor-id",
            "yeslab_208",
            "--full-market",
            "--exclude-bj",
            "--stock-batch-size",
            "5000",
            "--start",
            "2025-01-02",
            "--end",
            "2025-01-31",
            "--output",
            "/tmp/tushare-annual-stable-candidates.json",
        ]
    )

    assert args.factor_id == (
        "yeslab_165",
        "yeslab_167",
        "yeslab_168",
        "yeslab_206",
        "yeslab_208",
    )
    assert audit.required_market_fields(
        tuple(REPLAY_DEFINITIONS[factor_id] for factor_id in args.factor_id)
    ) == ("close", "high", "vol")


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


def test_normalize_tushare_market_drops_turnover_when_daily_basic_is_disabled() -> None:
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

    normalized = audit.normalize_tushare_market(
        raw, codes=("000001.SZ",), include_daily_basic=False
    )

    assert "turnover_rate" not in normalized.columns


def test_normalize_tushare_market_preserves_pre_close() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20250102"],
            "open": [10.5],
            "low": [10.0],
            "amount": [10.456],
            "pct_chg": [-0.4],
            "pre_close": [10.8],
        }
    )

    normalized = audit.normalize_tushare_market(raw, codes=("000001.SZ",))

    assert normalized.loc[0, "pre_close"] == pytest.approx(10.8)


def test_normalize_tushare_market_derives_hfq_fields_from_adj_factor() -> None:
    raw = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": ["20250102"],
            "open": [10.0],
            "high": [11.0],
            "low": [9.0],
            "close": [10.5],
            "pre_close": [10.2],
            "amount": [10.456],
            "pct_chg": [2.94],
            "adj_factor": [1.2],
        }
    )

    normalized = audit.normalize_tushare_market(raw, codes=("000001.SZ",))

    assert normalized.loc[0, "close_hfq"] == pytest.approx(12.6)
    assert normalized.loc[0, "high_hfq"] == pytest.approx(13.2)
    assert normalized.loc[0, "low_hfq"] == pytest.approx(10.8)


def test_merge_tushare_adjustment_factor_aligns_code_and_date() -> None:
    daily = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["20250102", "20250103"],
            "close": [10.5, 10.8],
        }
    )
    adjustment = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000001.SZ"],
            "trade_date": ["20250103", "20250102"],
            "adj_factor": [1.3, 1.2],
        }
    )

    merged = audit.merge_tushare_adjustment_factor(daily, adjustment)

    assert merged["adj_factor"].tolist() == [1.2, 1.3]


def test_required_market_fields_reflect_calculator_inputs() -> None:
    fields = audit.required_market_fields(
        (REPLAY_DEFINITIONS["yeslab_95"], REPLAY_DEFINITIONS["yeslab_163"])
    )

    assert fields == ("amount", "low", "pct_chg", "turnover_rate", "vol")


def test_tushare_market_session_intercepts_database_handle_market_query() -> None:
    class Base:
        def __init__(self) -> None:
            self.calls = 0

        def run(self, script: str) -> object:
            self.calls += 1
            raise AssertionError("stock_factor_daily query must not reach YesLab")

    base = Base()
    market = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2020-09-18")],
            "ts_code": ["000001.SZ"],
            "amount": [100.0],
            "turnover_rate": [0.5],
        }
    )
    wrapped = audit.TushareMarketSession(base, market, ("000001.SZ",))

    result = wrapped.run(
        'db = database("dfs://YesLab")\n'
        't = loadTable(db, "stock_factor_daily")\n'
        "select trade_date, ts_code, amount, turnover_rate from t"
    )

    assert isinstance(result, pd.DataFrame)
    assert result.to_dict("records") == market.to_dict("records")
    assert base.calls == 0


def test_cross_sectional_replay_requires_one_complete_batch() -> None:
    assert audit.requires_full_cross_section(("yeslab_376", "yeslab_163")) is True
    assert audit.requires_full_cross_section(("yeslab_443",)) is True
    assert audit.requires_full_cross_section(("yeslab_163", "yeslab_380")) is False


def test_alias_map_load_and_market_remap(tmp_path) -> None:
    path = tmp_path / "aliases.json"
    path.write_text('{"mapping": {"830001.BJ": "920001.BJ"}}', encoding="utf-8")

    aliases = audit.load_alias_map(path)
    market = pd.DataFrame(
        {
            "trade_date": [pd.Timestamp("2025-01-02")],
            "ts_code": ["920001.BJ"],
            "open": [1.0],
        }
    )

    remapped = audit.apply_tushare_aliases(market, aliases)

    assert remapped.loc[0, "ts_code"] == "830001.BJ"


def test_alias_map_rejects_duplicate_target_codes(tmp_path) -> None:
    path = tmp_path / "aliases.json"
    path.write_text(
        '{"mapping": {"830001.BJ": "920001.BJ", "830002.BJ": "920001.BJ"}}',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate alias target"):
        audit.load_alias_map(path)


def test_source_market_coverage_counts_uncovered_finite_rows() -> None:
    source = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "ts_code": ["000001.SZ", "830001.BJ"],
            "value": [1.0, 2.0],
        }
    )
    market = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-02"]),
            "ts_code": ["000001.SZ"],
        }
    )

    result = audit.source_market_coverage(source, market)

    assert result["source_market_uncovered_rows"] == 1
    assert result["source_market_uncovered_code_count"] == 1
    assert result["source_market_uncovered_date_count"] == 1


def test_source_market_coverage_catches_missing_date_for_known_code() -> None:
    source = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-02", "2020-01-03"]),
            "ts_code": ["000001.SZ", "000001.SZ"],
            "value": [1.0, 2.0],
        }
    )
    market = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-01-02"]),
            "ts_code": ["000001.SZ"],
        }
    )

    result = audit.source_market_coverage(source, market)

    assert result["source_market_uncovered_rows"] == 1
    assert result["source_market_uncovered_code_count"] == 1
    assert result["source_market_uncovered_date_count"] == 1


def test_missing_code_range_fetch_uses_daily_basic_when_required(tmp_path) -> None:
    class FakePro:
        def daily(self, **kwargs) -> pd.DataFrame:
            assert kwargs["ts_code"] == "830001.BJ"
            return pd.DataFrame(
                {
                    "ts_code": ["830001.BJ"],
                    "trade_date": ["20250102"],
                    "open": [10.0],
                    "low": [9.5],
                    "amount": [100.0],
                    "pct_chg": [1.0],
                }
            )

        def daily_basic(self, **kwargs) -> pd.DataFrame:
            assert kwargs["ts_code"] == "830001.BJ"
            return pd.DataFrame(
                {
                    "ts_code": ["830001.BJ"],
                    "trade_date": ["20250102"],
                    "turnover_rate": [0.5],
                }
            )

    result = audit.fetch_missing_code_ranges(
        FakePro(),
        ("830001.BJ",),
        date(2025, 1, 1),
        date(2025, 1, 2),
        cache_dir=tmp_path,
        include_daily_basic=True,
    )

    assert result.loc[0, "ts_code"] == "830001.BJ"
    assert result.loc[0, "turnover_rate"] == pytest.approx(0.5)


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
