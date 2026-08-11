from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import audit_formula_replay_consistency as audit
import pandas as pd
import pytest

DATABASE_PATH = Path(__file__).resolve().parents[4] / "数据库"
if str(DATABASE_PATH) not in sys.path:
    sys.path.insert(0, str(DATABASE_PATH))
from get_data import DataFetcher  # noqa: E402


def test_compare_replay_frames_passes_exact_finite_overlap() -> None:
    source = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-09-18", "2020-09-21"]),
            "ts_code": ["000001.SZ", "000001.SZ"],
            "value": [1.0, 2.0],
        }
    )
    replay = source.rename(columns={"value": "replay_value"})

    result = audit.compare_replay_frames(source, replay)

    assert result["decision_status"] == "FORMULA_REPLAY_PASS"
    assert result["overlap_finite_rows"] == 2
    assert result["source_finite_rows"] == 2
    assert result["replay_finite_rows"] == 2
    assert result["mismatch_rows"] == 0


def test_compare_replay_frames_blocks_finite_mismatch() -> None:
    source = pd.DataFrame(
        {
            "trade_date": pd.to_datetime(["2020-09-18", "2020-09-21"]),
            "ts_code": ["000001.SZ", "000001.SZ"],
            "value": [1.0, 2.0],
        }
    )
    replay = pd.DataFrame(
        {
            "trade_date": source["trade_date"],
            "ts_code": source["ts_code"],
            "replay_value": [1.0, 2.2],
        }
    )

    result = audit.compare_replay_frames(source, replay)

    assert result["decision_status"] == "FORMULA_REPLAY_BLOCKED"
    assert result["mismatch_rows"] == 1
    assert result["max_abs_diff"] == pytest.approx(0.2)
    assert result["source_finite_rows"] == 2
    assert result["replay_finite_rows"] == 2


def test_parser_rejects_locked_test_and_unknown_factor() -> None:
    with pytest.raises(ValueError, match="2026"):
        audit.parse_args(
            [
                "--factor-id",
                "yeslab_395",
                "--start",
                "2026-01-01",
                "--end",
                "2026-01-02",
                "--output",
                "/tmp/replay.json",
            ]
        )
    with pytest.raises(ValueError, match="unsupported"):
        audit.parse_args(
            [
                "--factor-id",
                "yeslab_999",
                "--start",
                "2020-09-14",
                "--end",
                "2020-09-25",
                "--output",
                "/tmp/replay.json",
            ]
        )


def test_parser_accepts_oos_candidate_replay_factor() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_443",
            "--start",
            "2020-09-14",
            "--end",
            "2020-09-25",
            "--output",
            "/tmp/replay.json",
        ]
    )

    assert args.factor_id == ("yeslab_443",)


def test_parser_accepts_financial_metric_aliases() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_222",
            "--factor-id",
            "yeslab_224",
            "--start",
            "2025-01-02",
            "--end",
            "2025-01-31",
            "--output",
            "/tmp/financial-alias-replay.json",
        ]
    )

    assert args.factor_id == ("yeslab_222", "yeslab_224")


def test_parser_accepts_alpha5_replay_factor() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_5",
            "--start",
            "2025-01-02",
            "--end",
            "2025-01-31",
            "--output",
            "/tmp/alpha5-replay.json",
        ]
    )

    assert args.factor_id == ("yeslab_5",)


def test_parser_accepts_alpha31_replay_factor() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_31",
            "--start",
            "2025-01-02",
            "--end",
            "2025-01-31",
            "--output",
            "/tmp/alpha31-replay.json",
        ]
    )

    assert args.factor_id == ("yeslab_31",)


def test_parser_accepts_warmup_recovered_factor() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_398",
            "--start",
            "2020-01-02",
            "--end",
            "2020-01-31",
            "--output",
            "/tmp/replay.json",
        ]
    )

    assert args.factor_id == ("yeslab_398",)


def test_parser_accepts_bounded_replay_sample() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_398",
            "--max-ts-codes",
            "100",
            "--start",
            "2020-01-02",
            "--end",
            "2020-01-31",
            "--output",
            "/tmp/replay.json",
        ]
    )

    assert args.max_ts_codes == 100
    assert audit.REPLAY_DEFINITIONS["yeslab_398"].replay_lookback_days == 180


def test_parser_accepts_structural_predictive_factor_for_replay() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_370",
            "--max-ts-codes",
            "64",
            "--start",
            "2020-09-14",
            "--end",
            "2020-09-25",
            "--output",
            "/tmp/replay.json",
        ]
    )

    assert args.factor_id == ("yeslab_370",)
    assert audit.REPLAY_DEFINITIONS["yeslab_370"].replay_lookback_days == 180


def test_registry_includes_existing_gp_calculators_for_shsz_quality_replay() -> None:
    expected = {
        "yeslab_377": ("GP_Factor_008_Calculator", "factor_data8", 1),
        "yeslab_387": ("GP_Factor_018_Calculator", "factor_data9", 1),
        "yeslab_388": ("GP_Factor_019_Calculator", "factor_data9", 10),
        "yeslab_391": ("GP_Factor_022_Calculator", "factor_data9", 60),
        "yeslab_432": ("GP_Factor_063_Calculator", "factor_data9", 5),
        "yeslab_433": ("GP_Factor_064_Calculator", "factor_data9", 60),
    }

    for factor_id, (calculator, store_factor, lookback) in expected.items():
        definition = audit.REPLAY_DEFINITIONS[factor_id]
        assert definition.calculator_module == calculator
        assert definition.calculator_class == calculator
        assert definition.store_factor == store_factor
        assert definition.lookback_days == lookback


def test_run_audit_records_calculator_errors_without_dropping_factor() -> None:
    class Client:
        def run(self, script: str) -> object:
            raise RuntimeError("calculator input unavailable")

    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_443",
            "--start",
            "2020-09-14",
            "--end",
            "2020-09-25",
            "--output",
            "/tmp/replay.json",
        ]
    )

    result = audit.run_audit(args, client=Client())

    assert result["rows"][0]["decision_status"] == "FORMULA_REPLAY_ERROR"
    assert result["rows"][0]["reason_codes"] == ["CALCULATOR_REPLAY_ERROR"]


def test_replay_dates_are_bounded_before_locked_test() -> None:
    args = audit.parse_args(
        ["--factor-id", "yeslab_395", "--start", "2020-09-14", "--end", "2020-09-25"]
        + ["--output", "/tmp/replay.json"]
    )
    assert args.start == date(2020, 9, 14)
    assert args.end == date(2020, 9, 25)


def test_data_fetcher_maps_current_industry_column_to_calculator_contract() -> None:
    class Session:
        def __init__(self) -> None:
            self.script = ""

        def run(self, script: str) -> pd.DataFrame:
            self.script = script
            return pd.DataFrame(
                {
                    "ts_code": ["000001.SZ"],
                    "l1_name": ["银行"],
                    "l1_code": ["银行"],
                }
            )

    session = Session()
    result = DataFetcher(session).get_stock_industry_info()

    assert result.to_dict("records") == [
        {"ts_code": "000001.SZ", "l1_name": "银行", "l1_code": "银行"}
    ]
    assert 'database("dfs://YesLab")' in session.script
    assert 'loadTable(db, "stock_info_table")' in session.script
    assert "industry as l1_name" in session.script
    assert "industry as l1_code" in session.script
