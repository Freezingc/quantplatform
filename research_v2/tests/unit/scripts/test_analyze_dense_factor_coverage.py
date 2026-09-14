"""Tests for the dense-factor coverage analyzer.

The point of this script is to answer "which factor, which securities, which period" for the rows
the frozen baseline cannot score, so the tests concentrate on the parts where a wrong answer would
still look plausible: which factors are missing TOGETHER (a single broken factor and two factors
sharing one dependency look identical in a per-column NaN table), whether one factor's missing set
nests inside another's, and which rows belong in the coverage denominator at all.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
# The analyzer's --repo is the directory that CONTAINS research_v2, which from a test file
# three levels down is one level above REPO_ROOT.
REPO = REPO_ROOT.parent
sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_dense_factor_coverage as module  # noqa: E402

COLUMNS = ["f00", "f01", "f12"]


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(rows)


class TestFactorSetParsing:
    def test_reads_both_sets_out_of_the_driver_source(self) -> None:
        sets = module.parse_factor_sets(REPO)
        assert len(sets["FROZEN_FACTOR_SET_V2"]) == 15
        assert len(sets["FROZEN_FACTOR_SET_V2_DENSE"]) == 9
        assert set(sets["FROZEN_FACTOR_SET_V2_DENSE"]).issubset(sets["FROZEN_FACTOR_SET_V2"])

    def test_dense_columns_use_the_canonical_feature_names(self) -> None:
        sets = module.parse_factor_sets(REPO)
        columns = module.dense_columns(
            sets["FROZEN_FACTOR_SET_V2"], sets["FROZEN_FACTOR_SET_V2_DENSE"]
        )
        assert len(columns) == 9
        assert columns[0] == "f00"  # yeslab_206 is first in the canonical order and is dense
        assert "f02" not in columns  # yeslab_372 is sparse, so it must not be treated as dense
        assert columns == sorted(columns)

    def test_a_repo_without_the_source_is_refused(self, tmp_path: Path) -> None:
        with pytest.raises(SystemExit):
            module.parse_factor_sets(tmp_path)


class TestMissingCombinations:
    def test_counts_combinations_not_columns(self) -> None:
        incomplete = _frame(
            [
                {"f00": 1.0, "f01": float("nan"), "f12": float("nan")},
                {"f00": 1.0, "f01": float("nan"), "f12": float("nan")},
                {"f00": float("nan"), "f01": 1.0, "f12": 1.0},
            ]
        )
        assert module.missing_combinations(incomplete, COLUMNS) == {"f01,f12": 2, "f00": 1}

    def test_the_two_shapes_that_a_per_column_table_cannot_tell_apart(self) -> None:
        shared_dependency = _frame(
            [{"f00": 1.0, "f01": float("nan"), "f12": float("nan")}] * 3
        )
        independent_gaps = _frame(
            [
                {"f00": 1.0, "f01": float("nan"), "f12": 1.0},
                {"f00": 1.0, "f01": 1.0, "f12": float("nan")},
                {"f00": float("nan"), "f01": 1.0, "f12": 1.0},
            ]
        )
        assert module.missing_combinations(shared_dependency, COLUMNS) == {"f01,f12": 3}
        assert module.missing_combinations(independent_gaps, COLUMNS) == {
            "f00": 1,
            "f01": 1,
            "f12": 1,
        }

    def test_an_empty_frame_has_no_combinations(self) -> None:
        empty = _frame([]).assign(**{column: [] for column in COLUMNS})
        combinations = module.missing_combinations(empty, COLUMNS)
        assert combinations == {}


class TestNesting:
    def test_a_perfectly_nested_pair_reports_zero_inner_only(self) -> None:
        frame = _frame(
            [
                {"f01": float("nan"), "f12": float("nan")},
                {"f01": float("nan"), "f12": float("nan")},
                {"f01": 1.0, "f12": float("nan")},
                {"f01": 1.0, "f12": 1.0},
            ]
        )
        result = module.nesting(frame, "f01", "f12")
        assert result["inner_only_rows"] == 0
        assert result["outer_only_rows"] == 1
        assert result["both_missing_rows"] == 2

    def test_an_unnested_pair_says_so(self) -> None:
        frame = _frame(
            [
                {"f01": float("nan"), "f12": 1.0},
                {"f01": 1.0, "f12": float("nan")},
            ]
        )
        result = module.nesting(frame, "f01", "f12")
        assert result["inner_only_rows"] == 1
        assert result["outer_only_rows"] == 1
        assert result["both_missing_rows"] == 0


class TestValidLabelRows:
    def test_only_rows_with_a_valid_label_and_a_finite_return_count(self) -> None:
        frame = _frame(
            [
                {"label_valid": True, "raw_return_5d": 0.01},
                {"label_valid": False, "raw_return_5d": 0.02},
                {"label_valid": True, "raw_return_5d": float("nan")},
                {"label_valid": None, "raw_return_5d": 0.03},
                {"label_valid": True, "raw_return_5d": -0.04},
            ]
        )
        kept = module.valid_label_rows(frame)
        assert len(kept) == 2
        assert list(kept["raw_return_5d"]) == [0.01, -0.04]


class TestCodePrefixes:
    def test_counts_by_prefix(self) -> None:
        assert module.code_prefixes(["688001.SH", "688002.SH", "000001.SZ"]) == {
            "000": 1,
            "688": 2,
        }


def _write_split(tmp_path: Path) -> list[Path]:
    rows = []
    # Two shards, four days, three securities: 688001.SH is missing f01 and f12 on the two
    # early days, which is the shape the real frozen-v2 data turned out to have.
    for day, star_ok in (("2023-07-05", False), ("2023-07-06", False),
                         ("2023-11-02", True), ("2023-11-03", True)):
        for code, is_star in (("688001.SH", True), ("000001.SZ", False), ("600000.SH", False)):
            missing = is_star and not star_ok
            rows.append(
                {
                    "signal_date": day,
                    "ts_code": code,
                    "f00": 1.0,
                    "f01": float("nan") if missing else 0.5,
                    "f12": float("nan") if missing else 0.25,
                    "raw_return_5d": 0.01,
                    "label_valid": True,
                }
            )
    frame = pd.DataFrame(rows)
    paths = []
    for index, half in enumerate((frame.iloc[:6], frame.iloc[6:])):
        path = tmp_path / f"part-{index:05d}.parquet"
        half.to_parquet(path)
        paths.append(path)
    return paths



class TestAnalyseSplit:
    def test_shards_are_concatenated_and_the_shape_is_reported(self, tmp_path: Path) -> None:
        result = module.analyse_split(_write_split(tmp_path), COLUMNS)
        assert result["valid_label_rows"] == 12
        assert result["distinct_securities"] == 3
        assert result["dense_incomplete_rows"] == 2
        assert result["dense_incomplete_share"] == pytest.approx(2 / 12)
        assert result["top_missing_combinations"] == {"f01,f12": 2}

    def test_the_star_market_concentration_is_measured(self, tmp_path: Path) -> None:
        affected = module.analyse_split(_write_split(tmp_path), COLUMNS)["affected_securities"]
        assert affected["count"] == 1
        assert affected["star_market_in_frame"] == 1
        assert affected["star_market_affected"] == 1
        assert affected["code_prefixes"] == {"688": 1}
        assert affected["top_5_by_days"] == {"688001.SH": 2}

    def test_per_day_and_per_month_shares_are_present(self, tmp_path: Path) -> None:
        result = module.analyse_split(_write_split(tmp_path), COLUMNS)
        per_day = result["per_day_incomplete_share"]
        assert per_day["days"] == 4
        assert per_day["max"] == pytest.approx(1 / 3)
        assert per_day["min"] == 0.0
        assert per_day["worst_5_days"]["2023-07-05"] == pytest.approx(1 / 3)
        assert result["per_month_incomplete_share"]["2023-11"] == 0.0
        assert result["per_month_incomplete_share"]["2023-07"] == pytest.approx(1 / 3)

    def test_the_worst_pair_nesting_is_reported(self, tmp_path: Path) -> None:
        # f01 and f12 go missing together for the STAR-market security, so neither has rows the
        # other lacks: inner_only and outer_only must both be zero, which is what distinguishes a
        # shared dependency from two independent gaps.
        result = module.analyse_split(_write_split(tmp_path), COLUMNS)
        reported = result["top_pair_nesting"]
        assert reported is not None
        assert {reported["inner"], reported["outer"]} == {"f01", "f12"}
        assert reported["both_missing_rows"] == 2
        assert reported["inner_only_rows"] == 0
        assert reported["outer_only_rows"] == 0


class TestRenderMarkdown:
    def test_renders_the_tables_and_names_the_companion_script(self, tmp_path: Path) -> None:
        result = module.analyse_split(_write_split(tmp_path), COLUMNS)
        payload = {
            "workspace": "/tmp/ws",
            "dense_factors": ["a", "b", "c"],
            "dense_columns": COLUMNS,
            "sparse_factors": ["d"],
            "years": {"2023": result},
        }
        text = module.render_markdown(payload)
        assert text == module.render_markdown(payload)  # deterministic
        assert "| 2023 | 12 | 2 |" in text
        assert "a (f00) |" in text
        assert "analyze_yearly_baseline_deltas.py" in text
