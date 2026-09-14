"""Tests for the pure half of analyze_yearly_baseline_deltas.py.

The frame assembly and the evaluator replay cannot be unit-tested without signed artifacts, so
they are guarded at run time instead: the script refuses to print a single new number unless it
first reproduces the signed report bit for bit (checks A, B and C in its own docstring). What is
tested here are the three pieces whose failure would be silent -- the equality predicate that
"bit for bit" rests on, the day-weighted rebuild that is check C, and the manifest scan that
decides which prediction artifact is the candidate and which is the baseline.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPTS_DIR = REPO_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

import analyze_yearly_baseline_deltas as module  # noqa: E402


def _row(year: int, days: int, delta: float) -> dict[str, Any]:
    return {"year": year, "candidate_valid_days": days, "delta_rank_ic": delta, "status": "OK"}


class TestSame:
    def test_identical_floats_are_the_same(self) -> None:
        assert module._same(0.0852268662737014, 0.0852268662737014)

    def test_floats_that_differ_in_the_last_bit_are_not(self) -> None:
        assert not module._same(0.1 + 0.2, 0.3)
        assert repr(0.1 + 0.2) != repr(0.3)

    def test_none_and_strings_compare_by_equality(self) -> None:
        assert module._same(None, None)
        assert not module._same(None, 0.0)
        assert module._same("PASS", "PASS")
        assert not module._same("PASS", "FAIL")

    def test_day_counts_compare_as_integers(self) -> None:
        assert module._same(945, 945)
        assert not module._same(945, 969)


class TestRebuiltPooledDelta:
    def test_rebuilds_the_pooled_delta_from_its_own_partition(self) -> None:
        rows = [_row(2022, 241, 0.012491), _row(2023, 219, 0.003192),
                _row(2024, 242, 0.013919), _row(2025, 243, 0.006453)]
        rebuilt, days = module.rebuilt_pooled_delta(rows)
        assert days == 945
        expected = (0.012491 * 241 + 0.003192 * 219 + 0.013919 * 242 + 0.006453 * 243) / 945
        assert abs(rebuilt - expected) < 1e-15

    def test_a_partition_that_does_not_add_back_up_is_detectable(self) -> None:
        rows = [_row(2022, 241, 0.012491), _row(2023, 219, 0.003192),
                _row(2024, 242, 0.013919), _row(2025, 243, 0.006453)]
        honest, _ = module.rebuilt_pooled_delta(rows)
        tampered = [dict(row) for row in rows]
        tampered[1]["delta_rank_ic"] = -0.005066  # the 2023 value as mis-read across day sets
        rebuilt, _ = module.rebuilt_pooled_delta(tampered)
        assert abs(rebuilt - honest) > 1e-4

    def test_missing_years_are_skipped_not_treated_as_zero(self) -> None:
        rows = [_row(2022, 241, 0.01), {"year": 2023, "status": "MISSING"}]
        rebuilt, days = module.rebuilt_pooled_delta(rows)
        assert days == 241
        assert rebuilt == 0.01

    def test_no_rows_is_nan_and_zero_days(self) -> None:
        rebuilt, days = module.rebuilt_pooled_delta([])
        assert days == 0
        assert math.isnan(rebuilt)


class TestPredictionIdsByFold:
    def _write(self, root: Path, prediction_id: str, fold: str, kind: str) -> None:
        # The function takes the WORKSPACE and appends "predictions", the way the driver's
        # layout does, so the fixture has to nest one level deeper than the id.
        directory = root / "predictions" / prediction_id
        directory.mkdir(parents=True)
        (directory / "manifest.json").write_text(
            json.dumps(
                {
                    "prediction_id": prediction_id,
                    "metadata": {"fold_id": fold, "kind": kind},
                }
            )
        )

    def test_maps_candidate_and_best_simple_per_fold(self, tmp_path: Path) -> None:
        self._write(tmp_path, "cand-f1", "F1", "candidate")
        self._write(tmp_path, "base-f1", "F1", "best_simple")
        self._write(tmp_path, "cand-f2", "F2", "candidate")
        found = module.prediction_ids_by_fold(tmp_path)
        assert found == {
            "F1": {"candidate": "cand-f1", "best_simple": "base-f1"},
            "F2": {"candidate": "cand-f2"},
        }

    def test_other_kinds_are_ignored(self, tmp_path: Path) -> None:
        self._write(tmp_path, "cand-f1", "F1", "candidate")
        self._write(tmp_path, "val-f1", "F1", "validation")
        found = module.prediction_ids_by_fold(tmp_path)
        assert found == {"F1": {"candidate": "cand-f1"}}

    def test_an_empty_predictions_root_is_an_empty_mapping(self, tmp_path: Path) -> None:
        assert module.prediction_ids_by_fold(tmp_path / "absent") == {}


class TestRenderMarkdown:
    def _result(self) -> dict[str, Any]:
        return {
            "family": "M0R",
            "report": "/artifacts/m0r_v2/report.json",
            "report_sha256": "d16b5708c4cdc961" * 4,
            "pooled": {
                "candidate_mean_rank_ic": 0.094376,
                "baseline_mean_rank_ic_on_candidate_days": 0.085227,
                "baseline_mean_rank_ic_before_coverage_matching": 0.085227,
                "baseline_valid_days_before_coverage_matching": 945,
                "delta_rank_ic": 0.009149,
                "candidate_valid_days": 945,
                "baseline_valid_days": 945,
            },
            "yearly": [
                {
                    "year": 2022,
                    "candidate_mean_rank_ic": 0.093128,
                    "candidate_valid_days": 241,
                    "baseline_mean_rank_ic_on_candidate_days": 0.080637,
                    "baseline_valid_days": 241,
                    "delta_rank_ic": 0.012491,
                    "baseline_valid_days_before_coverage_matching": 241,
                    "baseline_mean_rank_ic_before_coverage_matching": 0.080637,
                    "baseline_days_lost_to_coverage_matching": 0,
                    "baseline_mean_rank_ic_on_days_lost_to_matching": None,
                    "status": "OK",
                },
                {"year": 2023, "status": "MISSING"},
            ],
            "unevaluable_days": {
                "label_day_count": 969,
                "candidate_invalid_days": 24,
                "baseline_invalid_days_on_the_same_frame": 24,
                "by_year": {
                    2022: {
                        "label_dates": 242,
                        "valid_days": 241,
                        "invalid_days": 1,
                        "median_coverage_on_invalid_days": 0.011,
                        "median_sample_count_on_invalid_days": 8.0,
                        "median_valid_label_count_on_invalid_days": 727.0,
                        "baseline_invalid_days_same_frame": 1,
                    },
                    2023: {
                        "label_dates": 242,
                        "valid_days": 219,
                        "invalid_days": 23,
                        "median_coverage_on_invalid_days": 0.009,
                        "median_sample_count_on_invalid_days": 7.0,
                        "median_valid_label_count_on_invalid_days": 780.0,
                        "baseline_invalid_days_same_frame": 23,
                    },
                },
            },
        }

    def test_is_deterministic(self) -> None:
        result = self._result()
        assert module.render_markdown(result) == module.render_markdown(result)

    def test_states_the_acceptance_it_rests_on(self) -> None:
        text = module.render_markdown(self._result())
        assert "Acceptance: A the replay reproduces the signed report bit for bit" in text
        assert "C the day-weighted" in text

    def test_a_missing_year_is_shown_as_missing(self) -> None:
        text = module.render_markdown(self._result())
        assert "| 2023 | MISSING |" in text

    def test_a_lost_day_mean_of_none_renders_as_a_dash(self) -> None:
        text = module.render_markdown(self._result())
        row = next(line for line in text.splitlines() if line.startswith("| 2022 |"))
        assert row.endswith(" 0 | - |")

    def test_the_unevaluable_day_section_carries_the_counts(self) -> None:
        text = module.render_markdown(self._result())
        assert "## Days the candidate cannot be evaluated on" in text
        assert "969 OOS days" in text
        assert "unevaluable on 24" in text
        # Search inside the section only: the yearly table above also has a 2023 row, and it is
        # the MISSING one, so a whole-document scan finds the wrong line first.
        section = text.split("## Days the candidate cannot be evaluated on")[1]
        row = next(line for line in section.splitlines() if line.startswith("| 2023 |"))
        assert "| 242 | 219 | 23 | 23 |" in row
        assert "0.0090" in row  # the median coverage that made those days unevaluable

    def test_it_says_the_baseline_run_own_report_is_a_different_evaluation(self) -> None:
        text = module.render_markdown(self._result())
        assert "must not be mixed into this table" in text
