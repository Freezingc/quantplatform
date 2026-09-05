"""Tests for the frozen report-path pre-flight verifier.

Task 3.1 is a roughly fifteen-hour run whose report is written at the very end. The
``environment`` and ``assembly_coverage`` blocks added to that report had never been
exercised on the real frozen path, so a serialisation or dtype defect there would
surface after the whole run had already been paid for. This verifier assembles real
frozen segments, measures them, and proves the resulting report blocks serialise --
cheaply, before the expensive run rather than at the end of it.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "verify_frozen_report_path.py"


def _load():
    spec = importlib.util.spec_from_file_location("verify_frozen_report_path", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_frozen_report_path"] = module
    spec.loader.exec_module(module)
    return module


M = _load()

ENVIRONMENT = {
    "factor_source": "frozen",
    "factor_ids": ["yeslab_96", "yeslab_206"],
    "host_cpu_count": 104,
    "omp_num_threads": None,
    "library_versions": {"numpy": "2.3.3"},
    "torch": {"seed": -1, "cuda_device_names": ("NVIDIA GeForce RTX 4090",)},
}


def test_serialisable_blocks_produce_no_problems() -> None:
    coverage = {"F01": {"train": {"rows": 100, "median_width": 40, "survival_ratio": 0.4}}}

    assert M.report_block_problems(ENVIRONMENT, coverage) == []


def test_numpy_scalars_are_rejected_because_json_dumps_raises_on_them() -> None:
    """The real frames produce numpy ints; json.dumps would fail at hour fifteen."""
    import numpy as np

    coverage = {"F01": {"train": {"rows": np.int64(100)}}}

    problems = M.report_block_problems(ENVIRONMENT, coverage)

    assert any("not JSON serialisable" in problem for problem in problems)


def test_a_non_finite_survival_ratio_is_a_problem() -> None:
    """An empty segment divides by zero; NaN must not reach the archived report."""
    coverage = {"F01": {"train": {"rows": 0, "survival_ratio": float("nan")}}}

    problems = M.report_block_problems(ENVIRONMENT, coverage)

    assert any("survival_ratio" in problem for problem in problems)


def test_environment_must_name_the_factor_set_it_describes() -> None:
    problems = M.report_block_problems({**ENVIRONMENT, "factor_ids": []}, {})

    assert any("factor_ids" in problem for problem in problems)


def test_coverage_must_carry_every_role_of_every_requested_fold() -> None:
    coverage = {"F01": {"train": {"rows": 10, "survival_ratio": 1.0}}}

    problems = M.coverage_completeness_problems(coverage, ("F01",), M.REQUIRED_ROLES)
    assert any("validation" in problem for problem in problems)

    complete = {
        "F01": {role: {"rows": 10, "survival_ratio": 1.0} for role in M.REQUIRED_ROLES}
    }
    assert M.coverage_completeness_problems(complete, ("F01",), M.REQUIRED_ROLES) == []


def test_coverage_completeness_fails_closed_on_a_missing_fold() -> None:
    problems = M.coverage_completeness_problems({}, ("F01",), M.REQUIRED_ROLES)

    assert any("F01" in problem for problem in problems)


def test_widths_from_a_date_object_column_are_measured_not_crashed_on() -> None:
    """Assembled frames store signal_date as datetime.date, not pandas Timestamp.

    This exact dtype mismatch silently hung an ad-hoc measurement earlier: the calendar
    came back as Timestamps while the groupby index held date objects, so every lookup
    missed. The measurement normalises both sides, and this pins that.
    """
    from research_v2.datasets.coverage import summarize_frame_coverage

    frame = pd.DataFrame(
        {
            "signal_date": [date(2019, 1, 2)] * 3 + [date(2019, 1, 3)] * 2,
            "ts_code": ["a", "b", "c", "a", "b"],
            "f00": [1.0, 2.0, float("nan"), 3.0, 4.0],
        }
    )

    summary = summarize_frame_coverage(frame, ("f00",))

    assert summary.days == 2
    # 2019-01-02 keeps 2 of its 3 rows (one f00 is NaN); 2019-01-03 keeps both.
    assert summary.rows == 5
    assert summary.complete_rows == 4
    assert summary.median_width == 2
    assert summary.min_width == 2


def test_payload_round_trips_through_json() -> None:
    coverage = {
        "F01": {
            role: {"rows": 10, "complete_rows": 4, "survival_ratio": 0.4, "min_width": 0}
            for role in M.REQUIRED_ROLES
        }
    }

    text = json.dumps({"environment": ENVIRONMENT, "assembly_coverage": coverage})
    restored = json.loads(text)

    assert restored["environment"] == json.loads(json.dumps(ENVIRONMENT))
    assert set(restored["assembly_coverage"]["F01"]) == set(M.REQUIRED_ROLES)
