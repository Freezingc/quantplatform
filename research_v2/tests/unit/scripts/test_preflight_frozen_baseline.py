"""Tests for the frozen baseline pre-flight gate (reconstruction plan Task 3.1).

The gate exists because the frozen baseline becomes the trust root while G1-G6 cannot
see a thin cross-section, so these tests focus on the two ways a pre-flight can be
wrong: letting a failure through, and refusing to be waived by an accountable decision.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "preflight_frozen_baseline.py"


def _load():
    spec = importlib.util.spec_from_file_location("preflight_frozen_baseline", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["preflight_frozen_baseline"] = module
    spec.loader.exec_module(module)
    return module


M = _load()


def _result(name: str, code: int) -> M.GateResult:
    return M.GateResult(name, code, False, "")


def _all_pass() -> tuple:
    return tuple(_result(name, 0) for name in M.GATE_ORDER)


def test_all_gates_passing_clears_the_run() -> None:
    verdict = M.aggregate(_all_pass())
    assert verdict.cleared
    assert verdict.blocking == ()
    assert verdict.failures == ()


def test_a_single_failure_refuses_the_run() -> None:
    results = tuple(
        _result(name, 1 if name == "admission_consistency" else 0) for name in M.GATE_ORDER
    )
    verdict = M.aggregate(results)
    assert not verdict.cleared
    assert verdict.blocking == ("admission_consistency",)
    assert any("admission_consistency FAILED" in f for f in verdict.failures)


def test_a_reasoned_waiver_clears_and_is_recorded() -> None:
    results = tuple(
        _result(name, 1 if name == "admission_consistency" else 0) for name in M.GATE_ORDER
    )
    verdict = M.aggregate(results, {"admission_consistency": "decision 2026-09-06 option iii"})
    assert verdict.cleared
    assert verdict.waived == ("admission_consistency",)
    assert verdict.blocking == ()
    assert verdict.results[1].waiver_reason == "decision 2026-09-06 option iii"


def test_waiver_without_a_reason_is_rejected() -> None:
    results = tuple(
        _result(name, 1 if name == "fold_coverage" else 0) for name in M.GATE_ORDER
    )
    verdict = M.aggregate(results, {"fold_coverage": "   "})
    assert not verdict.cleared
    assert any("waiver without a reason" in f for f in verdict.failures)


def test_waiver_naming_an_unknown_gate_is_rejected() -> None:
    verdict = M.aggregate(_all_pass(), {"not_a_gate": "because"})
    assert not verdict.cleared
    assert any("unknown gate" in f for f in verdict.failures)


def test_a_gate_that_never_ran_refuses_the_run() -> None:
    verdict = M.aggregate(tuple(_result(name, 0) for name in M.GATE_ORDER[:2]))
    assert not verdict.cleared
    assert any("did not run" in f for f in verdict.failures)


def test_a_passing_gate_is_not_marked_waived() -> None:
    """A waiver on a gate that passed must not create a misleading WAIVED record, and the
    now-stale waiver must be surfaced so it does not silently mask a future failure."""
    verdict = M.aggregate(_all_pass(), {"fold_coverage": "stale waiver"})
    assert verdict.cleared
    assert verdict.waived == ()
    assert verdict.unused_waivers == ("fold_coverage",)


def test_a_waiver_actually_used_is_not_reported_as_stale() -> None:
    results = tuple(
        _result(name, 1 if name == "fold_coverage" else 0) for name in M.GATE_ORDER
    )
    verdict = M.aggregate(results, {"fold_coverage": "decision ref"})
    assert verdict.cleared
    assert verdict.waived == ("fold_coverage",)
    assert verdict.unused_waivers == ()


def test_run_gates_stops_early_when_the_data_is_incomplete() -> None:
    """Later gates would report on incomplete data and mislead, so they must not run."""
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> int:
        calls.append(argv)
        return 1 if argv[0] == "verify_factor_daily_backfill.py" else 0

    results = M.run_gates(runner=runner)
    assert [r.name for r in results] == list(M.GATE_ORDER)
    assert results[0].exit_code == 1
    assert all(r.exit_code == 1 for r in results[1:])
    assert len(calls) == 1, "no downstream gate may execute on incomplete data"


def test_run_gates_executes_in_order_and_takes_the_worst_screen_window() -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> int:
        calls.append(argv)
        if argv[0] == "screen_factor_daily_coverage.py" and argv[2] == "2020-01-01":
            return 1
        return 0

    results = M.run_gates(runner=runner)
    by_name = {r.name: r for r in results}
    assert by_name["factor_coverage_screen"].exit_code == 1
    assert by_name["backfill_completeness"].exit_code == 0
    scripts = [argv[0] for argv in calls]
    assert scripts[0] == "verify_factor_daily_backfill.py"
    assert scripts.count("screen_factor_daily_coverage.py") == len(M.DEFAULT_SCREEN_WINDOWS)
    assert scripts[-2] == "verify_frozen_fold_coverage.py"
    assert scripts[-1] == "verify_frozen_report_path.py"


def test_screen_windows_cover_the_whole_materialized_period() -> None:
    """2025 used to be unscreened: the two prior windows cover 2,676 of 2,919 trading days.

    Measured 2026-09-06, screening 2025 gives 243/243 days for all 17 factors with
    PASS 11/17 and the same six ELIGIBILITY_FAIL, so the classification is stable across
    all three windows -- but only if the pre-flight actually runs the third one. Without
    it the door would keep certifying a factor set whose most recent year was never
    measured, which is how the frozen set's own defect stayed invisible for so long.
    """
    assert M.DEFAULT_SCREEN_WINDOWS == (
        ("2014-01-01", "2019-12-31"),
        ("2020-01-01", "2024-12-31"),
        ("2025-01-01", "2025-12-31"),
    )


def test_a_failure_in_the_2025_screen_window_refuses_the_run() -> None:
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> int:
        calls.append(argv)
        if argv[0] == "screen_factor_daily_coverage.py" and argv[2] == "2025-01-01":
            return 1
        return 0

    results = M.run_gates(runner=runner)

    assert {r.name: r for r in results}["factor_coverage_screen"].exit_code == 1


def test_report_path_door_runs_last_and_its_failure_refuses_the_run() -> None:
    """Task 3.1 writes its report at the end of a ~15h run, so prove that works first.

    The environment and assembly_coverage blocks had never executed on the real frozen
    path. A numpy scalar or a NaN survival ratio would raise at serialisation time,
    after the whole run had been paid for.
    """
    calls: list[list[str]] = []

    def runner(argv: list[str]) -> int:
        calls.append(argv)
        return 1 if argv[0] == "verify_frozen_report_path.py" else 0

    results = M.run_gates(runner=runner)

    assert {r.name: r for r in results}["report_path"].exit_code == 1
    assert calls[-1][0] == "verify_frozen_report_path.py"


def test_main_rejects_a_waive_without_a_reason() -> None:
    code = M.main(["--waive", "fold_coverage"])
    assert code == 2


def test_main_rejects_an_unknown_waiver_target() -> None:
    with pytest.raises(SystemExit):
        M.main(["--waive", "not_a_gate", "--waiver-reason", "x"])
