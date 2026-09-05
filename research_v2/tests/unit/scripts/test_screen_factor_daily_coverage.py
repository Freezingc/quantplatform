"""Tests for the factor_daily coverage screen (C plan C6.2 on the consumed source).

Fixtures reproduce the two failure modes found in the frozen set so the screen is
regression-protected against both: mode A (low per-day coverage, as recorded for
yeslab_372 at median 0.4636 / min 0.0288) and mode B (whole missing periods with
healthy daily coverage, as recorded for yeslab_398 at median 0.9978 but
minimum_monthly_day_ratio 0.0).
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import date, timedelta
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "screen_factor_daily_coverage.py"


def _load():
    spec = importlib.util.spec_from_file_location("screen_factor_daily_coverage", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["screen_factor_daily_coverage"] = module
    spec.loader.exec_module(module)
    return module


M = _load()


def _days(start: date, count: int) -> list[date]:
    return [start + timedelta(days=offset) for offset in range(count)]


def _universe(days: list[date], size: int = 1000) -> dict[date, int]:
    return {day: size for day in days}


def test_full_coverage_passes() -> None:
    days = _days(date(2020, 1, 1), 60)
    result = M.screen_factor("yeslab_x", {d: 1000 for d in days}, _universe(days))
    assert result.coverage_gate_pass
    assert result.eligibility_status == "PASS"
    assert result.reason_codes == ("COVERAGE_GATE_PASS",)
    assert result.median_coverage_ratio == 1.0
    assert result.minimum_coverage_ratio == 1.0
    assert result.valid_day_count == result.expected_day_count == 60
    assert result.minimum_monthly_day_ratio == 1.0


def test_mode_a_low_daily_coverage_fails_like_yeslab_372() -> None:
    """Reproduce the recorded profile: median ~0.46, p05 ~0.13, min ~0.03."""
    days = _days(date(2020, 1, 1), 100)
    rows = {}
    for index, day in enumerate(days):
        if index < 5:
            rows[day] = 29          # min 0.029
        elif index < 20:
            rows[day] = 130         # p05 region ~0.13
        else:
            rows[day] = 460         # median ~0.46
    result = M.screen_factor("yeslab_372", rows, _universe(days))
    assert not result.coverage_gate_pass
    assert result.eligibility_status == "ELIGIBILITY_FAIL"
    assert "LOW_MEDIAN_SAMPLE_COVERAGE" in result.reason_codes
    assert "LOW_P05_SAMPLE_COVERAGE" in result.reason_codes
    assert result.minimum_coverage_ratio == 0.029


def test_mode_b_missing_period_fails_despite_healthy_daily_coverage() -> None:
    """yeslab_398/464: median ~0.998 but one month entirely absent."""
    days = _days(date(2020, 1, 1), 90)          # 2020-01-01 .. 2020-03-30
    february = {d for d in days if d.month == 2}
    rows = {d: 1000 for d in days if d not in february}
    result = M.screen_factor("yeslab_398", rows, _universe(days))
    assert not result.coverage_gate_pass
    assert "MISSING_MONTH_OR_LOW_MONTHLY_COVERAGE" in result.reason_codes
    assert result.minimum_monthly_day_ratio == 0.0
    assert result.median_coverage_ratio > 0.99
    # 90 days minus February 2020, which has 29 days in a leap year.
    assert result.valid_day_count == 90 - len(february) == 61
    assert result.expected_day_count == 90


def test_denominator_is_the_universe_not_the_factor_table() -> None:
    """C6.3: a factor covering half the universe must read 0.5, not 1.0.

    If the denominator were 'stocks appearing in the factor table' this would score a
    perfect 1.0 and the gate would be meaningless, which is the error C6.3 warns about.
    """
    days = _days(date(2020, 1, 1), 30)
    result = M.screen_factor("yeslab_half", {d: 500 for d in days}, _universe(days, size=1000))
    assert result.median_coverage_ratio == 0.5
    assert not result.coverage_gate_pass


def test_days_the_universe_does_not_cover_are_not_held_against_the_factor() -> None:
    days = _days(date(2020, 1, 1), 30)
    universe = _universe(days)
    rows = {d: 1000 for d in days[:20]}
    narrowed = {d: n for d, n in universe.items() if d in days[:20]}
    result = M.screen_factor("yeslab_x", rows, narrowed)
    assert result.expected_day_count == 20
    assert result.day_ratio == 1.0
    assert result.coverage_gate_pass


def test_duplicate_business_keys_fail() -> None:
    days = _days(date(2020, 1, 1), 30)
    result = M.screen_factor(
        "yeslab_x", {d: 1000 for d in days}, _universe(days), duplicate_row_count=3
    )
    assert not result.coverage_gate_pass
    assert "DUPLICATE_BUSINESS_KEYS" in result.reason_codes


def test_factor_absent_from_the_window_fails_closed() -> None:
    days = _days(date(2020, 1, 1), 30)
    result = M.screen_factor("yeslab_missing", {}, _universe(days))
    assert not result.coverage_gate_pass
    assert result.valid_day_count == 0
    assert result.median_coverage_ratio == 0.0
    assert "MISSING_MONTH_OR_LOW_MONTHLY_COVERAGE" in result.reason_codes


def test_empty_universe_is_handled_without_division_by_zero() -> None:
    result = M.screen_factor("yeslab_x", {}, {})
    assert result.expected_day_count == 0
    assert result.day_ratio == 0.0
    assert result.minimum_monthly_day_ratio == 0.0
    assert not result.coverage_gate_pass


def test_quantile_helper_edge_cases() -> None:
    assert M._quantile([], 0.5) == 0.0
    assert M._quantile([0.4], 0.05) == 0.4
    values = sorted([0.1, 0.2, 0.3, 0.4, 0.5])
    assert M._quantile(values, 0.0) == 0.1
    assert M._quantile(values, 1.0) == 0.5
    assert M._quantile(values, 0.5) == 0.3


def test_threshold_boundary_is_inclusive_at_the_gate() -> None:
    """Exactly 0.70 median must pass the C6.2 gate, since the rule is '<70% fails'."""
    days = _days(date(2020, 1, 1), 30)
    result = M.screen_factor("yeslab_edge", {d: 700 for d in days}, _universe(days, size=1000))
    assert result.median_coverage_ratio == 0.70
    assert result.coverage_gate_pass
    just_below = M.screen_factor(
        "yeslab_edge", {d: 699 for d in days}, _universe(days, size=1000)
    )
    assert not just_below.coverage_gate_pass
