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


# The five tests below came from a 15-mutation run over this screen, of which seven survived. All
# seven are pinned here. This screen is the instrument that produced "PASS 11/17 with the same six
# ELIGIBILITY_FAIL" on three windows, which is the measured basis for adjudication 4's option (iii)
# and for the C62_PASS_11 constant in verify_frozen_fold_coverage.py, so a threshold that can drift
# silently can move the frozen list silently.
#
# One of the seven was not a hole and is pinned anyway. Lowering COVERAGE_GATE alone changes no
# verdict, because gate_pass also requires an empty reasons list and MEDIAN_GATE -- same value --
# has already put LOW_MEDIAN_SAMPLE_COVERAGE in it. That is worth stating out loud: it makes the
# COVERAGE_GATE_FAIL branch below unreachable while the two constants agree, and that branch is the
#  reason code the *persisted snapshot* emits as a catch-all for six failing conjuncts. Same name,
# two
# instruments, alive in one and dead in the other.


def test_the_monthly_gate_is_0_90_and_a_month_at_exactly_0_90_passes() -> None:
    """`min_monthly < MONTHLY_DAY_RATIO_GATE`, and the gate is 0.90 rather than 0.50.

    Two faces, because one cannot pin both. A month present on exactly 9 of its 10 trading days is
    at the gate and must pass, which is what distinguishes `<` from `<=`. A month present on 7 of 10
    is between 0.50 and 0.90, so it fails under the real gate and would pass under a lowered one --
    that is what pins the value. 0.90 is also the one threshold of the three that the persisted
    snapshot does NOT share (the snapshot uses 0.70 for the same quantity), so it is the threshold
    most worth having pinned; see decision L.
    """
    days = _days(date(2020, 1, 1), 10)  # all ten fall inside January
    universe = _universe(days, size=1000)

    # Asserted on the reason code rather than on coverage_gate_pass: a day the factor is absent from
    # also enters the daily-coverage series as 0.0, so the p05 gate fires on the same fixture and
    # would mask the monthly one. Pinning the reason isolates the threshold under test.
    at_gate = M.screen_factor("yeslab_m", {d: 1000 for d in days[:9]}, universe)
    assert at_gate.minimum_monthly_day_ratio == 0.9
    assert "MISSING_MONTH_OR_LOW_MONTHLY_COVERAGE" not in at_gate.reason_codes, (
        "a month at exactly the 0.90 gate is at it, not below it"
    )

    between = M.screen_factor("yeslab_m", {d: 1000 for d in days[:7]}, universe)
    assert between.minimum_monthly_day_ratio == 0.7
    assert "MISSING_MONTH_OR_LOW_MONTHLY_COVERAGE" in between.reason_codes, (
        "0.70 is below the 0.90 gate; this half is what pins the value against a lowered one"
    )


def test_the_p05_gate_is_0_30_and_is_the_fifth_percentile_not_the_tenth() -> None:
    """A p05 exactly at 0.30 passes, and a series whose p05 fails but p10 passes must fail.

    The first half pins `<` against `<=`. The second half pins the quantile *level*: with ratios
    0.20, 0.25, 0.40 and then full coverage, the 5th percentile is 0.25 (below the gate) while the
    10th is 0.40 (above it), so computing p10 instead of p05 would pass a factor the C6.2 rule
    rejects. Both cases hold the median at 1.0 so the verdict is attributable to p05 alone.
    """
    days = _days(date(2020, 1, 1), 21)
    universe = _universe(days, size=1000)

    at_gate = {days[0]: 200, days[1]: 300, **{d: 1000 for d in days[2:]}}
    result = M.screen_factor("yeslab_p", at_gate, universe)
    assert result.p05_coverage_ratio == 0.30
    assert result.median_coverage_ratio == 1.0
    assert "LOW_P05_SAMPLE_COVERAGE" not in result.reason_codes
    assert result.coverage_gate_pass

    below = {days[0]: 200, days[1]: 250, days[2]: 400, **{d: 1000 for d in days[3:]}}
    lower = M.screen_factor("yeslab_p", below, universe)
    assert lower.p05_coverage_ratio == 0.25
    assert lower.median_coverage_ratio == 1.0
    assert "LOW_P05_SAMPLE_COVERAGE" in lower.reason_codes
    assert not lower.coverage_gate_pass


def test_median_gate_is_its_own_constant_and_supplies_the_reason_code() -> None:
    """MEDIAN_GATE and COVERAGE_GATE are two constants that happen to share the value 0.70.

    Asserted separately because they do different jobs and only one of them is load-bearing today.
    MEDIAN_GATE decides whether LOW_MEDIAN_SAMPLE_COVERAGE is reported; COVERAGE_GATE contributes to
    gate_pass, but since a median below it has already collected the MEDIAN_GATE reason, lowering
    COVERAGE_GATE alone cannot change any verdict -- which a mutation run confirmed. Pinning both
    values means a future divergence between them is a deliberate act rather than a silent one.
    """
    assert M.MEDIAN_GATE == 0.70
    assert M.COVERAGE_GATE == 0.70

    days = _days(date(2020, 1, 1), 21)
    below = M.screen_factor("yeslab_med", {d: 600 for d in days}, _universe(days, size=1000))
    assert below.median_coverage_ratio == 0.60
    assert "LOW_MEDIAN_SAMPLE_COVERAGE" in below.reason_codes
    assert not below.coverage_gate_pass


def test_the_quantile_takes_the_nearest_rank_not_the_floor() -> None:
    """`round(q * (n - 1))`, not `int(...)`: with four values the median is the third, not the
    second.

    The existing edge-case test uses five values, where q * (n - 1) is a whole number and rounding
    versus flooring agree, so it cannot tell them apart. Four values put the index at 1.5, where
    they
    differ. Nearest-rank is the documented rule in the helper's own docstring.
    """
    assert M._quantile([0.1, 0.2, 0.3, 0.4], 0.5) == 0.3
    assert M._quantile([0.1, 0.2, 0.3, 0.4], 0.05) == 0.1


def test_coverage_gate_fail_is_unreachable_while_the_two_median_constants_agree() -> None:
    """Sweep the whole ratio range and assert the catch-all label never appears.

    This is a pin on a dead branch, recorded because the same label is very much alive in the other
    instrument. record_structural_quality_snapshot.py emits COVERAGE_GATE_FAIL from a plain else for
    ANY of six failing conjuncts, and reading that label as "the median was too low" sent two rounds
    of the audit in the wrong direction before the conjunct that had actually failed -- a whole
    month
    with zero observations -- was extracted by hand. Here the label cannot be emitted at all, so a
    reader who sees COVERAGE_GATE_FAIL knows it came from the snapshot and not from this screen.

    If MEDIAN_GATE is ever lowered below COVERAGE_GATE the branch becomes live and this test fails,
    which is the point: that change would alter what the reason codes mean.
    """
    days = _days(date(2020, 1, 1), 21)
    universe = _universe(days, size=1000)
    seen: set[str] = set()
    for rows in range(0, 1001, 7):
        seen.update(M.screen_factor("yeslab_sweep", {d: rows for d in days}, universe).reason_codes)
    assert "COVERAGE_GATE_FAIL" not in seen
    assert M.COVERAGE_GATE == M.MEDIAN_GATE, (
        "the two constants have diverged, so COVERAGE_GATE_FAIL is now reachable and the reason "
        "codes this screen emits have changed meaning"
    )
