"""Tests for the factor_daily backfill verifier (reconstruction plan Task 3.0)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).resolve().parents[3] / "scripts" / "verify_factor_daily_backfill.py"


def _load():
    spec = importlib.util.spec_from_file_location("verify_factor_daily_backfill", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_factor_daily_backfill"] = module
    spec.loader.exec_module(module)
    return module


M = _load()

# Independently verified against the live published_version table: the
# YYYY-01-01..YYYY-12-31 window convention reproduces the registered hashes.
KNOWN_VERSIONS = {
    2014: "9ff10d94340f92b2ee06f2662d4701e1461a66830397168d8f123260f77dfd56",
    2020: "66f33abd966703aa87658fd3477959cf6b086ab9ef9c6e946f175b8e32f9cfeb",
    2025: "7f0636bc639a3f2e555b066abffb13b18580d66a63e20c27a0f78d3da65b6271",
}


def test_predicted_version_reproduces_registered_hashes() -> None:
    for year, expected in KNOWN_VERSIONS.items():
        assert M.predicted_version(year) == expected


def test_predicted_version_is_window_sensitive() -> None:
    """Each year must get its own version, which is why per-year runs are mandatory."""
    versions = {M.predicted_version(year) for year in range(2014, 2026)}
    assert len(versions) == 12


def test_parse_year_range() -> None:
    assert M.parse_year_range("2014..2016") == [2014, 2015, 2016]
    assert M.parse_year_range("2019") == [2019]
    with pytest.raises(ValueError):
        M.parse_year_range("2016..2014")
    with pytest.raises(ValueError):
        M.parse_year_range("14..2016")


def test_parse_dryrun_expectations_ignores_noise() -> None:
    log = "\n".join(
        [
            "frozen factors: 17; materializing: 17",
            "market rows: 1183530",
            "yeslab_206: 329071 universe-aligned rows",
            "yeslab_206: written",
            "AttentionSpillover 进度: 503/503",
            "yeslab_406: 195377 universe-aligned rows",
            "Dry-run only. Re-run with --apply to write.",
        ]
    )
    assert M.parse_dryrun_expectations(log) == {"yeslab_206": 329071, "yeslab_406": 195377}


def test_row_tolerance_is_four_or_two_hundredths_of_a_percent() -> None:
    assert M.row_tolerance(0) == 4
    assert M.row_tolerance(195377) == 4
    assert M.row_tolerance(1_000_000) == 20


def _clean_state(count: int = 100) -> dict:
    # 16 numbered ids plus the real yeslab_406, so the marginal-float case below can
    # target the factor that actually exhibited it without changing the factor count.
    names = [f"yeslab_{index}" for index in range(M.FROZEN_FACTOR_COUNT - 1)]
    names.append("yeslab_406")
    factors = {name: count for name in names}
    total = count * M.FROZEN_FACTOR_COUNT
    return {
        "versions": {M.predicted_version(2014): total},
        "factor_counts": factors,
        "duplicates": 0,
    }


def test_assess_year_passes_on_clean_state() -> None:
    assessment = M.assess_year(2014, **_clean_state())
    assert assessment.passed
    assert assessment.failures == ()
    assert assessment.factor_count == M.FROZEN_FACTOR_COUNT
    assert assessment.data_version == M.predicted_version(2014)


def test_assess_year_rejects_wrong_version() -> None:
    state = _clean_state()
    state["versions"] = {"deadbeef" * 8: 1700}
    assessment = M.assess_year(2014, **state)
    assert not assessment.passed
    assert any("data_version" in failure for failure in assessment.failures)


def test_assess_year_rejects_mixed_versions_in_one_year() -> None:
    """A single year must carry exactly one data_version (per-year window invariant)."""
    state = _clean_state()
    state["versions"] = {M.predicted_version(2014): 900, M.predicted_version(2015): 800}
    assessment = M.assess_year(2014, **state)
    assert not assessment.passed
    assert assessment.data_version is None


def test_assess_year_rejects_missing_factor() -> None:
    state = _clean_state()
    state["factor_counts"] = dict(list(state["factor_counts"].items())[:-1])
    assessment = M.assess_year(2014, **state)
    assert not assessment.passed
    assert any("factor count 16" in failure for failure in assessment.failures)


def test_assess_year_rejects_duplicate_business_keys() -> None:
    state = _clean_state()
    state["duplicates"] = 3
    assessment = M.assess_year(2014, **state)
    assert not assessment.passed
    assert any("duplicate" in failure for failure in assessment.failures)


def test_assess_year_tolerates_the_observed_one_row_marginal_float_delta() -> None:
    """The real 2014 case: yeslab_406 stored one row more than its dry-run predicted."""
    state = _clean_state()
    expectations = dict(state["factor_counts"])
    expectations["yeslab_406"] = 195377
    state["factor_counts"]["yeslab_406"] = 195378
    state["versions"] = {M.predicted_version(2014): sum(state["factor_counts"].values())}
    assessment = M.assess_year(2014, expectations=expectations, **state)
    assert assessment.passed
    assert any("yeslab_406" in note for note in assessment.notes)


def test_assess_year_fails_when_rows_exceed_tolerance() -> None:
    state = _clean_state()
    expectations = dict(state["factor_counts"])
    state["factor_counts"]["yeslab_5"] = 120
    state["versions"] = {M.predicted_version(2014): sum(state["factor_counts"].values())}
    assessment = M.assess_year(2014, expectations=expectations, **state)
    assert not assessment.passed
    assert any("yeslab_5" in failure for failure in assessment.failures)


def test_assess_year_fails_closed_on_empty_region() -> None:
    assessment = M.assess_year(2014, versions={}, factor_counts={}, duplicates=0)
    assert not assessment.passed
    assert any("factor count 0" in failure for failure in assessment.failures)


# The five tests below came from a 14-mutation run over this script, in which five survived and all
# five turned out to be genuine rather than masked. This is the FIRST of the five pre-flight doors:
# preflight_frozen_baseline.run_gates stops early when it fails and marks the other four as failed
# without running them, so a door that reports complete when the data is not is worse here than
# anywhere else in the chain -- the later doors never get the chance to disagree.
#
# Three of the five are cases where another check in the same function looked like it should have
# caught them and does not: the factor-count gate does not notice one factor being replaced by
# another, the total-rows gate does not notice a shortfall smaller than its own tolerance, and the
# version gate does not notice seventeen factors that are all present and all empty.


_FIDS = [f"yeslab_{i}" for i in range(17)]
_EXPECT = {f: 6_000_000 for f in _FIDS}


def _assess(factor_counts, expectations=_EXPECT, duplicates=0, year=2020):
    return M.assess_year(
        year,
        versions={M.predicted_version(year): sum(factor_counts.values())},
        factor_counts=factor_counts,
        duplicates=duplicates,
        expectations=expectations,
    )


def test_a_factor_replaced_by_an_impostor_fails_even_though_the_count_is_right() -> None:
    """`observed is None` is load-bearing: the count gate cannot see a swapped identity.

    Seventeen factors are present, so `factor_count != FROZEN_FACTOR_COUNT` does not fire, and the
    impostor is not in the expectations so it is never examined. The only thing that notices the
    expected factor is absent is the `observed is None` half of the per-factor check. Dropping it
    turns a correct FAIL into a PASS, which is the realistic shape of a factor being renamed or
    substituted during a re-materialization.
    """
    swapped = {f: 6_000_000 for f in _FIDS}
    del swapped[_FIDS[0]]
    swapped["yeslab_IMPOSTOR"] = 6_000_000

    result = _assess(swapped)
    assert result.factor_count == 17, "the count gate sees nothing wrong here, by construction"
    assert not result.passed
    assert any(_FIDS[0] in failure and "None" in failure for failure in result.failures), (
        f"the absent factor must be named: {result.failures}"
    )


def test_an_undercount_within_the_total_tolerance_is_still_caught_per_factor() -> None:
    """`abs()` is load-bearing, and the total gate cannot cover for it.

    Per-factor tolerance is max(4, 0.002% of 6,000,000) = 120 while the total tolerance over
    102,000,000 rows is 2,040. A single factor short by 500 exceeds its own tolerance but is well
    inside the total's, so removing abs() -- which leaves only over-counts
    detectable -- would pass a
    year that is missing 500 rows of one factor. Under-counting is the direction that matters here:
    it is what an interrupted materialization produces.
    """
    short = {f: 6_000_000 for f in _FIDS}
    short[_FIDS[1]] = 6_000_000 - 500

    assert M.row_tolerance(6_000_000) == 120
    assert M.row_tolerance(sum(_EXPECT.values())) == 2040
    result = _assess(short)
    assert not result.passed
    assert any(_FIDS[1] in failure for failure in result.failures)
    assert not any("total rows" in failure for failure in result.failures), (
        "the total gate does not catch this one, which is why the per-factor abs() must"
    )

    over = {f: 6_000_000 for f in _FIDS}
    over[_FIDS[1]] = 6_000_000 + 500
    assert not _assess(over).passed, "the same magnitude over-count must fail identically"


def test_a_total_over_by_exactly_its_tolerance_passes_and_one_more_row_does_not() -> None:
    """`>` not `>=` on the total, pinned from both sides.

    Spread the excess so no single factor exceeds its own tolerance: seventeen factors each over by
    exactly 120 puts the total over by exactly 2,040, which is the total tolerance. At the tolerance
    the year passes; one row more and it does not. Without the passing half, `>` and `>=` are
    indistinguishable and the published tolerance means something other than what it says.
    """
    at_tolerance = {f: 6_000_000 + 120 for f in _FIDS}
    total_excess = sum(at_tolerance.values()) - sum(_EXPECT.values())
    assert total_excess == M.row_tolerance(sum(_EXPECT.values()))

    assert _assess(at_tolerance).passed

    one_over = dict(at_tolerance)
    one_over[_FIDS[0]] += 1
    assert not _assess(one_over).passed


def test_seventeen_factors_present_and_all_empty_is_a_failure() -> None:
    """The `total <= 0` branch is the only thing standing between an empty year and a PASS.

    With all seventeen factor ids present, the factor-count gate is satisfied and the version gate
    can
    be satisfied, so an empty partition would clear every other check. Changing `<=` to `<` makes a
    year in which nothing was materialized report as complete -- and since this door short-circuits
    the other four, that empty year would then be waved through the whole pre-flight.
    """
    empty = {f: 0 for f in _FIDS}
    result = _assess(empty, expectations=None)
    assert not result.passed
    assert any("total rows is 0" in failure for failure in result.failures)


def test_main_exits_nonzero_when_a_year_fails(monkeypatch) -> None:
    """The exit code is what the pre-flight actually reads, and nothing tested it.

    `preflight_frozen_baseline._run` executes this script with runpy and takes the SystemExit code
    as
    the door's verdict, so `return 1 if failed else 0` is the only line that decides whether the
    first
    door passes. No test called main() at all: forcing that line to `return 0`
    left all thirteen green,
    which would have made an incomplete factor_daily clear the pre-flight and, because run_gates
    stops
    early on this door, mark the other four as failed without running them -- the opposite of the
    intended fail-closed behaviour in both directions at once.
    """
    import walk_forward_common

    monkeypatch.setattr(walk_forward_common, "new_holder", lambda: {})

    def state(holder, year):
        counts = {f: 6_000_000 for f in _FIDS}
        return {M.predicted_version(year): sum(counts.values())}, counts, 0

    monkeypatch.setattr(M, "collect_year_state", state)
    assert M.main(["--years", "2020"]) == 0

    def broken(holder, year):
        return {}, {}, 0

    monkeypatch.setattr(M, "collect_year_state", broken)
    assert M.main(["--years", "2020"]) == 1, "an empty year must make the process exit non-zero"

    monkeypatch.setattr(M, "collect_year_state", state)
    assert M.main(["--years", "2020..2021"]) == 0
    monkeypatch.setattr(
        M, "collect_year_state",
        lambda holder, year: broken(holder, year) if year == 2021 else state(holder, year),
    )
    assert M.main(["--years", "2020..2021"]) == 1, "one bad year in a range must fail the run"
