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
