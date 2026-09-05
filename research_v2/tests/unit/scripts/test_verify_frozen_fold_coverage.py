"""Tests for the frozen per-fold coverage gate (reconstruction plan Task 3.1b).

The fixtures are the measured 2026-09-05 numbers, so these tests encode the finding
itself: a day-share threshold cannot catch the crash-day collapse, because 11 lost
days inside a five-year train segment is a 99.1% valid share.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "verify_frozen_fold_coverage.py"
)


def _load():
    spec = importlib.util.spec_from_file_location("verify_frozen_fold_coverage", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["verify_frozen_fold_coverage"] = module
    spec.loader.exec_module(module)
    return module


M = _load()

# Measured 2015: 244 trading days, median all-17 width 211, minimum 2, and 11 days
# below the floor of 30 (2015-08-24 = 2 stocks, the ~2000-limit-down 'Black Monday').
TRAIN_2015_WIDTHS = [211] * 233 + [2, 4, 7, 8, 8, 9, 9, 13, 21, 25, 29]
# Measured test years 2020-2025: no day below the floor; minima 74-239.
TEST_WIDTHS = [599] * 200 + [121, 166, 76, 239, 94, 74]


def test_test_year_profile_passes() -> None:
    a = M.assess_coverage("F01", "test", (2020,), {2020: TEST_WIDTHS})
    assert a.passed
    assert a.days_below_floor == 0
    assert a.valid_day_share == 1.0
    assert a.min_width == 74


def test_crash_year_train_profile_fails() -> None:
    a = M.assess_coverage("F02", "train", (2015,), {2015: TRAIN_2015_WIDTHS})
    assert not a.passed
    assert a.days_below_floor == 11
    assert a.min_width == 2
    assert any("below the cross-section floor" in f for f in a.failures)


def test_day_share_alone_would_miss_the_defect() -> None:
    """The reason the hard gate counts days rather than trusting a share threshold.

    A realistic F02 train segment spans 2015-2019 (~1,220 days). Embedding the same 11
    lost crash days in it yields a 99.1% valid share, which sails through a 0.95 gate
    -- the same threshold G6 uses. The count-based gate must still fail.
    """
    five_year = {
        2015: TRAIN_2015_WIDTHS,
        2016: [269] * 244,
        2017: [360] * 244,
        2018: [450] * 244,
        2019: [520] * 244,
    }
    a = M.assess_coverage("F02", "train", tuple(five_year), five_year)
    assert a.days == 1220  # 5 years x 244 trading days
    assert a.days_below_floor == 11
    assert a.valid_day_share > 0.99, "expected the share gate to be far too weak here"
    assert not a.passed, "count-based gate must fail even though the share is 99%+"
    lenient = M.assess_coverage(
        "F02", "train", tuple(five_year), five_year, max_days_below_floor=10**6
    )
    assert lenient.passed


def test_empty_segment_fails_closed() -> None:
    a = M.assess_coverage("F01", "train", (2014,), {2014: []})
    assert not a.passed
    assert any("no trading days" in f for f in a.failures)


def test_zero_width_day_fails() -> None:
    a = M.assess_coverage("F01", "train", (2014,), {2014: [300, 0, 310]})
    assert not a.passed
    assert any("empty complete-factor cross-section" in f for f in a.failures)


def test_report_only_mode_records_without_failing() -> None:
    a = M.assess_coverage(
        "F02", "train", (2015,), {2015: TRAIN_2015_WIDTHS}, enforce=False
    )
    assert a.passed
    assert a.failures == ()
    assert a.days_below_floor == 11
    assert any("below the cross-section floor" in n for n in a.notes)


def test_thresholds_are_validated() -> None:
    with pytest.raises(ValueError):
        M.assess_coverage("F01", "train", (2014,), {2014: [300]}, minimum_cross_section=1)
    with pytest.raises(ValueError):
        M.assess_coverage("F01", "train", (2014,), {2014: [300]}, min_valid_day_share=0.0)
    with pytest.raises(ValueError):
        M.assess_coverage("F01", "train", (2014,), {2014: [300]}, max_days_below_floor=-1)


def test_asymmetry_note_reports_the_measured_shift() -> None:
    """Train median 211 vs test median 599 is the measured 2015/2020 contrast."""
    train = M.assess_coverage(
        "F02", "train", (2015,), {2015: TRAIN_2015_WIDTHS}, enforce=False
    )
    test = M.assess_coverage("F02", "test", (2020,), {2020: TEST_WIDTHS})
    note = M.asymmetry_note(train, test)
    assert note is not None
    assert "ratio 0.35" in note
    assert train.median_width == 211 and test.median_width == 599


def test_asymmetry_note_is_none_when_a_side_is_degenerate() -> None:
    empty = M.assess_coverage("F01", "train", (2014,), {2014: []}, enforce=False)
    test = M.assess_coverage("F01", "test", (2020,), {2020: TEST_WIDTHS})
    assert M.asymmetry_note(empty, test) is None


def test_fold_role_triples_skip_the_locked_final_fold() -> None:
    triples = M.fold_role_years()
    folds = {fold for fold, _, _ in triples}
    assert folds == {"F01", "F02", "F03", "F04", "F05", "F06"}
    assert len(triples) == 18
    roles = {role for _, role, _ in triples}
    assert roles == {"train", "validation", "test"}
    f01_train = next(years for f, r, years in triples if f == "F01" and r == "train")
    assert f01_train == (2014, 2015, 2016, 2017, 2018)
