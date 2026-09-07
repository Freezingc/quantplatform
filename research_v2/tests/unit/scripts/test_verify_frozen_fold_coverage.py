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
        "F02",
        "train",
        tuple(five_year),
        five_year,
        max_days_below_floor=10**6,
        # Relax the width criterion too, so this case isolates the day count alone.
        minimum_median_width=0,
    )
    assert lenient.passed


def test_empty_segment_fails_closed() -> None:
    a = M.assess_coverage("F01", "train", (2014,), {2014: []})
    assert not a.passed
    assert any("no trading days" in f for f in a.failures)


def test_zero_width_days_fail_once_they_exceed_the_calibrated_allowance() -> None:
    """An empty day is the extreme case of a day below the floor, not a separate rule.

    One empty day is tolerated because the accepted criterion-303 baseline has exactly
    one (2017-10-09, where yeslab_387 is absent at source). Two are not, and the
    failure message still names the empty cross-section so the diagnostic survives.
    """
    # A realistic segment length, so the day-share criterion is not what decides this.
    tolerated = M.assess_coverage("F01", "train", (2014,), {2014: [1402] * 1213 + [0]})
    assert tolerated.days_below_floor == 1
    assert tolerated.passed

    a = M.assess_coverage("F01", "train", (2014,), {2014: [300, 0, 0, 310]})
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
    with pytest.raises(ValueError):
        M.assess_coverage("F01", "train", (2014,), {2014: [300]}, minimum_median_width=-1)


# --- calibrated criteria (measured 2026-09-06) --------------------------------
#
# A gate threshold cannot be chosen from the defective side alone: it has to admit the
# baseline already accepted, otherwise adopting it retroactively invalidates the trust
# root instead of protecting the next one. audit_legacy_train_coverage.py measured the
# legacy-14 side from the assembled frames criterion-303 actually consumed:
#   train median cross-section  1402 / 1615 / 1949 / 2464 / 2656 / 2794
#   train days below floor           1 /   1 /   1 /   1 /   0 /   0
#   worst single train day             0 (2017-10-09, yeslab_387 absent at source)
# against frozen-17's
#   train median cross-section   310 /  343 /  419 /  513 /  600 /  673
#   train days below floor        15 /  15 /   4 /   0 /   0 /   0
#   worst single train day         2
# The widths do not overlap anywhere (4.2-4.8x per fold), which is what makes an
# absolute floor usable. The train/test ratios DO overlap (0.53-0.74 against
# 0.49-0.77), which is what makes the ratio unusable as a criterion.

LEGACY_TRAIN_MEDIANS = (1402, 1615, 1949, 2464, 2656, 2794)
FROZEN_TRAIN_MEDIANS = (310, 343, 419, 513, 600, 673)
LEGACY_DAYS_BELOW_FLOOR = (1, 1, 1, 1, 0, 0)
FROZEN_DAYS_BELOW_FLOOR = (15, 15, 4, 0, 0, 0)


def test_width_floor_sits_inside_the_measured_separation_gap() -> None:
    """Guard the constant against drifting out of the range the evidence supports."""
    floor = M.DEFAULT_MINIMUM_MEDIAN_WIDTH
    assert max(FROZEN_TRAIN_MEDIANS) < floor < min(LEGACY_TRAIN_MEDIANS)


def test_day_allowance_is_the_measured_legacy_worst_case() -> None:
    allowance = M.DEFAULT_MAX_DAYS_BELOW_FLOOR
    assert allowance == max(LEGACY_DAYS_BELOW_FLOOR)
    assert allowance < min(
        count for count in FROZEN_DAYS_BELOW_FLOOR if count > 0
    )


def _segment(
    median: int, days: int, *, below_floor: int, min_width: int = 2
) -> dict[int, list[int]]:
    """A width profile faithful to the measured summary of one train segment."""
    widths = [median] * (days - below_floor) + [min_width] * below_floor
    return {2019: widths}


def test_calibrated_gate_admits_every_legacy_fold() -> None:
    for index, median in enumerate(LEGACY_TRAIN_MEDIANS):
        below = LEGACY_DAYS_BELOW_FLOOR[index]
        # The legacy worst day is empty (width 0), not merely narrow.
        profile = _segment(median, 1214 - index, below_floor=below, min_width=0)
        a = M.assess_coverage(f"F0{index + 1}", "train", (2019,), profile)
        assert a.passed, f"legacy F0{index + 1} must be admitted: {a.failures}"


def test_calibrated_gate_rejects_the_frozen_folds_that_lose_days() -> None:
    for index, median in enumerate(FROZEN_TRAIN_MEDIANS):
        below = FROZEN_DAYS_BELOW_FLOOR[index]
        profile = _segment(median, 1220 - index, below_floor=below)
        a = M.assess_coverage(f"F0{index + 1}", "train", (2019,), profile)
        if below:
            assert not a.passed, f"frozen F0{index + 1} loses {below} days and must fail"
        # Every frozen fold is below the width floor, including the ones that lose no day.
        assert any("median" in failure for failure in a.failures), (
            f"frozen F0{index + 1} median {median} is below the floor and must fail on it"
        )


def test_width_floor_applies_to_validation_but_not_to_test() -> None:
    """Validation picks the winning baseline, so it is a decision role; test only scores.

    Frozen test medians (599-908) sit below the floor, but a narrow TEST cross-section
    is already governed by the >=30-per-day validity rule that G6 reads. Applying the
    training-adequacy floor there would gate a quantity the evaluation already handles.
    """
    validation = M.assess_coverage("F01", "validation", (2019,), {2019: [523] * 244})
    test = M.assess_coverage("F01", "test", (2020,), {2020: [599] * 243})

    assert not validation.passed
    assert any("median" in failure for failure in validation.failures)
    assert test.passed


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
    assert folds == {"F1", "F2", "F3", "F4"}
    assert len(triples) == 12
    roles = {role for _, role, _ in triples}
    assert roles == {"train", "validation", "test"}
    f1_train = next(years for f, r, years in triples if f == "F1" and r == "train")
    assert f1_train == (2017, 2018, 2019, 2020)


# --- measuring a candidate factor subset before it is chosen ------------------
#
# Adjudication 4's option (iii) keeps only the eleven factors that pass the C6.2
# coverage screen. Whether that set clears the calibrated width floor was an inference
# from survival percentages, not a measurement; these make it measurable without
# wiring anything, so the choice can be made on evidence either way.

FROZEN17 = (
    "yeslab_96", "yeslab_206", "yeslab_357", "yeslab_372", "yeslab_375",
    "yeslab_381", "yeslab_384", "yeslab_389", "yeslab_390", "yeslab_398",
    "yeslab_405", "yeslab_406", "yeslab_425", "yeslab_443_reg30_max20_v1",
    "yeslab_464", "yeslab_471", "yeslab_484",
)


def test_c62_pass_set_is_the_screened_eleven_and_excludes_the_six_defective() -> None:
    assert len(M.C62_PASS_11) == 11
    assert set(M.C62_PASS_11) <= set(FROZEN17)
    defective = {"yeslab_372", "yeslab_375", "yeslab_384", "yeslab_389",
                 "yeslab_390", "yeslab_406"}
    assert not (set(M.C62_PASS_11) & defective)
    assert set(FROZEN17) - set(M.C62_PASS_11) == defective


def test_resolve_factor_set_returns_the_frozen_set_by_default() -> None:
    assert M.resolve_factor_set("frozen17", FROZEN17) == FROZEN17


def test_resolve_factor_set_returns_the_screened_subset() -> None:
    resolved = M.resolve_factor_set("c62pass11", FROZEN17)

    assert resolved == M.C62_PASS_11
    assert len(resolved) == 11


def test_resolve_factor_set_rejects_an_unknown_name() -> None:
    with pytest.raises(ValueError, match="unknown factor set"):
        M.resolve_factor_set("dense12", FROZEN17)


def test_resolve_factor_set_fails_closed_when_the_subset_leaves_the_frozen_set() -> None:
    """The frozen set is the authority; a stale subset must not be measured silently."""
    shrunken = tuple(fid for fid in FROZEN17 if fid != "yeslab_484")

    with pytest.raises(ValueError, match="not a subset"):
        M.resolve_factor_set("c62pass11", shrunken)


def test_build_width_query_restricts_to_the_named_factors_and_their_count() -> None:
    script = M.build_width_query(2020, ("yeslab_96", "yeslab_206"), "T")

    assert "factor_id in [`yeslab_96,`yeslab_206]" in script
    assert "iif(c==2,1,0)" in script
    assert "2020.01.01" in script and "2021.01.01" in script
    assert "group by signal_date, ts_code" in script


def test_build_width_query_uses_count_star_because_distinct_is_not_nestable() -> None:
    """DolphinDB rejects count(distinct ...) inside a grouped select (RefId:S02032).

    count(*) is only equivalent to the number of distinct factors present when the
    table has no duplicate (factor, date, code) rows, which is verified separately:
    zero duplicate groups in 2020-01, and the C6.2 screen reports dups==0 for all 17
    factors over both 2014-2019 and 2020-2024.
    """
    script = M.build_width_query(2015, M.C62_PASS_11, "T")

    assert "count(*)" in script
    assert "distinct" not in script
    assert "iif(c==11,1,0)" in script


def test_build_width_query_rejects_an_empty_factor_set() -> None:
    with pytest.raises(ValueError, match="cannot be empty"):
        M.build_width_query(2020, (), "T")


# --- "at least N of the set" semantics (adjudication 4's option (i)) -----------
#
# B1 EqualWeightBaseline already tolerates partial coverage by design: its
# minimum_factor_coverage=0.60 means a row needs >=11 of the 17 frozen factors, and it
# omits missing factors rather than imputing them. Option (i) adopts that rule for the
# whole baseline set, so its cross-section is "pairs with at least 11 of 17", not "pairs
# with all 17" and not "pairs with all of some 11-factor subset". Measuring it needs a
# different completeness predicate, and the plan only had sampled-month survival
# percentages for it (84.4% / 98.5% / 99.2%), never per-fold widths.


def test_build_width_query_defaults_to_requiring_every_named_factor() -> None:
    script = M.build_width_query(2020, M.C62_PASS_11, "T")

    assert "iif(c==11,1,0)" in script
    assert ">=" not in script.split("iif(")[1].split(")")[0]


def test_build_width_query_supports_an_at_least_predicate() -> None:
    script = M.build_width_query(2020, FROZEN17, "T", minimum_factors=11)

    assert "iif(c>=11,1,0)" in script
    # Still restricted to the named set, or "at least 11" would count other factors.
    assert "factor_id in [`yeslab_96" in script
    assert "`yeslab_484]" in script


def test_build_width_query_rejects_an_unusable_at_least_threshold() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        M.build_width_query(2020, FROZEN17, "T", minimum_factors=18)
    with pytest.raises(ValueError, match="at least 1"):
        M.build_width_query(2020, FROZEN17, "T", minimum_factors=0)


def test_at_least_the_full_count_is_the_same_predicate_as_the_default() -> None:
    """Requiring >=17 of 17 must not silently differ from requiring all 17."""
    explicit = M.build_width_query(2020, FROZEN17, "T", minimum_factors=17)
    default = M.build_width_query(2020, FROZEN17, "T")

    assert "iif(c>=17,1,0)" in explicit
    assert "iif(c==17,1,0)" in default


# The three tests below pin the boundary of each numeric comparison in assess_coverage. They exist
# because a mutation run over that function found all three comparisons unpinned: flipping `<` to
# `<=` in the cross-section floor, in the valid-day share, and in the calibrated median width each
# left all 27 pre-existing tests green. Two further survivors were checked and are NOT holes -- the
# days-below-floor allowance is pinned once both of its occurrences are mutated together (mutating
# only one desynchronises the two branches that read the same constant, which is not a change anyone
# could ship), and dropping the `minimum_median_width > 0` guard is an equivalent mutation, since
# widths are non-negative so a floor of 0 fails nothing either way.
#
# Each test asserts both sides of its boundary. One side alone would not pin the comparison: a test
# asserting only that 29 fails leaves `<=` and `<` indistinguishable, which is exactly the shape
# C7.5 had to rewrite.


def test_a_day_exactly_at_the_cross_section_floor_is_not_below_it() -> None:
    """`width < minimum_cross_section`, not `<=`: a day at the floor is a valid day.

    The floor is the per-day minimum the assembly needs to compute a cross-sectional statistic at
    all, so a day with exactly that many stocks has one and must not be counted against the
    calibrated allowance. Counting it would consume the single tolerated day on a segment that is
    in fact fine.
    """
    at_floor = M.assess_coverage(
        "F01", "test", (2020,), {2020: [30, 30, 30, 30]}, minimum_cross_section=30
    )
    assert at_floor.days_below_floor == 0
    assert at_floor.min_width == 30
    assert at_floor.passed

    one_under = M.assess_coverage(
        "F01", "test", (2020,), {2020: [29, 29, 29, 29]}, minimum_cross_section=30
    )
    assert one_under.days_below_floor == 4
    assert not one_under.passed


def test_a_valid_day_share_exactly_at_the_threshold_passes() -> None:
    """`share < min_valid_day_share`, not `<=`: meeting the threshold exactly is meeting it.

    Nineteen valid days out of twenty is a share of exactly 0.95, which is the default threshold.
    A segment that lands precisely on a published threshold passes it; requiring strictly more would
    make the documented number mean something other than what it says.
    """
    widths = [1402] * 19 + [5]

    at_threshold = M.assess_coverage(
        "F01", "test", (2020,), {2020: widths}, min_valid_day_share=0.95
    )
    assert at_threshold.valid_day_share == pytest.approx(0.95)
    assert at_threshold.days_below_floor == 1  # inside the calibrated allowance of 1
    assert at_threshold.passed

    above_threshold = M.assess_coverage(
        "F01", "test", (2020,), {2020: widths}, min_valid_day_share=0.96
    )
    assert not above_threshold.passed
    assert any("valid day share" in failure for failure in above_threshold.failures)


def test_a_median_exactly_at_the_calibrated_floor_passes() -> None:
    """`median_width < minimum_median_width`, not `<=`, on the constant the calibration chose.

    1000 is not an arbitrary number: it was placed inside a measured 4.2-4.8x gap between the
    frozen-17 train medians (310-673) and the accepted legacy-14 ones (1402-2794). A segment whose
    median is exactly the floor satisfies the criterion the calibration defined, and one below it
    does not -- which is the distinction that makes all six frozen train segments fail and all six
    legacy ones pass.
    """
    at_floor = M.assess_coverage(
        "F01", "train", (2014,), {2014: [1000] * 10}, minimum_median_width=1000
    )
    assert at_floor.median_width == 1000
    assert at_floor.passed

    one_under = M.assess_coverage(
        "F01", "train", (2014,), {2014: [999] * 10}, minimum_median_width=1000
    )
    assert one_under.median_width == 999
    assert not one_under.passed
    assert any("below the calibrated floor" in failure for failure in one_under.failures)
