"""Per-fold coverage gate for the frozen factor set (reconstruction plan Task 3.1b).

The G1-G6 gates do not measure how much of each day's cross-section actually survives
assembly. Measured 2026-09-05, the frozen 17 contain six factors that raise a
sign-capable base to a real power (``pct_chg ** x`` and ``tanh(delta20(vol|turnover))
** x``), which numpy leaves undefined for falling stocks and declining volume. Their
coverage therefore equals the market's advancing breadth: on 2015-08-24 only 2 of
1,466 liquid-universe stocks had all 17 factors, and on the two circuit-breaker days
2016-01-04 / 01-07 only 3 and 4. Eleven days in 2015 and four in 2016 fall below the
``minimum_cross_section`` floor of 30 and are dropped silently.

All six TEST years pass G6 at 100% valid days, so a frozen baseline run would clear
every existing gate and become the new trust root while carrying that defect. This
script is the gate that makes it visible: it reports, per fold and role, the daily
width of the complete-factor cross-section, and fails closed on two calibrated
criteria -- a decision role (train or validation) that loses too many days to the
floor, and a decision role whose median cross-section is too narrow. Both constants
are calibrated against the accepted legacy-14 baseline by
``audit_legacy_train_coverage.py``; see the comments on the defaults for the measured
numbers and for why the train/test width ratio is reported but deliberately not gated.

Scope note: this reads ``factor_daily`` only. That is sound because the frozen table
has exactly one row per ``(factor_id, signal_date, ts_code)`` (verified: zero
duplicate business keys) and exactly one registered ``data_version`` per year, so a
raw count per pair is already version-correct. The legacy path stores multiple
observations per pair and needs ``filter_current`` first, so it is out of scope here;
legacy coverage is measured directly from its assembled frames instead.

Usage:
  verify_frozen_fold_coverage.py                       # all six folds, all three roles
  verify_frozen_fold_coverage.py --min-valid-day-share 0.99
"""

from __future__ import annotations

import argparse
import statistics
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_MINIMUM_CROSS_SECTION = 30
DEFAULT_MIN_VALID_DAY_SHARE = 0.95
# A day-share threshold alone is far too weak here: 11 lost days inside a five-year
# train segment of ~1,220 days is a 99.1% valid share and would pass a 0.95 gate, yet
# those 11 days are precisely 2015's crash days. The hard gate is therefore the count
# of days below the cross-section floor; the share is still reported because it is what
# G6 uses and the contrast is informative.
#
# The allowance is 1, not 0, because the accepted criterion-303 baseline itself loses
# exactly one train day in F01-F04 and that day's width is 0: yeslab_387 has no rows at
# all on 2017-10-09 in factor_observation_clean while the other 13 factors are normal
# (2,645-3,177 rows). A full-history scan of 2014-2025 -- 2,919 trading days x 14
# factors, 40,866 (factor, date) cells -- finds exactly one absent cell, so it is a
# single missing observation at source, not PIT gating and not a holiday-calendar
# effect. Zero tolerance would therefore reject the trust root this gate is meant to
# protect the successor of. Repair that cell and the allowance should return to 0.
# Frozen-17 loses 15/15/4 days in F01/F02/F03, so it fails at either allowance.
DEFAULT_MAX_DAYS_BELOW_FLOOR = 1
# Calibrated 2026-09-06 by audit_legacy_train_coverage.py, which measures the same
# quantity on the assembled frames the accepted baseline actually consumed:
#   train median cross-section   legacy-14  1402 1615 1949 2464 2656 2794
#                                frozen-17   310  343  419  513  600  673
# The two sets do not overlap on any fold (a 4.2-4.8x gap), so an absolute floor is
# usable and 1000 sits inside the measured gap with ~1.4x margin on both sides.
#
# The train/test width RATIO is deliberately NOT a criterion. Measured on the same two
# sets it is 0.53-0.74 for legacy against 0.49-0.77 for frozen-17: the distributions
# overlap almost completely, because a ratio near 0.5 is a property of the walk-forward
# design (a five-year train window dominated by early years, scored against one recent
# year over a universe that grew from ~2,700 to ~4,300 names) rather than of any factor
# set. A ratio threshold low enough to fail frozen-17 would also fail the accepted
# baseline, i.e. it would retroactively invalidate the trust root. asymmetry_note()
# keeps reporting the ratio and keeps not gating it.
DEFAULT_MINIMUM_MEDIAN_WIDTH = 1000
# Roles whose cross-section width determines a modelling decision: train fits the
# parameters, validation selects among B0-B3. TEST only scores, and a narrow test
# cross-section is already governed by the >=30-per-day validity rule that G6 reads,
# so the training-adequacy floor is not applied there.
DECISION_ROLES: tuple[str, ...] = ("train", "validation")


@dataclass(frozen=True)
class CoverageAssessment:
    fold_id: str
    role: str
    years: tuple[int, ...]
    days: int
    days_below_floor: int
    valid_day_share: float
    median_width: int
    min_width: int
    passed: bool
    failures: tuple[str, ...]
    notes: tuple[str, ...] = ()


def assess_coverage(
    fold_id: str,
    role: str,
    years: Sequence[int],
    daily_widths: Mapping[int, Sequence[int]],
    *,
    minimum_cross_section: int = DEFAULT_MINIMUM_CROSS_SECTION,
    min_valid_day_share: float = DEFAULT_MIN_VALID_DAY_SHARE,
    max_days_below_floor: int = DEFAULT_MAX_DAYS_BELOW_FLOOR,
    minimum_median_width: int = DEFAULT_MINIMUM_MEDIAN_WIDTH,
    enforce: bool = True,
) -> CoverageAssessment:
    """Pure verdict for one (fold, role) from its per-day complete-cross-section widths.

    ``daily_widths`` maps a trading day (any hashable key, e.g. an ordinal or ISO
    string) to the number of stocks having every frozen factor that day. ``enforce``
    gates the result; report-only callers pass False to record without failing.
    """
    if minimum_cross_section < 2:
        raise ValueError("minimum_cross_section must be at least 2")
    if not 0.0 < min_valid_day_share <= 1.0:
        raise ValueError("min_valid_day_share must be in (0, 1]")
    if max_days_below_floor < 0:
        raise ValueError("max_days_below_floor cannot be negative")
    if minimum_median_width < 0:
        raise ValueError("minimum_median_width cannot be negative")

    widths = [int(width) for values in daily_widths.values() for width in values]
    days = len(widths)
    failures: list[str] = []
    notes: list[str] = []
    if days == 0:
        failures.append("no trading days observed; assembly would produce an empty segment")
        return CoverageAssessment(
            fold_id=fold_id, role=role, years=tuple(years), days=0,
            days_below_floor=0, valid_day_share=0.0, median_width=0, min_width=0,
            passed=False, failures=tuple(failures), notes=(),
        )

    below = sum(1 for width in widths if width < minimum_cross_section)
    share = (days - below) / days
    median_width = int(statistics.median(widths))
    min_width = min(widths)

    if below > max_days_below_floor:
        message = (
            f"{below} of {days} days fall below the cross-section floor of "
            f"{minimum_cross_section} (allowed {max_days_below_floor}); narrowest day "
            f"had {min_width} stocks, median {median_width}"
        )
        if enforce:
            failures.append(message)
        else:
            notes.append(message)
    if enforce and share < min_valid_day_share:
        failures.append(f"valid day share {share:.3%} < {min_valid_day_share:.2%}")
    # An empty day is the extreme case of a day below the floor, so it is governed by
    # the same calibrated allowance rather than by a separate zero-tolerance rule; it
    # still gets its own message because "the model saw nothing at all on this date"
    # is a different diagnosis from "the model saw 20 names on this date".
    empty_days = sum(1 for width in widths if width == 0)
    if empty_days > max_days_below_floor:
        message = (
            f"{empty_days} day(s) have an empty complete-factor cross-section "
            f"(allowed {max_days_below_floor})"
        )
        if enforce:
            failures.append(message)
        else:
            notes.append(message)
    if role in DECISION_ROLES and minimum_median_width > 0 and median_width < minimum_median_width:
        message = (
            f"median cross-section {median_width} is below the calibrated floor of "
            f"{minimum_median_width} for a {role} segment"
        )
        if enforce:
            failures.append(message)
        else:
            notes.append(message)

    return CoverageAssessment(
        fold_id=fold_id, role=role, years=tuple(years), days=days,
        days_below_floor=below, valid_day_share=share, median_width=median_width,
        min_width=min_width, passed=not failures, failures=tuple(failures),
        notes=tuple(notes),
    )


def asymmetry_note(
    train: CoverageAssessment, test: CoverageAssessment
) -> str | None:
    """Report the train/test cross-section width ratio, which no G gate measures.

    The frozen set is fitted on a materially narrower cross-section than it is scored
    on (median 211-360 stocks in train years 2014-2017 versus 599-908 in test years
    2020-2025), and the narrower side is the more direction-selected of the two. This
    is reported rather than gated, and the threshold question is now closed: measuring
    the accepted legacy-14 baseline with the same instrument gives ratios of
    0.53/0.58/0.60/0.69/0.70/0.74 against this set's 0.52/0.49/0.60/0.66/0.77/0.74
    (``audit_legacy_train_coverage.py``, 2026-09-06). The two distributions overlap
    almost completely, so a ratio near 0.5 is a property of the walk-forward design --
    a five-year train window dominated by early years, scored against one recent year
    over a universe that grew from ~2,700 to ~4,300 names -- not of the frozen factor
    set. Any threshold low enough to fail this set would also fail the accepted
    baseline. Gate on the absolute train width instead, where the two do not overlap
    (legacy 1402-2794 versus frozen 310-673, a 4.2-4.8x gap on every fold).
    """
    if test.median_width <= 0 or train.median_width <= 0:
        return None
    ratio = train.median_width / test.median_width
    return (
        f"{train.fold_id}: train median cross-section {train.median_width} vs test "
        f"{test.median_width} -> ratio {ratio:.2f}"
    )


def fold_role_years() -> list[tuple[str, str, tuple[int, ...]]]:
    """The (fold, role, years) triples the frozen baseline will assemble."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from research_v2.datasets.splits import annual_folds

    triples: list[tuple[str, str, tuple[int, ...]]] = []
    for fold in annual_folds():
        if fold.locked:
            continue  # 2026 FINAL is never touched
        triples.append((fold.fold_id, "train", tuple(fold.train_years)))
        triples.append((fold.fold_id, "validation", tuple(fold.validation_years)))
        triples.append((fold.fold_id, "test", tuple(fold.test_years)))
    return triples


def daily_complete_widths(
    holder: dict[str, Any], years: Sequence[int], factor_count: int
) -> dict[int, list[int]]:
    """Per-day count of stocks having all ``factor_count`` frozen factors."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from walk_forward_common import TARGET_DATABASE, query

    table = f"loadTable('{TARGET_DATABASE}','factor_daily')"
    widths: dict[int, list[int]] = {}
    for year in years:
        script = (
            f"p = select signal_date, ts_code, count(*) as c from {table} "
            f"where signal_date >= {year}.01.01 and signal_date < {year + 1}.01.01 "
            f"group by signal_date, ts_code\n"
            f"select signal_date, sum(iif(c=={factor_count},1,0)) as complete "
            f"from p group by signal_date"
        )
        ok, frame = query(holder, script)
        if not ok or frame is None or not len(frame):
            widths[year] = []
            continue
        widths[year] = [int(v) for v in frame["complete"]]
    return widths


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--minimum-cross-section", type=int, default=DEFAULT_MINIMUM_CROSS_SECTION)
    parser.add_argument("--min-valid-day-share", type=float, default=DEFAULT_MIN_VALID_DAY_SHARE)
    parser.add_argument(
        "--max-days-below-floor", type=int, default=DEFAULT_MAX_DAYS_BELOW_FLOOR,
        help="tolerated days under the cross-section floor for an enforced role "
             "(default 1: the accepted criterion-303 baseline loses exactly one, "
             "2017-10-09, to a single missing factor cell; frozen-17 loses 15/15/4)",
    )
    parser.add_argument(
        "--minimum-median-width", type=int, default=DEFAULT_MINIMUM_MEDIAN_WIDTH,
        help="calibrated floor on the median complete-factor cross-section of a "
             "decision role (train/validation); 0 disables the criterion",
    )
    parser.add_argument(
        "--roles", default="train,validation,test",
        help="comma-separated roles to enforce the gate on (others are reported only)",
    )
    args = parser.parse_args(argv)
    enforced = {role.strip() for role in args.roles.split(",") if role.strip()}

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from walk_forward_common import new_holder
    from walk_forward_factor_daily import FROZEN_FACTOR_SET_V1

    factor_count = len(FROZEN_FACTOR_SET_V1)
    holder = new_holder()
    print(f"frozen factor count: {factor_count}; cross-section floor: "
          f"{args.minimum_cross_section}; enforced roles: {sorted(enforced)}; "
          f"max days below floor: {args.max_days_below_floor}; "
          f"minimum median width ({'/'.join(DECISION_ROLES)}): "
          f"{args.minimum_median_width or 'disabled'}")
    print(f"{'fold':6} {'role':11} {'years':22} {'days':>5} {'<floor':>7} "
          f"{'valid share':>12} {'median':>7} {'min':>6}  verdict")

    failed = False
    assessments: dict[tuple[str, str], CoverageAssessment] = {}
    for fold_id, role, years in fold_role_years():
        widths = daily_complete_widths(holder, years, factor_count)
        assessment = assess_coverage(
            fold_id, role, years, widths,
            minimum_cross_section=args.minimum_cross_section,
            min_valid_day_share=args.min_valid_day_share,
            max_days_below_floor=args.max_days_below_floor,
            minimum_median_width=args.minimum_median_width,
            enforce=role in enforced,
        )
        assessments[(fold_id, role)] = assessment
        verdict = "PASS" if assessment.passed else "FAIL"
        note = "" if role in enforced else "  (reported only)"
        print(f"{fold_id:6} {role:11} {str(years[0]) + '-' + str(years[-1]):22} "
              f"{assessment.days:>5} {assessment.days_below_floor:>7} "
              f"{assessment.valid_day_share:>11.3%} {assessment.median_width:>7} "
              f"{assessment.min_width:>6}  {verdict}{note}")
        for failure in assessment.failures:
            print(f"       FAIL: {failure}")
        for extra in assessment.notes:
            print(f"       NOTE: {extra}")
        failed |= not assessment.passed

    print("\ntrain/test cross-section asymmetry (reported only -- see DECISION_ROLES "
          "comment: the ratio does not separate frozen-17 from the accepted baseline):")
    for fold_id in sorted({key[0] for key in assessments}):
        train = assessments.get((fold_id, "train"))
        test = assessments.get((fold_id, "test"))
        if train is not None and test is not None:
            asymmetry = asymmetry_note(train, test)
            if asymmetry:
                print(f"  {asymmetry}")
    print("COVERAGE GATE:", "FAIL" if failed else "PASS")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
