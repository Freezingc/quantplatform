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
  verify_frozen_fold_coverage.py --factor-set c62pass11  # measure a candidate subset
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
# The eleven frozen factors that pass the C6.2 coverage screen on factor_daily, i.e.
# adjudication 4's option (iii). Provenance: coverage_screen_factor_daily_2014_2019.txt
# and coverage_screen_factor_daily_2020_2024.txt, which independently judged the same
# six factors ELIGIBILITY_FAIL (median 0.37-0.50 against the 0.70 gate) on two windows.
# This is a MEASUREMENT input, not a wiring: nothing trains on it, and the frozen set
# consumed by the walk-forward remains FROZEN_FACTOR_SET_V1 until adjudication 4 lands.
C62_PASS_11: tuple[str, ...] = (
    "yeslab_96",
    "yeslab_206",
    "yeslab_357",
    "yeslab_381",
    "yeslab_398",
    "yeslab_405",
    "yeslab_425",
    "yeslab_443_reg30_max20_v1",
    "yeslab_464",
    "yeslab_471",
    "yeslab_484",
)
# ``None`` means "whatever FROZEN_FACTOR_SET_V1 currently is", so the default door
# cannot drift from the wiring it is gating.
FACTOR_SETS: dict[str, tuple[str, ...] | None] = {
    "frozen17": None,
    "c62pass11": C62_PASS_11,
}


def resolve_factor_set(name: str, frozen_set: Sequence[str]) -> tuple[str, ...]:
    """Resolve a named measurement set, failing closed if it is not inside the frozen set.

    The frozen set is the authority. A subset that has drifted out of it -- because the
    frozen set changed, or because this constant went stale -- must not be measured
    silently, or the resulting widths would describe a factor set nothing can train on.
    """
    if name not in FACTOR_SETS:
        raise ValueError(f"unknown factor set {name!r}; choose from {sorted(FACTOR_SETS)}")
    preset = FACTOR_SETS[name]
    if preset is None:
        return tuple(frozen_set)
    if unknown := sorted(set(preset) - set(frozen_set)):
        raise ValueError(
            f"{name} is not a subset of the frozen set; unknown ids: {unknown}"
        )
    return preset


def build_width_query(
    year: int,
    factor_ids: Sequence[str],
    table: str,
    minimum_factors: int | None = None,
) -> str:
    """Per-day count of stocks having every named factor, for one calendar year.

    ``count(*)`` rather than ``count(distinct factor_id)`` because DolphinDB rejects a
    nested aggregate inside a grouped select (RefId:S02032). The two are equivalent only
    when the table has no duplicate ``(factor, date, code)`` rows, which is verified
    rather than assumed: zero duplicate groups in 2020-01, and the C6.2 screen reports
    ``dups==0`` for all 17 factors over both the 2014-2019 and 2020-2024 windows.

    ``minimum_factors`` switches the completeness predicate from "every named factor"
    to "at least this many of them", which is what adjudication 4's option (i) needs:
    B1 already tolerates partial coverage by design (``minimum_factor_coverage=0.60``
    means >=11 of 17, and it omits missing factors rather than imputing them), so under
    that option the trained-on cross-section is pairs with at least 11 of the 17, not
    pairs with all 17 and not pairs with all of some 11-factor subset. The query stays
    restricted to the named set either way, or "at least 11" would count other factors.
    """
    if not factor_ids:
        raise ValueError("factor_ids cannot be empty")
    required = len(factor_ids) if minimum_factors is None else minimum_factors
    if required < 1:
        raise ValueError("minimum_factors must be at least 1")
    if required > len(factor_ids):
        raise ValueError(
            f"minimum_factors {required} cannot exceed the {len(factor_ids)} named factors"
        )
    predicate = f"c=={required}" if minimum_factors is None else f"c>={required}"
    listed = ",".join(f"`{factor_id}" for factor_id in factor_ids)
    return (
        f"p = select signal_date, ts_code, count(*) as c from {table} "
        f"where signal_date >= {year}.01.01 and signal_date < {year + 1}.01.01 "
        f"and factor_id in [{listed}] group by signal_date, ts_code\n"
        f"select signal_date, sum(iif({predicate},1,0)) as complete "
        f"from p group by signal_date"
    )


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
    holder: dict[str, Any],
    years: Sequence[int],
    factor_ids: Sequence[str],
    minimum_factors: int | None = None,
) -> dict[int, list[int]]:
    """Per-day count of stocks having every (or at least ``minimum_factors``) factor."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from walk_forward_common import TARGET_DATABASE, query

    table = f"loadTable('{TARGET_DATABASE}','factor_daily')"
    widths: dict[int, list[int]] = {}
    for year in years:
        script = build_width_query(year, factor_ids, table, minimum_factors)
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
    parser.add_argument(
        "--factor-set", default="frozen17", choices=sorted(FACTOR_SETS),
        help="which factor set to measure. frozen17 is the wiring the pre-flight door "
             "gates; c62pass11 is the C6.2-screened subset, for measuring adjudication "
             "4's option (iii) before choosing it. Measurement only, nothing is rewired.",
    )
    parser.add_argument(
        "--minimum-factors", type=int, default=None,
        help="require at least this many of the named factors instead of all of them "
             "(adjudication 4's option (i): B1's own >=11-of-17 partial-coverage rule)",
    )
    args = parser.parse_args(argv)
    enforced = {role.strip() for role in args.roles.split(",") if role.strip()}

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from walk_forward_common import new_holder
    from walk_forward_factor_daily import FROZEN_FACTOR_SET_V1

    factor_ids = resolve_factor_set(args.factor_set, FROZEN_FACTOR_SET_V1)
    holder = new_holder()
    completeness = (
        f"all {len(factor_ids)}"
        if args.minimum_factors is None
        else f">={args.minimum_factors} of {len(factor_ids)}"
    )
    print(f"factor set: {args.factor_set} ({len(factor_ids)} factors); cross-section floor: "
          f"{args.minimum_cross_section}; enforced roles: {sorted(enforced)}; "
          f"max days below floor: {args.max_days_below_floor}; "
          f"minimum median width ({'/'.join(DECISION_ROLES)}): "
          f"{args.minimum_median_width or 'disabled'}; completeness: {completeness}")
    print(f"{'fold':6} {'role':11} {'years':22} {'days':>5} {'<floor':>7} "
          f"{'valid share':>12} {'median':>7} {'min':>6}  verdict")

    failed = False
    assessments: dict[tuple[str, str], CoverageAssessment] = {}
    for fold_id, role, years in fold_role_years():
        widths = daily_complete_widths(holder, years, factor_ids, args.minimum_factors)
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
