"""Calibrate a train-side coverage gate against the already-accepted baseline.

Why this exists
---------------
G1-G6 are computed exclusively from TEST-period evidence
(``evaluate_frozen_oos`` raises ``PermissionError`` for any non-TEST role, which
is the leakage firewall and must stay). Consequently nothing in the promotion
contract measures the cross-section the model was *fitted* on. Measured
2026-09-05, the frozen-17 set is fitted on a median cross-section of 310-673
stocks while being scored on 599-908 -- a ratio of 0.49-0.77 -- and loses 15,
15 and 4 train days entirely to the ``minimum_cross_section`` floor of 30 on
2015's crash and 2016's circuit-breaker days. All six TEST years still pass G6
at 100% valid days, so a frozen baseline would clear every existing gate and
become the new trust root while carrying that defect. A gate that cannot see a
defect is worse than a hard failure.

``verify_frozen_fold_coverage.py`` measures the frozen side and reports the
train/test asymmetry, but deliberately does not gate it: no threshold had been
measured. This script measures the missing half -- the legacy-14 frames that
produced the accepted criterion-303 trust root -- so a threshold can be set with
evidence on both sides.

Source of truth
---------------
It reads the assembled-frame parquet cache (``FOLD_CACHE_ROOT``) rather than
re-querying the database. That is not a convenience: those files are the exact
frames the accepted baseline consumed, so the widths measured here are the widths
the baseline actually saw. Cache keys are recomputed from the currently published
versions, and a key that resolves to no file fails closed -- the legacy reference
must be complete, otherwise the calibration is drawn from a different dataset
than the one being calibrated against.

Usage:
  audit_legacy_train_coverage.py                      # legacy reference only
  audit_legacy_train_coverage.py --sources legacy,frozen
  audit_legacy_train_coverage.py --json /path/out.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from research_v2.datasets.coverage import daily_complete_widths, summarize_widths
from research_v2.datasets.coverage import factor_columns as _library_factor_columns

ROLES: tuple[str, ...] = ("train", "validation", "test")
# Same floor the evaluation path uses (``evaluate_frozen_oos(minimum_cross_section=30)``),
# so a day counted as lost here is a day the baseline genuinely could not be fitted on.
DEFAULT_MINIMUM_CROSS_SECTION = 30
REFERENCE_SOURCE = "legacy"


@dataclass(frozen=True)
class SegmentCoverage:
    """Per-day complete-cross-section width summary for one (fold, role) segment."""

    fold_id: str
    role: str
    days: int
    days_below_floor: int
    median_width: int
    min_width: int


@dataclass(frozen=True)
class CalibrationBounds:
    """Worst-case legacy numbers; a G7 threshold must admit all of them.

    Every field is a worst case across folds, never a mean. Calibrating on an
    average would let the weakest accepted fold fail the new gate, which would
    retroactively invalidate the trust root instead of protecting future ones.
    """

    min_train_median_width: int
    worst_train_day_width: int
    max_train_days_below_floor: int
    min_train_test_ratio: float


@dataclass(frozen=True)
class FrameMeasurement:
    """One cached frame's width summary plus the identity checks it must satisfy."""

    coverage: SegmentCoverage
    factor_count: int
    observed_years: tuple[int, ...]
    widths: tuple[int, ...]


SegmentMap = dict[tuple[str, str], SegmentCoverage]
MeasurementMap = dict[tuple[str, str], FrameMeasurement]


def gate_from_widths(
    assess: Any,
    widths: Sequence[int],
    *,
    fold_id: str,
    role: str,
    years: Sequence[int],
    **thresholds: Any,
) -> Any:
    """Run the frozen coverage gate's own verdict function over measured widths.

    ``assess_coverage`` flattens ``daily_widths.values()`` and never reads the day
    keys, so a measured width list can be handed to it directly. This is what makes
    the calibration checkable instead of asserted: the same gate that rejects the
    frozen set must admit every segment of the baseline already accepted, and this
    runs that claim on the real frames rather than on a synthetic fixture.
    """
    return assess(fold_id, role, years, {0: [int(width) for width in widths]}, **thresholds)


def factor_columns(columns: Sequence[str]) -> tuple[str, ...]:
    """The ``fNN`` feature columns the runner passes to ``dropna(subset=...)``."""
    return _library_factor_columns(columns)


def daily_widths_after_dropna(frame: pd.DataFrame, factors: Sequence[str]) -> dict[Any, int]:
    """Per-day surviving row count, applying exactly the runner's ``dropna``.

    Delegates to :mod:`research_v2.datasets.coverage`, which owns this measurement so
    the runner, the coverage gate and this audit cannot drift apart. Kept as a named
    wrapper because the property it guarantees is the one that matters here: a day
    whose every row is dropped is reported as width 0 rather than omitted, since
    grouping the post-dropna frame alone would erase it from the calendar -- which is
    precisely how the frozen train segment lost 2015-08-24 (2 stocks of 1,466) and the
    two 2016 circuit-breaker days without any gate noticing.
    """
    return daily_complete_widths(frame, factors)


def years_consistent(
    observed_years: Sequence[int], expected_years: Sequence[int], role: str
) -> bool:
    """Cross-check a resolved cache file against its fold definition.

    Defense in depth: the cache key already identifies the segment, so this
    catches a key collision or a stale cache written by a different split. Only
    the test role may extend past its years, because test signals legitimately
    exit into the following January (``extract_dataset`` sets ``segment_end`` to
    ``max(test_years)+1`` January for that role alone).
    """
    observed = {int(year) for year in observed_years}
    expected = {int(year) for year in expected_years}
    if not expected:
        return False
    if not expected <= observed:
        return False
    extra = observed - expected
    if not extra:
        return True
    return role == "test" and extra <= {max(expected) + 1}


def cache_key_index(cache_root: Path) -> dict[str, Path]:
    """Map each cached frame's key (its file stem) to its path."""
    return {path.stem: path for path in sorted(cache_root.glob("*.parquet"))}


def resolve_cache_paths(
    wanted: Sequence[tuple[str, str, str]], index: Mapping[str, Path]
) -> dict[tuple[str, str], Path]:
    """Resolve ``(fold, role, cache_key)`` triples to paths, failing closed."""
    resolved: dict[tuple[str, str], Path] = {}
    missing: list[str] = []
    for fold_id, role, key in wanted:
        path = index.get(key)
        if path is None:
            missing.append(f"{fold_id}/{role}")
            continue
        resolved[(fold_id, role)] = path
    if missing:
        raise ValueError(
            f"no cached frame for {', '.join(missing)}; the reference chain must be "
            "complete before it can calibrate a gate (re-run the walk-forward to "
            "populate the cache, or drop the source)"
        )
    return resolved


def partition_cache_paths(
    wanted: Sequence[tuple[str, str, str]], index: Mapping[str, Path]
) -> tuple[dict[tuple[str, str], Path], tuple[str, ...]]:
    """Like :func:`resolve_cache_paths` but returns the missing segments instead of raising."""
    resolved: dict[tuple[str, str], Path] = {}
    missing: list[str] = []
    for fold_id, role, key in wanted:
        path = index.get(key)
        if path is None:
            missing.append(f"{fold_id}/{role}")
        else:
            resolved[(fold_id, role)] = path
    return resolved, tuple(missing)


def calibration_bounds(assessments: Sequence[SegmentCoverage]) -> CalibrationBounds:
    """Derive the worst-case legacy numbers a G7 threshold has to admit."""
    train = {item.fold_id: item for item in assessments if item.role == "train"}
    test = {item.fold_id: item for item in assessments if item.role == "test"}
    unmatched = sorted(set(train) ^ set(test))
    if not train or unmatched:
        raise ValueError(
            "calibration needs a matched train/test pair per fold to compute the "
            f"asymmetry ratio; unmatched folds: {unmatched or 'none resolved'}"
        )
    ratios = [
        train[fold].median_width / test[fold].median_width
        for fold in sorted(train)
        if test[fold].median_width > 0
    ]
    if not ratios:
        raise ValueError("every test segment has a zero median width; cannot calibrate a ratio")
    return CalibrationBounds(
        min_train_median_width=min(item.median_width for item in train.values()),
        worst_train_day_width=min(item.min_width for item in train.values()),
        max_train_days_below_floor=max(item.days_below_floor for item in train.values()),
        min_train_test_ratio=min(ratios),
    )


def _segments_for_source(source: str) -> list[tuple[str, str, tuple[int, ...]]]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from research_v2.datasets.splits import annual_folds

    segments: list[tuple[str, str, tuple[int, ...]]] = []
    for fold in annual_folds():
        if fold.locked:
            continue  # 2026 FINAL is never touched
        for role in ROLES:
            years = getattr(fold, f"{role}_years")
            segments.append((fold.fold_id, role, tuple(years)))
    if not segments:
        raise ValueError(f"no unlocked folds for source {source!r}")
    return segments


def _wanted_keys(holder: dict[str, Any], source: str) -> list[tuple[str, str, str]]:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from walk_forward_common import current_versions, factor_members_for
    from walk_forward_factor_daily import factor_source_plan, fold_cache_key_for

    factor_ids, version_tables = factor_source_plan(source)
    versions = {table: current_versions(holder, table) for table in version_tables}
    # Ask the source for its own key function. The frozen extractor scopes its cache key
    # by the requested factor set; using the legacy key function here silently resolves
    # to keys that were never written, which reads exactly like an empty cache.
    key_fn = fold_cache_key_for(source)
    members = factor_members_for(factor_ids)
    return [
        (fold_id, role, key_fn(fold_id, role, members, versions))
        for fold_id, role, _ in _segments_for_source(source)
    ]


def measure_frame(path: Path, minimum_cross_section: int) -> FrameMeasurement:
    """Measure one cached frame, reading only the columns the dropna needs.

    ``signal_date`` is read once and reused for both the width histogram and the
    year-span cross-check, so a 423 MB train frame is never materialised twice.
    """
    import pyarrow.parquet as pq

    available = pq.ParquetFile(path).schema_arrow.names
    factors = factor_columns(available)
    frame = pd.read_parquet(path, columns=["signal_date", *factors])
    try:
        dates = pd.DatetimeIndex(frame["signal_date"])
        observed_years = tuple(sorted(set(int(year) for year in dates.year)))
        widths = daily_widths_after_dropna(frame, factors)
        rows = int(len(frame))
    finally:
        del frame
    summary = summarize_widths(
        widths, rows=rows, minimum_cross_section=minimum_cross_section
    )
    coverage = SegmentCoverage(
        fold_id="",
        role="",
        days=summary.days,
        days_below_floor=summary.days_below_floor,
        median_width=summary.median_width,
        min_width=summary.min_width,
    )
    return FrameMeasurement(
        coverage,
        len(factors),
        observed_years,
        tuple(int(widths[day]) for day in sorted(widths, key=str)),
    )


def measure_source(
    holder: dict[str, Any],
    source: str,
    cache_root: Path,
    *,
    minimum_cross_section: int,
    require_complete: bool,
) -> tuple[SegmentMap, tuple[str, ...], int, MeasurementMap]:
    """Measure every cached segment of one factor source."""
    from dataclasses import replace

    wanted = _wanted_keys(holder, source)
    index = cache_key_index(cache_root)
    if require_complete:
        resolved = resolve_cache_paths(wanted, index)
        missing: tuple[str, ...] = ()
    else:
        resolved, missing = partition_cache_paths(wanted, index)
    years_by_segment = {(fold, role): years for fold, role, years in _segments_for_source(source)}

    results: SegmentMap = {}
    measurements: MeasurementMap = {}
    factor_count = 0
    for (fold_id, role), path in sorted(resolved.items()):
        measurement = measure_frame(path, minimum_cross_section)
        factor_count = measurement.factor_count
        expected_years = years_by_segment[(fold_id, role)]
        if not years_consistent(measurement.observed_years, expected_years, role):
            raise ValueError(
                f"cached frame for {source} {fold_id}/{role} spans "
                f"{list(measurement.observed_years)}, expected {list(expected_years)}; "
                "the cache is stale or the key collided"
            )
        results[(fold_id, role)] = replace(
            measurement.coverage, fold_id=fold_id, role=role
        )
        measurements[(fold_id, role)] = measurement
    return results, missing, factor_count, measurements


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", default=REFERENCE_SOURCE,
                        help="comma-separated factor sources to measure (legacy,frozen)")
    parser.add_argument("--minimum-cross-section", type=int,
                        default=DEFAULT_MINIMUM_CROSS_SECTION)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--json", type=Path, default=None, help="also write the measurements here")
    parser.add_argument(
        "--apply-coverage-gate", action="store_true",
        help="run verify_frozen_fold_coverage's verdict function over the measured "
             "widths, to check the calibrated gate admits what it is calibrated on",
    )
    args = parser.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from walk_forward_common import FOLD_CACHE_ROOT, new_holder

    cache_root = args.cache_root if args.cache_root is not None else FOLD_CACHE_ROOT
    sources = [item.strip() for item in args.sources.split(",") if item.strip()]
    holder = new_holder()
    payload: dict[str, Any] = {"minimum_cross_section": args.minimum_cross_section}
    for source in sources:
        # The reference chain must be complete: a calibration drawn from a subset
        # of folds would be drawn from a different dataset than the one being
        # calibrated against. Every other source is best-effort, because the frozen
        # chain has only been assembled for the folds proven so far.
        require_complete = source == REFERENCE_SOURCE
        results, missing, factor_count, measurements = measure_source(
            holder, source, cache_root,
            minimum_cross_section=args.minimum_cross_section,
            require_complete=require_complete,
        )
        print(f"\n=== {source} ({factor_count} factors, floor "
              f"{args.minimum_cross_section}) ===")
        print(f"{'fold':6} {'role':11} {'days':>6} {'<floor':>7} {'median':>8} {'min':>7}")
        ordered = sorted(results.values(), key=lambda entry: (entry.fold_id, entry.role))
        for item in ordered:
            print(f"{item.fold_id:6} {item.role:11} {item.days:>6} "
                  f"{item.days_below_floor:>7} {item.median_width:>8} {item.min_width:>7}")
        if missing:
            print(f"missing segments (not measured): {', '.join(missing)}")
        ratios = {
            fold: results[(fold, "train")].median_width / results[(fold, "test")].median_width
            for fold in sorted({key[0] for key in results})
            if (fold, "train") in results and (fold, "test") in results
            and results[(fold, "test")].median_width > 0
        }
        for fold, ratio in ratios.items():
            print(f"  {fold}: train/test median cross-section ratio {ratio:.2f}")
        segments = {f"{item.fold_id}/{item.role}": asdict(item) for item in ordered}
        payload[source] = {
            "factor_count": factor_count,
            "segments": segments,
            "missing": list(missing),
            "train_test_ratio": {fold: round(ratio, 4) for fold, ratio in ratios.items()},
        }
        if args.apply_coverage_gate:
            payload[f"{source}_coverage_gate"] = apply_coverage_gate(
                source, measurements, minimum_cross_section=args.minimum_cross_section
            )
        try:
            bounds = calibration_bounds(tuple(results.values()))
        except ValueError as error:
            print(f"  calibration: unavailable ({error})")
            continue
        print("  calibration bounds (worst case across folds):")
        print(f"    min train median cross-section : {bounds.min_train_median_width}")
        print(f"    worst single train day width   : {bounds.worst_train_day_width}")
        print(f"    max train days below floor     : {bounds.max_train_days_below_floor}")
        print(f"    min train/test median ratio    : {bounds.min_train_test_ratio:.3f}")
        payload[f"{source}_calibration"] = asdict(bounds)

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


def apply_coverage_gate(
    source: str,
    measurements: MeasurementMap,
    *,
    minimum_cross_section: int,
) -> dict[str, Any]:
    """Check the calibrated gate against the widths it was calibrated on.

    A threshold chosen only from the defective side is not a gate, it is a verdict
    written in advance. This runs ``verify_frozen_fold_coverage.assess_coverage`` --
    the same function that rejects the frozen set -- over the measured widths of the
    accepted baseline, and reports per segment whether it is admitted. If the
    reference chain is not fully admitted, the calibration constants are wrong and
    must be widened before the gate is used to block anything.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from verify_frozen_fold_coverage import (
        DEFAULT_MAX_DAYS_BELOW_FLOOR,
        DEFAULT_MINIMUM_MEDIAN_WIDTH,
        assess_coverage,
    )

    years_by_segment = {
        (fold_id, role): years for fold_id, role, years in _segments_for_source(source)
    }
    print("  coverage-gate verdict on the measured widths (same function, same constants):")
    verdicts: dict[str, Any] = {}
    admitted = 0
    for (fold_id, role), measurement in sorted(measurements.items()):
        assessment = gate_from_widths(
            assess_coverage,
            measurement.widths,
            fold_id=fold_id,
            role=role,
            years=years_by_segment[(fold_id, role)],
            minimum_cross_section=minimum_cross_section,
            max_days_below_floor=DEFAULT_MAX_DAYS_BELOW_FLOOR,
            minimum_median_width=DEFAULT_MINIMUM_MEDIAN_WIDTH,
        )
        admitted += bool(assessment.passed)
        verdicts[f"{fold_id}/{role}"] = {
            "passed": bool(assessment.passed),
            "failures": list(assessment.failures),
        }
        print(f"    {fold_id}/{role:11} {'PASS' if assessment.passed else 'FAIL'}")
        for failure in assessment.failures:
            print(f"       FAIL: {failure}")
    total = len(measurements)
    print(f"  gate admits {admitted}/{total} measured {source} segments")
    if admitted != total:
        print(
            "  CALIBRATION INVALID: the gate rejects part of the reference chain it "
            "was calibrated on; widen the constants before using it to block a run"
        )
    verdicts["admitted"] = admitted
    verdicts["total"] = total
    verdicts["calibration_valid"] = admitted == total
    return verdicts


if __name__ == "__main__":
    raise SystemExit(main())
