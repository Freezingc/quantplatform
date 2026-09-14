#!/usr/bin/env python3
"""Predict the unevaluable OOS day set from published tables, before a run spends nine days on it.

WHY THIS EXISTS

The score-health floor at ``src/research_v2/evaluation/gates.py:35`` is applied **per day** to the
coverage of the candidate∩baseline key set over that day's valid-label rows. The frozen best_simple
baseline is defined exactly on the rows where all nine dense factors are finite, so for a candidate
that scores every row, the day's coverage is the dense **intersection** divided by the valid-label
rows. Every coverage instrument in use measures **one factor at a time** -- median and p05, at
0.70/0.30 (live screen) or 0.80/0.70 (persisted snapshot) -- so none of them can see this:

    yeslab_357  median 0.9992  p05 0.9131  -> PASS on both instruments
    yeslab_471  median 0.9936  p05 0.9091  -> PASS on both instruments
    panel-measured 2023 intersection coverage, median  0.9093  -> 23 of the year's days within
                                                                  0.005 of the floor, 23 dropped

471's p05 and the intersection's median differ by 0.0002 because 357's missing set nests inside
471's, which makes the intersection equal to 471 alone. A per-factor quantile gate cannot constrain
an intersection floor. This screen measures the quantity the floor actually constrains.

WHAT IT MEASURES

Numerator: per day, the number of securities having **all** of the requested factors present in
``factor_daily`` at the published ``data_version``. Denominator: that day's ``label_valid=1`` rows
in ``label_daily``, also at the published version. That denominator is the evaluator's, not a proxy
-- on 2023-07-05 it is 3,627, which is the panel's valid-label row count for that day exactly.

The floor is **parsed out of the evaluator's own source** at run time rather than retyped here, and
a unit test fails if the two diverge. A screen that predicts a gate must read that gate.

STATUS: measurement, not policy. Nothing calls this script; it changes no threshold and gates
nothing. ``--fail-on-prediction`` exists so an owner can wire it into a preflight later without
another code change, and it is off by default because deciding what to do about a predicted day set
(re-publish a complete baseline, move the floor, or disclose the exclusion) is a governance decision
that changes signed reports.

SAFETY: SELECT-only against dfs://QuantResearchV2Test. No writes, no weights, no artifact mutation.

Usage:
    screen_dense_intersection_coverage.py --start 2022-01-01 --end 2025-12-31
    screen_dense_intersection_coverage.py --start 2022-01-01 --end 2025-12-31 \
        --compare /path/to/yearly_baseline_deltas/m0r_v2.json --json out.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, cast

RESEARCH_V2 = Path(__file__).resolve().parents[1]
GATES_PATH = RESEARCH_V2 / "src" / "research_v2" / "evaluation" / "gates.py"

# The evaluator's own line, as of this writing:
#     return 0.90 <= coverage <= 1.0 and self.unique_score_count >= 10 and score_std > 1e-6
_FLOOR_PATTERN = re.compile(r"([0-9]*\.?[0-9]+)\s*<=\s*coverage\s*<=\s*1\.0")
DEFAULT_FLOOR = 0.90
# Factor ids reach the database inside a quoted IN-list. The frozen set already contains an id with
# an underscore and digits (yeslab_443_reg30_max20_v1), so the character class has to allow both;
# what it must not allow is anything that could close the quote.
_FACTOR_ID_PATTERN = re.compile(r"^[A-Za-z0-9_]+$")


def parse_factor_list(raw: str) -> list[str]:
    """Validate an explicit factor list, refusing the shapes that would fake a clean result.

    A repeated id makes ``having count(*) = k`` unreachable, so the intersection query returns no
    days at all and the screen reports nothing below the floor -- indistinguishable from a factor
    set that covers everything. An id containing a quote would break out of the IN-list. Both are
    refused here rather than discovered as a mysteriously perfect report.
    """
    ids = [item.strip() for item in raw.split(",")]
    ids = [item for item in ids if item]
    if not ids:
        raise ValueError("--factors needs at least one factor id")
    invalid = [item for item in ids if not _FACTOR_ID_PATTERN.fullmatch(item)]
    if invalid:
        raise ValueError(f"--factors contains ids with unsupported characters: {invalid[:4]}")
    duplicates = sorted({item for item in ids if ids.count(item) > 1})
    if duplicates:
        raise ValueError(
            f"--factors contains duplicates: {duplicates[:4]}; a repeated id would make the "
            "intersection clause unsatisfiable and the screen would report a false all-clear"
        )
    return ids


class FloorNotFoundError(RuntimeError):
    """The evaluator's score-health floor could not be read out of its source."""


class UnpublishedPartitionError(RuntimeError):
    """The window touches a logical partition that has no published ``data_version``.

    Rows under an unpublished version cannot reach any run, and a day that cannot be measured must
    not be silently dropped from the report: the whole point of this screen is to name the days that
    will be missing, so quietly omitting one would defeat it.
    """


def parse_floor_from_gates(source_text: str) -> float:
    """Read the score-health floor out of the evaluator's source instead of restating it."""
    match = _FLOOR_PATTERN.search(source_text)
    if match is None:
        raise FloorNotFoundError(
            "could not find the score-health coverage floor in the evaluator source; expected a "
            "line of the form '<floor> <= coverage <= 1.0'. The floor must be read from the gate "
            "this screen predicts, never retyped here."
        )
    return float(match.group(1))


def current_floor(gates_path: Path = GATES_PATH) -> float:
    return parse_floor_from_gates(gates_path.read_text(encoding="utf-8"))


@dataclass(frozen=True)
class DayCoverage:
    """One day's dense-intersection coverage, on the evaluator's own denominator."""

    day: date
    intersection_rows: int
    valid_label_rows: int
    coverage: float
    below_floor: bool


def daily_coverage(
    intersection_per_day: Mapping[date, int],
    valid_label_per_day: Mapping[date, int],
    *,
    floor: float,
) -> tuple[DayCoverage, ...]:
    """Per-day coverage. A day with no valid-label rows has no verdict and is not counted."""
    entries: list[DayCoverage] = []
    for day in sorted(valid_label_per_day):
        label_rows = int(valid_label_per_day[day])
        if label_rows <= 0:
            continue
        intersection_rows = int(intersection_per_day.get(day, 0))
        coverage = intersection_rows / label_rows
        entries.append(
            DayCoverage(
                day=day,
                intersection_rows=intersection_rows,
                valid_label_rows=label_rows,
                coverage=coverage,
                below_floor=coverage < floor,
            )
        )
    return tuple(entries)


def _median(values: Sequence[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return float((ordered[middle - 1] + ordered[middle]) / 2)


def summarize(measured: Iterable[DayCoverage], *, floor: float) -> dict[str, Any]:
    """Totals, per-year breakdown and the day list, in the shape the reports already use."""
    entries = list(measured)
    below = [entry for entry in entries if entry.below_floor]
    by_year: dict[str, dict[str, Any]] = {}
    for entry in entries:
        bucket = by_year.setdefault(
            f"{entry.day.year}",
            {"days": 0, "days_below_floor": 0, "coverages": [], "days_below_floor_list": []},
        )
        bucket["days"] += 1
        bucket["coverages"].append(entry.coverage)
        if entry.below_floor:
            bucket["days_below_floor"] += 1
            bucket["days_below_floor_list"].append(entry.day.isoformat())
    for bucket in by_year.values():
        coverages = cast(list[float], bucket.pop("coverages"))
        bucket["coverage_min"] = min(coverages) if coverages else None
        bucket["coverage_median"] = _median(coverages)
        bucket["coverage_max"] = max(coverages) if coverages else None
        bucket["days_below_floor_list"] = sorted(bucket["days_below_floor_list"])
    return {
        "floor": floor,
        "total_days": len(entries),
        "days_below_floor": len(below),
        "days_below_floor_list": sorted(entry.day.isoformat() for entry in below),
        "coverage_min": min((entry.coverage for entry in entries), default=None),
        "coverage_median": _median([entry.coverage for entry in entries]),
        "by_year": {year: by_year[year] for year in sorted(by_year)},
    }


def compare_with_known(
    predicted: Iterable[Any], known: Iterable[Any]
) -> dict[str, Any]:
    """Set-difference a predicted day list against a day list already measured from a real run."""
    predicted_days = sorted(_iso(value) for value in predicted)
    known_days = sorted(_iso(value) for value in known)
    matched = sorted(set(predicted_days) & set(known_days))
    return {
        "predicted_days": len(predicted_days),
        "known_days": len(known_days),
        "matched": matched,
        "only_predicted": sorted(set(predicted_days) - set(known_days)),
        "only_known": sorted(set(known_days) - set(predicted_days)),
        "identical": predicted_days == known_days,
    }


def _iso(value: Any) -> str:
    if isinstance(value, date):
        return value.isoformat()
    return str(value)[:10]


def known_days_from_delta_report(payload: Mapping[str, Any]) -> list[str]:
    """Pull the measured unevaluable day list out of an analyze_yearly_baseline_deltas JSON."""
    unevaluable = payload.get("unevaluable_days")
    if not isinstance(unevaluable, Mapping):
        raise ValueError("the report has no unevaluable_days section; is it a delta report?")
    by_year = unevaluable.get("by_year")
    if not isinstance(by_year, Mapping):
        raise ValueError("unevaluable_days has no by_year section")
    days: list[str] = []
    for entry in by_year.values():
        if isinstance(entry, Mapping):
            days.extend(_iso(value) for value in entry.get("invalid_dates", ()))
    return sorted(days)


def _partitions_in_window(start: date, end: date) -> list[str]:
    partitions: list[str] = []
    year, month = start.year, start.month
    while (year, month) <= (end.year, end.month):
        partitions.append(f"{year}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return partitions


def _require_pins(
    pins: Mapping[str, str], partitions: Sequence[str], object_name: str
) -> list[str]:
    missing = [partition for partition in partitions if partition not in pins]
    if missing:
        raise UnpublishedPartitionError(
            f"{object_name} has no published data_version for {len(missing)} partition(s) in the "
            f"window: {missing[:6]}. Those days cannot be measured, and skipping them would hide "
            "exactly the kind of gap this screen exists to name. Publish the versions or narrow "
            "the window."
        )
    return sorted({pins[partition] for partition in partitions})


def _sql_list(values: Sequence[str]) -> str:
    """Render an IN-list, escaping the single quote the way the other screens do."""
    return ", ".join("'" + str(value).replace("'", "''") + "'" for value in values)


def collect(
    holder: dict[str, Any], start: date, end: date, factor_ids: Sequence[str]
) -> tuple[dict[date, int], dict[date, int], dict[str, Any]]:
    """Two aggregate queries: the dense intersection per day, and the denominator per day.

    Both are resolved to the published ``data_version``, because both tables keep every version they
    were published with and an unfiltered count measures the union of all of them.
    """
    sys.path.insert(0, str(RESEARCH_V2 / "scripts"))
    sys.path.insert(0, str(RESEARCH_V2 / "src"))
    import pandas as pd
    from walk_forward_common import TARGET_DATABASE, current_versions, query

    if not factor_ids:
        raise ValueError("at least one factor id is required")
    partitions = _partitions_in_window(start, end)
    factor_pins = current_versions(holder, "factor_daily")
    label_pins = current_versions(holder, "label_daily")
    factor_versions = _require_pins(factor_pins, partitions, "factor_daily")
    label_versions = _require_pins(label_pins, partitions, "label_daily")

    bounds = f"signal_date >= {start:%Y.%m.%d} and signal_date <= {end:%Y.%m.%d}"
    ids = ", ".join("'" + str(fid).replace("'", "''") + "'" for fid in factor_ids)
    label_version_list = ", ".join("'" + version + "'" for version in label_versions)

    ok_l, labels = query(
        holder,
        f"select signal_date, count(*) as n from loadTable('{TARGET_DATABASE}','label_daily') "
        f"where {bounds} and label_valid=1 and data_version in [{label_version_list}] "
        "group by signal_date",
    )
    if not ok_l:
        raise RuntimeError("valid-label denominator query failed")

    def per_day(frame: Any, label: str) -> dict[date, int]:
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return {}
        if "signal_date" not in frame.columns or "n" not in frame.columns:
            raise RuntimeError(f"{label} query returned an unexpected shape: {list(frame.columns)}")
        return {
            pd.Timestamp(cast(Any, row.signal_date)).date(): int(cast(Any, row.n))
            for row in frame.itertuples(index=False)
        }

    valid_label_per_day = per_day(labels, "valid-label")

    # The numerator must be restricted to the securities the evaluator actually scores that day: a
    # security holding all nine factors but no valid label is not a row of the panel, and counting
    # it inflates coverage. Measured on 2023-04-28 the unrestricted count is 3,237 and the
    # restricted one 3,212 -- the difference between predicting 0.9070 (day survives) and 0.89997
    # (day dropped), and the run dropped it. Three of the 24 measured days turn on exactly this, and
    # all three are misses in the unsafe direction, so the restriction is part of the contract.
    #
    # DolphinDB will not alias a subquery or a loadTable inside a join, so this runs as a script:
    # both sides are bound to variables and combined with ej(). The inner grouping deduplicates
    # (day, security, factor) first, so a repeated business key cannot make eight factors read as
    # nine. Chunked by year to bound what the server holds in memory at once.
    intersection_per_day: dict[date, int] = {}
    years_queried: list[int] = []
    for year in range(start.year, end.year + 1):
        year_start = max(start, date(year, 1, 1))
        year_end = min(end, date(year, 12, 31))
        year_partitions = [
            partition for partition in partitions if partition.startswith(f"{year}-")
        ]
        label_list = _sql_list(sorted({label_pins[partition] for partition in year_partitions}))
        factor_list = _sql_list(sorted({factor_pins[partition] for partition in year_partitions}))
        year_bounds = (
            f"signal_date >= {year_start:%Y.%m.%d} and signal_date <= {year_end:%Y.%m.%d}"
        )
        script = (
            f"lab = select signal_date, ts_code from loadTable('{TARGET_DATABASE}','label_daily') "
            f"where {year_bounds} and label_valid=1 and data_version in [{label_list}];"
            "fac = select signal_date, ts_code, factor_id "
            f"from loadTable('{TARGET_DATABASE}','factor_daily') "
            f"where {year_bounds} and factor_id in [{ids}] and data_version in [{factor_list}];"
            "distinct_keys = select signal_date, ts_code, factor_id from fac "
            "group by signal_date, ts_code, factor_id;"
            "complete = select signal_date, ts_code from distinct_keys "
            f"group by signal_date, ts_code having count(*) = {len(factor_ids)};"
            "joined = select lab.signal_date as signal_date, lab.ts_code as ts_code "
            "from ej(lab, complete, `signal_date`ts_code);"
            "select signal_date, count(*) as n from joined group by signal_date"
        )
        ok_i, intersection = query(holder, script)
        if not ok_i:
            raise RuntimeError(f"dense intersection query failed for {year}")
        intersection_per_day.update(per_day(intersection, f"intersection {year}"))
        years_queried.append(year)
        print(f"  intersection measured for {year}", flush=True)

    diagnostics = {
        "factor_count": len(factor_ids),
        "factor_ids": [str(fid) for fid in factor_ids],
        "partitions_resolved": len(partitions),
        "years_queried": years_queried,
        "factor_versions_resolved": len(factor_versions),
        "label_versions_resolved": len(label_versions),
        "numerator_restricted_to_valid_label_rows": True,
        "days_with_intersection_rows": len(intersection_per_day),
        "days_with_valid_label_rows": len(valid_label_per_day),
        "days_with_labels_but_no_intersection": sorted(
            str(day) for day in valid_label_per_day if day not in intersection_per_day
        ),
    }
    return intersection_per_day, valid_label_per_day, diagnostics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", type=date.fromisoformat, required=True)
    parser.add_argument("--end", type=date.fromisoformat, required=True)
    parser.add_argument(
        "--factor-source",
        choices=("dense_v2", "all_v2"),
        default=None,
        help="dense_v2 is the nine factors the frozen best_simple baseline is defined on, which is "
        "the intersection the evaluator's floor actually sees; all_v2 is the full frozen fifteen",
    )
    parser.add_argument(
        "--factors",
        default=None,
        help="an explicit comma-separated factor list, for screening a candidate factor set before "
        "it is committed to the driver's frozen constants; overrides --factor-source",
    )
    parser.add_argument(
        "--floor",
        type=float,
        default=None,
        help="override the score-health floor; by default it is parsed out of the evaluator source",
    )
    parser.add_argument(
        "--compare",
        type=Path,
        default=None,
        help="an analyze_yearly_baseline_deltas JSON whose measured unevaluable days to check "
        "the prediction against",
    )
    parser.add_argument(
        "--json", type=Path, default=None, help="write the full measurement as JSON"
    )
    parser.add_argument(
        "--fail-on-prediction",
        action="store_true",
        help="exit 1 if any day is predicted below the floor; off by default because acting on a "
        "prediction is a governance decision, not a measurement",
    )
    args = parser.parse_args(argv)

    if args.factors is not None and args.factor_source is not None:
        parser.error("--factors and --factor-source cannot both be given")
    explicit: list[str] | None = None
    if args.factors is not None:
        try:
            explicit = parse_factor_list(args.factors)
        except ValueError as error:
            parser.error(str(error))

    floor = args.floor if args.floor is not None else current_floor()
    floor_source = "command line" if args.floor is not None else f"parsed from {GATES_PATH}"

    sys.path.insert(0, str(RESEARCH_V2 / "scripts"))
    from walk_forward_common import new_holder
    from walk_forward_factor_daily import FROZEN_FACTOR_SET_V2, FROZEN_FACTOR_SET_V2_DENSE

    if explicit is not None:
        factor_ids: tuple[str, ...] = tuple(explicit)
        factor_source = "explicit"
    else:
        source = args.factor_source or "dense_v2"
        factor_ids = tuple(
            FROZEN_FACTOR_SET_V2_DENSE if source == "dense_v2" else FROZEN_FACTOR_SET_V2
        )
        factor_source = source
    print(
        f"window {args.start} .. {args.end}; factor source {factor_source} "
        f"({len(factor_ids)} factors); floor {floor} ({floor_source})",
        flush=True,
    )
    intersection, denominator, diagnostics = collect(new_holder(), args.start, args.end, factor_ids)
    measured = daily_coverage(intersection, denominator, floor=floor)
    summary = summarize(measured, floor=floor)

    print(
        f"days measured: {summary['total_days']}; predicted below the floor: "
        f"{summary['days_below_floor']}",
        flush=True,
    )
    print(
        f"coverage: min {summary['coverage_min']}, median {summary['coverage_median']}",
        flush=True,
    )
    for year, bucket in summary["by_year"].items():
        print(
            f"  {year}: {bucket['days']:>4} days, {bucket['days_below_floor']:>3} below floor, "
            f"coverage min/median {bucket['coverage_min']:.4f}/{bucket['coverage_median']:.4f}",
            flush=True,
        )
    worst = sorted(measured, key=lambda entry: entry.coverage)[:10]
    if worst:
        print("worst days:", flush=True)
        for entry in worst:
            flag = "BELOW" if entry.below_floor else "above"
            print(
                f"  {entry.day} {entry.coverage:.4f} {flag} floor "
                f"({entry.intersection_rows}/{entry.valid_label_rows})",
                flush=True,
            )
    if diagnostics["days_with_labels_but_no_intersection"]:
        print(
            "WARNING: days with valid labels but no complete intersection at all: "
            f"{diagnostics['days_with_labels_but_no_intersection'][:5]}",
            flush=True,
        )

    payload: dict[str, Any] = {
        "window": [args.start.isoformat(), args.end.isoformat()],
        "factor_source": factor_source,
        "floor": floor,
        "floor_source": floor_source,
        "diagnostics": diagnostics,
        "summary": summary,
        "per_day": [
            {
                "day": entry.day.isoformat(),
                "intersection_rows": entry.intersection_rows,
                "valid_label_rows": entry.valid_label_rows,
                "coverage": entry.coverage,
                "below_floor": entry.below_floor,
            }
            for entry in measured
        ],
    }

    exit_code = 0
    if args.compare is not None:
        known = known_days_from_delta_report(json.loads(args.compare.read_text(encoding="utf-8")))
        comparison = compare_with_known(summary["days_below_floor_list"], known)
        payload["comparison_with_measured_run"] = comparison | {"compared_again": str(args.compare)}
        print(
            f"\ncompared against {args.compare}:\n"
            f"  predicted {comparison['predicted_days']} vs measured {comparison['known_days']}; "
            f"matched {len(comparison['matched'])}; identical={comparison['identical']}",
            flush=True,
        )
        if comparison["only_predicted"]:
            print(f"  only predicted: {comparison['only_predicted'][:8]}", flush=True)
        if comparison["only_known"]:
            print(f"  only measured:  {comparison['only_known'][:8]}", flush=True)
        if not comparison["identical"]:
            exit_code = 1

    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\njson written: {args.json}", flush=True)

    if args.fail_on_prediction and summary["days_below_floor"]:
        return 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
