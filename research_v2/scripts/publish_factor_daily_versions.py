"""Registration-only version publisher for the materialized factor_daily table.

The frozen-set materializer wrote factor values directly into ``factor_daily``
(bypassing staging). ``V2BatchPublisher`` appends data rows, which would
duplicate the already-materialized data; therefore this publisher is
registration-only: it fail-closed pre-validates each calendar month region
(exactly one data_version, non-empty, full frozen-factor coverage, zero
duplicate business keys) and then writes only ``published_version`` pointer
rows (object_name='factor_daily'), preserving the "pointer advances last"
invariant. Dormant until executed against the live database with --execute.

Usage:
  publish_factor_daily_versions.py --months 2020-01..2025-12 [--execute]
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

MONTH_PATTERN = re.compile(r"^\d{4}-(0[1-9]|1[0-2])$")
OBJECT_NAME = "factor_daily"
BUSINESS_KEYS: tuple[str, ...] = ("factor_id", "signal_date", "ts_code")


class RegistrationRefused(ValueError):
    """Fail-closed refusal: a month region does not satisfy registration premises."""


@dataclass(frozen=True)
class PlanEntry:
    month: str
    data_version: str
    previous_version: str
    row_count: int
    action: str  # REGISTER / SKIP_CURRENT


def plan_registrations(
    month_states: Mapping[str, Mapping[str, Any]],
    *,
    expected_factor_count: int,
) -> list[PlanEntry]:
    """Build the fail-closed registration plan from per-month observed states.

    Each state carries: versions (distinct data_versions observed),
    rows_by_version, factors (distinct factor_ids), duplicate_key_count,
    and registered_version (current published_version pointer, if any).
    Pointer chains are per (object_name, logical_partition), so each month's
    previous_version comes only from its own registered_version.
    """
    plan: list[PlanEntry] = []
    for month in sorted(month_states):
        state = month_states[month]
        versions = list(state["versions"])
        if len(versions) != 1:
            raise RegistrationRefused(
                f"{month}: multiple data_versions present ({sorted(versions)}); "
                "registration requires exactly one version per month partition"
            )
        version = str(versions[0])
        row_count = int(state["rows_by_version"].get(version, 0))
        if row_count <= 0:
            raise RegistrationRefused(f"{month}: no rows for data_version {version}")
        factors = list(state["factors"])
        if len(factors) != expected_factor_count:
            raise RegistrationRefused(
                f"{month}: factor coverage {len(factors)} != expected {expected_factor_count}"
            )
        duplicates = int(state.get("duplicate_key_count", 0))
        if duplicates != 0:
            raise RegistrationRefused(
                f"{month}: {duplicates} duplicate business keys "
                f"({', '.join(BUSINESS_KEYS)}) must be zero"
            )
        registered = state.get("registered_version")
        if registered == version:
            plan.append(
                PlanEntry(
                    month=month,
                    data_version=version,
                    previous_version="",
                    row_count=row_count,
                    action="SKIP_CURRENT",
                )
            )
            continue
        plan.append(
            PlanEntry(
                month=month,
                data_version=version,
                previous_version=str(registered) if registered else "",
                row_count=row_count,
                action="REGISTER",
            )
        )
    return plan


def pointer_script(entry: PlanEntry, *, object_name: str, target_database: str) -> str:
    """Render the pointer-only insert (mirrors V2BatchPublisher._pointer_script)."""

    def quoted(value: str) -> str:
        return value.replace("'", "''")

    return (
        "publishPointer = table("
        f"[`{object_name}] as object_name, "
        f"['{quoted(entry.month)}'] as logical_partition, "
        f"['{quoted(entry.data_version)}'] as data_version, "
        f"['{quoted(entry.previous_version)}'] as previous_data_version, "
        "[now()] as published_at, [uuid()] as run_id)\n"
        "tableInsert{loadTable("
        f"'{target_database}', 'published_version')"
        "}(publishPointer)"
    )


def month_bounds(month: str) -> str:
    """DolphinDB date-literal bounds [month start, next month start) with year rollover."""
    year, mon = month[:4], month[5:]
    next_year = int(year) + (1 if mon == "12" else 0)
    next_mon = (int(mon) % 12) + 1
    return (
        f"signal_date >= {year}.{mon}.01 "
        f"and signal_date < {next_year:04d}.{next_mon:02d}.01"
    )


def parse_month_range(spec: str) -> list[str]:
    """Expand 'YYYY-MM..YYYY-MM' (inclusive) into the month list."""
    start, separator, end = spec.partition("..")
    if not separator:
        raise ValueError("month range must look like 2020-01..2025-12")
    for bound in (start, end):
        if not MONTH_PATTERN.fullmatch(bound):
            raise ValueError(f"invalid month bound: {bound}")
    months: list[str] = []
    year, month = int(start[:4]), int(start[5:])
    end_year, end_month = int(end[:4]), int(end[5:])
    while (year, month) <= (end_year, end_month):
        months.append(f"{year:04d}-{month:02d}")
        month += 1
        if month == 13:
            year, month = year + 1, 1
    return months


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--months", required=True, help="inclusive range, e.g. 2020-01..2025-12")
    parser.add_argument("--expected-factor-count", type=int, default=17)
    parser.add_argument("--execute", action="store_true", help="write pointers (default: dry run)")
    args = parser.parse_args()
    months = parse_month_range(args.months)

    # Live-database glue is imported lazily so unit tests never need DolphinDB.
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1] / "scripts"))
    from walk_forward_common import TARGET_DATABASE, new_holder, query

    holder = new_holder()
    states: dict[str, dict[str, Any]] = {}
    for month in months:
        bounds = month_bounds(month)
        ok, frame = query(
            holder,
            f"select data_version, count(*) as n from loadTable('{TARGET_DATABASE}','{OBJECT_NAME}') "
            f"where {bounds} group by data_version",
        )
        versions: list[str] = []
        rows_by_version: dict[str, int] = {}
        if ok and isinstance(frame, pd.DataFrame) and not frame.empty:
            versions = [str(v) for v in frame["data_version"]]
            rows_by_version = {str(v): int(n) for v, n in zip(frame["data_version"], frame["n"])}
        ok_f, frame_f = query(
            holder,
            f"select distinct(factor_id) as fid from loadTable('{TARGET_DATABASE}','{OBJECT_NAME}') "
            f"where {bounds}",
        )
        factors = (
            [str(f) for f in frame_f["fid"]]
            if ok_f and isinstance(frame_f, pd.DataFrame) and not frame_f.empty
            else []
        )
        ok_d, frame_d = query(
            holder,
            "select count(*) as n from (select factor_id, signal_date, ts_code "
            f"from loadTable('{TARGET_DATABASE}','{OBJECT_NAME}') where {bounds} "
            "group by factor_id, signal_date, ts_code having count(*) > 1)",
        )
        duplicates = int(frame_d.iloc[0]["n"]) if ok_d and isinstance(frame_d, pd.DataFrame) else -1
        ok_r, frame_r = query(
            holder,
            "select top 1 data_version from loadTable("
            f"'{TARGET_DATABASE}', 'published_version') where object_name=`{OBJECT_NAME}, "
            f"logical_partition='{month}' order by published_at desc",
        )
        registered = (
            str(frame_r.iloc[0]["data_version"])
            if ok_r and isinstance(frame_r, pd.DataFrame) and not frame_r.empty
            else None
        )
        states[month] = {
            "versions": versions,
            "rows_by_version": rows_by_version,
            "factors": factors,
            "duplicate_key_count": duplicates,
            "registered_version": registered,
        }

    plan = plan_registrations(states, expected_factor_count=args.expected_factor_count)
    for entry in plan:
        print(f"{entry.month}: {entry.action} version={entry.data_version} rows={entry.row_count}")
    if not args.execute:
        print(f"dry run complete; {sum(1 for e in plan if e.action == 'REGISTER')} pointer(s) to write")
        return
    for entry in plan:
        if entry.action != "REGISTER":
            continue
        from walk_forward_common import run_query_with_retry

        run_query_with_retry(holder["connect"], holder["session"], pointer_script(
            entry, object_name=OBJECT_NAME, target_database=TARGET_DATABASE
        ))
    print(f"executed; {sum(1 for e in plan if e.action == 'REGISTER')} pointer(s) written")


if __name__ == "__main__":
    main()
