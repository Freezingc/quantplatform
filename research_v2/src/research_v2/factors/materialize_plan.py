"""Plan a factor materialization run from metadata alone.

C5.3 asks the dry-run to report the factor count, the source tables and columns, the date chunks,
an estimated row count and the target partitions, while reading no values and writing nothing. This
module produces that plan.

Both prohibitions are structural rather than behavioural. The planner accepts **no session, client
or connection of any kind**: its inputs are the catalog entries, an optional schema map that some
*other* code already collected read-only, and a trading calendar. There is no parameter through
which a statement could reach a database, so "it did not read values" is a property of the signature
rather than a promise kept by discipline inside it.

The row estimate is the one number a metadata-only planner genuinely cannot know,
because it depends on how many securities trade each day. Rather than guess one -- a
guessed width yields a figure that looks measured -- the estimate is ``None`` unless the
caller supplies ``cross_section_size`` explicitly, and the formula is then stated in one
place. Obtaining a real width needs a count against the source, which is exactly what a
dry-run must not do.

The registry-facing commands in C5.3 are not built here. They are CLI over
``FactorRegistry``, which has no production caller, and which representation of a frozen
factor set is authoritative is an open decision. This planner needs no registry: catalog
plus schema dry-run plus calendar is enough.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from research_v2.backfill.windows import resolve_monthly_backfill_windows
from research_v2.factors.catalog_dry_run import (
    SourceColumn,
    entry_key,
    registered_source_columns,
    run_catalog_dry_run,
)
from research_v2.pit.calendar import TradingCalendar

#: Where planned rows would be published, and how that table is partitioned. Both are read off
#: migration 003 rather than invented, so a plan names partitions that actually exist.
DEFAULT_TARGET_TABLE = "factor_observation_clean"
DEFAULT_TARGET_PARTITION_COLUMN = "signal_date"
DEFAULT_LOOKBACK_TRADING_DAYS = 20


class FactorPlanError(ValueError):
    """Raised when a materialization plan cannot be produced, or cannot be trusted."""


@dataclass(frozen=True)
class FactorMaterializeChunk:
    """One publishable month of a planned run.

    ``trading_days`` counts the target month only. The lookback window is carried because the read
    path needs it to compute features, but those days are not published for this month, so they are
    deliberately excluded from ``estimated_rows``.
    """

    logical_partition: str
    target_start: date
    target_end: date
    lookback_start: date
    trading_days: int
    source_tables: tuple[str, ...]
    factor_count: int
    estimated_rows: int | None


@dataclass(frozen=True)
class FactorMaterializePlan:
    """A whole planned run: scope, source binding, chunks, estimate and verification status."""

    start_date: date
    end_date: date
    factor_count: int
    source_tables: tuple[str, ...]
    source_columns: Mapping[str, tuple[str, ...]]
    chunks: tuple[FactorMaterializeChunk, ...]
    cross_section_size: int | None
    estimated_rows_total: int | None
    schema_verified: bool
    blocked_factors: tuple[tuple[str, int], ...]
    unmapped_source_columns: Mapping[str, tuple[str, ...]]
    target_table: str = DEFAULT_TARGET_TABLE
    target_partition_column: str = DEFAULT_TARGET_PARTITION_COLUMN

    @property
    def materializable(self) -> bool:
        """Whether this plan may be handed to a materializer as it stands.

        A plan that was never checked against the source is **not** materializable. That is a
        deliberate asymmetry: "unverified" and "verified clean" must not report the same thing, or
        skipping the schema dry-run would look like passing it.
        """
        return self.schema_verified and not self.blocked_factors


def plan_factor_materialization(
    entries: Sequence[Mapping[str, Any]],
    *,
    calendar: TradingCalendar,
    start_date: date,
    end_date: date,
    schemas: Mapping[str, tuple[SourceColumn, ...] | None] | None = None,
    retired: frozenset[tuple[str, int]] = frozenset(),
    cross_section_size: int | None = None,
    lookback_trading_days: int = DEFAULT_LOOKBACK_TRADING_DAYS,
    strict: bool = True,
    target_table: str = DEFAULT_TARGET_TABLE,
    target_partition_column: str = DEFAULT_TARGET_PARTITION_COLUMN,
) -> FactorMaterializePlan:
    """Build the dry-run plan for materializing ``entries`` over a closed date range.

    ``strict`` decides what a blocked factor means. By default one stops the plan, because C5.1's
    principle is that a step which cannot succeed must not contribute visible rows, and a factor
    whose registered column does not exist in the source cannot be read. With ``strict=False`` the
    plan is still returned with the blocked factors named, so a caller can report the whole picture
    rather than only the first failure.
    """
    if start_date > end_date:
        raise FactorPlanError("start_date cannot be after end_date")
    if not entries:
        raise FactorPlanError("the catalog supplied no factor entries to plan")
    if cross_section_size is not None and cross_section_size < 1:
        raise FactorPlanError(
            "cross_section_size must be a positive number of securities, or omitted entirely"
        )

    in_scope = [entry for entry in entries if entry_key(entry) not in retired]
    if not in_scope:
        raise FactorPlanError("no factor entries remain in scope after retirement")
    factor_count = len(in_scope)

    blocked: tuple[tuple[str, int], ...] = ()
    unmapped: Mapping[str, tuple[str, ...]] = {}
    schema_verified = schemas is not None
    if schemas is not None:
        report = run_catalog_dry_run(entries, schemas, retired=retired)
        blocked = tuple(
            sorted(
                (finding.factor_id, finding.factor_version)
                for finding in report.blocked_findings
            )
        )
        unmapped = dict(report.unmapped_source_columns)
        if strict and blocked:
            detail = "; ".join(
                f"{factor_id}:{version} ({', '.join(finding.reasons)})"
                for finding in report.blocked_findings
                for factor_id, version in ((finding.factor_id, finding.factor_version),)
            )
            raise FactorPlanError(
                f"{len(blocked)} catalog entries cannot be read from their declared source: "
                f"{detail}"
            )

    source_tables = tuple(
        sorted(
            {
                str(entry["source_table"])
                for entry in in_scope
                if str(entry.get("source_table", ""))
            }
        )
    )
    source_columns = {
        table: registered_source_columns(entries, table, retired=retired)
        for table in source_tables
    }

    try:
        windows = resolve_monthly_backfill_windows(
            calendar,
            start_date=start_date,
            end_date=end_date,
            lookback_trading_days=lookback_trading_days,
        )
    except ValueError as error:
        raise FactorPlanError(str(error)) from error

    chunks: list[FactorMaterializeChunk] = []
    for window in windows:
        trading_days = len(calendar.window(window.target_start, window.target_end))
        estimated_rows = (
            None
            if cross_section_size is None
            else factor_count * trading_days * cross_section_size
        )
        chunks.append(
            FactorMaterializeChunk(
                logical_partition=window.month,
                target_start=window.target_start,
                target_end=window.target_end,
                lookback_start=window.lookback_start,
                trading_days=trading_days,
                source_tables=source_tables,
                factor_count=factor_count,
                estimated_rows=estimated_rows,
            )
        )

    estimated_rows_total = (
        None
        if cross_section_size is None
        else sum(chunk.estimated_rows or 0 for chunk in chunks)
    )

    return FactorMaterializePlan(
        start_date=start_date,
        end_date=end_date,
        factor_count=factor_count,
        source_tables=source_tables,
        source_columns=source_columns,
        chunks=tuple(chunks),
        cross_section_size=cross_section_size,
        estimated_rows_total=estimated_rows_total,
        schema_verified=schema_verified,
        blocked_factors=blocked,
        unmapped_source_columns=unmapped,
        target_table=target_table,
        target_partition_column=target_partition_column,
    )
