"""Fail-closed materialization command plans."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path

import typer
import yaml

from research_v2.factors.catalog_dry_run import schema_set_from_payload
from research_v2.factors.materialize_plan import FactorPlanError, plan_factor_materialization
from research_v2.pit.calendar import TradingCalendar

_TARGET_DATABASE = "dfs://QuantResearchV2Test"


@dataclass(frozen=True)
class MaterializePlan:
    operation: str
    start_date: str
    end_date: str
    source_tables: tuple[str, ...]
    target_database: str
    target_table: str
    dry_run: bool


_PLANS = {
    "security-master": (
        ("dfs://YesLab/stock_info_table", "dfs://YesLab/stock_used_name"),
        "security_master_scd",
    ),
    "market": (("dfs://YesLab/stock_factor_daily",), "market_daily_clean"),
    "universe": (
        (
            "dfs://YesLab/stock_info_table",
            "dfs://YesLab/stock_used_name",
            "dfs://YesLab/stock_trading_suspension_info",
            "dfs://QuantResearchV2Test/market_daily_clean",
        ),
        "universe_daily",
    ),
    "labels": (("dfs://QuantResearchV2Test/market_daily_clean",), "label_daily"),
}


def emit_plan(operation: str, start: str, end: str, dry_run: bool) -> None:
    """Validate date range and emit the fully qualified, no-write execution plan."""
    try:
        start_date = date.fromisoformat(start)
        end_date = date.fromisoformat(end)
    except ValueError as error:
        raise typer.BadParameter("dates must use YYYY-MM-DD") from error
    if start_date > end_date:
        raise typer.BadParameter("start cannot be after end")
    source_tables, target_table = _PLANS[operation]
    plan = MaterializePlan(
        operation=operation,
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        source_tables=source_tables,
        target_database=_TARGET_DATABASE,
        target_table=target_table,
        dry_run=dry_run,
    )
    typer.echo(json.dumps(asdict(plan), ensure_ascii=False, sort_keys=True))
    if not dry_run:
        typer.echo("runtime backend is not configured; refusing to materialize", err=False)
        raise typer.Exit(code=2)


def _parse_iso_date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as error:
        raise typer.BadParameter(f"{label} must use YYYY-MM-DD") from error


def _load_calendar_dates(path: Path) -> list[date]:
    """Read trading dates from a plain file: one ISO date per line, or a JSON list.

    The calendar is an explicit input rather than something fetched here, so this command stays
    metadata-only and runnable with no cluster reachable. That is also what makes its output
    reproducible: the same calendar file and catalog always give the same plan.
    """
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise typer.BadParameter(f"calendar file is empty: {path}")
    raw: list[str]
    if text.startswith("["):
        parsed = json.loads(text)
        if not isinstance(parsed, list):
            raise typer.BadParameter("calendar JSON must be a list of ISO dates")
        raw = [str(value) for value in parsed]
    else:
        raw = [line for line in text.splitlines()]
    dates = sorted(
        {
            date.fromisoformat(token.strip())
            for token in raw
            if token.strip() and not token.strip().startswith("#")
        }
    )
    if not dates:
        raise typer.BadParameter(f"calendar file contains no dates: {path}")
    return dates


def _load_catalog_entries(path: Path) -> list[dict[str, object]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8"))
    entries = list((document or {}).get("factors") or [])
    if not entries:
        raise typer.BadParameter(f"catalog has no factor entries: {path}")
    return entries


def plan_factors(
    catalog: Path,
    start: str,
    end: str,
    *,
    calendar_path: Path,
    schemas_path: Path | None,
    cross_section: int | None,
    lookback_trading_days: int,
    strict: bool,
    dry_run: bool,
    output: Path | None,
) -> None:
    """Emit a factor materialization plan built only from metadata.

    Reads no factor values and writes no table: the planner it calls accepts no database session of
    any kind. Without ``--schemas`` the plan is explicitly reported as unverified against the source
    and therefore not materializable; pass the file written by
    ``scripts/audit_factor_catalog_schema_dry_run.py --save-schemas`` to verify it offline.
    """
    start_date = _parse_iso_date(start, "--start")
    end_date = _parse_iso_date(end, "--end")
    if start_date > end_date:
        raise typer.BadParameter("--start cannot be after --end")
    if cross_section is not None and cross_section < 1:
        raise typer.BadParameter("--cross-section must be a positive number of securities")
    if lookback_trading_days < 1:
        raise typer.BadParameter("--lookback-trading-days must be positive")

    entries = _load_catalog_entries(catalog)
    calendar = TradingCalendar.from_dates(_load_calendar_dates(calendar_path))

    schemas = None
    schema_errors: dict[str, str] = {}
    if schemas_path is not None:
        payload = json.loads(schemas_path.read_text(encoding="utf-8"))
        schemas, schema_errors = schema_set_from_payload(payload)

    try:
        plan = plan_factor_materialization(
            entries,
            calendar=calendar,
            start_date=start_date,
            end_date=end_date,
            schemas=schemas,
            cross_section_size=cross_section,
            lookback_trading_days=lookback_trading_days,
            strict=strict,
        )
    except FactorPlanError as error:
        typer.echo(f"factor materialization plan refused: {error}", err=True)
        raise typer.Exit(code=2) from error

    document = {
        "start_date": plan.start_date.isoformat(),
        "end_date": plan.end_date.isoformat(),
        "factor_count": plan.factor_count,
        "source_tables": list(plan.source_tables),
        "source_columns": {table: list(columns) for table, columns in plan.source_columns.items()},
        "chunks": [asdict(chunk) for chunk in plan.chunks],
        "cross_section_size": plan.cross_section_size,
        "estimated_rows_total": plan.estimated_rows_total,
        "schema_verified": plan.schema_verified,
        "materializable": plan.materializable,
        "blocked_factors": [list(key) for key in plan.blocked_factors],
        "unmapped_source_columns": {
            table: list(columns) for table, columns in plan.unmapped_source_columns.items()
        },
        "schema_errors": schema_errors,
        "target_table": plan.target_table,
        "target_partition_column": plan.target_partition_column,
        "catalog": str(catalog),
        "calendar": str(calendar_path),
    }
    rendered = json.dumps(document, ensure_ascii=False, indent=2, sort_keys=True, default=str)
    typer.echo(rendered)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
        typer.echo(f"plan written to {output}", err=True)
    if not dry_run:
        typer.echo(
            "factor materialization backend is not wired to this command; refusing to materialize",
            err=False,
        )
        raise typer.Exit(code=2)
