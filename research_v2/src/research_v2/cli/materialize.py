"""Fail-closed materialization command plans."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import date

import typer

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
