"""Fail-closed commands for isolated V1-compatible prediction exports."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import typer

from research_v2.export.compatibility import verify_compatibility_export
from research_v2.registry.model_registry import validate_model_id


def verify(manifest: Path) -> None:
    """Verify a locally published compatibility manifest and its immutable file hashes."""
    try:
        valid = verify_compatibility_export(manifest)
    except ValueError as error:
        typer.echo(str(error))
        raise typer.Exit(code=2) from error
    typer.echo(json.dumps({"manifest": str(manifest), "valid": valid}, sort_keys=True))
    if not valid:
        raise typer.Exit(code=1)


def compatibility(signal_date: str, model_id: str, format: str) -> None:
    """Validate an export request before a future prediction repository adapter is allowed."""
    try:
        parsed_date = date.fromisoformat(signal_date)
        validate_model_id(model_id)
    except ValueError as error:
        typer.echo(str(error))
        raise typer.Exit(code=2) from error
    formats = tuple(item.strip() for item in format.split(",") if item.strip())
    if set(formats) != {"csv", "pkl"} or len(formats) != 2:
        typer.echo("compatibility export requires exactly --format csv,pkl")
        raise typer.Exit(code=2)
    typer.echo(
        json.dumps(
            {
                "formats": list(formats),
                "model_id": model_id,
                "signal_date": parsed_date.isoformat(),
                "target_root": "artifacts/exports/v2",
            },
            sort_keys=True,
        )
    )
    typer.echo("runtime backend is not configured; refusing to export")
    raise typer.Exit(code=2)
