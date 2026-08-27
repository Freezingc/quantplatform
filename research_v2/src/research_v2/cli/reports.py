"""Fail-closed experiment report command plans."""

from __future__ import annotations

import json

import typer

from research_v2.reporting.experiment import validate_experiment_id


def report_experiment(experiment_id: str, format: str) -> None:
    """Validate report identity and format before reading experiment artifacts."""
    try:
        validate_experiment_id(experiment_id)
    except ValueError as error:
        typer.echo(str(error))
        raise typer.Exit(code=2) from error
    if format not in {"json", "html"}:
        typer.echo("report format must be json or html")
        raise typer.Exit(code=2)
    typer.echo(
        json.dumps(
            {
                "experiment_id": experiment_id,
                "format": format,
            },
            sort_keys=True,
        )
    )
    typer.echo("runtime backend is not configured; refusing to render experiment report")
    raise typer.Exit(code=2)
