"""Fail-closed walk-forward evaluation command plans."""

from __future__ import annotations

import json

import typer

from research_v2.evaluation.walk_forward import ORDINARY_TEST_FOLDS
from research_v2.reporting.experiment import validate_experiment_id


def evaluate_walk_forward(
    experiment_id: str,
    folds: str,
    *,
    compare: str = "best-simple",
) -> None:
    """Validate ordinary fold scope before loading any experiment artifact."""
    try:
        validate_experiment_id(experiment_id)
    except ValueError as error:
        typer.echo(str(error))
        raise typer.Exit(code=2) from error
    fold_ids = tuple(item.strip() for item in folds.split(",") if item.strip())
    if set(fold_ids) != set(ORDINARY_TEST_FOLDS) or len(fold_ids) != len(ORDINARY_TEST_FOLDS):
        typer.echo("walk-forward evaluation requires exactly F01,F02,F03,F04,F05,F06")
        raise typer.Exit(code=2)
    if compare != "best-simple":
        typer.echo("walk-forward evaluation requires --compare best-simple")
        raise typer.Exit(code=2)
    typer.echo(
        json.dumps(
            {
                "compare": compare,
                "experiment_id": experiment_id,
                "folds": list(fold_ids),
                "role": "ORDINARY_TEST",
            },
            sort_keys=True,
        )
    )
    typer.echo("runtime backend is not configured; refusing to evaluate walk-forward")
    raise typer.Exit(code=2)
