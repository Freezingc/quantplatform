"""Fail-closed baseline run command plans."""

from __future__ import annotations

import json

import typer

from research_v2.baselines.orchestration import SIMPLE_BASELINE_IDS
from research_v2.evaluation.walk_forward import ORDINARY_TEST_FOLDS


def run_baseline(fold: str, models: str) -> None:
    """Validate a baseline run scope before a runtime data backend is consulted."""
    model_ids = tuple(item.strip() for item in models.split(",") if item.strip())
    if fold == "FINAL":
        typer.echo("FINAL test is locked and cannot enter the ordinary baseline command")
        raise typer.Exit(code=2)
    if fold not in ORDINARY_TEST_FOLDS:
        typer.echo("baseline run fold must be one of F01,F02,F03,F04,F05,F06")
        raise typer.Exit(code=2)
    if set(model_ids) != set(SIMPLE_BASELINE_IDS) or len(model_ids) != len(SIMPLE_BASELINE_IDS):
        missing = sorted(set(SIMPLE_BASELINE_IDS).difference(model_ids))
        unexpected = sorted(set(model_ids).difference(SIMPLE_BASELINE_IDS))
        details: list[str] = []
        if missing:
            details.append(f"missing: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected: {', '.join(unexpected)}")
        typer.echo("baseline run requires exactly B0,B1,B2,B3 (" + "; ".join(details) + ")")
        raise typer.Exit(code=2)
    typer.echo(
        json.dumps(
            {
                "fold_id": fold,
                "models": list(model_ids),
                "role": "TRAIN_VALIDATION",
            },
            sort_keys=True,
        )
    )
    typer.echo("runtime backend is not configured; refusing to run baseline")
    raise typer.Exit(code=2)
