"""Fail-closed candidate-model training command plans."""

from __future__ import annotations

import json

import typer

from research_v2.evaluation.model_walk_forward import MODEL_FAMILIES
from research_v2.evaluation.walk_forward import ORDINARY_TEST_FOLDS


def plan_model_train(family: str, fold: str) -> None:
    """Validate one ordinary model-training scope before a runtime backend is consulted."""
    if family not in MODEL_FAMILIES:
        typer.echo("model family must be one of M1,M2,M3")
        raise typer.Exit(code=2)
    if fold == "FINAL":
        typer.echo("FINAL test is locked and cannot enter the ordinary model command")
        raise typer.Exit(code=2)
    if fold not in ORDINARY_TEST_FOLDS:
        typer.echo("model train fold must be one of F01,F02,F03,F04,F05,F06")
        raise typer.Exit(code=2)
    if family != "M1":
        typer.echo(
            "model family prerequisite evidence is unavailable; only M1 can be planned "
            "before the registry runtime is configured"
        )
        raise typer.Exit(code=2)
    typer.echo(
        json.dumps(
            {
                "family": family,
                "fold_id": fold,
                "role": "TRAIN_VALIDATION",
            },
            sort_keys=True,
        )
    )
    typer.echo("runtime backend is not configured; refusing to train model")
    raise typer.Exit(code=2)
