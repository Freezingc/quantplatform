"""Top-level Typer application."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import typer

from research_v2.cli.baselines import run_baseline
from research_v2.cli.datasets import inspect_dataset, plan_build, verify_dataset
from research_v2.cli.evaluate import evaluate_walk_forward
from research_v2.cli.export import compatibility, verify
from research_v2.cli.inference import daily as inference_daily
from research_v2.cli.inference import retry as inference_retry
from research_v2.cli.inference import status as inference_status
from research_v2.cli.materialize import emit_plan
from research_v2.cli.models import plan_model_train
from research_v2.cli.registry import pointer_app, promote
from research_v2.cli.reports import report_experiment

app = typer.Typer(no_args_is_help=True)
materialize_app = typer.Typer(no_args_is_help=True)
dataset_app = typer.Typer(no_args_is_help=True)
baseline_app = typer.Typer(no_args_is_help=True)
model_app = typer.Typer(no_args_is_help=True)
registry_app = typer.Typer(no_args_is_help=True)
evaluate_app = typer.Typer(no_args_is_help=True)
report_app = typer.Typer(no_args_is_help=True)
export_app = typer.Typer(no_args_is_help=True)
inference_app = typer.Typer(no_args_is_help=True)
app.add_typer(materialize_app, name="materialize")
app.add_typer(dataset_app, name="dataset")
app.add_typer(baseline_app, name="baseline")
app.add_typer(model_app, name="model")
app.add_typer(registry_app, name="registry")
registry_app.add_typer(pointer_app, name="pointer")
app.add_typer(evaluate_app, name="evaluate")
app.add_typer(report_app, name="report")
app.add_typer(export_app, name="export")
app.add_typer(inference_app, name="inference")


def _materialize_command(operation: str) -> Callable[..., None]:
    def command(
        start: str = typer.Option(..., "--start"),
        end: str = typer.Option(..., "--end"),
        dry_run: bool = typer.Option(False, "--dry-run"),
    ) -> None:
        emit_plan(operation, start, end, dry_run)

    return command


materialize_app.command("security-master")(_materialize_command("security-master"))
materialize_app.command("market")(_materialize_command("market"))
materialize_app.command("universe")(_materialize_command("universe"))
materialize_app.command("labels")(_materialize_command("labels"))


@dataset_app.command("build")
def dataset_build(
    fold_id: str = typer.Option(..., "--fold"),
    factor_policy: str = typer.Option("stable-v1", "--factor-policy"),
    include_test: bool = typer.Option(False, "--include-test"),
) -> None:
    """Emit a validated build plan; the data reader is intentionally fail-closed for now."""
    plan_build(fold_id, factor_policy, include_test)


@dataset_app.command("inspect")
def dataset_inspect(dataset_id: str = typer.Option(..., "--dataset-id")) -> None:
    """Print an immutable dataset manifest."""
    inspect_dataset(dataset_id)


@dataset_app.command("verify")
def dataset_verify(dataset_id: str = typer.Option(..., "--dataset-id")) -> None:
    """Verify artifact shard hashes."""
    verify_dataset(dataset_id)


@baseline_app.command("run")
def baseline_run(
    fold: str = typer.Option(..., "--fold"),
    models: str = typer.Option("B0,B1,B2,B3", "--models"),
) -> None:
    run_baseline(fold, models)


@model_app.command("train")
def model_train(
    family: str = typer.Option(..., "--family"),
    fold: str = typer.Option(..., "--fold"),
) -> None:
    plan_model_train(family, fold)


@registry_app.command("promote")
def registry_promote(
    model_id: str = typer.Option(..., "--model-id"),
    target: str = typer.Option(..., "--to"),
    evidence: Path = typer.Option(..., "--evidence"),  # noqa: B008
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    promote(model_id, target, evidence, dry_run)


@evaluate_app.command("walk-forward")
def evaluate_walk_forward_command(
    experiment_id: str = typer.Option(..., "--experiment-id"),
    folds: str = typer.Option("F01,F02,F03,F04,F05,F06", "--folds"),
    compare: str = typer.Option("best-simple", "--compare"),
) -> None:
    evaluate_walk_forward(experiment_id, folds, compare=compare)


@report_app.command("experiment")
def report_experiment_command(
    experiment_id: str = typer.Option(..., "--experiment-id"),
    format: str = typer.Option("html", "--format"),
) -> None:
    report_experiment(experiment_id, format)


@export_app.command("compatibility")
def export_compatibility_command(
    signal_date: str = typer.Option(..., "--signal-date"),
    model_id: str = typer.Option(..., "--model-id"),
    format: str = typer.Option("csv,pkl", "--format"),
) -> None:
    compatibility(signal_date, model_id, format)


@export_app.command("verify")
def export_verify_command(
    manifest: Path = typer.Option(..., "--manifest"),  # noqa: B008
) -> None:
    verify(manifest)


@inference_app.command("daily")
def inference_daily_command(
    signal_date: str = typer.Option(..., "--signal-date"),
    model_id: str = typer.Option(..., "--model-id"),
    mode: str = typer.Option(..., "--mode"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    inference_daily(signal_date, model_id, mode, dry_run)


@inference_app.command("status")
def inference_status_command(
    signal_date: str = typer.Option(..., "--signal-date"),
    model_id: str = typer.Option(..., "--model-id"),
) -> None:
    inference_status(signal_date, model_id)


@inference_app.command("retry")
def inference_retry_command(
    run_id: str = typer.Option(..., "--run-id"),
    from_step: str = typer.Option(..., "--from-step"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    inference_retry(run_id, from_step, dry_run)


if __name__ == "__main__":
    app()
