"""Top-level Typer application."""

from __future__ import annotations

from collections.abc import Callable

import typer

from research_v2.cli.datasets import inspect_dataset, plan_build, verify_dataset
from research_v2.cli.materialize import emit_plan

app = typer.Typer(no_args_is_help=True)
materialize_app = typer.Typer(no_args_is_help=True)
dataset_app = typer.Typer(no_args_is_help=True)
app.add_typer(materialize_app, name="materialize")
app.add_typer(dataset_app, name="dataset")


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


if __name__ == "__main__":
    app()
