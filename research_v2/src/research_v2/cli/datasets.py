"""Command-line inspection and fail-closed build planning for immutable datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer

from research_v2.datasets.builder import DatasetBuildRequest, LockedDatasetError
from research_v2.datasets.storage import DatasetArtifactStore


def artifact_root() -> Path:
    """Return V2's standalone artifact root, never a legacy V1 output directory."""
    return Path(os.environ.get("QRV2_ARTIFACT_ROOT", "artifacts/datasets"))


def inspect_dataset(dataset_id: str) -> None:
    """Print one published artifact manifest without mutating it."""
    manifest_path = artifact_root() / dataset_id / "manifest.json"
    if not manifest_path.is_file():
        raise typer.BadParameter(f"dataset artifact was not found: {dataset_id}")
    typer.echo(manifest_path.read_text(encoding="utf-8"), nl=False)


def verify_dataset(dataset_id: str) -> None:
    """Check every parquet shard against the immutable manifest SHA-256 values."""
    valid = DatasetArtifactStore(artifact_root()).verify(dataset_id)
    typer.echo(json.dumps({"dataset_id": dataset_id, "valid": valid}, sort_keys=True))
    if not valid:
        raise typer.Exit(code=1)


def plan_build(fold_id: str, factor_policy: str, include_test: bool) -> None:
    """Validate a build request and fail closed until a published-data reader is configured."""
    if not factor_policy:
        raise typer.BadParameter("factor policy cannot be empty")
    roles = ["TRAIN", "VALIDATION"]
    if include_test:
        roles.append("TEST")
    try:
        for role in roles:
            DatasetBuildRequest(fold_id=fold_id, split_role=role)  # type: ignore[arg-type]
    except LockedDatasetError as error:
        typer.echo(str(error))
        raise typer.Exit(code=2) from error
    typer.echo(
        json.dumps(
            {
                "factor_policy": factor_policy,
                "fold_id": fold_id,
                "roles": roles,
                "target_artifact_root": artifact_root().as_posix(),
            },
            sort_keys=True,
        )
    )
    typer.echo("published-data runtime backend is not configured; refusing to build", err=False)
    raise typer.Exit(code=2)
