"""Fail-closed registry promotion command plans."""

from __future__ import annotations

import json
from pathlib import Path

import typer

from research_v2.registry.model_registry import (
    ModelStatus,
    load_walk_forward_evidence,
    validate_model_id,
)


def promote(model_id: str, target: str, evidence: Path, dry_run: bool) -> None:
    """Recalculate G1-G6 from immutable evidence before a registry backend is contacted."""
    if target != ModelStatus.WALK_FORWARD_PASSED.value:
        typer.echo("this registry command only permits --to WALK_FORWARD_PASSED")
        raise typer.Exit(code=2)
    try:
        validate_model_id(model_id)
        _, evidence_hash, gate_report = load_walk_forward_evidence(evidence)
    except (TypeError, ValueError) as error:
        typer.echo(str(error))
        raise typer.Exit(code=2) from error
    payload = {
        "dry_run": dry_run,
        "evidence_hash": evidence_hash,
        "gate_statuses": {
            gate: result.status.value for gate, result in sorted(gate_report.results.items())
        },
        "model_id": model_id,
        "to": target,
    }
    typer.echo(json.dumps(payload, sort_keys=True))
    if dry_run:
        return
    typer.echo("registry persistence backend is not configured; refusing to promote")
    raise typer.Exit(code=2)
