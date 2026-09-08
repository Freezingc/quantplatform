"""Fail-closed registry promotion command plans."""

from __future__ import annotations

import json
import re
import uuid
from pathlib import Path

import typer

from research_v2.monitoring.shadow import ShadowDeploymentApproval
from research_v2.registry.model_registry import (
    ModelStatus,
    load_walk_forward_evidence,
    validate_model_id,
)
from research_v2.registry.registry_store import ModelRegistryStore, RegistryStoreError

pointer_app = typer.Typer(no_args_is_help=True)
_CONSUMER_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_EVIDENCE_ID = re.compile(r"^[0-9a-f]{64}$")


def register(
    model_id: str, actor: str, parent_model_id: str | None, dry_run: bool
) -> None:
    """Record a DRAFT model so a later promotion has something to promote.

    ``promote_walk_forward`` only accepts a model already in DRAFT, so without this command the
    lifecycle has no entry point. The model_id is content-addressed; for a walk-forward
    candidate it comes from ``derive_walk_forward_model_id`` over the run's verified per-fold
    bundle digests, never from a name someone chose.
    """
    try:
        validate_model_id(model_id)
        if parent_model_id is not None:
            validate_model_id(parent_model_id)
    except ValueError as error:
        typer.echo(str(error))
        raise typer.Exit(code=2) from error
    if not actor.strip():
        typer.echo("--actor is required: every retained transition event names who caused it")
        raise typer.Exit(code=2)
    plan = {
        "actor": actor,
        "dry_run": dry_run,
        "model_id": model_id,
        "parent_model_id": parent_model_id,
        "status": ModelStatus.DRAFT.value,
    }
    typer.echo(json.dumps(plan, sort_keys=True))
    if dry_run:
        return
    store = _configured_store("register")
    try:
        record = store.register_draft(
            model_id, actor=actor, parent_model_id=parent_model_id
        )
    except (RegistryStoreError, ValueError) as error:
        typer.echo(str(error))
        raise typer.Exit(code=2) from error
    typer.echo(
        json.dumps(
            {
                "actor": actor,
                "lineage_id": record.lineage_id,
                "model_id": record.model_id,
                "status": record.status.value,
                "transition_events": len(store.registry.history(model_id)),
            },
            sort_keys=True,
        )
    )


def shadow(model_id: str, approval_root: Path, approval_id: str, dry_run: bool) -> None:
    """Record a SHADOW transition the evaluator has already approved.

    The deployer is QRV2_SHADOW_DEPLOYER_ID and must not be the approver; that separation is
    enforced by the registry, not here. This command only refuses early and legibly when the
    model id is malformed or the backend is not configured.
    """
    try:
        validate_model_id(model_id)
    except ValueError as error:
        typer.echo(str(error))
        raise typer.Exit(code=2) from error
    if not approval_id.strip():
        typer.echo("--approval-id is required: the signed approval this transition rests on")
        raise typer.Exit(code=2)
    plan = {
        "approval_id": approval_id,
        "approval_root": str(approval_root),
        "dry_run": dry_run,
        "model_id": model_id,
        "to": ModelStatus.SHADOW.value,
    }
    typer.echo(json.dumps(plan, sort_keys=True))
    if dry_run:
        return
    store = _configured_store("enter shadow")
    try:
        record = store.enter_shadow(
            model_id,
            approval=ShadowDeploymentApproval(approval_id, model_id),
            approval_root=approval_root,
        )
    except (RegistryStoreError, ValueError, OSError) as error:
        typer.echo(str(error))
        raise typer.Exit(code=2) from error
    typer.echo(
        json.dumps(
            {
                "approval_id": approval_id,
                "dry_run": False,
                "model_id": model_id,
                "status": record.status.value,
                "to": ModelStatus.SHADOW.value,
                "transition_events": len(store.registry.history(model_id)),
            },
            sort_keys=True,
        )
    )


def _configured_store(operation: str) -> ModelRegistryStore:
    """Open the persisted registry, or refuse and say exactly what is missing."""
    try:
        return ModelRegistryStore.from_environment()
    except RegistryStoreError as error:
        typer.echo(
            f"registry persistence backend is not configured; refusing to {operation}: {error}"
        )
        raise typer.Exit(code=2) from error


def promote(model_id: str, target: str, evidence: Path, actor: str, dry_run: bool) -> None:
    """Recalculate G1-G6 from immutable evidence, then record the transition if a backend exists."""
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
        # A dry run computes and changes nothing, so it needs no actor: there is no event to
        # attribute. The mutating path refuses without one.
        return
    if not actor.strip():
        typer.echo("--actor is required: every retained transition event names who caused it")
        raise typer.Exit(code=2)
    store = _configured_store("promote")
    try:
        record = store.promote_walk_forward(model_id, actor=actor, evidence_path=evidence)
    except (RegistryStoreError, ValueError) as error:
        typer.echo(str(error))
        raise typer.Exit(code=2) from error
    typer.echo(
        json.dumps(
            {
                "actor": actor,
                "dry_run": False,
                "evidence_hash": evidence_hash,
                "model_id": model_id,
                "status": record.status.value,
                "to": target,
                "transition_events": len(store.registry.history(model_id)),
            },
            sort_keys=True,
        )
    )


@pointer_app.command("plan")
def pointer_plan(
    consumer: str = typer.Option(..., "--consumer"),
    model_id: str = typer.Option(..., "--model-id"),
) -> None:
    """Print the checks a pointer backend must satisfy without changing any pointer."""
    try:
        _validate_consumer(consumer)
        validate_model_id(model_id)
    except ValueError as error:
        typer.echo(str(error))
        raise typer.Exit(code=2) from error
    typer.echo(
        json.dumps(
            {
                "checks": {
                    "artifact": "BACKEND_UNAVAILABLE",
                    "compatibility_export": "BACKEND_UNAVAILABLE",
                    "latest_prediction": "BACKEND_UNAVAILABLE",
                    "model_status": "BACKEND_UNAVAILABLE",
                    "schema": "BACKEND_UNAVAILABLE",
                },
                "consumer": consumer,
                "dry_run": True,
                "model_id": model_id,
            },
            sort_keys=True,
        )
    )


@pointer_app.command("switch")
def pointer_switch(
    consumer: str = typer.Option(..., "--consumer"),
    environment: str = typer.Option(..., "--environment"),
    model_id: str = typer.Option(..., "--model-id"),
    expected_current: str | None = typer.Option(None, "--expected-current"),
    expected_version: int | None = typer.Option(None, "--expected-version"),
    expected_pointer_version: int | None = typer.Option(None, "--expected-pointer-version"),
    expect_absent: bool = typer.Option(False, "--expect-absent"),
    reason: str = typer.Option(..., "--reason"),
    evidence: str = typer.Option(..., "--evidence"),
    run_id: str = typer.Option(..., "--run-id"),
    confirm_production: bool = typer.Option(False, "--confirm-production"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Validate a switch request, then refuse mutation until a pointer backend is configured."""
    _validate_pointer_request(
        consumer=consumer,
        environment=environment,
        model_id=model_id,
        expected_current=expected_current,
        expected_version=expected_version,
        expected_pointer_version=expected_pointer_version,
        expect_absent=expect_absent,
        reason=reason,
        evidence=evidence,
        run_id=run_id,
        confirm_production=confirm_production,
    )
    if dry_run:
        typer.echo(
            json.dumps(
                {
                    "consumer": consumer,
                    "dry_run": True,
                    "environment": environment,
                    "expected_current": expected_current,
                    "expected_pointer_version": expected_pointer_version,
                    "expected_version": expected_version,
                    "model_id": model_id,
                },
                sort_keys=True,
            )
        )
        return
    typer.echo("registry pointer backend is not configured; refusing to switch")
    raise typer.Exit(code=2)


@pointer_app.command("rollback")
def pointer_rollback(
    consumer: str = typer.Option(..., "--consumer"),
    environment: str = typer.Option(..., "--environment"),
    event_id: str = typer.Option(..., "--event-id"),
    expected_current: str = typer.Option(..., "--expected-current"),
    expected_version: int = typer.Option(..., "--expected-version"),
    expected_pointer_version: int = typer.Option(..., "--expected-pointer-version"),
    reason: str = typer.Option(..., "--reason"),
    evidence: str = typer.Option(..., "--evidence"),
    run_id: str = typer.Option(..., "--run-id"),
    confirm_production: bool = typer.Option(False, "--confirm-production"),
) -> None:
    """Validate a rollback request, then refuse mutation until a pointer backend is configured."""
    try:
        _validate_uuid(event_id, "event_id")
    except ValueError as error:
        typer.echo(str(error))
        raise typer.Exit(code=2) from error
    _validate_pointer_request(
        consumer=consumer,
        environment=environment,
        model_id=expected_current,
        expected_current=expected_current,
        expected_version=expected_version,
        expected_pointer_version=expected_pointer_version,
        expect_absent=False,
        reason=reason,
        evidence=evidence,
        run_id=run_id,
        confirm_production=confirm_production,
    )
    typer.echo("registry pointer backend is not configured; refusing to rollback")
    raise typer.Exit(code=2)


def _validate_pointer_request(
    *,
    consumer: str,
    environment: str,
    model_id: str,
    expected_current: str | None,
    expected_version: int | None,
    expected_pointer_version: int | None,
    expect_absent: bool,
    reason: str,
    evidence: str,
    run_id: str,
    confirm_production: bool,
) -> None:
    try:
        _validate_consumer(consumer)
        if environment not in {"shadow", "production"}:
            raise ValueError("pointer environment must be shadow or production")
        validate_model_id(model_id)
        if expect_absent:
            if any(value is not None for value in (expected_current, expected_version)):
                raise ValueError(
                    "--expect-absent cannot be combined with expected current pointer fields"
                )
            if expected_pointer_version is not None:
                raise ValueError("--expect-absent cannot include --expected-pointer-version")
        else:
            if expected_current is None:
                raise ValueError("pointer changes require --expected-current or --expect-absent")
            validate_model_id(expected_current)
            if isinstance(expected_version, bool) or not isinstance(expected_version, int):
                raise ValueError("pointer changes require --expected-version")
            if expected_version < 1:
                raise ValueError("pointer expected_version must be positive")
            if isinstance(expected_pointer_version, bool) or not isinstance(
                expected_pointer_version, int
            ):
                raise ValueError("pointer changes require --expected-pointer-version")
            if expected_pointer_version < 1:
                raise ValueError("pointer expected_pointer_version must be positive")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("pointer reason cannot be empty")
        if not isinstance(evidence, str) or not _EVIDENCE_ID.fullmatch(evidence):
            raise ValueError("pointer evidence must be a SHA-256 digest")
        _validate_uuid(run_id, "run_id")
        if environment == "production" and not confirm_production:
            raise ValueError("production pointer changes require --confirm-production")
    except ValueError as error:
        typer.echo(str(error))
        raise typer.Exit(code=2) from error


def _validate_consumer(value: str) -> None:
    if not isinstance(value, str) or not _CONSUMER_ID.fullmatch(value):
        raise ValueError("pointer consumer is invalid")


def _validate_uuid(value: str, name: str) -> None:
    try:
        valid = str(uuid.UUID(value)) == value
    except (AttributeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"pointer {name} is invalid")
