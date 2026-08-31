"""Fail-closed daily inference CLI plans."""

from __future__ import annotations

import json
import uuid
from collections.abc import Mapping
from datetime import date
from pathlib import Path

import typer
import yaml

from research_v2.registry.model_registry import validate_model_id

_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "inference" / "daily.yaml"
_ALLOWED_MODES = {"shadow", "production"}
_ALLOWED_FROM_STEP = "watermark"


def daily(signal_date: str, model_id: str, mode: str, dry_run: bool) -> None:
    """Validate a daily inference request and refuse live execution without a backend."""
    config = _load_config()
    try:
        parsed_date = date.fromisoformat(signal_date)
        validate_model_id(model_id)
        _validate_mode(mode)
    except ValueError as error:
        if isinstance(error, ValueError) and str(error) == "day is out of range for month":
            typer.echo(f"signal date is invalid: {error}")
        else:
            typer.echo(str(error))
        raise typer.Exit(code=2) from error

    plan = {
        "dry_run": dry_run,
        "mode": mode,
        "model_id": model_id,
        "retry_policy": {
            "max_retries": _require_int(config, "max_retries"),
            "retry_times": _require_list_of_strings(config, "retry_times"),
        },
        "schedule": {
            "slo_time": _require_str(config, "slo_time"),
            "timezone": _require_str(config, "timezone"),
            "trigger_time": _require_str(config, "trigger_time"),
        },
        "signal_date": parsed_date.isoformat(),
        "target_database": _require_str(config, "target_database"),
        "target_environment": _require_str(config, "target_environment"),
        "write": False,
    }
    if mode == "production":
        plan["active_pointer_required"] = True
        plan["runtime_requirements"] = ["active-pointer", "PRODUCTION"]
    typer.echo(json.dumps(plan, ensure_ascii=False, sort_keys=True))
    if dry_run:
        return
    if mode == "production":
        typer.echo(
            "production requires active-pointer and PRODUCTION runtime backend; "
            "refusing to run daily inference"
        )
    else:
        typer.echo("runtime backend is not configured; refusing to run daily inference")
    raise typer.Exit(code=2)


def status(signal_date: str, model_id: str) -> None:
    """Report the read-only status view without mutating any state."""
    config = _load_config()
    try:
        parsed_date = date.fromisoformat(signal_date)
        validate_model_id(model_id)
    except ValueError as error:
        typer.echo(str(error))
        raise typer.Exit(code=2) from error
    typer.echo(
        json.dumps(
            {
                "backend_status": "BACKEND_UNAVAILABLE",
                "model_id": model_id,
                "read_only": True,
                "signal_date": parsed_date.isoformat(),
                "target_database": _require_str(config, "target_database"),
                "target_environment": _require_str(config, "target_environment"),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def retry(run_id: str, from_step: str, dry_run: bool) -> None:
    """Create a fresh retry plan and refuse live execution without a backend."""
    config = _load_config()
    try:
        _validate_run_id(run_id)
        _validate_from_step(from_step)
    except ValueError as error:
        typer.echo(str(error))
        raise typer.Exit(code=2) from error

    retry_run_id = str(uuid.uuid4())
    plan = {
        "dry_run": dry_run,
        "from_step": from_step,
        "original_run_id": run_id,
        "retry_policy": {
            "max_retries": _require_int(config, "max_retries"),
            "retry_times": _require_list_of_strings(config, "retry_times"),
        },
        "retry_run_id": retry_run_id,
        "target_database": _require_str(config, "target_database"),
        "target_environment": _require_str(config, "target_environment"),
        "write": False,
    }
    typer.echo(json.dumps(plan, ensure_ascii=False, sort_keys=True))
    if dry_run:
        return
    typer.echo("runtime backend is not configured; refusing to retry daily inference")
    raise typer.Exit(code=2)


def _load_config() -> dict[str, object]:
    try:
        raw = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"unable to load configuration file: {_CONFIG_PATH}") from error
    if not isinstance(raw, Mapping):
        raise ValueError("daily inference config must contain a mapping")
    return dict(raw)


def _require_str(config: Mapping[str, object], key: str) -> str:
    value = config.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"daily inference config requires {key}")
    return value


def _require_int(config: Mapping[str, object], key: str) -> int:
    value = config.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"daily inference config requires {key}")
    return value


def _require_list_of_strings(config: Mapping[str, object], key: str) -> list[str]:
    value = config.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) and item for item in value):
        raise ValueError(f"daily inference config requires {key}")
    return list(value)


def _validate_mode(mode: str) -> None:
    if mode not in _ALLOWED_MODES:
        raise ValueError("mode must be shadow or production")


def _validate_from_step(value: str) -> None:
    if value != _ALLOWED_FROM_STEP:
        raise ValueError("retry can only start from watermark")


def _validate_run_id(value: str) -> None:
    try:
        valid = str(uuid.UUID(value)) == value
    except (AttributeError, ValueError):
        valid = False
    if not valid:
        raise ValueError("run_id must be a valid UUID")
