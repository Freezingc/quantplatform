"""Configuration loading entry points."""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Literal, cast

import yaml

from research_v2.config.models import AppSettings


def _load_yaml_mapping(path: Path) -> dict[str, object]:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise ValueError(f"unable to load configuration file: {path}") from error

    if raw is None:
        return {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"configuration file must contain a mapping: {path}")
    if not all(isinstance(key, str) for key in raw):
        raise ValueError(f"configuration keys must be strings: {path}")
    return dict(cast(Mapping[str, object], raw))


def _merge_mappings(
    base: Mapping[str, object], override: Mapping[str, object]
) -> dict[str, object]:
    merged = dict(base)
    for key, value in override.items():
        previous = merged.get(key)
        if isinstance(previous, Mapping) and isinstance(value, Mapping):
            merged[key] = _merge_mappings(
                cast(Mapping[str, object], previous), cast(Mapping[str, object], value)
            )
        else:
            merged[key] = value
    return merged


def _reject_plaintext_secrets(value: object) -> None:
    if not isinstance(value, Mapping):
        if isinstance(value, list):
            for item in value:
                _reject_plaintext_secrets(item)
        return

    for key, nested_value in value.items():
        if not isinstance(key, str):
            raise ValueError("configuration keys must be strings")
        if key in {"password", "secret", "token"} or key.endswith("_password"):
            raise ValueError("secret values must be supplied through environment variables")
        _reject_plaintext_secrets(nested_value)


def _resolve_environment_value(config: dict[str, object], field_name: str) -> None:
    environment_key = f"{field_name}_env"
    configured_environment = config.pop(environment_key, None)
    if configured_environment is None:
        return
    if field_name in config:
        raise ValueError(f"{field_name} and {environment_key} cannot both be set")
    if not isinstance(configured_environment, str) or not configured_environment:
        raise ValueError(f"{environment_key} must be a non-empty environment variable name")
    resolved = os.getenv(configured_environment)
    if not resolved:
        raise ValueError(f"secret environment variable is missing: {configured_environment}")
    config[field_name] = resolved


def load_settings(
    environment: Literal["dev", "test", "production"], *, config_root: Path
) -> AppSettings:
    """Load one approved environment without accepting plaintext secrets."""
    base = _load_yaml_mapping(config_root / "base.yaml")
    selected = _load_yaml_mapping(config_root / f"{environment}.yaml")
    _reject_plaintext_secrets(base)
    _reject_plaintext_secrets(selected)

    merged = _merge_mappings(base, selected)
    configured_environment = merged.get("environment")
    if configured_environment not in {None, environment}:
        raise ValueError("configuration environment does not match requested environment")
    merged["environment"] = environment

    raw_database = merged.get("dolphindb")
    if not isinstance(raw_database, Mapping):
        raise ValueError("dolphindb configuration must be a mapping")
    database = dict(cast(Mapping[str, object], raw_database))
    for field_name in ("host", "username", "password"):
        _resolve_environment_value(database, field_name)
    merged["dolphindb"] = database

    return AppSettings.model_validate(merged)
