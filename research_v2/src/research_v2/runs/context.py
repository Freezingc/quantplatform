"""Run identity and content-addressed data versions."""

from __future__ import annotations

import hashlib
import json
import math
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path


def _canonicalize(value: object) -> object:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("configuration contains a non-finite float")
        return value
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("configuration mapping keys must be strings")
        return {
            key: _canonicalize(value[key])
            for key in sorted(value)
        }
    if isinstance(value, (list, tuple)):
        return [_canonicalize(item) for item in value]
    if isinstance(value, (frozenset, set)):
        canonical_items = [_canonicalize(item) for item in value]
        return sorted(canonical_items, key=_canonical_json)
    raise ValueError(f"unsupported configuration value type: {type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


@dataclass(frozen=True)
class RunContext:
    run_id: str
    data_version: str

    @classmethod
    def build(
        cls,
        *,
        config: Mapping[str, object],
        watermarks: Iterable[str],
        code_commit: str,
        schema_version: str,
    ) -> RunContext:
        if not code_commit:
            raise ValueError("code_commit cannot be empty")
        if not schema_version:
            raise ValueError("schema_version cannot be empty")
        payload = {
            "code_commit": code_commit,
            "config": _canonicalize(config),
            "schema_version": schema_version,
            "watermarks": sorted(set(watermarks)),
        }
        data_version = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
        return cls(run_id=str(uuid.uuid4()), data_version=data_version)
