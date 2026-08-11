"""Immutable, integrity-checked model and optimizer checkpoint artifacts."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import torch
from torch import nn

_CHECKPOINT_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class CheckpointArtifact:
    checkpoint_id: str
    path: Path
    sha256: str


class CheckpointStore:
    """Atomically publish immutable checkpoint directories under one artifact root."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def save(
        self,
        checkpoint_id: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        *,
        metadata: Mapping[str, object],
    ) -> CheckpointArtifact:
        self._validate_id(checkpoint_id)
        destination = self._root / checkpoint_id
        if destination.exists():
            raise FileExistsError(f"checkpoint is immutable and already exists: {checkpoint_id}")
        self._root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".checkpoint-", dir=self._root))
        checkpoint_path = temporary / "checkpoint.pt"
        try:
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                checkpoint_path,
            )
            digest = _sha256_file(checkpoint_path)
            manifest = {
                "checkpoint_id": checkpoint_id,
                "metadata": _json_compatible(metadata),
                "sha256": digest,
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            if destination.exists():
                raise FileExistsError(
                    f"checkpoint is immutable and already exists: {checkpoint_id}"
                )
            temporary.rename(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return CheckpointArtifact(checkpoint_id, destination / "checkpoint.pt", digest)

    def verify(self, checkpoint_id: str) -> bool:
        try:
            manifest = self._read_manifest(checkpoint_id)
            path = self._root / checkpoint_id / "checkpoint.pt"
            return path.is_file() and _sha256_file(path) == manifest["sha256"]
        except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
            return False

    def load(
        self,
        checkpoint_id: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> dict[str, object]:
        if not self.verify(checkpoint_id):
            raise ValueError(f"checkpoint integrity verification failed: {checkpoint_id}")
        checkpoint_path = self._root / checkpoint_id / "checkpoint.pt"
        loaded = cast(
            dict[str, object],
            torch.load(checkpoint_path, map_location="cpu", weights_only=True),
        )
        model_state = loaded.get("model_state_dict")
        optimizer_state = loaded.get("optimizer_state_dict")
        if not isinstance(model_state, dict) or not isinstance(optimizer_state, dict):
            raise ValueError("checkpoint state dictionaries are malformed")
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)
        return self._read_manifest(checkpoint_id)

    def _read_manifest(self, checkpoint_id: str) -> dict[str, object]:
        self._validate_id(checkpoint_id)
        path = self._root / checkpoint_id / "manifest.json"
        if not path.is_file():
            raise ValueError(f"checkpoint manifest not found: {checkpoint_id}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict) or manifest.get("checkpoint_id") != checkpoint_id:
            raise ValueError("checkpoint manifest is malformed")
        if not isinstance(manifest.get("sha256"), str):
            raise ValueError("checkpoint manifest is missing its SHA-256")
        return cast(dict[str, object], manifest)

    @staticmethod
    def _validate_id(checkpoint_id: str) -> None:
        if not _CHECKPOINT_ID.fullmatch(checkpoint_id):
            raise ValueError("checkpoint_id contains unsupported characters")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_compatible(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("checkpoint metadata keys must be strings")
        return {key: _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    raise ValueError(f"unsupported checkpoint metadata type: {type(value).__name__}")
