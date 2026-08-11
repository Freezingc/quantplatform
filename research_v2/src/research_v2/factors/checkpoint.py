"""Immutable, content-addressed checkpoints for factor-materialization groups."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class StoredFactorCheckpoint:
    checkpoint_id: str
    path: Path


class FactorCheckpointStore:
    """Persist resumable clean/quarantine factor groups without overwriting them."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def write(
        self,
        *,
        identity: Mapping[str, object],
        clean: pd.DataFrame,
        quarantine: pd.DataFrame,
    ) -> StoredFactorCheckpoint:
        payload = json.dumps(identity, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        checkpoint_id = hashlib.sha256(payload.encode()).hexdigest()
        destination = self._root / checkpoint_id
        if destination.exists():
            raise FileExistsError(f"factor checkpoint already exists: {checkpoint_id}")
        self._root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".factor-", dir=self._root))
        try:
            clean_path = temporary / "clean.parquet"
            quarantine_path = temporary / "quarantine.parquet"
            clean.to_parquet(clean_path, engine="pyarrow", compression="zstd", index=False)
            quarantine.to_parquet(
                quarantine_path,
                engine="pyarrow",
                compression="zstd",
                index=False,
            )
            manifest = {
                "checkpoint_id": checkpoint_id,
                "identity": identity,
                "hashes": {
                    name: _hash(temporary / name)
                    for name in ("clean.parquet", "quarantine.parquet")
                },
            }
            (temporary / "manifest.json").write_text(
                json.dumps(manifest, sort_keys=True),
                encoding="utf-8",
            )
            temporary.rename(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return StoredFactorCheckpoint(checkpoint_id=checkpoint_id, path=destination)

    def verify(self, checkpoint_id: str) -> bool:
        try:
            root = self._root / checkpoint_id
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
            return manifest["checkpoint_id"] == checkpoint_id and all(
                _hash(root / name) == value for name, value in manifest["hashes"].items()
            )
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return False


def _hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
