"""Atomic, immutable parquet storage for content-addressed dataset artifacts."""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from research_v2.datasets.manifest import DatasetManifest


@dataclass(frozen=True)
class StoredDatasetArtifact:
    path: Path
    shard_hashes: Mapping[str, str]


class DatasetArtifactStore:
    """Persist artifacts under one root without allowing published IDs to be overwritten."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def write(
        self,
        manifest: DatasetManifest,
        splits: Mapping[str, pd.DataFrame],
        *,
        preprocessing: Mapping[str, object],
    ) -> StoredDatasetArtifact:
        if not splits:
            raise ValueError("at least one split is required")
        destination = self._root / manifest.dataset_id
        if destination.exists():
            raise FileExistsError(
                f"dataset artifact is immutable and already exists: {manifest.dataset_id}"
            )
        self._root.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=".dataset-", dir=self._root))
        try:
            shard_hashes = self._write_splits(temporary, splits)
            self._write_json(
                temporary / "preprocessing.json",
                {"preprocessing": _json_compatible(preprocessing)},
            )
            self._write_json(
                temporary / "manifest.json",
                {
                    "business_keys": list(manifest.business_keys),
                    "column_order": list(manifest.column_order),
                    "content_hash": manifest.content_hash,
                    "dataset_id": manifest.dataset_id,
                    "metadata": _json_compatible(manifest.metadata),
                    "row_count": manifest.row_count,
                    "shard_hashes": shard_hashes,
                },
            )
            if destination.exists():
                raise FileExistsError(
                    f"dataset artifact is immutable and already exists: {manifest.dataset_id}"
                )
            temporary.rename(destination)
        except Exception:
            shutil.rmtree(temporary, ignore_errors=True)
            raise
        return StoredDatasetArtifact(path=destination, shard_hashes=shard_hashes)

    def verify(self, dataset_id: str) -> bool:
        manifest_path = self._root / dataset_id / "manifest.json"
        if not manifest_path.is_file():
            return False
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            if manifest.get("dataset_id") != dataset_id:
                return False
            hashes = manifest.get("shard_hashes")
            if not isinstance(hashes, dict):
                return False
            artifact_root = manifest_path.parent.resolve()
            for relative_path, expected_hash in hashes.items():
                if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
                    return False
                shard_path = (artifact_root / relative_path).resolve()
                if artifact_root not in shard_path.parents or not shard_path.is_file():
                    return False
                if _sha256_file(shard_path) != expected_hash:
                    return False
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return False
        return True

    def _write_splits(
        self,
        temporary: Path,
        splits: Mapping[str, pd.DataFrame],
    ) -> dict[str, str]:
        shard_hashes: dict[str, str] = {}
        for split_name in sorted(splits):
            if split_name not in {"train", "validation", "test"}:
                raise ValueError(f"unsupported dataset split: {split_name}")
            frame = splits[split_name]
            if frame.empty:
                raise ValueError(f"cannot write an empty dataset split: {split_name}")
            if "signal_date" not in frame.columns or "ts_code" not in frame.columns:
                raise ValueError("dataset split requires signal_date and ts_code")
            ordered = frame.copy(deep=True)
            ordered["signal_date"] = pd.to_datetime(ordered["signal_date"], errors="raise").dt.date
            ordered = ordered.sort_values(["signal_date", "ts_code"], kind="mergesort")
            years = pd.to_datetime(ordered["signal_date"], errors="raise").dt.year
            for year in sorted(years.unique()):
                partition = ordered.loc[years == year].reset_index(drop=True)
                relative_path = Path(f"split={split_name}") / f"year={year}" / "part-00000.parquet"
                shard_path = temporary / relative_path
                shard_path.parent.mkdir(parents=True, exist_ok=True)
                partition.to_parquet(shard_path, engine="pyarrow", compression="zstd", index=False)
                shard_hashes[relative_path.as_posix()] = _sha256_file(shard_path)
        return shard_hashes

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, object]) -> None:
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json_compatible(value: object) -> object:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    raise ValueError(f"artifact JSON contains unsupported type: {type(value).__name__}")
