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

from research_v2.datasets.manifest import (
    DatasetManifest,
    build_dataset_manifest,
    build_split_dataset_manifest,
    validate_dataset_splits,
)


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
        _validate_manifest_binding(manifest, splits)
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
                    "split_content_hashes": dict(manifest.split_content_hashes),
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
            if not isinstance(manifest, dict):
                return False
            if manifest.get("dataset_id") != dataset_id:
                return False
            hashes = manifest.get("shard_hashes")
            if not isinstance(hashes, dict):
                return False
            business_keys = manifest.get("business_keys")
            column_order = manifest.get("column_order")
            metadata = manifest.get("metadata")
            row_count = manifest.get("row_count")
            content_hash = manifest.get("content_hash")
            split_content_hashes = manifest.get("split_content_hashes", {})
            if (
                not isinstance(business_keys, list)
                or not all(isinstance(key, str) for key in business_keys)
                or not isinstance(column_order, list)
                or not all(isinstance(column, str) for column in column_order)
                or not isinstance(metadata, Mapping)
                or not isinstance(row_count, int)
                or isinstance(row_count, bool)
                or row_count < 1
                or not isinstance(content_hash, str)
                or not isinstance(split_content_hashes, dict)
                or not all(
                    isinstance(name, str) and isinstance(value, str)
                    for name, value in split_content_hashes.items()
                )
            ):
                return False
            artifact_root = manifest_path.parent.resolve()
            shard_frames: list[pd.DataFrame] = []
            split_shards: dict[str, list[pd.DataFrame]] = {}
            for relative_path, expected_hash in hashes.items():
                if not isinstance(relative_path, str) or not isinstance(expected_hash, str):
                    return False
                shard_path = (artifact_root / relative_path).resolve()
                if artifact_root not in shard_path.parents or not shard_path.is_file():
                    return False
                if _sha256_file(shard_path) != expected_hash:
                    return False
                frame = pd.read_parquet(shard_path, engine="pyarrow")
                shard_frames.append(frame)
                parts = Path(relative_path).parts
                if len(parts) != 3 or not parts[0].startswith("split="):
                    return False
                split_name = parts[0][len("split=") :]
                if split_name not in {"train", "validation", "test"}:
                    return False
                split_shards.setdefault(split_name, []).append(frame)
            if not shard_frames:
                return False
            loaded_splits = {
                split_name: pd.concat(frames, ignore_index=True)
                for split_name, frames in split_shards.items()
            }
            combined = pd.concat(shard_frames, ignore_index=True)
            if split_content_hashes:
                if set(split_content_hashes) != set(loaded_splits):
                    return False
                recomputed = build_split_dataset_manifest(
                    loaded_splits,
                    business_keys=tuple(business_keys),
                    metadata=metadata,
                )
            else:
                if len(loaded_splits) != 1:
                    return False
                recomputed = build_dataset_manifest(
                    combined,
                    business_keys=tuple(business_keys),
                    metadata=metadata,
                )
            return (
                recomputed.dataset_id == dataset_id
                and recomputed.content_hash == content_hash
                and recomputed.row_count == row_count
                and list(recomputed.column_order) == column_order
                and dict(recomputed.split_content_hashes) == split_content_hashes
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError, KeyError, IndexError):
            return False

    def _write_splits(
        self,
        temporary: Path,
        splits: Mapping[str, pd.DataFrame],
    ) -> dict[str, str]:
        shard_hashes: dict[str, str] = {}
        for split_name in sorted(splits):
            frame = splits[split_name]
            ordered = frame.copy(deep=True)
            sort_keys = pd.DataFrame(
                {
                    "signal_date": pd.to_datetime(ordered["signal_date"], errors="raise"),
                    "ts_code": ordered["ts_code"].astype(str),
                    "row_position": range(len(ordered)),
                },
            )
            row_positions = sort_keys.sort_values(["signal_date", "ts_code"], kind="mergesort")[
                "row_position"
            ].to_numpy()
            ordered = ordered.iloc[row_positions].reset_index(drop=True)
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


def _validate_manifest_binding(
    manifest: DatasetManifest,
    splits: Mapping[str, pd.DataFrame],
) -> None:
    """Require the supplied shards to describe exactly the published manifest."""
    frames = []
    for frame in validate_dataset_splits(splits):
        if "signal_date" not in frame.columns or "ts_code" not in frame.columns:
            raise ValueError("dataset split requires signal_date and ts_code")
        frames.append(frame)
    combined = pd.concat(frames, ignore_index=True)
    if manifest.split_content_hashes:
        recomputed = build_split_dataset_manifest(
            splits,
            business_keys=manifest.business_keys,
            metadata=manifest.metadata,
        )
    else:
        if len(splits) != 1:
            raise ValueError("multi-split artifacts require a split-bound manifest")
        recomputed = build_dataset_manifest(
            combined,
            business_keys=manifest.business_keys,
            metadata=manifest.metadata,
        )
    if (
        recomputed.dataset_id != manifest.dataset_id
        or recomputed.content_hash != manifest.content_hash
        or recomputed.row_count != manifest.row_count
        or recomputed.business_keys != manifest.business_keys
        or recomputed.column_order != manifest.column_order
        or recomputed.split_content_hashes != manifest.split_content_hashes
    ):
        raise ValueError("dataset splits do not match manifest")


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
