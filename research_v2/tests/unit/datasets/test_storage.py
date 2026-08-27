from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from research_v2.datasets.manifest import (
    build_dataset_manifest,
    build_split_dataset_manifest,
)
from research_v2.datasets.storage import DatasetArtifactStore


def test_artifact_store_writes_atomically_and_verifies_shards(tmp_path: object) -> None:
    root = tmp_path / "datasets"  # type: ignore[operator]
    frame = pd.DataFrame(
        {
            "signal_date": [date(2025, 1, 3), date(2025, 1, 2)],
            "ts_code": ["b", "a"],
            "f1": [2.0, 1.0],
        }
    )
    manifest = build_dataset_manifest(
        frame,
        business_keys=("signal_date", "ts_code"),
        metadata={"fold_id": "F01"},
    )
    store = DatasetArtifactStore(root)

    artifact = store.write(manifest, {"train": frame}, preprocessing={"state_hash": "abc"})

    assert artifact.path == root / manifest.dataset_id
    assert (artifact.path / "manifest.json").is_file()
    assert store.verify(manifest.dataset_id)
    assert len(artifact.shard_hashes) == 1
    with pytest.raises(FileExistsError):
        store.write(manifest, {"train": frame}, preprocessing={"state_hash": "abc"})


def test_artifact_store_rejects_splits_that_do_not_match_manifest(tmp_path: object) -> None:
    root = tmp_path / "datasets"  # type: ignore[operator]
    frame = pd.DataFrame({"signal_date": [date(2025, 1, 2)], "ts_code": ["a"], "f1": [1.0]})
    manifest = build_dataset_manifest(
        frame,
        business_keys=("signal_date", "ts_code"),
        metadata={"fold_id": "F01"},
    )
    changed = frame.copy()
    changed.loc[0, "f1"] = 99.0

    with pytest.raises(ValueError, match="manifest"):
        DatasetArtifactStore(root).write(manifest, {"train": changed}, preprocessing={})
    assert not (root / manifest.dataset_id).exists()


def test_artifact_store_recomputes_manifest_identity_during_verify(tmp_path: object) -> None:
    root = tmp_path / "datasets"  # type: ignore[operator]
    frame = pd.DataFrame({"signal_date": [date(2025, 1, 2)], "ts_code": ["a"], "f1": [1.0]})
    manifest = build_dataset_manifest(
        frame,
        business_keys=("signal_date", "ts_code"),
        metadata={"fold_id": "F01"},
    )
    store = DatasetArtifactStore(root)
    artifact = store.write(manifest, {"train": frame}, preprocessing={})
    shard_path = next((artifact.path / "split=train").rglob("*.parquet"))
    changed = frame.copy()
    changed.loc[0, "f1"] = 99.0
    changed.to_parquet(shard_path, engine="pyarrow", compression="zstd", index=False)

    manifest_path = artifact.path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_path = next(iter(payload["shard_hashes"]))
    payload["shard_hashes"][relative_path] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    assert not store.verify(manifest.dataset_id)


def test_artifact_store_verification_fails_closed_for_invalid_parquet(tmp_path: object) -> None:
    root = tmp_path / "datasets"  # type: ignore[operator]
    frame = pd.DataFrame({"signal_date": [date(2025, 1, 2)], "ts_code": ["a"], "f1": [1.0]})
    manifest = build_dataset_manifest(
        frame,
        business_keys=("signal_date", "ts_code"),
        metadata={"fold_id": "F01"},
    )
    store = DatasetArtifactStore(root)
    artifact = store.write(manifest, {"train": frame}, preprocessing={})
    shard_path = next((artifact.path / "split=train").rglob("*.parquet"))
    shard_path.write_bytes(b"not a parquet file")

    manifest_path = artifact.path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    relative_path = next(iter(payload["shard_hashes"]))
    payload["shard_hashes"][relative_path] = hashlib.sha256(shard_path.read_bytes()).hexdigest()
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    assert not store.verify(manifest.dataset_id)


def test_artifact_store_requires_split_bound_manifest_for_multiple_splits(
    tmp_path: object,
) -> None:
    root = tmp_path / "datasets"  # type: ignore[operator]
    train = pd.DataFrame({"signal_date": [date(2025, 1, 2)], "ts_code": ["a"], "f1": [1.0]})
    test = pd.DataFrame({"signal_date": [date(2025, 1, 3)], "ts_code": ["b"], "f1": [2.0]})
    manifest = build_dataset_manifest(
        pd.concat([train, test], ignore_index=True),
        business_keys=("signal_date", "ts_code"),
        metadata={"fold_id": "F01"},
    )

    with pytest.raises(ValueError, match="split"):
        DatasetArtifactStore(root).write(
            manifest,
            {"train": train, "test": test},
            preprocessing={},
        )


def test_artifact_store_detects_swapped_split_shards(tmp_path: object) -> None:
    root = tmp_path / "datasets"  # type: ignore[operator]
    train = pd.DataFrame({"signal_date": [date(2025, 1, 2)], "ts_code": ["a"], "f1": [1.0]})
    test = pd.DataFrame({"signal_date": [date(2025, 1, 3)], "ts_code": ["b"], "f1": [2.0]})
    manifest = build_split_dataset_manifest(
        {"train": train, "test": test},
        business_keys=("signal_date", "ts_code"),
        metadata={"fold_id": "F01"},
    )
    store = DatasetArtifactStore(root)
    artifact = store.write(
        manifest,
        {"train": train, "test": test},
        preprocessing={},
    )
    assert store.verify(manifest.dataset_id)
    train_path = next((artifact.path / "split=train").rglob("*.parquet"))
    test_path = next((artifact.path / "split=test").rglob("*.parquet"))
    train_frame = pd.read_parquet(train_path)
    test_frame = pd.read_parquet(test_path)
    train_frame.to_parquet(test_path, engine="pyarrow", compression="zstd", index=False)
    test_frame.to_parquet(train_path, engine="pyarrow", compression="zstd", index=False)

    manifest_path = artifact.path / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["shard_hashes"] = {
        relative_path: hashlib.sha256((artifact.path / relative_path).read_bytes()).hexdigest()
        for relative_path in payload["shard_hashes"]
    }
    manifest_path.write_text(json.dumps(payload), encoding="utf-8")

    assert not store.verify(manifest.dataset_id)


def test_load_split_returns_a_hash_bound_snapshot_when_a_shard_changes_during_read(
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = tmp_path / "datasets"  # type: ignore[operator]
    frame = pd.DataFrame({"signal_date": [date(2025, 1, 2)], "ts_code": ["a"], "f1": [1.0]})
    manifest = build_dataset_manifest(
        frame,
        business_keys=("signal_date", "ts_code"),
        metadata={"fold_id": "F01"},
    )
    store = DatasetArtifactStore(root)
    artifact = store.write(manifest, {"train": frame}, preprocessing={})
    shard_path = next((artifact.path / "split=train").rglob("*.parquet")).resolve()
    changed = frame.copy()
    changed.loc[0, "f1"] = 99.0
    original_read_parquet = pd.read_parquet
    mutated = False

    def read_parquet_then_mutate(source: object, *args: object, **kwargs: object) -> pd.DataFrame:
        nonlocal mutated
        loaded = original_read_parquet(source, *args, **kwargs)
        if not mutated and isinstance(source, Path) and source.resolve() == shard_path:
            changed.to_parquet(shard_path, engine="pyarrow", compression="zstd", index=False)
            mutated = True
        return loaded

    monkeypatch.setattr(pd, "read_parquet", read_parquet_then_mutate)

    loaded = store.load_split(manifest.dataset_id, "train", expected_fold_id="F01")

    assert loaded.loc[0, "f1"] == 1.0
    assert mutated is False
