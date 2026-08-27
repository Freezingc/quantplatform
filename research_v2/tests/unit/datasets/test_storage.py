from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.datasets.manifest import build_dataset_manifest
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
