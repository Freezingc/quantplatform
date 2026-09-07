from __future__ import annotations

import json
from datetime import date

import pandas as pd
from typer.testing import CliRunner

from research_v2.cli.app import app
from research_v2.datasets.manifest import build_dataset_manifest
from research_v2.datasets.storage import DatasetArtifactStore


def test_dataset_inspect_and_verify_read_immutable_artifact(
    tmp_path: object, monkeypatch: object
) -> None:
    root = tmp_path / "artifacts"  # type: ignore[operator]
    frame = pd.DataFrame(
        {"signal_date": [date(2025, 1, 2)], "ts_code": ["a"], "f1": [1.0]}
    )
    manifest = build_dataset_manifest(
        frame,
        business_keys=("signal_date", "ts_code"),
        metadata={"fold_id": "F1"},
    )
    DatasetArtifactStore(root).write(manifest, {"train": frame}, preprocessing={})
    monkeypatch.setenv("QRV2_ARTIFACT_ROOT", str(root))  # type: ignore[attr-defined]
    runner = CliRunner()

    inspected = runner.invoke(app, ["dataset", "inspect", "--dataset-id", manifest.dataset_id])
    verified = runner.invoke(app, ["dataset", "verify", "--dataset-id", manifest.dataset_id])

    assert inspected.exit_code == 0
    assert json.loads(inspected.stdout)["dataset_id"] == manifest.dataset_id
    assert verified.exit_code == 0
    assert json.loads(verified.stdout) == {"dataset_id": manifest.dataset_id, "valid": True}


def test_dataset_build_rejects_locked_final_test_before_runtime_access() -> None:
    result = CliRunner().invoke(
        app,
        ["dataset", "build", "--fold", "FINAL", "--include-test"],
    )

    assert result.exit_code != 0
    assert "locked" in result.stdout.lower()
