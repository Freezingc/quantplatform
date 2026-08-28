from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
from typer.testing import CliRunner

from research_v2.cli.app import app
from research_v2.export.compatibility import export_compatibility_predictions

_MODEL_ID = "sha256:" + "a" * 64


def test_export_verify_command_reports_a_valid_manifest(tmp_path: Path) -> None:
    artifact = export_compatibility_predictions(
        pd.DataFrame(
            {
                "signal_date": [date(2026, 1, 2)],
                "ts_code": ["000001.SZ"],
                "score": [0.1],
            }
        ),
        model_id=_MODEL_ID,
        model_version=1,
        signal_date=date(2026, 1, 2),
        source_prediction_data_version="prediction-v1",
        artifact_root=tmp_path / "artifacts",
        legacy_output_roots=(tmp_path / "legacy-v1-output",),
    )

    result = CliRunner().invoke(
        app, ["export", "verify", "--manifest", str(artifact.manifest_path)]
    )

    assert result.exit_code == 0
    assert '"valid": true' in result.stdout


def test_export_compatibility_command_fails_closed_without_a_runtime_backend() -> None:
    result = CliRunner().invoke(
        app,
        [
            "export",
            "compatibility",
            "--signal-date",
            "2026-01-02",
            "--model-id",
            _MODEL_ID,
            "--format",
            "csv,pkl",
        ],
    )

    assert result.exit_code == 2
    assert "runtime backend is not configured" in result.stdout
