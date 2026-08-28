from __future__ import annotations

import hashlib
import json
import os
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from research_v2.export.compatibility import (
    UnsafeExportPath,
    export_compatibility_predictions,
    load_compatibility_contract,
    verify_compatibility_export,
)

_MODEL_ID = "sha256:" + "a" * 64


def _predictions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": [date(2026, 1, 2), date(2026, 1, 2)],
            "ts_code": ["600000.SH", "000001.SZ"],
            "score": [0.9, 0.1],
        }
    )


def _legacy_output_roots(tmp_path: Path) -> tuple[Path, ...]:
    return (tmp_path / "legacy-v1-output",)


def test_packaged_contract_matches_the_approved_contract() -> None:
    project_root = Path(__file__).parents[3]
    approved = project_root / "configs" / "export" / "v1_compatible.yaml"
    packaged = project_root / "src" / "research_v2" / "export" / "v1_compatible.yaml"

    assert packaged.read_bytes() == approved.read_bytes()


def test_exporter_writes_deterministic_v1_compatibility_views(tmp_path: Path) -> None:
    result = export_compatibility_predictions(
        _predictions(),
        model_id=_MODEL_ID,
        model_version=1,
        signal_date=date(2026, 1, 2),
        source_prediction_data_version="prediction-v1",
        artifact_root=tmp_path / "artifacts",
        legacy_output_roots=_legacy_output_roots(tmp_path),
    )

    assert result.csv_path.read_text(encoding="utf-8") == (
        "ts_code,20260102\n000001.SZ,0.1000000000\n600000.SH,0.9000000000\n"
    )
    pkl = pd.read_pickle(result.pkl_path)
    assert pkl.to_dict(orient="records") == [
        {"instrument": "000001.SZ", "datetime": "2026-01-02", "score": 0.1},
        {"instrument": "600000.SH", "datetime": "2026-01-02", "score": 0.9},
    ]
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    contract = Path(__file__).parents[3] / "configs" / "export" / "v1_compatible.yaml"
    assert manifest["contract_hash"] == hashlib.sha256(contract.read_bytes()).hexdigest()
    assert result.manifest_path.parent.is_symlink()
    assert verify_compatibility_export(result.manifest_path) is True


def test_exporter_rejects_legacy_v1_output_roots_without_writing(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy-v1-output"
    legacy_root.mkdir()

    with pytest.raises(UnsafeExportPath):
        export_compatibility_predictions(
            _predictions(),
            model_id=_MODEL_ID,
            model_version=1,
            signal_date=date(2026, 1, 2),
            source_prediction_data_version="prediction-v1",
            artifact_root=legacy_root,
            legacy_output_roots=(legacy_root,),
        )

    assert list(legacy_root.iterdir()) == []


def test_exporter_requires_explicit_v1_output_roots(tmp_path: Path) -> None:
    with pytest.raises(UnsafeExportPath):
        export_compatibility_predictions(
            _predictions(),
            model_id=_MODEL_ID,
            model_version=1,
            signal_date=date(2026, 1, 2),
            source_prediction_data_version="prediction-v1",
            artifact_root=tmp_path / "artifacts",
        )


def test_exporter_does_not_touch_a_read_only_v1_sentinel(tmp_path: Path) -> None:
    legacy_root = tmp_path / "legacy-v1-output"
    legacy_root.mkdir()
    sentinel = legacy_root / "sentinel.txt"
    sentinel.write_text("do not modify", encoding="utf-8")
    legacy_root.chmod(0o555)
    try:
        result = export_compatibility_predictions(
            _predictions(),
            model_id=_MODEL_ID,
            model_version=1,
            signal_date=date(2026, 1, 2),
            source_prediction_data_version="prediction-v1",
            artifact_root=tmp_path / "artifacts",
            legacy_output_roots=(legacy_root,),
        )
    finally:
        legacy_root.chmod(0o755)

    assert result.manifest_path.exists()
    assert sentinel.read_text(encoding="utf-8") == "do not modify"
    assert list(legacy_root.iterdir()) == [sentinel]


def test_exporter_rejects_an_intermediate_symlink_to_a_v1_root(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    legacy_root = tmp_path / "legacy-v1-output"
    legacy_root.mkdir()
    os.symlink(legacy_root, artifact_root / "exports", target_is_directory=True)

    with pytest.raises(UnsafeExportPath):
        export_compatibility_predictions(
            _predictions(),
            model_id=_MODEL_ID,
            model_version=1,
            signal_date=date(2026, 1, 2),
            source_prediction_data_version="prediction-v1",
            artifact_root=artifact_root,
            legacy_output_roots=(legacy_root,),
        )

    assert list(legacy_root.iterdir()) == []


def test_exporter_rejects_an_intermediate_symlink_outside_the_artifact_root(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    artifact_root.mkdir()
    external_root = tmp_path / "external"
    external_root.mkdir()
    os.symlink(external_root, artifact_root / "exports", target_is_directory=True)

    with pytest.raises(UnsafeExportPath, match="outside artifact root"):
        export_compatibility_predictions(
            _predictions(),
            model_id=_MODEL_ID,
            model_version=1,
            signal_date=date(2026, 1, 2),
            source_prediction_data_version="prediction-v1",
            artifact_root=artifact_root,
            legacy_output_roots=_legacy_output_roots(tmp_path),
        )

    assert list(external_root.iterdir()) == []


def test_exporter_refuses_to_overwrite_an_existing_export(tmp_path: Path) -> None:
    args = {
        "model_id": _MODEL_ID,
        "model_version": 1,
        "signal_date": date(2026, 1, 2),
        "source_prediction_data_version": "prediction-v1",
        "artifact_root": tmp_path / "artifacts",
        "legacy_output_roots": _legacy_output_roots(tmp_path),
    }
    export_compatibility_predictions(_predictions(), **args)

    with pytest.raises(FileExistsError, match="immutable"):
        export_compatibility_predictions(_predictions(), **args)


def test_exporter_rejects_a_preexisting_dangling_date_link(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    model_root = artifact_root / "exports" / "v2" / _MODEL_ID
    model_root.mkdir(parents=True)
    os.symlink("dangling-target", model_root / "2026-01-02", target_is_directory=True)

    with pytest.raises(FileExistsError, match="immutable"):
        export_compatibility_predictions(
            _predictions(),
            model_id=_MODEL_ID,
            model_version=1,
            signal_date=date(2026, 1, 2),
            source_prediction_data_version="prediction-v1",
            artifact_root=artifact_root,
            legacy_output_roots=_legacy_output_roots(tmp_path),
        )

    assert os.readlink(model_root / "2026-01-02") == "dangling-target"
    assert not (model_root / "dangling-target").exists()


def test_exporter_preserves_a_concurrently_created_empty_destination(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact_root = tmp_path / "artifacts"
    destination = artifact_root / "exports" / "v2" / _MODEL_ID / "2026-01-02"
    original_symlink = os.symlink

    def competing_publish(
        source: str,
        target: str,
        *,
        target_is_directory: bool = False,
        dir_fd: int | None = None,
    ) -> None:
        assert dir_fd is not None
        os.mkdir(target, dir_fd=dir_fd)
        original_symlink(
            source,
            target,
            target_is_directory=target_is_directory,
            dir_fd=dir_fd,
        )

    monkeypatch.setattr("research_v2.export.compatibility.os.symlink", competing_publish)

    with pytest.raises(FileExistsError, match="immutable"):
        export_compatibility_predictions(
            _predictions(),
            model_id=_MODEL_ID,
            model_version=1,
            signal_date=date(2026, 1, 2),
            source_prediction_data_version="prediction-v1",
            artifact_root=artifact_root,
            legacy_output_roots=_legacy_output_roots(tmp_path),
        )

    assert destination.is_dir()
    assert not destination.is_symlink()
    assert list(destination.iterdir()) == []


def test_manifest_verification_detects_a_tampered_export_file(tmp_path: Path) -> None:
    result = export_compatibility_predictions(
        _predictions(),
        model_id=_MODEL_ID,
        model_version=1,
        signal_date=date(2026, 1, 2),
        source_prediction_data_version="prediction-v1",
        artifact_root=tmp_path / "artifacts",
        legacy_output_roots=_legacy_output_roots(tmp_path),
    )
    result.csv_path.write_text("ts_code,20260102\n000001.SZ,999.0000000000\n", encoding="utf-8")

    assert verify_compatibility_export(result.manifest_path) is False


def test_manifest_verification_detects_a_tampered_row_count(tmp_path: Path) -> None:
    result = export_compatibility_predictions(
        _predictions(),
        model_id=_MODEL_ID,
        model_version=1,
        signal_date=date(2026, 1, 2),
        source_prediction_data_version="prediction-v1",
        artifact_root=tmp_path / "artifacts",
        legacy_output_roots=_legacy_output_roots(tmp_path),
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["row_count"] = 3
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert verify_compatibility_export(result.manifest_path) is False


def test_manifest_verification_detects_a_tampered_contract_hash(tmp_path: Path) -> None:
    result = export_compatibility_predictions(
        _predictions(),
        model_id=_MODEL_ID,
        model_version=1,
        signal_date=date(2026, 1, 2),
        source_prediction_data_version="prediction-v1",
        artifact_root=tmp_path / "artifacts",
        legacy_output_roots=_legacy_output_roots(tmp_path),
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    manifest["contract_hash"] = "0" * 64
    result.manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    assert verify_compatibility_export(result.manifest_path) is False


def test_manifest_verification_rejects_a_manifest_outside_the_v2_layout(
    tmp_path: Path,
) -> None:
    result = export_compatibility_predictions(
        _predictions(),
        model_id=_MODEL_ID,
        model_version=1,
        signal_date=date(2026, 1, 2),
        source_prediction_data_version="prediction-v1",
        artifact_root=tmp_path / "artifacts",
        legacy_output_roots=_legacy_output_roots(tmp_path),
    )
    wrong_root = tmp_path / "wrong-layout"
    wrong_root.mkdir()
    wrong_manifest = wrong_root / "manifest.json"
    wrong_manifest.write_bytes(result.manifest_path.read_bytes())
    (wrong_root / "predictions.csv").write_bytes(result.csv_path.read_bytes())
    (wrong_root / "predictions.pkl").write_bytes(result.pkl_path.read_bytes())

    with pytest.raises(ValueError, match="outside the V2 export layout"):
        verify_compatibility_export(wrong_manifest)


def test_manifest_verification_rejects_a_regular_directory_copy(tmp_path: Path) -> None:
    result = export_compatibility_predictions(
        _predictions(),
        model_id=_MODEL_ID,
        model_version=1,
        signal_date=date(2026, 1, 2),
        source_prediction_data_version="prediction-v1",
        artifact_root=tmp_path / "artifacts",
        legacy_output_roots=_legacy_output_roots(tmp_path),
    )
    copied_root = tmp_path / "copied" / "exports" / "v2" / _MODEL_ID / "2026-01-02"
    copied_root.mkdir(parents=True)
    copied_manifest = copied_root / "manifest.json"
    copied_manifest.write_bytes(result.manifest_path.read_bytes())
    (copied_root / "predictions.csv").write_bytes(result.csv_path.read_bytes())
    (copied_root / "predictions.pkl").write_bytes(result.pkl_path.read_bytes())

    with pytest.raises(ValueError, match="exclusive publication link"):
        verify_compatibility_export(copied_manifest)


def test_manifest_verification_rejects_an_arbitrary_symbolic_link(tmp_path: Path) -> None:
    result = export_compatibility_predictions(
        _predictions(),
        model_id=_MODEL_ID,
        model_version=1,
        signal_date=date(2026, 1, 2),
        source_prediction_data_version="prediction-v1",
        artifact_root=tmp_path / "artifacts",
        legacy_output_roots=_legacy_output_roots(tmp_path),
    )
    copied_model_root = tmp_path / "copied" / "exports" / "v2" / _MODEL_ID
    copied_bundle = copied_model_root / "copied-bundle"
    copied_bundle.mkdir(parents=True)
    (copied_bundle / "manifest.json").write_bytes(result.manifest_path.read_bytes())
    (copied_bundle / "predictions.csv").write_bytes(result.csv_path.read_bytes())
    (copied_bundle / "predictions.pkl").write_bytes(result.pkl_path.read_bytes())
    os.symlink("copied-bundle", copied_model_root / "2026-01-02", target_is_directory=True)

    with pytest.raises(ValueError, match="exclusive publication link"):
        verify_compatibility_export(copied_model_root / "2026-01-02" / "manifest.json")


def test_contract_loader_rejects_unknown_column_mappings(tmp_path: Path) -> None:
    contract = tmp_path / "invalid.yaml"
    frozen_contract = Path(__file__).parents[3] / "configs" / "export" / "v1_compatible.yaml"
    contract.write_text(
        frozen_contract.read_text(encoding="utf-8").replace(
            "    score:\n      source: score\n      transform: identity",
            "    score:\n      source: unsupported_score\n      transform: identity",
            1,
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contract values"):
        load_compatibility_contract(contract)


def test_contract_loader_rejects_non_frozen_date_format(tmp_path: Path) -> None:
    contract = tmp_path / "invalid.yaml"
    frozen_contract = Path(__file__).parents[3] / "configs" / "export" / "v1_compatible.yaml"
    contract.write_text(
        frozen_contract.read_text(encoding="utf-8").replace(
            'format: "%Y%m%d"', 'format: "%Y-%m-%d"', 1
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="contract values"):
        load_compatibility_contract(contract)


def test_exporter_does_not_allow_a_contract_override(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="contract_path"):
        export_compatibility_predictions(
            _predictions(),
            model_id=_MODEL_ID,
            model_version=1,
            signal_date=date(2026, 1, 2),
            source_prediction_data_version="prediction-v1",
            artifact_root=tmp_path / "artifacts",
            legacy_output_roots=_legacy_output_roots(tmp_path),
            contract_path=tmp_path / "alternate.yaml",
        )
