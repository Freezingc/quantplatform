from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from research_v2.config.loader import load_settings


def _write_config(path: Path, name: str, values: dict[str, object]) -> None:
    (path / f"{name}.yaml").write_text(
        yaml.safe_dump(values, allow_unicode=True), encoding="utf-8"
    )


def _write_valid_configs(path: Path) -> None:
    _write_config(
        path,
        "base",
        {
            "dolphindb": {
                "host": "127.0.0.1",
                "port": 8848,
                "username": "research_reader",
                "password_env": "QRV2_DDB_PASSWORD",
                "source_database": "dfs://YesLab",
            },
            "artifact_root": "artifacts",
            "lock_year": 2026,
        },
    )
    _write_config(
        path,
        "test",
        {"environment": "test", "dolphindb": {"target_database": "dfs://QuantResearchV2Test"}},
    )


def test_loader_resolves_secret_only_from_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_valid_configs(tmp_path)
    monkeypatch.setenv("QRV2_DDB_PASSWORD", "runtime-secret")

    settings = load_settings("test", config_root=tmp_path)

    assert settings.environment == "test"
    assert settings.dolphindb.password.get_secret_value() == "runtime-secret"
    assert settings.dolphindb.target_database == "dfs://QuantResearchV2Test"


def test_loader_rejects_plaintext_password_in_yaml(tmp_path: Path) -> None:
    _write_valid_configs(tmp_path)
    _write_config(
        tmp_path,
        "test",
        {
            "environment": "test",
            "dolphindb": {
                "target_database": "dfs://QuantResearchV2Test",
                "password": "must-not-live-here",
            },
        },
    )

    with pytest.raises(ValueError, match="secret"):
        load_settings("test", config_root=tmp_path)


def test_loader_rejects_a_non_test_target_for_test_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_valid_configs(tmp_path)
    _write_config(
        tmp_path,
        "test",
        {"environment": "test", "dolphindb": {"target_database": "dfs://QuantResearchV2"}},
    )
    monkeypatch.setenv("QRV2_DDB_PASSWORD", "runtime-secret")

    with pytest.raises(ValueError, match="test environment"):
        load_settings("test", config_root=tmp_path)


def test_repository_test_config_uses_only_environment_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository_root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("QRV2_DDB_HOST", "127.0.0.1")
    monkeypatch.setenv("QRV2_DDB_USERNAME", "research_reader")
    monkeypatch.setenv("QRV2_DDB_PASSWORD", "runtime-secret")

    settings = load_settings("test", config_root=repository_root / "configs")

    assert settings.dolphindb.target_database == "dfs://QuantResearchV2Test"
