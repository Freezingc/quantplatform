from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from research_v2.cli.app import app

_MODEL_ID = "sha256:" + "a" * 64
_SIGNAL_DATE = "2026-08-31"
_RUN_ID = "00000000-0000-0000-0000-000000000001"
_CONFIG_PATH = Path(__file__).resolve().parents[3] / "configs" / "inference" / "daily.yaml"


def _invoke(arguments: list[str]) -> object:
    return CliRunner().invoke(app, arguments)


def test_daily_config_targets_v2test_with_the_expected_schedule() -> None:
    config = yaml.safe_load(_CONFIG_PATH.read_text(encoding="utf-8"))

    assert config == {
        "target_database": "dfs://QuantResearchV2Test",
        "target_environment": "V2Test",
        "timezone": "Asia/Shanghai",
        "trigger_time": "19:30",
        "slo_time": "20:00",
        "retry_times": ["19:35", "19:45", "20:00"],
        "max_retries": 3,
    }


def test_daily_dry_run_emits_a_v2test_plan() -> None:
    result = _invoke(
        [
            "inference",
            "daily",
            "--signal-date",
            _SIGNAL_DATE,
            "--model-id",
            _MODEL_ID,
            "--mode",
            "shadow",
            "--dry-run",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["signal_date"] == _SIGNAL_DATE
    assert payload["model_id"] == _MODEL_ID
    assert payload["mode"] == "shadow"
    assert payload["target_database"] == "dfs://QuantResearchV2Test"
    assert payload["target_environment"] == "V2Test"
    assert payload["dry_run"] is True
    assert payload["write"] is False
    assert payload["schedule"] == {
        "timezone": "Asia/Shanghai",
        "trigger_time": "19:30",
        "slo_time": "20:00",
    }
    assert payload["retry_policy"] == {
        "retry_times": ["19:35", "19:45", "20:00"],
        "max_retries": 3,
    }


def test_daily_production_dry_run_requires_active_pointer_and_production_backend() -> None:
    result = _invoke(
        [
            "inference",
            "daily",
            "--signal-date",
            _SIGNAL_DATE,
            "--model-id",
            _MODEL_ID,
            "--mode",
            "production",
            "--dry-run",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["mode"] == "production"
    assert payload["runtime_requirements"] == ["active-pointer", "PRODUCTION"]
    assert payload["active_pointer_required"] is True


def test_daily_live_run_refuses_without_a_runtime_backend() -> None:
    result = _invoke(
        [
            "inference",
            "daily",
            "--signal-date",
            _SIGNAL_DATE,
            "--model-id",
            _MODEL_ID,
            "--mode",
            "shadow",
        ]
    )

    assert result.exit_code == 2
    assert "runtime backend is not configured" in result.stdout


def test_daily_production_live_run_mentions_active_pointer_and_production() -> None:
    result = _invoke(
        [
            "inference",
            "daily",
            "--signal-date",
            _SIGNAL_DATE,
            "--model-id",
            _MODEL_ID,
            "--mode",
            "production",
        ]
    )

    assert result.exit_code == 2
    assert "active-pointer" in result.stdout
    assert "PRODUCTION" in result.stdout


def test_daily_rejects_invalid_signal_date_model_id_and_mode() -> None:
    bad_date = _invoke(
        [
            "inference",
            "daily",
            "--signal-date",
            "2026-02-30",
            "--model-id",
            _MODEL_ID,
            "--mode",
            "shadow",
            "--dry-run",
        ]
    )
    bad_model = _invoke(
        [
            "inference",
            "daily",
            "--signal-date",
            _SIGNAL_DATE,
            "--model-id",
            "model-1",
            "--mode",
            "shadow",
            "--dry-run",
        ]
    )
    bad_mode = _invoke(
        [
            "inference",
            "daily",
            "--signal-date",
            _SIGNAL_DATE,
            "--model-id",
            _MODEL_ID,
            "--mode",
            "paper",
            "--dry-run",
        ]
    )

    assert bad_date.exit_code == 2
    assert "signal date" in bad_date.stdout.lower()
    assert bad_model.exit_code == 2
    assert "sha256:" in bad_model.stdout
    assert bad_mode.exit_code == 2
    assert "shadow or production" in bad_mode.stdout


def test_status_is_read_only_json() -> None:
    result = _invoke(
        [
            "inference",
            "status",
            "--signal-date",
            _SIGNAL_DATE,
            "--model-id",
            _MODEL_ID,
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["signal_date"] == _SIGNAL_DATE
    assert payload["model_id"] == _MODEL_ID
    assert payload["target_database"] == "dfs://QuantResearchV2Test"
    assert payload["read_only"] is True
    assert payload["backend_status"] == "BACKEND_UNAVAILABLE"


def test_retry_dry_run_uses_a_fresh_run_id(monkeypatch: pytest.MonkeyPatch) -> None:
    fresh_run_id = uuid.UUID("11111111-1111-1111-1111-111111111111")
    monkeypatch.setattr("research_v2.cli.inference.uuid.uuid4", lambda: fresh_run_id)

    result = _invoke(
        [
            "inference",
            "retry",
            "--run-id",
            _RUN_ID,
            "--from-step",
            "watermark",
            "--dry-run",
        ]
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["original_run_id"] == _RUN_ID
    assert payload["retry_run_id"] == str(fresh_run_id)
    assert payload["retry_run_id"] != _RUN_ID
    assert payload["from_step"] == "watermark"
    assert payload["dry_run"] is True


def test_retry_live_run_emits_a_fresh_plan_then_refuses_without_a_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fresh_run_id = uuid.UUID("22222222-2222-2222-2222-222222222222")
    monkeypatch.setattr("research_v2.cli.inference.uuid.uuid4", lambda: fresh_run_id)

    result = _invoke(
        [
            "inference",
            "retry",
            "--run-id",
            _RUN_ID,
            "--from-step",
            "watermark",
        ]
    )

    assert result.exit_code == 2
    payload = json.loads(result.stdout.splitlines()[0])
    assert payload["original_run_id"] == _RUN_ID
    assert payload["retry_run_id"] == str(fresh_run_id)
    assert payload["retry_run_id"] != _RUN_ID
    assert "runtime backend is not configured" in result.stdout


def test_retry_rejects_a_non_uuid_run_id_and_a_non_watermark_step() -> None:
    bad_run_id = _invoke(
        [
            "inference",
            "retry",
            "--run-id",
            "not-a-uuid",
            "--from-step",
            "watermark",
            "--dry-run",
        ]
    )
    bad_step = _invoke(
        [
            "inference",
            "retry",
            "--run-id",
            _RUN_ID,
            "--from-step",
            "build_batch",
            "--dry-run",
        ]
    )

    assert bad_run_id.exit_code == 2
    assert "uuid" in bad_run_id.stdout.lower()
    assert bad_step.exit_code == 2
    assert "watermark" in bad_step.stdout
