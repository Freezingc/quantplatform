from __future__ import annotations

import hashlib
import json

from typer.testing import CliRunner

from research_v2.cli.app import app


def _walk_forward_evidence() -> dict[str, object]:
    return {
        "schema_version": 1,
        "role": "ORDINARY_TEST",
        "gate_evidence": {
            "mean_rank_ic": 0.04,
            "hac_t_statistic": 2.5,
            "yearly_mean_rank_ic": {str(year): 0.04 for year in range(2020, 2026)},
            "candidate_delta_rank_ic": 0.006,
            "valid_test_day_count": 6,
            "score_health": [
                {"coverage": 0.95, "unique_score_count": 30, "score_std": 0.1} for _ in range(6)
            ],
        },
        "reported_gates": {
            gate: {"status": "FAIL"} for gate in ("G1", "G2", "G3", "G4", "G5", "G6")
        },
    }


def test_registry_promote_dry_run_recomputes_gates_from_raw_evidence(tmp_path: object) -> None:
    evidence_path = tmp_path / "walk_forward.json"  # type: ignore[operator]
    raw = json.dumps(_walk_forward_evidence(), sort_keys=True).encode("utf-8")
    evidence_path.write_bytes(raw)  # type: ignore[attr-defined]

    result = CliRunner().invoke(
        app,
        [
            "registry",
            "promote",
            "--model-id",
            "sha256:" + "a" * 64,
            "--to",
            "WALK_FORWARD_PASSED",
            "--evidence",
            str(evidence_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["evidence_hash"] == hashlib.sha256(raw).hexdigest()
    assert payload["gate_statuses"] == {
        gate: "PASS" for gate in ("G1", "G2", "G3", "G4", "G5", "G6")
    }


def test_registry_promote_rejects_a_non_content_addressed_model_id(tmp_path: object) -> None:
    evidence_path = tmp_path / "walk_forward.json"  # type: ignore[operator]
    evidence_path.write_text(json.dumps(_walk_forward_evidence()), encoding="utf-8")  # type: ignore[attr-defined]

    result = CliRunner().invoke(
        app,
        [
            "registry",
            "promote",
            "--model-id",
            "model-1",
            "--to",
            "WALK_FORWARD_PASSED",
            "--evidence",
            str(evidence_path),
            "--dry-run",
        ],
    )

    assert result.exit_code == 2
    assert "model_id" in result.stdout


def test_registry_pointer_plan_lists_required_checks_without_mutating_a_pointer() -> None:
    result = CliRunner().invoke(
        app,
        [
            "registry",
            "pointer",
            "plan",
            "--consumer",
            "paper-sim",
            "--model-id",
            "sha256:" + "a" * 64,
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["dry_run"] is True
    assert payload["checks"] == {
        "artifact": "BACKEND_UNAVAILABLE",
        "compatibility_export": "BACKEND_UNAVAILABLE",
        "latest_prediction": "BACKEND_UNAVAILABLE",
        "model_status": "BACKEND_UNAVAILABLE",
        "schema": "BACKEND_UNAVAILABLE",
    }


def test_registry_pointer_production_switch_requires_second_confirmation() -> None:
    result = CliRunner().invoke(
        app,
        [
            "registry",
            "pointer",
            "switch",
            "--consumer",
            "paper-sim",
            "--environment",
            "production",
            "--model-id",
            "sha256:" + "a" * 64,
            "--expected-current",
            "sha256:" + "b" * 64,
            "--expected-version",
            "1",
            "--expected-pointer-version",
            "3",
            "--reason",
            "production switch",
            "--evidence",
            "a" * 64,
            "--run-id",
            "00000000-0000-0000-0000-000000000001",
        ],
    )

    assert result.exit_code == 2
    assert "confirm-production" in result.stdout


def test_registry_pointer_switch_dry_run_validates_without_a_backend() -> None:
    result = CliRunner().invoke(
        app,
        [
            "registry",
            "pointer",
            "switch",
            "--consumer",
            "paper-sim",
            "--environment",
            "shadow",
            "--model-id",
            "sha256:" + "a" * 64,
            "--expected-current",
            "sha256:" + "b" * 64,
            "--expected-version",
            "1",
            "--expected-pointer-version",
            "3",
            "--reason",
            "shadow switch",
            "--evidence",
            "a" * 64,
            "--run-id",
            "00000000-0000-0000-0000-000000000001",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert json.loads(result.stdout) == {
        "consumer": "paper-sim",
        "dry_run": True,
        "environment": "shadow",
        "expected_current": "sha256:" + "b" * 64,
        "expected_pointer_version": 3,
        "expected_version": 1,
        "model_id": "sha256:" + "a" * 64,
    }


def test_registry_pointer_switch_rejects_an_ambiguous_cas_request() -> None:
    result = CliRunner().invoke(
        app,
        [
            "registry",
            "pointer",
            "switch",
            "--consumer",
            "paper-sim",
            "--environment",
            "shadow",
            "--model-id",
            "sha256:" + "a" * 64,
            "--expected-current",
            "sha256:" + "b" * 64,
            "--expected-version",
            "1",
            "--reason",
            "ambiguous switch",
            "--evidence",
            "a" * 64,
            "--run-id",
            "00000000-0000-0000-0000-000000000001",
            "--dry-run",
        ],
    )

    assert result.exit_code == 2
    assert "expected-pointer-version" in result.stdout


def test_registry_pointer_switch_dry_run_can_explicitly_expect_no_current_pointer() -> None:
    result = CliRunner().invoke(
        app,
        [
            "registry",
            "pointer",
            "switch",
            "--consumer",
            "paper-sim",
            "--environment",
            "shadow",
            "--model-id",
            "sha256:" + "a" * 64,
            "--expect-absent",
            "--reason",
            "initial switch",
            "--evidence",
            "a" * 64,
            "--run-id",
            "00000000-0000-0000-0000-000000000001",
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    payload = json.loads(result.stdout)
    assert payload["expected_current"] is None
    assert payload["expected_pointer_version"] is None
    assert payload["expected_version"] is None
