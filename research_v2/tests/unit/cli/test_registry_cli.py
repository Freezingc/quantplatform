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
