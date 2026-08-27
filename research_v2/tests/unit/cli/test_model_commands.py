from __future__ import annotations

import json

from typer.testing import CliRunner

from research_v2.cli.app import app


def test_model_train_plans_an_ordinary_family_fold_then_fails_closed() -> None:
    result = CliRunner().invoke(app, ["model", "train", "--family", "M1", "--fold", "F01"])

    assert result.exit_code == 2
    assert json.loads(result.stdout.splitlines()[0]) == {
        "family": "M1",
        "fold_id": "F01",
        "role": "TRAIN_VALIDATION",
    }
    assert "runtime backend" in result.stdout


def test_model_train_rejects_locked_or_unregistered_families() -> None:
    final = CliRunner().invoke(app, ["model", "train", "--family", "M1", "--fold", "FINAL"])
    unknown = CliRunner().invoke(app, ["model", "train", "--family", "M4", "--fold", "F01"])

    assert final.exit_code == 2
    assert "FINAL" in final.stdout
    assert unknown.exit_code == 2
    assert "M1" in unknown.stdout and "M3" in unknown.stdout


def test_model_train_fails_closed_when_prior_family_registry_evidence_is_unavailable() -> None:
    result = CliRunner().invoke(app, ["model", "train", "--family", "M2", "--fold", "F01"])

    assert result.exit_code == 2
    assert "prerequisite" in result.stdout.lower()


def test_walk_forward_cli_requires_best_simple_comparison_before_runtime_access() -> None:
    experiment_id = "sha256:" + "c" * 64
    result = CliRunner().invoke(
        app,
        [
            "evaluate",
            "walk-forward",
            "--experiment-id",
            experiment_id,
            "--compare",
            "best-simple",
        ],
    )

    assert result.exit_code == 2
    assert json.loads(result.stdout.splitlines()[0]) == {
        "compare": "best-simple",
        "experiment_id": experiment_id,
        "folds": ["F01", "F02", "F03", "F04", "F05", "F06"],
        "role": "ORDINARY_TEST",
    }
    assert "runtime backend" in result.stdout
