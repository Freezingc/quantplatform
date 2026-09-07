from __future__ import annotations

import json

from typer.testing import CliRunner

from research_v2.cli.app import app


def test_baseline_run_emits_plan_then_fails_closed() -> None:
    result = CliRunner().invoke(
        app,
        ["baseline", "run", "--fold", "F1", "--models", "B0,B1,B2,B3"],
    )

    assert result.exit_code == 2
    assert json.loads(result.stdout.splitlines()[0]) == {
        "fold_id": "F1",
        "models": ["B0", "B1", "B2", "B3"],
        "role": "TRAIN_VALIDATION",
    }
    assert "runtime backend" in result.stdout


def test_baseline_run_rejects_final_and_incomplete_candidates() -> None:
    final = CliRunner().invoke(
        app,
        ["baseline", "run", "--fold", "FINAL", "--models", "B0,B1,B2,B3"],
    )
    incomplete = CliRunner().invoke(
        app,
        ["baseline", "run", "--fold", "F1", "--models", "B0,B1"],
    )

    assert final.exit_code == 2
    assert "FINAL" in final.stdout
    assert incomplete.exit_code == 2
    assert "B2" in incomplete.stdout and "B3" in incomplete.stdout


def test_baseline_run_rejects_unknown_fold() -> None:
    result = CliRunner().invoke(
        app,
        ["baseline", "run", "--fold", "F07", "--models", "B0,B1,B2,B3"],
    )

    assert result.exit_code == 2
    assert "F1" in result.stdout and "F4" in result.stdout


def test_evaluate_and_report_commands_validate_scope_before_backend() -> None:
    evaluate = CliRunner().invoke(
        app,
        [
            "evaluate",
            "walk-forward",
            "--experiment-id",
            "sha256:" + "b" * 64,
            "--folds",
            "F1,F2,F3,F4",
        ],
    )
    report = CliRunner().invoke(
        app,
        [
            "report",
            "experiment",
            "--experiment-id",
            "sha256:" + "b" * 64,
            "--format",
            "html",
        ],
    )

    assert evaluate.exit_code == 2
    assert "runtime backend" in evaluate.stdout
    assert report.exit_code == 2
    assert "runtime backend" in report.stdout
