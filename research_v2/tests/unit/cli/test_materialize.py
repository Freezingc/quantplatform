from __future__ import annotations

import json

from typer.testing import CliRunner

from research_v2.cli.app import app


def test_market_materialize_dry_run_declares_read_and_write_boundaries() -> None:
    result = CliRunner().invoke(
        app,
        ["materialize", "market", "--start", "2025-01-02", "--end", "2025-01-03", "--dry-run"],
    )

    assert result.exit_code == 0
    plan = json.loads(result.stdout)
    assert plan["dry_run"] is True
    assert plan["source_tables"] == ["dfs://YesLab/stock_factor_daily"]
    assert plan["target_table"] == "market_daily_clean"
    assert plan["target_database"] == "dfs://QuantResearchV2Test"


def test_non_dry_run_fails_closed_until_a_runtime_backend_is_configured() -> None:
    result = CliRunner().invoke(
        app,
        ["materialize", "labels", "--start", "2025-01-02", "--end", "2025-01-03"],
    )

    assert result.exit_code != 0
    assert "runtime backend" in result.stdout
