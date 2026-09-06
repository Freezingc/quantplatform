from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from research_v2.cli.app import app
from research_v2.factors.catalog_dry_run import SourceColumn, schema_set_to_payload


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


# --- C5.3: the factor materialization dry-run command -------------------------------------------

_LIVE_TYPES = {
    "ts_code": "SYMBOL",
    "trade_date": "DATE",
    "factor_name": "STRING",
    "value": "DOUBLE",
}

_CATALOG_PATH = Path(__file__).resolve().parents[3] / "configs" / "factors" / "source_catalog.yaml"


def _write_calendar(path: Path) -> Path:
    dates = [f"2013-12-{day:02d}" for day in range(2, 32)]
    dates += ["2014-01-02", "2014-01-03", "2014-01-06", "2014-01-07", "2014-01-08"]
    dates += ["2014-02-03", "2014-02-04", "2014-02-05"]
    weekdays = [value for value in dates if date.fromisoformat(value).weekday() < 5]
    path.write_text("# trading dates\n" + "\n".join(weekdays) + "\n", encoding="utf-8")
    return path


def _write_catalog(path: Path, value_column: str = "value") -> Path:
    document = {
        "metadata_snapshot_hash": "a" * 64,
        "factors": [
            {
                "factor_id": f"yeslab_{index}",
                "factor_version": 1,
                "display_name": f"Alpha{index}",
                "source_database": "dfs://YesLab",
                "source_table": "factor_data1",
                "source_factor_name": f"Alpha{index}",
                "source_value_column": value_column,
                "source_date_column": "trade_date",
                "source_code_column": "ts_code",
            }
            for index in (1, 2)
        ],
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")
    return path


def _write_schemas(path: Path, types: dict[str, str] | None = None) -> Path:
    columns = tuple(
        SourceColumn(name=name, type_string=kind)
        for name, kind in (types if types is not None else _LIVE_TYPES).items()
    )
    path.write_text(
        json.dumps(schema_set_to_payload({"factor_data1": columns}, {}), indent=2),
        encoding="utf-8",
    )
    return path


def _invoke_factors(arguments: list[str]) -> object:
    return CliRunner().invoke(app, ["materialize", "factors", *arguments])


def _base_arguments(tmp_path: Path, **extra: str) -> list[str]:
    catalog = _write_catalog(tmp_path / "catalog.yaml")
    calendar = _write_calendar(tmp_path / "calendar.txt")
    arguments = [
        "--start",
        "2014-01-01",
        "--end",
        "2014-02-28",
        "--catalog",
        str(catalog),
        "--calendar",
        str(calendar),
    ]
    for key, value in extra.items():
        flag = f"--{key.replace('_', '-')}"
        # an empty value means the option is a boolean switch and takes no argument
        if value:
            arguments.extend([flag, value])
        else:
            arguments.append(flag)
    return arguments


def test_the_factor_dry_run_reports_every_field_c53_names(tmp_path: Path) -> None:
    """Factor count, source tables and columns, date chunks, row estimate, target partitions."""
    result = _invoke_factors(_base_arguments(tmp_path, dry_run="", cross_section="1000"))

    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout.split("plan written")[0])
    assert plan["factor_count"] == 2
    assert plan["source_tables"] == ["factor_data1"]
    assert plan["source_columns"] == {
        "factor_data1": ["factor_name", "trade_date", "ts_code", "value"]
    }
    assert [chunk["logical_partition"] for chunk in plan["chunks"]] == ["2014-01", "2014-02"]
    assert plan["chunks"][0]["trading_days"] == 5
    assert plan["chunks"][0]["estimated_rows"] == 2 * 5 * 1000
    assert plan["estimated_rows_total"] == 2 * 8 * 1000
    assert plan["target_table"] == "factor_observation_clean"
    assert plan["target_partition_column"] == "signal_date"


def test_without_dry_run_the_factor_command_fails_closed_like_its_siblings(tmp_path: Path) -> None:
    result = _invoke_factors(_base_arguments(tmp_path))

    assert result.exit_code == 2
    assert "refusing to materialize" in result.stdout


def test_an_unverified_plan_says_so_and_is_not_materializable(tmp_path: Path) -> None:
    result = _invoke_factors(_base_arguments(tmp_path, dry_run=""))

    assert result.exit_code == 0
    plan = json.loads(result.stdout)
    assert plan["schema_verified"] is False
    assert plan["materializable"] is False


def test_a_saved_schema_set_makes_the_plan_verified_and_materializable(tmp_path: Path) -> None:
    schemas = _write_schemas(tmp_path / "schemas.json")

    result = _invoke_factors(_base_arguments(tmp_path, dry_run="", schemas=str(schemas)))

    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    assert plan["schema_verified"] is True
    assert plan["materializable"] is True
    assert plan["blocked_factors"] == []


def test_a_factor_that_cannot_be_read_refuses_the_whole_plan(tmp_path: Path) -> None:
    schemas = _write_schemas(tmp_path / "schemas.json")
    catalog = _write_catalog(tmp_path / "catalog.yaml", value_column="absent_column")
    calendar = _write_calendar(tmp_path / "calendar.txt")

    result = _invoke_factors(
        [
            "--start",
            "2014-01-01",
            "--end",
            "2014-01-31",
            "--catalog",
            str(catalog),
            "--calendar",
            str(calendar),
            "--schemas",
            str(schemas),
            "--dry-run",
        ]
    )

    assert result.exit_code == 2
    assert "refused" in result.output
    assert "absent_column" in result.output


def test_no_strict_reports_the_blocked_factor_instead_of_refusing(tmp_path: Path) -> None:
    schemas = _write_schemas(tmp_path / "schemas.json")
    catalog = _write_catalog(tmp_path / "catalog.yaml", value_column="absent_column")
    calendar = _write_calendar(tmp_path / "calendar.txt")

    result = _invoke_factors(
        [
            "--start",
            "2014-01-01",
            "--end",
            "2014-01-31",
            "--catalog",
            str(catalog),
            "--calendar",
            str(calendar),
            "--schemas",
            str(schemas),
            "--no-strict",
            "--dry-run",
        ]
    )

    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    assert plan["blocked_factors"] == [["yeslab_1", 1], ["yeslab_2", 1]]
    assert plan["materializable"] is False


def test_the_row_estimate_is_null_rather_than_guessed_without_a_width(tmp_path: Path) -> None:
    result = _invoke_factors(_base_arguments(tmp_path, dry_run=""))

    plan = json.loads(result.stdout)
    assert plan["cross_section_size"] is None
    assert plan["estimated_rows_total"] is None
    assert all(chunk["estimated_rows"] is None for chunk in plan["chunks"])


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("start", "01-2014-01"),
        ("end", "2014-13-01"),
        ("cross-section", "0"),
        ("lookback-trading-days", "0"),
    ],
)
def test_bad_parameters_are_rejected_before_any_planning(
    tmp_path: Path, flag: str, value: str
) -> None:
    calendar = _write_calendar(tmp_path / "calendar.txt")
    catalog = _write_catalog(tmp_path / "catalog.yaml")

    result = _invoke_factors(
        [
            "--start",
            "2014-01-01",
            "--end",
            "2014-02-28",
            "--catalog",
            str(catalog),
            "--calendar",
            str(calendar),
            f"--{flag}",
            value,
            "--dry-run",
        ]
    )

    assert result.exit_code != 0


def test_an_inverted_range_is_rejected(tmp_path: Path) -> None:
    calendar = _write_calendar(tmp_path / "calendar.txt")
    catalog = _write_catalog(tmp_path / "catalog.yaml")

    result = _invoke_factors(
        [
            "--start",
            "2014-03-01",
            "--end",
            "2014-01-01",
            "--catalog",
            str(catalog),
            "--calendar",
            str(calendar),
            "--dry-run",
        ]
    )

    assert result.exit_code != 0


def test_an_empty_calendar_file_is_rejected(tmp_path: Path) -> None:
    catalog = _write_catalog(tmp_path / "catalog.yaml")
    calendar = tmp_path / "calendar.txt"
    calendar.write_text("# only a comment\n", encoding="utf-8")

    result = _invoke_factors(
        [
            "--start",
            "2014-01-01",
            "--end",
            "2014-01-31",
            "--catalog",
            str(catalog),
            "--calendar",
            str(calendar),
            "--dry-run",
        ]
    )

    assert result.exit_code != 0


def test_the_output_flag_writes_the_same_document_it_prints(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "plan.json"

    result = _invoke_factors(_base_arguments(tmp_path, dry_run="", output=str(target)))

    assert result.exit_code == 0, result.output
    assert target.exists()
    assert json.loads(target.read_text(encoding="utf-8")) == json.loads(result.stdout)


def test_the_shipped_catalog_plans_across_its_eleven_source_tables(tmp_path: Path) -> None:
    """End to end against the real 487-entry catalog, with no cluster involved."""
    calendar = _write_calendar(tmp_path / "calendar.txt")

    result = _invoke_factors(
        [
            "--start",
            "2014-01-01",
            "--end",
            "2014-02-28",
            "--catalog",
            str(_CATALOG_PATH),
            "--calendar",
            str(calendar),
            "--dry-run",
        ]
    )

    assert result.exit_code == 0, result.output
    plan = json.loads(result.stdout)
    assert plan["factor_count"] == 487
    assert len(plan["source_tables"]) == 11
    assert plan["schema_verified"] is False, (
        "no --schemas was supplied, so the plan must not claim to have checked the source"
    )
