"""The C1.5 dry-run script must be able to report, but never to write or to read values."""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Any

import audit_factor_catalog_schema_dry_run as audit
import pytest
import yaml

from research_v2.db.session import _MUTATING_COMMAND_PATTERN
from research_v2.factors.catalog_dry_run import SourceColumn

_LIVE_TYPES = {
    "ts_code": "SYMBOL",
    "trade_date": "DATE",
    "factor_name": "STRING",
    "value": "DOUBLE",
}


def _referenced_names(module_path: Path) -> set[str]:
    """Every identifier the module actually references, ignoring comments and docstrings."""
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    names.update(node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute))
    return names


def _string_literals(module_path: Path) -> list[str]:
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    return [
        node.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Constant) and isinstance(node.value, str)
    ]


def test_the_script_never_constructs_a_write_session() -> None:
    """C1.5 forbids writing to the database; this pins it structurally, not by inspection.

    Checked over the parsed AST rather than the raw text so the docstring and the inline comment
    that both *mention* ``V2WriteSession`` -- to say it is not used -- do not cause a false pass or
    a false failure. Adding the import anywhere, including inside ``main``, fails this.
    """
    names = _referenced_names(Path(audit.__file__).resolve())

    assert "V2WriteSession" not in names
    assert "UploadClient" not in names
    assert "YesLabReadOnlySession" in names, (
        "the read-only boundary is the only session this script may build"
    )


def test_the_script_issues_no_mutating_statement_even_as_a_literal() -> None:
    """Belt and braces: no DDB mutation verb appears anywhere in the module's strings."""
    offenders = [
        literal
        for literal in _string_literals(Path(audit.__file__).resolve())
        if _MUTATING_COMMAND_PATTERN.search(literal)
    ]

    assert offenders == []


def test_the_only_query_shape_the_script_can_emit_is_a_schema_query() -> None:
    """No value read is reachable: the query text is built in one place and is metadata-only."""
    from research_v2.factors.catalog_dry_run import _schema_query

    queries = [literal for literal in _string_literals(Path(audit.__file__).resolve())]
    assert any("colDefs" in literal for literal in queries)
    assert not any(
        re.search(r"\bselect\b.*\bfrom\b", literal, re.IGNORECASE) for literal in queries
    ), "the script must not contain a value-reading query"
    assert _schema_query("factor_data1") == (
        "schema(loadTable('dfs://YesLab', 'factor_data1')).colDefs"
    )


def test_saved_schemas_round_trip_to_an_identical_schema_set(tmp_path: Path) -> None:
    """The replay file must not lose information, or a re-examined verdict is not the same one."""
    schemas = {
        "factor_data1": tuple(
            SourceColumn(name=name, type_string=kind) for name, kind in _LIVE_TYPES.items()
        ),
        "factor_data9": None,
    }
    errors = {"factor_data9": "RuntimeError: table not found"}
    target = tmp_path / "schemas.json"
    target.write_text(
        json.dumps(audit._schema_payload(schemas, errors), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    restored = audit._schemas_from_disk(target)

    assert restored == schemas
    assert json.loads(target.read_text(encoding="utf-8"))["errors"] == errors


def test_the_real_catalog_loads_with_its_snapshot_hash() -> None:
    entries, snapshot_hash = audit._load_entries(audit._DEFAULT_CATALOG)

    assert entries, "the shipped catalog must not be empty"
    assert snapshot_hash is not None and re.fullmatch(r"[0-9a-f]{64}", snapshot_hash)
    required = {"factor_id", "factor_version", "source_table", "source_value_column"}
    assert all(required.issubset(entry) for entry in entries)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, frozenset()),
        ("", frozenset()),
        ("yeslab_7", frozenset({("yeslab_7", 1)})),
        ("yeslab_7:2", frozenset({("yeslab_7", 2)})),
        ("yeslab_7:2, yeslab_8", frozenset({("yeslab_7", 2), ("yeslab_8", 1)})),
    ],
)
def test_retired_parsing(raw: str | None, expected: frozenset[tuple[str, int]]) -> None:
    assert audit._parse_retired(raw) == expected


def _write_catalog(path: Path, value_column: str) -> None:
    document: dict[str, Any] = {
        "metadata_snapshot_hash": "a" * 64,
        "factors": [
            {
                "factor_id": "yeslab_1",
                "factor_version": 1,
                "display_name": "Alpha1",
                "source_database": "dfs://YesLab",
                "source_table": "factor_data1",
                "source_factor_name": "Alpha1",
                "source_value_column": value_column,
                "source_date_column": "trade_date",
                "source_code_column": "ts_code",
            }
        ],
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def _write_schemas(path: Path, types: dict[str, str]) -> None:
    payload = audit._schema_payload(
        {
            "factor_data1": tuple(
                SourceColumn(name=name, type_string=kind) for name, kind in types.items()
            )
        },
        {},
    )
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _run(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    monkeypatch.setattr("sys.argv", ["audit_factor_catalog_schema_dry_run.py", *argv])
    return audit.main()


def test_a_clean_catalog_exits_zero_without_touching_a_cluster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = tmp_path / "catalog.yaml"
    schemas = tmp_path / "schemas.json"
    report = tmp_path / "report.json"
    _write_catalog(catalog, "value")
    _write_schemas(schemas, _LIVE_TYPES)

    assert _run(monkeypatch, ["--catalog", str(catalog), "--schemas-from", str(schemas),
                              "--report", str(report)]) == 0

    written = json.loads(report.read_text(encoding="utf-8"))
    assert written["summary"]["mapped"] == 1
    assert written["summary"]["blocked"] == 0


def test_a_blocked_entry_exits_nonzero_so_it_can_gate_a_catalog_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    catalog = tmp_path / "catalog.yaml"
    schemas = tmp_path / "schemas.json"
    _write_catalog(catalog, "a_column_the_source_does_not_have")
    _write_schemas(schemas, _LIVE_TYPES)

    assert _run(monkeypatch, ["--catalog", str(catalog), "--schemas-from", str(schemas)]) == 1
    assert _run(
        monkeypatch,
        ["--catalog", str(catalog), "--schemas-from", str(schemas), "--report-only"],
    ) == 0, "--report-only downgrades the verdict to a summary"


def test_an_unmapped_source_column_alone_is_enough_to_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is blocked, yet the source carries a column the catalog does not describe."""
    catalog = tmp_path / "catalog.yaml"
    schemas = tmp_path / "schemas.json"
    report = tmp_path / "report.json"
    _write_catalog(catalog, "value")
    drifted = dict(_LIVE_TYPES)
    drifted["newly_added"] = "DOUBLE"
    _write_schemas(schemas, drifted)

    assert _run(monkeypatch, ["--catalog", str(catalog), "--schemas-from", str(schemas),
                              "--report", str(report)]) == 1

    summary = json.loads(report.read_text(encoding="utf-8"))["summary"]
    assert summary["blocked"] == 0
    assert summary["unmapped_source_columns"] == {"factor_data1": ["newly_added"]}
