"""C1.5: the catalog must be checked against the source schema without reading any values.

Two properties matter more than the individual checks and are pinned separately below:

* the dry-run may only ever issue ``schema(...).colDefs`` queries -- never a ``select`` that
  pulls factor values, because a governance check that reads the whole source table is not a
  dry-run;
* a source table outside the frozen ``factor_data1..11`` set is refused *before* any query is
  issued, so a mistyped catalog entry cannot be used to probe an arbitrary table.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest
import yaml

from research_v2.factors.catalog_dry_run import (
    SourceColumn,
    collect_source_schemas,
    registered_source_columns,
    run_catalog_dry_run,
)
from research_v2.sources.yeslab import _schema_fingerprint

_LIVE_TYPES = {
    "ts_code": "SYMBOL",
    "trade_date": "DATE",
    "factor_name": "STRING",
    "value": "DOUBLE",
}


def _entry(**overrides: Any) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "factor_id": "yeslab_1",
        "factor_version": 1,
        "display_name": "Alpha1",
        "source_database": "dfs://YesLab",
        "source_table": "factor_data1",
        "source_factor_name": "Alpha1",
        "source_value_column": "value",
        "source_date_column": "trade_date",
        "source_code_column": "ts_code",
    }
    entry.update(overrides)
    return entry


def _columns(types: dict[str, str]) -> tuple[SourceColumn, ...]:
    return tuple(SourceColumn(name=name, type_string=kind) for name, kind in types.items())


def _schemas(**tables: dict[str, str]) -> dict[str, tuple[SourceColumn, ...]]:
    return {table: _columns(types) for table, types in tables.items()}


def _schema_frame(types: dict[str, str] | None = None) -> pd.DataFrame:
    """What ``schema(loadTable(...)).colDefs`` comes back as from the session."""
    return pd.DataFrame(
        [
            {"name": name, "typeString": kind}
            for name, kind in (types if types is not None else _LIVE_TYPES).items()
        ]
    )


def test_a_catalog_matching_the_live_source_schema_maps_completely() -> None:
    """Baseline against the schema actually observed on factor_data1..11 on 2026-09-06."""
    report = run_catalog_dry_run(
        [_entry(), _entry(factor_id="yeslab_2", display_name="Alpha2")],
        _schemas(factor_data1=_LIVE_TYPES),
    )

    assert (report.mapped_count, report.blocked_count, report.retired_count) == (2, 0, 0)
    assert all(finding.status == "MAPPED" for finding in report.findings)
    assert all(finding.reasons == () for finding in report.findings)
    assert report.unmapped_source_columns == {"factor_data1": ()}


def test_extra_unregistered_source_columns_are_reported_but_not_loaded() -> None:
    """C2.1's remaining clause: an unregistered column is surfaced, and nothing reads it.

    A column added to the source that no catalog entry mentions is data the V2 pipeline
    silently ignores. Reporting it is the point; the reader's column list is built from the
    catalog, so the column is also never selected.
    """
    drifted = dict(_LIVE_TYPES)
    drifted["somebody_added_this"] = "DOUBLE"

    report = run_catalog_dry_run([_entry()], _schemas(factor_data1=drifted))

    assert report.mapped_count == 1
    assert report.unmapped_source_columns == {"factor_data1": ("somebody_added_this",)}
    assert report.findings[0].reasons == (), (
        "an extra source column must not block the factors that do not use it"
    )


def test_selected_columns_are_exactly_the_registered_role_columns() -> None:
    """The other half of C2.1's clause: what the reader would load is derived, not hardcoded."""
    assert registered_source_columns([_entry()], "factor_data1") == (
        "factor_name",
        "trade_date",
        "ts_code",
        "value",
    )


def test_a_missing_source_table_blocks_only_its_own_factors() -> None:
    report = run_catalog_dry_run(
        [_entry(), _entry(factor_id="yeslab_9", source_table="factor_data9")],
        _schemas(factor_data1=_LIVE_TYPES),
    )

    assert (report.mapped_count, report.blocked_count) == (1, 1)
    blocked = report.findings[1]
    assert blocked.status == "BLOCKED"
    assert any("factor_data9" in reason for reason in blocked.reasons)


@pytest.mark.parametrize(
    ("role_field", "column", "kind"),
    [
        ("source_value_column", "value", "STRING"),
        ("source_value_column", "value", "DATE"),
        ("source_date_column", "trade_date", "SYMBOL"),
        ("source_date_column", "trade_date", "DOUBLE"),
        ("source_code_column", "ts_code", "DOUBLE"),
        ("source_code_column", "ts_code", "DATE"),
    ],
)
def test_a_role_column_of_an_unconvertible_type_blocks_the_factor(
    role_field: str, column: str, kind: str
) -> None:
    """Each role is read a specific way, so an incompatible type must be refused up front.

    ``value`` goes through ``pd.to_numeric``: a STRING column would coerce to all-NaN and look
    like a factor with no data rather than a schema mismatch. ``trade_date`` goes through
    ``pd.to_datetime(errors="raise")``. ``ts_code`` must be a symbol/text key.
    """
    types = dict(_LIVE_TYPES)
    types[column] = kind

    report = run_catalog_dry_run([_entry()], _schemas(factor_data1=types))

    assert report.blocked_count == 1
    assert any(column in reason and kind in reason for reason in report.findings[0].reasons)


@pytest.mark.parametrize("missing", ["ts_code", "trade_date", "value"])
def test_a_registered_column_absent_from_the_table_blocks_the_factor(missing: str) -> None:
    types = {name: kind for name, kind in _LIVE_TYPES.items() if name != missing}

    report = run_catalog_dry_run([_entry()], _schemas(factor_data1=types))

    assert report.blocked_count == 1
    assert any(missing in reason for reason in report.findings[0].reasons)


def test_a_missing_discriminator_column_blocks_even_though_no_entry_registers_it() -> None:
    """``factor_name`` selects which factor a long-form row belongs to, so it is a key too.

    No catalog entry names it as a column -- the catalog stores the factor *name* as a value --
    but the reader filters on it, so a table without it cannot be read at all.
    """
    types = {name: kind for name, kind in _LIVE_TYPES.items() if name != "factor_name"}

    report = run_catalog_dry_run([_entry()], _schemas(factor_data1=types))

    assert report.blocked_count == 1
    assert any("factor_name" in reason for reason in report.findings[0].reasons)


def test_a_key_column_doubling_as_another_role_is_ambiguous_and_blocked() -> None:
    """The same column cannot be both the date key and the security key."""
    report = run_catalog_dry_run(
        [_entry(source_code_column="trade_date")],
        _schemas(factor_data1=_LIVE_TYPES),
    )

    assert report.blocked_count == 1
    assert any("ambiguous" in reason.lower() for reason in report.findings[0].reasons)


def test_every_defect_of_one_factor_is_reported_not_just_the_first() -> None:
    """A report that stops at the first problem makes the next dry-run find a new one."""
    report = run_catalog_dry_run(
        [_entry(source_value_column="absent_value", source_code_column="absent_code")],
        _schemas(factor_data1=_LIVE_TYPES),
    )

    assert report.blocked_count == 1
    joined = " ".join(report.findings[0].reasons)
    assert "absent_value" in joined and "absent_code" in joined
    assert len(report.findings[0].reasons) >= 2


def test_a_duplicate_column_name_in_the_source_schema_is_ambiguous() -> None:
    report = run_catalog_dry_run(
        [_entry()],
        {"factor_data1": _columns(_LIVE_TYPES) + (SourceColumn("value", "STRING"),)},
    )

    assert report.blocked_count == 1
    assert any("value" in reason for reason in report.findings[0].reasons)


def test_retired_entries_are_counted_as_retired_and_not_re_checked() -> None:
    """A retired factor is intentionally out, so source drift must not turn it into a blocker."""
    report = run_catalog_dry_run(
        [_entry(factor_id="yeslab_1"), _entry(factor_id="yeslab_2", source_table="factor_data9")],
        _schemas(factor_data1=_LIVE_TYPES),
        retired={("yeslab_2", 1)},
    )

    assert (report.mapped_count, report.blocked_count, report.retired_count) == (1, 0, 1)
    assert report.findings[1].status == "RETIRED"
    assert report.findings[1].reasons == ()


def test_columns_only_a_retired_entry_used_become_unmapped() -> None:
    """The reference set is what the pipeline still consumes, so retirement frees a column."""
    wide = dict(_LIVE_TYPES)
    wide["legacy_value"] = "DOUBLE"
    entries = [
        _entry(factor_id="yeslab_1", source_value_column="value"),
        _entry(factor_id="yeslab_2", source_value_column="legacy_value"),
    ]

    still_used = run_catalog_dry_run(entries, _schemas(factor_data1=wide))
    after_retirement = run_catalog_dry_run(
        entries, _schemas(factor_data1=wide), retired={("yeslab_2", 1)}
    )

    assert still_used.unmapped_source_columns == {"factor_data1": ()}
    assert after_retirement.unmapped_source_columns == {"factor_data1": ("legacy_value",)}
    assert after_retirement.mapped_count == 1, (
        "the surviving entry is unaffected by its neighbour's retirement"
    )


def test_a_table_outside_the_frozen_set_is_refused_without_being_queried() -> None:
    """A mistyped catalog entry must not become a way to probe an arbitrary YesLab table."""
    scripts: list[str] = []

    class _Recording:
        def query(self, script: str) -> object:
            scripts.append(script)
            return _schema_frame()

    with pytest.raises(ValueError, match="factor_data99"):
        collect_source_schemas(_Recording(), {"factor_data1", "factor_data99"})

    assert scripts == [], "no query may be issued once an unsafe table name is present"


def test_the_dry_run_issues_schema_queries_only_and_never_reads_values() -> None:
    """The whole point of a dry-run: metadata, not observations."""
    scripts: list[str] = []

    class _Recording:
        def query(self, script: str) -> object:
            scripts.append(script)
            return _schema_frame()

    schemas = collect_source_schemas(_Recording(), {"factor_data1", "factor_data3"})

    assert sorted(schemas) == ["factor_data1", "factor_data3"]
    assert sorted(scripts) == [
        "schema(loadTable('dfs://YesLab', 'factor_data1')).colDefs",
        "schema(loadTable('dfs://YesLab', 'factor_data3')).colDefs",
    ]
    assert not any("select" in script.lower() for script in scripts)


def test_a_table_whose_schema_query_fails_is_reported_not_swallowed() -> None:
    """An outage must not masquerade as schema drift."""

    class _Failing:
        def query(self, script: str) -> object:
            if "factor_data9" in script:
                raise RuntimeError("table not found")
            return _schema_frame()

    schemas = collect_source_schemas(_Failing(), {"factor_data1", "factor_data9"})

    assert schemas["factor_data9"] is None
    report = run_catalog_dry_run(
        [_entry(factor_id="yeslab_9", source_table="factor_data9")],
        schemas,  # type: ignore[arg-type]
        schema_errors={"factor_data9": "RuntimeError: table not found"},
    )
    assert report.blocked_count == 1
    assert any("RuntimeError" in reason for reason in report.findings[0].reasons)


def test_the_fingerprint_matches_the_source_adapter_for_the_same_schema() -> None:
    """Comparable with ``SourceBatch.schema_fingerprint``, so drift is attributable."""
    report = run_catalog_dry_run([_entry()], _schemas(factor_data1=_LIVE_TYPES))

    expected = _schema_fingerprint(_schema_frame())
    assert report.schema_fingerprints["factor_data1"] == expected


def test_the_report_is_independent_of_catalog_and_schema_ordering() -> None:
    entries = [
        _entry(factor_id="yeslab_1"),
        _entry(factor_id="yeslab_2", source_table="factor_data2"),
        _entry(factor_id="yeslab_3", source_table="factor_data2"),
    ]
    schemas = _schemas(factor_data1=_LIVE_TYPES, factor_data2=_LIVE_TYPES)
    reversed_schemas = {table: schemas[table] for table in reversed(sorted(schemas))}

    forward = run_catalog_dry_run(entries, schemas)
    backward = run_catalog_dry_run(list(reversed(entries)), reversed_schemas)

    assert forward.findings == backward.findings
    assert dict(forward.unmapped_source_columns) == dict(backward.unmapped_source_columns)
    assert [finding.factor_id for finding in forward.findings] == [
        "yeslab_1",
        "yeslab_2",
        "yeslab_3",
    ]


def test_a_column_referenced_by_one_table_is_not_credited_to_another() -> None:
    """Unmapped columns are per table: the same name in two tables is two separate facts."""
    left = dict(_LIVE_TYPES)
    left["extra"] = "DOUBLE"

    report = run_catalog_dry_run(
        [_entry(), _entry(factor_id="yeslab_2", source_table="factor_data2")],
        _schemas(factor_data1=left, factor_data2=_LIVE_TYPES),
    )

    assert report.unmapped_source_columns == {"factor_data1": ("extra",), "factor_data2": ()}


def test_a_missing_catalog_field_is_a_defect_of_the_entry_not_of_the_source() -> None:
    entry = _entry()
    del entry["source_value_column"]

    report = run_catalog_dry_run([entry], _schemas(factor_data1=_LIVE_TYPES))

    assert report.blocked_count == 1
    assert any("source_value_column" in reason for reason in report.findings[0].reasons)


# --- the shipped catalog, checked against the schema the source actually reported --------------

_CATALOG_PATH = Path(__file__).resolve().parents[3] / "configs" / "factors" / "source_catalog.yaml"


def _catalog_entries() -> list[dict[str, Any]]:
    document = yaml.safe_load(_CATALOG_PATH.read_text(encoding="utf-8"))
    return list(document["factors"])


def test_the_shipped_catalog_maps_cleanly_against_the_observed_source_schema() -> None:
    """Every registered factor must be readable as registered, or the catalog is describing a
    source that does not exist.

    ``_LIVE_TYPES`` is what ``schema(loadTable('dfs://YesLab', 'factor_dataN')).colDefs`` returned
    for all eleven tables on 2026-09-06 -- they are identical, and the dry-run of that date
    reported 487 mapped / 0 blocked / 0 retired and no unmapped columns. Deliberately not asserting
    the entry count: a catalog that legitimately grows should still pass. What must not change
    silently is the *verdict*.
    """
    entries = _catalog_entries()
    tables = {str(entry["source_table"]) for entry in entries}
    assert tables, "the catalog must name at least one source table"

    report = run_catalog_dry_run(entries, _schemas(**dict.fromkeys(sorted(tables), _LIVE_TYPES)))

    assert report.blocked_count == 0, [
        (finding.factor_id, finding.reasons) for finding in report.blocked_findings
    ][:10]
    assert report.mapped_count == len(entries)
    assert report.retired_count == 0
    assert set(report.unmapped_source_columns.values()) == {()}, (
        "a source column no catalog entry registers is data the pipeline silently ignores"
    )


def test_the_catalog_registration_agrees_with_the_columns_the_read_path_projects() -> None:
    """Two independent sources of truth for "which columns get read" must not drift apart.

    The catalog registers role columns per factor; ``source_backfill._SOURCE_COLUMNS`` is a
    hardcoded projection the read path actually issues. Today they agree. The dry-run checks
    catalog against *source*, and the query tests check the *reader* against a returned frame, so
    without this nothing would notice a catalog entry registering a column the read path never
    selects -- the factor would silently be read from the wrong column, or not at all.
    """
    from research_v2.factors.source_backfill import _SOURCE_COLUMNS

    entries = _catalog_entries()
    expected = tuple(sorted(_SOURCE_COLUMNS))

    disagreements = {
        (str(entry["factor_id"]), entry["source_table"]): registered_source_columns(
            [entry], str(entry["source_table"])
        )
        for entry in entries
        if registered_source_columns([entry], str(entry["source_table"])) != expected
    }

    assert disagreements == {}
    assert expected == ("factor_name", "trade_date", "ts_code", "value")


def test_a_non_integer_factor_version_is_a_blocked_finding_that_names_the_field() -> None:
    """Catalog drift must come back as a finding, never as an exception that prints nothing.

    ``factor_version`` is one of the six required entry fields, but "required" only ever
    checked presence and non-emptiness. An entry hand-edited to ``v1`` therefore reached
    ``int()`` inside the sort key that both ``run_catalog_dry_run`` and
    ``registered_source_columns`` use, and raised a bare ``ValueError``. The CLI refuses a
    plan through ``except FactorPlanError``, which does not catch that, so
    ``qr-v2 materialize factors --dry-run`` exited 1 having printed nothing at all -- not
    even which factor was at fault.

    Fixing only the crash would be worse than the crash: the pre-existing ``try/except`` in
    ``_check_entry`` coerces an unparseable version to 0 and appends *no* reason, so the
    entry would have been reported MAPPED with a silently invented version. A version that
    cannot be parsed is exactly the drift this gate exists to report, so it is a reason.
    ``test_a_catalog_matching_the_live_source_schema_maps_completely`` is the control that
    keeps the new reason from firing on well-formed entries.
    """
    from research_v2.factors.catalog_dry_run import BLOCKED

    report = run_catalog_dry_run(
        [_entry(factor_version="v1")],
        _schemas(factor_data1=_LIVE_TYPES),
    )

    assert report.blocked_count == 1
    assert report.mapped_count == 0
    finding = report.blocked_findings[0]
    assert finding.status == BLOCKED
    assert finding.factor_id == "yeslab_1"
    assert any("factor_version" in reason for reason in finding.reasons), finding.reasons
    assert finding.factor_version == 0, (
        "a finding still has to carry a well-defined identity; 0 is the documented fallback"
    )


def test_a_missing_factor_version_is_still_reported_as_missing_and_not_as_unparseable() -> None:
    """The two version defects are different findings, and only one of them is new.

    An absent or empty ``factor_version`` was already caught by the required-field check.
    Both paths now end in BLOCKED, but they must not collapse into one another: a reason
    that says "unparseable" for a field that is simply absent would send whoever edits the
    catalog looking for a typo where the real problem is a dropped key.
    """
    report = run_catalog_dry_run(
        [_entry(factor_version="")],
        _schemas(factor_data1=_LIVE_TYPES),
    )

    assert report.blocked_count == 1
    reasons = report.blocked_findings[0].reasons
    assert any("missing required field: factor_version" in reason for reason in reasons), reasons
    assert not any("cannot be read as an integer" in reason for reason in reasons), (
        "an absent field is not an unparseable one, and saying so sends the editor hunting "
        "for a typo where the real problem is a dropped key"
    )


def test_a_version_that_is_not_integral_is_refused_rather_than_truncated() -> None:
    """``int(1.5) == 1`` would identify a different factor version, and say nothing about it.

    A wrong identity that still looks valid is the worst outcome available here: the entry
    would be sorted, retired-matched and reported as version 1 while the catalog says
    something else. Refusing costs one BLOCKED line.
    """
    report = run_catalog_dry_run(
        [_entry(factor_version=1.5)],
        _schemas(factor_data1=_LIVE_TYPES),
    )

    assert report.blocked_count == 1
    reasons = report.blocked_findings[0].reasons
    assert any("cannot be read as an integer" in reason for reason in reasons), reasons


def test_a_version_spelled_as_an_integral_float_is_the_entry_it_looks_like() -> None:
    """``1.0`` is how a hand-edited YAML file often spells 1, and it does mean version 1.

    The other half of refusing ``1.5``: being strict about integrality must not turn into
    being strict about spelling, or the gate blocks a catalog that is semantically fine.
    """
    report = run_catalog_dry_run(
        [_entry(factor_version=1.0)],
        _schemas(factor_data1=_LIVE_TYPES),
    )

    assert report.blocked_count == 0
    assert report.mapped_count == 1
    assert report.findings[0].factor_version == 1
