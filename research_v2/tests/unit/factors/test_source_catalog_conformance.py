"""C1.1: the shipped catalog must be an explicit, closed, per-factor description of the source.

Every clause here is checked against the real ``configs/factors/source_catalog.yaml`` rather than a
fixture, because the property that matters is about the artifact the pipeline actually loads. A
synthetic catalog could satisfy all of it while the shipped one did not.

The complementary property -- that this file is exactly what the frozen metadata snapshot expands
to, with no factor added or dropped -- needs the source cluster and lives in
``scripts/verify_factor_catalog_reproducibility.py``. On 2026-09-06 that check reproduced the stored
``metadata_snapshot_hash`` and 487 byte-identical entries from the live ``factor_metadata`` table.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest
import yaml

from research_v2.factors.catalog import _ALLOWED_TABLES, CatalogBuildError, build_yeslab_catalog

_CATALOG_PATH = Path(__file__).resolve().parents[3] / "configs" / "factors" / "source_catalog.yaml"

#: The fifteen fields C1.1 requires on every record.
_REQUIRED_FIELDS = (
    "factor_id",
    "factor_version",
    "display_name",
    "source_database",
    "source_table",
    "source_value_column",
    "source_date_column",
    "source_code_column",
    "factor_type",
    "frequency",
    "expected_direction",
    "availability_policy",
    "max_staleness_calendar_days",
    "unit",
    "owner",
)

_WILDCARD = re.compile(r"[*?\[\]]")


@pytest.fixture(scope="module")
def document() -> dict[str, Any]:
    return yaml.safe_load(_CATALOG_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def entries(document: dict[str, Any]) -> list[dict[str, Any]]:
    return list(document["factors"])


def test_the_catalog_is_not_empty_and_is_a_document_with_a_snapshot_hash(
    document: dict[str, Any], entries: list[dict[str, Any]]
) -> None:
    assert entries
    assert re.fullmatch(r"[0-9a-f]{64}", str(document["metadata_snapshot_hash"])), (
        "the frozen metadata snapshot hash must be recorded so the catalog can be re-derived"
    )


def test_every_record_carries_all_fifteen_required_fields(
    entries: list[dict[str, Any]]
) -> None:
    incomplete = {
        str(entry.get("factor_id")): sorted(set(_REQUIRED_FIELDS).difference(entry))
        for entry in entries
        if set(_REQUIRED_FIELDS).difference(entry)
    }

    assert incomplete == {}


def test_no_required_field_is_present_but_empty(entries: list[dict[str, Any]]) -> None:
    """A field set to None or '' passes a key-presence check while describing nothing."""
    blank = {
        str(entry.get("factor_id")): sorted(
            field for field in _REQUIRED_FIELDS if entry.get(field) in (None, "")
        )
        for entry in entries
    }

    assert {key: value for key, value in blank.items() if value} == {}


def test_each_factor_has_exactly_one_entry(entries: list[dict[str, Any]]) -> None:
    ids = [str(entry["factor_id"]) for entry in entries]
    keys = [(str(entry["factor_id"]), int(entry["factor_version"])) for entry in entries]

    assert len(set(ids)) == len(ids), "a factor_id must not appear twice"
    assert len(set(keys)) == len(keys)
    assert all(re.fullmatch(r"yeslab_[0-9]+", factor_id) for factor_id in ids)


def test_no_entry_can_accept_a_factor_by_wildcard(entries: list[dict[str, Any]]) -> None:
    """C1.1 forbids runtime ``GP_*`` patterns: a glob here would silently admit new source columns.

    The failure mode is the one the whole registry exists to prevent -- a factor nobody defined,
    versioned or assigned an owner entering the pipeline because its name happened to match.
    """
    offenders = {
        str(entry["factor_id"]): sorted(
            field
            for field in ("factor_id", "display_name", "source_factor_name", "source_table")
            if _WILDCARD.search(str(entry.get(field, "")))
        )
        for entry in entries
    }

    assert {key: value for key, value in offenders.items() if value} == {}


def test_the_allowed_source_table_set_is_closed_and_exactly_the_frozen_eleven() -> None:
    expected = frozenset(f"factor_data{index}" for index in range(1, 12))

    assert expected == _ALLOWED_TABLES


def test_every_entry_names_a_table_inside_the_frozen_set(entries: list[dict[str, Any]]) -> None:
    outside = sorted(
        {str(entry["source_table"]) for entry in entries}.difference(_ALLOWED_TABLES)
    )

    assert outside == []


def test_the_source_binding_is_fixed_to_the_yeslab_long_form_contract(
    entries: list[dict[str, Any]]
) -> None:
    """C1.1 fixes these: any other value means the catalog describes a source that is not there."""
    assert {str(entry["source_database"]) for entry in entries} == {"dfs://YesLab"}
    assert {str(entry["source_value_column"]) for entry in entries} == {"value"}
    assert {str(entry["source_date_column"]) for entry in entries} == {"trade_date"}
    assert {str(entry["source_code_column"]) for entry in entries} == {"ts_code"}


def test_every_entry_declares_a_type_frequency_direction_policy_and_owner(
    entries: list[dict[str, Any]]
) -> None:
    from research_v2.contracts.factor import AvailabilityPolicy, FactorType

    factor_types = {str(entry["factor_type"]) for entry in entries}
    policies = {str(entry["availability_policy"]) for entry in entries}

    assert factor_types <= {member.value for member in FactorType}
    assert policies <= {member.value for member in AvailabilityPolicy}
    assert {str(entry["frequency"]) for entry in entries} == {"DAILY"}
    assert all(int(entry["expected_direction"]) in {-1, 0, 1} for entry in entries)
    assert all(int(entry["max_staleness_calendar_days"]) >= 0 for entry in entries)
    assert all(str(entry["owner"]).strip() for entry in entries), (
        "every factor must have a named owner"
    )
    assert all(str(entry["unit"]).strip() for entry in entries)


def test_technical_factors_carry_no_staleness_allowance(entries: list[dict[str, Any]]) -> None:
    """A technical factor is available at same-day close; any staleness here would permit ffill."""
    technical = [entry for entry in entries if str(entry["factor_type"]) == "TECHNICAL"]

    assert technical, "the catalog is expected to describe technical factors"
    assert all(int(entry["max_staleness_calendar_days"]) == 0 for entry in technical)
    assert all(
        str(entry["availability_policy"]) == "SAME_DAY_CLOSE" for entry in technical
    )


def test_the_generator_refuses_a_table_outside_the_frozen_set() -> None:
    """The closed set is enforced at generation time, not only checked afterwards."""
    import pandas as pd

    metadata = pd.DataFrame(
        [
            {
                "factor_id": 1,
                "factor_name": "Alpha1",
                "creator": "research",
                "update_frequency": "DAILY",
                "store_factor": "factor_data12",
            }
        ]
    )

    with pytest.raises(CatalogBuildError, match="factor_data12"):
        build_yeslab_catalog(metadata)


def test_the_generator_refuses_a_runtime_wildcard_factor_name() -> None:
    """A ``GP_*`` row in the source metadata must not become a catalog entry at all.

    C1.1 forbids accepting factors by wildcard. Recording ``GP_*`` literally would not expand it,
    but it would put a name into the catalog that matches nothing and describes no single factor --
    and the read path filters ``where factor_name in [...]``, so such an entry selects either
    nothing or, if the source ever stored a literal glob, something nobody defined. Refusing at
    generation time is what makes the clause hold for the *next* catalog, not just this one.
    """
    import pandas as pd

    metadata = pd.DataFrame(
        [
            {
                "factor_id": 1,
                "factor_name": "GP_*",
                "creator": "research",
                "update_frequency": "DAILY",
                "store_factor": "factor_data1",
            }
        ]
    )

    with pytest.raises(CatalogBuildError, match=r"GP_\*"):
        build_yeslab_catalog(metadata)


def test_a_wildcard_anywhere_in_a_generated_name_is_refused_not_only_a_trailing_star() -> None:
    import pandas as pd

    for name in ("Alpha?", "Alpha[1-9]", "Alpha*", "*"):
        metadata = pd.DataFrame(
            [
                {
                    "factor_id": 1,
                    "factor_name": name,
                    "creator": "research",
                    "update_frequency": "DAILY",
                    "store_factor": "factor_data1",
                }
            ]
        )

        with pytest.raises(CatalogBuildError):
            build_yeslab_catalog(metadata)
