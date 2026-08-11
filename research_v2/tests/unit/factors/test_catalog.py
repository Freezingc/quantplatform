from __future__ import annotations

import pandas as pd
import pytest

from research_v2.factors.catalog import (
    CatalogBuildError,
    build_yeslab_catalog,
    build_yeslab_catalog_document,
)


def test_catalog_builds_explicit_entries_for_each_metadata_factor() -> None:
    metadata = pd.DataFrame(
        {
            "factor_id": [1, 2],
            "factor_name": ["alpha", "beta"],
            "creator": ["research", "research"],
            "update_frequency": ["daily", "daily"],
            "store_factor": ["factor_data1", "factor_data2"],
        }
    )

    catalog = build_yeslab_catalog(metadata)

    assert [entry["factor_id"] for entry in catalog] == ["yeslab_1", "yeslab_2"]
    assert catalog[0]["source_table"] == "factor_data1"
    assert catalog[0]["source_factor_name"] == "alpha"
    assert catalog[0]["availability_policy"] == "SAME_DAY_CLOSE"


def test_catalog_rejects_unmapped_or_ambiguous_source_metadata() -> None:
    metadata = pd.DataFrame(
        {
            "factor_id": [1, 1],
            "factor_name": ["alpha", "alpha2"],
            "creator": ["research", "research"],
            "update_frequency": ["daily", "daily"],
            "store_factor": ["factor_data1", "unknown"],
        }
    )

    with pytest.raises(CatalogBuildError, match="duplicate"):
        build_yeslab_catalog(metadata)


def test_catalog_document_has_a_stable_metadata_snapshot_hash() -> None:
    metadata = pd.DataFrame(
        {
            "factor_id": [2, 1],
            "factor_name": ["beta", "alpha"],
            "creator": ["research", "research"],
            "update_frequency": ["daily", "daily"],
            "store_factor": ["factor_data2", "factor_data1"],
        }
    )

    document = build_yeslab_catalog_document(metadata)

    assert len(str(document["metadata_snapshot_hash"])) == 64
    assert [entry["factor_id"] for entry in document["factors"]] == ["yeslab_1", "yeslab_2"]
