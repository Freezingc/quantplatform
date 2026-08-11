from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.db.publish import StagedBatch
from research_v2.factors.materializer import FactorMaterializationBatches
from research_v2.factors.source_backfill import (
    FactorBackfillError,
    FactorMonth,
    audit_factor_materialization_batches,
    build_factor_definitions,
    build_factor_month_context,
    iter_clean_factor_batches,
    preflight_factor_month,
)


def _metadata() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "factor_id": [1, 2],
            "factor_name": ["alpha", "beta"],
            "creator": ["research", "research"],
            "update_frequency": ["daily", "daily"],
            "store_factor": ["factor_data1", "factor_data1"],
        }
    )


class SourceSession:
    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.queries: list[str] = []

    def query(self, script: str) -> object:
        self.queries.append(script)
        return self.frame.copy(deep=True)


def test_factor_definition_builder_maps_catalog_metadata_to_immutable_sources() -> None:
    definitions = build_factor_definitions(_metadata(), ("yeslab_2", "yeslab_1"))

    assert [definition.factor_id for definition in definitions] == ["yeslab_1", "yeslab_2"]
    assert definitions[0].display_name == "alpha"
    assert definitions[1].source_table == "factor_data1"


def test_factor_month_iterator_queries_only_selected_names_and_returns_clean_batches() -> None:
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "trade_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "factor_name": ["alpha", "beta"],
            "value": [1.0, 2.0],
        }
    )
    session = SourceSession(source)
    definitions = build_factor_definitions(_metadata(), ("yeslab_1", "yeslab_2"))
    month = FactorMonth.from_partition("2025-01")
    context = build_factor_month_context(month, definitions)

    batches = list(iter_clean_factor_batches(session, definitions, month, context))

    assert len(batches) == 2
    assert all(batch.logical_partition == "2025-01" for batch in batches)
    assert all(batch.data_version == context.data_version for batch in batches)
    assert "factor_name in ['alpha', 'beta']" in session.queries[0]
    assert "date('2025.01.01')" in session.queries[0]
    assert "date('2025.01.31')" in session.queries[0]


def test_factor_month_iterator_fails_closed_when_a_requested_factor_is_absent() -> None:
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ"],
            "trade_date": [date(2025, 1, 2)],
            "factor_name": ["alpha"],
            "value": [1.0],
        }
    )
    definitions = build_factor_definitions(_metadata(), ("yeslab_1", "yeslab_2"))
    month = FactorMonth.from_partition("2025-01")
    context = build_factor_month_context(month, definitions)

    with pytest.raises(FactorBackfillError, match="no clean observations"):
        list(iter_clean_factor_batches(SourceSession(source), definitions, month, context))


def test_factor_month_rejects_invalid_partition() -> None:
    with pytest.raises(FactorBackfillError, match="YYYY-MM"):
        FactorMonth.from_partition("2025-1")


def test_factor_month_retry_nonce_changes_version_without_changing_factor_scope() -> None:
    definitions = build_factor_definitions(_metadata(), ("yeslab_1", "yeslab_2"))
    month = FactorMonth.from_partition("2025-01")

    first = build_factor_month_context(month, definitions)
    retry = build_factor_month_context(month, definitions, attempt_nonce="retry-1")

    assert first.data_version != retry.data_version
    with pytest.raises(FactorBackfillError, match="attempt_nonce"):
        build_factor_month_context(month, definitions, attempt_nonce=" ")


def test_materialization_audit_counts_clean_and_quarantine_rows_by_factor() -> None:
    clean = StagedBatch.create(
        object_name="factor_observation_clean",
        logical_partition="2025-01",
        data_version="version-a",
        frame=pd.DataFrame(
            {
                "factor_id": ["yeslab_1", "yeslab_1", "yeslab_2"],
                "signal_date": [date(2025, 1, 2)] * 3,
                "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ"],
            }
        ),
        business_keys=("signal_date", "ts_code", "factor_id"),
    )
    quarantine = StagedBatch.create(
        object_name="factor_quarantine",
        logical_partition="2025-01",
        data_version="version-a",
        frame=pd.DataFrame(
            {
                "factor_id": ["yeslab_1", "yeslab_2"],
                "observation_date": [date(2025, 1, 2)] * 2,
                "ts_code": ["000001.SZ", "000002.SZ"],
                "reason": ["CONFLICT_QUARANTINE", "CONFLICT_QUARANTINE"],
            }
        ),
        business_keys=("observation_date", "ts_code", "factor_id"),
    )

    audit = audit_factor_materialization_batches(
        [FactorMaterializationBatches(clean, quarantine)],
        factor_ids=("yeslab_1", "yeslab_2"),
    )

    assert audit.logical_partition == "2025-01"
    assert audit.data_version == "version-a"
    assert audit.clean_rows_by_factor == (("yeslab_1", 2), ("yeslab_2", 1))
    assert audit.quarantine_rows_by_factor == (("yeslab_1", 1), ("yeslab_2", 1))
    assert audit.quarantine_reasons_by_factor == (
        ("yeslab_1", "CONFLICT_QUARANTINE", 1),
        ("yeslab_2", "CONFLICT_QUARANTINE", 1),
    )
    assert audit.total_clean_rows == 3
    assert audit.total_quarantine_rows == 2


def test_materialization_audit_rejects_mixed_publish_identity() -> None:
    first = StagedBatch.create(
        object_name="factor_observation_clean",
        logical_partition="2025-01",
        data_version="version-a",
        frame=pd.DataFrame(
            {"factor_id": ["yeslab_1"], "signal_date": [date(2025, 1, 2)], "ts_code": ["000001.SZ"]}
        ),
        business_keys=("signal_date", "ts_code", "factor_id"),
    )
    second = StagedBatch.create(
        object_name="factor_observation_clean",
        logical_partition="2025-01",
        data_version="version-b",
        frame=pd.DataFrame(
            {"factor_id": ["yeslab_2"], "signal_date": [date(2025, 1, 2)], "ts_code": ["000002.SZ"]}
        ),
        business_keys=("signal_date", "ts_code", "factor_id"),
    )

    with pytest.raises(FactorBackfillError, match="publish identity"):
        audit_factor_materialization_batches(
            [FactorMaterializationBatches(first, None), FactorMaterializationBatches(second, None)]
        )


def test_preflight_factor_month_materializes_once_without_target_writes() -> None:
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ"],
            "trade_date": [date(2025, 1, 2), date(2025, 1, 2)],
            "factor_name": ["alpha", "beta"],
            "value": [1.0, 2.0],
        }
    )
    session = SourceSession(source)

    audit = preflight_factor_month(
        session,
        _metadata(),
        factor_ids=("yeslab_1", "yeslab_2"),
        logical_partition="2025-01",
        attempt_nonce="preflight",
    )

    assert audit.logical_partition == "2025-01"
    assert audit.total_clean_rows == 2
    assert audit.total_quarantine_rows == 0
    assert len(session.queries) == 1


def test_clean_stream_allows_missing_values_but_blocks_conflicting_duplicates() -> None:
    source = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "000002.SZ", "000003.SZ"],
            "trade_date": [date(2025, 1, 2)] * 3,
            "factor_name": ["alpha"] * 3,
            "value": [1.0, None, 2.0],
        }
    )
    definitions = build_factor_definitions(_metadata(), ("yeslab_1",))
    month = FactorMonth.from_partition("2025-01")
    context = build_factor_month_context(month, definitions)

    batches = list(
        iter_clean_factor_batches(SourceSession(source), definitions, month, context)
    )
    assert len(batches) == 1
    assert len(batches[0].frame) == 2

    conflict = pd.concat(
        [
            source,
            pd.DataFrame(
                {
                    "ts_code": ["000003.SZ"],
                    "trade_date": [date(2025, 1, 2)],
                    "factor_name": ["alpha"],
                    "value": [3.0],
                }
            ),
        ],
        ignore_index=True,
    )
    with pytest.raises(FactorBackfillError, match="quarantined conflict"):
        list(iter_clean_factor_batches(SourceSession(conflict), definitions, month, context))
