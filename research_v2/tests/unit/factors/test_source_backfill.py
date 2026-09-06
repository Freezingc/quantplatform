from __future__ import annotations

import re
from datetime import date

import pandas as pd
import pytest

from research_v2.db.publish import StagedBatch
from research_v2.factors.materializer import FactorMaterializationBatches
from research_v2.factors.source_backfill import (
    _REPUBLISHED_UNCHANGED,
    FactorBackfillError,
    FactorMonth,
    _duplicate_payload_conflicts,
    _iter_client_repair_batches,
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


_REPAIR_KEYS = ["signal_date", "ts_code", "factor_id", "factor_version"]


class _RepairTargetSession:
    """Stands in for the target session, honouring the day the path asks about.

    ``_iter_client_repair_batches`` walks every calendar day of the month and queries one day
    at a time, so a stub that returned the whole frame for every day would multiply each row
    by the length of the month and hide exactly the per-day deduplication under test.
    """

    target_database = "dfs://QuantResearchV2Test"

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.queries: list[str] = []

    def query(self, script: str) -> object:
        self.queries.append(script)
        asked = re.search(r"signal_date=date\('(\d{4})\.(\d{2})\.(\d{2})'\)", script)
        assert asked is not None, f"the day filter changed shape; update the stub: {script[:120]}"
        wanted = date(int(asked.group(1)), int(asked.group(2)), int(asked.group(3)))
        held = self.frame.loc[
            pd.to_datetime(self.frame["signal_date"]).dt.date == wanted
        ]
        return held.copy(deep=True)


def _repair_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "signal_date": date(2025, 1, 2),
        "ts_code": "000001.SZ",
        "factor_id": "yeslab_1",
        "factor_version": 1,
        "value": 1.5,
        "observation_date": date(2025, 1, 2),
        "available_date": date(2025, 1, 2),
        "report_period": None,
        "stale_days": 0,
        "dedup_resolution": "UNIQUE",
        "source_revision_id": "r1",
        "run_id": "run-a",
        "data_version": "v-old",
        "created_at": pd.Timestamp("2025-01-02 00:00:00"),
        "source_watermark_id": "w1",
        "record_hash": "h-a",
    }
    row.update(overrides)
    return row


def test_duplicates_differing_only_on_overwritten_bookkeeping_are_not_a_conflict() -> None:
    """The shape this recovery path exists for must not be refused.

    An interrupted chunk stream re-writes rows, so duplicates legitimately disagree on
    ``run_id``, ``data_version``, ``created_at`` and ``record_hash``. The path overwrites all
    four, so none of them can reach the published version and none of them may trigger the
    guard -- otherwise the recovery path could never run on the data it was written for.
    """
    frame = pd.DataFrame(
        [_repair_row(), _repair_row(run_id="run-b", data_version="v-old-2",
                                    created_at=pd.Timestamp("2025-01-03 00:00:00"),
                                    record_hash="h-b")]
    )
    assert _duplicate_payload_conflicts(frame, _REPAIR_KEYS, _REPUBLISHED_UNCHANGED) == []

    batches = list(
        _iter_client_repair_batches(
            _RepairTargetSession(frame),
            old_data_version="v-old",
            new_data_version="v-new",
            month=FactorMonth.from_partition("2025-01"),
        )
    )
    # one day with rows -> one batch, duplicates collapsed, bookkeeping re-stamped
    payload = pd.concat([batch.frame for batch in batches], ignore_index=True)
    assert len(payload) == 1
    assert payload["data_version"].tolist() == ["v-new"]
    assert payload["value"].tolist() == [1.5]


def test_duplicates_disagreeing_on_a_republished_column_are_refused_not_resolved() -> None:
    """The sort cannot break this tie, so the path must refuse instead of publishing a value
    that whichever row the database happened to return first decided.

    ``_iter_client_repair_batches`` sorts by ``keys`` and then drops duplicates on ``keys``, so
    every row in a duplicate group is equal on all of the sort keys by construction -- that
    ``sort_values`` is a no-op for tie-breaking and ``keep="first"`` means "first returned".
    Unlike ``repair.py`` and ``compact_coverage.py``, whose tied rows are identical after their
    conflict quarantine (and which then project only keys plus value), this frame carries eight
    columns through unchanged, so a disagreement here would reach the published version.
    """
    for column, other in (("value", 2.5), ("source_revision_id", "r2"),
                          ("dedup_resolution", "SINGLE_NONNULL"), ("stale_days", 3)):
        frame = pd.DataFrame([_repair_row(), _repair_row(**{column: other})])
        conflicts = _duplicate_payload_conflicts(frame, _REPAIR_KEYS, _REPUBLISHED_UNCHANGED)
        assert len(conflicts) == 1, column
        assert conflicts[0][1] == [column], column
        with pytest.raises(FactorBackfillError, match="row order"):
            list(
                _iter_client_repair_batches(
                    _RepairTargetSession(frame),
                    old_data_version="v-old",
                    new_data_version="v-new",
                    month=FactorMonth.from_partition("2025-01"),
                )
            )


def test_the_guard_is_indifferent_to_row_order_which_is_the_point_of_it() -> None:
    """A guard that only fires in one row order would be as order-dependent as the code it
    replaces, so both permutations must be refused."""
    frame = pd.DataFrame([_repair_row(), _repair_row(value=2.5)])
    for candidate in (frame, frame.iloc[::-1].reset_index(drop=True)):
        with pytest.raises(FactorBackfillError, match="row order"):
            list(
                _iter_client_repair_batches(
                    _RepairTargetSession(candidate),
                    old_data_version="v-old",
                    new_data_version="v-new",
                    month=FactorMonth.from_partition("2025-01"),
                )
            )


def test_distinct_business_keys_are_never_treated_as_duplicates() -> None:
    """The guard must fire on ties only, or it would block ordinary multi-row days."""
    frame = pd.DataFrame(
        [_repair_row(), _repair_row(ts_code="000002.SZ", value=9.0),
         _repair_row(factor_id="yeslab_2", value=7.0)]
    )
    assert _duplicate_payload_conflicts(frame, _REPAIR_KEYS, _REPUBLISHED_UNCHANGED) == []
    batches = list(
        _iter_client_repair_batches(
            _RepairTargetSession(frame),
            old_data_version="v-old",
            new_data_version="v-new",
            month=FactorMonth.from_partition("2025-01"),
        )
    )
    payload = pd.concat([batch.frame for batch in batches], ignore_index=True)
    assert len(payload) == 3
