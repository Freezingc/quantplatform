from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.db.publish import PublishValidationError, V2BatchPublisher
from research_v2.factors.repair_persistence import (
    RepairAuditRecord,
    RepairPersistenceError,
    RepairQuarantineRow,
    build_repair_audit_batch,
    build_repair_identity,
    build_repair_quarantine_batch,
    repair_batch_quality_checks,
    repair_quality_checks,
)
from research_v2.quality.engine import can_publish


def _audit(
    *, factor_id: str = "yeslab_22", as_of_date: date = date(2025, 1, 31)
) -> RepairAuditRecord:
    return RepairAuditRecord(
        as_of_date=as_of_date,
        logical_partition="2025-01",
        factor_id=factor_id,
        factor_version=1,
        source_table="factor_data1",
        source_factor_name="GP_022",
        repair_method="KNOWN_FORMULA_RECONSTRUCTION",
        source_rows=10,
        candidate_rows=2,
        exact_duplicate_groups=0,
        exact_duplicate_rows=0,
        conflict_groups=0,
        nonfinite_rows=1,
        overlap_finite_rows=1,
        fillable_rows=1,
        filled_rows=1,
        added_rows=0,
        remaining_nonfinite_rows=0,
        mismatch_rows=0,
        max_abs_diff=0.0,
        max_rel_diff=0.0,
        pit_violation_rows=0,
        unresolved_rows=0,
        decision_status="REPAIR_AUDIT_PASS",
        rules_version=1,
        source_definition_hash="definition-hash-22",
        provider_hash="provider-hash-22",
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
    )


def _quarantine() -> RepairQuarantineRow:
    return RepairQuarantineRow(
        observation_date=date(2025, 1, 10),
        ts_code="000001.SZ",
        factor_id="yeslab_22",
        factor_version=1,
        factor_name="GP_022",
        reason="NO_FINITE_CANDIDATE",
        source_value=None,
        candidate_value=None,
        repair_method="KNOWN_FORMULA_RECONSTRUCTION",
        candidate_hash="candidate-hash-22",
        source_table="factor_data1",
        source_definition_hash="definition-hash-22",
        provider_hash="provider-hash-22",
        rules_version=1,
        start_date=date(2025, 1, 1),
        end_date=date(2025, 1, 31),
    )


def test_repair_batches_share_deterministic_run_and_data_version() -> None:
    audit_records = [_audit()]
    quarantine_rows = [_quarantine()]

    identity = build_repair_identity(audit_records, quarantine_rows)
    audit_batch = build_repair_audit_batch(audit_records, identity=identity)
    quarantine_batch = build_repair_quarantine_batch(quarantine_rows, identity=identity)

    assert quarantine_batch is not None
    assert audit_batch.object_name == "factor_repair_audit"
    assert quarantine_batch.object_name == "factor_repair_quarantine"
    assert audit_batch.frame.loc[0, "run_id"] == quarantine_batch.frame.loc[0, "run_id"]
    assert audit_batch.data_version == quarantine_batch.data_version
    assert audit_batch.frame.loc[0, "data_version"] == audit_batch.data_version
    assert (
        audit_batch.frame.loc[0, "source_watermark_id"]
        == quarantine_batch.frame.loc[0, "source_watermark_id"]
    )
    assert audit_batch.frame.loc[0, "rules_version"] == 1
    assert audit_batch.frame.columns.tolist() == [
        "as_of_date",
        "logical_partition",
        "factor_id",
        "factor_version",
        "source_table",
        "source_factor_name",
        "repair_method",
        "source_rows",
        "candidate_rows",
        "exact_duplicate_groups",
        "exact_duplicate_rows",
        "conflict_groups",
        "nonfinite_rows",
        "overlap_finite_rows",
        "fillable_rows",
        "filled_rows",
        "added_rows",
        "remaining_nonfinite_rows",
        "mismatch_rows",
        "max_abs_diff",
        "max_rel_diff",
        "pit_violation_rows",
        "unresolved_rows",
        "decision_status",
        "rules_version",
        "run_id",
        "data_version",
        "created_at",
        "source_watermark_id",
        "record_hash",
    ]
    assert audit_batch.frame.loc[0, "record_hash"]


def test_repair_identity_is_independent_of_input_order() -> None:
    first = _audit(factor_id="yeslab_22")
    second = _audit(factor_id="yeslab_64").__class__(
        **{**_audit(factor_id="yeslab_64").__dict__, "source_factor_name": "GP_064"}
    )

    left = build_repair_audit_batch([first, second])
    right = build_repair_audit_batch([second, first])

    assert left.data_version == right.data_version
    assert left.frame["factor_id"].tolist() == ["yeslab_22", "yeslab_64"]
    assert right.frame["factor_id"].tolist() == ["yeslab_22", "yeslab_64"]
    assert left.frame["record_hash"].tolist() == right.frame["record_hash"].tolist()


def test_no_arg_single_scope_builders_share_identity_when_lineage_matches() -> None:
    audit = _audit().__class__(
        **{**_audit().__dict__, "candidate_hash": "candidate-hash-22"}
    )
    quarantine = _quarantine()

    audit_batch = build_repair_audit_batch([audit])
    quarantine_batch = build_repair_quarantine_batch([quarantine])

    assert quarantine_batch is not None
    assert audit_batch.data_version == quarantine_batch.data_version
    assert audit_batch.frame.loc[0, "source_watermark_id"] == (
        quarantine_batch.frame.loc[0, "source_watermark_id"]
    )


def test_quality_checks_accept_an_explicit_generated_identity_watermark() -> None:
    audit = _audit()
    identity = build_repair_identity([audit])

    checks = repair_quality_checks(audit, identity=identity)

    watermark_check = next(item for item in checks if item.check_name == "repair_watermark")
    assert watermark_check.status.value == "PASS"


def test_month_partition_provides_shared_date_range_when_metadata_is_omitted() -> None:
    audit = _audit().__class__(
        **{**_audit().__dict__, "start_date": None, "end_date": None}
    )
    quarantine = _quarantine().__class__(
        **{**_quarantine().__dict__, "start_date": None, "end_date": None}
    )

    identity = build_repair_identity([audit], [quarantine])
    audit_batch = build_repair_audit_batch([audit], identity=identity)
    quarantine_batch = build_repair_quarantine_batch([quarantine], identity=identity)

    assert quarantine_batch is not None
    assert audit_batch.data_version == quarantine_batch.data_version


def test_duplicate_repair_audit_production_key_is_rejected_before_upload() -> None:
    batch = build_repair_audit_batch([_audit(), _audit()])

    with pytest.raises(PublishValidationError, match="duplicate business keys"):
        batch.validate()


def test_failed_repair_blocker_does_not_advance_pointer() -> None:
    class Session:
        target_database = "dfs://QuantResearchV2Test"

        def __init__(self) -> None:
            self.uploads: list[tuple[str, pd.DataFrame]] = []
            self.scripts: list[str] = []

        def upload_frame(self, name: str, frame: pd.DataFrame) -> None:
            self.uploads.append((name, frame.copy()))

        def execute(self, script: str) -> object:
            self.scripts.append(script)
            return None

        def query(self, script: str) -> object:
            if script.startswith("schema("):
                return pd.DataFrame(
                    {
                        "name": ["as_of_date", "run_id"],
                        "typeString": ["DATE", "UUID"],
                    }
                )
            if "count(*) as row_count" in script:
                return pd.DataFrame({"row_count": [0]})
            return pd.DataFrame(columns=["data_version"])

    blocked = _audit().__class__(
        **{
            **_audit().__dict__,
            "conflict_groups": 1,
            "decision_status": "REPAIR_BLOCKED",
        }
    )
    batch = build_repair_audit_batch([blocked])
    checks = repair_quality_checks(blocked)

    with pytest.raises(PublishValidationError, match="data-quality"):
        V2BatchPublisher(Session()).publish_chunk_stream([batch], checks)


def test_empty_quarantine_does_not_create_a_batch() -> None:
    assert build_repair_quarantine_batch([]) is None


def test_unknown_or_unresolved_repair_status_fails_closed() -> None:
    for status, unresolved_rows in (
        ("UNKNOWN_STATUS", 0),
        ("REPAIR_AUDIT_PASS_WITH_UNRESOLVED_QUARANTINE", 1),
    ):
        audit = _audit().__class__(
            **{
                **_audit().__dict__,
                "decision_status": status,
                "unresolved_rows": unresolved_rows,
                "source_watermark_id": "watermark",
            }
        )
        checks = repair_quality_checks(audit)
        assert any(result.status.value == "FAIL" for result in checks)

    blocked = _audit().__class__(
        **{
            **_audit().__dict__,
            "decision_status": "REPAIR_BLOCKED",
            "source_watermark_id": "watermark",
        }
    )
    assert all(result.status.value == "PASS" for result in repair_quality_checks(blocked)) is False


def test_shared_scope_identity_allows_quarantine_subset() -> None:
    audit_x = _audit(factor_id="yeslab_22")
    audit_y = _audit(factor_id="yeslab_64").__class__(
        **{
            **_audit(factor_id="yeslab_64").__dict__,
            "source_factor_name": "GP_064",
            "candidate_hash": "candidate-hash-64",
        }
    )
    quarantine_x = _quarantine()
    identity = build_repair_identity([audit_x, audit_y], [quarantine_x])

    audit_batch = build_repair_audit_batch([audit_x, audit_y], identity=identity)
    quarantine_batch = build_repair_quarantine_batch([quarantine_x], identity=identity)

    assert quarantine_batch is not None
    assert audit_batch.data_version == quarantine_batch.data_version == identity.data_version
    assert audit_batch.frame.loc[0, "run_id"] == quarantine_batch.frame.loc[0, "run_id"]
    assert audit_batch.frame.loc[0, "source_watermark_id"] == identity.source_watermark_id


def test_mixed_calendar_months_are_rejected_for_each_batch() -> None:
    february = _audit(as_of_date=date(2025, 2, 28)).__class__(
        **{**_audit(as_of_date=date(2025, 2, 28)).__dict__, "logical_partition": "2025-02"}
    )
    with pytest.raises(RepairPersistenceError, match="calendar month"):
        build_repair_audit_batch([_audit(), february])

    mismatched_partition = _audit().__class__(
        **{**_audit().__dict__, "logical_partition": "2025-02"}
    )
    with pytest.raises(RepairPersistenceError, match="logical_partition"):
        build_repair_audit_batch([mismatched_partition])

    january_quarantine = _quarantine()
    february_quarantine = january_quarantine.__class__(
        **{**january_quarantine.__dict__, "observation_date": date(2025, 2, 3)}
    )
    with pytest.raises(RepairPersistenceError, match="calendar month"):
        build_repair_quarantine_batch([january_quarantine, february_quarantine])


def test_identity_changes_when_repair_provenance_changes() -> None:
    baseline = build_repair_identity([_audit()], [_quarantine()])
    changed = _audit().__class__(
        **{
            **_audit().__dict__,
            "candidate_hash": "candidate-hash-new",
            "repair_method": "SECONDARY_SOURCE",
        }
    )
    changed_identity = build_repair_identity([changed], [_quarantine()])

    assert baseline.data_version != changed_identity.data_version
    assert baseline.source_watermark_id != changed_identity.source_watermark_id


def test_quarantine_rules_version_must_be_positive() -> None:
    invalid = _quarantine().__class__(**{**_quarantine().__dict__, "rules_version": 0})

    with pytest.raises(RepairPersistenceError, match="identity is invalid"):
        build_repair_quarantine_batch([invalid])


def test_batch_quality_checks_prefix_factor_ids_and_pass_publish_gate() -> None:
    audit_x = _audit(factor_id="yeslab_22")
    audit_y = _audit(factor_id="yeslab_64").__class__(
        **{**_audit(factor_id="yeslab_64").__dict__, "source_factor_name": "GP_064"}
    )
    identity = build_repair_identity([audit_x, audit_y])

    checks = repair_batch_quality_checks([audit_x, audit_y], identity=identity)

    assert len({result.check_name for result in checks}) == len(checks)
    assert all(result.check_name.startswith(("yeslab_22:", "yeslab_64:")) for result in checks)
    assert can_publish(checks)

    class Session:
        target_database = "dfs://QuantResearchV2Test"

        def __init__(self) -> None:
            self.uploads: list[tuple[str, pd.DataFrame]] = []
            self.scripts: list[str] = []

        def upload_frame(self, name: str, frame: pd.DataFrame) -> None:
            self.uploads.append((name, frame.copy()))

        def execute(self, script: str) -> object:
            self.scripts.append(script)
            return None

        def query(self, script: str) -> object:
            if script.startswith("schema("):
                return pd.DataFrame({"name": ["as_of_date"], "typeString": ["DATE"]})
            if "count(*) as row_count" in script:
                return pd.DataFrame({"row_count": [0]})
            return pd.DataFrame(columns=["data_version"])

    session = Session()
    V2BatchPublisher(session).publish_chunk_stream(
        [build_repair_audit_batch([audit_x, audit_y], identity=identity)],
        checks,
    )
    assert session.uploads
    assert "published_version" in session.scripts[-1]


def test_builder_rejects_record_outside_explicit_identity_scope() -> None:
    identity = build_repair_identity([_audit(factor_id="yeslab_22")])
    foreign = _audit(factor_id="yeslab_64").__class__(
        **{**_audit(factor_id="yeslab_64").__dict__, "source_factor_name": "GP_064"}
    )

    with pytest.raises(RepairPersistenceError, match="identity scope"):
        build_repair_audit_batch([foreign], identity=identity)


def test_factor_version_is_part_of_identity_scope() -> None:
    version_two = _audit().__class__(**{**_audit().__dict__, "factor_version": 2})
    identity = build_repair_identity([_audit()])

    assert identity.data_version != build_repair_identity([version_two]).data_version
    with pytest.raises(RepairPersistenceError, match="identity scope"):
        build_repair_audit_batch([version_two], identity=identity)


def test_quarantine_only_builder_requires_provenance_hashes() -> None:
    malformed = _quarantine().__class__(
        **{
            **_quarantine().__dict__,
            "source_definition_hash": "",
            "provider_hash": "",
        }
    )

    with pytest.raises(RepairPersistenceError, match="provenance"):
        build_repair_quarantine_batch([malformed])
