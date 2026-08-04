from __future__ import annotations

import json
from datetime import date

import pandas as pd
import pytest
from run_factor_repair_overlay import (
    SOURCE_SCOPE_ID,
    SOURCE_STATUS,
    main,
    parse_args,
    resolve_factor_ids,
)


class _FakeTarget:
    target_database = "dfs://QuantResearchV2Test"

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.queries: list[str] = []

    def query(self, script: str) -> pd.DataFrame:
        self.queries.append(script)
        if "factor_quality_snapshot" in script:
            return self.frame
        return pd.DataFrame()


class _FakeSource:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def query(self, script: str) -> pd.DataFrame:
        self.queries.append(script)
        if "factor_metadata" in script:
            return pd.DataFrame(
                {
                    "factor_id": [1],
                    "factor_name": ["GP_001"],
                    "creator": ["unit-test"],
                    "update_frequency": ["DAILY"],
                    "store_factor": ["factor_data1"],
                }
            )
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ"],
                "trade_date": [pd.Timestamp("2020-01-02")],
                "factor_name": ["GP_001"],
                "value": [1.0],
            }
        )

    def execute(self, script: str) -> None:
        raise AssertionError(f"source session write attempted: {script}")


def test_runner_resolves_only_source_qualified_scope_rows() -> None:
    target = _FakeTarget(
        pd.DataFrame(
            {
                "factor_id": ["yeslab_1", "yeslab_2", "yeslab_3"],
                "metrics_json": [
                    json.dumps({"decision_status": SOURCE_STATUS}),
                    json.dumps({"decision_status": "SOURCE_DQ_FAIL"}),
                    json.dumps({"decision_status": SOURCE_STATUS}),
                ],
            }
        )
    )

    assert resolve_factor_ids(target) == ("yeslab_1", "yeslab_3")
    assert len(target.queries) == 1
    assert SOURCE_SCOPE_ID in target.queries[0]
    assert "locked_test_audit" not in target.queries[0]


def test_parser_defaults_to_dry_run_train_scope() -> None:
    args = parse_args(["--phase", "train", "--start", "2020-01-01", "--end", "2024-12-31"])

    assert args.phase == "train"
    assert args.start == date(2020, 1, 1)
    assert args.end == date(2024, 12, 31)
    assert args.publish is False


def test_parser_rejects_unknown_phase_and_locked_test_end_date() -> None:
    with pytest.raises(SystemExit):
        parse_args(["--phase", "test", "--start", "2020-01-01", "--end", "2024-12-31"])

    with pytest.raises(ValueError, match="2026"):
        parse_args(["--phase", "train", "--start", "2025-01-01", "--end", "2026-01-01"])


def test_explicit_factor_id_must_be_in_resolved_scope() -> None:
    target = _FakeTarget(
        pd.DataFrame(
            {
                "factor_id": ["yeslab_1"],
                "metrics_json": [json.dumps({"decision_status": SOURCE_STATUS})],
            }
        )
    )
    with pytest.raises(ValueError, match="resolved source-qualified"):
        resolve_factor_ids(target, factor_id="yeslab_9")


def test_main_dry_run_reads_bounded_source_and_never_writes_source() -> None:
    target = _FakeTarget(
        pd.DataFrame(
            {
                "factor_id": ["yeslab_1"],
                "metrics_json": [json.dumps({"decision_status": SOURCE_STATUS})],
            }
        )
    )
    source = _FakeSource()
    result = main(
        ["--phase", "train", "--start", "2020-01-01", "--end", "2020-01-31"],
        source_session=source,
        target_session=target,
    )

    assert result.published is False
    assert result.counts["audit_rows"] == 1
    source_queries = "\n".join(source.queries)
    assert "trade_date between date('2020.01.01')" in source_queries
    assert "2026" not in source_queries
