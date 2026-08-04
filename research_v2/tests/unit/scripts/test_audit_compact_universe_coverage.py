from __future__ import annotations

from datetime import date

import audit_compact_universe_coverage as audit
import pandas as pd
import pytest


class _Source:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def query(self, script: str) -> pd.DataFrame:
        self.queries.append(script)
        if "factor_metadata" in script:
            return pd.DataFrame(
                {"factor_id": [395], "factor_name": ["GP_026"], "store_factor": ["factor_data9"]}
            )
        if "count(*) as finite_keys" in script:
            return pd.DataFrame(
                {
                    "factor_name": ["GP_026"],
                    "trade_date": [date(2020, 12, 1)],
                    "finite_keys": [1],
                }
            )
        return pd.DataFrame(
            {
                "trade_date": [date(2020, 12, 1), date(2020, 12, 1)],
                "ts_code": ["000001.SZ", "000002.SZ"],
                "value": [1.0, float("nan")],
            }
        )


class _Target:
    def __init__(self) -> None:
        self.queries: list[str] = []
        self.uploads: dict[str, pd.DataFrame] = {}

    def upload_frame(self, name: str, frame: pd.DataFrame) -> None:
        self.uploads[name] = frame.copy()

    def query(self, script: str) -> pd.DataFrame:
        self.queries.append(script)
        if "universe_daily" in script:
            return pd.DataFrame(
                {
                    "signal_date": [date(2020, 12, 1)] * 2,
                    "ts_code": ["000001.SZ", "000002.SZ"],
                    "is_member": [True, True],
                }
            )
        return pd.DataFrame(
            {
                "signal_date": [date(2020, 12, 1)] * 2,
                "ts_code": ["000001.SZ", "000002.SZ"],
                "label_valid": [True, True],
            }
        )


def test_parser_rejects_locked_test_and_unsafe_symbols() -> None:
    with pytest.raises(ValueError, match="2026"):
        audit.parse_args(
            ["--factor-id", "yeslab_395", "--start", "2026-01-01", "--end", "2026-01-02"]
        )
    with pytest.raises(ValueError, match="unsafe"):
        audit.parse_args(
            [
                "--factor-id",
                "yeslab_395",
                "--start",
                "2020-12-01",
                "--end",
                "2020-12-31",
                "--universe-id",
                "bad;drop",
            ]
        )


def test_run_audit_uses_mature_compact_denominator_without_writes() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_395",
            "--start",
            "2020-12-01",
            "--end",
            "2020-12-31",
        ]
    )
    source = _Source()
    target = _Target()

    rows = audit.run_audit(args, source_session=source, target_session=target)

    assert len(rows) == 1
    assert rows[0]["compact_rows"] == 2
    assert rows[0]["compact_finite_ratio"] == 0.5
    assert rows[0]["decision_status"] == "COMPACT_DQ_FAIL"
    assert all("2026" not in query for query in source.queries + target.queries)


def test_run_audit_server_aggregate_avoids_raw_value_transfer() -> None:
    args = audit.parse_args(
        [
            "--factor-id",
            "yeslab_395",
            "--start",
            "2020-12-01",
            "--end",
            "2020-12-31",
            "--server-aggregate",
        ]
    )
    source = _Source()
    target = _Target()

    rows = audit.run_audit(args, source_session=source, target_session=target)

    assert len(rows) == 1
    assert rows[0]["compact_rows"] == 2
    assert rows[0]["compact_finite_rows"] == 1
    assert rows[0]["compact_finite_ratio"] == 0.5
    aggregate_queries = [query for query in source.queries if "finite_keys" in query]
    assert len(aggregate_queries) == 1
    assert "s = select distinct factor_name,trade_date,ts_code" in aggregate_queries[0]
    assert "ej(s,compact_mature_keys" in aggregate_queries[0]
    assert "isValid(value)" in aggregate_queries[0]
    assert target.uploads["compact_mature_keys"].to_dict("records") == [
        {"trade_date": date(2020, 12, 1), "ts_code": "000001.SZ"},
        {"trade_date": date(2020, 12, 1), "ts_code": "000002.SZ"},
    ]


def test_oom_date_split_makes_progress_inside_one_year() -> None:
    chunks = audit._split_date_range(date(2020, 1, 1), date(2020, 1, 3))

    assert chunks == (
        (date(2020, 1, 1), date(2020, 1, 2)),
        (date(2020, 1, 3), date(2020, 1, 3)),
    )
