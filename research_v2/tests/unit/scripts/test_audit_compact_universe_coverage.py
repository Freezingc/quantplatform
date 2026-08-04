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
