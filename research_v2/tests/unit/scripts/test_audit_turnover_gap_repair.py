from __future__ import annotations

from datetime import date

import audit_turnover_gap_repair as audit
import pandas as pd
import pytest


class _Tushare:
    def daily_basic(self, **_: object) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "ts_code": ["000001.SZ", "000002.SZ"],
                "turnover_rate": [1.2, 2.3],
            }
        )


def test_parse_args_rejects_locked_test() -> None:
    with pytest.raises(ValueError, match="2026"):
        audit.parse_args(
            [
                "--gap-date",
                "2026-01-02",
                "--impact-start",
                "2026-01-01",
                "--impact-end",
                "2026-01-03",
                "--output",
                "/tmp/audit.json",
            ]
        )


def test_apply_tushare_turnover_gap_only_fills_missing_rows() -> None:
    market = pd.DataFrame(
        {
            "trade_date": [date(2020, 9, 18)] * 2,
            "ts_code": ["000001.SZ", "000002.SZ"],
            "turnover_rate": [float("nan"), 9.0],
        }
    )
    mature = pd.DataFrame(
        {
            "signal_date": [date(2020, 9, 18)] * 2,
            "ts_code": ["000001.SZ", "000002.SZ"],
        }
    )

    result, evidence = audit._apply_tushare_turnover_gap(
        market,
        mature,
        gap_date=date(2020, 9, 18),
        tushare_client=_Tushare(),
    )

    assert result.loc[result["ts_code"] == "000001.SZ", "turnover_rate"].iloc[0] == 1.2
    assert result.loc[result["ts_code"] == "000002.SZ", "turnover_rate"].iloc[0] == 9.0
    assert evidence["source_missing_mature_rows"] == 1
    assert evidence["replacement_rows"] == 1


def test_audit_factor_blocks_formula_overlap_mismatch() -> None:
    source = pd.DataFrame(
        {
            "trade_date": [date(2020, 9, 18)],
            "ts_code": ["000001.SZ"],
            "factor_name": ["GP_011"],
            "value": [1.0],
        }
    )
    candidate = pd.DataFrame(
        {
            "trade_date": [date(2020, 9, 18)],
            "ts_code": ["000001.SZ"],
            "candidate_value": [2.0],
        }
    )
    mature = pd.DataFrame(
        {
            "signal_date": [date(2020, 9, 18)],
            "ts_code": ["000001.SZ"],
        }
    )

    result = audit._audit_factor(
        factor_id="yeslab_380",
        factor_name="GP_011",
        source=source,
        candidate=candidate,
        mature=mature,
        gap_date=date(2020, 9, 18),
    )

    assert result["overlap_mismatch_rows"] == 1
    assert result["status"] == "TURNOVER_REPAIR_BLOCKED_OVERLAP"


def test_candidate_fill_frame_contains_only_nonfinite_source_rows(tmp_path) -> None:
    source = pd.DataFrame(
        {
            "trade_date": [date(2020, 9, 18), date(2020, 9, 18)],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "factor_name": ["GP_011", "GP_011"],
            "value": [float("nan"), 5.0],
        }
    )
    candidate = pd.DataFrame(
        {
            "trade_date": [date(2020, 9, 18), date(2020, 9, 18)],
            "ts_code": ["000001.SZ", "000002.SZ"],
            "candidate_value": [3.0, 6.0],
        }
    )
    mature = pd.DataFrame(
        {
            "signal_date": [date(2020, 9, 18)] * 2,
            "ts_code": ["000001.SZ", "000002.SZ"],
        }
    )

    frame = audit._candidate_fill_frame(
        factor_id="yeslab_380",
        factor_name="GP_011",
        source=source,
        candidate=candidate,
        mature=mature,
    )
    artifact = audit._write_candidate_artifact(
        tmp_path,
        factor_id="yeslab_380",
        frame=frame,
    )

    assert len(frame) == 1
    assert frame.iloc[0]["ts_code"] == "000001.SZ"
    assert artifact["rows"] == 1
    assert (tmp_path / "yeslab_380_turnover_gap_candidate.csv").is_file()
