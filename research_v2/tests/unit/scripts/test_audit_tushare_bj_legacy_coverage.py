from __future__ import annotations

from datetime import date

import pandas as pd

import audit_tushare_bj_legacy_coverage as audit


class FakePro:
    def daily(self, **kwargs):
        assert kwargs["ts_code"] == "830001.BJ"
        return pd.DataFrame({"trade_date": ["20200102", "20200103"]})

    def daily_basic(self, **kwargs):
        return pd.DataFrame(columns=["trade_date"])

    def stk_factor(self, **kwargs):
        raise RuntimeError("endpoint unavailable")


def test_summarize_frame_reports_date_range_and_rows() -> None:
    result = audit.summarize_frame(
        pd.DataFrame({"trade_date": ["20200103", "20200102", None]})
    )

    assert result == {
        "status": "ROWS_PRESENT",
        "rows": 3,
        "finite_trade_date_rows": 2,
        "min_trade_date": "2020-01-02",
        "max_trade_date": "2020-01-03",
    }


def test_audit_code_records_empty_and_endpoint_error() -> None:
    result = audit.audit_code(FakePro(), "830001.BJ", date(2020, 1, 2), date(2020, 1, 31))

    assert result["code"] == "830001.BJ"
    assert result["endpoints"]["daily"]["status"] == "ROWS_PRESENT"
    assert result["endpoints"]["daily"]["rows"] == 2
    assert result["endpoints"]["daily_basic"]["status"] == "NO_ROWS"
    assert result["endpoints"]["stk_factor"]["status"] == "ERROR"
    assert "endpoint unavailable" in result["endpoints"]["stk_factor"]["error"]
