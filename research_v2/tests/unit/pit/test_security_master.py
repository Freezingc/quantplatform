from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.pit.security_master import SecurityMasterConflict, build_security_master_scd


def _security_master() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "exchange": "SZSE",
                "market": "主板",
                "list_date": date(2020, 1, 2),
                "delist_date": pd.NaT,
                "name": "Alpha",
                "list_status": "L",
            },
            {
                "ts_code": "688001.SH",
                "exchange": "SSE",
                "market": "科创板",
                "list_date": date(2020, 1, 2),
                "delist_date": pd.NaT,
                "name": "Beta",
                "list_status": "L",
            },
            {
                "ts_code": "430001.BJ",
                "exchange": "BSE",
                "market": "北交所",
                "list_date": date(2020, 1, 2),
                "delist_date": pd.NaT,
                "name": "Gamma",
                "list_status": "L",
            },
        ]
    )


def test_security_master_scd_preserves_historical_st_interval_and_board() -> None:
    names = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "*ST Alpha",
                "start_date": date(2020, 2, 3),
                "end_date": date(2020, 2, 7),
            },
            {
                "ts_code": "000001.SZ",
                "name": "Alpha New",
                "start_date": date(2020, 2, 10),
                "end_date": pd.NaT,
            },
        ]
    )

    scd = build_security_master_scd(_security_master(), names)
    alpha = scd.loc[scd["ts_code"] == "000001.SZ"].reset_index(drop=True)

    assert alpha[["valid_from", "valid_to"]].to_dict("records") == [
        {"valid_from": date(2020, 1, 2), "valid_to": date(2020, 2, 3)},
        {"valid_from": date(2020, 2, 3), "valid_to": date(2020, 2, 8)},
        {"valid_from": date(2020, 2, 8), "valid_to": date(2020, 2, 10)},
        {"valid_from": date(2020, 2, 10), "valid_to": None},
    ]
    assert alpha["is_st"].tolist() == [False, True, False, False]
    assert alpha["security_name"].tolist() == [None, "*ST Alpha", None, "Alpha New"]
    assert set(scd.loc[scd["ts_code"] == "000001.SZ", "board"]) == {"SZ_MAIN"}
    assert set(scd.loc[scd["ts_code"] == "688001.SH", "board"]) == {"STAR"}
    assert set(scd.loc[scd["ts_code"] == "430001.BJ", "board"]) == {"BEIJING"}


def test_security_master_scd_rejects_conflicting_name_records() -> None:
    names = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "*ST Alpha",
                "start_date": date(2020, 2, 3),
                "end_date": pd.NaT,
            },
            {
                "ts_code": "000001.SZ",
                "name": "Alpha New",
                "start_date": date(2020, 2, 3),
                "end_date": pd.NaT,
            },
        ]
    )

    with pytest.raises(SecurityMasterConflict, match="conflicting active names"):
        build_security_master_scd(_security_master(), names)
