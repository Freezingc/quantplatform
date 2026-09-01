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


def test_security_master_scd_does_not_flag_names_that_merely_contain_st_substring() -> None:
    names = pd.DataFrame(
        [
            {
                "ts_code": "000001.SZ",
                "name": "FIRST Capital",
                "start_date": date(2020, 1, 2),
                "end_date": pd.NaT,
            }
        ]
    )

    scd = build_security_master_scd(_security_master(), names)
    alpha = scd.loc[scd["ts_code"] == "000001.SZ"]

    assert alpha["security_name"].tolist() == ["FIRST Capital"]
    assert alpha["is_st"].tolist() == [False]


def test_security_master_scd_flags_canonical_st_designation_prefixes() -> None:
    for st_name in ("ST长生", "*ST Alpha", "SST前锋", "S*ST达新"):
        names = pd.DataFrame(
            [
                {
                    "ts_code": "000001.SZ",
                    "name": st_name,
                    "start_date": date(2020, 1, 2),
                    "end_date": pd.NaT,
                }
            ]
        )
        scd = build_security_master_scd(_security_master(), names)
        alpha = scd.loc[scd["ts_code"] == "000001.SZ"]
        assert alpha["is_st"].tolist() == [True], st_name


def test_resolve_name_history_truncates_stale_tail_and_merges_repeats() -> None:
    from research_v2.pit.security_master import resolve_name_history_overlaps

    dirty = pd.DataFrame(
        [
            {
                "ts_code": "X1.SZ",
                "name": "旧名",
                "start_date": date(2010, 1, 1),
                "end_date": date(2026, 8, 10),
            },
            {
                "ts_code": "X1.SZ",
                "name": "新名",
                "start_date": date(2026, 7, 18),
                "end_date": None,
            },
            {
                "ts_code": "X1.SZ",
                "name": "新名",
                "start_date": date(2026, 8, 11),
                "end_date": None,
            },
        ]
    )
    resolved = resolve_name_history_overlaps(dirty)

    assert resolved["name"].tolist() == ["旧名", "新名"]
    assert resolved["start_date"].tolist() == [date(2010, 1, 1), date(2026, 7, 18)]
    assert resolved["end_date"].tolist() == [date(2026, 7, 17), None]


def test_resolve_name_history_rejects_same_start_conflicts() -> None:
    from research_v2.pit.security_master import (
        SecurityMasterConflict,
        resolve_name_history_overlaps,
    )

    dirty = pd.DataFrame(
        [
            {"ts_code": "X2.SZ", "name": "甲", "start_date": date(2020, 1, 1), "end_date": None},
            {"ts_code": "X2.SZ", "name": "乙", "start_date": date(2020, 1, 1), "end_date": None},
        ]
    )
    with pytest.raises(SecurityMasterConflict):
        resolve_name_history_overlaps(dirty)


def test_resolve_name_history_is_idempotent() -> None:
    from research_v2.pit.security_master import resolve_name_history_overlaps

    dirty = pd.DataFrame(
        [
            {
                "ts_code": "X3.SZ",
                "name": "旧名",
                "start_date": date(2015, 1, 1),
                "end_date": date(2024, 5, 5),
            },
            {
                "ts_code": "X3.SZ",
                "name": "新名",
                "start_date": date(2024, 5, 1),
                "end_date": None,
            },
        ]
    )
    once = resolve_name_history_overlaps(dirty)
    twice = resolve_name_history_overlaps(once)
    pd.testing.assert_frame_equal(once, twice)
