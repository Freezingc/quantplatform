from __future__ import annotations

from datetime import date

import audit_tushare_pit_industry as audit
import pandas as pd
import pytest


def test_build_pit_snapshot_uses_start_inclusive_end_exclusive_intervals() -> None:
    classifications = pd.DataFrame(
        {
            "index_code": ["801010.SI"],
            "industry_name": ["农林牧渔"],
            "level": ["L1"],
        }
    )
    members = pd.DataFrame(
        {
            "index_code": ["801010.SI", "801010.SI"],
            "con_code": ["000001.SZ", "000001.SZ"],
            "in_date": ["20200101", "20210101"],
            "out_date": ["20210101", None],
        }
    )

    snapshot = audit.build_pit_snapshot(
        members, classifications, as_of=date(2021, 1, 1)
    )

    assert snapshot.to_dict("records") == [
        {
            "ts_code": "000001.SZ",
            "industry_code": "801010.SI",
            "industry_name": "农林牧渔",
            "in_date": pd.Timestamp("2021-01-01"),
            "out_date": pd.NaT,
        }
    ]


def test_build_pit_snapshot_records_overlapping_industry_conflict() -> None:
    classifications = pd.DataFrame(
        {
            "index_code": ["801010.SI", "801030.SI"],
            "industry_name": ["农林牧渔", "基础化工"],
            "level": ["L1", "L1"],
        }
    )
    members = pd.DataFrame(
        {
            "index_code": ["801010.SI", "801030.SI"],
            "con_code": ["000001.SZ", "000001.SZ"],
            "in_date": ["20200101", "20200101"],
            "out_date": [None, None],
        }
    )

    snapshot = audit.build_pit_snapshot(
        members, classifications, as_of=date(2021, 1, 1)
    )

    assert len(snapshot) == 1
    assert snapshot.attrs["overlap_conflict_code_count"] == 1


def test_normalize_members_is_idempotent_for_datetime_intervals() -> None:
    raw = pd.DataFrame(
        {
            "index_code": ["801010.SI"],
            "con_code": ["000001.SZ"],
            "in_date": ["20200101"],
            "out_date": [None],
        }
    )

    once = audit.normalize_members(raw)
    twice = audit.normalize_members(once)

    pd.testing.assert_frame_equal(once, twice)


def test_summarize_snapshot_reports_shsz_coverage_and_static_label_overlap() -> None:
    snapshot = pd.DataFrame(
        {
            "ts_code": ["000001.SZ", "600000.SH"],
            "industry_code": ["801010.SI", "801030.SI"],
            "industry_name": ["农林牧渔", "基础化工"],
        }
    )
    universe = pd.DataFrame({"ts_code": ["000001.SZ", "600000.SH", "830001.BJ"]})
    static = pd.DataFrame(
        {"ts_code": ["000001.SZ", "600000.SH"], "industry": ["农林牧渔", "钢铁"]}
    )

    summary = audit.summarize_snapshot(snapshot, universe, static)

    assert summary["universe_shsz_count"] == 2
    assert summary["pit_shsz_coverage_count"] == 2
    assert summary["pit_shsz_coverage_ratio"] == pytest.approx(1.0)
    assert summary["static_label_overlap_count"] == 2
    assert summary["static_label_exact_name_count"] == 1


def test_parse_date_rejects_locked_test() -> None:
    with pytest.raises(ValueError, match="2026"):
        audit.parse_date("2026-01-01")
