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


def _provider_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    classifications = pd.DataFrame(
        {
            "index_code": ["801010.SI", "801020.SI"],
            "industry_name": ["农林牧渔", "采掘"],
            "level": ["L1", "L1"],
        }
    )
    members = pd.DataFrame(
        {
            "index_code": ["801010.SI", "801020.SI", "801020.SI"],
            "con_code": ["000001.SZ", "000001.SZ", "600000.SH"],
            "in_date": ["20200101", "20210101", "20200101"],
            "out_date": ["20210101", None, None],
        }
    )
    return classifications, members


def test_pit_provider_returns_date_specific_industry_labels() -> None:
    classifications, members = _provider_frames()
    provider = audit.PITIndustryProvider(members, classifications)

    labels = provider.get_industry_labels(
        [date(2020, 1, 2), date(2021, 1, 4)],
        ["000001.SZ", "600000.SH"],
    )

    assert labels.loc[pd.Timestamp("2020-01-02"), "000001.SZ"] == "农林牧渔"
    assert labels.loc[pd.Timestamp("2021-01-04"), "000001.SZ"] == "采掘"
    assert labels.loc[pd.Timestamp("2020-01-02"), "600000.SH"] == "采掘"


def test_pit_provider_strict_mode_blocks_overlapping_membership() -> None:
    classifications, members = _provider_frames()
    members = pd.concat(
        [
            members,
            pd.DataFrame(
                {
                    "index_code": ["801010.SI"],
                    "con_code": ["600000.SH"],
                    "in_date": ["20200101"],
                    "out_date": [None],
                }
            ),
        ],
        ignore_index=True,
    )
    provider = audit.PITIndustryProvider(members, classifications, strict=True)

    with pytest.raises(ValueError, match="overlap"):
        provider.get_industry_labels([date(2020, 1, 2)], ["600000.SH"])


def test_pit_provider_reports_coverage_without_static_fallback() -> None:
    classifications, members = _provider_frames()
    provider = audit.PITIndustryProvider(members, classifications, strict=False)

    report = provider.coverage_report(
        [date(2020, 1, 2)], ["000001.SZ", "600000.SH", "000002.SZ"]
    )

    assert report == {
        "requested_rows": 3,
        "mapped_rows": 2,
        "coverage_ratio": pytest.approx(2 / 3),
        "conflict_code_count": 0,
        "unmapped_codes": ["000002.SZ"],
    }


def test_pit_provider_loads_cached_classifications_and_members(tmp_path) -> None:
    classifications, members = _provider_frames()
    classifications.to_json(
        tmp_path / "pit_industry_sw2021_l1_classification.json",
        orient="table",
        date_format="iso",
    )
    for index_code, frame in members.groupby("index_code"):
        frame.to_json(
            tmp_path / f"pit_industry_members_{index_code.replace('.', '_')}.json",
            orient="table",
            date_format="iso",
        )

    provider = audit.PITIndustryProvider.from_cache(tmp_path)

    labels = provider.get_industry_labels([date(2020, 1, 2)], ["000001.SZ"])

    assert labels.iloc[0, 0] == "农林牧渔"


def test_versioned_pit_provider_switches_source_at_cutover() -> None:
    before_classifications, before_members = _provider_frames()
    after_classifications = before_classifications.copy()
    after_classifications["industry_name"] = ["农业", "能源"]
    before = audit.PITIndustryProvider(before_members, before_classifications)
    after = audit.PITIndustryProvider(before_members, after_classifications)
    provider = audit.VersionedPITIndustryProvider(
        before,
        after,
        cutover=date(2021, 12, 10),
    )

    labels = provider.get_industry_labels(
        [date(2020, 1, 2), date(2021, 12, 10)], ["000001.SZ"]
    )

    assert labels.loc[pd.Timestamp("2020-01-02"), "000001.SZ"] == "农林牧渔"
    assert labels.loc[pd.Timestamp("2021-12-10"), "000001.SZ"] == "能源"


def test_fetch_pit_members_uses_requested_classification_source(tmp_path) -> None:
    classifications, members = _provider_frames()

    class Pro:
        def __init__(self) -> None:
            self.classify_sources: list[str] = []
            self.member_codes: list[str] = []

        def index_classify(self, **kwargs) -> pd.DataFrame:
            self.classify_sources.append(kwargs["src"])
            return classifications

        def index_member(self, **kwargs) -> pd.DataFrame:
            self.member_codes.append(kwargs["index_code"])
            return members.loc[members["index_code"].eq(kwargs["index_code"])]

    pro = Pro()
    loaded_classifications, loaded_members = audit.fetch_pit_members(
        pro, source="SW2014", cache_dir=tmp_path
    )

    assert pro.classify_sources == ["SW2014"]
    assert set(pro.member_codes) == {"801010.SI", "801020.SI"}
    assert len(loaded_classifications) == 2
    assert len(loaded_members) == 3
    assert (tmp_path / "pit_industry_sw2014_l1_classification.json").is_file()


def test_coverage_for_keys_uses_active_date_code_denominator() -> None:
    classifications, members = _provider_frames()
    provider = audit.PITIndustryProvider(members, classifications)
    keys = pd.DataFrame(
        {
            "signal_date": pd.to_datetime(
                ["2020-01-02", "2020-01-02", "2020-01-02"]
            ),
            "ts_code": ["000001.SZ", "600000.SH", "000002.SZ"],
        }
    )

    report = audit.coverage_for_keys(provider, keys)

    assert report["requested_rows"] == 3
    assert report["mapped_rows"] == 2
    assert report["coverage_ratio"] == pytest.approx(2 / 3)
    assert report["unmapped_key_count"] == 1
