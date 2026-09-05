"""Tests for the factor_daily-backed fold extraction (reconstruction plan Task 1)."""

from __future__ import annotations

import importlib.util
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "walk_forward_factor_daily.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("walk_forward_factor_daily", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["walk_forward_factor_daily"] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()

VERSIONS = {
    "universe_daily": {"2020-01": "u1"},
    "factor_daily": {"2020-01": "fd1"},
    "label_daily": {"2020-01": "l1"},
}


class _Segment:
    def __init__(self, dates):
        self.signal_dates = dates


class _Resolver:
    def resolve(self, fold_id, role):
        return _Segment([date(2020, 1, 2), date(2020, 1, 3)])


def _universe_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": pd.to_datetime(
                ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"]
            ),
            "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
            "data_version": ["u1"] * 4,
        }
    )


def _factor_frame(extra_version: str = "fd1") -> pd.DataFrame:
    rows = []
    for day in ("2020-01-02", "2020-01-03"):
        for code in ("000001.SZ", "000002.SZ"):
            for index, factor_id in enumerate(("yeslab_206", "yeslab_357")):
                rows.append(
                    {
                        "signal_date": pd.Timestamp(day),
                        "ts_code": code,
                        "factor_id": factor_id,
                        "factor_version": 1,
                        "value": float(index) + 0.5,
                        "data_version": extra_version,
                    }
                )
    return pd.DataFrame(rows)


def _label_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": pd.to_datetime(
                ["2020-01-02", "2020-01-02", "2020-01-03", "2020-01-03"]
            ),
            "ts_code": ["000001.SZ", "000002.SZ", "000001.SZ", "000002.SZ"],
            "raw_return": [0.01, -0.02, 0.03, 0.0],
            "label_valid": [True, True, True, False],
            "exit_date": pd.to_datetime(["2020-01-09"] * 4),
            "data_version": ["l1"] * 4,
        }
    )


def _factor_frame_for(script: str, frame: pd.DataFrame) -> pd.DataFrame:
    """Respect the factor_id WHERE clause like the real database would."""
    if "factor_id='" in script:
        requested = script.split("factor_id='", 1)[1].split("'", 1)[0]
        return frame.loc[frame["factor_id"] == requested].reset_index(drop=True)
    return frame


def _fake_query(holder, script: str):
    if "universe_daily" in script:
        return None, _universe_frame()
    if "'factor_daily'" in script:
        return None, _factor_frame_for(script, _factor_frame())
    if "label_daily" in script:
        return None, _label_frame()
    raise AssertionError(f"unexpected query: {script[:120]}")


def test_extract_builds_pit_dataset_from_factor_daily(tmp_path: Path) -> None:
    members = MODULE.factor_members_for(("yeslab_206", "yeslab_357"))
    dataset = MODULE.extract_dataset_factor_daily(
        {},
        _Resolver(),
        "F03",
        "train",
        members,
        VERSIONS,
        query_fn=_fake_query,
        cache_root=tmp_path,
    )
    assert {"signal_date", "ts_code", "raw_return_5d", "label_valid", "f00", "f01"}.issubset(
        dataset.columns
    )
    assert len(dataset) == 4  # 2 days x 2 universe members
    assert set(dataset["ts_code"]) == {"000001.SZ", "000002.SZ"}


def test_rows_with_unregistered_versions_are_dropped(tmp_path: Path) -> None:
    def mixed_query(holder, script):
        if "'factor_daily'" in script:
            good = _factor_frame("fd1")
            stale = _factor_frame("OLD")
            return None, _factor_frame_for(script, pd.concat([good, stale], ignore_index=True))
        return _fake_query(holder, script)

    members = MODULE.factor_members_for(("yeslab_206", "yeslab_357"))
    dataset = MODULE.extract_dataset_factor_daily(
        {},
        _Resolver(),
        "F03",
        "train",
        members,
        VERSIONS,
        query_fn=mixed_query,
        cache_root=tmp_path,
    )
    # stale-version rows must not duplicate or contaminate the panel
    assert len(dataset) == 4


def test_missing_factor_daily_version_key_fails_closed(tmp_path: Path) -> None:
    members = MODULE.factor_members_for(("yeslab_206",))
    incomplete = {"universe_daily": {"2020-01": "u1"}, "label_daily": {"2020-01": "l1"}}
    with pytest.raises(KeyError):
        MODULE.extract_dataset_factor_daily(
            {},
            _Resolver(),
            "F03",
            "train",
            members,
            incomplete,
            query_fn=_fake_query,
            cache_root=tmp_path,
        )


def test_cache_is_isolated_by_versions_and_reused(tmp_path: Path) -> None:
    members = MODULE.factor_members_for(("yeslab_206", "yeslab_357"))
    calls: list[str] = []

    def counting_query(holder, script):
        calls.append(script)
        return _fake_query(holder, script)

    first = MODULE.extract_dataset_factor_daily(
        {}, _Resolver(), "F03", "train", members, VERSIONS,
        query_fn=counting_query, cache_root=tmp_path,
    )
    queries_first = len(calls)
    second = MODULE.extract_dataset_factor_daily(
        {}, _Resolver(), "F03", "train", members, VERSIONS,
        query_fn=counting_query, cache_root=tmp_path,
    )
    assert len(calls) == queries_first  # second run served entirely from cache
    pd.testing.assert_frame_equal(first, second)

    other_versions = dict(VERSIONS, factor_daily={"2020-01": "fd2"})
    MODULE.extract_dataset_factor_daily(
        {}, _Resolver(), "F03", "train", members, other_versions,
        query_fn=counting_query, cache_root=tmp_path,
    )
    assert len(calls) > queries_first  # different version -> different cache key -> re-query
