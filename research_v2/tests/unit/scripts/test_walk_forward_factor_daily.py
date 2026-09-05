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


def test_frozen_constant_matches_frozen_contract_json() -> None:
    import json

    research_root = Path(__file__).resolve().parents[3]
    contract = json.loads(
        (research_root / "artifacts/diagnostics/factor_set_frozen_v1.json").read_text(
            encoding="utf-8"
        )
    )
    assert tuple(contract["factors"]) == MODULE.FROZEN_FACTOR_SET_V1
    assert len(MODULE.FROZEN_FACTOR_SET_V1) == contract["count"] == 17


def test_cache_key_is_scoped_to_the_factor_set(tmp_path: Path) -> None:
    """A different factor set must never be served a stale cached frame.

    ``fold_cache_key`` hashes only ``(fold_id, role, versions)``. Since the frozen
    factor set is a constant rather than part of that payload, shrinking
    FROZEN_FACTOR_SET_V1 (adjudication 4 option iii, which the coverage and
    redundancy findings make the leading candidate) would reuse the 17-factor cache
    under an identical key and silently hand the driver the wrong columns -- the same
    class of silent failure as an unregistered month. The extractor therefore scopes
    its own cache key by the requested factor set.
    """
    full = MODULE.factor_members_for(("yeslab_206", "yeslab_357"))
    narrow = MODULE.factor_members_for(("yeslab_206",))
    calls: list[str] = []

    def counting_query(holder, script):
        calls.append(script)
        return _fake_query(holder, script)

    first = MODULE.extract_dataset_factor_daily(
        {}, _Resolver(), "F03", "train", full, VERSIONS,
        query_fn=counting_query, cache_root=tmp_path,
    )
    after_full = len(calls)
    wide_columns = [c for c in first.columns if c.startswith("f")]

    second = MODULE.extract_dataset_factor_daily(
        {}, _Resolver(), "F03", "train", narrow, VERSIONS,
        query_fn=counting_query, cache_root=tmp_path,
    )
    assert len(calls) > after_full, "narrower factor set was served the wider cache"
    narrow_columns = [c for c in second.columns if c.startswith("f")]
    assert wide_columns == ["f00", "f01"]
    assert narrow_columns == ["f00"]

    # Repeating the narrow request must now hit its own cache entry.
    after_narrow = len(calls)
    third = MODULE.extract_dataset_factor_daily(
        {}, _Resolver(), "F03", "train", narrow, VERSIONS,
        query_fn=counting_query, cache_root=tmp_path,
    )
    assert len(calls) == after_narrow
    pd.testing.assert_frame_equal(second, third)


def test_legacy_fold_cache_key_formula_is_pinned() -> None:
    """The shared key formula must not change: legacy keys are a live cache and part
    of the criterion-303 reproduction contract. Pinning it here means any future edit
    to ``walk_forward_common.fold_cache_key`` fails loudly instead of silently
    orphaning 2.9 GB of warm legacy fold cache and forcing full re-extraction."""
    import hashlib
    import json
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
    import walk_forward_common as common

    versions = {
        "universe_daily": {"2020-01": "u1"},
        "factor_observation_clean": {"2020-01": "f1"},
        "label_daily": {"2020-01": "l1"},
    }
    payload = json.dumps(
        {"fold_id": "F01", "role": "train",
         "versions": {k: dict(v) for k, v in versions.items()}},
        sort_keys=True,
        ensure_ascii=True,
    )
    assert common.fold_cache_key("F01", "train", versions) == hashlib.sha256(
        payload.encode("utf-8")
    ).hexdigest()


def test_frozen_feature_names_are_unique() -> None:
    members = MODULE.factor_members_for(MODULE.FROZEN_FACTOR_SET_V1)
    names = [member.feature_name for member in members]
    assert len(set(names)) == 17
    assert names[0] == "f00" and names[-1] == "f16"


def test_frozen_set_satisfies_every_model_yaml_factor_bound() -> None:
    """Reconstruction plan Task 2.2: lock the frozen-17 / yaml bound compatibility.

    The candidate plan recorded a contract tension because ``raw_factor_count_min``
    is 15 in every ``configs/models/*.yaml`` while the legacy walk-forward set had
    14 factors. The frozen set has 17, so no yaml needs changing; this test pins
    that fact so a future yaml bound raise (or a frozen-set shrink) fails loudly
    instead of silently re-opening the tension.
    """
    import yaml

    research_root = Path(__file__).resolve().parents[3]
    configs = sorted((research_root / "configs" / "models").glob("*.yaml"))
    assert configs, "expected model yaml configs to exist"

    frozen_count = len(MODULE.FROZEN_FACTOR_SET_V1)
    assert frozen_count == 17
    for config_path in configs:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        minimum = payload["raw_factor_count_min"]
        maximum = payload["raw_factor_count_max"]
        assert minimum <= frozen_count <= maximum, (
            f"{config_path.name}: frozen factor count {frozen_count} outside "
            f"[{minimum}, {maximum}]"
        )
        # The legacy 14-factor set is what violated the declared minimum; record
        # the contrast so the retired tension clause stays explainable.
        assert minimum > 14, (
            f"{config_path.name}: raw_factor_count_min dropped to {minimum}; the "
            "documented legacy-14 tension would need re-adjudication"
        )
