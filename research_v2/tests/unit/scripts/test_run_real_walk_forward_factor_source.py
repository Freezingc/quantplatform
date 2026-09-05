"""Tests for the --factor-source switch (reconstruction plan Task 2.3/2.4).

The legacy branch must stay bit-identical to the criterion-303 reproduction
contract. Because ``fold_cache_key`` hashes the whole versions mapping (table
names included), "bit-identical" concretely means: legacy resolves exactly the
same three tables and the same factor id tuple as before the flag existed, so
its cache keys and assembled frames cannot change. The frozen branch must be
key-disjoint from legacy so the two chains can share one cache root.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"
sys.path.insert(0, str(SCRIPTS))

# Import the helper modules through the normal import system so the driver shares
# the very same module objects; loading private copies would make the identity
# assertions in test_extractor_selection_matches_the_source compare two distinct
# function objects for the same source function.
import walk_forward_common as COMMON  # noqa: E402
import walk_forward_factor_daily as FROZEN  # noqa: E402


def _load(name: str, filename: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / filename)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


DRIVER = _load("rwf_factor_source", "run_real_walk_forward.py")
CANDIDATE = _load("rct_factor_source", "run_real_candidate_training.py")

LEGACY_TABLES = ("universe_daily", "factor_observation_clean", "label_daily")
FROZEN_TABLES = ("universe_daily", "factor_daily", "label_daily")


def test_cli_exposes_factor_source_defaulting_to_legacy() -> None:
    parser = DRIVER.build_arg_parser()
    args = parser.parse_args(["--report", "/tmp/x.json"])
    assert args.factor_source == "legacy"
    with pytest.raises(SystemExit):
        parser.parse_args(["--report", "/tmp/x.json", "--factor-source", "bogus"])
    frozen = parser.parse_args(["--report", "/tmp/x.json", "--factor-source", "frozen"])
    assert frozen.factor_source == "frozen"


def test_legacy_source_keeps_the_exact_pre_flag_contract() -> None:
    """Legacy must resolve the same tables and factor ids as before Task 2.3."""
    factor_ids, tables = DRIVER.factor_source_plan("legacy")
    assert tables == LEGACY_TABLES
    assert factor_ids == COMMON.WALK_FORWARD_FACTOR_IDS
    assert len(factor_ids) == 14
    assert "factor_daily" not in tables


def test_frozen_source_uses_frozen_set_and_factor_daily() -> None:
    factor_ids, tables = DRIVER.factor_source_plan("frozen")
    assert tables == FROZEN_TABLES
    assert factor_ids == FROZEN.FROZEN_FACTOR_SET_V1
    assert len(factor_ids) == 17


def test_unknown_source_is_rejected() -> None:
    with pytest.raises(ValueError):
        DRIVER.factor_source_plan("observation_clean")


def test_legacy_cache_keys_are_unchanged_by_the_flag() -> None:
    """The legacy fold cache key must equal the key computed from the pre-flag mapping."""
    versions = {
        "universe_daily": {"2020-01": "u1"},
        "factor_observation_clean": {"2020-01": "f1"},
        "label_daily": {"2020-01": "l1"},
    }
    _, tables = DRIVER.factor_source_plan("legacy")
    assert tables == tuple(versions)
    assert COMMON.fold_cache_key("F01", "train", versions) == COMMON.fold_cache_key(
        "F01", "train", {table: versions[table] for table in tables}
    )


def test_frozen_and_legacy_cache_keys_are_disjoint() -> None:
    """Both chains share FOLD_CACHE_ROOT, so their keys must never collide."""
    legacy_versions = {
        "universe_daily": {"2020-01": "u1"},
        "factor_observation_clean": {"2020-01": "f1"},
        "label_daily": {"2020-01": "l1"},
    }
    frozen_versions = {
        "universe_daily": {"2020-01": "u1"},
        "factor_daily": {"2020-01": "f1"},
        "label_daily": {"2020-01": "l1"},
    }
    for role in ("train", "validation", "test"):
        assert COMMON.fold_cache_key("F01", role, legacy_versions) != COMMON.fold_cache_key(
            "F01", role, frozen_versions
        )


def test_extractor_selection_matches_the_source() -> None:
    assert DRIVER.extract_fn_for("legacy") is COMMON.extract_dataset
    assert DRIVER.extract_fn_for("frozen") is FROZEN.extract_dataset_factor_daily
    with pytest.raises(ValueError):
        DRIVER.extract_fn_for("legacy ")


def test_candidate_cli_exposes_factor_source_defaulting_to_legacy() -> None:
    args = CANDIDATE.parse_args(["--family", "M1", "--report", "/tmp/x.json"])
    assert args.factor_source == "legacy"
    with pytest.raises(SystemExit):
        CANDIDATE.parse_args(
            ["--family", "M1", "--report", "/tmp/x.json", "--factor-source", "bogus"]
        )
    frozen = CANDIDATE.parse_args(
        ["--family", "M1", "--report", "/tmp/x.json", "--factor-source", "frozen"]
    )
    assert frozen.factor_source == "frozen"


def test_both_drivers_share_one_factor_source_contract() -> None:
    """Task 2.3 requires the same flag on both drivers; sharing the helpers by
    identity is what prevents the two from drifting apart."""
    assert CANDIDATE.factor_source_plan is DRIVER.factor_source_plan
    assert CANDIDATE.extract_fn_for is DRIVER.extract_fn_for
    assert CANDIDATE.FACTOR_SOURCES is DRIVER.FACTOR_SOURCES
    assert CANDIDATE.factor_source_plan("frozen") == DRIVER.factor_source_plan("frozen")
    assert CANDIDATE.factor_source_plan("legacy") == DRIVER.factor_source_plan("legacy")


def test_full_run_context_carries_the_extractor() -> None:
    """The extractor must travel with the context so both the cross-sectional
    (M1/M3) and temporal (M2) pipelines assemble from the same source."""
    names = {field.name for field in dataclasses.fields(CANDIDATE._FullRunContext)}
    assert "extract" in names
    assert {"versions", "factor_members", "factor_order"} <= names
