"""Tests for the registration-only factor_daily version publisher (Task 1.6)."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3] / "scripts" / "publish_factor_daily_versions.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("publish_factor_daily_versions", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["publish_factor_daily_versions"] = module
    spec.loader.exec_module(module)
    return module


MODULE = _load_module()

MONTHS = ("2020-01", "2020-02")
FROZEN = tuple(f"f{i:02d}" for i in range(17))


def _month_state(
    month: str,
    *,
    versions: tuple[str, ...] = ("v1",),
    rows_per_version: int = 100,
    factors: tuple[str, ...] = FROZEN,
    duplicate_keys: int = 0,
    already_registered: str | None = None,
) -> dict:
    return {
        "month": month,
        "versions": list(versions),
        "rows_by_version": {version: rows_per_version for version in versions},
        "factors": list(factors),
        "duplicate_key_count": duplicate_keys,
        "registered_version": already_registered,
    }


def test_plan_registers_each_month_with_independent_pointers() -> None:
    """published_version chains per (object_name, logical_partition): months never chain."""
    states = {m: _month_state(m) for m in MONTHS}
    plan = MODULE.plan_registrations(states, expected_factor_count=17)
    assert [entry.month for entry in plan] == list(MONTHS)
    assert all(entry.action == "REGISTER" for entry in plan)
    assert plan[0].previous_version == ""
    assert plan[1].previous_version == ""


def test_multi_version_month_is_refused() -> None:
    states = {"2020-01": _month_state("2020-01", versions=("v1", "v2"))}
    with pytest.raises(MODULE.RegistrationRefused, match="multiple data_versions"):
        MODULE.plan_registrations(states, expected_factor_count=17)


def test_empty_month_is_refused() -> None:
    states = {"2020-01": _month_state("2020-01", rows_per_version=0)}
    with pytest.raises(MODULE.RegistrationRefused, match="no rows"):
        MODULE.plan_registrations(states, expected_factor_count=17)


def test_missing_factors_is_refused() -> None:
    states = {"2020-01": _month_state("2020-01", factors=FROZEN[:16])}
    with pytest.raises(MODULE.RegistrationRefused, match="factor coverage"):
        MODULE.plan_registrations(states, expected_factor_count=17)


def test_duplicate_business_keys_are_refused() -> None:
    states = {"2020-01": _month_state("2020-01", duplicate_keys=3)}
    with pytest.raises(MODULE.RegistrationRefused, match="duplicate business keys"):
        MODULE.plan_registrations(states, expected_factor_count=17)


def test_already_registered_month_is_skipped_idempotently() -> None:
    states = {"2020-01": _month_state("2020-01", already_registered="v1")}
    plan = MODULE.plan_registrations(states, expected_factor_count=17)
    assert plan[0].action == "SKIP_CURRENT"


def test_registered_with_different_version_still_registers() -> None:
    states = {"2020-01": _month_state("2020-01", versions=("v2",), already_registered="v1")}
    plan = MODULE.plan_registrations(states, expected_factor_count=17)
    assert plan[0].action == "REGISTER"
    assert plan[0].data_version == "v2"
    assert plan[0].previous_version == "v1"


class TestPointerScript:
    def test_pointer_script_targets_published_version_only(self) -> None:
        entry = MODULE.PlanEntry(
            month="2020-01",
            data_version="v2",
            previous_version="v1",
            row_count=100,
            action="REGISTER",
        )
        script = MODULE.pointer_script(entry, object_name="factor_daily", target_database="dfs://X")
        assert "published_version" in script
        assert "factor_daily" in script
        assert "'v2'" in script
        assert "'v1'" in script
        # registration must never write data rows into the target table
        assert "factor_daily`)" not in script.replace("object_name", "")
