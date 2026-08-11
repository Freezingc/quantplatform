from __future__ import annotations

from datetime import date

import pytest

from research_v2.runs.context import RunContext


def test_data_version_is_stable_for_equivalent_orderings() -> None:
    config = {"universe": "cn_liquid_a:v1", "windows": [5, 20]}
    left = RunContext.build(
        config=config,
        watermarks=["market:2025-12-31", "factor:2025-12-31"],
        code_commit="UNVERSIONED",
        schema_version="4",
    )
    right = RunContext.build(
        config={"windows": [5, 20], "universe": "cn_liquid_a:v1"},
        watermarks=["factor:2025-12-31", "market:2025-12-31"],
        code_commit="UNVERSIONED",
        schema_version="4",
    )

    assert left.data_version == right.data_version
    assert left.run_id != right.run_id


def test_data_version_changes_when_a_watermark_changes() -> None:
    base = RunContext.build(
        config={"as_of": date(2025, 12, 31)},
        watermarks=["market:2025-12-31"],
        code_commit="UNVERSIONED",
        schema_version="4",
    )
    changed = RunContext.build(
        config={"as_of": date(2025, 12, 31)},
        watermarks=["market:2026-01-01"],
        code_commit="UNVERSIONED",
        schema_version="4",
    )

    assert base.data_version != changed.data_version


def test_run_context_rejects_non_finite_configuration_values() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        RunContext.build(
            config={"threshold": float("nan")},
            watermarks=["market:2025-12-31"],
            code_commit="UNVERSIONED",
            schema_version="4",
        )
