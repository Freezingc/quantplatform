from __future__ import annotations

import math
from time import perf_counter

import pandas as pd

from research_v2.factors.deduplication import deduplicate_factor_observations


def _row(value: float, source_row_id: str) -> dict[str, object]:
    return {
        "observation_date": "2025-01-02",
        "ts_code": "000001.SZ",
        "factor_id": "yeslab_1",
        "factor_version": 1,
        "raw_value": value,
        "source_row_id": source_row_id,
    }


def test_deduplication_collapses_identical_and_preserves_single_nonnull_value() -> None:
    observations = pd.DataFrame(
        [
            _row(1.0, "a"),
            _row(1.0, "b"),
            {**_row(2.0, "c"), "ts_code": "000002.SZ"},
            {**_row(math.nan, "d"), "ts_code": "000002.SZ"},
        ]
    )

    result = deduplicate_factor_observations(observations)

    assert result.quarantine.empty
    assert result.clean["dedup_resolution"].tolist() == ["IDENTICAL_COLLAPSE", "SINGLE_NONNULL"]
    assert result.clean["raw_value"].tolist() == [1.0, 2.0]


def test_deduplication_quarantines_conflicting_finite_values_without_choosing_one() -> None:
    observations = pd.DataFrame([_row(1.0, "a"), _row(2.0, "b")])

    result = deduplicate_factor_observations(observations)

    assert result.clean.empty
    assert len(result.quarantine) == 2
    assert set(result.quarantine["reason"]) == {"CONFLICT_QUARANTINE"}
    assert result.statistics.conflict_groups == 1


def test_deduplication_handles_large_unique_cross_section_without_group_loops() -> None:
    count = 10_000
    observations = pd.DataFrame(
        {
            "observation_date": ["2025-01-02"] * count,
            "ts_code": [f"{index:06d}.SZ" for index in range(count)],
            "factor_id": ["yeslab_1"] * count,
            "factor_version": [1] * count,
            "raw_value": [1.0] * count,
            "source_row_id": [f"source-{index}" for index in range(count)],
        }
    )
    started_at = perf_counter()

    result = deduplicate_factor_observations(observations)

    assert len(result.clean) == count
    assert perf_counter() - started_at < 1.0
