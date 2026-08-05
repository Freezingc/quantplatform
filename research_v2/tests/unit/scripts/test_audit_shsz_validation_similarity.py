from __future__ import annotations

import pandas as pd
import pytest

import audit_shsz_validation_similarity as audit


FACTOR_IDS = ("yeslab_376", "yeslab_380", "yeslab_396")


def test_audit_validation_similarity_requires_all_unordered_pairs() -> None:
    frame = pd.DataFrame(
        [
            {
                "factor_id_left": "yeslab_376",
                "factor_id_right": "yeslab_380",
                "median_daily_abs_spearman": 0.4,
                "valid_days": 237,
            },
            {
                "factor_id_left": "yeslab_376",
                "factor_id_right": "yeslab_396",
                "median_daily_abs_spearman": 0.5,
                "valid_days": 237,
            },
        ]
    )

    result = audit.audit_validation_similarity(frame, FACTOR_IDS)

    assert result["required_pair_count"] == 3
    assert result["observed_pair_count"] == 2
    assert result["missing_pairs"] == [["yeslab_380", "yeslab_396"]]
    assert result["status"] == "FAIL"


def test_audit_validation_similarity_passes_complete_below_threshold() -> None:
    frame = pd.DataFrame(
        [
            ["yeslab_376", "yeslab_380", 0.4, 237],
            ["yeslab_396", "yeslab_376", 0.5, 237],
            ["yeslab_380", "yeslab_396", 0.8, 237],
        ],
        columns=["factor_id_left", "factor_id_right", "median_daily_abs_spearman", "valid_days"],
    )

    result = audit.audit_validation_similarity(frame, FACTOR_IDS)

    assert result["status"] == "PASS"
    assert result["observed_pair_count"] == 3
    assert result["max_median_abs_spearman"] == pytest.approx(0.8)
    assert result["minimum_valid_days_observed"] == 237


def test_audit_validation_similarity_blocks_threshold_or_day_count_failure() -> None:
    frame = pd.DataFrame(
        [
            ["yeslab_376", "yeslab_380", 0.4, 237],
            ["yeslab_376", "yeslab_396", 0.9, 237],
            ["yeslab_380", "yeslab_396", 0.8, 30],
        ],
        columns=["factor_id_left", "factor_id_right", "median_daily_abs_spearman", "valid_days"],
    )

    result = audit.audit_validation_similarity(frame, FACTOR_IDS, minimum_valid_days=60)

    assert result["status"] == "FAIL"
    assert result["pair_failures"] == [
        ["yeslab_376", "yeslab_396"],
        ["yeslab_380", "yeslab_396"],
    ]
