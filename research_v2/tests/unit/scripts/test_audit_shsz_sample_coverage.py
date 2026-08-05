from __future__ import annotations

import pandas as pd

import audit_shsz_sample_coverage as audit


def _frame(rows):
    return pd.DataFrame(rows, columns=["factor_id", "signal_date", "sample_count"])


def test_sample_coverage_passes_stable_train_and_validation_panels() -> None:
    train = _frame(
        [
            ["yeslab_376", "2020-01-02", 80],
            ["yeslab_376", "2020-01-03", 100],
            ["yeslab_380", "2020-01-02", 90],
            ["yeslab_380", "2020-01-03", 100],
        ]
    )
    validation = _frame(
        [
            ["yeslab_376", "2025-01-02", 95],
            ["yeslab_376", "2025-01-03", 100],
            ["yeslab_380", "2025-01-02", 90],
            ["yeslab_380", "2025-01-03", 100],
        ]
    )

    result = audit.audit_sample_coverage(
        train, validation, ("yeslab_376", "yeslab_380"), minimum_valid_days=2
    )

    assert result["status"] == "PASS"
    assert result["failed_factor_ids"] == []
    assert result["rows"][0]["train_min_to_median"] == 80 / 90


def test_sample_coverage_blocks_a_daily_sample_collapse() -> None:
    train = _frame(
        [
            ["yeslab_376", "2020-01-02", 40],
            ["yeslab_376", "2020-01-03", 100],
        ]
    )
    validation = _frame(
        [
            ["yeslab_376", "2025-01-02", 90],
            ["yeslab_376", "2025-01-03", 100],
        ]
    )

    result = audit.audit_sample_coverage(
        train, validation, ("yeslab_376",), minimum_valid_days=2, minimum_min_to_median=0.7
    )

    assert result["status"] == "FAIL"
    assert result["failed_factor_ids"] == ["yeslab_376"]
