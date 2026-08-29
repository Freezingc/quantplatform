from __future__ import annotations

import json

import audit_shsz_performance_drift as audit
import pandas as pd


def test_performance_drift_passes_direction_and_ratio_gates() -> None:
    frame = pd.DataFrame(
        [
            ["yeslab_376", -0.08, -0.70, -0.09, -0.78],
            ["yeslab_442", 0.07, 0.60, 0.08, 0.62],
        ],
        columns=[
            "factor_id",
            "train_mean_rank_ic",
            "train_icir",
            "validation_mean_rank_ic",
            "validation_icir",
        ],
    )

    result = audit.audit_performance_drift(frame)

    assert result["status"] == "PASS"
    assert result["direction_flip_factor_ids"] == []
    assert all(row["drift_status"] == "PASS" for row in result["rows"])


def test_performance_drift_blocks_direction_flip_and_icir_collapse() -> None:
    frame = pd.DataFrame(
        [
            ["yeslab_bad_direction", -0.08, -0.70, 0.01, 0.60],
            ["yeslab_bad_icir", -0.08, -0.70, -0.03, -0.20],
        ],
        columns=[
            "factor_id",
            "train_mean_rank_ic",
            "train_icir",
            "validation_mean_rank_ic",
            "validation_icir",
        ],
    )

    result = audit.audit_performance_drift(frame)

    assert result["status"] == "FAIL"
    assert result["direction_flip_factor_ids"] == ["yeslab_bad_direction"]
    assert result["validation_predictive_fail_factor_ids"] == [
        "yeslab_bad_direction",
        "yeslab_bad_icir",
    ]
    assert result["drift_fail_factor_ids"] == ["yeslab_bad_direction", "yeslab_bad_icir"]


def test_load_quality_rows_merges_split_train_and_validation_layouts(tmp_path) -> None:
    train = tmp_path / "train.json"
    train.write_text(
        json.dumps(
            {
                "start": "2020-01-02",
                "summaries": [
                    {
                        "factor_id": "yeslab_376",
                        "mean_rank_ic": -0.08,
                        "icir": -0.70,
                    }
                ],
            }
        )
    )
    validation = tmp_path / "validation.json"
    validation.write_text(
        json.dumps(
            {
                "start": "2025-01-02",
                "summaries": [
                    {
                        "factor_id": "yeslab_376",
                        "mean_rank_ic": -0.09,
                        "icir": -0.78,
                    }
                ],
            }
        )
    )

    result = audit.load_quality_rows([train, validation], ("yeslab_376",))

    assert result.to_dict("records") == [
        {
            "factor_id": "yeslab_376",
            "train_mean_rank_ic": -0.08,
            "train_icir": -0.70,
            "validation_mean_rank_ic": -0.09,
            "validation_icir": -0.78,
        }
    ]
