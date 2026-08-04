from __future__ import annotations

import json

import pandas as pd
from compare_repaired_factor_quality import build_comparison_report, main


def _audit() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "factor_id": ["yeslab_2", "yeslab_1", "yeslab_1"],
            "filled_rows": [0, 3, 2],
            "added_rows": [0, 0, 1],
            "remaining_nonfinite_rows": [0, 1, 0],
            "decision_status": [
                "REPAIR_AUDIT_PASS",
                "REPAIR_AUDIT_PASS_WITH_UNRESOLVED_QUARANTINE",
                "REPAIR_AUDIT_PASS",
            ],
            "data_version": ["repair-v1"] * 3,
        }
    )


def test_comparison_report_is_sorted_and_contains_quality_deltas() -> None:
    before = pd.DataFrame(
        {
            "factor_id": ["yeslab_2", "yeslab_1"],
            "finite_ratio": [0.80, 0.70],
            "mean_rank_ic": [0.01, 0.02],
            "valid_day_count": [40, 50],
        }
    )
    after = pd.DataFrame(
        {
            "factor_id": ["yeslab_1", "yeslab_2"],
            "finite_ratio": [0.95, 0.80],
            "mean_rank_ic": [0.03, 0.015],
            "valid_day_count": [60, 40],
        }
    )

    report = build_comparison_report(_audit(), before, after)

    assert report["factor_id"].tolist() == ["yeslab_1", "yeslab_2"]
    assert {
        "before",
        "after",
        "filled_rows",
        "remaining_nonfinite_rows",
        "finite_ratio_delta",
        "mean_rank_ic_delta",
        "repair_status",
        "repair_data_version",
    }.issubset(report.columns)
    assert report.loc[0, "repair_status"] == "UNRESOLVED"
    assert report.loc[0, "finite_ratio_delta"] == 0.25
    assert json.loads(report.loc[0, "before"])["finite_ratio"] == 0.70


def test_main_writes_csv_from_injected_frames_without_database_access(tmp_path) -> None:
    output = tmp_path / "comparison.csv"
    result = main(
        ["--output", str(output)],
        audit_frame=_audit(),
        before_frame=pd.DataFrame({"factor_id": ["yeslab_1"], "finite_ratio": [0.7]}),
        after_frame=pd.DataFrame({"factor_id": ["yeslab_1"], "finite_ratio": [0.9]}),
    )

    assert output.exists()
    assert result["factor_id"].tolist() == ["yeslab_1", "yeslab_2"]
    assert pd.read_csv(output)["factor_id"].tolist() == ["yeslab_1", "yeslab_2"]


def test_main_writes_dependency_free_markdown(tmp_path) -> None:
    output = tmp_path / "comparison.md"
    main(
        ["--output", str(output), "--format", "markdown"],
        audit_frame=_audit(),
        before_frame=pd.DataFrame({"factor_id": ["yeslab_1"], "finite_ratio": [0.7]}),
        after_frame=pd.DataFrame({"factor_id": ["yeslab_1"], "finite_ratio": [0.9]}),
    )

    text = output.read_text(encoding="utf-8")
    assert text.startswith("| factor_id |")
    assert "yeslab_1" in text


def test_nested_snapshot_metrics_are_flattened_with_leaf_aliases() -> None:
    before = pd.DataFrame(
        {
            "factor_id": ["yeslab_1"],
            "metrics_json": [
                json.dumps(
                    {
                        "predictive": {"finite_ratio": 0.70, "mean_rank_ic": 0.02},
                        "validation": {"finite_ratio": 0.60},
                        "structural": {"valid_day_count": 40},
                    }
                )
            ],
        }
    )
    after = pd.DataFrame(
        {
            "factor_id": ["yeslab_1"],
            "metrics_json": [
                json.dumps(
                    {
                        "predictive": {"finite_ratio": 0.90, "mean_rank_ic": 0.03},
                        "validation": {"finite_ratio": 0.80},
                        "structural": {"valid_day_count": 60},
                    }
                )
            ],
        }
    )

    report = build_comparison_report(_audit().iloc[[1]], before, after)

    assert report.loc[0, "before_finite_ratio"] == 0.70
    assert report.loc[0, "after_finite_ratio"] == 0.90
    assert report.loc[0, "finite_ratio_delta"] == 0.20
    assert report.loc[0, "before_predictive.finite_ratio"] == 0.70
    assert report.loc[0, "before_predictive_finite_ratio"] == 0.70
    assert report.loc[0, "valid_day_count_delta"] == 20.0
    assert report.loc[0, "mean_rank_ic_delta"] == 0.01


def test_mixed_repair_statuses_use_blocker_precedence() -> None:
    audit = pd.DataFrame(
        {
            "factor_id": ["yeslab_3"] * 4,
            "decision_status": [
                "REPAIR_AUDIT_PASS",
                "REPAIRED_PENDING_QUALITY_RECHECK",
                "REPAIR_AUDIT_PASS_WITH_UNRESOLVED_QUARANTINE",
                "REPAIR_AUDIT_PASS",
            ],
            "data_version": ["v1"] * 4,
        }
    )

    report = build_comparison_report(
        audit,
        pd.DataFrame({"factor_id": []}),
        pd.DataFrame({"factor_id": []}),
    )

    assert report.loc[0, "decision_status"] == (
        "REPAIR_AUDIT_PASS_WITH_UNRESOLVED_QUARANTINE"
    )
    assert report.loc[0, "repair_status"] == "UNRESOLVED"
