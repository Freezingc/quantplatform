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
