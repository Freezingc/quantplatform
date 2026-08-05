from __future__ import annotations

import build_shsz_factor_quality_scoreboard as scoreboard


def test_build_scoreboard_combines_gates_and_marks_monitoring() -> None:
    result = scoreboard.build_scoreboard(
        factor_ids=("yeslab_380", "yeslab_376"),
        preflight={
            "preflight_status": "PASS_WITH_MONITORING",
            "monitoring_factor_ids": ["yeslab_380"],
            "candidate_results": [
                {
                    "factor_id": "yeslab_380",
                    "predictive_gate": "PASS",
                    "annual_stability_gate": "MONITOR",
                    "formula_scope_gate": "PASS",
                    "quality_status": "PASS",
                    "status": "PASS",
                },
                {
                    "factor_id": "yeslab_376",
                    "predictive_gate": "PASS",
                    "annual_stability_gate": "PASS",
                    "formula_scope_gate": "PASS",
                    "quality_status": "PASS",
                    "status": "PASS",
                },
            ],
        },
        drift={
            "rows": [
                {
                    "factor_id": "yeslab_380",
                    "direction_consistent": True,
                    "validation_predictive_status": "PASS",
                    "drift_status": "PASS",
                },
                {
                    "factor_id": "yeslab_376",
                    "direction_consistent": True,
                    "validation_predictive_status": "PASS",
                    "drift_status": "PASS",
                },
            ]
        },
        annual_pass_years={"yeslab_380": 4, "yeslab_376": 5},
        validation_similarity={"status": "PASS", "pair_results": []},
    )

    assert result["status"] == "PASS_WITH_MONITORING"
    assert result["factor_count"] == 2
    assert result["monitoring_factor_ids"] == ["yeslab_380"]
    assert result["rows"][0]["factor_id"] == "yeslab_376"
    assert result["rows"][1]["admission_status"] == "PASS_WITH_MONITORING"


def test_build_scoreboard_fails_when_a_factor_is_missing_from_a_gate() -> None:
    try:
        scoreboard.build_scoreboard(
            factor_ids=("yeslab_380", "yeslab_376"),
            preflight={
                "preflight_status": "PASS",
                "monitoring_factor_ids": [],
                "candidate_results": [
                    {
                        "factor_id": "yeslab_380",
                        "predictive_gate": "PASS",
                        "annual_stability_gate": "PASS",
                        "formula_scope_gate": "PASS",
                        "quality_status": "PASS",
                        "status": "PASS",
                    }
                ],
            },
            drift={"rows": []},
            annual_pass_years={"yeslab_380": 4},
            validation_similarity={"status": "PASS", "pair_results": []},
        )
    except ValueError as error:
        assert "yeslab_376" in str(error)
    else:
        raise AssertionError("missing factor evidence must fail closed")


def test_load_annual_pass_years_accepts_canonical_annual_key(tmp_path) -> None:
    artifact = tmp_path / "canonical.json"
    artifact.write_text(
        '{"annual_icir_abs_pass_years": {"yeslab_431": 3, "yeslab_437": 2}}'
    )

    result = scoreboard._load_annual_pass_years([artifact], ("yeslab_431", "yeslab_437"))

    assert result == {"yeslab_431": 3, "yeslab_437": 2}


def test_build_scoreboard_includes_sample_coverage_gate() -> None:
    result = scoreboard.build_scoreboard(
        factor_ids=("yeslab_380",),
        preflight={
            "preflight_status": "PASS",
            "monitoring_factor_ids": [],
            "candidate_results": [
                {
                    "factor_id": "yeslab_380",
                    "predictive_gate": "PASS",
                    "annual_stability_gate": "PASS",
                    "formula_scope_gate": "PASS",
                    "quality_status": "PASS",
                    "status": "PASS",
                }
            ],
        },
        drift={
            "rows": [
                {
                    "factor_id": "yeslab_380",
                    "direction_consistent": True,
                    "validation_predictive_status": "PASS",
                    "drift_status": "PASS",
                }
            ]
        },
        annual_pass_years={"yeslab_380": 4},
        validation_similarity={"status": "PASS", "pair_results": []},
        sample_coverage={
            "status": "PASS",
            "rows": [{"factor_id": "yeslab_380", "status": "PASS"}],
        },
    )

    assert result["sample_coverage_status"] == "PASS"
    assert result["rows"][0]["sample_coverage_status"] == "PASS"
