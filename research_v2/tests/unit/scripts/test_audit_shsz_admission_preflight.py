from __future__ import annotations

import json
from pathlib import Path

import audit_shsz_admission_preflight as preflight


def test_required_pairs_are_all_unique_unordered_edges() -> None:
    result = preflight.required_pairs(["yeslab_380", "yeslab_431", "yeslab_437", "yeslab_155"])

    assert len(result) == 6
    assert ("yeslab_380", "yeslab_431") in result
    assert ("yeslab_155", "yeslab_437") in result
    assert all(left < right for left, right in result)


def test_collect_similarity_evidence_reads_json_pairs_and_stored_pair(tmp_path: Path) -> None:
    matrix = tmp_path / "matrix.json"
    matrix.write_text(
        json.dumps(
            {
                "pairs": [
                    {
                        "factor_id_left": "yeslab_431",
                        "factor_id_right": "yeslab_437",
                        "median_abs_spearman": 0.78,
                    }
                ]
            }
        )
    )
    stored = tmp_path / "stored.json"
    stored.write_text(
        json.dumps(
            {
                "rematerialized_factor_ids": ["yeslab_155"],
                "stored_tushare_canonical_factor_ids": ["yeslab_380"],
                "median_abs_spearman": 0.25,
            }
        )
    )

    result = preflight.collect_similarity_evidence([matrix, stored])

    assert result[("yeslab_431", "yeslab_437")]["metric"] == 0.78
    assert result[("yeslab_155", "yeslab_380")]["metric"] == 0.25


def test_collect_similarity_evidence_keeps_train_and_validation_metrics_separate(
    tmp_path: Path,
) -> None:
    train = tmp_path / "train.csv"
    train.write_text(
        "factor_id_left,factor_id_right,median_daily_abs_spearman,valid_days\n"
        "yeslab_380,yeslab_405,0.60,1212\n"
    )
    validation = tmp_path / "validation.csv"
    validation.write_text(
        "factor_id_left,factor_id_right,median_daily_abs_spearman,valid_days\n"
        "yeslab_380,yeslab_405,0.70,243\n"
    )

    result: dict[tuple[str, str], dict[str, object]] = {}
    for path, period in ((train, "train"), (validation, "validation")):
        evidence = preflight.collect_similarity_evidence([path], period=period)
        for key, row in evidence.items():
            result.setdefault(key, {}).setdefault("period_metrics", {}).update(
                row["period_metrics"]
            )

    assert result[("yeslab_380", "yeslab_405")]["period_metrics"] == {
        "train": {"metric": 0.60, "source": str(train)},
        "validation": {"metric": 0.70, "source": str(validation)},
    }


def test_collect_similarity_evidence_reads_list_similarity_artifact(tmp_path: Path) -> None:
    path = tmp_path / "similarity.json"
    path.write_text(
        json.dumps(
            [
                {
                    "factor_id_left": "yeslab_380",
                    "factor_id_right": "yeslab_405",
                    "median_abs_spearman": 0.61,
                }
            ]
        )
    )

    result = preflight.collect_similarity_evidence([path], period="validation")

    assert result[("yeslab_380", "yeslab_405")]["period_metrics"]["validation"]["metric"] == 0.61


def test_build_preflight_fails_closed_when_a_required_pair_is_missing() -> None:
    proposal = {
        "market_scope": ["SH", "SZ"],
        "label": "forward_open_5d",
        "similarity_threshold": 0.9,
        "candidates": [
            {
                "factor_id": "yeslab_380",
                "predictive_gate": "PASS",
                "annual_stability_gate": "MONITOR",
            },
            {
                "factor_id": "yeslab_431",
                "predictive_gate": "PASS",
                "annual_stability_gate": "PASS",
            },
        ],
    }

    result = preflight.build_preflight(proposal, {}, {})

    assert result["preflight_status"] == "FAIL"
    assert result["publication_status"] == "NOT_PUBLISHED"
    assert result["missing_pairs"] == [["yeslab_380", "yeslab_431"]]
    assert result["database_writes"] is False


def test_build_preflight_passes_with_monitoring_when_all_edges_are_below_threshold() -> None:
    proposal = {
        "market_scope": ["SH", "SZ"],
        "label": "forward_open_5d",
        "similarity_threshold": 0.9,
        "candidates": [
            {
                "factor_id": "yeslab_380",
                "predictive_gate": "PASS",
                "annual_stability_gate": "MONITOR",
            },
            {
                "factor_id": "yeslab_431",
                "predictive_gate": "PASS",
                "annual_stability_gate": "PASS",
            },
        ],
    }
    pair = {("yeslab_380", "yeslab_431"): {"metric": 0.4, "source": "test"}}

    result = preflight.build_preflight(proposal, {}, pair)

    assert result["preflight_status"] == "PASS_WITH_MONITORING"
    assert result["missing_pairs"] == []
    assert result["monitoring_factor_ids"] == ["yeslab_380"]
    assert result["publication_status"] == "NOT_PUBLISHED"


def test_build_preflight_blocks_pending_formula_scope_gate() -> None:
    proposal = {
        "market_scope": ["SH", "SZ"],
        "label": "forward_open_5d",
        "similarity_threshold": 0.9,
        "candidates": [
            {
                "factor_id": "yeslab_376",
                "predictive_gate": "PASS",
                "strict_annual_icir_gate": "PASS",
                "formula_scope_gate": "PENDING_SCOPE_TRANSITION_REVIEW",
            },
            {
                "factor_id": "yeslab_380",
                "predictive_gate": "PASS",
                "annual_stability_gate": "MONITOR",
            },
        ],
    }
    pair = {("yeslab_376", "yeslab_380"): {"metric": 0.4, "source": "test"}}

    result = preflight.build_preflight(proposal, {}, pair)

    assert result["preflight_status"] == "FAIL"
    assert result["failed_checks"] == ["candidate:yeslab_376"]
    assert result["candidate_results"][0]["formula_scope_gate"] == "PENDING_SCOPE_TRANSITION_REVIEW"


def test_quality_status_accepts_root_level_aggregate_icir() -> None:
    assert preflight._quality_status({"factor_id": "yeslab_376", "icir": -0.70}) == "PASS"


def test_load_quality_summaries_accepts_scoreboard_rows(tmp_path: Path) -> None:
    path = tmp_path / "scoreboard.json"
    path.write_text(json.dumps({"rows": [{"factor_id": "yeslab_380", "train_icir": -0.59}]}))

    result = preflight.load_quality_summaries([path])

    assert result["yeslab_380"]["train_icir"] == -0.59
    assert preflight._quality_status(result["yeslab_380"]) == "PASS"


def test_build_preflight_requires_every_declared_similarity_period() -> None:
    proposal = {
        "market_scope": ["SH", "SZ"],
        "label": "forward_open_5d",
        "similarity_threshold": 0.9,
        "similarity_periods": ["train", "validation"],
        "candidates": [
            {"factor_id": "yeslab_380", "predictive_gate": "PASS", "annual_stability_gate": "PASS"},
            {"factor_id": "yeslab_405", "predictive_gate": "PASS", "annual_stability_gate": "PASS"},
        ],
    }
    evidence = {
        ("yeslab_380", "yeslab_405"): {
            "period_metrics": {"train": {"metric": 0.6, "source": "train"}}
        }
    }

    result = preflight.build_preflight(proposal, {}, evidence)

    assert result["preflight_status"] == "FAIL"
    assert result["missing_pairs"] == [["yeslab_380", "yeslab_405", "validation"]]
    assert result["required_pair_count"] == 2
