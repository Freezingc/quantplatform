from __future__ import annotations

import json
from pathlib import Path

import audit_shsz_admission_preflight as preflight


def test_required_pairs_are_all_unique_unordered_edges() -> None:
    result = preflight.required_pairs(
        ["yeslab_380", "yeslab_431", "yeslab_437", "yeslab_155"]
    )

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
