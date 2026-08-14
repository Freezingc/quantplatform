"""Regression checks for the committed Alpha95 audit evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[4]
DIAGNOSTICS = ROOT / "research_v2" / "artifacts" / "diagnostics"
LEDGER_PATH = DIAGNOSTICS / "tushare_shsz_factor_audit_coverage_v1.json"
QUALITY_PATH = (
    DIAGNOSTICS
    / "tushare_alpha95_relative_amount_volatility_v1"
    / "quality_summary.json"
)


def test_alpha95_ledger_metrics_match_the_canonical_quality_artifact() -> None:
    """Prevent manually transcribed Alpha95 quality metrics from drifting."""
    ledger = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
    quality = json.loads(QUALITY_PATH.read_text(encoding="utf-8"))

    candidates = ledger["alpha95_relative_amount_volatility_v1"]["candidates"]
    for candidate_id, ledger_candidate in candidates.items():
        actual = quality["candidates"][candidate_id]
        for ledger_window, artifact_window in (
            ("train", "train_2020_2024"),
            ("validation", "validation_2025"),
        ):
            expected = ledger_candidate[ledger_window]
            observed = actual[artifact_window]
            assert expected["valid_day_count"] == observed["valid_day_count"]
            assert expected["signed_mean_rank_ic"] == pytest.approx(
                observed["signed_mean_rank_ic"]
            )
            assert expected["signed_icir"] == pytest.approx(observed["signed_icir"])
            assert expected["directional_day_share"] == pytest.approx(
                observed["directional_day_share"]
            )
        assert ledger_candidate["validation"]["max_similarity_to_formal_nine"] == pytest.approx(
            actual["validation_2025"]["max_similarity"]
        )
