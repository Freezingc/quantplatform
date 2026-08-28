"""Tests for the canonical candidate-pair comparison contract."""

from __future__ import annotations

from datetime import date

import audit_canonical_candidate_pairs as audit
import pytest


def test_registry_has_exactly_six_versioned_candidates() -> None:
    assert [item.candidate_id for item in audit.CANDIDATES] == [
        "yeslab_114_tushare_shsz_v2",
        "yeslab_381_tushare_shsz_v1",
        "yeslab_405_tushare_shsz_v1",
        "yeslab_425_tushare_shsz_v2",
        "yeslab_484_tushare_shsz_v1",
        "yeslab_206_median_abs_v2",
    ]


def test_parser_defaults_to_the_fixed_comparison_window() -> None:
    args = audit.parse_args(["--output", "/tmp/x"])

    assert args.start == date(2020, 1, 2)
    assert args.end == date(2025, 12, 31)


def test_parser_requires_an_output_location() -> None:
    with pytest.raises(SystemExit):
        audit.parse_args([])


def test_parser_rejects_locked_test_range() -> None:
    with pytest.raises(SystemExit):
        audit.parse_args(
            ["--start", "2025-01-01", "--end", "2026-01-01", "--output", "/tmp/x"]
        )


def test_classify_pair_requires_complete_phase_days() -> None:
    assert audit.classify_pair(0.20, 1212, 0.20, 242) == "INCOMPLETE"


def test_classify_pair_marks_a_pair_below_monitor_threshold_as_unique() -> None:
    assert audit.classify_pair(0.79, 1212, 0.20, 243) == "PAIR_UNIQUE"


def test_classify_pair_marks_a_pair_at_monitor_threshold() -> None:
    assert audit.classify_pair(0.80, 1212, 0.20, 243) == "ELEVATED_PAIR_REDUNDANCY_MONITOR"


def test_classify_pair_marks_a_pair_at_redundancy_threshold() -> None:
    assert audit.classify_pair(0.20, 1212, 0.90, 243) == "REDUNDANT"
