"""Tests for the isolated canonical Alpha484 admission evaluator."""

from __future__ import annotations

import evaluate_alpha484_canonical as evaluation
import pytest


def test_candidate_identity_is_isolated_from_historical_factor() -> None:
    assert evaluation.CANDIDATE_ID == "yeslab_484_tushare_shsz_v1"
    assert evaluation.SOURCE_FACTOR_ID == "yeslab_484"
    assert evaluation.CANDIDATE_ID != evaluation.SOURCE_FACTOR_ID


def test_parse_args_rejects_the_locked_test_boundary() -> None:
    with pytest.raises(SystemExit):
        evaluation.parse_args(["--end", "2026-01-01", "--output", "/tmp/out"])
