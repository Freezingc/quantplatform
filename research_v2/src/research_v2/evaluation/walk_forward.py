"""Strict F01–F06 assembly for ordinary locked-free walk-forward OOS evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from research_v2.evaluation.model_evaluator import FrozenOosEvaluation, evaluate_frozen_oos

ORDINARY_TEST_FOLDS = ("F01", "F02", "F03", "F04", "F05", "F06")


@dataclass(frozen=True)
class OosFold:
    candidate_predictions: pd.DataFrame
    labels: pd.DataFrame
    baseline_predictions: pd.DataFrame


def evaluate_walk_forward_oos(
    folds: Mapping[str, OosFold],
    *,
    minimum_cross_section: int = 30,
) -> FrozenOosEvaluation:
    """Evaluate exactly F01–F06 together; 2026 FINAL is intentionally outside this path."""
    expected = set(ORDINARY_TEST_FOLDS)
    actual = set(folds)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        unexpected = sorted(actual.difference(expected))
        details = []
        if missing:
            details.append(f"missing folds: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected folds: {', '.join(unexpected)}")
        raise ValueError("walk-forward folds are incomplete: " + "; ".join(details))
    candidate_predictions = pd.concat(
        [folds[fold_id].candidate_predictions for fold_id in ORDINARY_TEST_FOLDS],
        ignore_index=True,
    )
    labels = pd.concat(
        [folds[fold_id].labels for fold_id in ORDINARY_TEST_FOLDS],
        ignore_index=True,
    )
    baseline_predictions = pd.concat(
        [folds[fold_id].baseline_predictions for fold_id in ORDINARY_TEST_FOLDS],
        ignore_index=True,
    )
    return evaluate_frozen_oos(
        candidate_predictions,
        labels,
        baseline_predictions=baseline_predictions,
        role="TEST",
        minimum_cross_section=minimum_cross_section,
    )
