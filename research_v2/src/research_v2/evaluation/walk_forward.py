"""Strict F1-F4 walk-forward assembly and validation-first baseline orchestration."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import pandas as pd

from research_v2.baselines.base import Baseline
from research_v2.baselines.orchestration import (
    BaselineFold,
    BaselineFoldResult,
    BaselineSelectionRecord,
    build_default_baseline_candidates,
    run_baseline_fold,
)
from research_v2.datasets.splits import annual_folds
from research_v2.evaluation.model_evaluator import FrozenOosEvaluation, evaluate_frozen_oos

# The ordinary (non-locked) evaluation folds and their test years are DERIVED from
# datasets.splits.annual_folds -- the single source of truth for the fold contract --
# rather than restated here. Restating them is exactly how this module kept requiring
# F01-F06 after the v2 contract moved the grid to F1-F4 (it then rejected the v2
# baseline run with "missing folds: F01..F06; unexpected folds: F1..F4"). Deriving them
# means the next fold-contract change cannot strand this gate the same way.
ORDINARY_TEST_FOLDS = tuple(fold.fold_id for fold in annual_folds() if not fold.locked)
_ORDINARY_TEST_YEARS = {
    fold.fold_id: fold.test_years[0] for fold in annual_folds() if not fold.locked
}


@dataclass(frozen=True)
class OosFold:
    candidate_predictions: pd.DataFrame
    labels: pd.DataFrame
    baseline_predictions: pd.DataFrame


@dataclass(frozen=True)
class BaselineWalkForwardResult:
    fold_results: Mapping[str, BaselineFoldResult]
    selection_records: tuple[BaselineSelectionRecord, ...]
    evaluation: FrozenOosEvaluation


def evaluate_walk_forward_oos(
    folds: Mapping[str, OosFold],
    *,
    minimum_cross_section: int = 30,
    is_baseline: bool = False,
) -> FrozenOosEvaluation:
    """Evaluate exactly the ordinary folds (F1-F4) together; 2026 FINAL is outside this path."""
    _require_ordinary_fold_set(folds)
    candidate_predictions, labels, baseline_predictions = _assemble_oos_frames(folds)
    return evaluate_frozen_oos(
        candidate_predictions,
        labels,
        baseline_predictions=baseline_predictions,
        role="TEST",
        minimum_cross_section=minimum_cross_section,
        is_baseline=is_baseline,
    )


def run_walk_forward_baselines(
    folds: Mapping[str, BaselineFold],
    *,
    candidates_factory: Callable[[BaselineFold], Mapping[str, Baseline]] | None = None,
    minimum_cross_section: int = 30,
    selection_recorder: Callable[[BaselineSelectionRecord], None] | None = None,
) -> BaselineWalkForwardResult:
    """Run B0-B3 independently per outer fold and evaluate only each winner on Test."""
    _require_ordinary_fold_set(folds)
    if minimum_cross_section < 2:
        raise ValueError("minimum_cross_section must be at least 2")
    factory = candidates_factory or (
        lambda fold: build_default_baseline_candidates(
            fold,
            minimum_cross_section=minimum_cross_section,
        )
    )
    records: list[BaselineSelectionRecord] = []
    fold_results: dict[str, BaselineFoldResult] = {}
    oos_folds: dict[str, OosFold] = {}

    def record_selection(record: BaselineSelectionRecord) -> None:
        records.append(record)
        if selection_recorder is not None:
            selection_recorder(record)

    for fold_id in ORDINARY_TEST_FOLDS:
        fold = folds[fold_id]
        result = run_baseline_fold(
            fold,
            candidates=factory(fold),
            minimum_cross_section=minimum_cross_section,
            selection_recorder=record_selection,
        )
        if result.test_evaluation is None:
            raise ValueError(f"{fold_id} baseline run did not produce a TEST evaluation")
        evaluation = result.test_evaluation
        oos_folds[fold_id] = OosFold(
            candidate_predictions=evaluation.predictions.frame,
            labels=_labels_with_lineage(
                evaluation.labels,
                fold_id,
                test_dataset_id=str(evaluation.predictions.frame["dataset_id"].iloc[0]),
            ),
            baseline_predictions=evaluation.predictions.frame,
        )
        fold_results[fold_id] = result

    return BaselineWalkForwardResult(
        fold_results=fold_results,
        selection_records=tuple(records),
        evaluation=evaluate_walk_forward_oos(
            oos_folds,
            minimum_cross_section=minimum_cross_section,
            is_baseline=True,
        ),
    )


def _require_ordinary_fold_set(folds: Mapping[str, object]) -> None:
    expected = set(ORDINARY_TEST_FOLDS)
    actual = set(folds)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        unexpected = sorted(actual.difference(expected))
        details: list[str] = []
        if missing:
            details.append(f"missing folds: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected folds: {', '.join(unexpected)}")
        raise ValueError("walk-forward folds are incomplete: " + "; ".join(details))


def _assemble_oos_frames(
    folds: Mapping[str, OosFold],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    candidate_frames: list[pd.DataFrame] = []
    label_frames: list[pd.DataFrame] = []
    baseline_frames: list[pd.DataFrame] = []
    for fold_id in ORDINARY_TEST_FOLDS:
        expected_year = _ORDINARY_TEST_YEARS[fold_id]
        fold = folds[fold_id]
        candidate_frames.append(
            _validate_oos_frame(
                fold.candidate_predictions,
                name=f"{fold_id} candidate predictions",
                fold_id=fold_id,
                expected_year=expected_year,
                required_columns={
                    "signal_date",
                    "ts_code",
                    "score",
                    "fold_id",
                    "model_id",
                    "dataset_id",
                },
            )
        )
        label_frames.append(
            _validate_oos_frame(
                fold.labels,
                name=f"{fold_id} labels",
                fold_id=fold_id,
                expected_year=expected_year,
                required_columns={
                    "signal_date",
                    "ts_code",
                    "raw_return_5d",
                    "label_valid",
                    "fold_id",
                    "dataset_id",
                },
            )
        )
        baseline_frames.append(
            _validate_oos_frame(
                fold.baseline_predictions,
                name=f"{fold_id} baseline predictions",
                fold_id=fold_id,
                expected_year=expected_year,
                required_columns={
                    "signal_date",
                    "ts_code",
                    "score",
                    "fold_id",
                    "model_id",
                    "dataset_id",
                },
            )
        )
    candidates = pd.concat(candidate_frames, ignore_index=True)
    labels = pd.concat(label_frames, ignore_index=True)
    baselines = pd.concat(baseline_frames, ignore_index=True)
    for name, frame in (
        ("candidate predictions", candidates),
        ("labels", labels),
        ("baseline predictions", baselines),
    ):
        if frame.duplicated(["signal_date", "ts_code"]).any():
            raise ValueError(f"{name} contain duplicate keys across ordinary test folds")
    for fold_id in ORDINARY_TEST_FOLDS:
        candidate = candidate_frames[ORDINARY_TEST_FOLDS.index(fold_id)]
        labels_frame = label_frames[ORDINARY_TEST_FOLDS.index(fold_id)]
        baseline = baseline_frames[ORDINARY_TEST_FOLDS.index(fold_id)]
        _require_matching_keys(
            candidate,
            labels_frame,
            f"{fold_id} candidate predictions and labels",
        )
        _require_baseline_covered_by_candidate(
            candidate,
            baseline,
            f"{fold_id} candidate and baseline predictions",
        )
        dataset_ids = {
            str(candidate["dataset_id"].iloc[0]),
            str(labels_frame["dataset_id"].iloc[0]),
            str(baseline["dataset_id"].iloc[0]),
        }
        if len(dataset_ids) != 1:
            raise ValueError(f"{fold_id} OOS frames do not share one dataset_id")
    return candidates, labels, baselines


def _validate_oos_frame(
    frame: pd.DataFrame,
    *,
    name: str,
    fold_id: str,
    expected_year: int,
    required_columns: set[str],
) -> pd.DataFrame:
    if missing := sorted(required_columns.difference(frame.columns)):
        raise ValueError(f"{name} are missing columns: {missing}")
    if frame.empty:
        raise ValueError(f"{name} cannot be empty")
    normalized = frame.copy(deep=True)
    normalized["signal_date"] = pd.to_datetime(
        normalized["signal_date"], errors="raise"
    ).dt.date
    if normalized["signal_date"].isna().any():
        raise ValueError(f"{name} contain null signal_date")
    years = pd.to_datetime(normalized["signal_date"], errors="raise").dt.year
    if not years.eq(expected_year).all():
        raise ValueError(f"{name} contain dates outside {fold_id} test year {expected_year}")
    if normalized.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError(f"{name} contain duplicate signal-date/security keys")
    for column in ("model_id", "dataset_id"):
        if column in normalized.columns:
            values = normalized[column]
            if values.isna().any() or values.astype(str).eq("").any():
                raise ValueError(f"{name} {column} cannot be null or empty")
            if values.nunique(dropna=False) != 1:
                raise ValueError(f"{name} must use one frozen {column} per fold")
    if "fold_id" in normalized.columns:
        values = normalized["fold_id"]
        if values.isna().any() or values.astype(str).ne(fold_id).any():
            raise ValueError(f"{name} fold_id does not match {fold_id}")
    return normalized.reset_index(drop=True)


def _require_matching_keys(left: pd.DataFrame, right: pd.DataFrame, name: str) -> None:
    keys = ["signal_date", "ts_code"]
    left_keys = left.loc[:, keys].sort_values(keys, kind="mergesort").reset_index(drop=True)
    right_keys = right.loc[:, keys].sort_values(keys, kind="mergesort").reset_index(drop=True)
    if not left_keys.equals(right_keys):
        raise ValueError(f"{name} do not contain identical business keys")


def _require_baseline_covered_by_candidate(
    candidate: pd.DataFrame, baseline: pd.DataFrame, name: str
) -> None:
    """The candidate must cover at least the baseline's test rows, not necessarily equal them.

    A candidate that scores more of the test cross-section than the baseline -- M0 with
    allow_nan scores the sparse rows the dense baseline skips -- is legitimate, and
    evaluate_frozen_oos already restricts both to their common keys via
    _match_prediction_coverage before computing the G4 delta. Requiring IDENTICAL keys
    here rejected exactly that case. What must not happen is the candidate covering FEWER
    of the baseline's rows, because then the delta would silently lose baseline coverage.
    """
    keys = ["signal_date", "ts_code"]
    candidate_keys = set(map(tuple, candidate.loc[:, keys].itertuples(index=False, name=None)))
    baseline_keys = set(map(tuple, baseline.loc[:, keys].itertuples(index=False, name=None)))
    if not baseline_keys.issubset(candidate_keys):
        raise ValueError(f"{name}: baseline rows are not all covered by the candidate")


def _labels_with_lineage(
    labels: pd.DataFrame,
    fold_id: str,
    *,
    test_dataset_id: str,
) -> pd.DataFrame:
    normalized = labels.copy(deep=True)
    normalized["fold_id"] = fold_id
    normalized["dataset_id"] = test_dataset_id
    return normalized


__all__ = [
    "ORDINARY_TEST_FOLDS",
    "BaselineWalkForwardResult",
    "OosFold",
    "evaluate_walk_forward_oos",
    "run_walk_forward_baselines",
]
