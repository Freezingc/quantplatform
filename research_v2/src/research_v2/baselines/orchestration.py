"""Validation-first orchestration for one outer walk-forward baseline fold."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

import pandas as pd

from research_v2.baselines.base import Baseline, BaselineDataset, FittedBaseline, PredictionFrame
from research_v2.baselines.equal_weight import EqualWeightBaseline
from research_v2.baselines.lgbm_ranker import LGBMRankerCandidate
from research_v2.baselines.linear import LinearBaselineCandidate
from research_v2.baselines.single_factor import SingleFactorBaseline
from research_v2.evaluation.rank_ic import RankICSummary, compute_daily_rank_ic, summarize_rank_ic

SIMPLE_BASELINE_IDS = ("B0", "B1", "B2", "B3")
_BASELINE_COMPLEXITY = {
    baseline_id: index for index, baseline_id in enumerate(SIMPLE_BASELINE_IDS)
}


@dataclass(frozen=True)
class BaselineFold:
    """Train/validation views plus a deferred loader for the outer test view."""

    fold_id: str
    train: BaselineDataset
    validation: BaselineDataset
    factor_order: tuple[str, ...]
    frozen_directions: Mapping[str, int]
    test_loader: Callable[[], BaselineDataset] | None = None

    def __post_init__(self) -> None:
        if not self.fold_id.strip():
            raise ValueError("fold_id cannot be empty")
        if self.train.fold_id != self.fold_id or self.validation.fold_id != self.fold_id:
            raise ValueError("all baseline datasets must belong to the fold")
        if self.train.role != "TRAIN" or self.validation.role != "VALIDATION":
            raise PermissionError("baseline folds require TRAIN and VALIDATION datasets")
        if self.test_loader is not None and not callable(self.test_loader):
            raise TypeError("test_loader must be callable")
        if not self.factor_order or len(set(self.factor_order)) != len(self.factor_order):
            raise ValueError("factor_order must be non-empty and unique")
        expected_columns = ("signal_date", "ts_code", *self.factor_order)
        if (
            tuple(self.train.features.columns) != expected_columns
            or tuple(self.validation.features.columns) != expected_columns
        ):
            raise ValueError("fold feature columns must exactly match factor_order")
        directions = dict(self.frozen_directions)
        if tuple(directions) != self.factor_order or any(
            direction not in {-1, 1} for direction in directions.values()
        ):
            raise ValueError("frozen_directions must align with factor_order and contain +/-1")
        object.__setattr__(self, "frozen_directions", MappingProxyType(directions))


@dataclass(frozen=True)
class BaselineValidationScore:
    baseline_id: str
    mean_rank_ic: float
    valid_day_count: int
    invalid_day_count: int


@dataclass(frozen=True)
class BaselineSelectionRecord:
    """The immutable decision record written before an outer test can be loaded."""

    fold_id: str
    selected_baseline_id: str
    validation_scores: tuple[BaselineValidationScore, ...]
    unselected_reasons: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class BaselineTestEvaluation:
    baseline_id: str
    predictions: PredictionFrame
    labels: pd.DataFrame
    daily_rank_ic: pd.DataFrame
    summary: RankICSummary


@dataclass(frozen=True)
class BaselineFoldResult:
    selection_record: BaselineSelectionRecord
    test_evaluation: BaselineTestEvaluation | None


def run_baseline_fold(
    fold: BaselineFold,
    *,
    candidates: Mapping[str, Baseline],
    minimum_cross_section: int = 30,
    selection_recorder: Callable[[BaselineSelectionRecord], None] | None = None,
) -> BaselineFoldResult:
    """Fit B0-B3, select by validation RankIC, then optionally evaluate one test winner.

    The test loader is deliberately a callback. It is invoked only after the selection
    record has been assembled and handed to selection_recorder.
    """
    if minimum_cross_section < 2:
        raise ValueError("minimum_cross_section must be at least 2")
    _require_complete_candidate_set(candidates)
    if fold.test_loader is not None and selection_recorder is None:
        raise ValueError("test evaluation requires a selection_recorder")

    fitted: dict[str, FittedBaseline] = {}
    validation_scores: list[BaselineValidationScore] = []
    for baseline_id in SIMPLE_BASELINE_IDS:
        fitted_baseline = candidates[baseline_id].fit(fold.train, fold.validation)
        if not callable(getattr(fitted_baseline, "predict", None)):
            raise TypeError(f"{baseline_id} fit did not return a fitted baseline")
        prediction = fitted_baseline.predict(fold.validation)
        _validate_prediction_contract(prediction, fold.validation, baseline_id, fold.fold_id)
        summary = _evaluate_predictions(
            prediction,
            fold.validation,
            minimum_cross_section=minimum_cross_section,
        )[1]
        fitted[baseline_id] = fitted_baseline
        validation_scores.append(
            BaselineValidationScore(
                baseline_id=baseline_id,
                mean_rank_ic=summary.mean_rank_ic,
                valid_day_count=summary.valid_day_count,
                invalid_day_count=summary.invalid_day_count,
            )
        )

    winner = _select_winner(validation_scores)
    selection_record = BaselineSelectionRecord(
        fold_id=fold.fold_id,
        selected_baseline_id=winner.baseline_id,
        validation_scores=tuple(validation_scores),
        unselected_reasons=tuple(
            (
                score.baseline_id,
                _unselected_reason(score, winner),
            )
            for score in validation_scores
            if score.baseline_id != winner.baseline_id
        ),
    )
    if selection_recorder is not None:
        selection_recorder(selection_record)

    if fold.test_loader is None:
        return BaselineFoldResult(selection_record=selection_record, test_evaluation=None)

    test = fold.test_loader()
    if test.role != "TEST" or test.fold_id != fold.fold_id:
        raise PermissionError("test_loader must return the same fold's TEST dataset")
    test_prediction = fitted[winner.baseline_id].predict(test)
    _validate_prediction_contract(test_prediction, test, winner.baseline_id, fold.fold_id)
    daily, summary = _evaluate_predictions(
        test_prediction,
        test,
        minimum_cross_section=minimum_cross_section,
    )
    return BaselineFoldResult(
        selection_record=selection_record,
        test_evaluation=BaselineTestEvaluation(
            baseline_id=winner.baseline_id,
            predictions=test_prediction,
            labels=test.labels,
            daily_rank_ic=daily,
            summary=summary,
        ),
    )


def build_default_baseline_candidates(
    fold: BaselineFold,
    *,
    minimum_cross_section: int = 30,
) -> dict[str, Baseline]:
    """Build the approved B0-B3 candidates from one fold's frozen factor metadata."""
    if minimum_cross_section < 2:
        raise ValueError("minimum_cross_section must be at least 2")
    directions = dict(fold.frozen_directions)
    return {
        "B0": SingleFactorBaseline(
            factor_order=fold.factor_order,
            frozen_directions=directions,
            minimum_cross_section=minimum_cross_section,
        ),
        "B1": EqualWeightBaseline(
            factor_order=fold.factor_order,
            frozen_directions=directions,
        ),
        "B2": LinearBaselineCandidate(
            factor_order=fold.factor_order,
            minimum_cross_section=minimum_cross_section,
        ),
        "B3": LGBMRankerCandidate(factor_order=fold.factor_order),
    }


def _require_complete_candidate_set(candidates: Mapping[str, Baseline]) -> None:
    actual = set(candidates)
    expected = set(SIMPLE_BASELINE_IDS)
    if actual != expected:
        missing = sorted(expected.difference(actual))
        unexpected = sorted(actual.difference(expected))
        details: list[str] = []
        if missing:
            details.append(f"missing baselines: {', '.join(missing)}")
        if unexpected:
            details.append(f"unexpected baselines: {', '.join(unexpected)}")
        raise ValueError("baseline candidates must contain exactly B0-B3: " + "; ".join(details))


def _evaluate_predictions(
    prediction: PredictionFrame,
    dataset: BaselineDataset,
    *,
    minimum_cross_section: int,
) -> tuple[pd.DataFrame, RankICSummary]:
    daily = compute_daily_rank_ic(
        prediction.frame.loc[:, ["signal_date", "ts_code", "score"]],
        dataset.labels,
        minimum_cross_section=minimum_cross_section,
        include_invalid=True,
        require_score_health=False,
    )
    return daily, summarize_rank_ic(daily)


def _select_winner(scores: list[BaselineValidationScore]) -> BaselineValidationScore:
    eligible = [score for score in scores if math.isfinite(score.mean_rank_ic)]
    if not eligible:
        raise ValueError("B0-B3 have no valid validation RankIC observations")
    return min(
        eligible,
        key=lambda score: (-score.mean_rank_ic, _BASELINE_COMPLEXITY[score.baseline_id]),
    )


def _unselected_reason(
    score: BaselineValidationScore,
    winner: BaselineValidationScore,
) -> str:
    if not math.isfinite(score.mean_rank_ic):
        return "NO_VALID_VALIDATION_RANKIC"
    if score.mean_rank_ic < winner.mean_rank_ic:
        return "LOWER_VALIDATION_RANKIC"
    return "COMPLEXITY_TIE_BREAK"


def _validate_prediction_contract(
    prediction: PredictionFrame,
    dataset: BaselineDataset,
    baseline_id: str,
    fold_id: str,
) -> None:
    frame = prediction.frame
    if set(frame["model_id"].astype(str)) != {baseline_id}:
        raise ValueError(f"{baseline_id} prediction model_id does not match the candidate")
    if set(frame["fold_id"].astype(str)) != {fold_id}:
        raise ValueError(f"{baseline_id} prediction fold_id does not match the fold")
    if set(frame["dataset_id"].astype(str)) != {dataset.dataset_id}:
        raise ValueError(f"{baseline_id} prediction dataset_id does not match the dataset")
    expected = dataset.features.loc[:, ["signal_date", "ts_code"]].copy()
    actual = frame.loc[:, ["signal_date", "ts_code"]].copy()
    expected["signal_date"] = pd.to_datetime(expected["signal_date"], errors="raise").dt.date
    actual["signal_date"] = pd.to_datetime(actual["signal_date"], errors="raise").dt.date
    joined = expected.merge(actual, on=["signal_date", "ts_code"], how="outer", indicator=True)
    if not joined["_merge"].eq("both").all() or len(joined) != len(expected):
        raise ValueError(f"{baseline_id} prediction keys do not match the dataset")


__all__ = [
    "SIMPLE_BASELINE_IDS",
    "BaselineFold",
    "BaselineFoldResult",
    "BaselineSelectionRecord",
    "BaselineTestEvaluation",
    "BaselineValidationScore",
    "build_default_baseline_candidates",
    "run_baseline_fold",
]
