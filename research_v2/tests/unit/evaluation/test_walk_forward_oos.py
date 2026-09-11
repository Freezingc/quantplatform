from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.evaluation.walk_forward import OosFold, evaluate_walk_forward_oos


def _fold(year: int) -> OosFold:
    rows: list[dict[str, object]] = []
    baseline: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    # Two test dates per fold: evaluate_frozen_oos refuses OOS with <=5 valid days,
    # and four folds at one date each only reach four.
    for signal_date in (date(year, 1, 2), date(year, 1, 3)):
        for index in range(30):
            key = {"signal_date": signal_date, "ts_code": f"S{index:03d}"}
            fold_id = f"F{year - 2021}"
            dataset_id = f"dataset-{year}"
            rows.append(
                {
                    **key,
                    "score": float(index),
                    "model_id": "candidate",
                    "fold_id": fold_id,
                    "dataset_id": dataset_id,
                }
            )
            baseline.append(
                {
                    **key,
                    "score": float(-index),
                    "model_id": "baseline",
                    "fold_id": fold_id,
                    "dataset_id": dataset_id,
                }
            )
            labels.append(
                {
                    **key,
                    "raw_return_5d": float(index),
                    "label_valid": True,
                    "fold_id": fold_id,
                    "dataset_id": dataset_id,
                }
            )
    return OosFold(
        candidate_predictions=pd.DataFrame(rows),
        labels=pd.DataFrame(labels),
        baseline_predictions=pd.DataFrame(baseline),
    )


def test_walk_forward_requires_all_four_ordinary_test_folds() -> None:
    folds = {f"F{index}": _fold(2021 + index) for index in range(1, 5)}

    evaluation = evaluate_walk_forward_oos(folds)

    assert evaluation.summary.valid_day_count == 8
    assert evaluation.delta_rank_ic == pytest.approx(2.0)


def _unlabelled_rows(fold_id: str, year: int, *, with_score: bool) -> pd.DataFrame:
    """Rows with no usable label, optionally scored, on both of the fold's test dates."""
    rows = [
        {
            "signal_date": date(year, 1, day),
            "ts_code": f"U{day:03d}",
            "raw_return_5d": float("nan"),
            "label_valid": False,
            "fold_id": fold_id,
            "dataset_id": f"dataset-{year}",
        }
        for day in (2, 3)
    ]
    frame = pd.DataFrame(rows)
    if with_score:
        frame["score"] = [float(index) for index in range(len(frame))]
        frame["model_id"] = "baseline"
    return frame


def test_walk_forward_accepts_a_baseline_row_no_candidate_could_ever_score() -> None:
    """The frozen simple baseline scores rows that have no usable label, and that is not a gap.

    Measured on the real F1 fold: the best_simple baseline covers 747,461 rows, the torch candidate
    803,572, and the 940 baseline rows the candidate lacks contain **zero** scoreable rows -- every
    one is ``label_valid=False`` or has a non-finite return. The metric's usable intersection is
    746,521 rows whether or not those 940 are covered, because ``evaluate_frozen_oos`` discards them
    either way. So requiring the candidate to cover them protected nothing at all, and it was the
    third blocker behind which no torch family could reach a verdict.

    What the check exists for is unchanged: a candidate that drops a row the baseline could be
    compared on would compute the G4 delta over a narrower cross-section than the baseline supports,
    and still return a plausible number. That case is pinned by the next test.
    """
    folds = {f"F{index}": _fold(2021 + index) for index in range(1, 5)}
    fold = folds["F1"]
    extra = _unlabelled_rows("F1", 2022, with_score=False)
    folds["F1"] = OosFold(
        candidate_predictions=fold.candidate_predictions,
        labels=pd.concat([fold.labels, extra], ignore_index=True),
        baseline_predictions=pd.concat(
            [fold.baseline_predictions, _unlabelled_rows("F1", 2022, with_score=True)],
            ignore_index=True,
        ),
    )

    evaluation = evaluate_walk_forward_oos(folds)

    assert evaluation.summary.valid_day_count == 8
    assert evaluation.delta_rank_ic == pytest.approx(2.0)


def test_walk_forward_rejects_a_baseline_row_outside_the_label_frame() -> None:
    """Relaxing coverage of unscoreable rows must not relax membership of the TEST frame.

    The old identity-based candidate check caught this as a side effect: a baseline row invented
    out of nowhere could not be in the candidate, and the candidate could not leave the label frame.
    With coverage required only over scoreable rows that implication is gone, so the requirement is
    stated directly instead of being left to fall out of another check.
    """
    folds = {f"F{index}": _fold(2021 + index) for index in range(1, 5)}
    fold = folds["F1"]
    invented = pd.DataFrame(
        [
            {
                "signal_date": date(2022, 1, 2),
                "ts_code": "NOT_IN_THE_TEST_FRAME",
                "score": 1.0,
                "model_id": "baseline",
                "fold_id": "F1",
                "dataset_id": "dataset-2022",
            }
        ]
    )
    folds["F1"] = OosFold(
        candidate_predictions=fold.candidate_predictions,
        labels=fold.labels,
        baseline_predictions=pd.concat(
            [fold.baseline_predictions, invented], ignore_index=True
        ),
    )

    with pytest.raises(ValueError, match="not in the verified TEST frame"):
        evaluate_walk_forward_oos(folds)
    with pytest.raises(ValueError, match="F4"):
        evaluate_walk_forward_oos({key: value for key, value in folds.items() if key != "F4"})


def test_walk_forward_rejects_a_fold_with_dates_outside_its_test_year() -> None:
    folds = {f"F{index}": _fold(2021 + index) for index in range(1, 5)}
    folds["F3"] = _fold(2022)

    with pytest.raises(ValueError, match="F3|2024"):
        evaluate_walk_forward_oos(folds)


def test_walk_forward_rejects_a_candidate_row_outside_the_label_frame() -> None:
    """Renamed and re-messaged: the mismatch this pinned is still refused, now by name.

    Relabelling one label row's security used to trip a requirement that the two frames hold
    IDENTICAL business keys. That requirement is gone -- the torch carriers legitimately score only
    the label-valid subset -- but this mutation is still rejected, because the candidate now names a
    row the label frame does not contain. The guard got more specific, not weaker.
    """
    folds = {f"F{index}": _fold(2021 + index) for index in range(1, 5)}
    mismatched = folds["F1"].labels.copy()
    mismatched.loc[0, "ts_code"] = "MISSING"
    folds["F1"] = OosFold(
        candidate_predictions=folds["F1"].candidate_predictions,
        labels=mismatched,
        baseline_predictions=folds["F1"].baseline_predictions,
    )

    with pytest.raises(ValueError, match="not in the verified TEST frame"):
        evaluate_walk_forward_oos(folds)


def test_walk_forward_rejects_a_candidate_that_drops_a_scoreable_label_row() -> None:
    """The other direction, and the one that would otherwise fail silently.

    Accepting a subset of the label frame is only safe because the subset must contain every row the
    metric can use. Drop one of those and RankIC would be computed over a quietly smaller
    cross-section while still returning a plausible number -- which is why coverage is required in
    both directions rather than identity in one.
    """
    folds = {f"F{index}": _fold(2021 + index) for index in range(1, 5)}
    trimmed = folds["F1"].candidate_predictions.iloc[1:].reset_index(drop=True)
    folds["F1"] = OosFold(
        candidate_predictions=trimmed,
        labels=folds["F1"].labels,
        baseline_predictions=folds["F1"].baseline_predictions,
    )

    with pytest.raises(ValueError, match="usable label are not scored"):
        evaluate_walk_forward_oos(folds)


def test_walk_forward_accepts_a_candidate_that_skips_rows_without_a_usable_label() -> None:
    """The torch carriers' row set, at the evaluator's own level.

    Both torch batch builders drop rows without a usable label before scoring, so their frozen
    candidate predictions are a strict subset of the verified TEST frame -- 803,572 of 804,575 rows
    on the real F1 fold. The metric never used the unlabelled rows anyway: it restricts itself to
    ``label_valid`` with a finite return. So the unlabelled rows may be absent from the candidate,
    and the delta must come out exactly as it does when they are present.
    """
    folds = {f"F{index}": _fold(2021 + index) for index in range(1, 5)}
    fold = folds["F1"]
    unlabelled = pd.DataFrame(
        [
            {
                "signal_date": date(2022, 1, day),
                "ts_code": f"U{day:03d}",
                "raw_return_5d": float("nan"),
                "label_valid": False,
                "fold_id": "F1",
                "dataset_id": "dataset-2022",
            }
            for day in (2, 3)
        ]
    )
    folds["F1"] = OosFold(
        candidate_predictions=fold.candidate_predictions,
        labels=pd.concat([fold.labels, unlabelled], ignore_index=True),
        baseline_predictions=fold.baseline_predictions,
    )

    evaluation = evaluate_walk_forward_oos(folds)

    assert evaluation.summary.valid_day_count == 8
    assert evaluation.delta_rank_ic == pytest.approx(2.0)


def test_walk_forward_rejects_empty_oos_frames() -> None:
    folds = {f"F{index}": _fold(2021 + index) for index in range(1, 5)}
    empty = folds["F1"].candidate_predictions.iloc[0:0].copy()
    folds["F1"] = OosFold(
        candidate_predictions=empty,
        labels=folds["F1"].labels,
        baseline_predictions=folds["F1"].baseline_predictions,
    )

    with pytest.raises(ValueError, match="cannot be empty"):
        evaluate_walk_forward_oos(folds)
