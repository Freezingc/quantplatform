from __future__ import annotations

import math
from collections.abc import Callable
from datetime import date

import pandas as pd
import pytest

from research_v2.evaluation.model_evaluator import evaluate_frozen_oos


def test_frozen_oos_evaluator_compares_only_test_predictions_to_baseline() -> None:
    prediction_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for year in range(2020, 2026):
        signal_date = date(year, 1, 2)
        for index in range(30):
            label_rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": f"S{index:03d}",
                    "raw_return_5d": float(index),
                    "label_valid": True,
                }
            )
            prediction_rows.append(
                {"signal_date": signal_date, "ts_code": f"S{index:03d}", "score": float(index)}
            )
            baseline_rows.append(
                {"signal_date": signal_date, "ts_code": f"S{index:03d}", "score": float(-index)}
            )

    evaluation = evaluate_frozen_oos(
        pd.DataFrame(prediction_rows),
        pd.DataFrame(label_rows),
        baseline_predictions=pd.DataFrame(baseline_rows),
        role="TEST",
    )

    assert evaluation.summary.mean_rank_ic == pytest.approx(1.0)
    assert evaluation.baseline_mean_rank_ic == pytest.approx(-1.0)
    assert evaluation.delta_rank_ic == pytest.approx(2.0)
    assert set(evaluation.yearly_mean_rank_ic) == set(range(2020, 2026))
    assert set(evaluation.yearly_summary["year"]) == set(range(2020, 2026))
    assert evaluation.gate_report.results["G6"].status.value == "PASS"
    with pytest.raises(PermissionError, match="TEST"):
        evaluate_frozen_oos(
            pd.DataFrame(prediction_rows),
            pd.DataFrame(label_rows),
            role="VALIDATION",
        )


def test_candidate_and_baseline_rank_ic_use_identical_key_coverage() -> None:
    prediction_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for day_offset in range(6):
        signal_date = date(2020 + day_offset, 1, 2)
        for index in range(30):
            label_rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": f"S{index:03d}",
                    "raw_return_5d": float(index),
                    "label_valid": True,
                }
            )
            prediction_rows.append(
                {"signal_date": signal_date, "ts_code": f"S{index:03d}", "score": float(index)}
            )
            if index != 0:
                baseline_rows.append(
                    {
                        "signal_date": signal_date,
                        "ts_code": f"S{index:03d}",
                        "score": float(-index),
                    }
                )

    evaluation = evaluate_frozen_oos(
        pd.DataFrame(prediction_rows),
        pd.DataFrame(label_rows),
        baseline_predictions=pd.DataFrame(baseline_rows),
        role="TEST",
        minimum_cross_section=29,
    )

    assert evaluation.daily_rank_ic["sample_count"].tolist() == [29] * 6


def test_frozen_oos_evaluator_blocks_a_collapsed_score_day() -> None:
    prediction_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for day_offset in range(7):
        signal_date = date(2020, 1, 2 + day_offset)
        for index in range(30):
            key = {"signal_date": signal_date, "ts_code": f"S{index:03d}"}
            label_rows.append({**key, "raw_return_5d": float(index), "label_valid": True})
            prediction_rows.append({**key, "score": float(index) if day_offset != 0 else 1.0})

    evaluation = evaluate_frozen_oos(
        pd.DataFrame(prediction_rows),
        pd.DataFrame(label_rows),
        role="TEST",
    )

    assert evaluation.summary.valid_day_count == 6
    assert evaluation.gate_report.results["G6"].status.value == "FAIL"


def test_frozen_oos_normalizes_intraday_dates_for_matching_and_gate_evidence() -> None:
    prediction_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for day_offset in range(6):
        day = pd.Timestamp("2020-01-02") + pd.Timedelta(days=day_offset)
        for index in range(30):
            key = {"signal_date": day + pd.Timedelta(hours=9), "ts_code": f"S{index:03d}"}
            label_rows.append({**key, "raw_return_5d": float(index), "label_valid": True})
            prediction_rows.append(
                {
                    "signal_date": day + pd.Timedelta(hours=10),
                    "ts_code": f"S{index:03d}",
                    "score": float(index),
                }
            )
            baseline_rows.append(
                {
                    "signal_date": day + pd.Timedelta(hours=11),
                    "ts_code": f"S{index:03d}",
                    "score": float(-index),
                }
            )

    evaluation = evaluate_frozen_oos(
        pd.DataFrame(prediction_rows),
        pd.DataFrame(label_rows),
        baseline_predictions=pd.DataFrame(baseline_rows),
        role="TEST",
    )

    assert evaluation.summary.valid_day_count == 6
    assert evaluation.gate_report.results["G6"].status.value == "PASS"


def test_frozen_oos_rejects_locked_year_from_ordinary_test_entrypoint() -> None:
    frame = pd.DataFrame({"signal_date": [pd.Timestamp("2026-01-02")]})

    with pytest.raises(PermissionError, match="2026|FINAL"):
        evaluate_frozen_oos(frame, frame, role="TEST")


def test_frozen_oos_baseline_report_marks_relative_gate_not_applicable() -> None:
    prediction_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for year in range(2020, 2026):
        signal_date = date(year, 1, 2)
        for index in range(30):
            key = {"signal_date": signal_date, "ts_code": f"S{index:03d}"}
            prediction_rows.append({**key, "score": float(index)})
            label_rows.append({**key, "raw_return_5d": float(index), "label_valid": True})

    evaluation = evaluate_frozen_oos(
        pd.DataFrame(prediction_rows),
        pd.DataFrame(label_rows),
        role="TEST",
        is_baseline=True,
    )

    assert evaluation.gate_report.results["G4"].status.value == "NOT_APPLICABLE"


# --- the evidence the gates are handed ---------------------------------------
#
# The six tests above check what the evaluator decides. These check what it
# assembles, because a gate that judges correctly on evidence assembled wrongly is
# not a working gate. Each corresponds to a mutation of model_evaluator.py that
# survived a full measurement round, and each was probed first so the assertion
# below states an observed difference rather than a suspected one.

_NINE_DAYS = [date(2020, 1, 2 + offset) for offset in range(4)] + [
    date(year, 1, 2) for year in range(2021, 2026)
]


def _nine_day_frames(
    *,
    stocks: Callable[[int], int] | None = None,
    label_valid: Callable[[int, int], object] | None = None,
    returns: Callable[[int, int], float] | None = None,
    scores: Callable[[int, int], float] | None = None,
    baseline: Callable[[int, int], float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame | None]:
    """Nine signal days over six years, 30 stocks a day, every gate satisfiable.

    Nine days rather than six so a test that invalidates one day still clears the
    evaluator's own "more than five valid days" requirement -- three of the probes
    behind these tests first measured that guard instead of what they were about.
    Six distinct years so G3 and G5 see a complete yearly picture.
    """
    predictions: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    for day_index, day in enumerate(_NINE_DAYS):
        count = 30 if stocks is None else stocks(day_index)
        for index in range(count):
            key = {"signal_date": day, "ts_code": f"S{index:03d}"}
            score = float(index) if scores is None else scores(day_index, index)
            value = float(index) if returns is None else returns(day_index, index)
            valid = True if label_valid is None else label_valid(day_index, index)
            predictions.append({**key, "score": score})
            labels.append({**key, "raw_return_5d": value, "label_valid": valid})
            if baseline is not None:
                baseline_rows.append({**key, "score": baseline(day_index, index)})
    return (
        pd.DataFrame(predictions),
        pd.DataFrame(labels),
        pd.DataFrame(baseline_rows) if baseline is not None else None,
    )


def test_nine_day_fixture_is_a_clean_complete_evaluation() -> None:
    """Guard the guard: if this fixture is not clean, the five tests below prove nothing."""
    predictions, labels, baseline = _nine_day_frames(baseline=lambda day, index: float(-index))

    evaluation = evaluate_frozen_oos(
        predictions, labels, baseline_predictions=baseline, role="TEST"
    )

    assert evaluation.summary.valid_day_count == 9
    assert sorted(evaluation.yearly_mean_rank_ic) == list(range(2020, 2026))
    assert evaluation.gate_report.results["G6"].status.value == "PASS"
    assert evaluation.delta_rank_ic == pytest.approx(2.0)


def test_locked_year_in_the_baseline_frame_is_refused_by_name() -> None:
    """The locked-date wall walks the baseline frame too, and must say so when it fires.

    Probed: with the baseline left out of the checked frames, a baseline artifact
    carrying a FINAL/2026 row does not leak into the numbers -- the coverage join
    drops it because no candidate key matches -- but it is silently tolerated, and
    the call returns a complete, normal-looking verdict (nine valid days, six years,
    delta 2.0). Refusing is the point. A mis-scoped baseline artifact is a governance
    event, and "the join cleaned it up afterwards" is not evidence the input was in
    scope; the next artifact may not be so conveniently unmatched.
    """
    predictions, labels, baseline = _nine_day_frames(baseline=lambda day, index: float(-index))
    assert baseline is not None
    locked_row = {"signal_date": date(2026, 1, 5), "ts_code": "S000", "score": 1.0}
    tainted = pd.concat([baseline, pd.DataFrame([locked_row])], ignore_index=True)

    with pytest.raises(PermissionError, match="baseline"):
        evaluate_frozen_oos(predictions, labels, baseline_predictions=tainted, role="TEST")


def test_a_baseline_with_no_valid_observation_is_refused_where_it_happens() -> None:
    """A collapsed baseline must raise here rather than travel onwards as NaN.

    Probed: with the guard removed the call succeeds and returns
    baseline_mean_rank_ic=nan and delta_rank_ic=nan. G4 still fails, so nothing is
    promoted -- but ``json.dumps`` of that field emits the bare token ``NaN``, which
    is not strict JSON, and the refusal moves downstream to the registry's
    finite-number check, far from the collapsed baseline that caused it.
    """
    predictions, labels, baseline = _nine_day_frames(baseline=lambda day, index: 1.0)

    with pytest.raises(ValueError, match="baseline has no valid"):
        evaluate_frozen_oos(predictions, labels, baseline_predictions=baseline, role="TEST")


def test_a_year_without_a_finite_mean_never_reaches_the_report() -> None:
    """Non-finite yearly means are filtered out of the evidence instead of passed on.

    Probed: without the filter the returned yearly_mean_rank_ic carries ``2023: nan``.
    Both readings fail G3 and G5, because gates.py requires every required year to be
    present and finite, so this is not a promotion risk -- but the NaN lands in the run
    report, and the registry then refuses the evidence for "not finite" instead of for
    "year missing", which is a much worse clue to whoever has to debug it.
    """
    predictions, labels, _ = _nine_day_frames(
        scores=lambda day, index: 1.0 if _NINE_DAYS[day].year == 2023 else float(index)
    )

    evaluation = evaluate_frozen_oos(predictions, labels, role="TEST")

    assert 2023 not in evaluation.yearly_mean_rank_ic
    assert all(math.isfinite(value) for value in evaluation.yearly_mean_rank_ic.values())
    assert evaluation.gate_report.results["G3"].status.value == "FAIL"


def test_g6_denominator_ignores_null_label_flags_and_non_finite_returns() -> None:
    """G6's denominator counts days that really carry labels, not days that look full.

    Both halves probed. A day with 20 labelled stocks and 25 whose ``label_valid`` is
    NULL must count as 8 of 9 days, not 9: treating NULL as valid makes it a test day,
    the numerator and denominator then disagree, and a clean evaluation fails G6 with
    observed_value None. A day whose 30 forward returns are all NaN likewise does not
    count. Neither direction can promote a model that should fail -- both make G6
    stricter -- but both write a wrong day count into the evidence file that the
    registry and any human reader take at face value.
    """
    null_flag_predictions, null_flag_labels, _ = _nine_day_frames(
        stocks=lambda day: 45 if day == 8 else 30,
        label_valid=lambda day, index: True if day != 8 else (True if index < 20 else None),
    )
    null_flag = evaluate_frozen_oos(null_flag_predictions, null_flag_labels, role="TEST")
    assert null_flag.gate_report.results["G6"].status.value == "PASS"
    assert null_flag.gate_report.results["G6"].observed_value == pytest.approx(1.0)

    nan_predictions, nan_labels, _ = _nine_day_frames(
        returns=lambda day, index: float("nan") if day == 8 else float(index),
    )
    nan_returns = evaluate_frozen_oos(nan_predictions, nan_labels, role="TEST")
    assert nan_returns.gate_report.results["G6"].status.value == "PASS"
    assert nan_returns.gate_report.results["G6"].observed_value == pytest.approx(1.0)


def test_a_thin_day_stays_out_of_the_g6_numerator_so_the_two_sides_agree() -> None:
    """G6's numerator and denominator must be counting the same set of days.

    Probed: dropping the day filter in ``_score_health_evidence`` lets a 25-stock day
    into the numerator while ``_valid_label_day_count`` still refuses it, so the counts
    disagree and gates.py fails G6 closed with observed_value None -- a clean nine-day
    evaluation that should pass instead fails. That is a false rejection, not a
    promotion risk, but it is the kind that reads as "the model got worse" when only
    the bookkeeping changed.
    """
    predictions, labels, _ = _nine_day_frames(stocks=lambda day: 25 if day == 8 else 30)

    evaluation = evaluate_frozen_oos(predictions, labels, role="TEST")

    assert evaluation.summary.valid_day_count == 8
    assert evaluation.gate_report.results["G6"].status.value == "PASS"
    assert evaluation.gate_report.results["G6"].observed_value == pytest.approx(1.0)
