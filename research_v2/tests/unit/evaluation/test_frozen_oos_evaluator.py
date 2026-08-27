from __future__ import annotations

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
