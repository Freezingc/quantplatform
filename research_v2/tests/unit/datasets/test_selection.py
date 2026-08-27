from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from research_v2.datasets.selection import (
    benjamini_hochberg,
    select_cluster_representatives,
    select_training_factors,
)


def test_benjamini_hochberg_is_deterministic_for_ties() -> None:
    adjusted = benjamini_hochberg(
        {("z", 1): 0.03, ("a", 1): 0.03, ("b", 1): 0.01}
    )

    assert adjusted[("b", 1)] == 0.03
    assert adjusted[("a", 1)] == 0.03
    assert adjusted[("z", 1)] == 0.03


def test_selector_uses_train_dates_and_freezes_unknown_direction() -> None:
    dates = [
        date(2015 + index // 26, 1, 1) + timedelta(days=index % 26)
        for index in range(130)
    ]
    rows: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    for signal_date in dates:
        for number in range(30):
            value = float(number)
            labels.append(
                {
                    "signal_date": signal_date,
                    "ts_code": f"S{number:03d}",
                    "raw_return_5d": value,
                    "label_valid": True,
                }
            )
            rows.extend(
                [
                    {
                        "signal_date": signal_date,
                        "ts_code": f"S{number:03d}",
                        "factor_id": "positive",
                        "factor_version": 1,
                        "value": value,
                    },
                    {
                        "signal_date": signal_date,
                        "ts_code": f"S{number:03d}",
                        "factor_id": "unknown_negative",
                        "factor_version": 1,
                        "value": -value,
                    },
                ]
            )
    observations = pd.DataFrame(rows)
    label_frame = pd.DataFrame(labels)
    specs = pd.DataFrame(
        [
            {"factor_id": "positive", "factor_version": 1, "expected_direction": 1},
            {"factor_id": "unknown_negative", "factor_version": 1, "expected_direction": 0},
        ]
    )

    result = select_training_factors(
        observations,
        label_frame,
        factor_specs=specs,
        train_start=dates[0],
        train_end=dates[-1],
        inner_validation_ics={
            ("positive", 1): (0.1, 0.1, -0.1),
            ("unknown_negative", 1): (-0.1, -0.1, -0.1),
        },
    )

    selected = result.candidates.set_index("factor_id")
    assert selected.loc["positive", "status"] == "ELIGIBLE"
    assert selected.loc["positive", "frozen_direction"] == 1
    assert selected.loc["unknown_negative", "status"] == "ELIGIBLE"
    assert selected.loc["unknown_negative", "frozen_direction"] == -1

    future_date = date(2030, 1, 1)
    future_observations = pd.concat(
        [
            observations,
            pd.DataFrame(
                [
                    {
                        "signal_date": future_date,
                        "ts_code": "S000",
                        "factor_id": "positive",
                        "factor_version": 1,
                        "value": -999.0,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    future_labels = pd.concat(
        [
            label_frame,
            pd.DataFrame(
                [
                    {
                        "signal_date": future_date,
                        "ts_code": "S000",
                        "raw_return_5d": 999.0,
                        "label_valid": True,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    unchanged = select_training_factors(
        future_observations,
        future_labels,
        factor_specs=specs,
        train_start=dates[0],
        train_end=dates[-1],
        inner_validation_ics={
            ("positive", 1): (0.1, 0.1, -0.1),
            ("unknown_negative", 1): (-0.1, -0.1, -0.1),
        },
    )
    assert result.candidates.equals(unchanged.candidates)


def test_cluster_representative_uses_stable_business_tie_breakers() -> None:
    candidates = pd.DataFrame(
        [
            {
                "factor_id": "b",
                "factor_version": 1,
                "status": "ELIGIBLE",
                "fdr_q_value": 0.02,
                "signed_mean_rank_ic": 0.10,
                "positive_year_count": 4,
                "coverage": 0.95,
                "conflict_rate": 0.01,
            },
            {
                "factor_id": "a",
                "factor_version": 1,
                "status": "ELIGIBLE",
                "fdr_q_value": 0.01,
                "signed_mean_rank_ic": 0.09,
                "positive_year_count": 3,
                "coverage": 0.90,
                "conflict_rate": 0.02,
            },
        ]
    )
    similarity = pd.DataFrame(
        [
            {
                "factor_id_left": "a",
                "factor_version_left": 1,
                "factor_id_right": "b",
                "factor_version_right": 1,
                "median_daily_abs_spearman": 0.95,
                "status": "PASS",
            }
        ]
    )

    chosen = select_cluster_representatives(candidates, similarity, minimum_features=1)

    assert chosen.selected_factor_keys == (("a", 1),)
    assert chosen.status == "READY"
