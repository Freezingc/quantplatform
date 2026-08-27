from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from research_v2.evaluation.hac import hac_mean_test
from research_v2.evaluation.rank_ic import compute_daily_rank_ic, summarize_rank_ic

_DATA_PATH = Path(__file__).parent / "data" / "dataset_50x60.parquet"
_EXPECTED_PATH = Path(__file__).parent / "data" / "expected_metrics.json"


def test_golden_dataset_has_fixed_50_by_60_shape_and_order() -> None:
    frame = pd.read_parquet(_DATA_PATH, engine="pyarrow")

    assert frame.shape == (300, 64)
    assert frame.columns[:4].tolist() == [
        "signal_date",
        "ts_code",
        "raw_return_5d",
        "label_valid",
    ]
    assert frame.columns[4:].tolist() == [f"factor_{index:02d}" for index in range(60)]
    assert frame["signal_date"].nunique() == 6
    assert frame.groupby("signal_date", sort=True)["ts_code"].nunique().tolist() == [50] * 6


def test_golden_daily_rank_ic_matches_independent_reference_metrics() -> None:
    frame = pd.read_parquet(_DATA_PATH, engine="pyarrow")
    expected = json.loads(_EXPECTED_PATH.read_text(encoding="utf-8"))
    predictions = frame.loc[:, ["signal_date", "ts_code", "factor_00"]].rename(
        columns={"factor_00": "score"}
    )
    labels = frame.loc[:, ["signal_date", "ts_code", "raw_return_5d", "label_valid"]]

    daily = compute_daily_rank_ic(
        predictions,
        labels,
        minimum_cross_section=30,
        include_invalid=True,
        require_score_health=True,
    )
    summary = summarize_rank_ic(daily)
    hac = hac_mean_test(daily.loc[daily["rank_ic_valid"], "rank_ic"], max_lags=5)

    assert daily["rank_ic"].tolist() == pytest.approx(expected["daily_rank_ic"])
    assert daily["sample_count"].tolist() == expected["sample_count"]
    assert daily["coverage"].tolist() == pytest.approx(expected["coverage"])
    assert daily["prediction_unique_count"].tolist() == expected["unique_score_count"]
    assert daily["rank_ic_valid"].tolist() == [True] * 6
    assert summary.mean_rank_ic == pytest.approx(expected["mean_rank_ic"])
    assert summary.rank_ic_std == pytest.approx(expected["rank_ic_std"])
    assert summary.icir == pytest.approx(expected["icir"])
    assert hac.mean == pytest.approx(expected["hac_mean"])
    assert hac.standard_error == pytest.approx(expected["hac_standard_error"])
    assert hac.t_statistic == pytest.approx(expected["hac_t_statistic"])
    assert hac.p_value == pytest.approx(expected["hac_p_value"])


def test_golden_reference_is_not_pooled_across_all_rows() -> None:
    frame = pd.read_parquet(_DATA_PATH, engine="pyarrow")
    daily_reference: list[float] = []
    for _, group in frame.groupby("signal_date", sort=True):
        valid = group["factor_00"].notna()
        daily_reference.append(
            float(
                spearmanr(
                    group.loc[valid, "factor_00"],
                    group.loc[valid, "raw_return_5d"],
                ).statistic
            )
        )

    assert np.mean(daily_reference) != pytest.approx(
        float(
            spearmanr(
                frame["factor_00"].fillna(0.0),
                frame["raw_return_5d"],
            ).statistic
        )
    )
