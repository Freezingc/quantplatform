from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from research_v2.baselines.lgbm_ranker import LGBMRankerBaseline


def test_lgbm_ranker_is_train_only_and_scores_frozen_feature_order() -> None:
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for day_offset in range(5):
        signal_date = date(2020, 1, 2) + timedelta(days=day_offset)
        for index in range(30):
            feature_rows.append(
                {"signal_date": signal_date, "ts_code": f"S{index:03d}", "f1": float(index)}
            )
            label_rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": f"S{index:03d}",
                    "raw_return_5d": float(index),
                    "label_valid": True,
                }
            )
    features = pd.DataFrame(feature_rows)
    labels = pd.DataFrame(label_rows)
    baseline = LGBMRankerBaseline.fit(
        features,
        labels,
        factor_order=("f1",),
        role="TRAIN",
        num_boost_round=30,
    )

    scores = baseline.predict(features)

    assert np.isfinite(scores).all()
    assert np.corrcoef(scores, features["f1"])[0, 1] > 0.7
    with pytest.raises(PermissionError, match="TRAIN"):
        LGBMRankerBaseline.fit(features, labels, factor_order=("f1",), role="TEST")


def test_lgbm_ranker_supports_cross_sections_larger_than_default_label_gain() -> None:
    feature_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    for day_offset in range(2):
        signal_date = date(2020, 2, 3) + timedelta(days=day_offset)
        for index in range(64):
            feature_rows.append(
                {"signal_date": signal_date, "ts_code": f"S{index:03d}", "f1": float(index)}
            )
            label_rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": f"S{index:03d}",
                    "raw_return_5d": float(index),
                    "label_valid": True,
                }
            )
    baseline = LGBMRankerBaseline.fit(
        pd.DataFrame(feature_rows),
        pd.DataFrame(label_rows),
        factor_order=("f1",),
        role="TRAIN",
        num_boost_round=5,
    )

    scores = baseline.predict(pd.DataFrame(feature_rows))

    assert np.isfinite(scores).all()
