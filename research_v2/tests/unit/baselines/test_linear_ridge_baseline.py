from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import pytest

from research_v2.baselines.linear import LinearRidgeBaseline


def test_linear_baseline_fits_train_only_and_freezes_feature_order() -> None:
    rows: list[dict[str, object]] = []
    labels: list[dict[str, object]] = []
    for signal_date in (date(2020, 1, 2), date(2020, 1, 3)):
        for index in range(30):
            first = float(index)
            second = float(29 - index)
            rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": f"S{index:03d}",
                    "f1": first,
                    "f2": second,
                }
            )
            labels.append(
                {
                    "signal_date": signal_date,
                    "ts_code": f"S{index:03d}",
                    "raw_return_5d": 2 * first - second,
                    "label_valid": True,
                }
            )
    baseline = LinearRidgeBaseline.fit(
        pd.DataFrame(rows),
        pd.DataFrame(labels),
        factor_order=("f1", "f2"),
        role="TRAIN",
        alpha=1e-8,
    )

    scores = baseline.predict(pd.DataFrame(rows))

    assert np.corrcoef(scores, pd.DataFrame(labels)["raw_return_5d"])[0, 1] > 0.999
    with pytest.raises(PermissionError, match="TRAIN"):
        LinearRidgeBaseline.fit(
            pd.DataFrame(rows),
            pd.DataFrame(labels),
            factor_order=("f1", "f2"),
            role="TEST",
        )
    with pytest.raises(ValueError, match="feature order"):
        baseline.predict(pd.DataFrame(rows).loc[:, ["signal_date", "ts_code", "f2", "f1"]])
