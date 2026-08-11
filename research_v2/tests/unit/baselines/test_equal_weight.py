from __future__ import annotations

import math

import pandas as pd

from research_v2.baselines.equal_weight import equal_weight_scores


def test_equal_weight_preserves_all_missing_rows() -> None:
    features = pd.DataFrame({"f1": [1.0, None], "f2": [3.0, None]})

    scores = equal_weight_scores(features, ("f1", "f2"))

    assert scores.iloc[0] == 2.0
    assert math.isnan(scores.iloc[1])
