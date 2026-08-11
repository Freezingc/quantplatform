from __future__ import annotations

import pandas as pd

from research_v2.evaluation.hac import hac_mean_test


def test_hac_mean_test_uses_configured_overlap_lag() -> None:
    result = hac_mean_test(pd.Series([0.01, 0.02, 0.03, 0.02, 0.01, 0.02, 0.03]), max_lags=5)

    assert result.observation_count == 7
    assert result.max_lags == 5
    assert result.mean > 0
    assert result.standard_error > 0
