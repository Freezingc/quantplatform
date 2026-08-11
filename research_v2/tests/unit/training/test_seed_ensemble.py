from __future__ import annotations

import pandas as pd
import pytest

from research_v2.training.ensemble import REQUIRED_SEEDS, build_seed_ensemble


def _predictions(first: float, second: float) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "signal_date": ["2025-01-02", "2025-01-02"],
            "ts_code": ["a", "b"],
            "percentile_rank": [first, second],
        }
    )


def test_seed_ensemble_uses_all_three_fixed_seeds_and_equal_weight() -> None:
    ensemble = build_seed_ensemble(
        {
            REQUIRED_SEEDS[0]: _predictions(1.0, 0.0),
            REQUIRED_SEEDS[1]: _predictions(0.5, 0.5),
            REQUIRED_SEEDS[2]: _predictions(0.0, 1.0),
        },
        validation_rank_ic={
            REQUIRED_SEEDS[0]: 0.01,
            REQUIRED_SEEDS[1]: 0.02,
            REQUIRED_SEEDS[2]: 0.03,
        },
    )

    assert ensemble.predictions["ensemble_score"].tolist() == [0.5, 0.5]
    assert ensemble.seed_mean_validation_rank_ic == pytest.approx(0.02)
    assert ensemble.worst_seed == REQUIRED_SEEDS[0]


def test_seed_ensemble_rejects_missing_seed_or_mismatched_business_keys() -> None:
    with pytest.raises(ValueError, match="all required"):
        build_seed_ensemble(
            {REQUIRED_SEEDS[0]: _predictions(1.0, 0.0)},
            validation_rank_ic={REQUIRED_SEEDS[0]: 0.01},
        )
    with pytest.raises(ValueError, match="identical"):
        build_seed_ensemble(
            {
                REQUIRED_SEEDS[0]: _predictions(1.0, 0.0),
                REQUIRED_SEEDS[1]: _predictions(0.5, 0.5),
                REQUIRED_SEEDS[2]: _predictions(0.0, 1.0).iloc[:1],
            },
            validation_rank_ic={seed: 0.01 for seed in REQUIRED_SEEDS},
        )
