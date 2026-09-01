from __future__ import annotations

from datetime import date

import pandas as pd
import pytest

from research_v2.datasets.builder import FactorInput
from research_v2.datasets.preprocessing import FoldPreprocessor
from research_v2.inference.builder import build_inference_batch_for_family
from research_v2.inference.predictor import predict_three_seed_ensemble
from research_v2.models.base import FeatureBatch, TemporalFeatureBatch
from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
from research_v2.models.temporal import TemporalGRU, TemporalGRUConfig
from research_v2.training.ensemble import REQUIRED_SEEDS

_SIGNAL = date(2024, 1, 5)
_DAYS = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), _SIGNAL]
_MEMBERS = (FactorInput(factor_id="alpha", factor_version=1, feature_name="f01"),)


def _universe(codes=("a", "b", "c")) -> pd.DataFrame:
    return pd.DataFrame([{"signal_date": _SIGNAL, "ts_code": code} for code in codes])


def _factors() -> pd.DataFrame:
    rows = []
    for day in _DAYS:
        for code in ("a", "b", "c"):
            rows.append(
                {
                    "signal_date": day,
                    "ts_code": code,
                    "factor_id": "alpha",
                    "factor_version": 1,
                    "value": 1.0,
                    "available_date": day,
                }
            )
    return pd.DataFrame(rows)


def _fitted_preprocessor(factor_order=("f01",)) -> FoldPreprocessor:
    train = pd.DataFrame({name: [float(v) for v in range(100)] for name in factor_order})
    preprocessor = FoldPreprocessor(factor_order=factor_order)
    preprocessor.fit(train, role="TRAIN")
    return preprocessor


def test_cross_sectional_families_dispatch_to_static_batches() -> None:
    for family in ("RESIDUAL_MLP", "SET_ATTENTION"):
        batch = build_inference_batch_for_family(
            family,
            _universe(),
            _factors(),
            factor_members=_MEMBERS,
            signal_date=_SIGNAL,
            preprocessor=_fitted_preprocessor(),
        )
        assert isinstance(batch, FeatureBatch)
        assert not isinstance(batch, TemporalFeatureBatch)


def test_temporal_family_dispatches_to_temporal_batches() -> None:
    batch = build_inference_batch_for_family(
        "TEMPORAL_GRU",
        _universe(),
        _factors(),
        factor_members=_MEMBERS,
        signal_date=_SIGNAL,
        preprocessor=_fitted_preprocessor(),
        history_length=4,
        minimum_valid_history_length=3,
    )
    assert isinstance(batch, TemporalFeatureBatch)
    assert batch.history_mask[0].all()


def test_unknown_family_is_rejected() -> None:
    with pytest.raises(ValueError, match="model family"):
        build_inference_batch_for_family(
            "TEMPORAL_TCN",
            _universe(),
            _factors(),
            factor_members=_MEMBERS,
            signal_date=_SIGNAL,
            preprocessor=_fitted_preprocessor(),
        )


def test_temporal_dispatch_feeds_three_seed_ensemble() -> None:
    batch = build_inference_batch_for_family(
        "TEMPORAL_GRU",
        _universe(),
        _factors(),
        factor_members=_MEMBERS,
        signal_date=_SIGNAL,
        preprocessor=_fitted_preprocessor(),
        history_length=4,
        minimum_valid_history_length=3,
    )
    models = {
        seed: TemporalGRU(
            TemporalGRUConfig(raw_factor_count=1, hidden_dim=8, dropout=0.0)
        ).eval()
        for seed in REQUIRED_SEEDS
    }

    prediction = predict_three_seed_ensemble(batch, models)

    assert set(prediction.seed_percentile_ranks) == set(REQUIRED_SEEDS)
    ensemble = prediction.ensemble
    assert ensemble.shape[0] == 3
    assert {"signal_date", "ts_code", "ensemble_score"}.issubset(ensemble.columns)
    ranks = ensemble["ensemble_score"].to_numpy(dtype=float)
    assert ((ranks >= 0.0) & (ranks <= 1.0)).all()


def test_static_dispatch_feeds_three_seed_ensemble() -> None:
    batch = build_inference_batch_for_family(
        "RESIDUAL_MLP",
        _universe(),
        _factors(),
        factor_members=_MEMBERS,
        signal_date=_SIGNAL,
        preprocessor=_fitted_preprocessor(),
    )
    models = {
        seed: ResidualMLP(
            ResidualMLPConfig(raw_factor_count=1, hidden_dim=8, num_blocks=1, dropout=0.0)
        ).eval()
        for seed in REQUIRED_SEEDS
    }

    prediction = predict_three_seed_ensemble(batch, models)

    assert prediction.ensemble.shape[0] == 3
    ranks = prediction.ensemble["ensemble_score"].to_numpy(dtype=float)
    assert ((ranks >= 0.0) & (ranks <= 1.0)).all()
