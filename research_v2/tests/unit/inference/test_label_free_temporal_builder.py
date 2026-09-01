from __future__ import annotations

from datetime import date

import pandas as pd
import pytest
import torch

from research_v2.datasets.builder import FactorInput
from research_v2.datasets.preprocessing import FoldPreprocessor
from research_v2.inference.builder import build_label_free_temporal_inference_batch
from research_v2.models.base import predictions_to_frame
from research_v2.models.temporal import TemporalGRU, TemporalGRUConfig

_SIGNAL = date(2024, 1, 5)
_DAYS = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4), _SIGNAL, date(2024, 1, 8)]
_MEMBERS = (FactorInput(factor_id="alpha", factor_version=1, feature_name="f01"),)


def _universe(codes=("a", "b")) -> pd.DataFrame:
    return pd.DataFrame([{"signal_date": _SIGNAL, "ts_code": code} for code in codes])


def _factor_row(day, code, value, *, available=None) -> dict:
    return {
        "signal_date": day,
        "ts_code": code,
        "factor_id": "alpha",
        "factor_version": 1,
        "value": value,
        "available_date": available or day,
    }


def _fitted_preprocessor(factor_order: tuple[str, ...] = ("f01",)) -> FoldPreprocessor:
    train = pd.DataFrame(
        {name: [float(value) for value in range(100)] for name in factor_order}
    )
    preprocessor = FoldPreprocessor(factor_order=factor_order)
    preprocessor.fit(train, role="TRAIN")
    return preprocessor


def test_temporal_inference_assembles_causal_left_aligned_windows() -> None:
    factors = pd.DataFrame(
        [
            _factor_row(_DAYS[1], "a", 1.0),
            _factor_row(_DAYS[2], "a", 2.0),
            _factor_row(_DAYS[3], "a", 3.0),
            _factor_row(_DAYS[3], "b", 9.0),
        ]
    )
    batch = build_label_free_temporal_inference_batch(
        _universe(),
        factors,
        factor_members=_MEMBERS,
        signal_date=_SIGNAL,
        preprocessor=_fitted_preprocessor(),
        history_length=3,
        minimum_valid_history_length=2,
    )

    assert batch.signal_dates == (_SIGNAL,)
    assert batch.ts_codes == (("a",),)
    assert batch.features.shape == (1, 1, 3, 1)
    assert batch.history_mask[0, 0].tolist() == [True, True, True]
    assert batch.stock_mask[0].tolist() == [True]
    # Transformed values preserve the frozen-train ordering of the raw window.
    window = batch.features[0, 0, :, 0].tolist()
    assert window == sorted(window)
    assert batch.features[0, 0, 2, 0].item() == pytest.approx(
        batch.features[0, 0, 2, 0].item()
    )


def test_temporal_inference_ignores_rows_after_the_signal_date() -> None:
    clean = pd.DataFrame(
        [
            _factor_row(_DAYS[2], "a", 2.0),
            _factor_row(_DAYS[3], "a", 3.0),
        ]
    )
    leaked = pd.concat(
        [clean, pd.DataFrame([_factor_row(_DAYS[4], "a", 99.0)])], ignore_index=True
    )
    preprocessor = _fitted_preprocessor()
    base = build_label_free_temporal_inference_batch(
        _universe(codes=("a",)),
        clean,
        factor_members=_MEMBERS,
        signal_date=_SIGNAL,
        preprocessor=preprocessor,
        history_length=3,
        minimum_valid_history_length=2,
    )
    perturbed = build_label_free_temporal_inference_batch(
        _universe(codes=("a",)),
        leaked,
        factor_members=_MEMBERS,
        signal_date=_SIGNAL,
        preprocessor=preprocessor,
        history_length=3,
        minimum_valid_history_length=2,
    )

    assert torch.equal(base.features, perturbed.features)
    assert torch.equal(base.history_mask, perturbed.history_mask)


def test_temporal_inference_excludes_securities_without_enough_history() -> None:
    factors = pd.DataFrame([_factor_row(_DAYS[3], "b", 9.0)])
    with pytest.raises(ValueError, match="sufficient history"):
        build_label_free_temporal_inference_batch(
            _universe(codes=("b",)),
            factors,
            factor_members=_MEMBERS,
            signal_date=_SIGNAL,
            preprocessor=_fitted_preprocessor(),
            history_length=3,
            minimum_valid_history_length=2,
        )


def test_temporal_inference_rejects_history_rows_not_yet_available() -> None:
    factors = pd.DataFrame(
        [
            _factor_row(_DAYS[2], "a", 2.0),
            _factor_row(_DAYS[3], "a", 3.0, available=_DAYS[4]),
        ]
    )
    with pytest.raises(ValueError, match="not available"):
        build_label_free_temporal_inference_batch(
            _universe(codes=("a",)),
            factors,
            factor_members=_MEMBERS,
            signal_date=_SIGNAL,
            preprocessor=_fitted_preprocessor(),
            history_length=3,
            minimum_valid_history_length=2,
        )


def test_temporal_inference_rejects_invalid_configuration() -> None:
    factors = pd.DataFrame([_factor_row(_DAYS[3], "a", 1.0)])
    with pytest.raises(ValueError):
        build_label_free_temporal_inference_batch(
            _universe(codes=("a",)),
            factors,
            factor_members=_MEMBERS,
            signal_date=_SIGNAL,
            preprocessor=_fitted_preprocessor(),
            history_length=3,
            minimum_valid_history_length=4,
        )


def test_temporal_inference_batch_scores_through_gru() -> None:
    members = (
        FactorInput(factor_id="alpha", factor_version=1, feature_name="f01"),
        FactorInput(factor_id="beta", factor_version=1, feature_name="f02"),
        FactorInput(factor_id="gamma", factor_version=1, feature_name="f03"),
    )
    rows = []
    for day in _DAYS[:4]:
        for code in ("a", "b"):
            for factor_id in ("alpha", "beta", "gamma"):
                row = _factor_row(day, code, 1.0)
                row["factor_id"] = factor_id
                rows.append(row)
    factors = pd.DataFrame(rows)
    batch = build_label_free_temporal_inference_batch(
        _universe(),
        factors,
        factor_members=members,
        signal_date=_SIGNAL,
        preprocessor=_fitted_preprocessor(("f01", "f02", "f03")),
        history_length=4,
        minimum_valid_history_length=3,
    )

    model = TemporalGRU(
        TemporalGRUConfig(raw_factor_count=3, hidden_dim=8, dropout=0.0)
    ).eval()
    with torch.no_grad():
        scores = model(batch)
    frame = predictions_to_frame(batch, scores)

    assert frame.shape[0] == 2
    assert {"signal_date", "ts_code", "percentile_rank"}.issubset(frame.columns)
    ranks = frame["percentile_rank"].to_numpy(dtype=float)
    assert ((ranks >= 0.0) & (ranks <= 1.0)).all()
