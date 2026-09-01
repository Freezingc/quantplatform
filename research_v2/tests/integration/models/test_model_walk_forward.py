"""Synthetic smoke end-to-end model orchestration evidence (E8.5).

Verifies train -> checkpoint -> immutable bundle -> verified inference load ->
frozen TEST evaluation -> G1-G6 gate report for M1/M2/M3 smoke configurations.
The test only proves orchestration; synthetic performance is never model
passing evidence.
"""

from __future__ import annotations

import os
from dataclasses import asdict
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from research_v2.evaluation.model_evaluator import FrozenOosEvaluation, evaluate_frozen_oos
from research_v2.evaluation.rank_ic import compute_daily_rank_ic, summarize_rank_ic
from research_v2.inference.artifact_loader import load_inference_artifact
from research_v2.models.base import FeatureBatch, TemporalFeatureBatch
from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
from research_v2.models.set_attention import SetAttention, SetAttentionConfig
from research_v2.models.temporal import TemporalGRU, TemporalGRUConfig
from research_v2.training.artifacts import CheckpointStore
from research_v2.training.batches import TemporalTrainingBatch, TrainingBatch
from research_v2.training.ensemble import REQUIRED_SEEDS, build_seed_ensemble
from research_v2.training.model_bundle import ModelArtifactBundleStore
from research_v2.training.torch_loop import TorchDateTrainingLoop

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(
        os.environ.get("QRV2_INTEGRATION") != "1",
        reason="set QRV2_INTEGRATION=1 to run model walk-forward integration tests",
    ),
]

_STOCK_COUNT = 40
_TRAIN_DAYS = [date(2024, 1, 2), date(2024, 1, 3), date(2024, 1, 4)]
_VALIDATION_DAY = date(2024, 1, 5)
_TEST_DAYS = [
    date(2024, 1, 8),
    date(2024, 1, 9),
    date(2024, 1, 10),
    date(2024, 1, 11),
    date(2024, 1, 12),
    date(2024, 1, 15),
    date(2024, 1, 16),
    date(2024, 1, 17),
]


def _codes() -> tuple[str, ...]:
    return tuple(f"s{index:02d}" for index in range(_STOCK_COUNT))


def _alpha(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal(_STOCK_COUNT)


def _percentile_rank(scores: torch.Tensor) -> torch.Tensor:
    ranks = scores.argsort().argsort().to(dtype=torch.float)
    return ranks / (_STOCK_COUNT - 1)


def _cross_section_batches(factor_count: int):
    alpha = _alpha(21)
    generator = np.random.default_rng(22)
    train_batches: dict[date, TrainingBatch] = {}
    for day in _TRAIN_DAYS:
        noise = generator.standard_normal((_STOCK_COUNT, factor_count))
        features = torch.from_numpy(noise).to(dtype=torch.float)
        features[:, 0] = torch.from_numpy(alpha).to(dtype=torch.float)
        target_raw = torch.from_numpy(alpha + generator.standard_normal(_STOCK_COUNT) * 0.5)
        train_batches[day] = TrainingBatch(
            inputs=FeatureBatch(
                signal_dates=(day,),
                ts_codes=(_codes(),),
                features=features.unsqueeze(0),
                feature_missing=torch.zeros((1, _STOCK_COUNT, factor_count), dtype=torch.bool),
                stock_mask=torch.ones((1, _STOCK_COUNT), dtype=torch.bool),
            ),
            target_raw=target_raw.unsqueeze(0),
            target_rank=_percentile_rank(target_raw).unsqueeze(0),
        )
    validation_features = torch.zeros((_STOCK_COUNT, factor_count))
    validation_features[:, 0] = torch.from_numpy(alpha).to(dtype=torch.float)
    validation_labels = alpha + np.random.default_rng(23).standard_normal(_STOCK_COUNT) * 0.5
    validation_batch = FeatureBatch(
        signal_dates=(_VALIDATION_DAY,),
        ts_codes=(_codes(),),
        features=validation_features.unsqueeze(0),
        feature_missing=torch.zeros((1, _STOCK_COUNT, factor_count), dtype=torch.bool),
        stock_mask=torch.ones((1, _STOCK_COUNT), dtype=torch.bool),
    )
    test_frames: list[pd.DataFrame] = []
    label_rows: list[dict[str, object]] = []
    test_generator = np.random.default_rng(24)
    for day in _TEST_DAYS:
        returns = alpha + test_generator.standard_normal(_STOCK_COUNT) * 0.5
        for index, code in enumerate(_codes()):
            test_frames.append(
                pd.DataFrame(
                    {
                        "signal_date": [day],
                        "ts_code": [code],
                        "score": [float(alpha[index])],
                    }
                )
            )
            label_rows.append(
                {
                    "signal_date": day,
                    "ts_code": code,
                    "raw_return_5d": float(returns[index]),
                    "label_valid": True,
                }
            )
    baseline_predictions = pd.concat(test_frames, ignore_index=True)
    labels = pd.DataFrame(label_rows)
    test_batch = FeatureBatch(
        signal_dates=tuple(_TEST_DAYS),
        ts_codes=(_codes(),) * len(_TEST_DAYS),
        features=torch.from_numpy(np.repeat(alpha, len(_TEST_DAYS), axis=0))
        .reshape(len(_TEST_DAYS), _STOCK_COUNT, 1)
        .expand(len(_TEST_DAYS), _STOCK_COUNT, factor_count)
        .clone()
        .to(dtype=torch.float),
        feature_missing=torch.zeros(
            (len(_TEST_DAYS), _STOCK_COUNT, factor_count), dtype=torch.bool
        ),
        stock_mask=torch.ones((len(_TEST_DAYS), _STOCK_COUNT), dtype=torch.bool),
    )
    return (
        train_batches,
        validation_batch,
        validation_labels,
        test_batch,
        baseline_predictions,
        labels,
    )


def _temporal_batches(factor_count: int):
    alpha = _alpha(31)
    history_length = 20
    generator = np.random.default_rng(32)
    train_batches: dict[date, TemporalTrainingBatch] = {}
    for day in _TRAIN_DAYS:
        features = torch.from_numpy(
            generator.standard_normal((_STOCK_COUNT, history_length, factor_count)) * 0.1
        ).to(dtype=torch.float)
        features[:, :, 0] = (
            torch.from_numpy(alpha).unsqueeze(1).expand(_STOCK_COUNT, history_length)
        )
        target_raw = torch.from_numpy(alpha + generator.standard_normal(_STOCK_COUNT) * 0.5)
        train_batches[day] = TemporalTrainingBatch(
            inputs=TemporalFeatureBatch(
                signal_dates=(day,),
                ts_codes=(_codes(),),
                features=features.unsqueeze(0),
                feature_missing=torch.zeros(
                    (1, _STOCK_COUNT, history_length, factor_count), dtype=torch.bool
                ),
                stock_mask=torch.ones((1, _STOCK_COUNT), dtype=torch.bool),
                history_mask=torch.ones((1, _STOCK_COUNT, history_length), dtype=torch.bool),
            ),
            target_raw=target_raw.unsqueeze(0),
            target_rank=_percentile_rank(target_raw).unsqueeze(0),
        )
    validation_features = torch.zeros((_STOCK_COUNT, history_length, factor_count))
    validation_features[:, :, 0] = torch.from_numpy(alpha).unsqueeze(1)
    validation_labels = alpha + np.random.default_rng(33).standard_normal(_STOCK_COUNT) * 0.5
    validation_batch = TemporalFeatureBatch(
        signal_dates=(_VALIDATION_DAY,),
        ts_codes=(_codes(),),
        features=validation_features.unsqueeze(0),
        feature_missing=torch.zeros(
            (1, _STOCK_COUNT, history_length, factor_count), dtype=torch.bool
        ),
        stock_mask=torch.ones((1, _STOCK_COUNT), dtype=torch.bool),
        history_mask=torch.ones((1, _STOCK_COUNT, history_length), dtype=torch.bool),
    )
    test_features = torch.zeros((len(_TEST_DAYS), _STOCK_COUNT, history_length, factor_count))
    test_features[:, :, :, 0] = torch.from_numpy(alpha).reshape(1, -1, 1).expand(
        len(_TEST_DAYS), _STOCK_COUNT, history_length
    )
    test_batch = TemporalFeatureBatch(
        signal_dates=tuple(_TEST_DAYS),
        ts_codes=(_codes(),) * len(_TEST_DAYS),
        features=test_features,
        feature_missing=torch.zeros_like(test_features, dtype=torch.bool),
        stock_mask=torch.ones((len(_TEST_DAYS), _STOCK_COUNT), dtype=torch.bool),
        history_mask=torch.ones((len(_TEST_DAYS), _STOCK_COUNT, history_length), dtype=torch.bool),
    )
    baseline_rows: list[dict[str, object]] = []
    label_rows: list[dict[str, object]] = []
    test_generator = np.random.default_rng(34)
    for day in _TEST_DAYS:
        returns = alpha + test_generator.standard_normal(_STOCK_COUNT) * 0.5
        for index, code in enumerate(_codes()):
            baseline_rows.append(
                {"signal_date": day, "ts_code": code, "score": float(alpha[index])}
            )
            label_rows.append(
                {
                    "signal_date": day,
                    "ts_code": code,
                    "raw_return_5d": float(returns[index]),
                    "label_valid": True,
                }
            )
    return (
        train_batches,
        validation_batch,
        validation_labels,
        test_batch,
        pd.DataFrame(baseline_rows),
        pd.DataFrame(label_rows),
    )


def _seed_predictions(model, batch) -> pd.DataFrame:
    model.eval()
    with torch.no_grad():
        scores = model(batch)
    rows: list[dict[str, object]] = []
    for date_index, signal_date in enumerate(batch.signal_dates):
        ranks = _percentile_rank(scores[date_index])
        for stock_index, code in enumerate(batch.ts_codes[date_index]):
            rows.append(
                {
                    "signal_date": signal_date,
                    "ts_code": code,
                    "percentile_rank": float(ranks[stock_index]),
                }
            )
    return pd.DataFrame(rows)


def _validation_rank_ic(predictions: pd.DataFrame, labels: pd.DataFrame) -> float:
    daily = compute_daily_rank_ic(
        predictions.rename(columns={"percentile_rank": "score"}),
        labels,
        minimum_cross_section=30,
    )
    summary = summarize_rank_ic(daily)
    return summary.mean_rank_ic


def _run_smoke_pipeline(family: str, tmp_path: Path) -> FrozenOosEvaluation:
    if family == "RESIDUAL_MLP":
        factor_count = 15
        config = ResidualMLPConfig(
            raw_factor_count=factor_count, hidden_dim=16, num_blocks=1, dropout=0.0
        )
        config_payload = {
            "raw_factor_count": factor_count,
            "hidden_dim": 16,
            "num_blocks": 1,
            "dropout": 0.0,
        }
        factory = lambda: ResidualMLP(config)  # noqa: E731
        batches = _cross_section_batches(factor_count)
    elif family == "SET_ATTENTION":
        factor_count = 15
        config_sa = SetAttentionConfig(
            raw_factor_count=factor_count,
            hidden_dim=16,
            num_heads=2,
            inducing_points=4,
            num_blocks=1,
            dropout=0.0,
        )
        config_payload = {
            "raw_factor_count": factor_count,
            "hidden_dim": 16,
            "num_heads": 2,
            "inducing_points": 4,
            "num_blocks": 1,
            "dropout": 0.0,
        }
        factory = lambda: SetAttention(config_sa)  # noqa: E731
        batches = _cross_section_batches(factor_count)
    else:
        factor_count = 6
        config_gru = TemporalGRUConfig(raw_factor_count=factor_count, hidden_dim=16, dropout=0.0)
        config_payload = {"raw_factor_count": factor_count, "hidden_dim": 16, "dropout": 0.0}
        factory = lambda: TemporalGRU(config_gru)  # noqa: E731
        batches = _temporal_batches(factor_count)

    (
        train_batches,
        validation_batch,
        validation_labels,
        test_batch,
        baseline_predictions,
        labels,
    ) = batches
    validation_label_frame = pd.DataFrame(
        {
            "signal_date": [_VALIDATION_DAY] * _STOCK_COUNT,
            "ts_code": _codes(),
            "raw_return_5d": validation_labels,
            "label_valid": True,
        }
    )

    checkpoint_store = CheckpointStore(tmp_path / "checkpoints")
    seed_checkpoints: dict[int, Path] = {}
    seed_validation_predictions: dict[int, pd.DataFrame] = {}
    seed_rank_ic: dict[int, float] = {}
    prototype = factory()
    for seed in REQUIRED_SEEDS:
        torch.manual_seed(seed)
        model = factory()
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
        loop = TorchDateTrainingLoop(
            model, optimizer, train_batches, seed=seed, gradient_clip_norm=1.0
        )
        for epoch in range(8):
            loop.train_epoch(epoch)
        artifact = checkpoint_store.save(
            f"smoke-{family.lower()}-{seed}",
            model,
            optimizer,
            metadata={"family": family, "seed": seed},
        )
        seed_checkpoints[seed] = artifact.path
        seed_validation_predictions[seed] = _seed_predictions(model, validation_batch)
        seed_rank_ic[seed] = _validation_rank_ic(
            seed_validation_predictions[seed], validation_label_frame
        )

    ensemble = build_seed_ensemble(seed_validation_predictions, validation_rank_ic=seed_rank_ic)
    validation_predictions = ensemble.predictions.rename(
        columns={"ensemble_score": "percentile_rank"}
    )

    bundle_store = ModelArtifactBundleStore(tmp_path / "models")
    model_id = f"smoke-{family.lower()}"
    bundle_store.write(
        model_id,
        model_card={
            "family": family,
            "dataset_id": "synthetic-train",
            "model_status": "WALK_FORWARD_PASSED",
            "model_code_version": {
                "RESIDUAL_MLP": "residual-mlp-1",
                "TEMPORAL_GRU": "temporal-gru-1",
                "SET_ATTENTION": "set-attention-1",
            }[family],
        },
        config=config_payload,
        signature=asdict(prototype.artifact_signature()),
        feature_order=tuple(f"f{index:02d}" for index in range(factor_count)),
        preprocessing={"state_hash": "sha256:" + "0" * 64},
        seed_checkpoints=seed_checkpoints,
        validation_predictions=validation_predictions,
    )
    with pytest.raises(FileExistsError):
        bundle_store.write(
            model_id,
            model_card={
                "family": family,
                "dataset_id": "synthetic-train",
                "model_status": "WALK_FORWARD_PASSED",
            },
            config=config_payload,
            signature=asdict(prototype.artifact_signature()),
            feature_order=tuple(f"f{index:02d}" for index in range(factor_count)),
            preprocessing={"state_hash": "sha256:" + "0" * 64},
            seed_checkpoints=seed_checkpoints,
            validation_predictions=validation_predictions,
        )

    loaded = load_inference_artifact(
        bundle_store,
        model_id,
        expected_dataset_id="synthetic-train",
        expected_feature_order=tuple(f"f{index:02d}" for index in range(factor_count)),
        expected_preprocessor_hash="sha256:" + "0" * 64,
        device=torch.device("cpu"),
    )
    assert len(loaded.models) == len(REQUIRED_SEEDS)

    rows: list[dict[str, object]] = []
    for date_index, signal_date in enumerate(test_batch.signal_dates):
        accumulated = torch.zeros(_STOCK_COUNT)
        for seed_model in loaded.models:
            single = _select_date(test_batch, date_index)
            with torch.no_grad():
                scores = seed_model(single)[0]
            accumulated = accumulated + _percentile_rank(scores)
        ranks = accumulated / len(REQUIRED_SEEDS)
        for stock_index, code in enumerate(_codes()):
            rows.append(
                {"signal_date": signal_date, "ts_code": code, "score": float(ranks[stock_index])}
            )
    candidate_predictions = pd.DataFrame(rows)

    evaluation = evaluate_frozen_oos(
        candidate_predictions,
        labels,
        baseline_predictions=baseline_predictions,
        role="TEST",
        minimum_cross_section=30,
    )
    assert set(evaluation.gate_report.results) == {"G1", "G2", "G3", "G4", "G5", "G6"}
    assert evaluation.summary.valid_day_count == len(_TEST_DAYS)
    assert evaluation.baseline_mean_rank_ic is not None
    assert np.isfinite(evaluation.baseline_mean_rank_ic)
    repeated = evaluate_frozen_oos(
        candidate_predictions,
        labels,
        baseline_predictions=baseline_predictions,
        role="TEST",
        minimum_cross_section=30,
    )
    assert evaluation.gate_report == repeated.gate_report
    assert evaluation.summary == repeated.summary
    return evaluation


def _select_date(batch, date_index: int):
    if isinstance(batch, TemporalFeatureBatch):
        return TemporalFeatureBatch(
            signal_dates=(batch.signal_dates[date_index],),
            ts_codes=(batch.ts_codes[date_index],),
            features=batch.features[date_index : date_index + 1],
            feature_missing=batch.feature_missing[date_index : date_index + 1],
            stock_mask=batch.stock_mask[date_index : date_index + 1],
            history_mask=batch.history_mask[date_index : date_index + 1],
        )
    return FeatureBatch(
        signal_dates=(batch.signal_dates[date_index],),
        ts_codes=(batch.ts_codes[date_index],),
        features=batch.features[date_index : date_index + 1],
        feature_missing=batch.feature_missing[date_index : date_index + 1],
        stock_mask=batch.stock_mask[date_index : date_index + 1],
    )


def test_m1_residual_mlp_smoke_pipeline(tmp_path) -> None:
    _run_smoke_pipeline("RESIDUAL_MLP", tmp_path)


def test_m2_temporal_gru_smoke_pipeline(tmp_path) -> None:
    _run_smoke_pipeline("TEMPORAL_GRU", tmp_path)


def test_m3_set_attention_smoke_pipeline(tmp_path) -> None:
    _run_smoke_pipeline("SET_ATTENTION", tmp_path)
