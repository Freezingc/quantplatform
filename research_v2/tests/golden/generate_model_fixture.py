"""Generate deterministic tensor-only M1/M2/M3 golden fixtures."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path

import torch

from research_v2.models.base import FeatureBatch, TemporalFeatureBatch
from research_v2.models.residual_mlp import ResidualMLP, ResidualMLPConfig
from research_v2.models.set_attention import SetAttention, SetAttentionConfig
from research_v2.models.temporal import TemporalGRU, TemporalGRUConfig


def _fill_parameters(model: torch.nn.Module) -> None:
    with torch.no_grad():
        for parameter_index, parameter in enumerate(model.parameters()):
            values = torch.arange(parameter.numel(), dtype=torch.float32).reshape_as(parameter)
            parameter.copy_(((values.remainder(17) - 8.0) / 100.0) + parameter_index / 1000.0)


def _feature_batch() -> FeatureBatch:
    return FeatureBatch(
        signal_dates=(date(2025, 1, 2),),
        ts_codes=(("a", "b", ""),),
        features=torch.tensor(
            [[[0.10, 0.20], [0.30, 0.40], [0.00, 0.00]]],
            dtype=torch.float32,
        ),
        feature_missing=torch.tensor([[[False, True], [False, False], [False, False]]]),
        stock_mask=torch.tensor([[True, True, False]]),
    )


def _temporal_batch() -> TemporalFeatureBatch:
    return TemporalFeatureBatch(
        signal_dates=(date(2025, 1, 2),),
        ts_codes=(("a", "b", ""),),
        features=torch.tensor(
            [
                [
                    [
                        [0.10, 0.20],
                        [0.20, 0.30],
                        [0.30, 0.40],
                    ],
                    [
                        [0.40, 0.50],
                        [0.00, 0.00],
                        [0.00, 0.00],
                    ],
                    [
                        [0.00, 0.00],
                        [0.00, 0.00],
                        [0.00, 0.00],
                    ],
                ]
            ],
            dtype=torch.float32,
        ),
        feature_missing=torch.tensor(
            [
                [
                    [
                        [False, True],
                        [False, False],
                        [False, False],
                    ],
                    [
                        [False, False],
                        [False, False],
                        [False, False],
                    ],
                    [
                        [False, False],
                        [False, False],
                        [False, False],
                    ],
                ]
            ],
        ),
        stock_mask=torch.tensor([[True, True, False]]),
        history_mask=torch.tensor(
            [[[True, True, True], [True, False, False], [False, False, False]]]
        ),
    )


def main() -> None:
    root = Path(__file__).with_name("data")
    root.mkdir(parents=True, exist_ok=True)
    feature_batch = _feature_batch()
    temporal_batch = _temporal_batch()
    cases: dict[str, tuple[torch.nn.Module, object, str, dict[str, object]]] = {
        "M1": (
            ResidualMLP(
                ResidualMLPConfig(raw_factor_count=2, hidden_dim=4, num_blocks=1, dropout=0.0)
            ),
            feature_batch,
            "m1_state_dict",
            {"raw_factor_count": 2, "hidden_dim": 4, "num_blocks": 1, "dropout": 0.0},
        ),
        "M2_GRU": (
            TemporalGRU(TemporalGRUConfig(raw_factor_count=2, hidden_dim=4, dropout=0.0)),
            temporal_batch,
            "m2_gru_state_dict",
            {"raw_factor_count": 2, "hidden_dim": 4, "dropout": 0.0},
        ),
        "M3": (
            SetAttention(
                SetAttentionConfig(
                    raw_factor_count=2,
                    hidden_dim=4,
                    num_heads=2,
                    inducing_points=2,
                    num_blocks=1,
                    dropout=0.0,
                )
            ),
            feature_batch,
            "m3_state_dict",
            {
                "raw_factor_count": 2,
                "hidden_dim": 4,
                "num_heads": 2,
                "inducing_points": 2,
                "num_blocks": 1,
                "dropout": 0.0,
            },
        ),
    }
    fixture: dict[str, object] = {
        "cross_section_features": feature_batch.features,
        "cross_section_missing": feature_batch.feature_missing,
        "cross_section_stock_mask": feature_batch.stock_mask,
        "temporal_features": temporal_batch.features,
        "temporal_missing": temporal_batch.feature_missing,
        "temporal_stock_mask": temporal_batch.stock_mask,
        "temporal_history_mask": temporal_batch.history_mask,
    }
    expected_cases: dict[str, object] = {}
    for name, (model, batch, state_key, config) in cases.items():
        _fill_parameters(model)
        model.eval()
        with torch.no_grad():
            scores = model(batch)  # type: ignore[arg-type, operator]
        fixture[state_key] = model.state_dict()
        expected_cases[name] = {
            "config": config,
            "signature": asdict(model.artifact_signature()),  # type: ignore[attr-defined]
            "scores": scores.tolist(),
        }
    torch.save(fixture, root / "model_batch.pt")
    (root / "expected_model_outputs.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "feature_order": ["factor_a", "factor_b"],
                "cases": expected_cases,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
