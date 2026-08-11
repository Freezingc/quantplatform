"""M1 residual MLP score model for a date-isolated factor cross-section."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import cast

import torch
from torch import nn

from research_v2.models.base import FeatureBatch, ModelSignature
from research_v2.models.components import ResidualFeedForwardBlock


@dataclass(frozen=True)
class ResidualMLPConfig:
    raw_factor_count: int
    hidden_dim: int = 128
    num_blocks: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if not 1 <= self.raw_factor_count <= 60:
            raise ValueError("raw_factor_count must be between 1 and 60")
        if self.hidden_dim < 1 or self.num_blocks < 1 or not 0 <= self.dropout < 1:
            raise ValueError("residual MLP dimensions or dropout are invalid")


class ResidualMLP(nn.Module):
    """Score every valid stock independently after appending raw missingness indicators."""

    model_family = "RESIDUAL_MLP"

    def __init__(self, config: ResidualMLPConfig) -> None:
        super().__init__()
        self._config = config
        self._input_projection = nn.Linear(2 * config.raw_factor_count, config.hidden_dim)
        self._blocks = nn.ModuleList(
            [
                ResidualFeedForwardBlock(config.hidden_dim, config.dropout)
                for _ in range(config.num_blocks)
            ]
        )
        self._head = nn.Linear(config.hidden_dim, 1)

    def forward(self, batch: FeatureBatch) -> torch.Tensor:
        if batch.features.shape[2] != self._config.raw_factor_count:
            raise ValueError("feature factor count does not match the residual MLP configuration")
        values = torch.cat(
            (batch.features, batch.feature_missing.to(dtype=batch.features.dtype)),
            dim=-1,
        )
        hidden = self._input_projection(values)
        for block in self._blocks:
            hidden = block(hidden)
        scores = self._head(hidden).squeeze(-1)
        return cast(torch.Tensor, scores.masked_fill(~batch.stock_mask, 0.0))

    def artifact_signature(self) -> ModelSignature:
        config_hash = hashlib.sha256(
            json.dumps(asdict(self._config), separators=(",", ":"), sort_keys=True).encode()
        ).hexdigest()
        return ModelSignature(
            model_family=self.model_family,
            raw_factor_count=self._config.raw_factor_count,
            config_hash=config_hash,
        )
