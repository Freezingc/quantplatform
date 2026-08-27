"""Induced set-attention cross-sectional model without stock identifiers or positions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn

from research_v2.models.base import FeatureBatch, ModelSignature


@dataclass(frozen=True)
class SetAttentionConfig:
    raw_factor_count: int
    hidden_dim: int = 128
    num_heads: int = 4
    inducing_points: int = 64
    num_blocks: int = 2
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if not 1 <= self.raw_factor_count <= 60:
            raise ValueError("raw_factor_count must be between 1 and 60")
        if (
            self.hidden_dim < 1
            or self.num_heads < 1
            or self.hidden_dim % self.num_heads != 0
            or self.inducing_points < 1
            or self.num_blocks < 1
            or not 0 <= self.dropout < 1
        ):
            raise ValueError("set-attention dimensions or dropout are invalid")


class _InducedSetAttentionBlock(nn.Module):
    """ISAB with O(N×M) cross-sectional attention and no positional signal."""

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        inducing_points: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self._inducing = nn.Parameter(torch.empty(1, inducing_points, hidden_dim))
        nn.init.xavier_uniform_(self._inducing)
        self._to_inducing = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self._from_inducing = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self._inducing_norm = nn.LayerNorm(hidden_dim)
        self._output_norm = nn.LayerNorm(hidden_dim)
        self._feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self._feed_forward_norm = nn.LayerNorm(hidden_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        inducing = self._inducing.expand(values.shape[0], -1, -1)
        induced, _ = self._to_inducing(inducing, values, values, need_weights=False)
        induced = self._inducing_norm(inducing + induced)
        attended, _ = self._from_inducing(values, induced, induced, need_weights=False)
        attended = self._output_norm(values + attended)
        return cast(torch.Tensor, self._feed_forward_norm(attended + self._feed_forward(attended)))


class SetAttention(nn.Module):
    """Apply set attention independently to every date's eligible stock cross-section."""

    model_family = "SET_ATTENTION"

    def __init__(self, config: SetAttentionConfig) -> None:
        super().__init__()
        self._config = config
        self._input_projection = nn.Linear(2 * config.raw_factor_count, config.hidden_dim)
        self._blocks = nn.ModuleList(
            [
                _InducedSetAttentionBlock(
                    config.hidden_dim,
                    config.num_heads,
                    config.inducing_points,
                    config.dropout,
                )
                for _ in range(config.num_blocks)
            ]
        )
        self._summary_query = nn.Parameter(torch.empty(1, 1, config.hidden_dim))
        nn.init.xavier_uniform_(self._summary_query)
        self._summary_attention = nn.MultiheadAttention(
            config.hidden_dim,
            config.num_heads,
            dropout=config.dropout,
            batch_first=True,
        )
        self._head = nn.Linear(config.hidden_dim, 1)

    def forward(self, batch: FeatureBatch) -> torch.Tensor:
        if batch.features.shape[-1] != self._config.raw_factor_count:
            raise ValueError("feature factor count does not match the set-attention configuration")
        inputs = torch.cat(
            (batch.features, batch.feature_missing.to(dtype=batch.features.dtype)),
            dim=-1,
        )
        output = torch.zeros(
            batch.stock_mask.shape,
            dtype=inputs.dtype,
            device=inputs.device,
        )
        for date_index in range(len(batch.signal_dates)):
            positions = torch.nonzero(batch.stock_mask[date_index], as_tuple=False).flatten()
            if not len(positions):
                continue
            values = self._input_projection(inputs[date_index, positions]).unsqueeze(0)
            for block in self._blocks:
                values = block(values)
            summary, _ = self._summary_attention(
                self._summary_query,
                values,
                values,
                need_weights=False,
            )
            scores = self._head(values + summary).squeeze(0).squeeze(-1)
            output[date_index, positions] = scores
        return output

    def artifact_signature(self) -> ModelSignature:
        return ModelSignature(
            model_family=self.model_family,
            raw_factor_count=self._config.raw_factor_count,
            config_hash=(
                f"set-{self._config.raw_factor_count}-{self._config.hidden_dim}-"
                f"{self._config.num_heads}-{self._config.inducing_points}-"
                f"{self._config.num_blocks}-{self._config.dropout}"
            ),
        )
