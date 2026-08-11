"""Causal GRU temporal candidate over each stock's own factor history."""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import torch
from torch import nn
from torch.nn import functional as functional
from torch.nn.utils.rnn import pack_padded_sequence

from research_v2.models.base import ModelSignature, TemporalFeatureBatch


@dataclass(frozen=True)
class TemporalGRUConfig:
    raw_factor_count: int
    hidden_dim: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if not 1 <= self.raw_factor_count <= 60:
            raise ValueError("raw_factor_count must be between 1 and 60")
        if self.hidden_dim < 1 or not 0 <= self.dropout < 1:
            raise ValueError("temporal GRU dimensions or dropout are invalid")


class TemporalGRU(nn.Module):
    """Encode only left-aligned causal history and return one score per valid stock."""

    model_family = "TEMPORAL_GRU"

    def __init__(self, config: TemporalGRUConfig) -> None:
        super().__init__()
        self._config = config
        self._gru = nn.GRU(2 * config.raw_factor_count, config.hidden_dim, batch_first=True)
        self._dropout = nn.Dropout(config.dropout)
        self._head = nn.Linear(config.hidden_dim, 1)

    def forward(self, batch: TemporalFeatureBatch) -> torch.Tensor:
        if batch.features.shape[-1] != self._config.raw_factor_count:
            raise ValueError("temporal factor count does not match the GRU configuration")
        date_count, stock_count, history_length, _ = batch.features.shape
        input_values = torch.cat(
            (batch.features, batch.feature_missing.to(dtype=batch.features.dtype)),
            dim=-1,
        )
        history = batch.history_mask.reshape(date_count * stock_count, history_length)
        sequences = input_values.reshape(date_count * stock_count, history_length, -1)
        sequences = sequences.masked_fill(~history.unsqueeze(-1), 0.0)
        valid_positions = torch.nonzero(batch.stock_mask.reshape(-1), as_tuple=False).flatten()
        valid_sequences = sequences[valid_positions]
        lengths = history[valid_positions].sum(dim=1).to(dtype=torch.int64).cpu()
        packed = pack_padded_sequence(
            valid_sequences,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, hidden = self._gru(packed)
        valid_scores = self._head(self._dropout(hidden[-1])).squeeze(-1)
        scores = torch.zeros(
            date_count * stock_count,
            dtype=valid_scores.dtype,
            device=valid_scores.device,
        )
        scores[valid_positions] = valid_scores
        return scores.reshape(date_count, stock_count)

    def artifact_signature(self) -> ModelSignature:
        return ModelSignature(
            model_family=self.model_family,
            raw_factor_count=self._config.raw_factor_count,
            config_hash=f"gru-{self._config.raw_factor_count}-{self._config.hidden_dim}-{self._config.dropout}",
        )


@dataclass(frozen=True)
class TemporalTCNConfig:
    raw_factor_count: int
    hidden_dim: int = 128
    dropout: float = 0.1

    def __post_init__(self) -> None:
        if not 1 <= self.raw_factor_count <= 60:
            raise ValueError("raw_factor_count must be between 1 and 60")
        if self.hidden_dim < 1 or not 0 <= self.dropout < 1:
            raise ValueError("temporal TCN dimensions or dropout are invalid")


class _CausalDepthwiseBlock(nn.Module):
    """A left-padded depthwise-separable residual block with no temporal normalization."""

    def __init__(self, hidden_dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self._left_padding = 2 * dilation
        self._depthwise = nn.Conv1d(
            hidden_dim,
            hidden_dim,
            kernel_size=3,
            dilation=dilation,
            groups=hidden_dim,
        )
        self._pointwise = nn.Conv1d(hidden_dim, hidden_dim, kernel_size=1)
        self._dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        encoded = functional.pad(values, (self._left_padding, 0))
        encoded = self._depthwise(encoded)
        encoded = functional.gelu(encoded)
        encoded = self._pointwise(encoded)
        return cast(torch.Tensor, values + self._dropout(encoded))


class TemporalTCN(nn.Module):
    """Causal dilated TCN with receptive field derived entirely from past time steps."""

    model_family = "TEMPORAL_TCN"

    def __init__(self, config: TemporalTCNConfig) -> None:
        super().__init__()
        self._config = config
        self._input_projection = nn.Conv1d(2 * config.raw_factor_count, config.hidden_dim, 1)
        self._blocks = nn.ModuleList(
            [
                _CausalDepthwiseBlock(config.hidden_dim, dilation, config.dropout)
                for dilation in (1, 2, 4)
            ]
        )
        self._head = nn.Linear(config.hidden_dim, 1)

    def forward(self, batch: TemporalFeatureBatch) -> torch.Tensor:
        if batch.features.shape[-1] != self._config.raw_factor_count:
            raise ValueError("temporal factor count does not match the TCN configuration")
        date_count, stock_count, history_length, _ = batch.features.shape
        input_values = torch.cat(
            (batch.features, batch.feature_missing.to(dtype=batch.features.dtype)),
            dim=-1,
        )
        history = batch.history_mask.reshape(date_count * stock_count, history_length)
        sequences = input_values.reshape(date_count * stock_count, history_length, -1)
        sequences = sequences.masked_fill(~history.unsqueeze(-1), 0.0)
        valid_positions = torch.nonzero(batch.stock_mask.reshape(-1), as_tuple=False).flatten()
        valid_sequences = sequences[valid_positions].transpose(1, 2)
        encoded = self._input_projection(valid_sequences)
        for block in self._blocks:
            encoded = block(encoded)
        lengths = history[valid_positions].sum(dim=1).to(dtype=torch.int64) - 1
        rows = torch.arange(len(valid_positions), device=encoded.device)
        final_state = encoded[rows, :, lengths]
        valid_scores = self._head(final_state).squeeze(-1)
        scores = torch.zeros(
            date_count * stock_count,
            dtype=valid_scores.dtype,
            device=valid_scores.device,
        )
        scores[valid_positions] = valid_scores
        return scores.reshape(date_count, stock_count)

    def artifact_signature(self) -> ModelSignature:
        return ModelSignature(
            model_family=self.model_family,
            raw_factor_count=self._config.raw_factor_count,
            config_hash=f"tcn-{self._config.raw_factor_count}-{self._config.hidden_dim}-{self._config.dropout}",
        )
