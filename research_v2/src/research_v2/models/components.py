"""Reusable neural components that preserve per-stock, per-date isolation."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn


class ResidualFeedForwardBlock(nn.Module):
    """LayerNorm residual block without batch statistics or cross-stock interaction."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self._network = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return cast(torch.Tensor, values + self._network(values))
