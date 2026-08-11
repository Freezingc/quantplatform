"""Date-equal sampling schedules for cross-sectional training."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date


@dataclass(frozen=True)
class DateBalancedSampler:
    """Return a deterministic full-date permutation, never stock-level chunks."""

    dates: tuple[date, ...]
    seed: int

    def __post_init__(self) -> None:
        if not self.dates:
            raise ValueError("sampler requires at least one training date")
        if len(set(self.dates)) != len(self.dates):
            raise ValueError("sampler dates must be unique")

    def for_epoch(self, epoch: int) -> tuple[date, ...]:
        """Return each date exactly once; epoch changes order without changing inclusion."""
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        ordered = sorted(
            self.dates,
            key=lambda signal_date: hashlib.sha256(
                f"{self.seed}|{signal_date.isoformat()}".encode()
            ).hexdigest(),
        )
        offset = epoch % len(ordered)
        return tuple(ordered[offset:] + ordered[:offset])
