"""Versioned universe snapshot construction."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Any, cast

import pandas as pd

from research_v2.universe.rules import UniverseRules, build_universe


@dataclass(frozen=True)
class UniverseDefinition:
    """An immutable identifier and rule set for one universe version."""

    universe_id: str
    version: int
    rules: UniverseRules

    def __post_init__(self) -> None:
        if not self.universe_id.strip():
            raise ValueError("universe_id cannot be empty")
        if self.version < 1:
            raise ValueError("universe version must be positive")


def _signal_date(candidates: pd.DataFrame) -> date:
    if "trade_date" not in candidates.columns:
        raise ValueError("universe candidates are missing trade_date")
    values = candidates["trade_date"].dropna().unique()
    if len(values) != 1 or candidates["trade_date"].isna().any():
        raise ValueError("universe snapshot requires exactly one signal date")
    value = cast(Any, values[0])
    return value if isinstance(value, date) else pd.Timestamp(value).date()


def build_universe_snapshot(
    candidates: pd.DataFrame,
    definition: UniverseDefinition,
) -> pd.DataFrame:
    """Build one reproducible cross-sectional universe snapshot for signal date T."""
    if candidates.empty:
        raise ValueError("universe snapshot cannot be built from empty candidates")
    signal_date = _signal_date(candidates)
    resolved = build_universe(candidates, definition.rules)
    resolved.insert(0, "signal_date", signal_date)
    resolved.insert(1, "universe_id", definition.universe_id)
    resolved.insert(2, "universe_version", definition.version)
    preferred_columns = [
        "signal_date",
        "universe_id",
        "universe_version",
        "ts_code",
        "is_member",
        "exclusion_reason",
        "reason_bitmap",
        "adv20_rank",
        "membership_rank",
    ]
    return resolved.loc[
        :, [column for column in preferred_columns if column in resolved.columns]
    ]
