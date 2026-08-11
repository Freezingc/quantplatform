"""Deterministic liquid A-share universe rules."""

from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class UniverseRules:
    """Approved daily liquid-A-share eligibility constraints."""

    allowed_boards: frozenset[str] = frozenset({"SH_MAIN", "STAR", "SZ_MAIN", "CHINEXT"})
    minimum_listing_trading_days: int = 252
    minimum_valid_days_20: int = 15
    retained_adv20_fraction: float = 0.80

    def __post_init__(self) -> None:
        if self.minimum_listing_trading_days < 1 or self.minimum_valid_days_20 < 1:
            raise ValueError("universe history thresholds must be positive")
        if not 0 < self.retained_adv20_fraction <= 1:
            raise ValueError("retained_adv20_fraction must be in (0, 1]")


def build_universe(candidates: pd.DataFrame, rules: UniverseRules) -> pd.DataFrame:
    """Apply eligibility and stable ADV20 selection to one daily cross-section."""
    required = {
        "ts_code",
        "board",
        "is_listed",
        "is_st",
        "is_suspended",
        "listing_age_trading_days",
        "valid_days_20",
        "adv20",
    }
    missing = sorted(required.difference(candidates.columns))
    if missing:
        raise ValueError(f"universe candidates are missing columns: {missing}")
    universe = candidates.copy(deep=True)
    universe["is_member"] = False
    universe["exclusion_reason"] = pd.NA
    universe["adv20_rank"] = pd.Series(pd.NA, index=universe.index, dtype="Int64")
    universe["membership_rank"] = pd.Series(pd.NA, index=universe.index, dtype="Int64")
    if universe["ts_code"].duplicated().any():
        raise ValueError("universe candidates contain duplicate ts_code values")
    ordered_rules = (
        (~universe["board"].isin(rules.allowed_boards), "OUTSIDE_BOARD"),
        (~universe["is_listed"], "NOT_LISTED"),
        (universe["is_st"], "ST"),
        (universe["is_suspended"], "SUSPENDED"),
        (
            universe["listing_age_trading_days"] < rules.minimum_listing_trading_days,
            "YOUNG_LISTING",
        ),
        (universe["valid_days_20"] < rules.minimum_valid_days_20, "INSUFFICIENT_HISTORY"),
    )
    for mask, reason in ordered_rules:
        universe.loc[mask & universe["exclusion_reason"].isna(), "exclusion_reason"] = reason

    eligible = universe[universe["exclusion_reason"].isna()].copy()
    eligible = eligible.sort_values(
        ["adv20", "ts_code"],
        ascending=[False, True],
        na_position="last",
    )
    eligible["adv20_rank"] = range(1, len(eligible) + 1)
    keep_count = math.ceil(len(eligible) * rules.retained_adv20_fraction)
    retained = eligible.head(keep_count).copy()
    retained["membership_rank"] = range(1, len(retained) + 1)
    adv20_rank_by_code = dict(zip(eligible["ts_code"], eligible["adv20_rank"], strict=True))
    membership_rank_by_code = dict(
        zip(retained["ts_code"], retained["membership_rank"], strict=True)
    )
    keep_codes = set(retained["ts_code"])
    universe["adv20_rank"] = universe["ts_code"].map(adv20_rank_by_code).astype("Int64")
    universe["membership_rank"] = (
        universe["ts_code"].map(membership_rank_by_code).astype("Int64")
    )
    universe.loc[universe["ts_code"].isin(keep_codes), "is_member"] = True
    liquidity_rejected = universe["exclusion_reason"].isna() & ~universe["is_member"]
    universe.loc[liquidity_rejected, "exclusion_reason"] = "LOW_LIQUIDITY"
    return universe.sort_values("ts_code", ignore_index=True)
