"""Mandatory three-seed percentile-rank ensemble construction."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED_SEEDS = (20260801, 20260802, 20260803)


@dataclass(frozen=True)
class SeedEnsemble:
    predictions: pd.DataFrame
    seed_mean_validation_rank_ic: float
    seed_std_validation_rank_ic: float
    worst_seed: int


def build_seed_ensemble(
    predictions_by_seed: Mapping[int, pd.DataFrame],
    *,
    validation_rank_ic: Mapping[int, float],
) -> SeedEnsemble:
    """Equal-weight all required seed ranks; never select or omit a convenient seed."""
    if set(validation_rank_ic) != set(REQUIRED_SEEDS):
        raise ValueError("all required seeds must be present exactly once")
    if any(not math.isfinite(value) for value in validation_rank_ic.values()):
        raise ValueError("seed validation RankIC values must be finite")
    result = combine_seed_percentile_ranks(predictions_by_seed)
    values = np.asarray([validation_rank_ic[seed] for seed in REQUIRED_SEEDS], dtype=float)
    return SeedEnsemble(
        predictions=result,
        seed_mean_validation_rank_ic=float(values.mean()),
        seed_std_validation_rank_ic=float(values.std(ddof=1)),
        worst_seed=min(REQUIRED_SEEDS, key=lambda seed: (validation_rank_ic[seed], seed)),
    )


def combine_seed_percentile_ranks(predictions_by_seed: Mapping[int, pd.DataFrame]) -> pd.DataFrame:
    """Combine exactly three keyed percentile-rank frames without accessing labels or metrics."""
    if set(predictions_by_seed) != set(REQUIRED_SEEDS):
        raise ValueError("all required seeds must be present exactly once")
    keys = ["signal_date", "ts_code"]
    normalized: dict[int, pd.DataFrame] = {}
    reference_keys: set[tuple[object, object]] | None = None
    for seed in REQUIRED_SEEDS:
        frame = predictions_by_seed[seed]
        required = set(keys + ["percentile_rank"])
        if missing := sorted(required.difference(frame.columns)):
            raise ValueError(f"seed predictions are missing columns: {missing}")
        if frame.duplicated(keys).any():
            raise ValueError("seed predictions contain duplicate business keys")
        ranks = pd.to_numeric(frame["percentile_rank"], errors="coerce")
        if not np.isfinite(ranks).all() or ((ranks < 0) | (ranks > 1)).any():
            raise ValueError("seed percentile ranks must be finite values in [0, 1]")
        cleaned = frame.loc[:, keys].copy()
        cleaned["percentile_rank"] = ranks
        frame_keys = set(zip(cleaned["signal_date"], cleaned["ts_code"], strict=True))
        if reference_keys is None:
            reference_keys = frame_keys
        elif frame_keys != reference_keys:
            raise ValueError("all seed predictions must have identical business-key coverage")
        normalized[seed] = cleaned
    result = normalized[REQUIRED_SEEDS[0]].loc[:, keys].copy()
    for seed in REQUIRED_SEEDS:
        result = result.merge(
            normalized[seed].rename(columns={"percentile_rank": f"seed_{seed}_rank"}),
            on=keys,
            how="inner",
            validate="one_to_one",
        )
    score_columns = [f"seed_{seed}_rank" for seed in REQUIRED_SEEDS]
    result["ensemble_score"] = result.loc[:, score_columns].mean(axis=1)
    result = result.sort_values(keys, kind="mergesort", ignore_index=True)
    return result.loc[:, keys + ["ensemble_score"]]
