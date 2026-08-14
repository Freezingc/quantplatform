"""Evaluate isolated scale-normalized Alpha95 research candidates."""

from collections.abc import Mapping

import numpy as np
import pandas as pd

CANDIDATE_WINDOWS = (10, 20, 40)


def compute_relative_amount_volatility(market: pd.DataFrame, window: int) -> pd.DataFrame:
    """Return trailing amount standard deviation relative to trailing mean."""
    required = {"trade_date", "ts_code", "amount"}
    if missing := sorted(required.difference(market.columns)):
        raise ValueError(f"market frame is missing columns: {missing}")
    amount = market.pivot(index="trade_date", columns="ts_code", values="amount").sort_index()
    deviation = amount.rolling(window=window, min_periods=window).std(ddof=1)
    baseline = amount.rolling(window=window, min_periods=window).mean()
    return deviation.div(baseline.replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)


def passes_training_gate(row: Mapping[str, float]) -> bool:
    """Return whether one candidate clears every fixed training gate."""
    return (
        row["signed_mean_rank_ic"] > 0.02
        and row["signed_icir"] >= 0.50
        and row["directional_day_share"] >= 0.55
        and row["annual_double_gate_pass_years"] >= 4
        and row["max_similarity"] < 0.90
    )


def select_training_candidate(rows: Mapping[str, Mapping[str, float]]) -> str | None:
    """Select the highest-ICIR eligible candidate using training-only evidence."""
    eligible = [
        (candidate_id, row)
        for candidate_id, row in rows.items()
        if passes_training_gate(row)
    ]
    if not eligible:
        return None
    return max(
        eligible,
        key=lambda item: (item[1]["signed_icir"], -item[1]["max_similarity"], item[0]),
    )[0]
