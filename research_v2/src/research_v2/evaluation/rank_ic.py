"""Daily cross-sectional RankIC and OOS ICIR summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class RankICSummary:
    valid_day_count: int
    mean_rank_ic: float
    rank_ic_std: float
    icir: float


def compute_daily_rank_ic(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    minimum_cross_section: int = 30,
    score_column: str = "score",
    label_column: str = "raw_return_5d",
) -> pd.DataFrame:
    """Compute Spearman RankIC for each valid signal-date cross-section."""
    required_predictions = {"signal_date", "ts_code", score_column}
    required_labels = {"signal_date", "ts_code", label_column, "label_valid"}
    if missing := sorted(required_predictions.difference(predictions.columns)):
        raise ValueError(f"predictions are missing columns: {missing}")
    if missing := sorted(required_labels.difference(labels.columns)):
        raise ValueError(f"labels are missing columns: {missing}")
    keys = ["signal_date", "ts_code"]
    if predictions.duplicated(keys).any() or labels.duplicated(keys).any():
        raise ValueError("predictions and labels must have unique signal-date keys")
    joined = predictions.loc[:, keys + [score_column]].merge(
        labels.loc[:, keys + [label_column, "label_valid"]],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    joined[score_column] = pd.to_numeric(joined[score_column], errors="coerce")
    joined[label_column] = pd.to_numeric(joined[label_column], errors="coerce")
    joined = joined.loc[
        joined["label_valid"]
        & np.isfinite(joined[score_column])
        & np.isfinite(joined[label_column])
    ]
    rows: list[dict[str, object]] = []
    for signal_date, cross_section in joined.groupby("signal_date", sort=True):
        if len(cross_section) < minimum_cross_section:
            continue
        rank_ic = cross_section[score_column].rank(method="average").corr(
            cross_section[label_column].rank(method="average"),
            method="pearson",
        )
        if pd.notna(rank_ic) and math.isfinite(float(rank_ic)):
            rows.append(
                {
                    "signal_date": signal_date,
                    "rank_ic": float(rank_ic),
                    "sample_count": len(cross_section),
                }
            )
    result = pd.DataFrame(rows, columns=["signal_date", "rank_ic", "sample_count"])
    return result.sort_values("signal_date", ignore_index=True)


def summarize_rank_ic(daily_rank_ic: pd.DataFrame) -> RankICSummary:
    """Summarize OOS daily RankIC values without pooling cross-sections."""
    if "rank_ic" not in daily_rank_ic.columns:
        raise ValueError("daily RankIC frame is missing rank_ic")
    values = pd.to_numeric(daily_rank_ic["rank_ic"], errors="coerce").dropna()
    if values.empty:
        return RankICSummary(0, float("nan"), float("nan"), float("nan"))
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    icir = mean / std if math.isfinite(std) and std > 0 else float("nan")
    return RankICSummary(len(values), mean, std, icir)
