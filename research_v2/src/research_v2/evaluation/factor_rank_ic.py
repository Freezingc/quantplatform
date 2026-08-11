"""Cross-sectional RankIC/ICIR evaluation for a panel of factor values."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd


def compute_factor_daily_rank_ic(
    factors: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    minimum_cross_section: int = 30,
    factor_name_column: str = "factor_name",
    factor_value_column: str = "value",
    label_column: str = "raw_return",
) -> pd.DataFrame:
    """Compute one Spearman RankIC row for every factor/date cross-section.

    The function accepts the long-form YesLab factor schema and the published
    V2 label schema. Invalid labels and non-finite factor values are excluded
    independently for each factor/date; no cross-sectional values are pooled
    across dates.
    """
    if minimum_cross_section < 2:
        raise ValueError("minimum_cross_section must be at least 2")
    required_factors = {"signal_date", "ts_code", factor_name_column, factor_value_column}
    if missing := sorted(required_factors.difference(factors.columns)):
        raise ValueError(f"factors are missing columns: {missing}")
    resolved_label_column = _resolve_label_column(labels, label_column)
    required_labels = {"signal_date", "ts_code", resolved_label_column, "label_valid"}
    if missing := sorted(required_labels.difference(labels.columns)):
        raise ValueError(f"labels are missing columns: {missing}")
    keys = ["signal_date", "ts_code"]
    if factors.duplicated(keys + [factor_name_column]).any():
        raise ValueError("factors must have unique factor/date/security keys")
    if labels.duplicated(keys).any():
        raise ValueError("labels must have unique date/security keys")
    factor_frame = factors.loc[:, keys + [factor_name_column, factor_value_column]].copy()
    label_frame = labels.loc[:, keys + [resolved_label_column, "label_valid"]].copy()
    factor_frame["signal_date"] = pd.to_datetime(
        factor_frame["signal_date"], errors="raise"
    ).dt.date
    label_frame["signal_date"] = pd.to_datetime(
        label_frame["signal_date"], errors="raise"
    ).dt.date
    factor_frame[factor_value_column] = pd.to_numeric(
        factor_frame[factor_value_column], errors="coerce"
    )
    label_frame[resolved_label_column] = pd.to_numeric(
        label_frame[resolved_label_column], errors="coerce"
    )
    joined = factor_frame.merge(label_frame, on=keys, how="inner", validate="many_to_one")
    joined = joined.loc[
        joined["label_valid"].astype(bool)
        & np.isfinite(joined[factor_value_column])
        & np.isfinite(joined[resolved_label_column])
    ]
    rows: list[dict[str, object]] = []
    for (factor_name, signal_date), group in joined.groupby(
        [factor_name_column, "signal_date"], sort=True
    ):
        if len(group) < minimum_cross_section:
            continue
        factor_rank = group[factor_value_column].rank(method="average")
        label_rank = group[resolved_label_column].rank(method="average")
        if factor_rank.nunique(dropna=False) < 2 or label_rank.nunique(dropna=False) < 2:
            continue
        rank_ic = factor_rank.corr(label_rank, method="pearson")
        if pd.notna(rank_ic) and math.isfinite(float(rank_ic)):
            rows.append(
                {
                    "factor_name": str(factor_name),
                    "signal_date": signal_date,
                    "rank_ic": float(rank_ic),
                    "sample_count": len(group),
                }
            )
    result = pd.DataFrame(
        rows,
        columns=["factor_name", "signal_date", "rank_ic", "sample_count"],
    )
    return result.sort_values(["factor_name", "signal_date"], ignore_index=True)


def summarize_factor_rank_ic(daily_rank_ic: pd.DataFrame) -> pd.DataFrame:
    """Summarize daily RankIC values into factor-level ICIR metrics."""
    required = {"factor_name", "signal_date", "rank_ic", "sample_count"}
    if missing := sorted(required.difference(daily_rank_ic.columns)):
        raise ValueError(f"daily RankIC is missing columns: {missing}")
    frame = daily_rank_ic.copy()
    frame["rank_ic"] = pd.to_numeric(frame["rank_ic"], errors="coerce")
    frame["sample_count"] = pd.to_numeric(frame["sample_count"], errors="coerce")
    frame = frame.loc[np.isfinite(frame["rank_ic"]) & np.isfinite(frame["sample_count"])]
    rows: list[dict[str, object]] = []
    for factor_name, group in frame.groupby("factor_name", sort=True):
        values = group["rank_ic"]
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
        rows.append(
            {
                "factor_name": str(factor_name),
                "valid_day_count": len(values),
                "mean_rank_ic": mean,
                "rank_ic_std": std,
                "icir": mean / std if math.isfinite(std) and std > 0 else float("nan"),
                "positive_day_share": float((values > 0).mean()),
                "mean_sample_count": float(group["sample_count"].mean()),
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "factor_name",
            "valid_day_count",
            "mean_rank_ic",
            "rank_ic_std",
            "icir",
            "positive_day_share",
            "mean_sample_count",
        ],
    )


def _resolve_label_column(labels: pd.DataFrame, requested: str) -> str:
    if requested in labels.columns:
        return requested
    if requested == "raw_return" and "raw_return_5d" in labels.columns:
        return "raw_return_5d"
    raise ValueError(f"labels are missing columns: [{requested!r}]")
