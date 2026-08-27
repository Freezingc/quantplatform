"""Daily cross-sectional RankIC and OOS ICIR summaries."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from research_v2.evaluation.gates import ScoreHealthEvidence


@dataclass(frozen=True)
class RankICSummary:
    valid_day_count: int
    mean_rank_ic: float
    rank_ic_std: float
    icir: float
    invalid_day_count: int = 0
    positive_day_share: float = float("nan")
    mean_sample_count: float = float("nan")
    sample_count_p05: float = float("nan")
    sample_count_median: float = float("nan")
    sample_count_p95: float = float("nan")


def compute_daily_rank_ic(
    predictions: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    minimum_cross_section: int = 30,
    score_column: str = "score",
    label_column: str = "raw_return_5d",
    include_invalid: bool = False,
    require_score_health: bool | None = None,
) -> pd.DataFrame:
    """Compute daily Spearman RankIC and optional score-health diagnostics.

    The default keeps the historical API and returns only valid RankIC days.  With
    ``include_invalid=True``, every label date is retained with ``rank_ic_valid``
    and score-health fields so callers can distinguish an unevaluable day from a
    genuine zero correlation.  ``require_score_health`` applies the production
    score-health thresholds to RankIC eligibility.  It defaults to
    ``include_invalid`` because the diagnostic OOS mode is the production
    acceptance path; callers using small synthetic cross-sections can explicitly
    disable it.
    """
    if minimum_cross_section < 2:
        raise ValueError("minimum_cross_section must be at least 2")
    enforce_score_health = include_invalid if require_score_health is None else require_score_health
    required_predictions = {"signal_date", "ts_code", score_column}
    required_labels = {"signal_date", "ts_code", label_column, "label_valid"}
    if missing := sorted(required_predictions.difference(predictions.columns)):
        raise ValueError(f"predictions are missing columns: {missing}")
    if missing := sorted(required_labels.difference(labels.columns)):
        raise ValueError(f"labels are missing columns: {missing}")
    keys = ["signal_date", "ts_code"]
    prediction_frame = predictions.loc[:, keys + [score_column]].copy()
    label_frame = labels.loc[:, keys + [label_column, "label_valid"]].copy()
    prediction_frame["signal_date"] = pd.to_datetime(
        prediction_frame["signal_date"], errors="raise"
    ).dt.date
    label_frame["signal_date"] = pd.to_datetime(label_frame["signal_date"], errors="raise").dt.date
    if prediction_frame.duplicated(keys).any() or label_frame.duplicated(keys).any():
        raise ValueError("predictions and labels must have unique signal-date keys")
    label_frame["label_valid"] = label_frame["label_valid"].fillna(False).astype(bool)
    joined = label_frame.merge(
        prediction_frame,
        on=keys,
        how="left",
        validate="one_to_one",
    )
    joined[score_column] = pd.to_numeric(joined[score_column], errors="coerce")
    joined[label_column] = pd.to_numeric(joined[label_column], errors="coerce")
    rows: list[dict[str, object]] = []
    for signal_date, cross_section in joined.groupby("signal_date", sort=True):
        valid_label = cross_section["label_valid"] & np.isfinite(cross_section[label_column])
        valid_labels = cross_section.loc[valid_label]
        finite_score = valid_label & np.isfinite(cross_section[score_column])
        scored = cross_section.loc[finite_score]
        eligible_count = len(cross_section)
        valid_label_count = len(valid_labels)
        coverage = len(scored) / valid_label_count if valid_label_count > 0 else 0.0
        unique_score_count = int(scored[score_column].nunique())
        score_std = (
            float(np.std(scored[score_column].to_numpy(dtype=float), ddof=0))
            if not scored.empty
            else float("nan")
        )
        score_health_valid = ScoreHealthEvidence(
            coverage=float(coverage),
            unique_score_count=unique_score_count,
            score_std=score_std,
        ).passes()
        rank_ic = float("nan")
        rank_ic_valid = False
        if len(scored) >= minimum_cross_section and (
            not enforce_score_health or score_health_valid
        ):
            score_rank = scored[score_column].rank(method="average")
            label_rank = scored[label_column].rank(method="average")
            if score_rank.nunique(dropna=False) >= 2 and label_rank.nunique(dropna=False) >= 2:
                candidate_rank_ic = score_rank.corr(label_rank, method="pearson")
                if pd.notna(candidate_rank_ic) and math.isfinite(float(candidate_rank_ic)):
                    rank_ic = float(candidate_rank_ic)
                    rank_ic_valid = True
        rows.append(
            {
                "signal_date": signal_date,
                "rank_ic": rank_ic,
                "sample_count": len(scored),
                "eligible_count": eligible_count,
                "valid_label_count": valid_label_count,
                "coverage": float(coverage),
                "prediction_unique_count": unique_score_count,
                "prediction_std": score_std,
                "rank_ic_valid": rank_ic_valid,
            }
        )
    diagnostic_columns = [
        "signal_date",
        "rank_ic",
        "sample_count",
        "eligible_count",
        "valid_label_count",
        "coverage",
        "prediction_unique_count",
        "prediction_std",
        "rank_ic_valid",
    ]
    result = pd.DataFrame(rows, columns=diagnostic_columns)
    if not include_invalid:
        result = result.loc[result["rank_ic_valid"]].copy()
        result = result.loc[:, ["signal_date", "rank_ic", "sample_count"]]
    return result.sort_values("signal_date", ignore_index=True)


def summarize_rank_ic(daily_rank_ic: pd.DataFrame) -> RankICSummary:
    """Summarize OOS daily RankIC values without pooling cross-sections."""
    if "rank_ic" not in daily_rank_ic.columns:
        raise ValueError("daily RankIC frame is missing rank_ic")
    numeric_rank_ic = pd.to_numeric(daily_rank_ic["rank_ic"], errors="coerce")
    finite = np.isfinite(numeric_rank_ic)
    if "rank_ic_valid" in daily_rank_ic.columns:
        valid = daily_rank_ic["rank_ic_valid"].fillna(False).astype(bool) & finite
    else:
        valid = finite
    values = numeric_rank_ic.loc[valid]
    invalid_day_count = int(len(daily_rank_ic) - len(values))
    if values.empty:
        return RankICSummary(0, float("nan"), float("nan"), float("nan"), invalid_day_count)
    mean = float(values.mean())
    std = float(values.std(ddof=1)) if len(values) > 1 else float("nan")
    icir = mean / std if math.isfinite(std) and std > 0 else float("nan")
    sample_counts = (
        pd.to_numeric(daily_rank_ic.loc[valid, "sample_count"], errors="coerce")
        if "sample_count" in daily_rank_ic.columns
        else pd.Series(dtype=float)
    )
    sample_counts = sample_counts.loc[np.isfinite(sample_counts)]
    return RankICSummary(
        valid_day_count=len(values),
        mean_rank_ic=mean,
        rank_ic_std=std,
        icir=icir,
        invalid_day_count=invalid_day_count,
        positive_day_share=float((values > 0).mean()),
        mean_sample_count=(
            float(sample_counts.mean()) if not sample_counts.empty else float("nan")
        ),
        sample_count_p05=(
            float(sample_counts.quantile(0.05)) if not sample_counts.empty else float("nan")
        ),
        sample_count_median=(
            float(sample_counts.quantile(0.50)) if not sample_counts.empty else float("nan")
        ),
        sample_count_p95=(
            float(sample_counts.quantile(0.95)) if not sample_counts.empty else float("nan")
        ),
    )


def summarize_rank_ic_by_year(daily_rank_ic: pd.DataFrame) -> pd.DataFrame:
    """Return deterministic RankIC and sample-count summaries by calendar year."""
    if "signal_date" not in daily_rank_ic.columns:
        raise ValueError("daily RankIC frame is missing signal_date")
    columns = [
        "year",
        "valid_day_count",
        "invalid_day_count",
        "mean_rank_ic",
        "rank_ic_std",
        "icir",
        "positive_day_share",
        "mean_sample_count",
        "sample_count_p05",
        "sample_count_median",
        "sample_count_p95",
    ]
    if daily_rank_ic.empty:
        return pd.DataFrame(columns=columns)
    working = daily_rank_ic.copy()
    working["signal_date"] = pd.to_datetime(working["signal_date"], errors="raise")
    rows: list[dict[str, object]] = []
    for year, group in working.groupby(working["signal_date"].dt.year, sort=True):
        summary = summarize_rank_ic(group)
        rows.append(
            {
                "year": int(year),
                "valid_day_count": summary.valid_day_count,
                "invalid_day_count": summary.invalid_day_count,
                "mean_rank_ic": summary.mean_rank_ic,
                "rank_ic_std": summary.rank_ic_std,
                "icir": summary.icir,
                "positive_day_share": summary.positive_day_share,
                "mean_sample_count": summary.mean_sample_count,
                "sample_count_p05": summary.sample_count_p05,
                "sample_count_median": summary.sample_count_median,
                "sample_count_p95": summary.sample_count_p95,
            }
        )
    return pd.DataFrame(rows, columns=columns)
