"""The validation metric the config search selects on, defined once.

Three callers need this number: the driver's serial search loops and both spawned search workers.
It is the value ``max`` picks the winning configuration by, so three copies would be three chances
for "the model the search selected" to mean something subtly different from "the model the parent
then fits and freezes". ``minimum_cross_section=30`` is the same floor the cross-sectional losses
and the OOS evaluation use, so a date too thin to score is too thin to select on as well.
"""

from __future__ import annotations

import pandas as pd

from research_v2.evaluation.rank_ic import compute_daily_rank_ic, summarize_rank_ic


def validation_rank_ic(predictions: pd.DataFrame, labels: pd.DataFrame) -> float:
    """Mean daily RankIC of within-date percentile predictions against the labels.

    The join is on business keys, so a frame whose row order differs from the labels' still scores
    correctly; predictions are expected to carry ``percentile_rank`` and the labels
    ``raw_return_5d`` with ``label_valid``.
    """
    merged = predictions.merge(
        labels.loc[:, ["signal_date", "ts_code", "raw_return_5d", "label_valid"]],
        on=["signal_date", "ts_code"],
        how="inner",
    )
    daily = compute_daily_rank_ic(
        merged.loc[:, ["signal_date", "ts_code", "percentile_rank"]].rename(
            columns={"percentile_rank": "score"}
        ),
        merged.loc[:, ["signal_date", "ts_code", "raw_return_5d", "label_valid"]],
        minimum_cross_section=30,
    )
    return float(summarize_rank_ic(daily).mean_rank_ic)


__all__ = ["validation_rank_ic"]
