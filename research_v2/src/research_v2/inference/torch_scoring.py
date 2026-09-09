"""Date-isolated scoring for the torch candidate families.

This logic lived in the candidate training driver, a script, which meant nothing outside that
script could score a torch bundle -- including the reproduction verifier, which has to recompute a
run's frozen TEST scores from its persisted bundle to show they are what the bundle produces rather
than merely what is recorded against it. Moving it here makes it importable and independently
testable while the driver keeps the same behaviour through aliased imports.

The padding contract in ``score_batch`` is load-bearing and its comment is kept verbatim: scoring
``batch.inputs.ts_codes`` directly instead of going through ``predictions_to_frame`` emits one row
per padding slot, all keyed ``ts_code=""``, and ranks real stocks against a denominator that grows
with the padding. That is what killed the 2026-09-05 full M1 run after 7h58m of training.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import cast

import pandas as pd
import torch

from research_v2.datasets.preprocessing import FoldPreprocessor
from research_v2.models.base import predictions_to_frame
from research_v2.training.batches import TemporalTrainingBatch, TrainingBatch

PredictionBatch = TrainingBatch | TemporalTrainingBatch


def score_batch(model: torch.nn.Module, batch: PredictionBatch) -> pd.DataFrame:
    """Score one date-isolated batch through the shared padding-dropping contract.

    ``predictions_to_frame`` is the same function the inference predictor and the
    validation-governed training loop already use: it keeps only ``stock_mask``
    positions, rejects non-finite scores for valid stocks, and ranks within each
    date. Reading ``batch.inputs.ts_codes`` directly instead emits one row per
    padding slot -- every one of them keyed ``ts_code=""`` -- and ranks the real
    stocks against a denominator that grows with the padding, which differs on
    every date narrower than the segment's widest one. That combination is what
    killed the 2026-09-05 full M1 run after 7h58m of training with "seed
    predictions contain duplicate business keys", and it would have biased the
    validation RankIC that selects the winning config even when it did not crash.
    """
    with torch.no_grad():
        scores = cast(torch.Tensor, model(batch.inputs))
    return predictions_to_frame(batch.inputs, scores)


def percentile_frame(
    model: torch.nn.Module, batches: Mapping[date, PredictionBatch]
) -> pd.DataFrame:
    """Per-seed within-date percentile ranks for a whole segment, padding dropped."""
    model.eval()
    frames = [score_batch(model, batches[day]) for day in sorted(batches)]
    if not frames:
        raise ValueError("no batches to predict on")
    combined = pd.concat(frames, ignore_index=True)
    return combined.loc[:, ["signal_date", "ts_code", "percentile_rank"]]


def mean_seed_percentile_scores(
    models: Sequence[torch.nn.Module], batch: PredictionBatch
) -> pd.DataFrame:
    """Average the seed percentile ranks of one date, aligned on business keys.

    ``predictions_to_frame`` sorts each date by score, so two seeds generally emit
    their rows in different orders and averaging the columns positionally would
    blend different stocks. ``validate="one_to_one"`` turns any residual key
    duplication into an error here rather than inside the OOS evaluation.
    """
    if not models:
        raise ValueError("at least one seed model is required")
    keys = ["signal_date", "ts_code"]
    frames = [score_batch(model, batch) for model in models]
    merged = frames[0].loc[:, [*keys, "percentile_rank"]]
    for index, frame in enumerate(frames[1:], start=1):
        merged = merged.merge(
            frame.loc[:, [*keys, "percentile_rank"]].rename(
                columns={"percentile_rank": f"seed_rank_{index}"}
            ),
            on=keys,
            validate="one_to_one",
        )
    rank_columns = [
        "percentile_rank", *(f"seed_rank_{index}" for index in range(1, len(frames)))
    ]
    merged["score"] = merged.loc[:, rank_columns].mean(axis=1)
    return merged.loc[:, [*keys, "score"]]


def concat_date_scores(
    models: Sequence[torch.nn.Module], batches: Mapping[date, PredictionBatch]
) -> pd.DataFrame:
    """Seed-mean percentile scores for every date of a segment, in date order."""
    if not batches:
        raise ValueError("no batches to predict on")
    return pd.concat(
        [mean_seed_percentile_scores(models, batches[day]) for day in sorted(batches)],
        ignore_index=True,
    )


def prepare_frame(
    frame: pd.DataFrame, preprocessor: FoldPreprocessor, factor_order: tuple[str, ...]
) -> pd.DataFrame:
    """Apply the train-fitted preprocessor and re-join business keys and labels."""
    transformed = preprocessor.transform(frame.loc[:, list(factor_order)])
    keys = ["signal_date", "ts_code"]
    extras = [c for c in ("raw_return_5d", "label_valid") if c in frame.columns]
    base = frame.loc[:, keys + extras].reset_index(drop=True)
    return pd.concat([base, transformed.values.reset_index(drop=True)], axis=1)


__all__ = [
    "PredictionBatch",
    "concat_date_scores",
    "mean_seed_percentile_scores",
    "percentile_frame",
    "prepare_frame",
    "score_batch",
]
