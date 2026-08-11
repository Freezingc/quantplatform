"""Pre-registered, content-addressed trial scheduling and validation-only selection."""

from __future__ import annotations

import hashlib
import itertools
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True)
class SearchTrial:
    model_family: str
    config: Mapping[str, object]
    config_hash: str


def build_pre_registered_trials(
    model_family: str,
    search_space: Mapping[str, Sequence[object]],
    *,
    maximum_trials: int = 30,
) -> tuple[SearchTrial, ...]:
    """Enumerate a declared grid, sort by canonical config hash, and apply the hard cap."""
    if not model_family or not search_space or not 1 <= maximum_trials <= 30:
        raise ValueError("model_family, search_space, or maximum_trials is invalid")
    names = sorted(search_space)
    values_per_name: list[list[object]] = []
    for name in names:
        values = list(search_space[name])
        if not name or not values:
            raise ValueError("search dimensions must have names and values")
        canonical_values = sorted(values, key=_canonical_json)
        if len({_canonical_json(value) for value in canonical_values}) != len(canonical_values):
            raise ValueError(f"search dimension contains duplicate values: {name}")
        values_per_name.append(canonical_values)
    trials: list[SearchTrial] = []
    for combination in itertools.product(*values_per_name):
        config = dict(zip(names, combination, strict=True))
        payload = {"model_family": model_family, "config": config}
        config_hash = hashlib.sha256(_canonical_json(payload).encode()).hexdigest()
        trials.append(SearchTrial(model_family, config, config_hash))
    return tuple(sorted(trials, key=lambda trial: trial.config_hash)[:maximum_trials])


def select_validation_winner(
    validation_rank_ic: Mapping[str, float],
    trials: Sequence[SearchTrial],
) -> SearchTrial:
    """Choose a completed trial solely from validation RankIC, with a stable hash tie-break."""
    if not trials:
        raise ValueError("at least one trial is required")
    hashes = {trial.config_hash for trial in trials}
    if set(validation_rank_ic) != hashes:
        raise ValueError("validation results must cover exactly the declared trials")
    if any(not math.isfinite(value) for value in validation_rank_ic.values()):
        raise ValueError("validation RankIC values must be finite")
    return min(
        trials,
        key=lambda trial: (-validation_rank_ic[trial.config_hash], trial.config_hash),
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
