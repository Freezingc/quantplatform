"""Resume a fold's config search from persisted per-config results.

The candidate driver's search loop has no fold- or config-level resume: ``_existing_evidence_id``
makes the *evidence* writes idempotent, but nothing skips the search, so an interruption at fold 3
config 20 restarts at fold 1 config 0 seed 1. Measured on this host that is ~2.1 hours per fit and
27 configs x 3 seeds per fold, i.e. the runs this protects (M2, M3) are multi-day to multi-week,
and one interruption costs everything. Recorded as an open operational risk in
docs/superpowers/plans/2026-09-04-real-candidate-model-training-plan.md §3.4.

Two properties are load-bearing.

Persisting changes no number. A record is one extra JSON file inside the run's own workspace, and
it is written whether or not resume is enabled, so the default path is byte-for-byte what it was.

Adopting is earned, not assumed. A record may replace hours of training only when every
numerics-affecting input it was produced under is still the one this run would use: the
content-addressed train and validation dataset ids (so any data change refuses), the factor order,
the epoch budget, the family and its code version, the config payload and its hash, and an explicit
runtime descriptor. The runtime descriptor is the part that is easy to omit and expensive to miss:
torch's intra-op thread count is neither pinned nor declared, defaults to half the host core count,
and changes checkpoint digests (measured: three distinct digests at 1/4/8 threads), so a record
produced under a different thread count is a different result and is refused. Callers declare which
runtime fields bind them -- ``torch_runtime_bindings()`` for the torch families, a LightGBM
descriptor for the tree families -- because which properties affect numerics is family knowledge.

Adoption is all-or-nothing per config and restores the exact result dict the fresh search would have
appended (same float bits, integer seed keys put back after JSON stringified them). That is what
lets a resumed run's report equal a fresh run's report instead of merely resembling one.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from research_v2.training.ensemble import REQUIRED_SEEDS

SCHEMA = "qrv2-search-record-1"


@dataclass(frozen=True)
class SearchRecordLookup:
    """One config's adoption attempt: the restored result, or why there is none.

    ``reason`` is never empty when ``record`` is None, because a transcript that does not say why a
    config was trained again is unauditable -- and because "resumed everything" and "resumed
    nothing" look identical in the report otherwise.
    """

    record: dict[str, Any] | None
    reason: str = ""
    path: Path | None = None


def torch_runtime_bindings() -> dict[str, object]:
    """The numerics-affecting torch runtime, OBSERVED and never configured.

    Pinning the thread count would change results and is an adjudicated change (see the compute
    decision brief); recording it is free, and it is what makes a record produced under another
    thread count refusable rather than silently adopted.
    """
    return {
        "torch_version": torch.__version__,
        "torch_intraop_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
    }


def config_hash(config: Mapping[str, Any]) -> str:
    """Canonical hash of one search config; a record's filename is bound to it."""
    payload = json.dumps(dict(config), sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class SearchResume:
    """Per-fold persistence and adoption of config-search results."""

    def __init__(
        self,
        *,
        workspace: Path,
        fold_id: str,
        family: str,
        model_code_version: str,
        train_dataset_id: str,
        validation_dataset_id: str,
        factor_order: tuple[str, ...],
        runtime: Mapping[str, object],
        enabled: bool = False,
        log: Callable[[str], None] | None = None,
    ) -> None:
        self._root = Path(workspace) / "search" / fold_id
        self._fold_id = fold_id
        self._fold_bindings: dict[str, Any] = {
            "fold_id": fold_id,
            "family": family,
            "model_code_version": model_code_version,
            "train_dataset_id": train_dataset_id,
            "validation_dataset_id": validation_dataset_id,
            "factor_order": list(factor_order),
            "runtime": dict(runtime),
        }
        self._enabled = enabled
        self._log = log

    def _bindings(self, budget: Mapping[str, Any]) -> dict[str, Any]:
        """The fold's bindings plus this config's effective budget, normalised for comparison.

        The budget is per config, not per fold, because for the tree families it includes the
        effective hyperparameters -- the config payload plus the library-side defaults it did not
        name. Those defaults live in this repository and ``model_code_version`` is a hand-maintained
        label that does not move when they do, so a config hash alone would adopt a record produced
        under different effective hyperparameters. Normalising through JSON is what makes the
        comparison survive a write/read cycle: tuples become lists and integer keys become strings
        on both sides alike, so equality means equality of the thing recorded, not of its encoding.
        """
        return dict(self._fold_bindings, budget=_json_safe(budget))

    def record_path(self, config_index: int, config: Mapping[str, Any]) -> Path:
        """Where one config's record lives: bound to its grid index and to its own content hash."""
        return self._root / f"c{config_index}-{config_hash(config)[:16]}.json"

    def persist(
        self,
        config_index: int,
        config: Mapping[str, Any],
        result: Mapping[str, Any],
        *,
        budget: Mapping[str, Any],
    ) -> Path:
        """Write one config's result so a later run may adopt it. Unconditional, numerics-free."""
        validated = _validated_result(result)
        document = {
            "schema": SCHEMA,
            "config_index": config_index,
            "config": dict(config),
            "config_hash": config_hash(config),
            "bindings": self._bindings(budget),
            "result": validated,
        }
        path = self.record_path(config_index, config)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write-then-rename, so an interruption mid-write cannot leave a half-written record that a
        # later run reads as a result. A truncated document would fail to parse and be refused, but
        # atomicity means the reader never sees one at all.
        temporary = path.with_name(path.name + ".partial")
        temporary.write_text(
            json.dumps(document, sort_keys=True, indent=2, ensure_ascii=True), encoding="utf-8"
        )
        temporary.replace(path)
        return path

    def adopt(
        self, config_index: int, config: Mapping[str, Any], *, budget: Mapping[str, Any]
    ) -> SearchRecordLookup:
        """Return this config's persisted result when every binding still holds, else refuse it."""
        path = self.record_path(config_index, config)
        if not self._enabled:
            # Silent on purpose: logging a refusal for every config of every ordinary run would
            # bury the transcript of the runs that are not resuming at all.
            return SearchRecordLookup(record=None, reason="resume is disabled", path=path)
        if not path.exists():
            return self._refuse(path, f"no record at {path.name}")
        try:
            document = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            return self._refuse(path, f"record is unreadable ({error})")
        if not isinstance(document, dict):
            return self._refuse(path, "record is not an object")
        if document.get("schema") != SCHEMA:
            return self._refuse(
                path, f"record schema {document.get('schema')!r} is not {SCHEMA!r}"
            )
        bindings = document.get("bindings")
        if not isinstance(bindings, dict):
            return self._refuse(path, "record carries no bindings")
        for field, expected in self._bindings(budget).items():
            recorded = bindings.get(field)
            if recorded != expected:
                return self._refuse(
                    path,
                    f"{field} differs: the record was produced with {recorded!r}, this run has "
                    f"{expected!r}",
                )
        if document.get("config_hash") != config_hash(config):
            return self._refuse(path, "config hash differs from this run's config")
        if document.get("config") != dict(config):
            return self._refuse(path, "config differs from the record's own config")
        result = document.get("result")
        if not isinstance(result, dict):
            return self._refuse(path, "record carries no result")
        try:
            restored = _restored_result(result)
        except ValueError as error:
            return self._refuse(path, f"record result is unusable: {error}")
        self._emit(
            f"  {self._fold_id} config {config_index}: resumed from {path.name} "
            f"(ensemble validation RankIC={restored['ensemble_validation_rank_ic']:.4f})"
        )
        return SearchRecordLookup(record=restored, path=path)

    def _refuse(self, path: Path, reason: str) -> SearchRecordLookup:
        self._emit(f"  {self._fold_id} record {path.name} not resumable: {reason}")
        return SearchRecordLookup(record=None, reason=reason, path=path)

    def _emit(self, message: str) -> None:
        if self._log is not None:
            self._log(message)


def _json_safe(value: Any) -> Any:
    """Round-trip a binding value through JSON so stored and live values compare on content."""
    return json.loads(json.dumps(value, sort_keys=True, ensure_ascii=True))


def _validated_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Check and normalise a search result before storing it, so nothing unreadable is persisted."""
    config = result.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("search result carries no config")
    ensemble = result.get("ensemble_validation_rank_ic")
    if not isinstance(ensemble, float) or not math.isfinite(ensemble):
        raise ValueError("ensemble_validation_rank_ic must be a finite float")
    seeds = result.get("seed_validation_rank_ic")
    if not isinstance(seeds, Mapping) or set(seeds) != set(REQUIRED_SEEDS):
        raise ValueError(
            "seed_validation_rank_ic must cover exactly the required seeds "
            f"{sorted(REQUIRED_SEEDS)}"
        )
    seed_map: dict[int, float] = {}
    for seed, value in seeds.items():
        if not isinstance(value, float) or not math.isfinite(value):
            raise ValueError(f"seed_validation_rank_ic[{seed}] must be a finite float")
        seed_map[int(seed)] = value
    worst = result.get("worst_seed")
    if not isinstance(worst, int) or isinstance(worst, bool) or worst not in REQUIRED_SEEDS:
        raise ValueError("worst_seed must be one of the required seeds")
    return {
        "config": _json_safe(dict(config)),
        "ensemble_validation_rank_ic": ensemble,
        "seed_validation_rank_ic": seed_map,
        "worst_seed": worst,
    }


def _restored_result(result: Mapping[str, Any]) -> dict[str, Any]:
    """Read a stored result back into the exact shape the fresh search would have appended.

    JSON stringifies integer keys and cannot be trusted to have kept types, so every field is
    re-checked here rather than cast: a record that restores to a different shape than a fresh
    search would produce is a record whose run's report would differ from its fresh equivalent.
    """
    config = result.get("config")
    if not isinstance(config, Mapping):
        raise ValueError("record result carries no config")
    ensemble = result.get("ensemble_validation_rank_ic")
    if not isinstance(ensemble, float) or not math.isfinite(ensemble):
        raise ValueError("record result has no finite float ensemble_validation_rank_ic")
    seeds = result.get("seed_validation_rank_ic")
    if not isinstance(seeds, Mapping):
        raise ValueError("record result has no seed_validation_rank_ic")
    seed_map: dict[int, float] = {}
    for key, value in seeds.items():
        try:
            seed = int(key)
        except (TypeError, ValueError) as error:
            raise ValueError(f"record result has a non-integer seed key {key!r}") from error
        if not isinstance(value, float) or not math.isfinite(value):
            raise ValueError(f"record result seed {seed} has no finite float RankIC")
        seed_map[seed] = value
    if set(seed_map) != set(REQUIRED_SEEDS):
        raise ValueError(
            f"record result covers {sorted(seed_map)}, not exactly {sorted(REQUIRED_SEEDS)}"
        )
    worst = result.get("worst_seed")
    if not isinstance(worst, int) or isinstance(worst, bool) or worst not in REQUIRED_SEEDS:
        raise ValueError("record result has no usable worst_seed")
    return {
        "config": dict(config),
        "ensemble_validation_rank_ic": ensemble,
        "seed_validation_rank_ic": seed_map,
        "worst_seed": worst,
    }


__all__ = [
    "SCHEMA",
    "SearchRecordLookup",
    "SearchResume",
    "config_hash",
    "torch_runtime_bindings",
]
