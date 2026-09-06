"""Training-window-only statistical factor selection and correlation de-duplication."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any, TypeAlias, cast

import pandas as pd

from research_v2.evaluation.hac import hac_mean_test
from research_v2.evaluation.rank_ic import compute_daily_rank_ic

FactorKey: TypeAlias = tuple[str, int]


@dataclass(frozen=True)
class FactorSelectionRules:
    minimum_cross_section: int = 30
    minimum_valid_days: int = 120
    hac_max_lags: int = 5
    fdr_q_threshold: float = 0.10
    minimum_positive_years: int = 3
    minimum_positive_inner_folds: int = 2
    inner_fold_count: int = 3

    def __post_init__(self) -> None:
        if self.minimum_cross_section < 2 or self.minimum_valid_days <= self.hac_max_lags:
            raise ValueError("selection sample thresholds are invalid")
        if not 0 < self.fdr_q_threshold <= 1:
            raise ValueError("fdr_q_threshold must be in (0, 1]")
        if not 1 <= self.minimum_positive_inner_folds <= self.inner_fold_count:
            raise ValueError("inner-fold thresholds are invalid")


_DEFAULT_SELECTION_RULES = FactorSelectionRules()


@dataclass(frozen=True)
class FactorSelectionResult:
    candidates: pd.DataFrame


@dataclass(frozen=True)
class ClusterSelection:
    selected_factor_keys: tuple[FactorKey, ...]
    status: str
    validation_reason: str | None
    clusters: tuple[FactorCluster, ...] = ()


@dataclass(frozen=True)
class FactorCluster:
    """One connected component of the similarity threshold graph, with a stable identity.

    ``cluster_id`` is SHA-256 over the sorted member identities, so it depends only on which
    factors are in the cluster -- not on the order they were read, the order the edges
    arrived in, or which member happened to be chosen as the representative. Without an
    identity a clustering result cannot be referenced in evidence or compared across folds,
    and it would be the only grouping in this pipeline that is not content addressed:
    ``factor_set_id`` already is.
    """

    cluster_id: str
    members: tuple[FactorKey, ...]


def cluster_identity(members: Sequence[FactorKey]) -> str:
    """Content-address a cluster over its sorted member identities.

    The sort here looks redundant next to the ``tuple(sorted(members))`` at the call site,
    and it is not. ``select_cluster_representatives`` builds its key set with a set
    comprehension, and Python randomises string hashing per process, so the order members
    arrive in differs between runs. With this sort removed, cluster ids became
    PYTHONHASHSEED-dependent -- verified across four seeds -- which is a reproducibility
    defect rather than a cosmetic one: the same clustering would not be referable by the
    same id twice. Keeping the sort in the function itself also means any future caller
    gets that guarantee without having to remember to pre-sort.
    """
    payload = json.dumps(
        [[factor_id, version] for factor_id, version in sorted(members)],
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def benjamini_hochberg(p_values: Mapping[FactorKey, float]) -> dict[FactorKey, float]:
    """Return stable Benjamini–Hochberg adjusted q-values for one fold's family."""
    if not p_values:
        return {}
    if any(not math.isfinite(value) or value < 0 or value > 1 for value in p_values.values()):
        raise ValueError("p-values must be finite values in [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0][0], item[0][1]))
    family_size = len(ordered)
    adjusted: dict[FactorKey, float] = {}
    running_minimum = 1.0
    for position in range(family_size, 0, -1):
        factor_key, p_value = ordered[position - 1]
        running_minimum = min(running_minimum, p_value * family_size / position)
        adjusted[factor_key] = running_minimum
    return adjusted


def select_training_factors(
    observations: pd.DataFrame,
    labels: pd.DataFrame,
    *,
    factor_specs: pd.DataFrame,
    train_start: date,
    train_end: date,
    inner_validation_ics: Mapping[FactorKey, Sequence[float]],
    rules: FactorSelectionRules = _DEFAULT_SELECTION_RULES,
) -> FactorSelectionResult:
    """Score factors from a closed training interval, never from validation or test rows."""
    if train_start > train_end:
        raise ValueError("train_start must not be after train_end")
    required_observations = {"signal_date", "ts_code", "factor_id", "factor_version", "value"}
    required_labels = {"signal_date", "ts_code", "raw_return_5d", "label_valid"}
    required_specs = {"factor_id", "factor_version", "expected_direction"}
    _require_columns(observations, required_observations, "observations")
    _require_columns(labels, required_labels, "labels")
    _require_columns(factor_specs, required_specs, "factor_specs")
    if factor_specs.duplicated(["factor_id", "factor_version"]).any():
        raise ValueError("factor_specs contains duplicate factor identities")
    if observations.duplicated(["signal_date", "ts_code", "factor_id", "factor_version"]).any():
        raise ValueError("observations contains duplicate business keys")
    if labels.duplicated(["signal_date", "ts_code"]).any():
        raise ValueError("labels contains duplicate business keys")

    scoped_observations = _scope_to_train(observations, train_start, train_end)
    scoped_labels = _scope_to_train(labels, train_start, train_end)
    specs = factor_specs.copy(deep=True)
    specs["factor_id"] = specs["factor_id"].astype(str)
    specs["factor_version"] = specs["factor_version"].astype(int)
    candidate_rows: list[dict[str, object]] = []
    p_values: dict[FactorKey, float] = {}

    for spec in specs.sort_values(["factor_id", "factor_version"], kind="mergesort").to_dict(
        orient="records"
    ):
        factor_id = str(spec["factor_id"])
        factor_version = int(spec["factor_version"])
        factor_key = (factor_id, factor_version)
        expected_direction = int(spec["expected_direction"])
        if expected_direction not in {-1, 0, 1}:
            raise ValueError(f"expected_direction must be -1, 0, or 1: {factor_key}")
        structural_eligible = bool(spec.get("structural_eligible", True))
        factor_observations = scoped_observations.loc[
            (scoped_observations["factor_id"].astype(str) == factor_id)
            & (scoped_observations["factor_version"].astype(int) == factor_version),
            ["signal_date", "ts_code", "value"],
        ].rename(columns={"value": "score"})
        daily = compute_daily_rank_ic(
            factor_observations,
            scoped_labels,
            minimum_cross_section=rules.minimum_cross_section,
        )
        raw_mean = float(daily["rank_ic"].mean()) if not daily.empty else float("nan")
        frozen_direction = expected_direction
        if expected_direction == 0 and math.isfinite(raw_mean) and raw_mean != 0:
            frozen_direction = 1 if raw_mean > 0 else -1
        signed_daily = (
            daily["rank_ic"] * frozen_direction
            if frozen_direction
            else pd.Series(dtype=float)
        )
        p_value = float("nan")
        if structural_eligible and len(signed_daily) >= rules.minimum_valid_days:
            p_value = hac_mean_test(signed_daily, max_lags=rules.hac_max_lags).p_value
            p_values[factor_key] = p_value
        signed_mean = float(signed_daily.mean()) if not signed_daily.empty else float("nan")
        positive_year_count = _positive_year_count(daily, frozen_direction)
        inner_ics = tuple(float(value) for value in inner_validation_ics.get(factor_key, ()))
        if inner_ics and len(inner_ics) != rules.inner_fold_count:
            raise ValueError(
                f"inner validation IC count must be {rules.inner_fold_count}: {factor_key}"
            )
        positive_inner_count = sum(value * frozen_direction > 0 for value in inner_ics)
        candidate_rows.append(
            {
                "factor_id": factor_id,
                "factor_version": factor_version,
                "expected_direction": expected_direction,
                "frozen_direction": frozen_direction,
                "structural_eligible": structural_eligible,
                "valid_day_count": len(daily),
                "raw_mean_rank_ic": raw_mean,
                "signed_mean_rank_ic": signed_mean,
                "hac_p_value": p_value,
                "positive_year_count": positive_year_count,
                "positive_inner_count": positive_inner_count,
                "coverage": float(spec.get("coverage", 1.0)),
                "conflict_rate": float(spec.get("conflict_rate", 0.0)),
            }
        )

    q_values = benjamini_hochberg(p_values)
    family_size = len(q_values)
    for row in candidate_rows:
        factor_key = (str(row["factor_id"]), _to_int(row["factor_version"]))
        q_value = q_values.get(factor_key, float("nan"))
        row["fdr_q_value"] = q_value
        row["fdr_family_size"] = family_size
        row["status"] = _selection_status(row, q_value, rules)
    candidates = pd.DataFrame(candidate_rows).sort_values(
        ["factor_id", "factor_version"], kind="mergesort", ignore_index=True
    )
    return FactorSelectionResult(candidates=candidates)


def select_cluster_representatives(
    candidates: pd.DataFrame,
    similarity: pd.DataFrame,
    *,
    similarity_threshold: float = 0.90,
    maximum_features: int = 40,
    hard_maximum_features: int = 60,
    minimum_features: int = 15,
) -> ClusterSelection:
    """Keep a deterministic representative of each high-similarity training-only cluster."""
    if not 0 <= similarity_threshold <= 1 or not 1 <= minimum_features <= maximum_features:
        raise ValueError("feature-count or similarity thresholds are invalid")
    if maximum_features > hard_maximum_features:
        raise ValueError("maximum_features exceeds the unconditional hard maximum")
    required_candidates = {
        "factor_id",
        "factor_version",
        "status",
        "fdr_q_value",
        "signed_mean_rank_ic",
        "positive_year_count",
        "coverage",
        "conflict_rate",
    }
    _require_columns(candidates, required_candidates, "candidates")
    eligible = candidates.loc[candidates["status"] == "ELIGIBLE"].copy()
    if eligible.duplicated(["factor_id", "factor_version"]).any():
        raise ValueError("candidates contains duplicate factor identities")
    keys = {
        (str(row["factor_id"]), _to_int(row["factor_version"]))
        for row in cast(list[dict[str, object]], eligible.to_dict(orient="records"))
    }
    parents = {key: key for key in keys}

    def find(key: FactorKey) -> FactorKey:
        while parents[key] != key:
            parents[key] = parents[parents[key]]
            key = parents[key]
        return key

    def union(left: FactorKey, right: FactorKey) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parents[max(left_root, right_root)] = min(left_root, right_root)

    similarity_required = {
        "factor_id_left",
        "factor_version_left",
        "factor_id_right",
        "factor_version_right",
        "median_daily_abs_spearman",
        "status",
    }
    _require_columns(similarity, similarity_required, "similarity")
    qualifying_similarity = similarity.loc[
        (similarity["status"] == "PASS")
        & (
            pd.to_numeric(similarity["median_daily_abs_spearman"], errors="coerce")
            >= similarity_threshold
        )
    ]
    for row in cast(list[dict[str, object]], qualifying_similarity.to_dict(orient="records")):
        left = (str(row["factor_id_left"]), _to_int(row["factor_version_left"]))
        right = (str(row["factor_id_right"]), _to_int(row["factor_version_right"]))
        if left in parents and right in parents:
            union(left, right)

    grouped: dict[FactorKey, list[FactorKey]] = {}
    for key in keys:
        grouped.setdefault(find(key), []).append(key)
    clusters = tuple(
        sorted(
            (
                FactorCluster(cluster_identity(members), tuple(sorted(members)))
                for members in grouped.values()
            ),
            key=lambda cluster: cluster.cluster_id,
        )
    )
    indexed = eligible.set_index(["factor_id", "factor_version"], drop=False)
    representatives: list[FactorKey] = []
    for members in grouped.values():
        representatives.append(
            min(members, key=lambda key: _representative_sort_key(_candidate_row(indexed, key)))
        )
    representatives.sort(key=lambda key: _representative_sort_key(_candidate_row(indexed, key)))
    selected = tuple(representatives[:maximum_features])
    if len(selected) < minimum_features:
        return ClusterSelection(selected, "DRAFT", "INSUFFICIENT_FEATURES", clusters)
    return ClusterSelection(selected, "READY", None, clusters)


def _scope_to_train(frame: pd.DataFrame, train_start: date, train_end: date) -> pd.DataFrame:
    scoped = frame.copy(deep=True)
    scoped["signal_date"] = pd.to_datetime(scoped["signal_date"], errors="raise").dt.date
    return scoped.loc[
        (scoped["signal_date"] >= train_start) & (scoped["signal_date"] <= train_end)
    ].copy()


def _positive_year_count(daily: pd.DataFrame, direction: int) -> int:
    if direction == 0 or daily.empty:
        return 0
    calendar_year = pd.to_datetime(daily["signal_date"], errors="raise").dt.year
    signed = daily["rank_ic"] * direction
    return int((signed.groupby(calendar_year).mean() > 0).sum())


def _selection_status(
    row: Mapping[str, object], q_value: float, rules: FactorSelectionRules
) -> str:
    if not bool(row["structural_eligible"]):
        return "STRUCTURAL_DQ_FAILED"
    if _to_int(row["valid_day_count"]) < rules.minimum_valid_days:
        return "INSUFFICIENT_DAILY_IC"
    if _to_int(row["frozen_direction"]) == 0 or _to_float(row["signed_mean_rank_ic"]) <= 0:
        return "NON_POSITIVE_MEAN_IC"
    if not math.isfinite(q_value) or q_value > rules.fdr_q_threshold:
        return "FDR_FAILED"
    if _to_int(row["positive_year_count"]) < rules.minimum_positive_years:
        return "INSUFFICIENT_POSITIVE_YEARS"
    if _to_int(row["positive_inner_count"]) < rules.minimum_positive_inner_folds:
        return "INSUFFICIENT_POSITIVE_INNER_FOLDS"
    return "ELIGIBLE"


def _representative_sort_key(row: pd.Series) -> tuple[float, float, int, float, float, str, int]:
    return (
        _to_float(row["fdr_q_value"]),
        -_to_float(row["signed_mean_rank_ic"]),
        -_to_int(row["positive_year_count"]),
        -_to_float(row["coverage"]),
        _to_float(row["conflict_rate"]),
        str(row["factor_id"]),
        _to_int(row["factor_version"]),
    )


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"{name} is missing columns: {missing}")


def _candidate_row(indexed: pd.DataFrame, key: FactorKey) -> pd.Series:
    return cast(pd.Series, indexed.loc[key])


def _to_int(value: object) -> int:
    return int(cast(Any, value))


def _to_float(value: object) -> float:
    return float(cast(Any, value))
