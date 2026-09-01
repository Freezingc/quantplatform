"""Point-in-time label-free feature construction for one daily inference signal date."""

from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd
import torch

from research_v2.datasets.builder import FactorInput
from research_v2.datasets.preprocessing import FoldPreprocessor
from research_v2.models.base import FeatureBatch, PredictionInputBatch, TemporalFeatureBatch


def build_label_free_inference_frame(
    universe: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    factor_members: tuple[FactorInput, ...],
    signal_date: date,
) -> pd.DataFrame:
    """Join same-day available factors onto current universe members without target columns."""
    if not factor_members:
        raise ValueError("factor_members cannot be empty")
    if len({member.feature_name for member in factor_members}) != len(factor_members):
        raise ValueError("factor feature names must be unique")
    _require_columns(universe, {"signal_date", "ts_code"}, "universe")
    _require_columns(
        factors,
        {"signal_date", "ts_code", "factor_id", "factor_version", "value", "available_date"},
        "factors",
    )
    keys = ["signal_date", "ts_code"]
    normalized_universe = universe.copy(deep=True)
    normalized_universe["signal_date"] = pd.to_datetime(
        normalized_universe["signal_date"], errors="raise"
    ).dt.date
    normalized_universe = normalized_universe.loc[normalized_universe["signal_date"] == signal_date]
    if "is_member" in normalized_universe.columns:
        normalized_universe = normalized_universe.loc[normalized_universe["is_member"].astype(bool)]
    if normalized_universe.empty:
        raise ValueError("inference universe is empty")
    if normalized_universe.duplicated(keys).any():
        raise ValueError("universe contains duplicate business keys")
    result = normalized_universe.loc[:, keys].copy()
    normalized_factors = factors.copy(deep=True)
    normalized_factors["signal_date"] = pd.to_datetime(
        normalized_factors["signal_date"], errors="raise"
    ).dt.date
    normalized_factors["available_date"] = pd.to_datetime(
        normalized_factors["available_date"], errors="raise"
    ).dt.date
    for member in factor_members:
        rows = normalized_factors.loc[
            (normalized_factors["signal_date"] == signal_date)
            & (normalized_factors["factor_id"].astype(str) == member.factor_id)
            & (normalized_factors["factor_version"].astype(int) == member.factor_version),
            keys + ["value", "available_date"],
        ].copy()
        if rows.duplicated(keys).any():
            raise ValueError(f"factor has duplicate business keys: {member.factor_id}")
        if rows["available_date"].isna().any():
            raise ValueError(f"factor has null available_date: {member.factor_id}")
        if (rows["available_date"] > signal_date).any():
            raise ValueError(f"factor is not available by signal date: {member.factor_id}")
        result = result.merge(
            rows.loc[:, keys + ["value"]].rename(columns={"value": member.feature_name}),
            on=keys,
            how="left",
            validate="one_to_one",
        )
    return result.sort_values(keys, kind="mergesort", ignore_index=True)


def build_label_free_inference_batch(
    universe: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    factor_members: tuple[FactorInput, ...],
    signal_date: date,
    preprocessor: FoldPreprocessor,
) -> FeatureBatch:
    """Transform one label-free factor frame with frozen train state into a keyed model batch."""
    frame = build_label_free_inference_frame(
        universe,
        factors,
        factor_members=factor_members,
        signal_date=signal_date,
    )
    factor_order = tuple(member.feature_name for member in factor_members)
    transformed = preprocessor.transform(frame.loc[:, list(factor_order)])
    feature_tensor = torch.tensor(
        transformed.values.to_numpy(dtype=float),
        dtype=torch.float32,
    ).unsqueeze(0)
    missing_tensor = torch.tensor(transformed.missing_mask.to_numpy(dtype=bool)).unsqueeze(0)
    return FeatureBatch(
        signal_dates=(signal_date,),
        ts_codes=(tuple(frame["ts_code"].astype(str)),),
        features=feature_tensor,
        feature_missing=missing_tensor,
        stock_mask=torch.ones((1, len(frame)), dtype=torch.bool),
    )


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    if missing := sorted(required.difference(frame.columns)):
        raise ValueError(f"{name} is missing columns: {missing}")


_CROSS_SECTIONAL_FAMILIES = frozenset({"RESIDUAL_MLP", "SET_ATTENTION"})
_TEMPORAL_FAMILIES = frozenset({"TEMPORAL_GRU"})


def build_inference_batch_for_family(
    model_family: str,
    universe: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    factor_members: tuple[FactorInput, ...],
    signal_date: date,
    preprocessor: FoldPreprocessor,
    history_length: int = 20,
    minimum_valid_history_length: int = 15,
) -> PredictionInputBatch:
    """Dispatch label-free inference batch assembly by the frozen model family.

    Cross-sectional families receive one same-day keyed frame; temporal families
    receive one causal left-aligned history window per security.  Unknown
    families fail closed instead of falling back to a default shape.
    """
    if model_family in _CROSS_SECTIONAL_FAMILIES:
        return build_label_free_inference_batch(
            universe,
            factors,
            factor_members=factor_members,
            signal_date=signal_date,
            preprocessor=preprocessor,
        )
    if model_family in _TEMPORAL_FAMILIES:
        return build_label_free_temporal_inference_batch(
            universe,
            factors,
            factor_members=factor_members,
            signal_date=signal_date,
            preprocessor=preprocessor,
            history_length=history_length,
            minimum_valid_history_length=minimum_valid_history_length,
        )
    raise ValueError(f"unknown model family for inference batch assembly: {model_family}")


def build_label_free_temporal_inference_batch(
    universe: pd.DataFrame,
    factors: pd.DataFrame,
    *,
    factor_members: tuple[FactorInput, ...],
    signal_date: date,
    preprocessor: FoldPreprocessor,
    history_length: int = 20,
    minimum_valid_history_length: int = 15,
) -> TemporalFeatureBatch:
    """Assemble one causal left-aligned history batch for temporal model inference.

    Each universe member's window is its most recent ``history_length`` observed
    factor rows at or before ``signal_date`` (oldest first).  Securities with fewer
    than ``minimum_valid_history_length`` observed steps cannot be scored and are
    excluded; if none remain the call fails closed.  Rows whose ``available_date``
    is after the signal date are rejected before assembly.
    """
    if history_length < 1:
        raise ValueError("history_length must be at least one")
    if not 1 <= minimum_valid_history_length <= history_length:
        raise ValueError("minimum_valid_history_length must be within [1, history_length]")
    current_frame = build_label_free_inference_frame(
        universe,
        factors,
        factor_members=factor_members,
        signal_date=signal_date,
    )
    factor_order = tuple(member.feature_name for member in factor_members)
    codes = tuple(current_frame["ts_code"].astype(str))

    _require_columns(
        factors,
        {"signal_date", "ts_code", "factor_id", "factor_version", "value", "available_date"},
        "factors",
    )
    normalized = factors.copy(deep=True)
    normalized["signal_date"] = pd.to_datetime(normalized["signal_date"], errors="raise").dt.date
    normalized["available_date"] = pd.to_datetime(
        normalized["available_date"], errors="raise"
    ).dt.date
    normalized = normalized.loc[normalized["signal_date"] <= signal_date]
    if (normalized["available_date"] > signal_date).any():
        raise ValueError("factor is not available by signal date: history rows")
    if normalized["available_date"].isna().any() or normalized["signal_date"].isna().any():
        raise ValueError("factor history rows cannot contain null dates")

    member_mask = pd.Series(False, index=normalized.index)
    for member in factor_members:
        member_mask |= (normalized["factor_id"].astype(str) == member.factor_id) & (
            normalized["factor_version"].astype(int) == member.factor_version
        )
    normalized = normalized.loc[member_mask]

    observations_by_code: dict[str, list[date]] = {}
    for day, code in normalized.loc[:, ["signal_date", "ts_code"]].itertuples(
        index=False, name=None
    ):
        observations_by_code.setdefault(str(code), []).append(day)

    eligible: list[str] = []
    windows: dict[str, tuple[date, ...]] = {}
    for code in codes:
        history_dates = tuple(sorted(set(observations_by_code.get(code, ()))))
        if len(history_dates) < minimum_valid_history_length:
            continue
        eligible.append(code)
        windows[code] = history_dates[-history_length:]
    if not eligible:
        raise ValueError("no security has sufficient history for temporal inference")

    transformed_by_step: dict[tuple[str, date], tuple[np.ndarray, np.ndarray]] = {}
    for code, history_dates in windows.items():
        for day in history_dates:
            if (code, day) in transformed_by_step:
                continue
            day_rows = normalized.loc[normalized["signal_date"] == day]
            day_frame = day_rows.pivot_table(
                index="ts_code",
                columns="factor_id",
                values="value",
                aggfunc="first",
            )
            named = pd.DataFrame(index=day_frame.index)
            for member in factor_members:
                column = member.factor_id
                named[member.feature_name] = (
                    day_frame[column] if column in day_frame.columns else np.nan
                )
            transformed = preprocessor.transform(named.loc[:, list(factor_order)])
            values = transformed.values.to_numpy(dtype=float)
            missing = transformed.missing_mask.to_numpy(dtype=bool)
            for position, code in enumerate(day_frame.index.astype(str)):
                transformed_by_step[(code, day)] = (values[position], missing[position])

    stock_count = len(eligible)
    factor_count = len(factor_order)
    features = np.zeros((1, stock_count, history_length, factor_count), dtype=np.float32)
    missing = np.zeros((1, stock_count, history_length, factor_count), dtype=bool)
    history_mask = np.zeros((1, stock_count, history_length), dtype=bool)
    for position, code in enumerate(eligible):
        for step, day in enumerate(windows[code]):
            values, mask = transformed_by_step[(code, day)]
            features[0, position, step] = values.astype(np.float32)
            missing[0, position, step] = mask
            history_mask[0, position, step] = True

    return TemporalFeatureBatch(
        signal_dates=(signal_date,),
        ts_codes=(tuple(eligible),),
        features=torch.from_numpy(features),
        feature_missing=torch.from_numpy(missing),
        stock_mask=torch.ones((1, stock_count), dtype=torch.bool),
        history_mask=torch.from_numpy(history_mask),
    )
