"""Point-in-time label-free feature construction for one daily inference signal date."""

from __future__ import annotations

from datetime import date

import pandas as pd
import torch

from research_v2.datasets.builder import FactorInput
from research_v2.datasets.preprocessing import FoldPreprocessor
from research_v2.models.base import FeatureBatch


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
