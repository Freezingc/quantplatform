"""Normalize registered YesLab long-form factor observations."""

from __future__ import annotations

import hashlib
import math
from typing import Any, cast

import pandas as pd

from research_v2.contracts.factor import FactorDefinition, FactorType


class SourceFactorSchemaError(ValueError):
    """Raised when a factor source batch does not satisfy the registered contract."""


def _source_row_id(
    source_table: str,
    ts_code: str,
    observation_date: object,
    factor_name: str,
    raw_value: object,
    occurrence: int,
) -> str:
    payload = f"{source_table}|{ts_code}|{observation_date}|{factor_name}|{raw_value}|{occurrence}"
    return hashlib.sha256(payload.encode()).hexdigest()


def normalize_technical_factor_source(
    source: pd.DataFrame,
    definition: FactorDefinition,
    *,
    source_factor_name: str,
    source_watermark_id: str,
    include_source_row_id: bool = True,
) -> pd.DataFrame:
    """Map a single registered YesLab technical factor into canonical long-form rows."""
    if definition.factor_type is not FactorType.TECHNICAL:
        raise ValueError("technical source reader requires a TECHNICAL factor definition")
    required = {
        definition.source_code_column,
        definition.source_date_column,
        "factor_name",
        definition.source_value_column,
    }
    missing = sorted(required.difference(source.columns))
    if missing:
        raise SourceFactorSchemaError(f"factor source is missing columns: {missing}")
    selected = source.loc[source["factor_name"].astype(str) == source_factor_name].copy()
    selected["observation_date"] = pd.to_datetime(
        selected[definition.source_date_column], errors="raise"
    ).dt.date
    selected["raw_value"] = pd.to_numeric(
        selected[definition.source_value_column], errors="coerce"
    )
    selected["value_is_valid"] = selected["raw_value"].map(
        lambda value: pd.notna(value) and math.isfinite(float(value))
    )
    selected["invalid_reason"] = pd.NA
    selected.loc[selected["raw_value"].isna(), "invalid_reason"] = "MISSING_VALUE"
    selected.loc[
        selected["raw_value"].notna() & ~selected["value_is_valid"], "invalid_reason"
    ] = "NONFINITE_VALUE"
    selected = selected.sort_values(
        [definition.source_code_column, "observation_date", "raw_value"],
        kind="mergesort",
        na_position="last",
    ).reset_index(drop=True)
    if include_source_row_id:
        occurrences = selected.groupby(
            [definition.source_code_column, "observation_date", "factor_name", "raw_value"],
            dropna=False,
            sort=False,
        ).cumcount()
        source_row_ids: list[str] = []
        for row, occurrence in zip(
            cast(list[dict[str, Any]], selected.to_dict(orient="records")),
            occurrences.tolist(),
            strict=True,
        ):
            source_row_ids.append(
                _source_row_id(
                    definition.source_table,
                    str(row[definition.source_code_column]),
                    row["observation_date"],
                    source_factor_name,
                    row["raw_value"],
                    int(occurrence),
                )
            )
    else:
        # Preflight only needs a deterministic tie-break for duplicate rows;
        # it does not publish source revision IDs or quarantine payloads.
        source_row_ids = [f"preflight-{index:012d}" for index in range(len(selected))]
    return pd.DataFrame(
        {
            "ts_code": selected[definition.source_code_column],
            "factor_id": definition.factor_id,
            "factor_version": definition.factor_version,
            "observation_date": selected["observation_date"],
            "source_available_date": selected["observation_date"],
            "raw_value": selected["raw_value"],
            "value_is_valid": selected["value_is_valid"],
            "invalid_reason": selected["invalid_reason"],
            "source_row_id": source_row_ids,
            "source_table": definition.source_table,
            "source_watermark_id": source_watermark_id,
        }
    )
