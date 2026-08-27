"""Content-addressed manifests for immutable research dataset snapshots."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class DatasetManifest:
    """The immutable identity and provenance of one assembled dataset."""

    dataset_id: str
    content_hash: str
    row_count: int
    business_keys: tuple[str, ...]
    column_order: tuple[str, ...]
    metadata: Mapping[str, object]
    split_content_hashes: tuple[tuple[str, str], ...] = ()


def build_dataset_manifest(
    frame: pd.DataFrame,
    *,
    business_keys: tuple[str, ...],
    metadata: Mapping[str, object],
    split_frames: Mapping[str, pd.DataFrame] | None = None,
) -> DatasetManifest:
    """Hash dataset contents after validating and ordering by immutable business keys.

    Column order remains significant because it defines the model input order.  Row order
    is deliberately not significant: the same keyed observations must have one identity
    regardless of database or parquet read ordering.
    """
    if not business_keys or len(set(business_keys)) != len(business_keys):
        raise ValueError("business_keys must be non-empty and unique")
    missing_keys = set(business_keys).difference(frame.columns)
    if missing_keys:
        raise ValueError(f"dataset is missing business-key columns: {sorted(missing_keys)}")
    if frame.loc[:, list(business_keys)].isna().any().any():
        raise ValueError("business keys cannot contain null values")
    if frame.duplicated(subset=list(business_keys)).any():
        raise ValueError("dataset contains duplicate business keys")

    ordered = frame.sort_values(list(business_keys), kind="mergesort", ignore_index=True)
    column_order = tuple(str(column) for column in ordered.columns)
    if len(set(column_order)) != len(column_order):
        raise ValueError("dataset columns must be unique")
    canonical_metadata = _canonicalize(metadata)
    if not isinstance(canonical_metadata, dict):
        raise ValueError("metadata must be a string-keyed mapping")
    split_content_hashes = _build_split_content_hashes(
        split_frames,
        business_keys=business_keys,
        metadata=metadata,
    )
    payload = {
        "business_keys": list(business_keys),
        "columns": [
            {"name": column, "dtype": str(ordered[column].dtype)} for column in column_order
        ],
        "metadata": canonical_metadata,
        "rows": [
            [_canonicalize(value) for value in row]
            for row in ordered.loc[:, list(column_order)].itertuples(index=False, name=None)
        ],
    }
    if split_content_hashes:
        payload["split_content_hashes"] = dict(split_content_hashes)
    content_hash = hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()
    return DatasetManifest(
        dataset_id=content_hash,
        content_hash=content_hash,
        row_count=len(ordered),
        business_keys=business_keys,
        column_order=column_order,
        metadata=canonical_metadata,
        split_content_hashes=split_content_hashes,
    )


def build_split_dataset_manifest(
    splits: Mapping[str, pd.DataFrame],
    *,
    business_keys: tuple[str, ...],
    metadata: Mapping[str, object],
) -> DatasetManifest:
    """Build a manifest whose identity includes each row's assigned dataset split."""
    frames = validate_dataset_splits(splits)
    combined = pd.concat(frames, ignore_index=True)
    return build_dataset_manifest(
        combined,
        business_keys=business_keys,
        metadata=metadata,
        split_frames=splits,
    )


def _build_split_content_hashes(
    splits: Mapping[str, pd.DataFrame] | None,
    *,
    business_keys: tuple[str, ...],
    metadata: Mapping[str, object],
) -> tuple[tuple[str, str], ...]:
    if splits is None:
        return ()
    validate_dataset_splits(splits)
    return tuple(
        (
            split_name,
            build_dataset_manifest(
                splits[split_name],
                business_keys=business_keys,
                metadata=metadata,
            ).content_hash,
        )
        for split_name in sorted(splits)
    )


def validate_dataset_splits(splits: Mapping[str, pd.DataFrame]) -> list[pd.DataFrame]:
    """Validate and return non-empty, schema-aligned train/validation/test frames."""
    if not splits:
        raise ValueError("at least one split is required")
    frames: list[pd.DataFrame] = []
    expected_columns: tuple[object, ...] | None = None
    for split_name in sorted(splits):
        if split_name not in {"train", "validation", "test"}:
            raise ValueError(f"unsupported dataset split: {split_name}")
        frame = splits[split_name]
        if not isinstance(frame, pd.DataFrame):
            raise TypeError(f"dataset split must be a pandas DataFrame: {split_name}")
        if frame.empty:
            raise ValueError(f"cannot write an empty dataset split: {split_name}")
        if expected_columns is None:
            expected_columns = tuple(frame.columns)
        elif tuple(frame.columns) != expected_columns:
            raise ValueError("dataset splits must have identical columns in identical order")
        frames.append(frame)
    return frames


def _canonicalize(value: object) -> object:
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _canonicalize(value.item())
    if isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if math.isinf(value):
            return {"non_finite": "positive" if value > 0 else "negative"}
        return value
    if isinstance(value, (date, datetime, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, pd.Timedelta):
        return value.isoformat()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, Mapping):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("metadata mapping keys must be strings")
        return {key: _canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_canonicalize(item) for item in value]
    raise ValueError(f"unsupported manifest value type: {type(value).__name__}")


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
