"""Content-addressed manifests for immutable research dataset snapshots."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd

_ROW_CHUNK_ROWS = 512
"""How many canonicalised rows are held at once while the content hash is streamed.

The digest cannot depend on this. ``json.dumps`` with ``separators=(",", ":")`` encodes a list as
``"["``, then the elements joined by ``","``, then ``"]"``, so encoding a chunk and dropping its
outer brackets is byte-identical to encoding those elements in place. That claim is pinned by
``test_the_digest_does_not_depend_on_the_chunking``, because getting a separator wrong here would
silently give every dataset built afterwards a new identity while two calls of the new code still
agreed with each other.

The value is small because the peak is linear in it and a chunk larger than the frame degenerates
into the old behaviour plus extra copies of the text: measured on a 4,000-row frame, 512 rows per
chunk peaks at 2.00x the frame's own deep size, 2,048 at 4.17x and 8,192 at 7.02x, all four with an
identical digest. At 512 the chunk itself is a few hundred kilobytes even at 60 factors, and the
extra ``json.dumps`` calls cost nothing next to canonicalising every cell.
"""


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
    content_hash = _hash_payload(
        ordered,
        business_keys=business_keys,
        column_order=column_order,
        canonical_metadata=canonical_metadata,
        split_content_hashes=split_content_hashes,
    )
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


def _hash_payload(
    ordered: pd.DataFrame,
    *,
    business_keys: tuple[str, ...],
    column_order: tuple[str, ...],
    canonical_metadata: object,
    split_content_hashes: tuple[tuple[str, str], ...],
) -> str:
    """Hash the payload's canonical JSON without ever holding that JSON.

    The bytes are the ones the previous implementation hashed, which is the whole requirement: this
    hash is the ``dataset_id``, so a changed byte would orphan every published artifact. The parts
    are written in ``sort_keys`` order -- ``business_keys``, ``columns``, ``metadata``, ``rows``,
    then ``split_content_hashes`` when present -- each encoded with the same separators and the
    rows in bounded chunks, so the concatenation equals
    ``json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)``.

    What this removes is measured, not speculative: on the published F1 artifact, building the
    manifest for 3,798,054 rows was 185.7 s of a 188.9 s ``load_split`` and lifted peak RSS from
    3.01 GiB to 8.62 GiB, because it held one Python object per cell plus the entire JSON text.
    """
    hasher = hashlib.sha256()

    def update(text: str) -> None:
        hasher.update(text.encode("utf-8"))

    update('{"business_keys":')
    update(_canonical_json(list(business_keys)))
    update(',"columns":')
    update(
        _canonical_json(
            [{"name": column, "dtype": str(ordered[column].dtype)} for column in column_order]
        )
    )
    update(',"metadata":')
    update(_canonical_json(canonical_metadata))
    update(',"rows":[')
    separator = ""
    for chunk in _canonical_row_chunks(ordered, column_order):
        update(separator)
        # The chunk is encoded as a JSON array; the rows list supplies the outer brackets, so only
        # the elements between them belong here.
        update(_canonical_json(chunk)[1:-1])
        separator = ","
    update("]")
    if split_content_hashes:
        update(',"split_content_hashes":')
        update(_canonical_json(dict(split_content_hashes)))
    update("}")
    return hasher.hexdigest()


def _canonical_row_chunks(
    ordered: pd.DataFrame, column_order: tuple[str, ...]
) -> Iterator[list[list[object]]]:
    """Yield canonicalised rows in bounded chunks, so the peak does not scale with the dataset."""
    chunk_rows = max(1, int(_ROW_CHUNK_ROWS))
    rows = ordered.loc[:, list(column_order)].itertuples(index=False, name=None)
    chunk: list[list[object]] = []
    for row in rows:
        chunk.append([_canonicalize(value) for value in row])
        if len(chunk) >= chunk_rows:
            yield chunk
            chunk = []
    if chunk:
        yield chunk
