"""Golden-master equivalence and footprint for the dataset content hash.

``build_dataset_manifest`` decides what a dataset IS: its ``content_hash`` is the ``dataset_id``, so
a single changed byte would orphan every published artifact and every frozen chain that points at
one. That makes this the least forgiving place in the codebase to optimise, and the reason the
equivalence test below compares against a verbatim frozen copy of the previous implementation rather
than against a second call of the same one -- two calls of the same function agree no matter what it
does wrong.

The property being added is that the canonical JSON is fed to the hasher incrementally instead of
being materialised. Measured on the real F1 artifact, building the manifest for 3,798,054 rows took
185.7 s of a 188.9 s ``load_split`` and lifted peak RSS from 3.01 GiB to 8.62 GiB, because it held
every cell of the dataset as a Python object plus the whole JSON text at once. Streaming cannot
change the digest: ``json.dumps`` with ``separators=(",", ":")`` emits ``"{"`` then ``key:value``
pairs joined by ``","`` then ``"}"``, and a list as ``"["`` then elements joined by ``","`` then
``"]"``, so concatenating the encodings of the parts is byte-for-byte the encoding of the whole.
``test_the_digest_does_not_depend_on_the_chunking`` pins exactly that claim.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import tracemalloc
from collections.abc import Mapping, Sequence
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from research_v2.datasets import manifest as manifest_module
from research_v2.datasets.manifest import build_dataset_manifest, build_split_dataset_manifest


def _reference_canonicalize(value: object) -> object:
    """Verbatim copy of ``manifest._canonicalize`` as of c64cbac. DO NOT EDIT."""
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, np.generic):
        return _reference_canonicalize(value.item())
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
        return {key: _reference_canonicalize(value[key]) for key in sorted(value)}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_reference_canonicalize(item) for item in value]
    raise ValueError(f"unsupported manifest value type: {type(value).__name__}")


def _reference_canonical_json(value: object) -> str:
    """Verbatim copy of ``manifest._canonical_json`` as of c64cbac. DO NOT EDIT."""
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _reference_build_dataset_manifest(
    frame: pd.DataFrame,
    *,
    business_keys: tuple[str, ...],
    metadata: Mapping[str, object],
    split_frames: Mapping[str, pd.DataFrame] | None = None,
) -> manifest_module.DatasetManifest:
    """Verbatim copy of ``manifest.build_dataset_manifest`` as of c64cbac. DO NOT EDIT."""
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
    canonical_metadata = _reference_canonicalize(metadata)
    if not isinstance(canonical_metadata, dict):
        raise ValueError("metadata must be a string-keyed mapping")
    split_content_hashes = _reference_build_split_content_hashes(
        split_frames, business_keys=business_keys, metadata=metadata
    )
    payload = {
        "business_keys": list(business_keys),
        "columns": [
            {"name": column, "dtype": str(ordered[column].dtype)} for column in column_order
        ],
        "metadata": canonical_metadata,
        "rows": [
            [_reference_canonicalize(value) for value in row]
            for row in ordered.loc[:, list(column_order)].itertuples(index=False, name=None)
        ],
    }
    if split_content_hashes:
        payload["split_content_hashes"] = dict(split_content_hashes)
    content_hash = hashlib.sha256(
        _reference_canonical_json(payload).encode("utf-8")
    ).hexdigest()
    return manifest_module.DatasetManifest(
        dataset_id=content_hash,
        content_hash=content_hash,
        row_count=len(ordered),
        business_keys=business_keys,
        column_order=column_order,
        metadata=canonical_metadata,
        split_content_hashes=split_content_hashes,
    )


def _reference_build_split_content_hashes(
    splits: Mapping[str, pd.DataFrame] | None,
    *,
    business_keys: tuple[str, ...],
    metadata: Mapping[str, object],
) -> tuple[tuple[str, str], ...]:
    """Verbatim copy of ``manifest._build_split_content_hashes`` as of c64cbac. DO NOT EDIT."""
    if splits is None:
        return ()
    manifest_module.validate_dataset_splits(splits)
    return tuple(
        (
            split_name,
            _reference_build_dataset_manifest(
                splits[split_name], business_keys=business_keys, metadata=metadata
            ).content_hash,
        )
        for split_name in sorted(splits)
    )


_KEYS = ("signal_date", "ts_code")

_FLOATS = [
    0.0, -0.0, 1.0, -1.0, 0.1 + 0.2, 1 / 3, 5e-324, 1.7976931348623157e308,
    float("nan"), float("inf"), float("-inf"), 1234567890.0987654,
]


def _frame(rows: int, *, seed: int = 0, prefix: str = "") -> pd.DataFrame:
    """A frame whose every column takes a route through ``_canonicalize`` that matters.

    Business keys are unique and non-null because the builder rejects anything else; everything
    non-key is chosen to be a value whose canonical form is easy to get subtly wrong -- signed zero,
    both infinities, NaN, a float whose shortest repr is not its decimal literal, non-ASCII text,
    an empty string, dates as both ``datetime.date`` and ``pd.Timestamp``, and a nullable integer
    column holding ``pd.NA``.

    ``prefix`` keeps the splits' business keys disjoint, which the combined manifest requires: the
    three splits are concatenated before they are hashed, so shared keys would be duplicates.
    """
    rng = np.random.default_rng(seed)
    base = datetime(2024, 1, 1)
    days = [(base + timedelta(days=index // 5)).date() for index in range(rows)]
    frame = pd.DataFrame(
        {
            "signal_date": [days[index] for index in range(rows)],
            "ts_code": [f"{prefix}{index:06d}.SZ" for index in range(rows)],
            "float_value": [_FLOATS[index % len(_FLOATS)] for index in range(rows)],
            "noise": rng.standard_normal(rows),
            "int_value": np.arange(rows, dtype=np.int64),
            "nullable_int": pd.array(
                [None if index % 7 == 0 else index for index in range(rows)], dtype=pd.Int64Dtype()
            ),
            "bool_value": [bool(index % 2) for index in range(rows)],
            "text": [f"名称-{index}" if index % 3 else "" for index in range(rows)],
            "timestamp": pd.to_datetime(
                [f"2024-02-{1 + index % 28:02d} 03:04:05" for index in range(rows)]
            ),
        }
    )
    return frame


_METADATA: dict[str, object] = {
    "fold_id": "F1",
    "嵌套": {"b": 2, "a": [1, 2.5, None, True, "文本"]},
    "zeta": 0.1 + 0.2,
    "alpha": None,
    "count": 3,
    "path": Path("a/b/c.parquet"),
}


def test_streamed_manifest_hashes_identically_to_the_frozen_reference() -> None:
    frame = _frame(37, seed=1)

    single = build_dataset_manifest(frame, business_keys=_KEYS, metadata=_METADATA)
    wanted = _reference_build_dataset_manifest(frame, business_keys=_KEYS, metadata=_METADATA)
    assert single.content_hash == wanted.content_hash
    assert single.dataset_id == wanted.dataset_id
    assert single.row_count == wanted.row_count
    assert single.column_order == wanted.column_order
    assert single.metadata == wanted.metadata
    assert single.split_content_hashes == wanted.split_content_hashes

    splits = {
        "train": _frame(23, seed=2, prefix="1"),
        "validation": _frame(11, seed=3, prefix="2"),
        "test": _frame(7, seed=4, prefix="3"),
    }
    produced = build_split_dataset_manifest(splits, business_keys=_KEYS, metadata=_METADATA)
    expected = _reference_build_dataset_manifest(
        pd.concat(list(splits.values()), ignore_index=True),
        business_keys=_KEYS,
        metadata=_METADATA,
        split_frames=splits,
    )
    assert produced.content_hash == expected.content_hash
    assert produced.dataset_id == expected.dataset_id
    assert tuple(produced.split_content_hashes) == tuple(expected.split_content_hashes)
    assert produced.row_count == expected.row_count


def test_the_digest_does_not_depend_on_the_chunking(monkeypatch: pytest.MonkeyPatch) -> None:
    """Streaming is only safe if the chunk boundary cannot reach the digest.

    A row that straddles a boundary is the whole risk: get the separator wrong and every dataset
    built after the change gets a new identity, silently, while two calls of the new code still
    agree with each other.

    ``monkeypatch.setattr`` is used rather than direct assignment so that a missing chunk constant
    is an error here instead of a silent pass -- a test that cannot fail is worse than no test.
    """
    frame = _frame(29, seed=5)
    digests = set()
    for chunk_rows in (1, 2, 3, 7, 28, 29, 30, 4096):
        monkeypatch.setattr(manifest_module, "_ROW_CHUNK_ROWS", chunk_rows)
        digests.add(
            build_dataset_manifest(frame, business_keys=_KEYS, metadata=_METADATA).content_hash
        )
    assert len(digests) == 1, f"chunk size changed the digest: {sorted(digests)}"


def test_manifest_build_does_not_materialize_every_cell() -> None:
    """The traced peak must stay within a small multiple of the frame's own deep size.

    The frozen reference builds a Python list of one list per row holding one canonicalised object
    per cell, then the entire JSON text, so its peak is many times the frame it was handed. A
    streamed builder holds one chunk. The bound is expressed against the frame rather than in
    absolute bytes so it stays meaningful as the fixture grows.
    """
    frame = _frame(4000, seed=6)
    frame_bytes = int(frame.memory_usage(deep=True).sum())
    gc.collect()

    tracemalloc.start()
    try:
        build_dataset_manifest(frame, business_keys=_KEYS, metadata=_METADATA)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert peak < 4 * frame_bytes, (
        f"traced peak {peak:,} bytes is {peak / frame_bytes:.1f}x the {frame_bytes:,} bytes of the "
        "frame; the canonical JSON is still being materialised whole"
    )
