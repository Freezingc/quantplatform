"""Content-addressed, explicitly label-free manifests for daily inference inputs."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date

import pandas as pd

from research_v2.datasets.manifest import build_dataset_manifest


@dataclass(frozen=True)
class InferenceManifest:
    manifest_id: str
    content_hash: str
    signal_date: date
    row_count: int
    contains_labels: bool


def build_inference_manifest(
    frame: pd.DataFrame,
    *,
    model_id: str,
    universe_id: str,
    factor_set_id: str,
    preprocessor_hash: str,
    input_versions: Mapping[str, str],
) -> InferenceManifest:
    """Hash keyed inference content and provenance while rejecting all target-like columns."""
    if not all((model_id, universe_id, factor_set_id, preprocessor_hash)):
        raise ValueError("inference manifest identity fields cannot be empty")
    forbidden = [
        column
        for column in frame.columns
        if any(token in str(column).lower() for token in ("label", "target", "raw_return"))
    ]
    if forbidden:
        raise ValueError(f"inference frame must not contain label columns: {sorted(forbidden)}")
    if "signal_date" not in frame.columns:
        raise ValueError("inference frame is missing signal_date")
    dates = pd.to_datetime(frame["signal_date"], errors="raise").dt.date
    if dates.nunique() != 1:
        raise ValueError("inference manifest requires exactly one signal_date")
    signal_date = dates.iloc[0]
    content = frame.copy(deep=True)
    content["signal_date"] = dates
    dataset_manifest = build_dataset_manifest(
        content,
        business_keys=("signal_date", "ts_code"),
        metadata={
            "factor_set_id": factor_set_id,
            "input_versions": dict(input_versions),
            "model_id": model_id,
            "preprocessor_hash": preprocessor_hash,
            "universe_id": universe_id,
        },
    )
    return InferenceManifest(
        manifest_id=dataset_manifest.dataset_id,
        content_hash=dataset_manifest.content_hash,
        signal_date=signal_date,
        row_count=dataset_manifest.row_count,
        contains_labels=False,
    )
