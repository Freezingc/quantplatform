"""Immutable contracts for batches extracted from external sources."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime

import pandas as pd


@dataclass(frozen=True)
class SourceBatch:
    """A source extract bound to a schema fingerprint and watermark."""

    frame: pd.DataFrame
    source_name: str
    table_name: str
    watermark_id: str
    max_observation_date: date
    source_row_count: int
    schema_fingerprint: str
    extracted_at: datetime
