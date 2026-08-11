"""Small, lineage-light contracts for latest-only compact tables."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class CompactManifestStatus(StrEnum):
    """Visibility state of one compact batch."""

    STAGED = "STAGED"
    PUBLISHED = "PUBLISHED"
    RETIRED = "RETIRED"


_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def compute_batch_id(object_name: str, logical_partition: str, source_data_version: str) -> int:
    """Return a stable positive signed 64-bit identity for one source batch."""
    if not all(value.strip() for value in (object_name, logical_partition, source_data_version)):
        raise ValueError("batch identity fields cannot be empty")
    payload = "|".join((object_name, logical_partition, source_data_version)).encode()
    value = int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)
    return value or 1


@dataclass(frozen=True)
class LatestManifestEntry:
    """One compact batch publication record."""

    object_name: str
    logical_partition: str
    batch_id: int
    source_object: str
    source_data_version: str
    published_at: datetime
    row_count: int
    business_hash: str
    status: CompactManifestStatus

    def __post_init__(self) -> None:
        if not self.object_name.strip() or not self.logical_partition.strip():
            raise ValueError("manifest object and partition cannot be empty")
        if self.batch_id <= 0:
            raise ValueError("batch_id must be positive")
        if not self.source_object.strip() or not self.source_data_version.strip():
            raise ValueError("manifest source identity cannot be empty")
        if self.row_count < 1:
            raise ValueError("row_count must be positive")
        if not _SHA256.fullmatch(self.business_hash):
            raise ValueError("business_hash must be a lowercase SHA-256 hex digest")
