"""Build explicit V2 factor-catalog entries from a frozen YesLab metadata snapshot."""

from __future__ import annotations

import hashlib
import re
from typing import Any, cast

import pandas as pd


class CatalogBuildError(ValueError):
    """Raised when source metadata cannot be turned into an explicit catalog."""


_REQUIRED_COLUMNS = frozenset(
    {"factor_id", "factor_name", "creator", "update_frequency", "store_factor"}
)
_ALLOWED_TABLES = frozenset(f"factor_data{index}" for index in range(1, 12))
#: C1.1 forbids accepting factors by runtime wildcard. A glob in a factor name would put an entry
#: into the catalog that describes no single factor, while the read path filters
#: ``where factor_name in [...]`` -- so it would select nothing, or something nobody defined.
_WILDCARD_PATTERN = re.compile(r"[*?\[\]]")


def build_yeslab_catalog(metadata: pd.DataFrame) -> list[dict[str, object]]:
    """Create one explicit technical-factor entry for every frozen metadata row."""
    missing = sorted(_REQUIRED_COLUMNS.difference(metadata.columns))
    if missing:
        raise CatalogBuildError(f"factor metadata is missing columns: {missing}")
    if metadata["factor_id"].isna().any() or metadata["factor_id"].duplicated().any():
        raise CatalogBuildError("factor metadata contains duplicate or null factor_id")
    entries: list[dict[str, object]] = []
    for row in cast(
        list[dict[str, Any]], metadata.loc[:, sorted(_REQUIRED_COLUMNS)].to_dict(orient="records")
    ):
        source_table = str(row["store_factor"])
        if source_table not in _ALLOWED_TABLES:
            raise CatalogBuildError(f"unmapped source factor table: {source_table}")
        factor_name = str(row["factor_name"]).strip()
        owner = str(row["creator"]).strip()
        if not factor_name or factor_name.lower() == "nan" or not owner or owner.lower() == "nan":
            raise CatalogBuildError("factor metadata has an empty factor_name or creator")
        if _WILDCARD_PATTERN.search(factor_name):
            raise CatalogBuildError(
                f"factor name must be explicit, not a runtime wildcard: {factor_name}"
            )
        factor_number = int(row["factor_id"])
        frequency = str(row["update_frequency"]).strip().upper()
        if frequency != "DAILY":
            raise CatalogBuildError(f"unsupported non-daily factor frequency: {frequency}")
        entries.append(
            {
                "factor_id": f"yeslab_{factor_number}",
                "factor_version": 1,
                "display_name": factor_name,
                "source_database": "dfs://YesLab",
                "source_table": source_table,
                "source_factor_name": factor_name,
                "source_value_column": "value",
                "source_date_column": "trade_date",
                "source_code_column": "ts_code",
                "factor_type": "TECHNICAL",
                "frequency": "DAILY",
                "expected_direction": 0,
                "availability_policy": "SAME_DAY_CLOSE",
                "max_staleness_calendar_days": 0,
                "unit": "UNKNOWN",
                "owner": owner,
            }
        )
    return sorted(entries, key=lambda entry: str(entry["factor_id"]))


def build_yeslab_catalog_document(metadata: pd.DataFrame) -> dict[str, object]:
    """Return explicit entries together with a reproducible source-metadata snapshot hash."""
    entries = build_yeslab_catalog(metadata)
    normalized_metadata = metadata.loc[:, sorted(_REQUIRED_COLUMNS)].copy()
    canonical = normalized_metadata.sort_values("factor_id").to_csv(index=False)
    return {
        "metadata_snapshot_hash": hashlib.sha256(canonical.encode()).hexdigest(),
        "factors": entries,
    }
