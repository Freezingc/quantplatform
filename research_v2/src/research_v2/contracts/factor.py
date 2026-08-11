"""Immutable factor-definition contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum


class FactorType(StrEnum):
    TECHNICAL = "TECHNICAL"
    FUNDAMENTAL = "FUNDAMENTAL"
    ALTERNATIVE = "ALTERNATIVE"


class AvailabilityPolicy(StrEnum):
    SAME_DAY_CLOSE = "SAME_DAY_CLOSE"
    NEXT_TRADING_DAY_AFTER_ANNOUNCEMENT = "NEXT_TRADING_DAY_AFTER_ANNOUNCEMENT"
    EXPLICIT_AVAILABLE_DATE = "EXPLICIT_AVAILABLE_DATE"


class FactorStatus(StrEnum):
    DRAFT = "DRAFT"
    VALIDATED = "VALIDATED"
    ACTIVE = "ACTIVE"
    RETIRED = "RETIRED"


@dataclass(frozen=True)
class FactorDefinition:
    """Versioned factor semantic and source mapping, excluding mutable display metadata."""

    factor_id: str
    factor_version: int
    display_name: str
    source_database: str
    source_table: str
    source_value_column: str
    source_date_column: str
    source_code_column: str
    factor_type: FactorType
    frequency: str
    expected_direction: int
    availability_policy: AvailabilityPolicy
    max_staleness_calendar_days: int
    unit: str
    owner: str

    def __post_init__(self) -> None:
        if not self.factor_id or self.factor_version < 1 or not self.owner:
            raise ValueError("factor_id, positive factor_version, and owner are required")
        if self.source_database != "dfs://YesLab":
            raise ValueError("factor source_database must be dfs://YesLab")
        if not all(
            (
                self.source_table,
                self.source_value_column,
                self.source_date_column,
                self.source_code_column,
            )
        ):
            raise ValueError("factor source mapping fields cannot be empty")
        if self.expected_direction not in {-1, 0, 1}:
            raise ValueError("expected_direction must be -1, 0, or 1")
        if self.max_staleness_calendar_days < 0:
            raise ValueError("max_staleness_calendar_days cannot be negative")
        if self.factor_type is FactorType.TECHNICAL and self.max_staleness_calendar_days != 0:
            raise ValueError("technical factors must have zero staleness")
        if (
            self.factor_type is FactorType.FUNDAMENTAL
            and self.availability_policy
            is not AvailabilityPolicy.NEXT_TRADING_DAY_AFTER_ANNOUNCEMENT
        ):
            raise ValueError("fundamental factors require announcement-based availability")

    @property
    def definition_hash(self) -> str:
        payload = {
            "availability_policy": self.availability_policy.value,
            "expected_direction": self.expected_direction,
            "factor_id": self.factor_id,
            "factor_type": self.factor_type.value,
            "factor_version": self.factor_version,
            "frequency": self.frequency,
            "max_staleness_calendar_days": self.max_staleness_calendar_days,
            "source_code_column": self.source_code_column,
            "source_database": self.source_database,
            "source_date_column": self.source_date_column,
            "source_table": self.source_table,
            "source_value_column": self.source_value_column,
            "unit": self.unit,
        }
        canonical = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()
