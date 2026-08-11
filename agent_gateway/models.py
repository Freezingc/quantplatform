from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class FieldSpec:
    name: str
    table: str
    category: str = "unknown"
    frequency: str = "daily"
    description: str = ""
    aliases: tuple[str, ...] = ()
    expression_enabled: bool = True
    coverage: str | None = None
    missing_rate: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FactorCandidate:
    factor_name: str
    hypothesis: str
    required_fields: list[str]
    expression: str | None
    notes: str = ""
    source: str = "unknown"
    missing_fields: list[str] = field(default_factory=list)
    confidence: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ValidationResult:
    ok: bool
    expression: str
    used_fields: list[str] = field(default_factory=list)
    missing_fields: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
