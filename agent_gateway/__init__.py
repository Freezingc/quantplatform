"""Agent-facing research gateway for Quant-Platform-Prod.

This package adds a thin, production-safe adaptation layer around the existing
factor database, factor evaluation, and calculator workflow. It is intentionally
kept separate from the legacy scripts so it can be adopted incrementally.
"""

from .dsl import ExpressionEngine
from .field_registry import FieldRegistry
from .ingest import FactorSourceIngestor
from .models import FactorCandidate, FieldSpec, ValidationResult
from .promote import CalculatorPromoter
from .research_log import ResearchLog

__all__ = [
    "CalculatorPromoter",
    "ExpressionEngine",
    "FactorCandidate",
    "FactorSourceIngestor",
    "FieldRegistry",
    "FieldSpec",
    "ResearchLog",
    "ValidationResult",
]
