from __future__ import annotations

import json
from pathlib import Path

try:
    from mcp.server.fastmcp import FastMCP
except ModuleNotFoundError as exc:  # pragma: no cover - exercised at runtime
    raise SystemExit(
        "Missing dependency: mcp. Install it with `pip install mcp` in the Python "
        "environment used by Claude Code."
    ) from exc

from .dsl import ExpressionEngine
from .factor_pipeline import run_factor_gate
from .field_registry import FieldRegistry
from .generate_field_reference import generate_field_reference
from .ingest import FactorSourceIngestor
from .knowledge_base import KnowledgeBase
from .promote import CalculatorPromoter
from .research_log import ResearchLog
from .validation import score_factor_summary


mcp = FastMCP(
    "quant-platform-agent-gateway",
    instructions=(
        "Agent-facing tools for Quant-Platform-Prod. Use list_available_fields "
        "and read_field_reference before generating factor expressions. "
        "Generated expressions must validate before promotion. After mining, "
        "similarity checks, or factor evaluation, distill reusable lessons into "
        "the research knowledge base."
    ),
)


def _json(payload) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


@mcp.tool()
def list_available_fields(query: str = "", expression_enabled_only: bool = True, limit: int = 200) -> str:
    """List database fields available for factor generation.

    Args:
        query: Optional substring to filter by field name, alias, table, category, or description.
        expression_enabled_only: If true, hide identifier/metadata-only fields.
        limit: Maximum number of fields to return.
    """
    registry = FieldRegistry.from_project()
    fields = registry.list_fields(expression_enabled=True if expression_enabled_only else None)
    if query:
        q = query.lower()
        fields = [
            f for f in fields
            if q in f.name.lower()
            or q in f.table.lower()
            or q in f.category.lower()
            or q in f.description.lower()
            or any(q in alias.lower() for alias in f.aliases)
        ]
    return _json({"count": len(fields), "fields": [f.to_dict() for f in fields[:limit]]})


@mcp.tool()
def read_field_reference(query: str = "", max_chars: int = 12000) -> str:
    """Read the generated daily-task database field reference document.

    Args:
        query: Optional substring filter; returns matching lines with nearby table headers.
        max_chars: Maximum characters returned.
    """
    path = generate_field_reference()
    text = path.read_text(encoding="utf-8")
    if not query:
        return text[:max_chars]

    q = query.lower()
    lines = text.splitlines()
    selected: list[str] = []
    current_header = ""
    for line in lines:
        if line.startswith("### "):
            current_header = line
        if q in line.lower():
            if current_header and (not selected or selected[-1] != current_header):
                selected.append(current_header)
            selected.append(line)
    return "\n".join(selected)[:max_chars]


@mcp.tool()
def validate_expression(expression: str) -> str:
    """Validate a DSL factor expression against the database field registry."""
    result = ExpressionEngine(FieldRegistry.from_project()).validate(expression)
    return _json(result.to_dict())


@mcp.tool()
def ingest_factor_source(source: str, source_type: str = "auto", max_candidates: int = 6) -> str:
    """Extract factor candidates from URL, PDF path, text file path, or raw text."""
    result = FactorSourceIngestor(FieldRegistry.from_project()).ingest_factor_source(
        source=source,
        source_type=source_type,
        max_candidates=max_candidates,
    )
    result["knowledge_asset"] = KnowledgeBase().distill_from_event(
        "factor_mining",
        {"source_type": source_type, "candidates": list(result.get("candidates", []))},
    )
    return _json(result)


@mcp.tool()
def generate_factor_expressions(description: str, max_candidates: int = 6) -> str:
    """Generate registry-constrained DSL factor candidates from a natural language description."""
    candidates = FactorSourceIngestor(FieldRegistry.from_project()).generate_factor_expressions(
        description,
        max_candidates=max_candidates,
    )
    result = {"description": description, "candidates": [c.to_dict() for c in candidates]}
    result["knowledge_asset"] = KnowledgeBase().distill_from_event("factor_mining", dict(result))
    return _json(result)


@mcp.tool()
def promote_to_calculator(
    factor_name: str,
    expression: str,
    creator: str = "agent_gateway",
    description: str = "Generated from DSL candidate. Review before production use.",
    lookback_days: int | None = None,
    write: bool = False,
) -> str:
    """Generate or write a production FactorCalculator draft from a validated expression.

    Keep write=false for review. Set write=true only after the user explicitly asks
    to create the file.
    """
    result = CalculatorPromoter(FieldRegistry.from_project()).promote_to_calculator(
        factor_name=factor_name,
        expression=expression,
        creator=creator,
        description=description,
        lookback_days=lookback_days,
        write=write,
    )
    return _json(result)


@mcp.tool()
def score_factor(summary_json: str, validation_json: str = "") -> str:
    """Score an existing factor evaluation summary JSON object.

    Args:
        summary_json: Existing factor metrics as JSON.
        validation_json: Optional rolling/placebo/out-of-sample validation JSON.
    """
    summary = json.loads(summary_json)
    validation = json.loads(validation_json) if validation_json else None
    result = score_factor_summary(summary, validation)
    result["knowledge_asset"] = KnowledgeBase().distill_from_event(
        "score_factor",
        {"summary": summary, "validation": validation, "evaluation": dict(result)},
    )
    return _json(result)


@mcp.tool()
def gate_factor_for_db_promotion(
    factor_name: str,
    method: str,
    lookback_days: int,
    formula: str = "",
    description: str = "",
    creator: str = "agent_gateway",
    category: str = "agent_generated",
    start_time: str = "2020.01.01",
    end_time: str = "",
    similarity_threshold: float = 0.7,
    min_abs_ic: float = 0.01,
    min_abs_icir: float = 0.3,
    commit: bool = False,
) -> str:
    """Calculate a candidate factor, run similarity and IC gates, and optionally insert it into DolphinDB.

    Set commit=false for dry-run validation. Set commit=true only after the user explicitly confirms DB insertion.
    """
    result = run_factor_gate(
        factor_name=factor_name,
        method=method,
        lookback_days=lookback_days,
        formula=formula,
        description=description,
        creator=creator,
        category=category,
        start_time=start_time,
        end_time=end_time or None,
        similarity_threshold=similarity_threshold,
        min_abs_ic=min_abs_ic,
        min_abs_icir=min_abs_icir,
        commit=commit,
    )
    return _json(result)


@mcp.tool()
def append_research_log(event_json: str) -> str:
    """Append an experiment/research event to the local JSONL research log."""
    event = json.loads(event_json)
    return _json(ResearchLog().append(event))


@mcp.tool()
def list_recent_research_logs(limit: int = 20) -> str:
    """List recent local agent research log entries."""
    return _json({"records": ResearchLog().list_recent(limit=limit)})


@mcp.tool()
def distill_research_knowledge(event_type: str, payload_json: str) -> str:
    """Distill a research event into rules/findings/failures knowledge assets.

    Args:
        event_type: One of factor_mining, factor_eval, similarity_check, or a custom event type.
        payload_json: Event payload JSON, such as similarity results or factor test output.
    """
    payload = json.loads(payload_json)
    return _json(KnowledgeBase().distill_from_event(event_type, payload))


@mcp.tool()
def list_research_knowledge(asset_type: str = "", limit: int = 20) -> str:
    """List recent structured research knowledge assets."""
    records = KnowledgeBase().list_recent(asset_type=asset_type or None, limit=limit)
    return _json({"records": records, "count": len(records)})


@mcp.tool()
def search_research_knowledge(query: str, asset_type: str = "", limit: int = 20) -> str:
    """Search structured research knowledge assets before repeating an experiment."""
    records = KnowledgeBase().search(query=query, asset_type=asset_type or None, limit=limit)
    return _json({"records": records, "count": len(records)})


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
