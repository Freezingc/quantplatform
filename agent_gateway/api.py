from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .dsl import ExpressionEngine
from .factor_pipeline import run_factor_gate
from .field_registry import FieldRegistry
from .ingest import FactorSourceIngestor
from .knowledge_base import KnowledgeBase
from .promote import CalculatorPromoter
from .research_log import ResearchLog
from .validation import score_factor_summary


app = FastAPI(title="Quant Platform Agent Gateway", version="0.1.0")


class IngestRequest(BaseModel):
    source: str
    source_type: str = "auto"
    max_candidates: int = Field(default=6, ge=1, le=20)


class GenerateRequest(BaseModel):
    description: str
    max_candidates: int = Field(default=6, ge=1, le=20)


class ValidateRequest(BaseModel):
    expression: str


class PromoteRequest(BaseModel):
    factor_name: str
    expression: str
    creator: str = "agent_gateway"
    description: str = "Generated from DSL candidate. Review before production use."
    lookback_days: int | None = None
    write: bool = False


class ScoreRequest(BaseModel):
    summary: dict[str, Any]
    validation: dict[str, Any] | None = None


class GateFactorRequest(BaseModel):
    factor_name: str
    method: str
    lookback_days: int
    formula: str = ""
    description: str = ""
    creator: str = "agent_gateway"
    category: str = "agent_generated"
    start_time: str = "2020.01.01"
    end_time: str | None = None
    create_date: str | None = None
    similarity_threshold: float = 0.7
    min_abs_ic: float = 0.01
    min_abs_icir: float = 0.3
    min_common_stocks: int = 50
    min_valid_dates: int = 20
    sample_similarity: bool = True
    sample_days: int = 60
    commit: bool = False
    env_path: str | None = None


class LogRequest(BaseModel):
    event: dict[str, Any]


class KnowledgeRequest(BaseModel):
    event_type: str
    payload: dict[str, Any]


@app.get("/api/v1/fields")
def list_available_fields(expression_enabled: bool | None = None) -> dict[str, Any]:
    registry = FieldRegistry.from_project()
    fields = registry.list_fields(expression_enabled=expression_enabled)
    return {"fields": [field.to_dict() for field in fields], "count": len(fields)}


@app.post("/api/v1/factor-sources/ingest")
def ingest_factor_source(req: IngestRequest) -> dict[str, Any]:
    payload = FactorSourceIngestor().ingest_factor_source(req.source, req.source_type, req.max_candidates)
    payload["knowledge_asset"] = KnowledgeBase().distill_from_event(
        "factor_mining",
        {"source_type": req.source_type, "candidates": list(payload.get("candidates", []))},
    )
    return payload


@app.post("/api/v1/factors/generate-expressions")
def generate_factor_expressions(req: GenerateRequest) -> dict[str, Any]:
    candidates = FactorSourceIngestor().generate_factor_expressions(req.description, req.max_candidates)
    payload = {"description": req.description, "candidates": [candidate.to_dict() for candidate in candidates]}
    payload["knowledge_asset"] = KnowledgeBase().distill_from_event("factor_mining", dict(payload))
    return payload


@app.post("/api/v1/expressions/validate")
def validate_expression(req: ValidateRequest) -> dict[str, Any]:
    return ExpressionEngine(FieldRegistry.from_project()).validate(req.expression).to_dict()


@app.post("/api/v1/factors/promote-to-calculator")
def promote_to_calculator(req: PromoteRequest) -> dict[str, Any]:
    try:
        result = CalculatorPromoter().promote_to_calculator(
            factor_name=req.factor_name,
            expression=req.expression,
            creator=req.creator,
            description=req.description,
            lookback_days=req.lookback_days,
            write=req.write,
        )
    except FileExistsError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if not result["ok"]:
        raise HTTPException(status_code=400, detail=result["errors"])
    return result


@app.post("/api/v1/factors/score")
def score_factor(req: ScoreRequest) -> dict[str, Any]:
    result = score_factor_summary(req.summary, req.validation)
    result["knowledge_asset"] = KnowledgeBase().distill_from_event(
        "score_factor",
        {"summary": req.summary, "validation": req.validation, "evaluation": dict(result)},
    )
    return result


@app.post("/api/v1/factors/gate")
def gate_factor(req: GateFactorRequest) -> dict[str, Any]:
    payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    return run_factor_gate(**payload)


@app.post("/api/v1/research-log")
def append_research_log(req: LogRequest) -> dict[str, Any]:
    return ResearchLog().append(req.event)


@app.get("/api/v1/research-log")
def list_research_log(limit: int = 50) -> dict[str, Any]:
    records = ResearchLog().list_recent(limit=limit)
    return {"records": records, "count": len(records)}


@app.post("/api/v1/knowledge")
def distill_knowledge(req: KnowledgeRequest) -> dict[str, Any]:
    return KnowledgeBase().distill_from_event(req.event_type, req.payload)


@app.get("/api/v1/knowledge")
def list_knowledge(asset_type: str | None = None, limit: int = 50) -> dict[str, Any]:
    records = KnowledgeBase().list_recent(asset_type=asset_type, limit=limit)
    return {"records": records, "count": len(records)}


@app.get("/api/v1/knowledge/search")
def search_knowledge(query: str, asset_type: str | None = None, limit: int = 20) -> dict[str, Any]:
    records = KnowledgeBase().search(query=query, asset_type=asset_type, limit=limit)
    return {"records": records, "count": len(records)}
