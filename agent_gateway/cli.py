from __future__ import annotations

import argparse
import json

from .dsl import ExpressionEngine
from .factor_pipeline import run_factor_gate
from .field_registry import FieldRegistry
from .ingest import FactorSourceIngestor
from .knowledge_base import KnowledgeBase
from .promote import CalculatorPromoter
from .research_log import ResearchLog
from .validation import score_factor_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent gateway for factor research workflows")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list-fields")

    p_ingest = sub.add_parser("ingest")
    p_ingest.add_argument("source")
    p_ingest.add_argument("--source-type", default="auto")
    p_ingest.add_argument("--max-candidates", type=int, default=6)

    p_gen = sub.add_parser("generate")
    p_gen.add_argument("description")
    p_gen.add_argument("--max-candidates", type=int, default=6)

    p_val = sub.add_parser("validate")
    p_val.add_argument("expression")

    p_promote = sub.add_parser("promote")
    p_promote.add_argument("factor_name")
    p_promote.add_argument("expression")
    p_promote.add_argument("--creator", default="agent_gateway")
    p_promote.add_argument("--description", default="Generated from DSL candidate. Review before production use.")
    p_promote.add_argument("--lookback-days", type=int)
    p_promote.add_argument("--write", action="store_true")

    p_score = sub.add_parser("score")
    p_score.add_argument("summary_json")
    p_score.add_argument("--validation-json")

    p_gate = sub.add_parser("gate-factor")
    p_gate.add_argument("factor_name")
    p_gate.add_argument("method")
    p_gate.add_argument("--lookback-days", type=int, required=True)
    p_gate.add_argument("--formula", default="")
    p_gate.add_argument("--description", default="")
    p_gate.add_argument("--creator", default="agent_gateway")
    p_gate.add_argument("--category", default="agent_generated")
    p_gate.add_argument("--start-time", default="2020.01.01")
    p_gate.add_argument("--end-time")
    p_gate.add_argument("--create-date")
    p_gate.add_argument("--similarity-threshold", type=float, default=0.7)
    p_gate.add_argument("--min-abs-ic", type=float, default=0.01)
    p_gate.add_argument("--min-abs-icir", type=float, default=0.3)
    p_gate.add_argument("--min-common-stocks", type=int, default=50)
    p_gate.add_argument("--min-valid-dates", type=int, default=20)
    p_gate.add_argument("--sample-days", type=int, default=60)
    p_gate.add_argument("--no-sample-similarity", action="store_true")
    p_gate.add_argument("--commit", action="store_true")
    p_gate.add_argument("--env-path")

    p_log = sub.add_parser("log")
    p_log.add_argument("event_json")

    p_knowledge = sub.add_parser("knowledge")
    p_knowledge.add_argument("event_type")
    p_knowledge.add_argument("payload_json")

    p_list_knowledge = sub.add_parser("list-knowledge")
    p_list_knowledge.add_argument("--asset-type", choices=["rules", "findings", "failures"])
    p_list_knowledge.add_argument("--limit", type=int, default=20)

    p_search_knowledge = sub.add_parser("search-knowledge")
    p_search_knowledge.add_argument("query")
    p_search_knowledge.add_argument("--asset-type", choices=["rules", "findings", "failures"])
    p_search_knowledge.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()
    registry = FieldRegistry.from_project()

    if args.cmd == "list-fields":
        payload = {"fields": registry.as_dicts()}
    elif args.cmd == "ingest":
        payload = FactorSourceIngestor(registry).ingest_factor_source(args.source, args.source_type, args.max_candidates)
        payload["knowledge_asset"] = KnowledgeBase().distill_from_event(
            "factor_mining",
            {"source_type": args.source_type, "candidates": list(payload.get("candidates", []))},
        )
    elif args.cmd == "generate":
        payload = {"candidates": [c.to_dict() for c in FactorSourceIngestor(registry).generate_factor_expressions(args.description, args.max_candidates)]}
        payload["knowledge_asset"] = KnowledgeBase().distill_from_event("factor_mining", {"description": args.description, **dict(payload)})
    elif args.cmd == "validate":
        payload = ExpressionEngine(registry).validate(args.expression).to_dict()
    elif args.cmd == "promote":
        payload = CalculatorPromoter(registry).promote_to_calculator(
            args.factor_name,
            args.expression,
            creator=args.creator,
            description=args.description,
            lookback_days=args.lookback_days,
            write=args.write,
        )
    elif args.cmd == "score":
        payload = score_factor_summary(
            json.loads(args.summary_json),
            json.loads(args.validation_json) if args.validation_json else None,
        )
        payload["knowledge_asset"] = KnowledgeBase().distill_from_event(
            "score_factor",
            {
                "summary": json.loads(args.summary_json),
                "validation": json.loads(args.validation_json) if args.validation_json else None,
                "evaluation": dict(payload),
            },
        )
    elif args.cmd == "gate-factor":
        payload = run_factor_gate(
            factor_name=args.factor_name,
            method=args.method,
            lookback_days=args.lookback_days,
            formula=args.formula,
            description=args.description,
            creator=args.creator,
            category=args.category,
            start_time=args.start_time,
            end_time=args.end_time,
            create_date=args.create_date,
            similarity_threshold=args.similarity_threshold,
            min_abs_ic=args.min_abs_ic,
            min_abs_icir=args.min_abs_icir,
            min_common_stocks=args.min_common_stocks,
            min_valid_dates=args.min_valid_dates,
            sample_similarity=not args.no_sample_similarity,
            sample_days=args.sample_days,
            commit=args.commit,
            env_path=args.env_path,
        )
    elif args.cmd == "log":
        payload = ResearchLog().append(json.loads(args.event_json))
    elif args.cmd == "knowledge":
        payload = KnowledgeBase().distill_from_event(args.event_type, json.loads(args.payload_json))
    elif args.cmd == "list-knowledge":
        payload = {"records": KnowledgeBase().list_recent(args.asset_type, args.limit)}
    elif args.cmd == "search-knowledge":
        payload = {"records": KnowledgeBase().search(args.query, args.asset_type, args.limit)}
    else:
        raise SystemExit(f"Unknown command: {args.cmd}")

    print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
