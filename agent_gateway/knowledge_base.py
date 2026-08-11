from __future__ import annotations

import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any


class KnowledgeBase:
    """Structured research asset store for reusable factor research knowledge."""

    def __init__(self, root: str | Path = "agent_gateway_runs"):
        self.root = Path(root)
        self.base_dir = self.root / "knowledge_base"
        self.assets_path = self.base_dir / "assets.jsonl"
        for name in ("rules", "findings", "failures"):
            (self.base_dir / name).mkdir(parents=True, exist_ok=True)

    def distill_from_event(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        event_type = event_type.strip().lower()
        if event_type in {"factor_eval", "factor_test", "score_factor"}:
            asset = self._from_factor_evaluation(payload)
        elif event_type in {"factor_mining", "mine_factor", "generate_factor"}:
            asset = self._from_factor_mining(payload)
        elif event_type in {"similarity_check", "factor_similarity"}:
            asset = self._from_similarity_check(payload)
        else:
            asset = self._generic_asset(event_type, payload)
        return self.append(asset)

    def append(self, asset: dict[str, Any]) -> dict[str, Any]:
        asset_type = asset.get("asset_type") or "findings"
        if asset_type not in {"rules", "findings", "failures"}:
            asset_type = "findings"
        record = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "asset_type": asset_type,
            **asset,
        }
        self.assets_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path = self._write_markdown(record)
        record["path"] = str(markdown_path)
        with self.assets_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record

    def list_recent(self, asset_type: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        records = self._read_all()
        if asset_type:
            records = [record for record in records if record.get("asset_type") == asset_type]
        return records[-limit:]

    def search(self, query: str, asset_type: str | None = None, limit: int = 20) -> list[dict[str, Any]]:
        terms = [term.lower() for term in re.split(r"\s+", query.strip()) if term]
        records = self._read_all()
        if asset_type:
            records = [record for record in records if record.get("asset_type") == asset_type]
        if not terms:
            return records[-limit:]

        scored = []
        for record in records:
            text = json.dumps(record, ensure_ascii=False, default=str).lower()
            score = sum(text.count(term) for term in terms)
            if score:
                scored.append((score, record))
        scored.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in scored[:limit]]

    def _from_factor_evaluation(self, payload: dict[str, Any]) -> dict[str, Any]:
        evaluation = payload.get("evaluation") or payload.get("score") or payload
        recommendation_code = evaluation.get("recommendation_code")
        risk_flags = evaluation.get("risk_flags") or []
        metrics = evaluation.get("metrics_used") or payload.get("summary") or {}
        factor_name = payload.get("factor_name") or payload.get("name") or "unnamed_factor"

        if recommendation_code == "enter_pool":
            asset_type = "rules"
            title = f"{factor_name}: validated candidate worth pool review"
            lesson = "This factor passed the current score gate and can be reused as a positive reference."
        elif recommendation_code == "stop" or _has_major_risk(risk_flags):
            asset_type = "failures"
            title = f"{factor_name}: avoid repeating weak or risky setup"
            lesson = "This setup should not be repeated without changing the signal, horizon, or risk controls."
        else:
            asset_type = "findings"
            title = f"{factor_name}: candidate needs further optimization"
            lesson = "The factor has research value, but current evidence is not strong enough for pool inclusion."

        return {
            "asset_type": asset_type,
            "event_type": "factor_eval",
            "title": title,
            "factor_name": factor_name,
            "expression": payload.get("expression"),
            "recommendation_code": recommendation_code,
            "grade": evaluation.get("grade"),
            "score": evaluation.get("score"),
            "risk_flags": risk_flags,
            "metrics": metrics,
            "lesson": lesson,
            "evidence": evaluation.get("diagnostic_reasons") or [],
            "next_actions": evaluation.get("improvement_suggestions") or [],
            "tags": _compact_tags(["factor_eval", recommendation_code, *risk_flags]),
            "source_payload": payload,
        }

    def _from_factor_mining(self, payload: dict[str, Any]) -> dict[str, Any]:
        candidates = payload.get("candidates") or []
        valid_count = sum(1 for item in candidates if item.get("expression"))
        missing_fields = sorted({
            field
            for item in candidates
            for field in item.get("missing_fields", [])
        })
        asset_type = "failures" if candidates and valid_count == 0 else "findings"
        return {
            "asset_type": asset_type,
            "event_type": "factor_mining",
            "title": f"Factor mining produced {valid_count}/{len(candidates)} implementable candidates",
            "factor_name": payload.get("factor_name"),
            "lesson": (
                "Reusable candidate expressions were found."
                if valid_count
                else "No implementable candidate was found; field mapping or source interpretation needs work."
            ),
            "evidence": [
                f"candidate_count={len(candidates)}",
                f"valid_expression_count={valid_count}",
            ],
            "next_actions": _mining_next_actions(valid_count, missing_fields),
            "risk_flags": ["missing_fields"] if missing_fields else [],
            "missing_fields": missing_fields,
            "tags": _compact_tags(["factor_mining", "missing_fields" if missing_fields else "has_candidates"]),
            "source_payload": payload,
        }

    def _from_similarity_check(self, payload: dict[str, Any]) -> dict[str, Any]:
        max_similarity = _first_float(payload, ["max_similarity", "max_corr", "correlation", "corr"], 0.0)
        threshold = _first_float(payload, ["threshold"], 0.8)
        factor_name = payload.get("factor_name") or payload.get("name") or "unnamed_factor"
        similar_to = payload.get("similar_to") or payload.get("nearest_factor")
        is_duplicate = max_similarity >= threshold
        return {
            "asset_type": "failures" if is_duplicate else "findings",
            "event_type": "similarity_check",
            "title": (
                f"{factor_name}: high similarity to existing factor"
                if is_duplicate
                else f"{factor_name}: acceptable similarity distance"
            ),
            "factor_name": factor_name,
            "similar_to": similar_to,
            "score": max_similarity,
            "lesson": (
                "Avoid promoting this factor unless it adds orthogonal information or reduces cost/risk."
                if is_duplicate
                else "This candidate is not obviously redundant under the supplied similarity threshold."
            ),
            "evidence": [f"max_similarity={max_similarity}", f"threshold={threshold}"],
            "next_actions": (
                ["Compare residual IC after neutralizing against the nearest factor.", "Prefer combining or replacing instead of adding a redundant factor."]
                if is_duplicate
                else ["Continue with factor evaluation and cost checks."]
            ),
            "risk_flags": ["high_similarity"] if is_duplicate else [],
            "tags": _compact_tags(["similarity_check", "high_similarity" if is_duplicate else "orthogonal_candidate"]),
            "source_payload": payload,
        }

    def _generic_asset(self, event_type: str, payload: dict[str, Any]) -> dict[str, Any]:
        status = str(payload.get("status") or payload.get("recommendation_code") or "").lower()
        asset_type = "failures" if status in {"failed", "fail", "stop"} else "findings"
        return {
            "asset_type": asset_type,
            "event_type": event_type,
            "title": payload.get("title") or f"Research event: {event_type}",
            "lesson": payload.get("lesson") or "Captured for later review.",
            "evidence": payload.get("evidence") or [],
            "next_actions": payload.get("next_actions") or [],
            "risk_flags": payload.get("risk_flags") or [],
            "tags": _compact_tags(["research_event", event_type, *payload.get("tags", [])]),
            "source_payload": payload,
        }

    def _write_markdown(self, record: dict[str, Any]) -> Path:
        created = record["created_at"].replace(":", "").replace("-", "")
        slug = _slugify(record.get("title") or record.get("event_type") or "asset")
        path = self.base_dir / record["asset_type"] / f"{created}_{slug}.md"
        lines = [
            f"# {record.get('title', 'Research Asset')}",
            "",
            f"- asset_type: {record.get('asset_type')}",
            f"- event_type: {record.get('event_type')}",
            f"- created_at: {record.get('created_at')}",
            f"- factor_name: {record.get('factor_name')}",
            f"- recommendation_code: {record.get('recommendation_code')}",
            f"- grade: {record.get('grade')}",
            f"- score: {record.get('score')}",
            f"- risk_flags: {', '.join(record.get('risk_flags') or [])}",
            "",
            "## Lesson",
            str(record.get("lesson") or ""),
            "",
            "## Evidence",
            *_bullet_lines(record.get("evidence") or []),
            "",
            "## Next Actions",
            *_bullet_lines(record.get("next_actions") or []),
            "",
            "## Tags",
            *_bullet_lines(record.get("tags") or []),
            "",
        ]
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _read_all(self) -> list[dict[str, Any]]:
        if not self.assets_path.exists():
            return []
        records = []
        for line in self.assets_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                records.append(json.loads(line))
        return records


def _has_major_risk(risk_flags: list[str]) -> bool:
    return any(flag in {"high_turnover", "high_similarity", "weak_out_of_sample", "high_drawdown"} for flag in risk_flags)


def _first_float(payload: dict[str, Any], keys: list[str], default: float) -> float:
    for key in keys:
        if key in payload:
            try:
                return float(payload[key])
            except (TypeError, ValueError):
                return default
    return default


def _mining_next_actions(valid_count: int, missing_fields: list[str]) -> list[str]:
    actions = []
    if valid_count:
        actions.append("Run expression validation, similarity check, and factor evaluation.")
    if missing_fields:
        actions.append("Resolve missing field mappings before generating production calculators.")
    if not actions:
        actions.append("Review extracted candidates manually.")
    return actions


def _compact_tags(values: list[Any]) -> list[str]:
    return [str(value) for value in values if value]


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-").lower()
    return slug[:80] or "asset"


def _bullet_lines(values: list[Any]) -> list[str]:
    if not values:
        return ["- none"]
    return [f"- {value}" for value in values]
