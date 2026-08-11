from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any


class ResearchLog:
    """Append-only JSONL experiment log for agent-generated factor research."""

    def __init__(self, root: str | Path = "agent_gateway_runs"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.path = self.root / "experiments.jsonl"

    def append(self, event: dict[str, Any]) -> dict[str, Any]:
        record = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            **event,
        }
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
        return record

    def list_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines[-limit:]]
