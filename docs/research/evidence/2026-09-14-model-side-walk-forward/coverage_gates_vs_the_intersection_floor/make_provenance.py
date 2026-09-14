#!/usr/bin/env python3
"""Compute provenance.json for this directory: every hash measured, none typed.

Also records the exact line the 0.90 floor lives on, because the whole finding is about the distance
between that line and the thresholds the two screening instruments use -- if the floor ever moves,
this record should show that it moved.

Usage: python make_provenance.py [--repo /data/.../Quant-Platform-Prod_CODEX]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO_DEFAULT = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX")

CITED_SOURCES = (
    "research_v2/scripts/screen_factor_daily_coverage.py",
    "research_v2/scripts/record_structural_quality_snapshot.py",
    "research_v2/scripts/walk_forward_common.py",
    "research_v2/scripts/walk_forward_factor_daily.py",
    "research_v2/src/research_v2/evaluation/gates.py",
    "research_v2/src/research_v2/evaluation/walk_forward.py",
    "docs/reports/2026-09-14-dense-factor-coverage-gap-and-the-24-days.md",
    "docs/handoffs/2026-09-06-c-plan-audit-decision-list.md",
)
FLOOR_FILE = "research_v2/src/research_v2/evaluation/gates.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def floor_line(repo: Path) -> dict:
    """Quote the score-health floor verbatim so a later edit cannot silently change the argument."""
    path = repo / FLOOR_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    hits = [
        {"line_number": index, "text": text.strip()}
        for index, text in enumerate(lines, start=1)
        if "0.90 <= coverage" in text
    ]
    return {"file": FLOOR_FILE, "sha256": sha256(path), "matches": hits}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    args = parser.parse_args()

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": {"path": "make_provenance.py", "sha256": sha256(HERE / "make_provenance.py")},
        "safety": (
            "read-only investigation: SELECT-only aggregate queries against "
            "dfs://QuantResearchV2Test. No database write, no weight loading, no artifact mutation, "
            "and no production script was edited -- the screen reuses the existing script's pure "
            "functions by import instead of changing them."
        ),
        "repository_main_tree": {
            "path": str(args.repo),
            "branch": git(args.repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "commit": git(args.repo, "rev-parse", "HEAD"),
            "subject": git(args.repo, "log", "-1", "--pretty=%s"),
            "dirty": bool(git(args.repo, "status", "--porcelain")),
        },
        "this_evidence_branch": {
            "branch": git(HERE, "rev-parse", "--abbrev-ref", "HEAD"),
            "commit_at_generation": git(HERE, "rev-parse", "HEAD"),
        },
        "score_health_floor": floor_line(args.repo),
        "cited_source_sha256": {
            relative: sha256(args.repo / relative)
            for relative in CITED_SOURCES
            if (args.repo / relative).is_file()
        },
        "cited_source_missing": [
            relative for relative in CITED_SOURCES if not (args.repo / relative).is_file()
        ],
        "probe_scripts": {
            path.name: sha256(path) for path in sorted((HERE / "probes").glob("*.py"))
        },
        "measurements": {
            path.name: sha256(path) for path in sorted((HERE / "measurements").iterdir())
        },
        "excluded_by_design": [
            "model weights / checkpoints (*.pt)",
            "parquet panels and fold caches (read in place for cross-checks, never copied)",
            "per-security rows: only server-side per-day aggregates leave the database",
        ],
    }
    (HERE / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"score-health floor: {provenance['score_health_floor']['matches']}")
    print(
        f"hashed {len(provenance['probe_scripts'])} probe(s), "
        f"{len(provenance['measurements'])} measurement(s), "
        f"{len(provenance['cited_source_sha256'])} cited source(s)"
    )
    if provenance["cited_source_missing"]:
        print(f"WARNING missing cited sources: {provenance['cited_source_missing']}")
    print(f"wrote {HERE / 'provenance.json'}")


if __name__ == "__main__":
    main()
