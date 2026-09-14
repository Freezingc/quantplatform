#!/usr/bin/env python3
"""Compute provenance.json for this directory: every hash measured, none typed.

Walks the directory, hashes the probes, the measurements and the narrative, and records the repository
state the findings were read from -- including the factor source catalog, because the census resolves
each factor's upstream table and name from it rather than retyping them, so a catalog edit would
silently change what was compared.

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
    "research_v2/configs/factors/source_catalog.yaml",
    "research_v2/scripts/materialize_frozen_factor_daily.py",
    "research_v2/scripts/walk_forward_common.py",
    "research_v2/scripts/walk_forward_factor_daily.py",
    "research_v2/scripts/audit_formula_replay_consistency.py",
    "research_v2/src/research_v2/evaluation/gates.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True, check=True).stdout.strip()


def directory(root: Path, pattern: str) -> dict:
    return {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size} for p in sorted(root.glob(pattern))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    args = parser.parse_args()

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": {"path": "make_provenance.py", "sha256": sha256(HERE / "make_provenance.py")},
        "safety": (
            "read-only investigation: SELECT-only queries against dfs://YesLab and "
            "dfs://QuantResearchV2Test, plus local file reads. No database write, no weight loading, "
            "no artifact mutation, no code or threshold change in the repository."
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
        "cited_source_sha256": {
            rel: sha256(args.repo / rel) for rel in CITED_SOURCES if (args.repo / rel).is_file()
        },
        "cited_source_missing": [rel for rel in CITED_SOURCES if not (args.repo / rel).is_file()],
        "probe_scripts": directory(HERE / "probes", "*.py"),
        "measurements": directory(HERE / "measurements", "*.json"),
        "narratives": directory(HERE, "*.md"),
        "factor_catalog": {
            "path": "research_v2/configs/factors/source_catalog.yaml",
            "sha256": sha256(args.repo / "research_v2" / "configs" / "factors" / "source_catalog.yaml"),
            "note": "the census resolves every upstream table and factor name from this file",
        },
        "excluded_by_design": [
            "model weights / checkpoints (*.pt)",
            "parquet panels and fold caches (read in place, never copied)",
            "the upstream factor observations themselves: only aggregate counts, correlations and "
            "differences are recorded, never the values",
        ],
    }
    (HERE / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"hashed {len(provenance['probe_scripts'])} probes, {len(provenance['measurements'])} measurements, "
          f"{len(provenance['narratives'])} narratives, {len(provenance['cited_source_sha256'])} cited sources")
    if provenance["cited_source_missing"]:
        print(f"WARNING missing cited sources: {provenance['cited_source_missing']}")
    print(f"wrote {HERE / 'provenance.json'}")


if __name__ == "__main__":
    main()
