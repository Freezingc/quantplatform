#!/usr/bin/env python3
"""Compute provenance.json for this evidence directory.

Every hash here is computed, never typed: this walks the directory, hashes the probe scripts, the
measurement outputs and the vendored cache rows, and records the repository state the findings were read
from. Re-running it after adding a probe or re-measuring keeps the record honest.

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

# Repository sources the root-cause narrative quotes or depends on. Hashed so a later edit cannot
# silently change the meaning of a citation.
CITED_SOURCES = (
    "research_v2/scripts/materialize_frozen_factor_daily.py",
    "research_v2/scripts/walk_forward_common.py",
    "research_v2/scripts/audit_formula_replay_consistency.py",
    "research_v2/src/research_v2/factors/materializer.py",
    "research_v2/src/research_v2/factors/source_evaluation.py",
    "research_v2/configs/factors/source_catalog.yaml",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def walk(sub: str, patterns: tuple[str, ...]) -> dict:
    out = {}
    d = HERE / sub
    if not d.is_dir():
        return out
    for p in sorted(d.rglob("*")):
        if not p.is_file() or p.name in ("MANIFEST.json", "provenance.json"):
            continue
        if patterns and not any(p.name.endswith(x) for x in patterns):
            continue
        out[p.relative_to(HERE).as_posix()] = {"sha256": sha256(p), "bytes": p.stat().st_size}
    return out


def git(repo: Path, *a: str) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True,
                              timeout=30).stdout.strip() or None
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    args = ap.parse_args()
    repo = args.repo

    cited, missing = {}, []
    for rel in CITED_SOURCES:
        p = repo / rel
        if p.is_file():
            cited[rel] = sha256(p)
        else:
            missing.append(rel)

    cache = repo / "research_v2" / "artifacts" / "tushare_cache"
    daily_files = sorted(cache.glob("daily_[0-9]*.json")) if cache.is_dir() else []

    doc = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": {"path": "make_provenance.py", "sha256": sha256(HERE / "make_provenance.py")},
        "narratives": {p.name: {"sha256": sha256(p), "bytes": p.stat().st_size}
                       for p in sorted(HERE.glob("*.md"))},
        "probe_scripts": walk("probes", (".py",)),
        "measurements": walk("measurements", (".json",)),
        "vendored_cache_rows": walk("vendored_cache_rows", (".json",)),
        "cited_source_sha256": cited,
        "cited_source_missing": missing,
        "repository_main_tree": {
            "path": str(repo),
            "branch": git(repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "head": git(repo, "rev-parse", "HEAD"),
            "note": "read-only; this evidence branch is a separate worktree",
        },
        "cache_read": {
            "path": str(cache),
            "daily_files": len(daily_files),
            "span": [daily_files[0].stem.replace("daily_", ""), daily_files[-1].stem.replace("daily_", "")]
            if daily_files else None,
            "note": "read in place, never copied except the 46 rows vendored above",
        },
        "databases_read": ["dfs://YesLab (SELECT only)", "dfs://QuantResearchV2Test (SELECT only)"],
        "safety": {
            "database_writes": False,
            "cache_writes": False,
            "repository_files_modified": False,
            "weights_or_checkpoints_written": False,
        },
        "excluded_by_design": [
            "model weights / checkpoints (*.pt)",
            "parquet panels and fold caches (read in place, never copied)",
            "full cache daily_*.json files (only the 46 offending/neighbour rows are vendored)",
            "database exports beyond the aggregate statistics quoted in root_cause.md",
        ],
    }
    (HERE / "provenance.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1, sort_keys=True))
    print(json.dumps({k: (v if not isinstance(v, dict) else f"{len(v)} entries")
                      for k, v in doc.items()}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
