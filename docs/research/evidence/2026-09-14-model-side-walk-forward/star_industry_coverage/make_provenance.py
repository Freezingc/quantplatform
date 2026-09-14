#!/usr/bin/env python3
"""Compute provenance.json and vendored_industry_cache/MANIFEST.json for this evidence directory.

Every hash in this pack is computed, never typed: this walks the directory, hashes the probe scripts,
the measurement outputs and the vendored industry-cache files, and records the repository state the
findings were read from. Re-running it after adding a probe or re-measuring keeps the record honest.

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

# Source files the root-cause narrative quotes. Hashed so a later edit cannot silently change meaning.
CITED_SOURCES = (
    "research_v2/scripts/materialize_frozen_factor_daily.py",
    "research_v2/scripts/audit_tushare_pit_industry.py",
    "research_v2/scripts/audit_formula_replay_consistency.py",
    "research_v2/src/research_v2/factors/deduplication.py",
    "research_v2/src/research_v2/factors/materializer.py",
    "research_v2/src/research_v2/factors/source_evaluation.py",
    "research_v2/scripts/walk_forward_factor_daily.py",
    "research_v2/scripts/walk_forward_common.py",
    "research_v2/configs/factors/source_catalog.yaml",
    "docs/reports/2026-09-14-dense-factor-coverage-gap-and-the-24-days.md",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    args = parser.parse_args()

    manifest = {
        "description": (
            "Byte copies of the SW2021 level-1 industry membership files the frozen-chain replay read "
            "from research_v2/artifacts/tushare_cache. Vendored because the vendor restates this "
            "history: re-fetching the same security today returns a different industry and in_date."
        ),
        "source_dir": str(args.repo / "research_v2" / "artifacts" / "tushare_cache"),
        "files": {},
    }
    cache_dir = HERE / "vendored_industry_cache"
    for path in sorted(cache_dir.glob("*.json")):
        if path.name == "MANIFEST.json":
            continue
        manifest["files"][path.name] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    combined = hashlib.sha256()
    for name in sorted(manifest["files"]):
        combined.update(hashlib.sha256(name.encode()).digest())
        combined.update(bytes.fromhex(manifest["files"][name]["sha256"]))
    manifest["combined_sha256"] = combined.hexdigest()
    manifest["file_count"] = len(manifest["files"])
    (cache_dir / "MANIFEST.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

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
        "vendored_industry_cache": {
            "file_count": manifest["file_count"],
            "combined_sha256": manifest["combined_sha256"],
        },
        "excluded_by_design": [
            "model weights / checkpoints (*.pt)",
            "parquet panels and fold caches (read in place, never copied)",
            "database exports beyond the aggregate counts quoted in root_cause.md",
        ],
    }
    (HERE / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"vendored {manifest['file_count']} cache files, combined {manifest['combined_sha256'][:16]}…")
    print(f"hashed {len(provenance['probe_scripts'])} probes, {len(provenance['measurements'])} measurements, "
          f"{len(provenance['cited_source_sha256'])} cited sources")
    if provenance["cited_source_missing"]:
        print(f"WARNING missing cited sources: {provenance['cited_source_missing']}")
    print(f"wrote {HERE / 'provenance.json'}")


if __name__ == "__main__":
    main()
