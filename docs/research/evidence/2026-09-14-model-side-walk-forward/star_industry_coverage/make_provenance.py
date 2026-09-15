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


def vendored_member_all(here: Path) -> dict:
    """Hash the second vendor API's byte copy, the same way the industry cache is hashed.

    Added 2026-09-15: option 1's feasibility rests on what pro.index_member_all returned on one
    particular day, so that answer is vendored and hashed rather than quoted. Returns an absent
    record instead of raising when the directory is missing, so this generator still runs on a
    checkout that predates the fetch.
    """
    directory = here / "vendored_member_all"
    if not directory.is_dir():
        return {"present": False}
    files = {
        path.name: {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(directory.glob("*.json"))
        if path.name != "MANIFEST.json"
    }
    combined = hashlib.sha256()
    for name in sorted(files):
        combined.update(hashlib.sha256(name.encode()).digest())
        combined.update(bytes.fromhex(files[name]["sha256"]))
    manifest_path = directory / "MANIFEST.json"
    return {
        "present": True,
        "file_count": len(files),
        "files": files,
        "combined_sha256": combined.hexdigest(),
        "manifest_sha256": sha256(manifest_path) if manifest_path.is_file() else None,
    }


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def tree(root: Path, pattern: str) -> dict:
    """Hash every matching file under root, recursing, keyed by path relative to root.

    Measurements now include per-run subdirectories (a sweep writes one file per day per variant), and a
    flat iterdir() plus sha256() raises IsADirectoryError on the first one. Keys are relative paths so a
    sweep's files stay grouped and a later re-run of this generator produces the same keys.
    """
    if not root.is_dir():
        return {}
    return {
        str(path.relative_to(root)): {"sha256": sha256(path), "bytes": path.stat().st_size}
        for path in sorted(root.rglob(pattern))
        if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    args = parser.parse_args()

    manifest = {
        "description": (
            "Byte copies of the SW2021 level-1 industry membership files the frozen-chain replay read "
            "from research_v2/artifacts/tushare_cache. Vendored because these are the exact bytes the "
            "replay saw, and the API that produced them (pro.index_member) clamps every STAR-Market "
            "in_date to 2023-09-25. An earlier version of this description said the vendor restates "
            "this history; that was wrong, and option1_feasibility.md section 1 corrects it -- the one "
            "security it was based on (688002.SH) changed industry on 2026-08-21, a week AFTER this "
            "cache was built on 2026-08-14, so the cache was right when it was taken."
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
        "probe_scripts": tree(HERE / "probes", "*.py"),
        "measurements": tree(HERE / "measurements", "*.json"),
        "vendored_industry_cache": {
            "file_count": manifest["file_count"],
            "combined_sha256": manifest["combined_sha256"],
        },
        "vendored_member_all": vendored_member_all(HERE),
        "narratives": {path.name: sha256(path) for path in sorted(HERE.glob("*.md"))},
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
    vma = provenance["vendored_member_all"]
    if vma.get("present"):
        print(f"vendored_member_all {vma['file_count']} files, combined {vma['combined_sha256'][:16]}…")
    print(f"hashed {len(provenance['narratives'])} narratives")
    if provenance["cited_source_missing"]:
        print(f"WARNING missing cited sources: {provenance['cited_source_missing']}")
    print(f"wrote {HERE / 'provenance.json'}")


if __name__ == "__main__":
    main()
