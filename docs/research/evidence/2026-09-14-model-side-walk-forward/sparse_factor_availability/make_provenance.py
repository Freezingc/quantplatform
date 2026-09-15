#!/usr/bin/env python3
"""Compute provenance.json for this directory: every hash measured, none typed.

One line is quoted verbatim on purpose. walk_forward_factor_daily.py's comment above
FROZEN_FACTOR_SET_V2_DENSE says the six sparse factors "are NaN on roughly half of all days by formula
(their sparsity IS the candidate signal)". §BA measures that sentence: the half is real, but it is a
day-level regime rather than a per-security property, and the two clusters §AN.4 found turn out to be
two largely independent on/off switches. If that comment ever changes, this record should show it.

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
    "research_v2/scripts/walk_forward_factor_daily.py",
    "research_v2/scripts/walk_forward_common.py",
    "research_v2/artifacts/diagnostics/factor_set_frozen_v2.json",
    "research_v2/configs/factors/source_catalog.yaml",
    "research_v2/scripts/materialize_frozen_factor_daily.py",
    "research_v2/scripts/audit_formula_replay_consistency.py",
    "research_v2/scripts/audit_formula_replay_tushare.py",
    # the six sparse calculators and the two dense controls §BD reads; they live under a
    # non-ASCII directory in the main tree and are opened read-only
    "数据库/factor_calculators/GP_Factor_003_Calculator.py",
    "数据库/factor_calculators/GP_Factor_006_Calculator.py",
    "数据库/factor_calculators/GP_Factor_012_Calculator.py",
    "数据库/factor_calculators/GP_Factor_015_Calculator.py",
    "数据库/factor_calculators/GP_Factor_020_Calculator.py",
    "数据库/factor_calculators/GP_Factor_021_Calculator.py",
    "数据库/factor_calculators/GP_Factor_037_Calculator.py",
    "数据库/factor_calculators/GP_Factor_056_Calculator.py",
)
SPARSITY_COMMENT_FILE = "research_v2/scripts/walk_forward_factor_daily.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                          check=True).stdout.strip()


def sparsity_comment(repo: Path) -> dict:
    path = repo / SPARSITY_COMMENT_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    hits = [{"line_number": i, "text": ln.strip()} for i, ln in enumerate(lines, start=1)
            if "sparsity IS the candidate signal" in ln or "NaN" in ln and "half of all days" in ln]
    return {"file": SPARSITY_COMMENT_FILE, "sha256": sha256(path), "matches": hits}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    args = ap.parse_args()
    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": {"path": "make_provenance.py", "sha256": sha256(HERE / "make_provenance.py")},
        "safety": (
            "read-only. No database access at all: every input is a parquet panel the frozen pipeline "
            "already published, read in place. No weight or model was loaded, nothing was written into "
            "any artifact directory, no process was signalled, and no repository file was edited."
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
        "sparsity_comment_quoted": sparsity_comment(args.repo),
        "cited_source_sha256": {r: sha256(args.repo / r) for r in CITED_SOURCES
                                if (args.repo / r).is_file()},
        "cited_source_missing": [r for r in CITED_SOURCES if not (args.repo / r).is_file()],
        "probe_scripts": {p.name: sha256(p) for p in sorted((HERE / "probes").glob("*.py"))},
        "measurements": {p.name: sha256(p) for p in sorted((HERE / "measurements").iterdir())},
        "excluded_by_design": [
            "model weights / checkpoints (*.pt)",
            "the parquet panels themselves (read in place, never copied)",
            "per-security rows: only per-day coverage aggregates and per-factor summaries are recorded",
        ],
    }
    (HERE / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"sparsity comment: {provenance['sparsity_comment_quoted']['matches']}")
    print(f"hashed {len(provenance['probe_scripts'])} probe(s), "
          f"{len(provenance['measurements'])} measurement(s), "
          f"{len(provenance['cited_source_sha256'])} cited source(s)")
    if provenance["cited_source_missing"]:
        print(f"WARNING missing cited sources: {provenance['cited_source_missing']}")
    print(f"wrote {HERE / 'provenance.json'}")


if __name__ == "__main__":
    main()
