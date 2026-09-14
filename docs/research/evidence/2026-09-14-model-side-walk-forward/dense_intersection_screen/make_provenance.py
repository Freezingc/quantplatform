#!/usr/bin/env python3
"""Compute provenance.json for this directory: every hash measured, none typed.

The instrument itself lives on an unmerged branch, so its two source files are hashed from that
worktree and recorded with the branch and commit they came from -- hashing them from the main tree
would either fail or, worse, silently pick up a different version after the branch merges.
The evaluator's floor line is quoted verbatim as well, because the whole instrument is defined
relative to that one line.

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
BRANCH_DEFAULT = Path("/data/zhangyudong/qrv2_worktrees/dense-intersection-screen")

CITED_SOURCES = (
    "research_v2/src/research_v2/evaluation/gates.py",
    "research_v2/src/research_v2/evaluation/walk_forward.py",
    "research_v2/scripts/walk_forward_common.py",
    "research_v2/scripts/walk_forward_factor_daily.py",
    "research_v2/scripts/analyze_yearly_baseline_deltas.py",
    "docs/reports/2026-09-14-dense-factor-coverage-gap-and-the-24-days.md",
)
BRANCH_SOURCES = (
    "research_v2/scripts/screen_dense_intersection_coverage.py",
    "research_v2/tests/unit/scripts/test_screen_dense_intersection_coverage.py",
)
FLOOR_FILE = "research_v2/src/research_v2/evaluation/gates.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def floor_line(repo: Path) -> dict:
    path = repo / FLOOR_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    return {
        "file": FLOOR_FILE,
        "sha256": sha256(path),
        "matches": [
            {"line_number": index, "text": text.strip()}
            for index, text in enumerate(lines, start=1)
            if "coverage <=" in text and "<=" in text.split("coverage")[0]
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    parser.add_argument("--branch-worktree", type=Path, default=BRANCH_DEFAULT)
    args = parser.parse_args()
    branch = args.branch_worktree

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": {"path": "make_provenance.py", "sha256": sha256(HERE / "make_provenance.py")},
        "safety": (
            "read-only investigation and a read-only instrument: SELECT-only queries against "
            "dfs://QuantResearchV2Test. No database write, no weight loading, no artifact mutation, "
            "no threshold change, and nothing in the repository calls the instrument."
        ),
        "repository_main_tree": {
            "path": str(args.repo),
            "branch": git(args.repo, "rev-parse", "--abbrev-ref", "HEAD"),
            "commit": git(args.repo, "rev-parse", "HEAD"),
            "subject": git(args.repo, "log", "-1", "--pretty=%s"),
            "dirty": bool(git(args.repo, "status", "--porcelain")),
        },
        "instrument_branch": {
            "worktree": str(branch),
            "branch": git(branch, "rev-parse", "--abbrev-ref", "HEAD"),
            "commit_at_generation": git(branch, "rev-parse", "HEAD"),
            "merged": False,
            "sources_sha256": {
                relative: sha256(branch / relative)
                for relative in BRANCH_SOURCES
                if (branch / relative).is_file()
            },
            "sources_missing": [
                relative for relative in BRANCH_SOURCES if not (branch / relative).is_file()
            ],
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
        "measurements": {
            path.name: sha256(path) for path in sorted((HERE / "measurements").iterdir())
        },
        "probes": {
            path.name: sha256(path) for path in sorted((HERE / "probes").glob("*.py"))
        },
        "probe_note": (
            "screen_dense_intersection_coverage.py and its tests are copies of the files on the "
            "instrument branch and are hashed there too; profile_candidate_feature_nan.py and "
            "probe_dense_per_day_attribution.py are local read-only probes that exist only here "
            "and in /data/zhangyudong/qrv2_research_artifacts"
        ),
        "excluded_by_design": [
            "model weights / checkpoints (*.pt)",
            "parquet panels and fold caches",
            "per-security rows: only per-day aggregates leave the database",
        ],
    }
    (HERE / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"floor: {provenance['score_health_floor']['matches']}")
    print(
        f"instrument branch {provenance['instrument_branch']['branch']} "
        f"{provenance['instrument_branch']['commit_at_generation'][:12]} "
        f"sources={len(provenance['instrument_branch']['sources_sha256'])}"
    )
    print(
        f"hashed {len(provenance['measurements'])} measurement(s), "
        f"{len(provenance['cited_source_sha256'])} cited source(s)"
    )
    missing = provenance["cited_source_missing"] + provenance["instrument_branch"]["sources_missing"]
    if missing:
        print(f"WARNING missing sources: {missing}")
    print(f"wrote {HERE / 'provenance.json'}")


if __name__ == "__main__":
    main()
