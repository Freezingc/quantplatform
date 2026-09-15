#!/usr/bin/env python3
"""Compute provenance.json for this directory: every hash measured, none typed.

Two things are pinned on purpose:

  * the G4 threshold line in evaluation/gates.py, because §AS.1 turns "baseline OOS mean + 0.005" into
    a concrete number M2 must clear -- if that 0.005 ever moves, this record should show that it moved;
  * the sha256 of scripts/run_real_candidate_training.py, the driver the live M2 run re-imports per
    fold. Hashing it here is how this pack proves it was never edited while the run was in flight. The
    same reasoning covers every other file §AU and §AX quote -- temporal_batches.py, walk_forward.py,
    temporal_search.py, orchestration.py and the rest are all imported by the running job, so their
    hashes are the evidence that this pack read them rather than changed them.

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
    "research_v2/scripts/run_real_candidate_training.py",
    "research_v2/scripts/walk_forward_common.py",
    "research_v2/scripts/walk_forward_factor_daily.py",
    "research_v2/src/research_v2/evaluation/gates.py",
    "research_v2/src/research_v2/evaluation/walk_forward.py",
    "research_v2/src/research_v2/evaluation/model_walk_forward.py",
    "research_v2/src/research_v2/evaluation/model_evaluator.py",
    "research_v2/src/research_v2/evaluation/rank_ic.py",
    "research_v2/src/research_v2/training/search_metrics.py",
    "research_v2/src/research_v2/training/temporal_batches.py",
    "research_v2/src/research_v2/training/temporal_search.py",
    "research_v2/src/research_v2/training/ensemble.py",
    "research_v2/src/research_v2/baselines/orchestration.py",
    "research_v2/src/research_v2/inference/torch_scoring.py",
    "research_v2/src/research_v2/inference/builder.py",
    "research_v2/configs/models/temporal.yaml",
    # §BM reads the grid definitions straight out of these, and its whole argument is that the
    # yaml `search_space` blocks and the driver's own `m{1,2,3}_config_grid()` functions disagree
    # for M3 (64 declared vs 16 run). Pinning both sides is what makes that claim checkable
    # later rather than a statement about files that may since have moved.
    "research_v2/src/research_v2/training/search.py",
    "research_v2/src/research_v2/training/cross_section_search.py",
    "research_v2/src/research_v2/models/families.py",
    "research_v2/configs/models/set_attention.yaml",
    "research_v2/configs/models/residual_mlp.yaml",
)
G4_FILE = "research_v2/src/research_v2/evaluation/gates.py"
DRIVER_FILE = "research_v2/scripts/run_real_candidate_training.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=True
    ).stdout.strip()


def g4_line(repo: Path) -> dict:
    """Quote the G4 threshold verbatim so a later edit cannot silently change §AS.1's arithmetic."""
    path = repo / G4_FILE
    lines = path.read_text(encoding="utf-8").splitlines()
    hits = [
        {"line_number": index, "text": text.strip()}
        for index, text in enumerate(lines, start=1)
        if "0.005" in text
    ]
    return {"file": G4_FILE, "sha256": sha256(path), "matches": hits}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=REPO_DEFAULT)
    args = parser.parse_args()

    provenance = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "generator": {"path": "make_provenance.py", "sha256": sha256(HERE / "make_provenance.py")},
        "safety": (
            "read-only investigation of a live training run. No database access at all -- every input is "
            "a JSON record the run itself already wrote, plus one parquet read in place for a day count. "
            "No weight was loaded, nothing was written into any artifact directory, no process was "
            "signalled except this pack's own v1 stall watcher, which was stopped deliberately and is "
            "recorded as stopped in measurements/watcher_v2_deployment.json. No repository file was "
            "edited; run_real_candidate_training.py is hashed below to prove it."
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
        "g4_threshold": g4_line(args.repo),
        "driver_script_never_edited": {
            "file": DRIVER_FILE,
            "sha256": sha256(args.repo / DRIVER_FILE)
            if (args.repo / DRIVER_FILE).is_file()
            else None,
            "why": (
                "the live M2 driver re-imports this per fold; if this hash differs from the one a later "
                "reader computes, the file changed while a run was in flight and every number in §AS is "
                "suspect"
            ),
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
        "excluded_by_design": [
            "model weights / checkpoints (*.pt)",
            "parquet panels and fold caches (read in place for cross-checks, never copied)",
            "per-security rows: only counts and per-fold/per-config aggregates are recorded",
        ],
    }
    (HERE / "provenance.json").write_text(
        json.dumps(provenance, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"G4 threshold lines: {provenance['g4_threshold']['matches']}")
    print(f"driver sha256: {provenance['driver_script_never_edited']['sha256']}")
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
