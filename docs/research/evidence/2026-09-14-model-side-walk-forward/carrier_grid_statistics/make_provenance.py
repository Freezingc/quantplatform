#!/usr/bin/env python3
"""Compute provenance.json for this directory: every hash measured, none typed.

This directory holds the cross-carrier grid-search statistics: the blocked variance decompositions of
the hyperparameter grids (§BL ran M2's F1; the probe here runs M1 across all four folds). Its inputs are
JSON search records the training runs already wrote, so the pins that matter are the two files that
define what a "config" and a "selection" are:

  * scripts/run_real_candidate_training.py -- it owns m1_config_grid / m2_config_grid / m3_config_grid,
    i.e. the grids being decomposed, and the live M2 run re-imports it per fold. Hashing it here is how
    this pack proves it was never edited while that run was in flight.
  * training/search.py -- it owns select_model_config, the rule whose behaviour §BN questions (it ranks
    configs by ensemble rank IC and takes the argmax), and build_pre_registered_trials, the hash-ordered
    truncation §BM records as never being called from production.

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
    "research_v2/src/research_v2/training/search.py",
    "research_v2/src/research_v2/training/cross_section_search.py",
    "research_v2/src/research_v2/training/search_metrics.py",
    "research_v2/src/research_v2/training/ensemble.py",
    "research_v2/src/research_v2/models/families.py",
    "research_v2/src/research_v2/evaluation/gates.py",
    "research_v2/configs/models/residual_mlp.yaml",
    "research_v2/configs/models/temporal.yaml",
    "research_v2/configs/models/set_attention.yaml",
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
            "read-only analysis of search records that finished training runs already wrote. No database "
            "access at all. No weight or checkpoint was loaded -- every number here comes from the "
            "per-seed and ensemble validation RankIC fields inside JSON. Nothing was written into any "
            "artifact directory. No repository file was edited; run_real_candidate_training.py is hashed "
            "below to prove it, and the live M2 driver re-imports that file per fold. No process was "
            "signalled."
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
                "reader computes, the file changed while a run was in flight and every number in §BL, "
                "§BM and §BN is suspect"
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
