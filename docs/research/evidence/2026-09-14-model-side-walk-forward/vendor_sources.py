#!/usr/bin/env python3
"""Copy the text-only sources this evidence pack quotes into ``vendored/``.

Why vendoring is needed at all: this branch (``codex/research-conditional-factor-review``)
is ~95 commits behind ``codex/publish-quant-platform``, so the reports and handoffs
that the summary quotes do not exist on this branch. Copying them makes the pack
self-contained for reading on another machine, and ``vendored/MANIFEST.json`` records
each source's repository path, sha256 and the git HEAD it was taken from, so a reader
can tell a vendored copy from a stale one.

Text only: markdown and the small signed ``report.json`` files. No weights, no
checkpoints, no parquet, no promotion_evidence payloads.

    python3 vendor_sources.py [--repo ...] [--artifacts ...]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

DOCS = (
    "docs/reports/2026-09-12-candidate-standings-comparison.md",
    "docs/reports/2026-09-12-m0r-real-candidate-walk-forward.md",
    "docs/reports/2026-09-12-m0r-shadow-approval-preconditions.md",
    "docs/reports/2026-09-13-m1-full-grid-real-candidate-walk-forward.md",
    "docs/reports/2026-09-14-yearly-deltas-and-the-24-unevaluable-days.md",
    "docs/reports/2026-09-14-dense-factor-coverage-gap-and-the-24-days.md",
    "docs/reports/2026-09-14-prediction-reproduction-audits.md",
    "docs/reports/2026-09-14-m1-full-grid-shadow-approval-preconditions.md",
    "docs/handoffs/2026-09-08-m0-attribution-and-carrier-findings.md",
    "docs/handoffs/2026-09-09-sparse-factor-value-alignment-brief.md",
    "docs/handoffs/2026-09-09-model-status-fail-open-finding.md",
    "docs/handoffs/2026-09-11-m1-evaluation-crash-lineage-and-footprint.md",
    "docs/handoffs/2026-09-12-m1-full-grid-launch-and-measured-cost-model.md",
    "docs/handoffs/2026-09-14-m1-full-grid-passed-and-m2-launched.md",
)
REPORTS = (
    "baseline_v2",
    "m0_v2",
    "m0_v3",
    "m0r_v2",
    "m0r_v3",
    "m1_parallel_v1",
    "m1_full_v1",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX"),
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("/data/zhangyudong/qrv2_research_artifacts"),
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent / "vendored"
    head = subprocess.run(
        ["git", "-C", str(args.repo), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()

    entries = []
    for rel in DOCS:
        source = args.repo / rel
        target = root / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        entries.append(
            {
                "kind": "repository-document",
                "source": str(source),
                "vendored_as": str(target.relative_to(root.parent)),
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
                "git_head": head,
            }
        )
    for workspace in REPORTS:
        source = args.artifacts / workspace / "report.json"
        if not source.exists():
            continue
        target = root / "artifacts" / f"{workspace}_report.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        entries.append(
            {
                "kind": "signed-run-report",
                "source": str(source),
                "vendored_as": str(target.relative_to(root.parent)),
                "sha256": sha256_file(source),
                "bytes": source.stat().st_size,
                "git_head": None,
            }
        )

    manifest = {
        "generated_at_utc": datetime.now(tz=timezone.utc).isoformat(timespec="seconds"),
        "repository_git_head": head,
        "note": (
            "Vendored copies. The authoritative originals live on branch "
            "codex/publish-quant-platform (documents) and in the artifacts tree "
            "(run reports). Compare sha256 before quoting."
        ),
        "files": entries,
    }
    (root / "MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    )
    print(f"vendored {len(entries)} files into {root}")
    for entry in entries:
        print(f"  {entry['sha256'][:12]} {entry['bytes']:>7} {entry['vendored_as']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
