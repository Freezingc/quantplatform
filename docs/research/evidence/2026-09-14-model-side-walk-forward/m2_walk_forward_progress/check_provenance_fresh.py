#!/usr/bin/env python3
"""Maintenance check: does provenance.json still describe the files actually on disk?

This exists because the mistake it catches has already happened. On the §CL round, provenance.json was
regenerated, and THEN the probe's source was edited (three line-number citations were corrected from
`walk_forward_factor_daily.py:243` to `:234` with the hit path at `:246`/`:248`/`:254`, and
`@dataclass(frozen=True)` from `:2208` to `:2198`/`:2199`), and the commit went out without
regenerating. The result was a provenance.json in git whose recorded sha256 for
probe_m2_f3f4_fold_cache_identity.py was a6cd825c… while the file in the very same commit hashed to
5334db63…. Two entries were stale: that probe and its measurement JSON, which the corrected probe had
re-run and so also changed.

Nothing about that commit looked wrong. `git status` was clean, check_section_order.py passed, the
prose was correct. The only symptom is that provenance.json -- the file whose entire purpose is to let
a later reader verify that the recorded numbers came from the recorded tools -- silently stopped
describing them. A reader who tried to reproduce §CL by checking hashes first would have concluded the
evidence had been tampered with.

This is the same class of trap as §CF's section-ordering check: an invariant that is easy to violate by
an innocent-looking edit late in a round, invisible in every other check, and expensive to discover by
hand later. So it gets a gate.

Run it after editing any probe or measurement, and before every commit. It checks three things:
  1. every sha256 recorded in provenance.json matches the file on disk;
  2. every probe/measurement file on disk is recorded (a new file cannot slip in unhashed);
  3. the driver sha256 recorded in provenance.json matches the live driver in the main worktree, and
     the main worktree is still clean -- because a provenance file that says "the driver was never
     edited" while the driver has since changed is worse than no provenance file at all.

It reads only this pack's own files plus the driver hash. No database, no network, no repository module
imported or executed, nothing written. Exit status is 0 when fresh and 1 when stale, so it can gate a
commit.

Validated in the direction that proves it can fail, per this pack's convention: appending one byte to
a COPY of a probe under /tmp and pointing the check at it yields "MISMATCH"; deleting a measurement
from the recorded dict yields "NOT RECORDED"; and pointing the driver path at a modified copy yields
"driver sha256 differs". The pack's own files were never modified during that negative testing.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MAIN_REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX")
DRIVER = "research_v2/scripts/run_real_candidate_training.py"
EXPECTED_DRIVER_SHA = "33742880f314ca2151b76b02e65531f5fbf7554e3e1d93a858f458f1f580b113"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def main() -> int:
    prov = json.load(open(HERE / "provenance.json", encoding="utf-8"))
    problems: list[str] = []

    for kind, subdir in (("probe_scripts", "probes"), ("measurements", "measurements")):
        recorded: dict[str, str] = prov.get(kind, {})
        on_disk = {p.name for p in (HERE / subdir).glob("*")
                   if p.is_file() and p.suffix in (".py", ".json", ".log", ".csv", ".md")}

        for name, want in sorted(recorded.items()):
            p = HERE / subdir / name
            if not p.is_file():
                problems.append(f"{kind}/{name}: recorded but MISSING on disk")
                continue
            got = sha256(p)
            if got != want:
                problems.append(
                    f"{kind}/{name}: STALE hash\n"
                    f"      recorded {want[:16]}…\n"
                    f"      on disk  {got[:16]}…\n"
                    f"      most likely cause: the file was edited (or re-run) AFTER provenance.json "
                    f"was generated. Re-run `python make_provenance.py` and re-stage both files.")

        for name in sorted(on_disk - set(recorded)):
            problems.append(f"{kind}/{name}: on disk but NOT RECORDED in provenance.json "
                            f"(run `python make_provenance.py`)")

    rec_driver = (prov.get("driver_script_never_edited") or {}).get("sha256")
    live = MAIN_REPO / DRIVER
    if not live.is_file():
        problems.append(f"driver not found at {live}")
    else:
        got = sha256(live)
        if rec_driver != got:
            problems.append(f"driver sha256 recorded {rec_driver} != live {got} -- the driver CHANGED")
        if got != EXPECTED_DRIVER_SHA:
            problems.append(f"live driver sha256 {got} != the value every section cites "
                            f"({EXPECTED_DRIVER_SHA})")
    if git(MAIN_REPO, "status", "--porcelain"):
        problems.append("main worktree is NOT clean")

    n_prov = len(prov.get("probe_scripts", {})) + len(prov.get("measurements", {}))
    print(f"provenance.json: {len(prov.get('probe_scripts', {}))} probes, "
          f"{len(prov.get('measurements', {}))} measurements ({n_prov} hashes checked)")
    if problems:
        print("FAIL")
        for p in problems:
            print("  -", p)
        return 1
    print(f"OK: every recorded hash matches disk, nothing unrecorded, driver is "
          f"{EXPECTED_DRIVER_SHA[:16]}… and the main worktree is clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
