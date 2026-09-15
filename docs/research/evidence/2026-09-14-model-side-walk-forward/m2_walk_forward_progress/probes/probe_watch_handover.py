#!/usr/bin/env python3
"""Read-only probe: the measured facts behind handing M2's stall monitoring from watcher v1 to v2.

v1 alarmed on 3 h of file silence plus a spinning worker. M2's F1 final-ensemble stage was legitimately
silent for 2 h 49 m, i.e. 93% of that threshold, and every remaining fold has the same closing stage.
This records the measurement that justifies v2's thresholds and the handover itself, so the numbers in
the write-up are computed here rather than typed.

Reads: the two watcher scripts, their logs, /proc for liveness, and checkpoint/model mtimes inside the
M2 workspace. Writes: one JSON under --out. Never signals a process, never touches a database, the
registry, or any repository file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
from datetime import datetime
from pathlib import Path

ART = Path("/data/zhangyudong/qrv2_research_artifacts")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def pid_of(script_name: str) -> int | None:
    """Find the interpreter process running script_name.

    Matching on the raw cmdline is not enough: the shell that launched this probe carries the whole
    heredoc in its own cmdline and would match its own text. Require argv[0] to be a python
    interpreter and some later argv to *end with* the script name, and skip this process.
    """
    self_pid = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        pid = int(entry)
        if pid == self_pid:
            continue
        try:
            argv = Path(f"/proc/{entry}/cmdline").read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        argv = [a for a in argv if a]
        if not argv or "python" not in os.path.basename(argv[0]):
            continue
        if any(a.endswith("/" + script_name) or a == script_name for a in argv[1:]):
            return pid
    return None


def tail(path: Path, n: int) -> list[str]:
    if not path.is_file():
        return []
    return path.read_text(encoding="utf-8", errors="replace").splitlines()[-n:]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=ART)
    ap.add_argument("--workspace", type=Path, default=ART / "m2_v1")
    ap.add_argument("--driver", type=int, default=3588838)
    ap.add_argument("--fold", default="f1")
    ap.add_argument("--out", type=Path, default=ART / "m2_f1_grid_and_standing")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    v1_script = args.artifacts / "m2_progress_stall_watcher.py"
    v2_script = args.artifacts / "m2_progress" / "m2_progress_stall_watcher_v2.py"
    v1_log = args.artifacts / "m2_progress_stall.log"
    v2_log = args.artifacts / "m2_progress_stall_v2.log"

    # the silent tail, measured from artifact mtimes rather than from memory
    ck = sorted((args.workspace / "checkpoints").glob(f"m2-{args.fold}-final-*"))
    last_ckpt = max((p.stat().st_mtime for p in ck), default=None)
    bundle = args.workspace / "models" / f"m2-{args.fold}"
    bundle_files = sorted(bundle.glob("*")) if bundle.is_dir() else []
    last_bundle = max((p.stat().st_mtime for p in bundle_files), default=None)
    tail_h = (last_bundle - last_ckpt) / 3600 if last_ckpt and last_bundle else None

    v2_lines = tail(v2_log, 40)
    verdicts = [ln for ln in v2_lines if re.search(r"\] (ok|quiet|ALARM-\w+)", ln)]
    io_zero = [ln for ln in verdicts if "io_delta=0," in ln]

    out = {
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "question": (
            "What was measured that justifies replacing M2's stall watcher, and did the replacement "
            "actually come up?"
        ),
        "read_only": True,
        "database_access": False,
        "processes_signalled_by_this_probe": [],
        "driver": {"pid": args.driver, "alive": alive(args.driver)},
        "measured_silent_tail": {
            "fold": args.fold.upper(),
            "what": (
                "elapsed time between the last per-epoch checkpoint of the final-ensemble stage and the "
                "model bundle that closes the fold -- the stretch in which the run writes no file at all"
            ),
            "last_final_checkpoint_mtime": datetime.fromtimestamp(last_ckpt).isoformat(timespec="seconds")
            if last_ckpt else None,
            "model_bundle_mtime": datetime.fromtimestamp(last_bundle).isoformat(timespec="seconds")
            if last_bundle else None,
            "hours": tail_h,
            "final_checkpoint_dirs_matched": len(ck),
            "v1_threshold_hours": 3.0,
            "fraction_of_v1_threshold_consumed": (tail_h / 3.0) if tail_h else None,
            "v2_spin_threshold_hours": 6.0,
            "v2_spin_threshold_over_measured_tail": (6.0 / tail_h) if tail_h else None,
        },
        "watchers": {
            "v1": {
                "script": str(v1_script),
                "sha256": sha256(v1_script),
                "pid": pid_of("m2_progress_stall_watcher.py"),
                "note": (
                    "pid is None because v1 was stopped deliberately after v2 came up; two watchers on "
                    "one run would have left v1's 3 h rule free to fire during the next fold's silent tail"
                ),
                "log_last_lines": tail(v1_log, 3),
                "alarm_file_present": (args.artifacts / "m2_progress_stall.ALARM").is_file(),
            },
            "v2": {
                "script": str(v2_script),
                "sha256": sha256(v2_script),
                "pid": pid_of("m2_progress_stall_watcher_v2.py"),
                "alive": alive(pid_of("m2_progress_stall_watcher_v2.py") or -1),
                "log": str(v2_log),
                "startup_lines": v2_lines[:2],
                "verdict_lines": verdicts[-5:],
                "alarm_file_present": (args.artifacts / "m2_progress_stall_v2.ALARM").is_file(),
            },
        },
        "spin_discriminator_is_weak": {
            "observation": (
                "v2's SPIN rule requires a zero I/O delta, but a zero I/O delta also occurs during "
                "legitimate heavy compute: tensor training with the panel already in memory issues no "
                "read/write syscalls for tens of seconds at a time"
            ),
            "verdict_lines_with_zero_io_delta": len(io_zero),
            "examples": io_zero[:3],
            "consequence": (
                "in practice SPIN degenerates into a 6 h file-silence rule. That is still safe (6 h is "
                "2.1x the measured tail) but it means v2's real improvement over v1 is the DEAD rule -- "
                "no CPU advance anywhere in the process tree and nothing in uninterruptible sleep -- not "
                "the SPIN rule"
            ),
        },
    }
    path = args.out / "watcher_v2_deployment.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"wrote {path}")
    print(json.dumps(out["measured_silent_tail"], ensure_ascii=False, indent=1))
    print("v2 pid", out["watchers"]["v2"]["pid"], "v1 pid", out["watchers"]["v1"]["pid"])


if __name__ == "__main__":
    main()
