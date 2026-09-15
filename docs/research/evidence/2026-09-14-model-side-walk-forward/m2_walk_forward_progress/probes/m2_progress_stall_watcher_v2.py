#!/usr/bin/env python3
"""Stall watcher v2: separate "hung" from "busy but silent", using measured phase lengths.

v1 (`m2_progress_stall_watcher.py`) alarmed on SILENCE(3 h) + SPIN(one core, zero I/O delta). On
2026-09-15 that came within 11 minutes of a false alarm: M2's F1 final-ensemble stage wrote its last
checkpoint at 11:31 and its model bundle at 14:20, i.e. **2 h 49 min of legitimate file silence while
a worker burned a full core in memory** -- 93% of the 3 h threshold. F2, F3 and F4 each have the same
closing stage, so v1's margin is one short epoch wide.

v2 keeps v1's conservatism but adds the measurement v1 lacked -- cumulative CPU time of the *whole*
process tree -- and splits the verdict in two:

  DEAD  no process in the tree advanced its CPU time across the sample window, nothing is in
        uninterruptible sleep (state D, i.e. a slow disk/NFS read), and no file for --dead-hours
        (default 1.0). A blocked or deadlocked pool looks like this and should be reported fast.
  SPIN  the tree IS advancing CPU time, its /proc/*/io counters are NOT moving at all, and no file
        for --spin-hours (default 6.0 = 2.1x the measured 2 h 49 m silent tail). This is the
        pathology v1 was written for: cores pinned, zero syscalls, nothing landing.

Everything else is logged as ok/quiet with the tree's core count and I/O delta attached, so the log
alone shows whether compute is advancing. Long silence with advancing CPU and moving I/O counters is
explicitly NOT an alarm; it is the known closing stage.

Read-only. Writes only its own log and alarm file. Never signals a process, never touches the
registry, a database, the workspace, or any repository file. Exits when the driver exits.

Usage: setsid nohup python m2_progress_stall_watcher_v2.py --driver 3588838 \
           --workspace /data/.../m2_v1 [--dead-hours 1] [--spin-hours 6] [--poll-minutes 10] &
"""
from __future__ import annotations

import argparse
import os
import time
from datetime import datetime
from pathlib import Path

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
IO_KEYS = ("rchar", "wchar", "syscr", "syscw", "read_bytes", "write_bytes")


def log(path: Path, msg: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    with path.open("a") as fh:
        fh.write(line + "\n")


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def tree(root: int) -> list[int]:
    """root plus every descendant, via /proc/*/stat ppid (no subprocess, no ps)."""
    ppid: dict[int, int] = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            fields = Path(f"/proc/{entry}/stat").read_text().rsplit(")", 1)[1].split()
            ppid[int(entry)] = int(fields[1])
        except Exception:
            continue
    kids: dict[int, list[int]] = {}
    for child, parent in ppid.items():
        kids.setdefault(parent, []).append(child)
    out, stack = [], [root]
    while stack:
        pid = stack.pop()
        out.append(pid)
        stack.extend(kids.get(pid, []))
    return sorted(set(out))


def newest_file(workspace: Path) -> tuple[float, str]:
    best, name = 0.0, ""
    for p in workspace.rglob("*"):
        try:
            if p.is_file() and (m := p.stat().st_mtime) > best:
                best, name = m, str(p)
        except OSError:
            pass
    return best, name


def sample(pids: list[int], seconds: float) -> dict:
    hz = os.sysconf("SC_CLK_TCK")

    def snap() -> dict:
        out = {}
        for p in pids:
            try:
                st = Path(f"/proc/{p}/stat").read_text()
                head, rest = st.split(")", 1)
                f = rest.split()
                io = {}
                try:
                    for line in Path(f"/proc/{p}/io").read_text().splitlines():
                        k, _, v = line.partition(":")
                        io[k.strip()] = int(v.strip())
                except OSError:
                    pass
                out[p] = {
                    "ticks": int(f[11]) + int(f[12]),       # utime + stime
                    "state": f[0],
                    "io": io,
                    "name": head[head.find("(") + 1:],
                }
            except Exception:
                pass
        return out

    a = snap()
    t0 = time.time()
    time.sleep(seconds)
    b = snap()
    dt = time.time() - t0 or 1e-9
    per = []
    for p, s in b.items():
        if p not in a:
            continue
        dio = sum(s["io"].get(k, 0) - a[p]["io"].get(k, 0) for k in IO_KEYS)
        per.append({
            "pid": p, "name": s["name"], "state": s["state"],
            "cores": (s["ticks"] - a[p]["ticks"]) / hz / dt,
            "io_delta": dio,
        })
    return {
        "window_seconds": dt,
        "sampled": len(per),
        "tree_cores": sum(x["cores"] for x in per),
        "tree_io_delta": sum(x["io_delta"] for x in per),
        "cpu_advanced": any(x["cores"] > 0.005 for x in per),
        "any_uninterruptible": any(x["state"] == "D" for x in per),
        "per_process": sorted(per, key=lambda x: -x["cores"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", type=int, required=True)
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--dead-hours", type=float, default=1.0)
    ap.add_argument("--spin-hours", type=float, default=6.0)
    ap.add_argument("--quiet-hours", type=float, default=2.0,
                    help="annotate the log from this much file silence (the measured tail is 2.82 h)")
    ap.add_argument("--poll-minutes", type=float, default=10.0)
    ap.add_argument("--sample-seconds", type=float, default=30.0)
    ap.add_argument("--carrier", default="M2")
    ap.add_argument("--log", type=Path, default=ART / "m2_progress_stall_v2.log")
    ap.add_argument("--alarm", type=Path, default=ART / "m2_progress_stall_v2.ALARM")
    args = ap.parse_args()

    log(args.log, f"stall watcher v2 start: driver={args.driver} workspace={args.workspace} "
                  f"dead_hours={args.dead_hours} spin_hours={args.spin_hours} "
                  f"poll_min={args.poll_minutes} sample_s={args.sample_seconds}")
    log(args.log, "thresholds are set from a measurement, not a guess: M2's F1 final-ensemble stage "
                  "wrote nothing from 11:31 to 14:20 (2.82 h) while a worker burned a full core, so "
                  f"spin_hours={args.spin_hours} is 2.1x that and file silence alone is never an alarm")
    latched = ""
    while True:
        if not alive(args.driver):
            log(args.log, f"driver {args.driver} no longer running; terminal outcomes belong to the "
                          "other watchers, so this one exits")
            return
        mtime, name = newest_file(args.workspace)
        silent_h = (time.time() - mtime) / 3600 if mtime else float("inf")
        pids = tree(args.driver)
        s = sample(pids, args.sample_seconds)
        top = s["per_process"][0] if s["per_process"] else None
        head = (f"tree={s['sampled']} proc, {s['tree_cores']:.2f} cores, io_delta={s['tree_io_delta']}, "
                f"newest file {silent_h:.2f} h old")
        if top:
            head += f" (busiest pid {top['pid']} {top['name']} {top['cores']:.2f} cores state {top['state']})"

        verdict = "ok"
        detail = ""
        if not s["cpu_advanced"] and not s["any_uninterruptible"] and silent_h >= args.dead_hours:
            verdict = "ALARM-DEAD"
            detail = (f"no process in the tree advanced CPU time across {s['window_seconds']:.0f}s, "
                      f"none is in uninterruptible sleep, and no file for {silent_h:.2f} h "
                      f"(newest {name})")
        elif (s["cpu_advanced"] and s["tree_io_delta"] == 0
              and silent_h >= args.spin_hours):
            verdict = "ALARM-SPIN"
            detail = (f"the tree burns {s['tree_cores']:.2f} cores with a zero I/O delta across "
                      f"{s['window_seconds']:.0f}s and no file for {silent_h:.2f} h "
                      f"(newest {name}) -- {args.spin_hours} h is 2.1x the measured silent tail")
        elif silent_h >= args.quiet_hours:
            verdict = "quiet"
            detail = ("CPU is advancing, so this is the known closing stage of a fold, not a stall")

        if verdict.startswith("ALARM"):
            if latched != verdict:
                log(args.log, f"{verdict} {detail} [{head}]")
                with args.alarm.open("a") as fh:
                    fh.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {args.carrier} {verdict} {detail}\n")
                latched = verdict
            else:
                log(args.log, f"{verdict} still latched, not repeating [{head}]")
        else:
            if latched:
                log(args.log, f"progress resumed ({verdict}); clearing the latched {latched}")
                latched = ""
            log(args.log, f"{verdict}{': ' + detail if detail else ''} [{head}]")
        time.sleep(args.poll_minutes * 60)


if __name__ == "__main__":
    main()
