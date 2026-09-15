#!/usr/bin/env python3
"""Alarm when a training driver is alive and burning CPU but has written nothing for hours.

The three watchers already running over this pipeline all wait for a terminal event: the chain watcher waits
for report.json plus driver exit, the fold preflight watcher waits for a selection to land so it can run a
reproduction audit, and neither can see a run that is alive and making no progress. On 2026-09-15 the M2
driver sat in futex_wait with two workers each burning exactly one core at zero syscalls while one sibling
worker had been idle in pipe_read since its seed early-stopped, and nothing in the monitoring stack would
ever have reported it. This fills that gap.

It is deliberately conservative, because a false alarm on a run that has already cost more than a day of
compute is expensive. Two conditions must hold together before it alarms:

  SILENCE  no FILE under the workspace has been written for --stall-hours (default 3). Directory mtimes do
           not count -- they move on create, rename and delete, so a prune could look like progress. The
           checkpoint cadence observed for this run is about 17 minutes, and it stops entirely when the epoch
           loop ends, so an hour of silence is normal and three is not.
  SPIN     at least one worker is burning >0.9 of a core while its /proc/PID/io counters do not move across
           a --sample-seconds window. A real evaluation phase runs in memory and could legitimately show no
           I/O, so spin alone is not enough -- but spin plus hours of silence is.

If the driver exits, this watcher exits too; terminal outcomes belong to the other watchers.

Read-only. Writes only its own log and alarm file. Never signals a process, never touches the registry, a
database, the workspace, or any repository file.

Usage: setsid nohup python m2_progress_stall_watcher.py --driver 3588838 \
           --workspace /data/.../m2_v1 [--stall-hours 3] [--poll-minutes 10] &
"""
from __future__ import annotations
import argparse, os, time
from datetime import datetime
from pathlib import Path

ART = Path("/data/zhangyudong/qrv2_research_artifacts")


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


def newest_file(workspace: Path) -> tuple[float, str]:
    best, name = 0.0, ""
    for p in workspace.rglob("*"):
        try:
            if not p.is_file():
                continue
            m = p.stat().st_mtime
            if m > best:
                best, name = m, str(p)
        except OSError:
            pass
    return best, name


def children(pid: int) -> list[int]:
    try:
        return [int(x) for x in os.popen(f"ps --ppid {pid} -o pid= 2>/dev/null").read().split()]
    except Exception:
        return []


def cpu_and_io(pids: list[int], seconds: float) -> list[dict]:
    hz = os.sysconf("SC_CLK_TCK")

    def snap():
        out = {}
        for p in pids:
            try:
                st = Path(f"/proc/{p}/stat").read_text().split()
                io = {}
                for line in Path(f"/proc/{p}/io").read_text().splitlines():
                    k, _, v = line.partition(":")
                    io[k.strip()] = int(v.strip())
                out[p] = (int(st[13]) + int(st[14]), io)
            except Exception:
                pass
        return out

    a = snap(); t0 = time.time(); time.sleep(seconds); b = snap(); dt = time.time() - t0
    res = []
    for p, (ticks, io) in b.items():
        if p not in a:
            continue
        dticks = ticks - a[p][0]
        dio = sum(io[k] - a[p][1].get(k, 0) for k in ("rchar", "wchar", "syscr", "syscw"))
        res.append({"pid": p, "cores": dticks / hz / dt if dt else 0.0, "io_delta": dio})
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", type=int, required=True)
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--stall-hours", type=float, default=3.0)
    ap.add_argument("--poll-minutes", type=float, default=10.0)
    ap.add_argument("--sample-seconds", type=float, default=30.0)
    ap.add_argument("--carrier", default="M2")
    ap.add_argument("--log", type=Path, default=ART / "m2_progress_stall.log")
    ap.add_argument("--alarm", type=Path, default=ART / "m2_progress_stall.ALARM")
    args = ap.parse_args()

    log(args.log, f"stall watcher start: driver={args.driver} workspace={args.workspace} "
                  f"stall_hours={args.stall_hours} poll_min={args.poll_minutes} sample_s={args.sample_seconds}")
    already = False
    while True:
        if not alive(args.driver):
            log(args.log, f"driver {args.driver} no longer running; terminal outcomes belong to the other "
                          "watchers, so this one exits")
            return
        mtime, name = newest_file(args.workspace)
        silent_h = (time.time() - mtime) / 3600 if mtime else float("inf")
        if silent_h >= args.stall_hours:
            kids = children(args.driver)
            samples = cpu_and_io(kids, args.sample_seconds) if kids else []
            spinning = [s for s in samples if s["cores"] > 0.9 and s["io_delta"] == 0]
            if spinning and not already:
                detail = (f"no file written for {silent_h:.2f} h (newest {name}) while "
                          f"{len(spinning)} of {len(samples)} worker(s) burn a full core with zero I/O "
                          f"delta over {args.sample_seconds:.0f}s: "
                          + ", ".join(f"pid {s['pid']} at {s['cores']:.2f} cores" for s in spinning))
                log(args.log, f"ALARM {detail}")
                with args.alarm.open("a") as fh:
                    fh.write(f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {args.carrier} {detail}\n")
                already = True
            elif spinning:
                log(args.log, f"still stalled ({silent_h:.2f} h, {len(spinning)} spinning worker(s)); "
                              "alarm already raised, not repeating")
            else:
                log(args.log, f"silent {silent_h:.2f} h but no worker is spinning "
                              f"({len(samples)} sampled) -- not alarming; a blocked or finished pool is the "
                              "other watchers' business")
                already = False
        else:
            if already:
                log(args.log, f"progress resumed (newest file {silent_h:.2f} h old); clearing the latched alarm")
            already = False
            log(args.log, f"ok: newest file {silent_h:.2f} h old, driver alive, {len(children(args.driver))} worker(s)")
        time.sleep(args.poll_minutes * 60)


if __name__ == "__main__":
    main()
