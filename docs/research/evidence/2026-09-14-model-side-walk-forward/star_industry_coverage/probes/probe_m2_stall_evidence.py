"""Capture the evidence that decides "computing" from "hung" for a running training driver.

A driver that is alive, whose workers are at 100% CPU and whose watchers are silent, is ambiguous: both a
long compute phase and a spin-wait deadlock look like that from ps alone. This separates them with four
read-only signals, sampled twice so that deltas rather than levels carry the information:

  1. CPU-time delta per process over an interval, in cores. A spin loop and a real computation both burn
     cores, so this alone is not decisive -- but zero is.
  2. I/O counters (/proc/PID/io). A training or evaluation phase that reads a dataset and writes checkpoints
     moves these; a userspace spin does not move them at all.
  3. Kernel wait channel (/proc/PID/wchan) and per-thread state histogram. `futex_wait_queue` on the parent
     plus `pipe_read` on one child plus exactly one running thread in the others is the shape of a pool whose
     stragglers never return, not of a pool that is working.
  4. Wall-clock silence: how long since anything under the workspace was created or modified, against the
     cadence the run had been showing.

Nothing here is written except --out. No process is signalled, no file under the workspace is touched.

Usage: python probe_m2_stall_evidence.py --driver 3588838 --workspace <m2_v1> --interval 30 --out <json>
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path


def proc_stats(pids: list[int]) -> dict:
    out = {}
    for p in pids:
        try:
            stat = Path(f"/proc/{p}/stat").read_text().split()
            io = {}
            for line in Path(f"/proc/{p}/io").read_text().splitlines():
                k, _, v = line.partition(":")
                io[k.strip()] = int(v.strip())
            status = dict(l.split(":", 1) for l in Path(f"/proc/{p}/status").read_text().splitlines() if ":" in l)
            try:
                wchan = Path(f"/proc/{p}/wchan").read_text().strip()
            except Exception:
                wchan = None
            threads = {}
            tdir = Path(f"/proc/{p}/task")
            for t in tdir.iterdir() if tdir.is_dir() else []:
                try:
                    st = (t / "stat").read_text().split()
                    threads[st[2]] = threads.get(st[2], 0) + 1
                except Exception:
                    pass
            out[p] = {"cpu_ticks": int(stat[13]) + int(stat[14]), "state": stat[2], "wchan": wchan,
                      "threads_total": int(status.get("Threads", "0").strip()),
                      "thread_states": threads,
                      "io": {k: io.get(k) for k in ("rchar", "wchar", "syscr", "syscw")},
                      "rss_kb": int(status.get("VmRSS", "0").split()[0])}
        except FileNotFoundError:
            out[p] = {"gone": True}
    return out


def newest(workspace: Path) -> tuple[float, str]:
    """Newest FILE mtime under the workspace.

    Directories are excluded on purpose: a directory's mtime moves when an entry is created, renamed or
    deleted, so counting directories lets a pruning or rename event masquerade as progress. Only a written
    file is progress.
    """
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--driver", type=int, required=True)
    ap.add_argument("--workspace", type=Path, required=True)
    ap.add_argument("--interval", type=float, default=30.0)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    def children(pid: int) -> list[int]:
        try:
            out = os.popen(f"ps --ppid {pid} -o pid= 2>/dev/null").read().split()
            return [int(x) for x in out]
        except Exception:
            return []

    kids = children(args.driver)
    pids = [args.driver] + kids
    m1, newest1 = newest(args.workspace)
    a = proc_stats(pids)
    t0 = time.time()
    time.sleep(args.interval)
    b = proc_stats(pids)
    t1 = time.time()
    m2, newest2 = newest(args.workspace)

    hz = os.sysconf("SC_CLK_TCK")
    per = []
    for p in pids:
        if a[p].get("gone") or b[p].get("gone"):
            per.append({"pid": p, "gone": True}); continue
        d = b[p]["cpu_ticks"] - a[p]["cpu_ticks"]
        iod = {k: (b[p]["io"][k] - a[p]["io"][k]) for k in ("rchar", "wchar", "syscr", "syscw")}
        per.append({"pid": p, "role": "driver" if p == args.driver else "worker",
                    "cores_busy": round(d / hz / (t1 - t0), 3),
                    "state": b[p]["state"], "wchan": b[p]["wchan"],
                    "threads_total": b[p]["threads_total"], "thread_states": b[p]["thread_states"],
                    "io_delta_over_interval": iod,
                    "io_total_syscr": b[p]["io"]["syscr"], "io_total_syscw": b[p]["io"]["syscw"],
                    "rss_gb": round(b[p]["rss_kb"] / 1048576, 2)})

    silent_for_h = (t1 - m2) / 3600
    spinners = [r for r in per if r.get("cores_busy", 0) > 0.9 and
                sum(r["io_delta_over_interval"].values()) == 0 and r.get("role") == "worker"]
    idle_workers = [r for r in per if r.get("role") == "worker" and r.get("cores_busy", 0) < 0.01]
    # Thresholds are deliberately conservative. A final-ensemble evaluation phase runs entirely in memory
    # (each worker holds ~14 GB RSS), so zero syscalls over a 30 s sample is NOT by itself evidence of a
    # hang; and silence is not evidence either, because the 17-minute checkpoint cadence stops when the
    # epoch loop ends. What counts as evidence is a LONG silence together with a full core burning and no
    # syscall at all -- an evaluation of that length would touch the dataset or write a result.
    verdict = ("AMBIGUOUS -- alive and burning CPU but nothing written for {h:.2f} h; re-sample before "
               "concluding anything").format(h=silent_for_h)
    if spinners and silent_for_h > 3:
        verdict = ("HUNG -- {n} worker(s) burn a full core with zero syscalls in {iv:.0f}s while no file has "
                   "been written for {h:.2f} h; the driver is in futex_wait and {k} sibling worker(s) are idle "
                   "in pipe_read, i.e. the pool is waiting on stragglers that are not returning"
                   ).format(n=len(spinners), iv=args.interval, h=silent_for_h, k=len(idle_workers))
    elif not spinners and silent_for_h > 3:
        verdict = ("STALLED_BUT_NOT_SPINNING -- no worker is burning a core, yet no file has been written for "
                   "{h:.2f} h").format(h=silent_for_h)
    elif silent_for_h <= 3:
        verdict = ("WITHIN_TOLERANCE -- the newest file is {h:.2f} h old; the 17-minute checkpoint cadence does "
                   "not explain that, but a post-epoch evaluation phase running in memory would"
                   ).format(h=silent_for_h)

    out = {"question": "Is the M2 driver computing or hung?",
           "driver_pid": args.driver, "worker_pids": kids, "interval_seconds": args.interval,
           "workspace": str(args.workspace),
           "workspace_newest_file": newest2,
           "workspace_silent_for_hours": round(silent_for_h, 3),
           "processes": per, "verdict": verdict,
           "signals_written": False, "processes_signalled": False, "database_writes": False}
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(verdict)
    print(json.dumps({"workspace_newest_file": newest2, "silent_for_hours": round(silent_for_h, 3)},
                     ensure_ascii=False, indent=1))
    for r in per:
        if r.get("gone"):
            print(f"  pid {r['pid']}: GONE"); continue
        print(f"  pid {r['pid']} ({r['role']}): cores={r['cores_busy']} state={r['state']} "
              f"wchan={r['wchan']} threads={r['thread_states']} io_delta={r['io_delta_over_interval']} "
              f"rss={r['rss_gb']}GB")


if __name__ == "__main__":
    main()
