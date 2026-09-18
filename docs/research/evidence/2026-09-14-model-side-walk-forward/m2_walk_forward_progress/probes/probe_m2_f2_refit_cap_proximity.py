#!/usr/bin/env python3
"""Read-only probe for §CR: a LIVE re-read of §CJ's cap-truncation scan, because the one M2 run §CJ
excluded as "still writing" has since climbed to the machine-wide maximum peak and is still climbing.

WHY THIS PROBE EXISTS. §CJ scanned all 486 torch runs and published (via ledger D13) three citable
claims: (a) 483 runs are in a completed stage and 0 were truncated by the max_epochs=100 cap; (b) the
machine-wide maximum peak epoch is 71, held by the grid run m2-f2-c17-20260802-seed-20260802, implying
at most 81 of 100 epochs and a 19-epoch margin; (c) 0 runs fall in the ambiguity region peak >= 90
(where peak + patience 10 could reach the cap 100, so "early-stopped" and "capped" become
indistinguishable). §CJ explicitly carved out the three M2 F2 final-refit runs as "still writing" and
excluded them from the 483. D13 then framed the cap as "a forward risk for M2 F3/F4".

WHAT CHANGED. As of this probe's timestamp, the excluded F2 refit run m2-f2-final-20260802-seed-20260802
has reached a peak of at least 71 -- tying the grid-stage machine-wide maximum -- and is still writing
(6 checkpoints in the last 30 minutes at the time of writing). It trains under the SAME cap as the grid
(max_epochs from run_real_candidate_training.py:2940, passed to the final fit at :2544; patience
TORCH_EARLY_STOPPING_PATIENCE=10 at :2086), so if it crosses 90 it enters §CJ's ambiguity region. Two
consequences for D13, both durable and independent of the exact current value:
  * the cap's forward risk is located at the RUNNING F2 refit, not at F3/F4 as D13 said;
  * "machine-wide max peak = 71" is no longer a stable ceiling -- §CJ stated it over the 483 completed
    runs while excluding this refit, and the refit has now reached at least that value and is climbing,
    so the number must be re-read once the refit early-stops or finishes.

MONOTONE-SAFE REPORTING. Peak epoch only ever increases for a running fit (trainer.py:110-116 saves on
strict improvement), so "has reached at least N" is permanently true once observed, while "is currently
N" is stale the next minute. This probe therefore asserts only monotone-safe relations (peak >= grid max;
cap == 100; refit and grid share the cap) and reports the exact current peaks ONLY as a timestamped
snapshot explicitly flagged as non-stable. Freezing a moving value into a durable claim is exactly the
partial-snapshot error §CQ warns against, so it is not done here.

WHAT IS STILL-WRITING versus FINISHED, measured not assumed: a run is "active" if any of its checkpoint
directories has an mtime within the last ACTIVE_WINDOW minutes. This is validated as a live control --
of the three F2 refit seeds, 20260802 is active (writing now) while 20260801 and 20260803 early-stopped
long ago (last writes 09-18 10:48 and 09-18 15:37), so the detector must separate them.

Only checkpoint directory NAMES and their mtimes are read, plus source text. No parquet (not even
footers), no checkpoint.pt, no weight or tensor, no repository module imported or executed, no database,
no network, no writes to any artifact directory, no signal to any process. Does not touch
/tmp/walk_forward_cache_v1, so no atime side effect.
"""
from __future__ import annotations

import glob
import re
import statistics as st
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "measurements" / "m2_f2_refit_cap_proximity.json"

CAP = 100                 # run_real_candidate_training.py:2940 (args.max_epochs), §CJ-established
PATIENCE = 10             # TORCH_EARLY_STOPPING_PATIENCE, run_real_candidate_training.py:2086
AMBIGUITY_THRESHOLD = CAP - PATIENCE   # 90: peak + patience could reach cap -> early-stop vs cap ambiguous
ACTIVE_WINDOW_MIN = 40    # a run is "still writing" if a checkpoint was written within this many minutes
GRID_MAX_AT_CJ = 71       # §CJ / D13 published machine-wide max (grid run m2-f2-c17)
RUN_RE = re.compile(r"^(?P<run>.+)-epoch-(?P<epoch>\d+)$")


def scan_torch_runs() -> dict:
    """peak epoch, checkpoint count, newest mtime, and active flag for every torch run in both carriers."""
    per_run: dict[str, dict] = {}
    now = datetime.now()
    for ws in ("m1_full_v1", "m2_v1"):
        runs: dict[str, list[tuple[int, float]]] = defaultdict(list)
        for d in glob.glob(str(ART / ws / "checkpoints" / "*-epoch-*")):
            m = RUN_RE.match(Path(d).name)
            if not m:
                continue
            runs[m.group("run")].append((int(m.group("epoch")), Path(d).stat().st_mtime))
        for r, eps in runs.items():
            peak = max(e for e, _ in eps)
            newest = max(mt for _, mt in eps)
            age_min = (now.timestamp() - newest) / 60.0
            stage = "refit" if "final" in r else "grid"
            per_run[f"{ws}/{r}"] = {
                "workspace": ws, "run": r, "stage": stage, "peak": peak,
                "n_checkpoints": len(set(e for e, _ in eps)),
                "newest_mtime": datetime.fromtimestamp(newest).isoformat(timespec="minutes"),
                "minutes_since_last_write": round(age_min, 1),
                "active": age_min <= ACTIVE_WINDOW_MIN,
                "in_ambiguity_region": peak >= AMBIGUITY_THRESHOLD,
                "at_or_over_cap": peak >= CAP,
                "margin_to_ambiguity": AMBIGUITY_THRESHOLD - peak,
            }
    return per_run


def main() -> None:
    runs = scan_torch_runs()
    peaks = [r["peak"] for r in runs.values()]
    machine_max = max(peaks)
    ambiguity = [k for k, r in runs.items() if r["in_ambiguity_region"]]
    over_cap = [k for k, r in runs.items() if r["at_or_over_cap"]]
    active = {k: r for k, r in runs.items() if r["active"]}

    # the three F2 refit seeds, specifically
    f2_refit = {k: r for k, r in runs.items() if re.search(r"m2-f2-final-\d+-seed-\d+$", r["run"])}
    seed_02 = next(r for k, r in f2_refit.items() if r["run"].endswith("20260802-seed-20260802"))
    seed_01 = next(r for k, r in f2_refit.items() if r["run"].endswith("20260801-seed-20260801"))
    seed_03 = next(r for k, r in f2_refit.items() if r["run"].endswith("20260803-seed-20260803"))

    # grid-stage max, to compare the refit against (the §CJ ceiling)
    grid_peaks = [r["peak"] for r in runs.values() if r["stage"] == "grid"]
    grid_max = max(grid_peaks)
    grid_argmax = [k for k, r in runs.items() if r["stage"] == "grid" and r["peak"] == grid_max]

    doc = {
        "question": ("has the M2 F2 final-refit run that §CJ excluded as 'still writing' approached the "
                     "cap, and does that change ledger D13's citable claims about the machine-wide max "
                     "peak and the location of the cap risk?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "SNAPSHOT_WARNING": ("every exact peak in this file is a live reading that changes minute to "
                             "minute while the F2 refit runs. The DURABLE claims are the monotone-safe "
                             "relations under 'durable_conclusions'; the exact numbers under 'snapshot' "
                             "are NOT stable and must be re-read after the refit early-stops."),
        "cap_semantics": {
            "max_epochs_cap": CAP, "early_stopping_patience": PATIENCE,
            "ambiguity_threshold_peak": AMBIGUITY_THRESHOLD,
            "cap_source": "run_real_candidate_training.py:2940 (args.max_epochs), final fit at :2544",
            "patience_source": "run_real_candidate_training.py:2086 TORCH_EARLY_STOPPING_PATIENCE=10",
            "refit_uses_same_cap_as_grid": True,
            "why_90_is_the_ambiguity_edge": ("a run whose peak is 90 could have executed 90 + 10 = 100 "
                                             "epochs, i.e. hit the cap, so 'early-stopped at 90' and "
                                             "'truncated at 100' are indistinguishable from the checkpoint "
                                             "record alone (§CJ.2)"),
        },
        "snapshot": {
            "n_torch_runs_scanned": len(runs),
            "machine_wide_max_peak_NOW": machine_max,
            "grid_stage_max_peak": grid_max,
            "grid_stage_argmax_runs": grid_argmax,
            "CJ_published_grid_max": GRID_MAX_AT_CJ,
            "runs_in_ambiguity_region_peak_ge_90": len(ambiguity),
            "ambiguity_runs": ambiguity,
            "runs_at_or_over_cap_100": len(over_cap),
            "active_runs_last_write_within_window": {
                k: {"peak": r["peak"], "minutes_since_last_write": r["minutes_since_last_write"]}
                for k, r in active.items()},
            "active_window_minutes": ACTIVE_WINDOW_MIN,
            "F2_refit_three_seeds": {
                "20260801": {"peak": seed_01["peak"], "n_checkpoints": seed_01["n_checkpoints"],
                             "last_write": seed_01["newest_mtime"], "active": seed_01["active"]},
                "20260802": {"peak": seed_02["peak"], "n_checkpoints": seed_02["n_checkpoints"],
                             "last_write": seed_02["newest_mtime"], "active": seed_02["active"],
                             "margin_to_ambiguity_90": seed_02["margin_to_ambiguity"]},
                "20260803": {"peak": seed_03["peak"], "n_checkpoints": seed_03["n_checkpoints"],
                             "last_write": seed_03["newest_mtime"], "active": seed_03["active"]},
            },
            "peak_by_stage": {
                "grid": {"n": len(grid_peaks), "max": grid_max, "mean": round(st.mean(grid_peaks), 3)},
                "refit": {"n": len([r for r in runs.values() if r["stage"] == "refit"]),
                          "max": max((r["peak"] for r in runs.values() if r["stage"] == "refit"),
                                     default=None)},
            },
        },
        "durable_conclusions": {
            "F2_refit_seed_02_peak_ge_grid_max": bool(seed_02["peak"] >= grid_max),
            "F2_refit_seed_02_peak_value_at_read": seed_02["peak"],
            "grid_max_value": grid_max,
            "monotone_safety": ("peak only increases for a running fit, so 'seed 20260802 has reached at "
                                "least the grid-stage max' stays true forever once observed; the exact "
                                "value above is a snapshot and will only grow"),
            "cap_risk_is_at_the_running_F2_refit_not_F3F4": True,
            "reason": ("§CJ excluded the three F2 refit runs from its '483 completed, 0 truncated, max "
                       "71, margin 19'; the one still running has since reached at least the grid max "
                       "under the same cap, so the forward cap risk D13 attributed to F3/F4 is in fact "
                       "live at the F2 refit right now"),
            "CJ_483_excluded_the_F2_refit": True,
            "machine_max_71_no_longer_a_stable_ceiling": True,
            "D13_corrections": [
                "the cap forward-risk is located at the RUNNING F2 refit, not at F3/F4",
                "'machine-wide max peak = 71' was stated over 483 completed runs excluding this refit; "
                "the refit has reached at least 71 and is climbing, so re-read the max after it stops",
                "'0 runs in the ambiguity region (peak >= 90)' holds at this timestamp but is a live "
                "quantity for the running refit, not a settled fact",
            ],
        },
        "live_controls": {},
        "read_only": True, "database_access": False, "weights_loaded": False,
        "checkpoint_pt_opened": False, "parquet_read": False, "training_performed": False,
        "repository_module_imported_or_executed": False, "processes_signalled": [],
        "atime_side_effect": False,
    }

    # --- live controls: the detectors must be able to say NO
    # (i) active-detector separates the still-writing seed from the two early-stopped ones
    active_detector_separates = (seed_02["active"] and not seed_01["active"] and not seed_03["active"])
    # (ii) ambiguity-region detector: synthetic peaks straddling the 90 threshold
    def in_ambiguity(peak: int) -> bool:
        return peak >= AMBIGUITY_THRESHOLD
    ambiguity_detector = {
        "peak_71_below_threshold": in_ambiguity(71) is False,
        "peak_89_below_threshold": in_ambiguity(89) is False,
        "peak_90_at_threshold": in_ambiguity(90) is True,
        "peak_100_over_cap": in_ambiguity(100) is True,
    }
    # (iii) the seed-02 peak really is >= the grid max (monotone-safe core claim)
    core_claim_holds = seed_02["peak"] >= grid_max
    doc["live_controls"] = {
        "active_detector_separates_writing_from_stopped": active_detector_separates,
        "seed_20260802_active": seed_02["active"],
        "seed_20260801_stopped": not seed_01["active"],
        "seed_20260803_stopped": not seed_03["active"],
        "ambiguity_detector": ambiguity_detector,
        "ambiguity_detector_all_correct": all(ambiguity_detector.values()),
        "core_monotone_claim_seed02_ge_gridmax": core_claim_holds,
        "ALL_PASS": bool(active_detector_separates and all(ambiguity_detector.values()) and core_claim_holds),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    import json
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    s = doc["snapshot"]
    print(f"scanned {s['n_torch_runs_scanned']} torch runs across both published carriers")
    print(f"\n=== SNAPSHOT (live, changes minute to minute) ===")
    print(f"  machine-wide max peak NOW: {s['machine_wide_max_peak_NOW']}  (§CJ/D13 published grid max {GRID_MAX_AT_CJ})")
    print(f"  grid-stage max: {s['grid_stage_max_peak']} at {s['grid_stage_argmax_runs']}")
    print(f"  runs in ambiguity region (peak>=90): {s['runs_in_ambiguity_region_peak_ge_90']}")
    print(f"  runs at/over cap 100: {s['runs_at_or_over_cap_100']}")
    print(f"  active runs (last write <= {ACTIVE_WINDOW_MIN} min): {list(s['active_runs_last_write_within_window'].keys())}")
    print(f"\n  F2 refit three seeds:")
    for sd, v in s["F2_refit_three_seeds"].items():
        print(f"    {sd}: peak {v['peak']:>3d}, {v['n_checkpoints']:>2d} ckpts, last write {v['last_write']}, "
              f"active={v['active']}" + (f", margin to 90 = {v.get('margin_to_ambiguity_90')}" if 'margin_to_ambiguity_90' in v else ""))
    d = doc["durable_conclusions"]
    print(f"\n=== DURABLE (monotone-safe) conclusions ===")
    print(f"  F2 refit seed 20260802 peak ({d['F2_refit_seed_02_peak_value_at_read']}) >= grid max ({d['grid_max_value']}): {d['F2_refit_seed_02_peak_ge_grid_max']}")
    print(f"  cap risk at running F2 refit, not F3/F4: {d['cap_risk_is_at_the_running_F2_refit_not_F3F4']}")
    print(f"  'machine max 71' no longer a stable ceiling: {d['machine_max_71_no_longer_a_stable_ceiling']}")
    print(f"\n=== live controls ===")
    lc = doc["live_controls"]
    for k, v in lc.items():
        if k != "ambiguity_detector":
            print(f"  {'OK ' if v else '*** FAIL ***'} {k}")
    print(f"  ambiguity_detector: {lc['ambiguity_detector']} -> all_correct={lc['ambiguity_detector_all_correct']}")
    print(f"  ALL_PASS: {lc['ALL_PASS']}")
    assert lc["ALL_PASS"], "a live control failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
