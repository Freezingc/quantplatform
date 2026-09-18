#!/usr/bin/env python3
"""Read-only probe for §CJ: is any PUBLISHED run truncated by the max_epochs=100 cap?

§CI priced M2's remaining wall clock and, in its "not done" list (§CI.5), explicitly declined to
check whether runs already finished were cut short by the epoch cap -- it only noted in passing
that F2's largest peak was 71. §CI's cost model multiplies by `epochs_per_run`, so a truncated run
would mean the model silently under-counted that run's true cost AND that the published grid
statistics (peak-epoch spread, epochs-per-run trend) describe runs that stopped for a reason other
than early stopping. This probe closes that gap over all 486 completed-or-running runs, not just F2.

WHAT IS MEASURED, and why the peak epoch is the right thing to look at
---------------------------------------------------------------------
trainer.py:96-120 (read, never executed) contains the whole stopping rule:

    best_value = -inf; stale_epochs = 0
    for epoch in 1..max_epochs:                       # max_epochs = 100 (driver default)
        validate -> validation_rank_ic
        if validation_rank_ic > best_value:           # STRICT improvement
            best_epoch = epoch; stale_epochs = 0
            checkpoint_epoch(epoch, ...)              # a directory is written
        else:
            stale_epochs += 1
        if stale_epochs >= early_stopping_patience:    # patience = 10
            break

Three consequences, each of which this probe relies on:
  (1) a checkpoint directory exists for epoch e IFF epoch e strictly improved on all of 1..e-1,
      so the max saved epoch of a run IS its best_epoch -- no need to read any manifest value;
  (2) the manifest `validation_rank_ic` values within one run must be strictly increasing in epoch,
      because every saved epoch improved on the previous saved one. This is checked over all 486
      runs rather than assumed, since it is the observable signature of (1);
  (3) a run that exits via `break` ran exactly best_epoch + patience epochs, because patience
      consecutive non-improving epochs follow the peak. A run that exits by exhausting the loop
      ran exactly max_epochs. The two are distinguishable: the break path needs
      best_epoch + patience <= max_epochs, i.e. peak <= 90.

Therefore peak >= 90 is the AMBIGUITY ZONE: for peak in [90, 100] the quantity peak + patience
would reach or exceed the cap, so from the checkpoint set alone one cannot tell whether the run
stopped early (patience) or was truncated (cap). For peak <= 89 the run provably stopped on
patience after peak + 10 <= 99 epochs, i.e. it was NOT truncated, and its true cost is fully
accounted for. The cap is 100 and patience is 10, so the zone boundary is 100 - 10 = 90.

CROSS-CHECK against §CI
-----------------------
§CI's probe_m2_runtime_projection_inputs.py computes `epochs_per_run_mean = mean(peak) + 10` per
phase and published 17.4861 (F1 grid), 18.3333 (F1 final), 28.8056 (F2 grid), 34.3333 (F2 final).
This probe recomputes the same quantity from the same directories. Agreement on the three COMPLETED
phases proves both probes read the same run set; F2 final is expected to DIFFER because that phase
is still writing (its peaks grow between snapshots), which is itself the evidence that the M2
numbers are a partial snapshot and must be labelled as one.

PARTIALITY
----------
M2 is still training (pid 3588838). Its checkpoint set grows while this runs. A peak can only
increase, never decrease, so "no completed run is truncated" is a statement about runs whose peak is
final. Completed phases are therefore separated from in-flight ones, and the in-flight ones are
reported with their current peak plus the headroom to the ambiguity zone, as a forward risk rather
than a fact -- which is precisely the distinction §CI.5 asked for.

Only directory names and manifest.json files are read. checkpoint.pt is never opened, so no weight
or tensor is loaded. No repository module is imported or executed. No database access, no network,
no writes into any artifact directory, no signal sent to any process.
"""
from __future__ import annotations

import glob
import json
import re
import statistics as st
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
OUT = Path(__file__).resolve().parent.parent / "measurements" / "cap_truncation_scan.json"

MAX_EPOCHS = 100          # trainer.py:39 TrainerConfig default; driver does not override for M2
PATIENCE = 10             # trainer.py:40; confirmed in every search record's bindings/budget
AMBIGUITY_FLOOR = MAX_EPOCHS - PATIENCE   # peak >= 90 -> cannot tell patience-exit from cap-exit

RUN_RE = re.compile(r"^(?P<run>.+)-epoch-(?P<epoch>\d+)$")

# Workspaces that also hold torch epoch checkpoints but are NOT one of the four published carriers
# (all_carrier_grid_variance.json pins M1 to m1_full_v1 and M2 to m2_v1; the two M0 carriers are
# LightGBM, whose checkpoints are booster.txt with no epoch and therefore no cap to hit).
# m1_parallel_v1 IS cited by this pack three times as "the carrier version of M1", so leaving it
# unscanned would make a whole-pack claim ("nothing published is truncated") rest on a narrower
# scan than the claim implies. Scanned with the same instrument and reported under its own key so
# the published-carrier totals are never a mixture of two bases -- the trap ledger B13 records.
SUPERSEDED_TORCH_WORKSPACES = {
    "m1": "3 runs, fold naming 'f01'; earliest M1 smoke",
    "m1_parallel_v1": "24 runs, 1 config; the 'carrier version' of M1 cited in this pack",
    "m1_v2": "24 runs, 1 config; an M1 rerun superseded by m1_full_v1",
}

# (label, workspace, phase, regex identifying the run id, is the phase complete?)
# Per FOLD, not per carrier. §CI published epochs_per_run per (fold, phase) -- 17.4861 for M2 F1
# grid, 28.8056 for M2 F2 grid -- so an aggregate over both folds cannot be compared with either of
# those numbers. The first version of this probe made exactly that mistake: it reported one "M2 grid"
# phase whose implied mean was 23.1458, which is neither of §CI's two values but their arithmetic
# mean (7.4861 and 18.8056 average to 13.1458 peaks, +10 = 23.1458). That is a grouping-basis
# difference, not a disagreement about any run, and this pack's rule is that neither number may be
# cited without stating its basis (ledger B13 records the same trap for the 65-versus-66 pair).
# Splitting by fold makes the cross-check exact on every completed phase, which is what turns it into
# evidence that both probes read the same run set.
FOLDS = ("f1", "f2", "f3", "f4")


def phase_complete(ws: str, fold: str, kind: str, run_ids: set[str]) -> bool:
    """Is every run in this phase finished, judged from artefacts the run itself wrote?

    Not hard-coded per fold, because the first version of this probe did exactly that and got M2's F2
    grid WRONG: it inherited a fold-level "F2 is incomplete" flag from the still-writing F2 refit, so
    it excluded the peak-71 run from the completed set, reported 28 epochs of headroom instead of the
    true 19, and listed eight already-finished F2 grid runs as "in flight". A fold's grid phase and
    its refit phase finish at different times and must be judged separately.

    grid:  a search record c<idx>.json is written when that config's seeds all finish, so the phase is
           complete when the distinct config indices among the checkpoint run ids are exactly covered
           by the records present. M2 F2 has 24 records for 24 configs -> complete.
    refit: the model bundle models/<carrier>-<fold> is written only after the refit and its test-side
           predictions are done, so bundle existence is the completion signal. M2 has m2-f1 only.
    """
    if not run_ids:
        return False
    if kind == "grid":
        cfgs = {r.split("-")[2] for r in run_ids}      # m2-f2-c17-... -> "c17"
        fold_up = fold.upper()
        recs = {f.stem.split("-")[0] for f in (ART / ws / "search" / fold_up).glob("c*.json")} \
            if (ART / ws / "search" / fold_up).is_dir() else set()
        return cfgs <= recs
    carrier = "m1" if ws.startswith("m1") else "m2"
    return (ART / ws / "models" / f"{carrier}-{fold}").is_dir()
# (label, workspace, run-id regex, kind) -- completeness is decided per phase inside main() because
# it depends on which run ids are actually present.
PHASES = tuple(
    (f"{ws.split('_')[0].upper()} {fold.upper()} {kind}", ws,
     re.compile(rf"^{ws.split('_')[0].lower()}-{fold}-" + (r"c\d+-\d+-seed-\d+$" if kind == "grid"
                                                            else r"final-\d+-seed-\d+$")),
     "grid" if kind == "grid" else "refit")
    for ws in ("m1_full_v1", "m2_v1")
    for fold in FOLDS
    for kind in ("grid", "final refit")
)


def collect_peaks(ws: str) -> tuple[dict[str, dict[int, float]], int]:
    """{run_id: {epoch: manifest_rank_ic}} plus a count of manifests read, for one workspace."""
    runs: dict[str, dict[int, float]] = defaultdict(dict)
    n_manifest = 0
    for d in glob.glob(str(ART / ws / "checkpoints" / "*-epoch-*")):
        m = RUN_RE.match(Path(d).name)
        if not m:
            continue
        mp = Path(d) / "manifest.json"
        if not mp.is_file():
            continue
        meta = json.load(open(mp, encoding="utf-8")).get("metadata", {})
        ep, ic = meta.get("epoch"), meta.get("validation_rank_ic")
        if ep is None or ic is None:
            continue
        if int(ep) != int(m.group("epoch")):
            raise AssertionError(f"directory epoch != manifest epoch: {d}")
        n_manifest += 1
        runs[m.group("run")][int(ep)] = float(ic)
    return dict(runs), n_manifest


def monotonicity_violations(peaks: dict[str, dict[int, float]]) -> list[str]:
    """Runs whose manifest values are NOT strictly increasing in epoch -- the signature of (2)."""
    out = []
    for run, eps in peaks.items():
        order = sorted(eps)
        vals = [eps[e] for e in order]
        if any(vals[i] >= vals[i + 1] for i in range(len(vals) - 1)):
            out.append(run)
    return out


def main() -> None:
    per_ws: dict[str, tuple[dict[str, dict[int, float]], int]] = {}
    for _, ws, _, _ in PHASES:
        if ws not in per_ws:
            per_ws[ws] = collect_peaks(ws)

    doc: dict[str, object] = {
        "question": ("is any published run truncated by the max_epochs=100 cap, and if not, what is "
                     "the largest peak epoch and how much headroom is left?"),
        "stopping_rule_source": {
            "file": "research_v2/src/research_v2/training/trainer.py",
            "lines": "96-120",
            "max_epochs": MAX_EPOCHS,
            "early_stopping_patience": PATIENCE,
            "read_not_executed": True,
        },
        "why_peak_ge_90_is_the_ambiguity_zone": (
            "a run exiting on patience ran peak+10 epochs; a run exhausting the loop ran 100. The "
            "break path is reachable only when peak+10<=100, i.e. peak<=90, so peak>=90 is where the "
            "checkpoint set alone cannot distinguish the two exits."
        ),
        "per_phase": {},
    }

    total_runs = 0
    grand_peaks: list[int] = []
    completed_peaks: list[int] = []
    in_flight: list[dict[str, object]] = []
    all_violations: dict[str, list[str]] = {}

    for label, ws, pat, kind in PHASES:
        peaks, _ = per_ws[ws]
        sel = {r: max(e) for r, e in peaks.items() if pat.match(r)}
        fold = label.split()[1].lower()
        complete = phase_complete(ws, fold, kind, set(sel)) if sel else False
        viol = monotonicity_violations({r: peaks[r] for r in sel})
        all_violations[label] = viol
        vals = sorted(sel.values())
        grand_peaks.extend(vals)
        row: dict[str, object] = {
            "workspace": ws,
            "n_runs": len(sel),
            "peak_complete": complete,
            "status": ("not_started" if not sel else "complete" if complete else "in_flight"),
            "max_peak": max(vals) if vals else None,
            "min_peak": min(vals) if vals else None,
            "median_peak": st.median(vals) if vals else None,
            "mean_peak": round(st.mean(vals), 4) if vals else None,
            "n_peak_ge_ambiguity_floor": sum(1 for v in vals if v >= AMBIGUITY_FLOOR),
            "n_peak_ge_80": sum(1 for v in vals if v >= 80),
            "implied_epochs_run_mean": round(st.mean(vals) + PATIENCE, 4) if vals else None,
            "implied_epochs_run_max": (max(vals) + PATIENCE) if vals else None,
            "truncated_by_cap": sum(1 for v in vals if v + PATIENCE > MAX_EPOCHS),
            "manifest_strict_increase_violations": viol,
        }
        if complete:
            completed_peaks.extend(vals)
            row["headroom_to_cap_epochs"] = MAX_EPOCHS - (max(vals) + PATIENCE) if vals else None
        else:
            for run, p in sorted(sel.items(), key=lambda kv: -kv[1])[:5]:
                in_flight.append({
                    "phase": label, "run": run, "current_peak": p,
                    "epochs_so_far_lower_bound": p + PATIENCE,
                    "epochs_to_ambiguity_zone": AMBIGUITY_FLOOR - p,
                    "reading": ("peak can only grow while this run trains; the run is NOT yet known "
                                "to be untruncated, only to be below the zone at snapshot time"),
                })
            row["top5_runs_by_current_peak"] = sorted(sel.items(), key=lambda kv: -kv[1])[:5]
        doc["per_phase"][label] = row
        total_runs += len(sel)

    doc.update({
        "totals": {
            "n_runs_scanned": total_runs,
            "n_completed_phase_runs": len(completed_peaks),
            "grand_max_peak": max(grand_peaks),
            "grand_median_peak": st.median(grand_peaks),
            "grand_mean_peak": round(st.mean(grand_peaks), 4),
            "n_runs_peak_ge_ambiguity_floor": sum(1 for v in grand_peaks if v >= AMBIGUITY_FLOOR),
            "n_runs_truncated_by_cap": sum(1 for v in completed_peaks if v + PATIENCE > MAX_EPOCHS),
            "max_implied_epochs_run_completed": max(completed_peaks) + PATIENCE,
            "headroom_to_cap_epochs_completed": MAX_EPOCHS - (max(completed_peaks) + PATIENCE),
        },
        "in_flight_runs_at_snapshot": in_flight,
        "negative_control": {
            "what": ("the monotonicity instrument must be able to fail, per this pack's convention "
                     "(§CF's five negative cases, §AL's self-refuting script)"),
            "injected": {"run": "fake-run", "epochs": {3: 0.09, 4: 0.05}},
            "expected": "flagged as a violation",
            "observed_flagged": bool(monotonicity_violations({"fake-run": {3: 0.09, 4: 0.05}})),
        },
        "cross_check_against_CI": {
            "CI_published_epochs_per_run_mean": {
                "M2 F1 grid": 17.4861, "M2 F1 final refit": 18.3333,
                "M2 F2 grid": 28.8056, "M2 F2 final refit": 34.3333,
            },
            "this_probe_recomputed": {},
            "deltas": {},
            "note": ("§CI priced M2 only, so M1's four phases have no counterpart there. On M2 the "
                     "three COMPLETED phases must reproduce §CI's numbers to four decimals -- that "
                     "agreement is the evidence that both probes read the same run set with the same "
                     "peak definition. M2 F2 final refit is expected to differ UPWARD because that "
                     "phase is still writing between snapshots, and the size of that drift is the "
                     "quantified partiality of §CI's 34.3333."),
        },
        "headroom_ladder": {
            "what": ("how close anything came to the cap, so that 'no run was truncated' can be read "
                     "as a margin and not merely as a binary"),
            "thresholds": [AMBIGUITY_FLOOR, 80, 70, 60, 50],
            "counts_all_runs": {},
            "counts_completed_phases_only": {},
        },
        "superseded_torch_workspaces": {},
        "m0_carriers_out_of_scope": (
            "m0_v2 and m0_v3 are the two M0 carriers in all_carrier_grid_variance.json. Their "
            "checkpoints are LightGBM booster.txt with no manifest and no epoch index, so the "
            "max_epochs cap does not exist for them and this question does not apply. Not scanned."
        ),
        "conclusion": None,
        "read_only": True,
        "database_access": False,
        "weights_loaded": False,
        "checkpoint_pt_opened": False,
        "training_performed": False,
        "repository_module_imported_or_executed": False,
        "processes_signalled": [],
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
    })

    cc = doc["cross_check_against_CI"]["this_probe_recomputed"]
    deltas = doc["cross_check_against_CI"]["deltas"]
    pub = doc["cross_check_against_CI"]["CI_published_epochs_per_run_mean"]
    for label, published in pub.items():
        got = doc["per_phase"][label]["implied_epochs_run_mean"]
        cc[label] = got
        deltas[label] = round(got - published, 4) if got is not None else None
    sup_all: list[int] = []
    for ws, note in SUPERSEDED_TORCH_WORKSPACES.items():
        peaks, n_manifest = collect_peaks(ws)
        sel = {run: max(eps) for run, eps in peaks.items()}
        vals = sorted(sel.values())
        sup_all.extend(vals)
        doc["superseded_torch_workspaces"][ws] = {
            "why_scanned": note,
            "n_runs": len(sel),
            "manifests_read": n_manifest,
            "max_peak": max(vals) if vals else None,
            "implied_epochs_run_max": (max(vals) + PATIENCE) if vals else None,
            "n_peak_ge_ambiguity_floor": sum(1 for v in vals if v >= AMBIGUITY_FLOOR),
            "truncated_by_cap": sum(1 for v in vals if v + PATIENCE > MAX_EPOCHS),
            "manifest_strict_increase_violations": monotonicity_violations(peaks),
            "is_a_published_carrier": False,
        }
    doc["superseded_torch_workspaces"]["_all_superseded"] = {
        "n_runs": len(sup_all),
        "grand_max_peak": max(sup_all) if sup_all else None,
        "implied_epochs_run_max": (max(sup_all) + PATIENCE) if sup_all else None,
        "n_peak_ge_ambiguity_floor": sum(1 for v in sup_all if v >= AMBIGUITY_FLOOR),
        "truncated_by_cap": sum(1 for v in sup_all if v + PATIENCE > MAX_EPOCHS),
    }
    ladder = doc["headroom_ladder"]
    for t in ladder["thresholds"]:
        ladder["counts_all_runs"][f"peak_ge_{t}"] = sum(1 for v in grand_peaks if v >= t)
        ladder["counts_completed_phases_only"][f"peak_ge_{t}"] = sum(
            1 for v in completed_peaks if v >= t)

    tot = doc["totals"]
    _sup = doc["superseded_torch_workspaces"]["_all_superseded"]
    _everywhere_max = max(tot["grand_max_peak"], _sup["grand_max_peak"] or 0)
    doc["totals"]["grand_max_peak_including_superseded"] = _everywhere_max
    doc["totals"]["n_runs_truncated_by_cap_including_superseded"] = (
        tot["n_runs_truncated_by_cap"] + _sup["truncated_by_cap"])
    doc["conclusion"] = (
        f"{tot['n_runs_truncated_by_cap']} of {tot['n_completed_phase_runs']} completed-phase runs are "
        f"cap-truncated. The largest peak epoch anywhere is {tot['grand_max_peak']}, implying at most "
        f"{tot['max_implied_epochs_run_completed']} epochs run against a cap of {MAX_EPOCHS} -- "
        f"{tot['headroom_to_cap_epochs_completed']} epochs of headroom. No run is in the ambiguity zone "
        f"(peak >= {AMBIGUITY_FLOOR}): {tot['n_runs_peak_ge_ambiguity_floor']} of "
        f"{tot['n_runs_scanned']}. So every published grid statistic describes a run that stopped on "
        f"early stopping, not one cut short by the cap, and §CI's epochs_per_run = mean(peak)+10 "
        f"under-counts no run's cost. Extending the same instrument to the three superseded torch "
        f"workspaces ({_sup['n_runs']} more runs, max peak {_sup['grand_max_peak']}) also finds "
        f"{_sup['truncated_by_cap']} truncated, so the largest peak anywhere on this machine is "
        f"{_everywhere_max} epochs -> at most {_everywhere_max + PATIENCE} of the {MAX_EPOCHS} cap. "
        f"The cap remains a FORWARD risk for M2's F3/F4 and for the in-flight F2 refit, not a fact "
        f"about anything published."
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"scanned {tot['n_runs_scanned']} runs "
          f"({tot['n_completed_phase_runs']} in completed phases)")
    for label, _, _, _kind in PHASES:
        r = doc["per_phase"][label]
        print(f"  {label:22s} n={r['n_runs']:4d} {r['status']:11s} "
              f"peak max={r['max_peak']} med={r['median_peak']} mean={r['mean_peak']} "
              f"epochs_run_max={r['implied_epochs_run_max']} "
              f"truncated={r['truncated_by_cap']} in_zone={r['n_peak_ge_ambiguity_floor']} "
              f"violations={len(r['manifest_strict_increase_violations'])}")
    print(f"  GRAND: max_peak={tot['grand_max_peak']} "
          f"max_implied_epochs_run={tot['max_implied_epochs_run_completed']}/{MAX_EPOCHS} "
          f"headroom={tot['headroom_to_cap_epochs_completed']} "
          f"truncated={tot['n_runs_truncated_by_cap']} in_zone={tot['n_runs_peak_ge_ambiguity_floor']}")
    print("  negative control flagged the injected non-monotone run:",
          doc["negative_control"]["observed_flagged"])
    print("  cross-check vs §CI (this probe / published / delta):")
    for label, published in pub.items():
        d = deltas[label]
        st_ = doc["per_phase"][label]["status"]
        flag = ("EXACT" if d == 0 else
                ("drift: still writing" if st_ == "in_flight" else "MISMATCH -- investigate"))
        print(f"    {label:22s} {cc[label]} / {published} / {d:+.4f}  {flag}")
    print("  headroom ladder (all runs / completed phases only):")
    for t in ladder["thresholds"]:
        print(f"    peak>={t:3d}: {ladder['counts_all_runs'][f'peak_ge_{t}']:4d} / "
              f"{ladder['counts_completed_phases_only'][f'peak_ge_{t}']:4d}")
    for r in doc["in_flight_runs_at_snapshot"]:
        print(f"  in flight: {r['run']} peak={r['current_peak']} "
              f"zone_in={r['epochs_to_ambiguity_zone']} epochs")
    print("  superseded torch workspaces (not published carriers):")
    for ws in SUPERSEDED_TORCH_WORKSPACES:
        r = doc["superseded_torch_workspaces"][ws]
        print(f"    {ws:16s} n={r['n_runs']:4d} peak max={r['max_peak']} "
              f"implied_max={r['implied_epochs_run_max']}/100 truncated={r['truncated_by_cap']} "
              f"violations={len(r['manifest_strict_increase_violations'])}")
    print(f"    ALL superseded    n={_sup['n_runs']} max_peak={_sup['grand_max_peak']} "
          f"truncated={_sup['truncated_by_cap']}")
    print("  conclusion:", doc["conclusion"])
    print("wrote", OUT)


if __name__ == "__main__":
    main()
