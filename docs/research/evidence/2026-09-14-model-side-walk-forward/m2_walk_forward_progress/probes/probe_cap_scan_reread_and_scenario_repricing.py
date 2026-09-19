#!/usr/bin/env python3
"""Read-only probe for §DA: three follow-ups that the live M2 state has now made answerable, plus one
structural explanation for why §CI's refit "uplift factor" can never be a single constant.

(1) §CJ/D13's cap scan, re-read now that the M2 F2 refit has FINISHED. §CR/B24 left two open re-reads:
    "the machine-wide max peak must be re-read once the fit early-stops", and "is the cap risk live at the
    running F2 refit?". The bundle models/m2-f2 exists (07:19), so that fit has definitively ended. The
    scan is redone over every torch run: how many runs are now in a completed stage, the machine-wide max
    peak and WHO holds it, and the counts in the ambiguity region (peak >= 90) and at/over the cap (100).
    §CR predicted the max would be a TIE between the grid run m2-f2-c17 and the refit run
    m2-f2-final-20260802; this checks that prediction rather than assuming it.
    The run classification must be by WORKSPACE and fold and stage, not by a fold-name substring: M1 also
    has an f3, so filtering on "-f3-" to isolate M2's in-progress fold would silently drop M1's completed
    F3 runs. An ad-hoc scan made exactly that mistake and reported 402 completed runs; the correct count
    reconciles to §CJ's published 486 plus the newly-created M2 F3 runs.

(2) §CK.5/D14's four scenarios, re-priced with the FINAL refit uplift 2.362x from §CS in place of §CI's
    1.45x. Every scenario's refit term scales, so all totals move. The published interval 173.5-547.8 h and
    the effect-B lower branch 139.0 h are recomputed, and the percentage shift is reported per scenario so
    the correction's materiality is visible rather than asserted.

(3) WHY §CI's uplift factor is not a constant, and cannot be made one with the data available. The three
    refit seeds run IN PARALLEL (verified: all three seeds of a fold share the same first-checkpoint
    minute), so the refit stage's wall-clock is the SLOWEST seed -- a MAX over seeds. But §CI's cost model
    multiplies k_final by the MEAN epochs/run. A max is not proportional to a mean; their ratio depends on
    how skewed the three seeds' peaks are, and that skew varies by fold (max/mean is 1.440 for F1, 1.954
    for F2). So the "uplift" §CI fitted is partly a mean->max conversion, which is fold-dependent by
    construction -- and on F1, the fold k_final itself was fitted on, the mean-based uplift is exactly
    1.000, which is circular. Testing whether a max-based model is stable instead: its uplift is 0.833 for
    F1 and 1.351 for F2, so it is NOT stable either. With two folds there is no basis to fit a better
    model, so this probe reports the structural reason and declines to propose a replacement constant.

CONTROLS: (i) the run-count classification must reconcile with §CJ's published 486 completed-stage runs
    once the M2 F2 refit is counted as complete -- a mismatch means the classifier is wrong; (ii) the
    scenario arithmetic at uplift 1.45 must reproduce §CK.5's published 173.5 / 263.7(263.8) / 355.7(355.4)
    / 547.8 and effect B's 139.0, proving the re-pricing isolates the uplift change and nothing else;
    (iii) the parallelism claim must be measured (three seeds' first-checkpoint times within a small
    window), not assumed, since the whole max-vs-mean argument rests on it; (iv) the max-based uplift must
    be reported as NOT stable, so the probe cannot be read as having found a fix; (v) every scenario's
    total at 2.362x must exceed its total at 1.45x (the uplift is > 1.45, so this must hold; if any
    scenario moved the other way the arithmetic would be wrong).

Reads only checkpoint directory names and mtimes, the published §CI/§CK constants carried as literals, and
bundle presence. No parquet, no checkpoint.pt, no weight or tensor, no repository module imported or
executed, no database, no network, no writes to any artifact directory, no signal to any process. Does not
touch /tmp/walk_forward_cache_v1, so no atime side effect. Deterministic apart from reading the live M2 F3
grid state, which is reported as a snapshot; the completed-stage figures (M1 and M2 F1/F2) are immutable.
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
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "measurements" / "cap_scan_reread_and_scenario_repricing.json"

# §CI / §CK published constants, carried as literals (not recomputed)
K_GRID = 0.00219327
K_FINAL = 0.00020381721864733542
UP_CI = 1.45                    # §CI.2's uplift, from an unfinished run
UP_FINAL = 2.3622               # §CS's final same-basis uplift (20.59 h / 8.72 h)
GAP_H, OOS_H = 5.0, 2.0
ROWS = {"F1": 2993479, "F2": 3323361, "F3": 3650743, "F4": 3955048}   # measured, §CK.1/A15
ANCHOR_EPR = 28.8056            # M2 F2 grid epochs/run, measured
R_F3_F2, R_F4_F3 = 0.8202, 0.8887   # M1 fold ratios, §CK.5 effect B
CAP = 100
PATIENCE = 10
AMBIGUITY = CAP - PATIENCE      # 90
# §CJ published: 486 torch runs in total, of which 483 were in a completed stage and 3 (the M2 F2
# refit seeds) were "still writing". Those 3 have since finished, so all 486 are now completed-stage and
# the count that must reconcile is §CJ's TOTAL, not its 483. Naming this constant after the 483 would
# have made the probe expect the wrong number; naming it after 486 without this comment would have
# misquoted §CJ.
CJ_TOTAL_RUNS = 486
CJ_COMPLETED_THEN = 483
CK5_PUBLISHED = {"optimistic_epochs_flat": 173.5, "linear_growth_capped": 263.7,
                 "central_geometric_capped": 355.7, "pessimistic_all_hit_cap": 547.8,
                 "effect_B_M1_shaped": 139.0}
SEEDS = ("20260801", "20260802", "20260803")
RUN_RE = re.compile(r"^(?P<run>.+)-epoch-(?P<epoch>\d+)$")


def scan_runs() -> dict:
    """Every torch run in both carriers, classified by workspace / fold / stage."""
    runs: dict[tuple[str, str], list[int]] = defaultdict(list)
    for ws in ("m1_full_v1", "m2_v1"):
        for d in glob.glob(str(ART / ws / "checkpoints" / "*-epoch-*")):
            m = RUN_RE.match(Path(d).name)
            if m:
                runs[(ws, m.group("run"))].append(int(m.group("epoch")))
    per: dict[tuple[str, str, str], list[int]] = defaultdict(list)   # (ws, fold, stage) -> peaks
    peaks: dict[tuple[str, str], int] = {}
    fold_of: dict[tuple[str, str], str] = {}
    for (ws, r), eps in runs.items():
        g = re.match(r"^m[12]-(f\d)-(final|c\d+)-", r)
        if not g:
            continue
        fold = g.group(1)
        stage = "refit" if g.group(2) == "final" else "grid"
        pk = max(eps)
        peaks[(ws, r)] = pk
        fold_of[(ws, r)] = fold
        per[(ws, fold, stage)].append(pk)

    # A CONTROL CAUGHT THIS BUG. The first draft tested `in_progress(k)` with k = (workspace, run_name)
    # but indexed k[1] as if it were the FOLD, so the predicate was never true: M2's in-progress F3 runs
    # were counted as completed (535 instead of 486) and the "in progress" bucket came back empty. The
    # reconciliation control against §CJ's published 486 is what failed and exposed it. Fold is now
    # carried explicitly per run.
    #
    # M2's F3 grid is the only stage still running: models/m2-f3 does not exist and search/F3 has no
    # records yet, so those runs have not reached a completed stage. Note the fold NAME is not enough on
    # its own -- M1 also has an f3, and M1's f3 finished long ago, which is why the workspace must be part
    # of the test (an ad-hoc scan that filtered on the substring "-f3-" silently dropped M1's completed
    # F3 runs and reported 402).
    def in_progress(k):
        return k[0] == "m2_v1" and fold_of[k] == "f3"
    completed = {k: v for k, v in peaks.items() if not in_progress(k)}
    return {"runs": runs, "peaks": peaks, "per_bucket": dict(per), "fold_of": fold_of,
            "completed": completed,
            "in_progress": {k: v for k, v in peaks.items() if in_progress(k)}}


def cap_scan(scan: dict) -> dict:
    comp = scan["completed"]
    mx = max(comp.values())
    holders = sorted(f"{ws}/{r}" for (ws, r), v in comp.items() if v == mx)
    buckets = {}
    for (ws, fold, stage), pks in sorted(scan["per_bucket"].items()):
        buckets[f"{ws}/{fold}/{stage}"] = {"n_runs": len(pks), "max_peak": max(pks)}
    return {
        "n_completed_stage_runs": len(comp),
        "CJ_published_total_runs": CJ_TOTAL_RUNS,
        "CJ_published_completed_then": CJ_COMPLETED_THEN,
        "the_3_that_were_still_writing_have_finished": len(comp) == CJ_TOTAL_RUNS,
        "reconciles_with_CJ": len(comp) == CJ_TOTAL_RUNS,
        "completed_grew_by": len(comp) - CJ_COMPLETED_THEN,
        "machine_wide_max_peak": mx,
        "max_holders": holders,
        "max_is_a_tie": len(holders) > 1,
        "CR_predicted_this_tie": True,
        "n_in_ambiguity_region_peak_ge_90": sum(1 for v in comp.values() if v >= AMBIGUITY),
        "n_at_or_over_cap_100": sum(1 for v in comp.values() if v >= CAP),
        "margin_to_ambiguity_epochs": AMBIGUITY - mx,
        "n_in_progress_runs_m2_f3": len(scan["in_progress"]),
        "in_progress_max_peak_so_far": max(scan["in_progress"].values(), default=None),
        "per_bucket": buckets,
    }


def refit_parallelism() -> dict:
    """Are a fold's three refit seeds concurrent? The whole max-vs-mean argument rests on this."""
    out = {}
    for ws, car, fold in (("m2_v1", "m2", "f1"), ("m2_v1", "m2", "f2")):
        firsts = []
        for s in SEEDS:
            ds = glob.glob(str(ART / ws / "checkpoints" / f"{car}-{fold}-final-{s}-seed-{s}-epoch-*"))
            firsts.append(min(Path(d).stat().st_mtime for d in ds))
        spread_min = (max(firsts) - min(firsts)) / 60.0
        out[f"{fold}"] = {
            "seed_start_times": [datetime.fromtimestamp(f).isoformat(timespec="minutes") for f in firsts],
            "spread_minutes": round(spread_min, 1),
            "concurrent": spread_min <= 5.0,
        }
    return out


def refit_mean_vs_max(ws: str, car: str, fold: str, rows_key: str) -> dict:
    firsts, lasts, peaks = [], [], []
    for s in SEEDS:
        ds = glob.glob(str(ART / ws / "checkpoints" / f"{car}-{fold}-final-{s}-seed-{s}-epoch-*"))
        mts = [Path(d).stat().st_mtime for d in ds]
        firsts.append(min(mts)); lasts.append(max(mts))
        peaks.append(max(int(re.search(r"epoch-(\d+)$", d).group(1)) for d in ds))
    span = (max(lasts) - min(firsts)) / 3600.0
    epr_mean = st.mean(peaks) + PATIENCE
    epr_max = max(peaks) + PATIENCE
    pred_mean = K_FINAL * epr_mean * ROWS[rows_key] / 3600.0
    pred_max = K_FINAL * epr_max * ROWS[rows_key] / 3600.0
    return {"per_seed_peaks": peaks, "stage_span_h": round(span, 3),
            "epr_mean": round(epr_mean, 3), "epr_max": epr_max,
            "pred_mean_h": round(pred_mean, 3), "pred_max_h": round(pred_max, 3),
            "uplift_mean_basis": round(span / pred_mean, 4),
            "uplift_max_basis": round(span / pred_max, 4),
            "peak_skew_max_over_mean": round(max(peaks) / st.mean(peaks), 4)}


def scenarios(uplift: float) -> dict:
    e_flat = (ANCHOR_EPR, ANCHOR_EPR)
    e_lin = (ANCHOR_EPR + 11.3, min(ANCHOR_EPR + 2 * 11.3, CAP))
    e_geo = (ANCHOR_EPR * 1.647, min(ANCHOR_EPR * 1.647 ** 2, CAP))
    e_cap = (CAP, CAP)
    e_b = (ANCHOR_EPR * R_F3_F2, ANCHOR_EPR * R_F3_F2 * R_F4_F3)
    out = {}
    for name, (e3, e4) in (("optimistic_epochs_flat", e_flat), ("linear_growth_capped", e_lin),
                           ("central_geometric_capped", e_geo), ("pessimistic_all_hit_cap", e_cap),
                           ("effect_B_M1_shaped", e_b)):
        grid = (K_GRID * e3 * ROWS["F3"] + K_GRID * e4 * ROWS["F4"]) / 3600.0
        refit = (K_FINAL * e3 * ROWS["F3"] * uplift + K_FINAL * e4 * ROWS["F4"] * uplift) / 3600.0
        total = grid + refit + 4 * GAP_H + OOS_H
        out[name] = {"f3_epochs_per_run": round(e3, 2), "f4_epochs_per_run": round(e4, 2),
                     "grid_h": round(grid, 1), "refit_h": round(refit, 1),
                     "gaps_and_oos_h": round(4 * GAP_H + OOS_H, 1),
                     "total_h": round(total, 1), "total_days": round(total / 24, 1),
                     "refit_share_pct": round(100 * refit / total, 1)}
    return out


def main() -> None:
    scan = scan_runs()
    cap = cap_scan(scan)
    par = refit_parallelism()
    f1 = refit_mean_vs_max("m2_v1", "m2", "f1", "F1")
    f2 = refit_mean_vs_max("m2_v1", "m2", "f2", "F2")
    sc_old = scenarios(UP_CI)
    sc_new = scenarios(UP_FINAL)

    repricing = {}
    for name in sc_old:
        a, b = sc_old[name], sc_new[name]
        repricing[name] = {
            "published_CK5_total_h": CK5_PUBLISHED[name],
            "recomputed_at_1.45_h": a["total_h"],
            "reproduces_CK5": abs(a["total_h"] - CK5_PUBLISHED[name]) <= 0.4,
            "at_final_uplift_2.362_h": b["total_h"],
            "delta_h": round(b["total_h"] - a["total_h"], 1),
            "delta_pct": round(100 * (b["total_h"] - a["total_h"]) / a["total_h"], 1),
            "refit_share_pct_at_1.45": a["refit_share_pct"],
        }

    doc = {
        "question": ("now that the M2 F2 refit has finished: what does §CJ's cap scan read, how do "
                     "§CK.5's scenarios move under §CS's final uplift, and why can §CI's refit uplift "
                     "never be a single constant?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "cap_scan_reread": cap,
        "refit_seed_parallelism": par,
        "refit_mean_vs_max": {"F1": f1, "F2": f2,
                              "why_the_uplift_is_not_a_constant": (
                                  "the three refit seeds run in parallel, so stage wall-clock is the "
                                  "SLOWEST seed (a MAX), while §CI's model multiplies k_final by the MEAN "
                                  "epochs/run; max/mean depends on per-fold peak skew, which differs by "
                                  "fold (F1 1.440, F2 1.954), so the fitted 'uplift' is partly a "
                                  "mean->max conversion and is fold-dependent by construction"),
                              "F1_uplift_is_circular": (
                                  "k_final was FITTED on F1, so F1's mean-basis uplift is exactly 1.000 "
                                  "by construction and carries no information"),
                              "max_basis_is_not_stable_either": (
                                  f"max-basis uplift is {f1['uplift_max_basis']} for F1 and "
                                  f"{f2['uplift_max_basis']} for F2, so switching to a max basis does not "
                                  "produce a constant either; with two folds there is no basis to fit a "
                                  "better model, so no replacement constant is proposed")},
        "scenario_repricing": repricing,
        "summary": {
            "completed_runs_reconcile_with_CJ": cap["reconciles_with_CJ"],
            "machine_max_peak": cap["machine_wide_max_peak"],
            "max_is_a_tie_between_grid_and_refit": cap["max_is_a_tie"],
            "zero_in_ambiguity_region": cap["n_in_ambiguity_region_peak_ge_90"] == 0,
            "zero_at_or_over_cap": cap["n_at_or_over_cap_100"] == 0,
            "all_scenarios_reproduced_at_1.45": all(v["reproduces_CK5"] for v in repricing.values()),
            "all_scenarios_increase_at_final_uplift": all(
                v["delta_pct"] > 0 for v in repricing.values()),
            "new_published_interval_h": [repricing["optimistic_epochs_flat"]["at_final_uplift_2.362_h"],
                                         repricing["pessimistic_all_hit_cap"]["at_final_uplift_2.362_h"]],
            "delta_pct_range": [min(v["delta_pct"] for v in repricing.values()),
                                max(v["delta_pct"] for v in repricing.values())],
        },
        "controls": {},
        "read_only": True, "database_access": False, "weights_loaded": False,
        "checkpoint_pt_opened": False, "parquet_read": False, "training_performed": False,
        "repository_module_imported_or_executed": False, "processes_signalled": [],
        "atime_side_effect": False,
        "live_snapshot_note": ("the M2 F3 grid counts are a live snapshot and will grow; the "
                               "completed-stage figures (M1 all folds, M2 F1/F2) are immutable"),
    }
    doc["controls"] = {
        "i_run_classification_reconciles_with_CJ_total_486": cap["reconciles_with_CJ"],
        "i_b_completed_grew_by_exactly_the_3_still_writing": cap["completed_grew_by"] == 3,
        "ii_scenarios_reproduce_CK5_at_1.45": all(v["reproduces_CK5"] for v in repricing.values()),
        "iii_refit_seeds_are_concurrent": all(v["concurrent"] for v in par.values()),
        "iv_max_basis_uplift_reported_as_not_stable": f1["uplift_max_basis"] != f2["uplift_max_basis"],
        "v_every_scenario_increases_at_final_uplift": all(
            v["delta_pct"] > 0 for v in repricing.values()),
        "vi_no_run_in_ambiguity_region": cap["n_in_ambiguity_region_peak_ge_90"] == 0,
        # M1's F3 is complete and MUST be counted as completed, while M2's F3 is in progress and must not.
        # Confusing the two is the exact trap that produced the wrong 402 and the wrong 535.
        "vii_M1_f3_counted_as_completed": all(
            fold_of_ok for fold_of_ok in [
                all((k[0] == "m1_full_v1") == (k not in scan["in_progress"])
                    for k in scan["peaks"] if scan["fold_of"][k] == "f3")]),
        "viii_M2_f3_all_in_progress": all(
            k[0] == "m2_v1" for k in scan["in_progress"]) and len(scan["in_progress"]) > 0,
        "ALL_PASS": None,
    }
    doc["controls"]["ALL_PASS"] = all(
        v for k, v in doc["controls"].items() if k != "ALL_PASS")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    c = cap
    print("=== (1) §CJ cap scan, re-read with the F2 refit complete ===")
    print(f"  completed-stage runs: {c['n_completed_stage_runs']}")
    print(f"    §CJ published {CJ_TOTAL_RUNS} total / {CJ_COMPLETED_THEN} completed (3 still writing); "
          f"those 3 finished, so completed grew by {c['completed_grew_by']} to equal §CJ's total "
          f"-> reconciles: {c['reconciles_with_CJ']}")
    print(f"  machine-wide max peak: {c['machine_wide_max_peak']} held by {len(c['max_holders'])} run(s):")
    for h in c["max_holders"]:
        print(f"      {h}")
    print(f"  tie as §CR predicted: {c['max_is_a_tie']}")
    print(f"  ambiguity region (peak>=90): {c['n_in_ambiguity_region_peak_ge_90']} | at/over cap 100: "
          f"{c['n_at_or_over_cap_100']} | margin to 90: {c['margin_to_ambiguity_epochs']} epochs")
    print(f"  M2 F3 grid in progress: {c['n_in_progress_runs_m2_f3']} runs, max peak so far "
          f"{c['in_progress_max_peak_so_far']}")
    print("\n=== (3) refit seeds concurrent? (the max-vs-mean argument rests on this) ===")
    for k, v in par.items():
        print(f"  {k}: starts {v['seed_start_times']} spread {v['spread_minutes']} min "
              f"-> concurrent={v['concurrent']}")
    print("\n  mean basis vs max basis:")
    for lbl, r in (("F1", f1), ("F2", f2)):
        print(f"    {lbl}: peaks {r['per_seed_peaks']} skew max/mean {r['peak_skew_max_over_mean']} | "
              f"span {r['stage_span_h']} h | uplift mean-basis {r['uplift_mean_basis']} "
              f"max-basis {r['uplift_max_basis']}")
    print("\n=== (2) scenarios re-priced at the final uplift 2.362x ===")
    print(f"  {'scenario':28s} {'@1.45':>8s} {'CK5 pub':>8s} {'repro':>6s} {'@2.362':>8s} {'delta':>7s} {'%':>6s}")
    for name, v in repricing.items():
        print(f"  {name:28s} {v['recomputed_at_1.45_h']:>8.1f} {v['published_CK5_total_h']:>8.1f} "
              f"{str(v['reproduces_CK5']):>6s} {v['at_final_uplift_2.362_h']:>8.1f} "
              f"{v['delta_h']:>+7.1f} {v['delta_pct']:>5.1f}%")
    s = doc["summary"]
    print(f"\n  new interval: {s['new_published_interval_h'][0]} - {s['new_published_interval_h'][1]} h "
          f"(was 173.5 - 547.8); shift range {s['delta_pct_range']}")
    print("\n=== controls ===")
    for k, v in doc["controls"].items():
        print(f"  {'OK ' if v else '*** FAIL ***'} {k}")
    assert doc["controls"]["ALL_PASS"], "a control failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
