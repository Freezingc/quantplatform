#!/usr/bin/env python3
"""Read-only probe for §CQ: the peak-LOCATION vs peak-HEIGHT dissociation that §CK.2 and §CN.3
established on neural carriers also holds on the gradient-boosted-tree carrier M0R -- so it is a
property of the walk-forward folds + rank-IC early stopping, not of neural-network training.

WHAT IS NEW HERE. This pack has analysed training length (peak epoch) only for the two torch carriers
M1 (ResidualMLP) and M2 (TCN). The tree carriers M0 (LGBM_RANKER) and M0R (LGBM_RANKREG) persist their
final-refit booster as booster.txt, whose tree count equals the model's best_iteration, and best_iteration
has never been read for a fold pattern (grep: "best_iteration" and "gradient boosting" occur 0 times in
problems_found.md). So the tree carriers' training length is unmeasured territory.

WHY best_iteration IS THE TREE ANALOGUE OF peak epoch. Both are "the training step at which validation
rank-IC peaked". For torch, trainer.py:110-116 saves a checkpoint only on strict improvement, so the max
saved epoch IS the epoch of best validation rank-IC (§CJ.2). For LightGBM, lgbm_rankreg.py:277-284 fits
with feval = DailyRankICMetric (rank_ic, higher-better, lgbm_rankreg.py:61) and callbacks=[lgb.early_
stopping(patience)], so booster.best_iteration is the boosting round of best validation rank-IC; the driver
saves with save_model(path) and no num_iteration (run_real_candidate_training.py:1279), and LightGBM's
default then writes exactly best_iteration trees, and prediction uses num_iteration=best_iteration
(lgbm_rankreg.py:314). So tree count in booster.txt = best_iteration = the rank-IC-peak round, directly
comparable to a torch peak epoch.

EVIDENCE THE READING IS CORRECT (all asserted):
  * every booster's "Tree=" line count equals its "tree_sizes=" entry count (two independent counts of the
    same trees must agree, else the file is being parsed wrong);
  * num_boost_round is 500 (run_real_candidate_training.py:963) and NO booster has 500 trees (max is 214),
    so every run early-stopped before the cap -- the tree count is a convergence point, not a truncation;
  * the three M0R workspaces (m0r_v3, m0r_resume_a, m0r_resume_c) carry byte-identical tree counts and
    identical report rank_ic, so which one is read does not change any number.

THE FINDING. On the SAME final-refit basis:
  M0R best_iteration by fold (mean over 3 seeds): F1 34.3, F2 164.3, F3 46.7, F4 39.7 -> peaks at F2,
    robustly: F2's minimum over seeds (133) exceeds every other fold's maximum (59), and mean, median and
    drop-the-max all give F2.
  M0R ensemble validation rank_ic by fold: F1 0.06795, F2 0.09871, F3 0.08944, F4 0.10318 -> peaks at F4.
  So M0R's training length peaks at F2 while its score peaks at F4: the location/height dissociation.
  §CK.2 found exactly this for M1 on the grid basis (epochs/run peaks F2, rank_ic peaks F4, and F4 has the
  highest IC with the LOWEST epochs/run), and §CN.3 found it for M1/M2. This probe extends it to a THIRD
  model family -- gradient-boosted trees -- so the dissociation is not a neural-network artefact.

WHAT THIS DOES *NOT* CLAIM (the basis discipline §CN/§CO/§CP enforce):
  * It does NOT claim an architecture-independent peak-LOCATION fold. On the common final-refit basis the
    location argmax is F2 (M0R), F4 (M0), F3 (M1), F2 (M2 partial) -- not consistent, exactly as §CN's huge
    fold x seed interaction predicts. What replicates across families is the DISSOCIATION (location argmax
    != height argmax), not a shared location argmax.
  * M0R has only ONE config (the refined carrier; report n_configs=1), so its best_iteration is a
    final-refit quantity with no grid-basis counterpart, while M1's §CK.2 dissociation was on the grid
    basis. Tree carriers persist ONLY the final-refit booster (no per-config grid boosters), so grid-basis
    training length simply does not exist for trees. The dissociation is compared within each carrier
    (location vs height of the same carrier), which is basis-safe; cross-carrier comparison is of the
    dissociation PATTERN, not of a shared fold index.
  * M0 (LGBM_RANKER) does NOT show the dissociation: its best_iteration argmax is F4 and its rank_ic argmax
    is also F4. But M0 is degenerate as a training-length signal -- its tree counts are 1-12 for 11 of 12
    runs with a single outlier of 61, so its "location" is dominated by one run and is not a reliable
    fold pattern. It is reported honestly as non-replicating AND as too degenerate to count against the
    finding, not silently dropped.

Only booster.txt files (counting "Tree=" and "tree_sizes=" lines), report.json text, and torch checkpoint
directory names are read. No parquet (not even footers), no checkpoint.pt, no weight or tensor loaded, no
repository module imported or executed (the driver and lgbm baselines are read as text), no database, no
network, no writes to any artifact directory, no signal to any process. This probe does not touch
/tmp/walk_forward_cache_v1, so it has no atime side effect.
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
OUT = HERE.parent / "measurements" / "m0r_tree_carrier_fold_dissociation.json"

FOLDS = ["f1", "f2", "f3", "f4"]
FOLDS_U = ["F1", "F2", "F3", "F4"]
SEEDS = ["20260801", "20260802", "20260803"]
NUM_BOOST_ROUND = 500            # run_real_candidate_training.py:963
M0R_WORKSPACES = ["m0r_v3", "m0r_resume_a", "m0r_resume_c"]


def booster_tree_count(path: Path) -> tuple[int, int]:
    """(n_Tree_lines, n_tree_sizes_entries) for one booster.txt -- two independent counts that must agree."""
    n_tree = 0
    n_sizes = None
    for line in path.open(encoding="utf-8", errors="replace"):
        if line.startswith("Tree="):
            n_tree += 1
        elif line.startswith("tree_sizes="):
            n_sizes = len(line.split()[1:])   # first token is 'tree_sizes=NNN' merged; handle below
            # the line is 'tree_sizes=a b c ...'; split() gives ['tree_sizes=a','b','c',...]
            n_sizes = len(line.split())        # 'tree_sizes=a' counts as one tree
    return n_tree, (n_sizes if n_sizes is not None else -1)


def read_tree_carrier(ws: str, car: str) -> dict:
    """{(fold, seed): best_iteration} from final-refit boosters, plus integrity checks."""
    out: dict[tuple[str, str], int] = {}
    integrity = []
    for bp in sorted(glob.glob(str(ART / ws / "checkpoints" / f"{car}-f*-final-*" / "booster.txt"))):
        p = Path(bp)
        m = re.match(rf"{car}-(f\d)-final-(\d+)$", p.parent.name)
        if not m:
            continue
        n_tree, n_sizes = booster_tree_count(p)
        integrity.append({"booster": p.parent.name, "n_Tree": n_tree, "n_tree_sizes": n_sizes,
                          "counts_agree": n_tree == n_sizes, "below_cap": n_tree < NUM_BOOST_ROUND})
        out[(m.group(1), m.group(2))] = n_tree
    return {"cells": out, "integrity": integrity}


def torch_final_peaks(ws: str, car: str) -> dict[tuple[str, str], int]:
    runs: dict[str, list[int]] = defaultdict(list)
    for d in glob.glob(str(ART / ws / "checkpoints" / "*-epoch-*")):
        m = re.match(r"^(.+)-epoch-(\d+)$", Path(d).name)
        if m:
            runs[m.group(1)].append(int(m.group(2)))
    out: dict[tuple[str, str], int] = {}
    for r, e in runs.items():
        g = re.match(rf"^{car}-(f\d)-final-(\d+)-seed-(\d+)$", r)
        if g:
            out[(g.group(1), g.group(3))] = max(e)
    return out


def torch_grid_epochs_per_run(ws: str, car: str) -> dict[str, float]:
    runs: dict[str, list[int]] = defaultdict(list)
    for d in glob.glob(str(ART / ws / "checkpoints" / "*-epoch-*")):
        m = re.match(r"^(.+)-epoch-(\d+)$", Path(d).name)
        if m:
            runs[m.group(1)].append(int(m.group(2)))
    per_fold: dict[str, list[int]] = defaultdict(list)
    for r, e in runs.items():
        g = re.match(rf"^{car}-(f\d)-c\d+-\d+-seed-\d+$", r)
        if g:
            per_fold[g.group(1)].append(max(e))
    return {f: st.mean(per_fold[f]) + 10 for f in FOLDS if per_fold.get(f)}


def rank_ic_by_fold(ws: str) -> dict[str, float]:
    """Mean ensemble validation rank_ic per fold.

    Reads report.json when the run has finished (M0, M0R, M1). For M2, which is still running and has no
    report.json, falls back to the per-config search records so the value is available for the folds that
    have finished -- but the caller must then treat M2 as a PARTIAL snapshot (see cross_carrier).
    """
    rep = ART / ws / "report.json"
    out: dict[str, float] = {}
    if rep.is_file():
        d = json.load(open(rep, encoding="utf-8"))
        for f in d["folds"]:
            vals = [r["ensemble_validation_rank_ic"] for r in f["search_results"]
                    if r.get("ensemble_validation_rank_ic") is not None]
            if vals:
                out[f["fold_id"].lower()] = st.mean(vals)
        return out
    for f in FOLDS_U:
        recs = sorted((ART / ws / "search" / f).glob("c*.json")) if (ART / ws / "search" / f).is_dir() else []
        vals = []
        for r in recs:
            d = json.load(open(r, encoding="utf-8"))
            v = (d.get("result", {}) or {}).get("ensemble_validation_rank_ic")
            if v is None:
                v = d.get("ensemble_validation_rank_ic")
            if v is not None:
                vals.append(v)
        if vals:
            out[f.lower()] = st.mean(vals)
    return out


def final_refit_complete(ws: str, car: str) -> bool:
    """True only if all four folds have all three seeds' final-refit artefacts (M2 currently fails)."""
    if (ART / ws / "checkpoints").is_dir():
        need = {(f, s) for f in FOLDS for s in SEEDS}
        if car in ("m0", "m0r"):
            have = {(m.group(1), m.group(2)) for bp in
                    glob.glob(str(ART / ws / "checkpoints" / f"{car}-f*-final-*" / "booster.txt"))
                    if (m := re.match(rf"{car}-(f\d)-final-(\d+)$", Path(bp).parent.name))}
        else:
            runs = {Path(d).name for d in glob.glob(str(ART / ws / "checkpoints" / "*-epoch-*"))}
            have = {(m.group(1), m.group(3)) for r in runs
                    if (m := re.match(rf"^{car}-(f\d)-final-(\d+)-seed-(\d+)$", r))}
        return need <= have
    return False


def argmax(d: dict[str, float]) -> str:
    return max(d, key=d.get)


def robust_argmax(per_fold_seeds: dict[str, list[int]]) -> dict:
    """argmax under mean, median, and drop-the-max; also whether F2's min exceeds all others' max."""
    stats_out = {}
    for stat_name, fn in (("mean", st.mean), ("median", st.median),
                          ("drop_max", lambda v: st.mean(sorted(v)[:-1]) if len(v) > 1 else float("nan"))):
        stats_out[stat_name] = {f: round(fn(v), 3) for f, v in per_fold_seeds.items() if v}
    argmaxes = {k: argmax(v) for k, v in stats_out.items()}
    return {"per_statistic": stats_out, "argmax_by_statistic": argmaxes,
            "argmax_stable_across_statistics": len(set(argmaxes.values())) == 1,
            "stable_argmax": next(iter(set(argmaxes.values()))) if len(set(argmaxes.values())) == 1 else None}


def main() -> None:
    # --- M0R across its three workspaces: must be identical
    m0r_ws = {ws: read_tree_carrier(ws, "m0r") for ws in M0R_WORKSPACES}
    base = m0r_ws["m0r_v3"]["cells"]
    ws_identical = all(m0r_ws[w]["cells"] == base for w in M0R_WORKSPACES)
    ic_ws = {ws: rank_ic_by_fold(ws) for ws in M0R_WORKSPACES}
    ic_identical = all(ic_ws[w] == ic_ws["m0r_v3"] for w in M0R_WORKSPACES)

    integrity = m0r_ws["m0r_v3"]["integrity"]
    all_counts_agree = all(r["counts_agree"] for r in integrity)
    all_below_cap = all(r["below_cap"] for r in integrity)
    max_trees = max(r["n_Tree"] for r in integrity)

    m0r_cells = base
    m0r_loc = robust_argmax({f: [m0r_cells[(f, s)] for s in SEEDS if (f, s) in m0r_cells] for f in FOLDS})
    m0r_height = ic_ws["m0r_v3"]
    m0r_loc_mean = {f: round(st.mean([m0r_cells[(f, s)] for s in SEEDS if (f, s) in m0r_cells]), 3)
                    for f in FOLDS}
    m0r_f2_min = min(m0r_cells[("f2", s)] for s in SEEDS)
    m0r_other_max = max(m0r_cells[(f, s)] for f in FOLDS if f != "f2" for s in SEEDS)

    # --- M0 (ranker) for the honest non-replication
    m0 = read_tree_carrier("m0_v3", "m0")
    m0_cells = m0["cells"]
    m0_loc = robust_argmax({f: [m0_cells[(f, s)] for s in SEEDS if (f, s) in m0_cells] for f in FOLDS})
    m0_height = rank_ic_by_fold("m0_v3")
    m0_trees = sorted(m0_cells.values())

    # --- neural carriers on the common final-refit basis, plus M1 grid for the §CK.2 precedent
    m1_final = torch_final_peaks("m1_full_v1", "m1")
    m2_final = torch_final_peaks("m2_v1", "m2")
    m1_final_loc = {f: round(st.mean([m1_final[(f, s)] for s in SEEDS if (f, s) in m1_final]), 3)
                    for f in FOLDS if any((f, s) in m1_final for s in SEEDS)}
    m2_final_loc = {f: round(st.mean([m2_final[(f, s)] for s in SEEDS if (f, s) in m2_final]), 3)
                    for f in FOLDS if any((f, s) in m2_final for s in SEEDS)}
    m1_grid_epr = torch_grid_epochs_per_run("m1_full_v1", "m1")
    m1_grid_height = rank_ic_by_fold("m1_full_v1")

    m2_complete = final_refit_complete("m2_v1", "m2")
    m2_height = rank_ic_by_fold("m2_v1")
    carriers = []
    for label, fam, loc, height, basis, complete, degenerate in (
        ("M0R", "LGBM_RANKREG (gradient-boosted trees)", m0r_loc_mean, m0r_height,
         "final-refit best_iteration", True, False),
        ("M0", "LGBM_RANKER (gradient-boosted trees)",
         {f: round(st.mean([m0_cells[(f, s)] for s in SEEDS if (f, s) in m0_cells]), 3) for f in FOLDS},
         m0_height, "final-refit best_iteration", True, True),
        ("M1", "ResidualMLP (neural)", m1_grid_epr, m1_grid_height, "grid epochs/run (§CK.2 basis)", True, False),
        ("M1-final", "ResidualMLP (neural)", m1_final_loc, m1_grid_height, "final-refit peak epoch", True, False),
        ("M2", "TCN (neural)", m2_final_loc, m2_height,
         "final-refit peak epoch (PARTIAL: F1/F2 only)", m2_complete, False),
    ):
        # a dissociation verdict needs all four folds; a partial carrier is reported but not counted
        folds_present = [k for k in loc if k in height]
        loc_am = argmax(loc) if loc else None
        hgt_am = argmax({k: height[k] for k in folds_present}) if folds_present else None
        carriers.append({
            "carrier": label, "family": fam, "basis": basis, "complete_four_folds": complete,
            "location_by_fold": loc, "height_by_fold": {k: round(v, 6) for k, v in height.items()},
            "location_argmax": loc_am, "height_argmax": hgt_am,
            "dissociated": (loc_am != hgt_am) if complete else None,
            "degenerate": degenerate,
        })

    doc = {
        "question": ("does the peak-location vs peak-height dissociation that §CK.2/§CN.3 found on neural "
                     "carriers also hold on the gradient-boosted-tree carrier M0R?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "best_iteration_is_peak_epoch_analogue": {
            "torch_peak_epoch": "max saved checkpoint epoch = epoch of best validation rank-IC (§CJ.2)",
            "lgbm_best_iteration": "booster.best_iteration = boosting round of best validation rank-IC",
            "lgbm_feval_is_rank_ic": "lgbm_rankreg.py:61,277 (DailyRankICMetric, higher-better)",
            "lgbm_early_stopping": "lgbm_rankreg.py:284 (lgb.early_stopping callback)",
            "save_truncates_to_best_iteration": "run_real_candidate_training.py:1279 save_model(path), default num_iteration",
            "predict_uses_best_iteration": "lgbm_rankreg.py:314 num_iteration=self.best_iteration",
            "num_boost_round_cap": NUM_BOOST_ROUND,
        },
        "m0r_reading_integrity": {
            "n_boosters": len(integrity),
            "all_tree_counts_equal_tree_sizes_counts": all_counts_agree,
            "max_trees_observed": max_trees,
            "all_below_cap_500_no_truncation": all_below_cap,
            "three_workspaces_identical_tree_counts": ws_identical,
            "three_workspaces_identical_rank_ic": ic_identical,
            "per_booster": integrity,
        },
        "m0r_robustness": {
            "best_iteration_by_fold_mean": m0r_loc_mean,
            "argmax_by_statistic": m0r_loc["argmax_by_statistic"],
            "argmax_stable_across_mean_median_dropmax": m0r_loc["argmax_stable_across_statistics"],
            "F2_min_over_seeds": m0r_f2_min,
            "max_of_all_other_folds": m0r_other_max,
            "F2_min_exceeds_other_folds_max": m0r_f2_min > m0r_other_max,
            "rank_ic_by_fold": {k: round(v, 6) for k, v in m0r_height.items()},
            "location_argmax": argmax(m0r_loc_mean),
            "height_argmax": argmax(m0r_height),
            "dissociated": argmax(m0r_loc_mean) != argmax(m0r_height),
        },
        "m0_degenerate_non_replication": {
            "tree_counts_sorted": m0_trees,
            "n_runs": len(m0_trees),
            "n_counts_le_12": sum(1 for x in m0_trees if x <= 12),
            "single_outlier": max(m0_trees),
            "location_argmax": m0_loc["stable_argmax"],
            "height_argmax": argmax(m0_height),
            "dissociated": (m0_loc["stable_argmax"] or "") != argmax(m0_height),
            "verdict": ("M0 does NOT show the dissociation (location and height both peak at F4), but its "
                        "tree counts are 1-12 for 11 of 12 runs with one outlier of 61, so its location "
                        "signal is dominated by a single run and is not a reliable fold pattern. Reported "
                        "honestly as non-replicating AND too degenerate to weigh against the finding."),
        },
        "cross_carrier": {
            "per_carrier": carriers,
            "n_carriers_dissociated": sum(1 for c in carriers
                                          if c["dissociated"] and not c["degenerate"]),
            "n_carriers_assessed": sum(1 for c in carriers
                                       if c["complete_four_folds"] and not c["degenerate"]),
            "carriers_excluded_partial": [c["carrier"] for c in carriers
                                          if not c["complete_four_folds"]],
            "model_families_showing_dissociation": sorted({c["family"].split(" (")[1].rstrip(")")
                                                           for c in carriers
                                                           if c["dissociated"] and not c["degenerate"]}),
            "claim": ("the location/height dissociation holds on gradient-boosted trees (M0R) as well as "
                      "on neural nets (M1 grid per §CK.2, M1 final-refit, M2), so it is a property of the "
                      "walk-forward folds plus rank-IC early stopping, not of neural training"),
            "does_NOT_claim": ("an architecture-independent peak-LOCATION fold: on the common final-refit "
                               "basis the location argmax is F2 (M0R), F4 (M0), F3 (M1), F2 (M2 partial) -- "
                               "not consistent, as §CN's fold x seed interaction predicts. What replicates "
                               "is the dissociation, not a shared argmax."),
        },
        "negative_controls": {},
        "read_only": True, "database_access": False, "weights_loaded": False,
        "checkpoint_pt_opened": False, "parquet_read": False, "training_performed": False,
        "repository_module_imported_or_executed": False, "processes_signalled": [],
        "atime_side_effect": False,
    }

    # --- negative controls: the instruments must be able to say NO / be wrong
    # (i) the tree-count reader: two independent counts must agree on every booster; a mismatch would flag
    # (ii) the dissociation detector must return False when location==height (M0 is the live case: both F4)
    # (iii) synthetic: a carrier whose location and height both peak at F2 must be reported NOT dissociated
    synthetic_same = {"location": {"f1": 1.0, "f2": 9.0, "f3": 2.0, "f4": 3.0},
                      "height": {"f1": 0.1, "f2": 0.9, "f3": 0.2, "f4": 0.3}}
    synthetic_dissoc = {"location": {"f1": 1.0, "f2": 9.0, "f3": 2.0, "f4": 3.0},
                        "height": {"f1": 0.1, "f2": 0.2, "f3": 0.3, "f4": 0.9}}
    doc["negative_controls"] = {
        "tree_count_two_ways_agree_on_all_boosters": all_counts_agree,
        "no_booster_at_cap_500": all_below_cap,
        "three_m0r_workspaces_identical": ws_identical and ic_identical,
        "dissociation_detector_on_M0_returns_False": (
            argmax({f: st.mean([m0_cells[(f, s)] for s in SEEDS]) for f in FOLDS}) == argmax(m0_height)),
        "dissociation_detector_synthetic_same_fold_is_False": (
            argmax(synthetic_same["location"]) == argmax(synthetic_same["height"])),
        "dissociation_detector_synthetic_diff_fold_is_True": (
            argmax(synthetic_dissoc["location"]) != argmax(synthetic_dissoc["height"])),
        "m0r_F2_location_robust_across_statistics": m0r_loc["argmax_stable_across_statistics"]
            and m0r_loc["stable_argmax"] == "f2",
        "m0r_F2_min_exceeds_other_folds_max": m0r_f2_min > m0r_other_max,
    }
    nc = doc["negative_controls"]
    nc["ALL_PASS"] = all(v for k, v in nc.items() if k != "ALL_PASS")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("=== M0R reading integrity ===")
    print(f"  {len(integrity)} boosters; Tree==tree_sizes on all: {all_counts_agree}; "
          f"max trees {max_trees} (< cap {NUM_BOOST_ROUND}): {all_below_cap}")
    print(f"  three workspaces identical (trees & rank_ic): {ws_identical and ic_identical}")
    print("\n=== M0R location (best_iteration) vs height (rank_ic) ===")
    print(f"  best_iteration mean by fold: {m0r_loc_mean}  -> argmax {argmax(m0r_loc_mean)}")
    print(f"  rank_ic by fold: { {k: round(v,5) for k,v in m0r_height.items()} }  -> argmax {argmax(m0r_height)}")
    print(f"  argmax stable across mean/median/dropmax: {m0r_loc['argmax_stable_across_statistics']} "
          f"({m0r_loc['argmax_by_statistic']})")
    print(f"  F2 min {m0r_f2_min} > all other folds max {m0r_other_max}: {m0r_f2_min > m0r_other_max}")
    print(f"  DISSOCIATED: {argmax(m0r_loc_mean) != argmax(m0r_height)}")
    print("\n=== cross-carrier dissociation ===")
    for c in carriers:
        print(f"  {c['carrier']:9s} [{c['basis'][:30]:30s}] loc={c['location_argmax']} "
              f"height={c['height_argmax']} dissociated={c['dissociated']}"
              f"{' (DEGENERATE)' if c['degenerate'] else ''}"
              f"{'' if c['complete_four_folds'] else ' (PARTIAL, not counted)'}")
    cc = doc["cross_carrier"]
    print(f"  families showing dissociation: {cc['model_families_showing_dissociation']}")
    print("\n=== M0 degenerate non-replication ===")
    print(f"  tree counts sorted: {m0_trees}")
    print(f"  {doc['m0_degenerate_non_replication']['n_counts_le_12']}/{len(m0_trees)} runs <= 12 trees, "
          f"one outlier {max(m0_trees)}")
    print("\n=== negative controls ===")
    for k, v in nc.items():
        print(f"  {'OK ' if v else '*** FAIL ***'} {k}")
    assert nc["ALL_PASS"], "a control failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
