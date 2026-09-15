#!/usr/bin/env python3
"""Read-only probe: the training-dynamics half of §BO.9 -- is M1's noise unstable optimisation?

The companion probe (probe_m1_config_effect_zero_cause.py) shows that M1's zero config effect is not
an artefact of power, and that the reason blocking cannot rescue it is structural: seed identity
explains 95.2% of M2's within-(config,fold) variation but only 4.2% of M1's, so 95.8% of M1's
run-to-run variation is not attributable to the seed and no blocking scheme removes it. This probe
asks what that unremovable variation actually is, using the checkpoint manifests.

CRITICAL CONSTRAINT discovered while writing this probe, and the reason it never mixes two metrics:
the per-epoch `validation_rank_ic` inside checkpoints/*/manifest.json is NOT the same quantity as the
search record's `seed_validation_rank_ic`. On m1-f1-c0-seed-20260801 the best saved manifest value is
0.05771882804489137 at epoch 6 while the search record reports 0.04697782260461206 for that seed --
the reported value is lower than every saved epoch including epoch 1. So the two are different
estimators (different row set or different aggregation). This probe therefore uses manifest values
ONLY for training-dynamics statistics (where a run peaked, how many improvements it made, how
erratic the peak epoch is) and NEVER as a substitute for the decomposed metric. The mismatch itself
is measured across all runs rather than asserted from one example, because §BP and §AU both turned on
whether two RankIC quantities are the same estimator.

Only manifest.json files are read. checkpoint.pt files are never opened, so no weight or tensor is
loaded. No database access, no repository module imported or executed, nothing written into any
artifact directory, no signal sent to any process. M2 (pid 3588838) is still training while this runs,
which is why M2's checkpoint set is read as a snapshot and its counts are reported as partial.
"""
from __future__ import annotations

import glob
import json
import math
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
OUT = Path(__file__).resolve().parent.parent / "measurements" / "m1_training_dynamics.json"
CARRIERS = (("M1", "m1_full_v1", 324), ("M2", "m2_v1", 288))
RUN_RE = re.compile(r"^(?P<run>.+)-epoch-(?P<epoch>\d+)$")


def collect(ws: Path):
    """{(fold, cfg_index, seed): {epoch: val_ic}} from manifest.json only."""
    runs = defaultdict(dict)
    n_manifest = 0
    excluded_final_refits = set()
    for d in glob.glob(str(ws / "checkpoints" / "*-epoch-*")):
        m = RUN_RE.match(Path(d).name)
        if not m:
            continue
        mp = Path(d) / "manifest.json"
        if not mp.is_file():
            continue
        meta = json.load(open(mp, encoding="utf-8")).get("metadata", {})
        ic = meta.get("validation_rank_ic")
        ep = meta.get("epoch")
        if ic is None or ep is None:
            continue
        n_manifest += 1
        name = m.group("run")
        mm = re.match(r"^[a-z0-9]+-f(?P<fold>\d+)-c(?P<cfg>\d+)-(?P<tag>\d+)-seed-(?P<seed>\d+)$", name)
        if mm:
            key = (f"F{mm.group('fold')}", int(mm.group("cfg")), str(mm.group("seed")))
        else:
            # e.g. "m1-f1-final-20260802-seed-20260802": the post-selection refit of the winning
            # config, not a grid point. Kept under a None config index so every per-config statistic
            # below skips it, and counted so the exclusion is visible rather than silent.
            key = (name, None, None)
            excluded_final_refits.add(name)
        runs[key][int(ep)] = float(ic)
    return dict(runs), n_manifest, sorted(excluded_final_refits)


def search_records(ws: Path):
    """{(fold, cfg_index, seed): reported seed_validation_rank_ic} from search/<fold>/c*.json."""
    out = {}
    cfg_by_fold = {}
    for d in sorted(glob.glob(str(ws / "search" / "F*"))):
        fold = Path(d).name
        paths = sorted(glob.glob(d + "/*.json"))
        keys = []
        for p in paths:
            j = json.load(open(p, encoding="utf-8"))
            keys.append(json.dumps(j["config"], sort_keys=True))
        order = {k: i for i, k in enumerate(sorted(set(keys)))}
        cfg_by_fold[fold] = order
        for p in paths:
            j = json.load(open(p, encoding="utf-8"))
            r = j.get("result", {})
            sric = r.get("seed_validation_rank_ic") or {}
            ci = order[json.dumps(j["config"], sort_keys=True)]
            for s, v in sric.items():
                out[(fold, ci, str(s))] = float(v)
    return out


def config_axes(ws: Path):
    """{cfg_index: {axis: level}} using F1's sorted config keys, matching search_records' indexing."""
    d = sorted(glob.glob(str(ws / "search" / "F*")))[0]
    keys = sorted({json.dumps(json.load(open(p, encoding="utf-8"))["config"], sort_keys=True)
                   for p in glob.glob(d + "/*.json")})
    return {i: json.loads(k) for i, k in enumerate(keys)}


def best_epoch_by_axis(runs, axes):
    """Does the peak epoch depend on the searched axis? A config effect on dynamics that the metric
    never shows would mean the axis changes training but not the selected model's quality."""
    vals = defaultdict(lambda: defaultdict(list))
    for (_fold, ci, _seed), eps in runs.items():
        if ci is None or ci not in axes or not eps:
            continue
        be = max(eps, key=lambda e: eps[e])
        for name, lvl in axes[ci].items():
            vals[name][str(lvl)].append(be)
    out = {}
    for name, by_lvl in vals.items():
        if len(by_lvl) < 2:
            continue
        groups = [np.asarray(v, dtype=float) for v in by_lvl.values() if len(v) >= 2]
        p_kw = float(stats.kruskal(*groups).pvalue) if len(groups) >= 2 else None
        out[name] = {"levels": {k: {"n": len(v), "mean_best_epoch": float(np.mean(v)),
                                    "median_best_epoch": float(np.median(v)),
                                    "sd": float(np.std(v, ddof=1)) if len(v) > 1 else None}
                                for k, v in sorted(by_lvl.items())},
                     "kruskal_p": p_kw,
                     "spread_of_level_means": float(max(np.mean(v) for v in by_lvl.values())
                                                    - min(np.mean(v) for v in by_lvl.values()))}
    return out


def key_integrity(runs, recs, excluded):
    """Assert the grid keys on the checkpoint side and the search side are the same set, so the
    dynamics statistics and the decomposed metric are describing the same 324 runs."""
    ck = {k for k in runs if k[1] is not None}
    sk = set(recs)
    return {"checkpoint_grid_keys": len(ck), "search_grid_keys": len(sk),
            "checkpoint_only": len(ck - sk), "search_only": len(sk - ck),
            "identical": ck == sk,
            "search_keys_are_a_subset_of_checkpoint_keys": sk <= ck,
            "excluded_non_grid_runs": len(excluded),
            "excluded_examples": excluded[:4],
            "excluded_are_final_refits": all("-final-" in e for e in excluded)}


def main() -> None:
    doc = {"question": ("§BO.9 mechanism half: M1's within-(config,fold) variation is 95.8% "
                        "unexplained by seed identity, so blocking cannot remove it. What is it? "
                        "Read from checkpoint manifests (JSON only, no .pt opened)."),
           "metric_mismatch_warning": ("manifest validation_rank_ic is NOT the search record's "
                                       "seed_validation_rank_ic; measured below, never substituted"),
           "per_carrier": {}}
    for label, ws_name, expected in CARRIERS:
        ws = ART / ws_name
        runs, n_man, excluded = collect(ws)
        recs = search_records(ws)
        axes = config_axes(ws)
        # --- measured metric mismatch, over every run present in both
        diffs, ratios, mism = [], [], 0
        for key, eps in runs.items():
            if key not in recs or not eps:
                continue
            best_man = max(eps.values())
            rep = recs[key]
            diffs.append(best_man - rep)
            if rep != 0:
                ratios.append(best_man / rep)
            if abs(best_man - rep) > 1e-12:
                mism += 1
        # --- training dynamics
        best_ep, n_imp, first_ep = [], [], []
        by_cfg_best_ep = defaultdict(list)
        by_cfgfold_seeds = defaultdict(list)
        for (fold, ci, seed), eps in runs.items():
            if not eps:
                continue
            be = max(eps, key=lambda e: eps[e])
            best_ep.append(be)
            n_imp.append(len(eps))
            first_ep.append(min(eps))
            if ci is not None:
                by_cfg_best_ep[ci].append(be)
                by_cfgfold_seeds[(fold, ci)].append(be)
        be = np.asarray(best_ep, dtype=float)
        # how erratic is the peak epoch across the 3 seeds of the SAME config+fold?
        spreads = [max(v) - min(v) for v in by_cfgfold_seeds.values() if len(v) >= 2]
        sds = [float(np.std(v, ddof=1)) for v in by_cfgfold_seeds.values() if len(v) >= 2]
        doc["per_carrier"][label] = {
            "workspace": ws_name, "n_runs_with_checkpoints": len(runs),
            "n_manifests_read": n_man, "n_search_seed_records": len(recs),
            "expected_runs_full_grid": expected,
            "partial_snapshot": len(runs) < expected,
            "key_integrity": key_integrity(runs, recs, excluded),
            "metric_mismatch": {
                "n_runs_compared": len(diffs), "n_differing": mism,
                "all_differ": mism == len(diffs) and len(diffs) > 0,
                "mean_best_manifest_minus_reported": float(np.mean(diffs)) if diffs else None,
                "median_abs_diff": float(np.median(np.abs(diffs))) if diffs else None,
                "max_abs_diff": float(np.max(np.abs(diffs))) if diffs else None,
                "median_ratio_best_manifest_over_reported": float(np.median(ratios)) if ratios else None,
            },
            "best_epoch": {"n": int(be.size), "mean": float(be.mean()) if be.size else None,
                           "median": float(np.median(be)) if be.size else None,
                           "min": float(be.min()) if be.size else None,
                           "max": float(be.max()) if be.size else None,
                           "sd": float(be.std(ddof=1)) if be.size > 1 else None,
                           "p10": float(np.quantile(be, 0.10)) if be.size else None,
                           "p90": float(np.quantile(be, 0.90)) if be.size else None},
            "n_improvements_saved": {"mean": float(np.mean(n_imp)) if n_imp else None,
                                     "median": float(np.median(n_imp)) if n_imp else None,
                                     "min": int(np.min(n_imp)) if n_imp else None,
                                     "max": int(np.max(n_imp)) if n_imp else None},
            "peak_epoch_spread_across_seeds_same_config_and_fold": {
                "n_groups": len(spreads),
                "mean_range": float(np.mean(spreads)) if spreads else None,
                "median_range": float(np.median(spreads)) if spreads else None,
                "max_range": int(np.max(spreads)) if spreads else None,
                "mean_sd": float(np.mean(sds)) if sds else None,
                "share_of_groups_with_range_ge_5": float(np.mean([s >= 5 for s in spreads])) if spreads else None,
            },
            "per_config_best_epoch_mean": {str(k): float(np.mean(v)) for k, v in sorted(by_cfg_best_ep.items())},
            "best_epoch_by_grid_axis": best_epoch_by_axis(runs, axes),
        }

    m1 = doc["per_carrier"]["M1"]
    m2 = doc["per_carrier"]["M2"]
    for label in ("M1", "M2"):
        ki = doc["per_carrier"][label]["key_integrity"]
        if doc["per_carrier"][label]["partial_snapshot"]:
            # a carrier still training has checkpoints for folds whose search records are not written
            # yet; the only safe requirement is that every search key has checkpoints behind it
            assert ki["search_keys_are_a_subset_of_checkpoint_keys"], \
                f"{label}: a search record has no checkpoints behind it"
        elif ki["checkpoint_grid_keys"]:
            assert ki["identical"], f"{label}: checkpoint and search grid keys differ"
    if m1["peak_epoch_spread_across_seeds_same_config_and_fold"]["n_groups"] and \
       m2["peak_epoch_spread_across_seeds_same_config_and_fold"]["n_groups"]:
        doc["cross_carrier"] = {
            "m1_peak_epoch_spread_mean": m1["peak_epoch_spread_across_seeds_same_config_and_fold"]["mean_range"],
            "m2_peak_epoch_spread_mean": m2["peak_epoch_spread_across_seeds_same_config_and_fold"]["mean_range"],
            "reading": ("the peak epoch of a given config+fold differs across its 3 seeds by this many "
                        "epochs; a large value means optimisation itself is run-idiosyncratic, which is "
                        "exactly variation that blocking on seed cannot remove"),
        }
    doc.update({"read_only": True, "database_access": False, "weights_loaded": False,
                "checkpoint_pt_opened": False, "training_performed": False,
                "repository_module_imported_or_executed": False,
                "generated_at_local": datetime.now().isoformat(timespec="seconds")})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    for label in ("M1", "M2"):
        p = doc["per_carrier"][label]
        print(f"=== {label} ({p['workspace']}) ===")
        print(f"  runs with checkpoints {p['n_runs_with_checkpoints']} (full grid would be "
              f"{p['expected_runs_full_grid']}, partial={p['partial_snapshot']}), manifests {p['n_manifests_read']}")
        mm = p["metric_mismatch"]
        print(f"  METRIC MISMATCH: {mm['n_differing']}/{mm['n_runs_compared']} runs differ; "
              f"mean(best_manifest - reported)={mm['mean_best_manifest_minus_reported']}, "
              f"median ratio={mm['median_ratio_best_manifest_over_reported']}, max|diff|={mm['max_abs_diff']}")
        b = p["best_epoch"]
        print(f"  best epoch: median {b['median']} mean {b['mean']:.1f} min {b['min']} max {b['max']} "
              f"sd {b['sd']:.1f} p10-p90 {b['p10']}-{b['p90']}")
        s = p["peak_epoch_spread_across_seeds_same_config_and_fold"]
        print(f"  peak-epoch spread across the 3 seeds of the SAME config+fold: n={s['n_groups']} "
              f"mean range {s['mean_range']:.1f} median {s['median_range']:.1f} max {s['max_range']} "
              f"(>=5 epochs in {s['share_of_groups_with_range_ge_5']*100:.0f}% of groups)")
        print(f"  improvements saved per run: mean {p['n_improvements_saved']['mean']:.1f} "
              f"min {p['n_improvements_saved']['min']} max {p['n_improvements_saved']['max']}")
        ki = p["key_integrity"]
        print(f"  key integrity: {ki['checkpoint_grid_keys']} checkpoint grid keys vs "
              f"{ki['search_grid_keys']} search keys, identical={ki['identical']} "
              f"(search is a subset={ki['search_keys_are_a_subset_of_checkpoint_keys']}); "
              f"{ki['excluded_non_grid_runs']} non-grid runs excluded "
              f"(all final refits={ki['excluded_are_final_refits']})")
        print("  does the PEAK EPOCH depend on the searched axis?")
        for name, v in p["best_epoch_by_grid_axis"].items():
            lvl = "  ".join(f"{k}: mean {d['mean_best_epoch']:.1f} (n={d['n']})"
                            for k, d in v["levels"].items())
            print(f"    {name:18s} spread {v['spread_of_level_means']:5.1f} epochs  "
                  f"kruskal p={v['kruskal_p']:.3e}   {lvl}")
        print()
    if "cross_carrier" in doc:
        print("cross-carrier:", json.dumps({k: v for k, v in doc["cross_carrier"].items() if k != "reading"},
                                           ensure_ascii=False))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
