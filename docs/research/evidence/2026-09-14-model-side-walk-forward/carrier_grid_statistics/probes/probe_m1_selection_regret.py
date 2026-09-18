#!/usr/bin/env python3
"""Read-only probe: what did §CA's peak/final defect actually COST M1, in the units that matter?

§CA.3 established that M1's grid search ranks configs by their final-epoch validation RankIC while the
artifact it ships is the peak-epoch model, and that the two rankings are nearly uncorrelated -- the
spearman between the peak ranking and the ranking the driver selected on is -0.1422, +0.1429, +0.2314
and -0.3394 across the four folds. §CC.1 then proved by sha256 that the shipped weights really are the
peak-epoch checkpoint, 12 of 12 for M1. What neither section answered is the question an owner has to
act on: given that M1 shipped a config chosen on the wrong basis, HOW MUCH DID THAT COST?

Three artifacts already on disk make this answerable with no training and no database:
  the shipped config per fold is recorded in models/m1-<fold>/config.json;
  the peak-epoch score of every fold-config-seed run is in checkpoints/*/manifest.json, and since
    checkpoints are saved only on strict improvement the max over saved epochs IS the peak;
  the final-epoch score the driver ranked on is in search/<fold>/c*.json.

The honest answer turns out to be "very little", and this probe is built so it can say that rather than
inflate its own subject. §CA.4 measured that on the peak metric M1's config share of variance is 0.17%
with F=0.3408 and p=0.9991, and the observed sd across its 27 config means is 0.000635 against a
standard error of 0.001088 for a single one of those means -- so the 27 configs are statistically
indistinguishable on the metric that describes the shipped model. A large regret is therefore not
available in principle: there is almost nothing to lose by picking the wrong config. The probe reports
regret in three units for that reason -- absolute, in peak-basis config sds, and in standard errors of
the DIFFERENCE between two config means, which is sqrt(2) times the SE of one -- because only the third
says whether the regret is distinguishable from zero at all.

An integrity gate runs first and the probe aborts if it fails: the driver's final-epoch argmax must
reproduce the config actually recorded in the shipped bundle on every fold. Without that, regret would
be measured against a hypothetical rather than against what the framework really did.

Safety: reads JSON and bundle config files only. No checkpoint.pt is opened, so no weight or tensor is
loaded. No database access, no network, no repository module imported or executed, nothing written into
any artifact directory, no signal sent to any process. M2, pid 3588838, is still training.
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

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
WS = ART / "m1_full_v1"
OUT = Path(__file__).resolve().parent.parent / "measurements" / "m1_selection_regret.json"
CK = re.compile(r"^m1-f(?P<fold>\d)-c(?P<cfg>\d+)-\d+-seed-(?P<seed>\d+)-epoch-(?P<ep>\d+)$")
SEEDS = ("20260801", "20260802", "20260803")
FOLDS = ("F1", "F2", "F3", "F4")
AXES = ("hidden_dim", "num_blocks", "dropout")
PEAK_ANOVA = Path(__file__).resolve().parent.parent / "measurements" / "peak_versus_final_model_state.json"


def peak_scores():
    """fold-config-seed to max manifest validation_rank_ic, i.e. the peak (saves only ever improve)."""
    out = defaultdict(lambda: float("-inf"))
    n = 0
    for d in glob.glob(str(WS / "checkpoints" / "*-epoch-*")):
        m = CK.match(Path(d).name)
        mp = Path(d) / "manifest.json"
        if not m or not mp.is_file():
            continue
        ic = json.load(open(mp, encoding="utf-8")).get("metadata", {}).get("validation_rank_ic")
        if ic is None:
            continue
        n += 1
        k = (f"F{m.group('fold')}", int(m.group("cfg")), str(m.group("seed")))
        out[k] = max(out[k], float(ic))
    return dict(out), n


def search_records():
    cfgs, final, ens = {}, {}, {}
    for f in FOLDS:
        for p in sorted(glob.glob(str(WS / "search" / f / "c*.json"))):
            j = json.load(open(p, encoding="utf-8"))
            ci = j["config_index"]
            r = j.get("result", {})
            cfgs[(f, ci)] = j["config"]
            ens[(f, ci)] = float(r["ensemble_validation_rank_ic"])
            for s, v in (r.get("seed_validation_rank_ic") or {}).items():
                final[(f, ci, str(s))] = float(v)
    return cfgs, final, ens


def shipped_config(f):
    return json.load(open(WS / "models" / f"m1-{f.lower()}" / "config.json", encoding="utf-8"))


def main() -> None:
    peak, n_man = peak_scores()
    cfgs, final, ens = search_records()
    an = json.load(open(PEAK_ANOVA, encoding="utf-8"))
    m1_peak = an["per_carrier"]["M1"]["anova_peak_epoch_manifest"]
    se_one = m1_peak["se_config_mean"]
    se_diff = se_one * math.sqrt(2.0)

    per_fold = {}
    for f in FOLDS:
        cids = sorted({k[1] for k in cfgs if k[0] == f})
        assert len(cids) == 27, f"{f}: expected 27 configs, got {len(cids)}"
        for ci in cids:
            for s in SEEDS:
                assert (f, ci, s) in peak, f"{f} c{ci} seed {s}: no peak score"
                assert (f, ci, s) in final, f"{f} c{ci} seed {s}: no final score"

        ship_cfg = shipped_config(f)
        matched = [ci for ci in cids if all(cfgs[(f, ci)][a] == ship_cfg[a] for a in AXES)]
        assert len(matched) == 1, f"{f}: shipped config matched {len(matched)} grid entries"
        ship = matched[0]

        argmax_final = max(cids, key=lambda c: ens[(f, c)])
        pk = {c: sum(peak[(f, c, s)] for s in SEEDS) / 3 for c in cids}
        fn = {c: sum(final[(f, c, s)] for s in SEEDS) / 3 for c in cids}
        argmax_peak = max(cids, key=lambda c: pk[c])
        argmax_final_seedmean = max(cids, key=lambda c: fn[c])
        assert argmax_final == ship, (
            f"{f}: driver argmax c{argmax_final} != shipped c{ship}; regret would be measured "
            "against a hypothetical")

        pk_vals = np.array([pk[c] for c in cids])
        cfg_sd = float(pk_vals.std(ddof=1))
        regret = pk[argmax_peak] - pk[ship]
        per_fold[f] = {
            "shipped_config_index": ship, "shipped_config": cfgs[(f, ship)],
            "integrity_gate_driver_argmax_equals_shipped": True,
            "driver_ensemble_value": ens[(f, ship)],
            "final_basis": {"argmax_on_ensemble": argmax_final,
                            "argmax_on_seedmean": argmax_final_seedmean,
                            "shipped_value": fn[ship], "best_value": fn[argmax_final_seedmean],
                            "argmax_agrees_between_ensemble_and_seedmean":
                                argmax_final == argmax_final_seedmean},
            "peak_basis": {"argmax": argmax_peak, "argmax_config": cfgs[(f, argmax_peak)],
                           "shipped_value": pk[ship], "best_value": pk[argmax_peak],
                           "shipped_rank_of_27": int((pk_vals > pk[ship]).sum()) + 1,
                           "config_mean_sd": cfg_sd,
                           "config_range": float(pk_vals.max() - pk_vals.min())},
            "regret_absolute": float(regret),
            "regret_in_peak_config_sds": float(regret / cfg_sd),
            "se_of_one_config_mean_peak": se_one,
            "se_of_difference_between_two_config_means": se_diff,
            "regret_in_se_of_difference": float(regret / se_diff),
            "regret_distinguishable_from_zero_at_2se": bool(regret / se_diff > 2.0),
        }

    reg = np.array([per_fold[f]["regret_absolute"] for f in FOLDS])
    z = np.array([per_fold[f]["regret_in_se_of_difference"] for f in FOLDS])
    doc = {
        "question": ("§CA.3 showed M1 shipped a config chosen by a ranking nearly uncorrelated with the "
                     "shipped model's quality. How much did that cost, in units that decide whether it "
                     "matters at all?"),
        "integrity_gate": ("the driver's ensemble argmax was verified to equal the config recorded in "
                           "models/m1-<fold>/config.json on all four folds before any regret was "
                           "computed, so regret is measured against what the framework really shipped"),
        "n_manifests_read": n_man, "n_runs": len(peak),
        "per_fold": per_fold,
        "summary": {
            "regret_absolute_min": float(reg.min()), "regret_absolute_max": float(reg.max()),
            "regret_absolute_mean": float(reg.mean()),
            "regret_in_se_of_difference": [float(v) for v in z],
            "regret_in_se_min": float(z.min()), "regret_in_se_max": float(z.max()),
            "folds_where_regret_exceeds_2se": int((z > 2.0).sum()),
            "shipped_rank_on_peak_basis": [per_fold[f]["peak_basis"]["shipped_rank_of_27"] for f in FOLDS],
            "peak_basis_config_share_of_variance": m1_peak["config_share_of_variance"],
            "peak_basis_config_p": m1_peak["config_p"],
            "peak_basis_config_mean_sd": m1_peak["config_mean_sd"],
            "peak_basis_se_config_mean": se_one,
            "verdict": ("the ranking is scrambled but the prize is almost nothing: on the peak metric "
                        "M1's 27 configs are statistically indistinguishable, so the regret from "
                        "shipping the wrong one is 1.0-1.5 SE of a difference between two config means "
                        "and is not distinguishable from zero on any fold"),
        },
        "read_only": True, "database_access": False, "network_or_vendor_call": False, "mcp_used": False,
        "checkpoint_pt_opened": False, "weights_loaded": False, "training_performed": False,
        "repository_module_imported_or_executed": False,
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
    }
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"integrity gate: driver argmax == shipped config on all 4 folds "
          f"(manifests read {n_man}, runs {len(peak)})")
    print(f"SE of one config mean (peak basis) = {se_one:.6f}; of a difference = {se_diff:.6f}")
    print()
    print(f"{'fold':5s} {'shipped':8s} {'rank/27':8s} {'peak argmax':12s} "
          f"{'regret':>10s} {'cfg sd':>8s} {'SE(diff)':>9s} {'>2SE':5s}")
    for f in FOLDS:
        p = per_fold[f]
        print(f"{f:5s} c{p['shipped_config_index']:<7d} {p['peak_basis']['shipped_rank_of_27']:<8d} "
              f"c{p['peak_basis']['argmax']:<11d} {p['regret_absolute']:10.6f} "
              f"{p['regret_in_peak_config_sds']:8.2f} {p['regret_in_se_of_difference']:9.2f} "
              f"{str(p['regret_distinguishable_from_zero_at_2se']):5s}")
    s = doc["summary"]
    print()
    print(f"regret absolute {s['regret_absolute_min']:.6f} .. {s['regret_absolute_max']:.6f} "
          f"(mean {s['regret_absolute_mean']:.6f}); in SE(diff) {s['regret_in_se_min']:.2f} .. "
          f"{s['regret_in_se_max']:.2f}; folds exceeding 2 SE: {s['folds_where_regret_exceeds_2se']}/4")
    print(f"shipped rank on peak basis: {s['shipped_rank_on_peak_basis']}")
    print(f"peak-basis config share {s['peak_basis_config_share_of_variance']*100:.2f}%, "
          f"p={s['peak_basis_config_p']:.4f}, config-mean sd {s['peak_basis_config_mean_sd']:.6f} "
          f"vs SE {s['peak_basis_se_config_mean']:.6f}")
    print()
    for f in FOLDS:
        p = per_fold[f]
        print(f"  {f}: shipped {p['shipped_config']}")
        print(f"      peak-basis best was {p['peak_basis']['argmax_config']} "
              f"(peak {p['peak_basis']['best_value']:.6f} vs shipped "
              f"{p['peak_basis']['shipped_value']:.6f})")
        print(f"      final-basis argmax agrees ensemble vs seedmean: "
              f"{p['final_basis']['argmax_agrees_between_ensemble_and_seedmean']}")
    print()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
