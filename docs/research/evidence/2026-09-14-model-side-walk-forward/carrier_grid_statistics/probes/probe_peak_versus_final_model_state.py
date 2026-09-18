#!/usr/bin/env python3
"""Read-only probe: the search record scores the FINAL-epoch weights, the shipped checkpoint is the PEAK.

§BY.6 measured that checkpoints/*/manifest.json's `validation_rank_ic` differs from the search
record's `seed_validation_rank_ic` in 324/324 M1 runs and 72/72 M2 runs, with the manifest value
systematically higher (median ratio 1.0837 / 1.0886), and did not explain it. Reading the source as
data explains it completely, and the explanation is not about the metric at all -- it is about WHICH
MODEL each number describes:

  * trainer.py:106-116  `validation_rank_ic = validation_epoch(epoch)`; a checkpoint is saved ONLY on
    strict improvement (`if validation_rank_ic > best_value`), and the value handed to
    `checkpoint_epoch(epoch, validation_rank_ic)` is what lands in the manifest. So the manifest value
    is the PEAK-epoch score, and because saves happen only on improvement,
    `run.checkpoint_ids[-1]` is the peak-epoch checkpoint.
  * trainer.py:118-119  the loop `break`s once `stale_epochs >= early_stopping_patience`, leaving the
    live `model` object holding the FINAL epoch's weights, up to `patience` epochs past the peak.
  * runner.py           `run_validation_governed_training` returns
    `ValidationGovernedRun(training, checkpoint_ids)` and never restores anything into `model`;
    `rg load_state_dict` over runner.py, cross_section_search.py and temporal_search.py returns nothing.
  * cross_section_search.py:203-208 (M1) and temporal_search.py:374-379 (M2) then do
    `predictions = percentile_frame(model, validation_batches)` and
    `validation_rank_ic = validation_rank_ic(predictions, validation_labels)`, i.e. they score the
    FINAL-epoch weights, while `checkpoint_path` points at `run.checkpoint_ids[-1]`, the PEAK.
  * trainer.py:55-58    `TrainingResult.best_validation_rank_ic` holds the peak value, and
    `rg best_validation_rank_ic --glob '*.py'` finds exactly ONE non-test reference: its own field
    declaration. Nothing in the production path ever reads it.

Three candidate causes are therefore eliminated by reading, not by measuring: the score column cannot
matter because `percentile_rank` is a within-date rank of `raw_score` (models/base.py:154) and daily
Spearman is invariant to a monotone within-date transform; `minimum_cross_section` is 30 on both paths
(validation.py:28 default, driver `_validation_rank_ic` explicit); and both call the same
`compute_daily_rank_ic` + `summarize_rank_ic`. Only the model state differs.

CONSEQUENCE, and the reason this probe exists: the persisted artifact is the peak-epoch model but the
number the driver's `argmax` selects on (`ensemble_validation_rank_ic`, §BP) is the final-epoch model's
score. Since every peak value is already on disk in the manifests, the counterfactual "what if
selection had scored the model that actually ships" can be evaluated with NO training at all. This
probe asks two things: does the winning config change, and is the peak-based metric less noisy -- the
second because §BY.2 concluded M1's config effect is genuinely small using a residual that this
finding says is inflated by post-peak drift, which would make that conclusion conservative in the
wrong direction.

Caveat stated up front and carried into the output: the driver's ensemble averages PREDICTIONS across
seeds and then scores the averaged frame, so an ensemble value cannot be rebuilt exactly from scalar
per-seed scores. This probe therefore compares config rankings using the MEAN OVER SEEDS of each
metric, which is a proxy, and says so; it never claims to reproduce `ensemble_validation_rank_ic`.

Safety: reads manifest.json and search-record JSON only. checkpoint.pt is never opened, so no weight or
tensor is loaded. No database access, no network, no repository module imported or executed, nothing
written into any artifact directory, no signal sent to any process. M2 (pid 3588838) is still training.
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
OUT = Path(__file__).resolve().parent.parent / "measurements" / "peak_versus_final_model_state.json"
CKPT_RE = re.compile(r"^[a-z0-9]+-f(?P<fold>\d+)-c(?P<cfg>\d+)-\d+-seed-(?P<seed>\d+)-epoch-(?P<ep>\d+)$")
CARRIERS = (("M1", "m1_full_v1"), ("M2", "m2_v1"))


def peaks(ws: Path):
    """{(fold, cfg_index, seed): peak manifest validation_rank_ic} -- max over saved epochs."""
    out = defaultdict(lambda: float("-inf"))
    n = 0
    for d in glob.glob(str(ws / "checkpoints" / "*-epoch-*")):
        m = CKPT_RE.match(Path(d).name)
        mp = Path(d) / "manifest.json"
        if not m or not mp.is_file():
            continue
        ic = json.load(open(mp, encoding="utf-8")).get("metadata", {}).get("validation_rank_ic")
        if ic is None:
            continue
        n += 1
        key = (f"F{m.group('fold')}", int(m.group("cfg")), str(m.group("seed")))
        out[key] = max(out[key], float(ic))
    return dict(out), n


def reported(ws: Path):
    """{(fold, cfg_index, seed): seed_validation_rank_ic}, {(fold,cfg): ensemble}, {(fold,cfg): config}.

    The ensemble value is what the driver's argmax actually selects on (§BP), so it is carried
    separately: the mean-over-seeds used below is only a proxy, and the proxy has to be validated
    against the real selection metric before any "the winner changed" claim is allowed.
    """
    out, cfg_of, ens_of = {}, {}, {}
    for d in sorted(glob.glob(str(ws / "search" / "F*"))):
        fold = Path(d).name
        for p in sorted(glob.glob(d + "/*.json")):
            j = json.load(open(p, encoding="utf-8"))
            r = j.get("result", {})
            sric = r.get("seed_validation_rank_ic") or {}
            ci = j.get("config_index")
            cfg_of[(fold, ci)] = j.get("config")
            if r.get("ensemble_validation_rank_ic") is not None:
                ens_of[(fold, ci)] = float(r["ensemble_validation_rank_ic"])
            for s, v in sric.items():
                out[(fold, ci, str(s))] = float(v)
    return out, cfg_of, ens_of


def decompose(Y: np.ndarray) -> dict:
    """§BO's blocked decomposition, reused so the two metrics are compared with one instrument."""
    a, b, c = Y.shape
    gm = float(Y.mean())
    m_cfg = Y.mean(axis=(1, 2)); m_fold = Y.mean(axis=(0, 2))
    m_seed = Y.mean(axis=(0, 1)); m_cf = Y.mean(axis=2)
    e_cfg = m_cfg - gm; e_fold = m_fold - gm; e_seed = m_seed - gm
    e_int = m_cf - m_cfg[:, None] - m_fold[None, :] + gm
    resid = Y - m_cf[:, :, None] - e_seed[None, None, :]
    SS_cfg = b * c * float((e_cfg ** 2).sum())
    SS_tot = float(((Y - gm) ** 2).sum())
    SS_seed = a * b * float((e_seed ** 2).sum())
    SS_res = float((resid ** 2).sum())
    SS_within = float(((Y - m_cf[:, :, None]) ** 2).sum())
    df_cfg, df_res = a - 1, (a * b - 1) * (c - 1)
    MS_res = SS_res / df_res
    F = (SS_cfg / df_cfg) / MS_res
    return {"a": a, "b": b, "c": c, "config_F": float(F),
            "config_p": float(stats.f.sf(F, df_cfg, df_res)),
            "config_share_of_variance": SS_cfg / SS_tot,
            "resid_sd": float(math.sqrt(MS_res)),
            "sigma_c": float(math.sqrt(max((SS_cfg / df_cfg - MS_res) / (b * c), 0.0))),
            "sigma_c_raw": float((SS_cfg / df_cfg - MS_res) / (b * c)),
            "within_cell_variation_explained_by_seed_identity": SS_seed / SS_within,
            "config_mean_sd": float(m_cfg.std(ddof=1)),
            "se_config_mean": float(math.sqrt(MS_res / (b * c))),
            "gm": gm, "SS_config": SS_cfg, "SS_tot": SS_tot, "df_res": df_res,
            "df_config": df_cfg}


def axis_effects(Y: np.ndarray, axes_idx: dict, dec: dict) -> dict:
    """Orthogonal per-axis main-effect SS on a given metric tensor.

    Exact because every grid here is a balanced full factorial. Run on BOTH the final-epoch and the
    peak-epoch tensor so §CA.5's "the config effect mostly vanishes on the shipped model's metric" can
    be pushed down to axis level: an axis significant on the final metric but not on the peak one is an
    axis whose apparent answer was an artefact of scoring the wrong model state.
    """
    a, b, c = Y.shape
    out = {}
    for name, spec in axes_idx.items():
        idx = np.asarray(spec["idx"])
        values = spec["values"]
        levels = sorted(set(idx.tolist()))
        L = len(levels)
        if L < 2:
            continue
        counts = np.array([(idx == t).sum() for t in levels])
        assert (counts == counts[0]).all(), f"{name}: grid is not a full factorial on this axis"
        n_obs_per_level = int(counts[0]) * b * c
        assert n_obs_per_level * L == Y.size, f"{name}: unbalanced"
        lvl_means = np.array([Y[idx == t].mean() for t in levels])
        SS = float(n_obs_per_level * ((lvl_means - dec["gm"]) ** 2).sum())
        df = L - 1
        F = (SS / df) / (dec["resid_sd"] ** 2)
        best = int(np.argmax(lvl_means))
        out[name] = {"level_indices": [int(t) for t in levels],
                     "levels": [str(values[t]) for t in levels],
                     "level_means": [float(v) for v in lvl_means],
                     "best_level": str(values[levels[best]]),
                     "spread": float(lvl_means.max() - lvl_means.min()),
                     "SS": SS, "df": df, "F": float(F),
                     "p": float(stats.f.sf(F, df, dec["df_res"])),
                     "share_of_SS_config": SS / dec["SS_config"] if dec["SS_config"] > 0 else None}
    return out


def main() -> None:
    doc = {"question": ("§BY.6's unexplained metric mismatch: the search record scores the final-epoch "
                        "weights while the persisted checkpoint is the peak-epoch model. Does selecting "
                        "on the shipped model's score change the winner, and is that metric less noisy?"),
           "static_finding": {
               "manifest_value_is": "peak-epoch score (trainer.py:106-116, saved only on strict improvement)",
               "search_record_value_is": ("final-epoch score (cross_section_search.py:203-208, "
                                          "temporal_search.py:374-379: percentile_frame(model, ...) "
                                          "after fit() returned)"),
               "shipped_checkpoint_is": "run.checkpoint_ids[-1] == the peak-epoch checkpoint",
               "best_checkpoint_restored_before_scoring": False,
               "load_state_dict_in_runner_or_search_modules": False,
               "TrainingResult_best_validation_rank_ic_non_test_references": 1,
               "that_one_reference_is": "its own field declaration at trainer.py:57",
               "eliminated_by_reading": [
                   "score column: percentile_rank is a within-date rank of raw_score "
                   "(models/base.py:154), so daily Spearman is invariant",
                   "minimum_cross_section: 30 on both paths (validation.py:28, driver "
                   "_validation_rank_ic)",
                   "aggregator: both call compute_daily_rank_ic + summarize_rank_ic",
               ],
               "only_remaining_difference": "the model state the score is computed from",
           },
           "ensemble_caveat": ("the driver's ensemble averages PREDICTIONS across seeds then scores the "
                               "averaged frame, so an ensemble value cannot be rebuilt from scalar "
                               "per-seed scores; rankings below use the MEAN OVER SEEDS as a proxy and "
                               "this probe never claims to reproduce ensemble_validation_rank_ic"),
           "per_carrier": {}}

    for label, ws_name in CARRIERS:
        ws = ART / ws_name
        pk, n_man = peaks(ws)
        rep, cfg_of, ens_of = reported(ws)
        keys = sorted(set(pk) & set(rep))
        pairs = [(k, pk[k], rep[k]) for k in keys]
        assert pairs, f"{label}: no joinable runs"
        diffs = np.array([p - r for _, p, r in pairs])

        folds = sorted({k[0] for k, _, _ in pairs})
        cfgs = sorted({k[1] for k, _, _ in pairs})
        seeds = sorted({k[2] for k, _, _ in pairs})
        # winner per fold under each metric, using the mean over seeds as the proxy score
        winners = {}
        for f in folds:
            score_pk, score_rp, score_ens = {}, {}, {}
            for ci in cfgs:
                vp = [p for (ff, c2, _s), p, _r in pairs if ff == f and c2 == ci]
                vr = [r for (ff, c2, _s), _p, r in pairs if ff == f and c2 == ci]
                if len(vp) == len(seeds):
                    score_pk[ci] = float(np.mean(vp)); score_rp[ci] = float(np.mean(vr))
                    if (f, ci) in ens_of:
                        score_ens[ci] = ens_of[(f, ci)]
            if score_pk:
                common = sorted(set(score_pk) & set(score_ens))
                wp, wr = max(score_pk, key=score_pk.get), max(score_rp, key=score_rp.get)
                we = max(common, key=score_ens.get) if common else None
                winners[f] = {
                    "winner_by_peak": wp, "winner_by_final": wr, "same": wp == wr,
                    "winner_by_driver_ensemble": we,
                    "proxy_final_matches_driver_ensemble": (we is not None and we == wr),
                    "proxy_peak_matches_driver_ensemble": (we is not None and we == wp),
                    "spearman_proxyfinal_vs_driver_ensemble": (float(stats.spearmanr(
                        [score_rp[c] for c in common], [score_ens[c] for c in common]).statistic)
                        if len(common) > 2 else None),
                    "spearman_proxypeak_vs_driver_ensemble": (float(stats.spearmanr(
                        [score_pk[c] for c in common], [score_ens[c] for c in common]).statistic)
                        if len(common) > 2 else None),
                    "driver_ensemble_config": cfg_of.get((f, we)),
                    "peak_config": cfg_of.get((f, wp)), "final_config": cfg_of.get((f, wr)),
                    "n_configs_compared": len(score_pk),
                    "rank_correlation_peak_vs_final": float(stats.spearmanr(
                        [score_pk[c] for c in sorted(score_pk)],
                        [score_rp[c] for c in sorted(score_pk)]).statistic),
                }

        entry = {"workspace": ws_name, "n_manifests_read": n_man, "n_joinable_runs": len(pairs),
                 "peak_minus_final": {"mean": float(diffs.mean()), "median": float(np.median(diffs)),
                                      "min": float(diffs.min()), "max": float(diffs.max()),
                                      "share_peak_higher": float((diffs > 0).mean()),
                                      "median_ratio": float(np.median([p / r for _, p, r in pairs
                                                                       if r != 0]))},
                 "winner_comparison": winners,
                 "folds_where_winner_changed": [f for f, w in winners.items() if not w["same"]]}

        # noise comparison, only where the grid is complete (M1: 4 folds; M2: F1 only, so skipped)
        if len(folds) > 1:
            # each pair is (peak, final): index 0 is the peak-epoch manifest value, index 1 the
            # final-epoch value the search record reports
            for name, idx in (("anova_final_epoch_reported", 1), ("anova_peak_epoch_manifest", 0)):
                Y = np.full((len(cfgs), len(folds), len(seeds)), np.nan)
                ok = True
                lookup = {(k[0], k[1], k[2]): (p, r) for k, p, r in pairs}
                for i, ci in enumerate(cfgs):
                    for j, f in enumerate(folds):
                        for kk, s in enumerate(seeds):
                            v = lookup.get((f, ci, s))
                            if v is None:
                                ok = False; break
                            Y[i, j, kk] = v[idx]
                if not ok or np.isnan(Y).any():
                    entry[name] = {"skipped": "incomplete tensor"}
                    continue
                dec = decompose(Y)
                # axis level indices, from the config payloads the search records carry
                proto = cfg_of.get((folds[0], cfgs[0])) or {}
                axes_idx = {}
                for ax in proto:
                    vals = sorted({(cfg_of.get((folds[0], ci)) or {}).get(ax) for ci in cfgs},
                                  key=lambda v: (isinstance(v, str), v))
                    if len(vals) < 2:
                        continue
                    axes_idx[ax] = {
                        "values": vals,
                        "idx": [vals.index((cfg_of.get((folds[0], ci)) or {}).get(ax)) for ci in cfgs],
                    }
                dec["axis_main_effects"] = axis_effects(Y, axes_idx, dec)
                dec["grid_axes"] = {k: [str(x) for x in sorted(
                    {(cfg_of.get((folds[0], ci)) or {}).get(k) for ci in cfgs},
                    key=lambda v: (isinstance(v, str), v))] for k in axes_idx}
                entry[name] = dec

        # ---- the cancellation ratio that decides whether pooling folds can rescue a seed effect.
        # §BO.3 predicted M2's seed share would collapse to ~1% once more folds landed, on the
        # reasoning that every multi-fold carrier's seed direction flips between folds and the flips
        # cancel in the pooled main effect. The ratio pooled/mean(within-fold) measures that
        # cancellation directly: ~0.2 for the three carriers whose ordering flips, and it should stay
        # near 1.0 for a carrier whose ordering does not.
        if len(folds) > 1:
            Yf = np.full((len(cfgs), len(folds), len(seeds)), np.nan)
            lookup = {(k[0], k[1], k[2]): (p, r) for k, p, r in pairs}
            for i, ci in enumerate(cfgs):
                for j, f in enumerate(folds):
                    for kk, s in enumerate(seeds):
                        v = lookup.get((f, ci, s))
                        if v is not None:
                            Yf[i, j, kk] = v[1]
            if not np.isnan(Yf).any():
                within = []
                orderings = []
                for j in range(Yf.shape[1]):
                    sm = Yf[:, j, :].mean(axis=0)
                    within.append(float(sm.max() - sm.min()))
                    orderings.append(tuple(int(x) for x in np.argsort(-sm)))
                pooled = Yf.mean(axis=(0, 1))
                entry["seed_cancellation"] = {
                    "pooled_seed_swing": float(pooled.max() - pooled.min()),
                    "within_fold_seed_swings": within,
                    "mean_within_fold_swing": float(np.mean(within)),
                    "cancellation_ratio_pooled_over_mean_within": float(
                        (pooled.max() - pooled.min()) / np.mean(within)),
                    "n_distinct_seed_orderings": len(set(orderings)),
                    "seed_orderings_by_fold": [list(o) for o in orderings],
                    "seed_labels": seeds,
                    "reading": ("ratio 1.0 means pooling cancels nothing, so the seed main effect "
                                "survives pooling intact and cannot collapse to ~1%. Measured on the "
                                "three carriers §BO.3 relied on, whose seed ordering DOES flip between "
                                "folds, the ratio is 0.667 (M0-base), 0.673 (M0-refined) and 0.288 "
                                "(M1) -- so partial cancellation plus a much smaller pooled swing is "
                                "what drove their seed share to ~1%. M2 is the counterexample to "
                                "§BO.3's reasoning: it is a multi-fold carrier whose ordering does NOT "
                                "flip (1 distinct ordering across both folds, 20260803>20260802>"
                                "20260801 in each), so nothing cancels."),
                    "cancellation_ratio_other_carriers_for_contrast": {
                        "M0-base": {"pooled": 0.00431, "mean_within_fold": 0.00645, "ratio": 0.667,
                                    "distinct_orderings": 4, "seed_share_pct": 0.99},
                        "M0-refined": {"pooled": 0.00444, "mean_within_fold": 0.00660, "ratio": 0.673,
                                       "distinct_orderings": 3, "seed_share_pct": 1.06},
                        "M1": {"pooled": 0.00426, "mean_within_fold": 0.01480, "ratio": 0.288,
                               "distinct_orderings": 3, "seed_share_pct": 0.88},
                        "note": ("within-fold swings for these three are transcribed from §BO.3's "
                                 "published table; M2's are computed live in this run"),
                    },
                }
        doc["per_carrier"][label] = entry

    doc.update({"read_only": True, "database_access": False, "network_or_vendor_call": False,
                "mcp_used": False, "checkpoint_pt_opened": False, "weights_loaded": False,
                "training_performed": False, "repository_module_imported_or_executed": False,
                "generated_at_local": datetime.now().isoformat(timespec="seconds")})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("static finding: manifest = PEAK-epoch score, search record = FINAL-epoch score,")
    print("  shipped checkpoint = peak; best checkpoint never restored; "
          "TrainingResult.best_validation_rank_ic has 1 non-test reference (its own declaration)")
    for label, _ in CARRIERS:
        e = doc["per_carrier"][label]
        d = e["peak_minus_final"]
        print(f"\n=== {label} ({e['workspace']}) ===")
        print(f"  joinable runs {e['n_joinable_runs']} (manifests {e['n_manifests_read']})")
        print(f"  peak - final: mean {d['mean']:+.6f} median {d['median']:+.6f} "
              f"min {d['min']:+.6f} max {d['max']:+.6f}; peak higher in "
              f"{d['share_peak_higher']*100:.1f}% of runs; median ratio {d['median_ratio']:.4f}")
        print(f"  folds where the WINNER changed: {e['folds_where_winner_changed'] or 'none'}")
        for f, w in e["winner_comparison"].items():
            print(f"    {f}: peak->c{w['winner_by_peak']} final->c{w['winner_by_final']} "
                  f"same={w['same']} rank corr {w['rank_correlation_peak_vs_final']:.4f} "
                  f"(n={w['n_configs_compared']})")
            print(f"        driver's actual argmax (ensemble_validation_rank_ic) -> "
                  f"c{w['winner_by_driver_ensemble']}; proxy-final reproduces it = "
                  f"{w['proxy_final_matches_driver_ensemble']}; "
                  f"spearman(proxy-final, driver) = "
                  f"{w['spearman_proxyfinal_vs_driver_ensemble']:.4f}; "
                  f"spearman(proxy-peak, driver) = "
                  f"{w['spearman_proxypeak_vs_driver_ensemble']:.4f}")
        for name in ("anova_final_epoch_reported", "anova_peak_epoch_manifest"):
            a = e.get(name)
            if not a:
                continue
            if "skipped" in a:
                print(f"  {name}: skipped ({a['skipped']})")
                continue
            print(f"  {name}: config F={a['config_F']:.4f} p={a['config_p']:.4g} "
                  f"share={a['config_share_of_variance']*100:.2f}% resid_sd={a['resid_sd']:.5f} "
                  f"sigma_c={a['sigma_c']:.6f} (raw {a['sigma_c_raw']:+.3e}) "
                  f"seed_explains={a['within_cell_variation_explained_by_seed_identity']*100:.1f}% "
                  f"cfg_mean_sd={a['config_mean_sd']:.6f} se={a['se_config_mean']:.6f}")
            for ax, v in a.get("axis_main_effects", {}).items():
                print(f"      axis {ax:14s} levels={v['levels']} best={v['best_level']} "
                      f"spread={v['spread']:.5f} F={v['F']:8.3f} p={v['p']:.3e} "
                      f"share_of_SS_config={v['share_of_SS_config']*100:6.2f}%")
        sc = e.get("seed_cancellation")
        if sc:
            print(f"  seed cancellation: pooled swing {sc['pooled_seed_swing']:.5f} vs mean within-fold "
                  f"{sc['mean_within_fold_swing']:.5f} -> ratio "
                  f"{sc['cancellation_ratio_pooled_over_mean_within']:.3f}; distinct orderings "
                  f"{sc['n_distinct_seed_orderings']} {sc['seed_orderings_by_fold']}")
    print("\nwrote", OUT)


if __name__ == "__main__":
    main()
