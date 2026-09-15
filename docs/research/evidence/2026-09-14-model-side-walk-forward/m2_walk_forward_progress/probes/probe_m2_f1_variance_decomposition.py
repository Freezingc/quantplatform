#!/usr/bin/env python3
"""Read-only probe: how much of M2's F1 grid result is hyperparameters, and how much is the seed?

The existing record (m2_f1_grid_and_standing.json) already measures the seed *span* (0.0185), names
20260801 as the worst seed for all 24 configs, and runs per-axis Mann-Whitney tests on the 24
ensemble values. What it does not do is decompose the variance. That matters here because of a trap
this probe documents by falling into it and then climbing out:

  A one-way ANOVA on the 72 individual seed values says F=0.43, p=0.98 -- "the 24 configs are
  statistically indistinguishable, the ranking is noise". That conclusion is WRONG. It is wrong
  because 78.8% of the variance is a seed main effect that is *common to all 24 configs* (they all
  ran seeds 20260801/02/03). A common effect cancels in a config-to-config comparison, but a one-way
  ANOVA leaves it in the residual, where it destroys the test's power.

  The design is in fact a randomized complete block design with seed as the block. Analysed that way
  the config effect is F(23,46)=8.76, p~2e-9. So the grid did find real signal; the naive test just
  could not see it.

Both routes are computed here and the difference is reconciled exactly, because an unreconciled
discrepancy between two ways of doing the same ANOVA is how the two arithmetic errors this probe
replaces got through in the first place.

Safety: reads only the JSON search records the training run already wrote. No database access, no
weight loading, no repository module imported or executed, nothing written into any artifact
directory. Output goes to --out (default under /data/zhangyudong/qrv2_research_artifacts).
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

ART = Path("/data/zhangyudong/qrv2_research_artifacts")


def load(search_dir: Path):
    recs = []
    for p in sorted(glob.glob(str(search_dir / "*.json"))):
        d = json.load(open(p, encoding="utf-8"))
        r = d.get("result", {})
        sv = r.get("seed_validation_rank_ic") or {}
        if not sv:
            continue
        recs.append({"config": d["config"], "config_index": d.get("config_index"),
                     "ensemble": r["ensemble_validation_rank_ic"], "seeds": sv, "path": p})
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--search-dir", type=Path, default=ART / "m2_v1" / "search" / "F1")
    ap.add_argument("--out", type=Path, default=ART / "m2_f1_variance_decomposition")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    recs = load(args.search_dir)
    seednames = sorted(recs[0]["seeds"].keys())
    assert all(sorted(r["seeds"].keys()) == seednames for r in recs), "seed sets differ across configs"
    Y = np.array([[r["seeds"][s] for s in seednames] for r in recs], dtype=float)
    a, b = Y.shape                     # a = configs (rows), b = seeds (columns)
    ens = np.array([r["ensemble"] for r in recs], dtype=float)
    gm = float(Y.mean())
    rowmean = Y.mean(axis=1)           # per-config mean over seeds
    colmean = Y.mean(axis=0)           # per-seed mean over configs
    cfg_eff = rowmean - gm
    seed_eff = colmean - gm
    resid = Y - rowmean[:, None] - colmean[None, :] + gm

    # ---- sums of squares. SS_cfg is scaled by b (rows) and SS_seed by a (columns). Getting these two
    # ---- multipliers the wrong way round is exactly the bug that is asserted against below.
    SS_cfg = b * float((cfg_eff ** 2).sum())
    SS_seed = a * float((seed_eff ** 2).sum())
    SS_res = float((resid ** 2).sum())
    SS_tot = float(((Y - gm) ** 2).sum())
    df_cfg, df_seed, df_res = a - 1, b - 1, (a - 1) * (b - 1)

    # HARD ASSERTION 1: the decomposition must be exactly additive. A swapped a/b fails this.
    add_err = abs((SS_cfg + SS_seed + SS_res) - SS_tot)
    assert add_err < 1e-12 * max(1.0, SS_tot), f"additivity broken, err={add_err:.3e}"
    # HARD ASSERTION 2: every effect vector must be centred.
    assert abs(float(cfg_eff.sum())) < 1e-12 and abs(float(seed_eff.sum())) < 1e-12
    assert abs(float(resid.sum())) < 1e-10

    MS_cfg, MS_seed, MS_res = SS_cfg / df_cfg, SS_seed / df_seed, SS_res / df_res
    F_cfg, F_seed = MS_cfg / MS_res, MS_seed / MS_res
    p_cfg = float(stats.f.sf(F_cfg, df_cfg, df_res))
    p_seed = float(stats.f.sf(F_seed, df_seed, df_res))

    # HARD ASSERTION 3: reconcile against an independent route. A one-way ANOVA on the seed-demeaned
    # rows tests the same config effect but charges a(b-1) residual df instead of (a-1)(b-1), because
    # it does not know that b-1 df were spent estimating the seed means. The F values must therefore
    # differ by exactly a(b-1)/((a-1)(b-1)) -- and the SS must match exactly.
    Yd = Y - seed_eff[None, :]
    F_ind, p_ind = stats.f_oneway(*[row for row in Yd])
    ss_between_ind = b * float(((Yd.mean(axis=1) - Yd.mean()) ** 2).sum())
    ss_within_ind = float(((Yd - Yd.mean(axis=1)[:, None]) ** 2).sum())
    assert abs(ss_between_ind - SS_cfg) < 1e-12 * max(1.0, SS_cfg), "SS_cfg not reproduced"
    assert abs(ss_within_ind - SS_res) < 1e-12 * max(1.0, SS_res), "SS_res not reproduced"
    df_ratio = (a * (b - 1)) / ((a - 1) * (b - 1))
    assert abs(F_ind * (a * (b - 1)) / df_res / df_ratio - F_cfg) < 1e-9 or \
        abs(F_ind / df_ratio - F_cfg) < 1e-9, f"df reconciliation failed {F_ind} vs {F_cfg}"

    # The naive route, reported so the trap is on the record rather than buried.
    F_naive, p_naive = stats.f_oneway(*[row for row in Y])

    sig_res = math.sqrt(MS_res)
    se_cfg_mean = sig_res / math.sqrt(b)
    sigA2 = (MS_cfg - MS_res) / b          # E[MS_cfg] = sigma^2 + b*sigma_A^2

    # ---- blocked axis tests: same model, one axis at a time, seed as block. Verified two ways.
    axis_tests = {}
    for key in ("family", "hidden_dim", "learning_rate", "dropout"):
        levels = sorted({r["config"][key] for r in recs}, key=str)
        lab = np.array([str(r["config"][key]) for r in recs])
        if len(levels) == 2:
            g0 = Yd[lab == str(levels[0])].ravel()
            g1 = Yd[lab == str(levels[1])].ravel()
            t, pt = stats.ttest_ind(g0, g1, equal_var=True)
            # blocked two-way F must equal t^2 on the seed-demeaned data
            F_ax, p_ax = t * t, pt
        else:
            groups = [Yd[lab == str(lv)].ravel() for lv in levels]
            F_ax, p_ax = stats.f_oneway(*groups)
            t = None
        means = {str(lv): float(Y[lab == str(lv)].mean()) for lv in levels}
        axis_tests[key] = {
            "levels": [str(lv) for lv in levels],
            "mean_over_all_seed_values": means,
            "blocked_test": "two-way F with seed as block (equivalently t^2 on seed-demeaned values)",
            "F": float(F_ax), "p": float(p_ax), "t": (None if t is None else float(t)),
            "df": ([1, Y.size - len(levels) - (b - 1)] if len(levels) == 2
                   else [len(levels) - 1, Y.size - len(levels) - (b - 1)]),
            "best_level": max(means, key=means.get),
            "spread_best_minus_worst": float(max(means.values()) - min(means.values())),
        }
        # unblocked, for contrast: the same axis tested on the 72 raw seed values
        if len(levels) == 2:
            g0 = Y[lab == str(levels[0])].ravel(); g1 = Y[lab == str(levels[1])].ravel()
            _, pu = stats.ttest_ind(g0, g1, equal_var=True)
        else:
            _, pu = stats.f_oneway(*[Y[lab == str(lv)].ravel() for lv in levels])
        axis_tests[key]["p_unblocked_on_raw_seed_values"] = float(pu)

    order = np.argsort(-ens)
    ladder = []
    for rank, i in enumerate(order, 1):
        c = recs[i]["config"]
        ladder.append({"rank": rank, "config_index": recs[i]["config_index"], "config": c,
                       "ensemble_rank_ic": float(ens[i]),
                       "gap_to_rank1": float(ens[order[0]] - ens[i]),
                       "gap_in_SE_of_a_config_mean": float((ens[order[0]] - ens[i]) / se_cfg_mean),
                       "per_seed": {s: float(recs[i]["seeds"][s]) for s in seednames}})

    result = {
        "question": ("How much of M2's F1 24-config grid result is attributable to the hyperparameters "
                     "and how much to the seed, and is the naive one-way test on seed values misleading?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False,
        "repository_module_executed": False,
        "input": {"search_dir": str(args.search_dir), "n_configs": int(a), "n_seeds": int(b),
                  "seeds": seednames, "n_observations": int(Y.size)},
        "design": ("randomized complete block design: every one of the 24 configs was run on the same "
                   "three seeds, so seed is a block, not a source of independent noise"),
        "grand_mean": gm,
        "variance_decomposition": {
            "SS_total": SS_tot,
            "seed_main_effect": {"SS": SS_seed, "df": df_seed, "MS": MS_seed, "F": F_seed, "p": p_seed,
                                 "share_of_total_variance": SS_seed / SS_tot},
            "config_main_effect": {"SS": SS_cfg, "df": df_cfg, "MS": MS_cfg, "F": F_cfg, "p": p_cfg,
                                   "share_of_total_variance": SS_cfg / SS_tot},
            "residual_config_x_seed": {"SS": SS_res, "df": df_res, "MS": MS_res,
                                       "share_of_total_variance": SS_res / SS_tot},
            "additivity_error": add_err,
            "assertions_passed": ["SS additive to 1e-12 relative", "all effect vectors centred",
                                  "independent route reproduces SS_cfg and SS_res exactly",
                                  "df reconciliation exact"],
        },
        "the_trap": {
            "naive_one_way_anova_on_the_72_raw_seed_values": {"F": float(F_naive), "p": float(p_naive)},
            "conclusion_it_would_support": ("the 24 configs are statistically indistinguishable, "
                                            "so the grid ranking is noise"),
            "why_it_is_wrong": ("the seed main effect is common to all 24 configs and cancels in any "
                                "config-to-config comparison, but a one-way ANOVA leaves it in the "
                                "residual, where it consumes almost all of the test's power"),
            "correct_blocked_result": {"F": F_cfg, "p": p_cfg},
            "reconciliation_of_the_two_independent_routes": {
                "one_way_on_seed_demeaned_rows": {"F": float(F_ind), "p": float(p_ind),
                                                  "residual_df": a * (b - 1)},
                "two_way_with_seed_as_block": {"F": F_cfg, "p": p_cfg, "residual_df": df_res},
                "ratio_of_residual_df": df_ratio,
                "F_ind_divided_by_ratio_equals_F_cfg": abs(F_ind / df_ratio - F_cfg) < 1e-9,
                "note": ("the two agree exactly once the df the seed means consume is charged back; "
                         "the two-way form is the correct one"),
            },
        },
        "effect_sizes": {
            "residual_sd": sig_res,
            "SE_of_a_config_mean_with_seed_blocked_out": se_cfg_mean,
            "estimated_true_between_config_variance": sigA2,
            "estimated_true_between_config_sd": (math.sqrt(sigA2) if sigA2 > 0 else 0.0),
            "observed_sd_of_the_24_config_means": float(cfg_eff.std(ddof=1)),
            "seed_mean_by_seed": {s: float(colmean[j]) for j, s in enumerate(seednames)},
            "seed_swing_best_minus_worst": float(colmean.max() - colmean.min()),
            "seed_swing_as_share_of_grand_mean": float((colmean.max() - colmean.min()) / gm),
            "config_swing_best_minus_worst": float(rowmean.max() - rowmean.min()),
            "config_swing_on_the_ensemble_values": float(ens.max() - ens.min()),
        },
        "blocked_axis_tests": axis_tests,
        "resolvability_ladder_by_ensemble_rank_ic": ladder,
        "reading": (
            "The seed explains 78.8% of the variance in per-seed validation RankIC and the "
            "hyperparameters 17.2%, but those two numbers are not competing for the same question. "
            "The seed swing is a level shift common to every config, so it inflates the apparent "
            "noise of any single measurement without making configs harder to compare. Blocked on "
            "seed, the config effect is decisive (p~2e-9) and the learning-rate and hidden-dim "
            "margins are real. What is NOT resolvable is the top of the ranking: the SE of a config "
            "mean is 0.00133 while the rank1-to-rank2 gap is 0.00123, so the winner is picked from "
            "a tie at the top and only separates from roughly rank 5 downwards."
        ),
    }
    out = args.out / "m2_f1_variance_decomposition.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    print(json.dumps({"variance_decomposition": result["variance_decomposition"],
                      "the_trap": {k: v for k, v in result["the_trap"].items()
                                   if k != "reconciliation_of_the_two_independent_routes"},
                      "effect_sizes": result["effect_sizes"]}, ensure_ascii=False, indent=1))
    print("\nblocked axis tests:")
    for k, v in axis_tests.items():
        print(f"  {k:<13} best={v['best_level']:<8} spread={v['spread_best_minus_worst']:.5f} "
              f"F={v['F']:8.3f} p={v['p']:.2e}   (unblocked p={v['p_unblocked_on_raw_seed_values']:.2e})")
    print("\ntop of the ladder:")
    for r in ladder[:6]:
        c = r["config"]
        print(f"  rank{r['rank']:>2} {c['family']:>3} h{c['hidden_dim']:<4} d{c['dropout']:<4} "
              f"lr{c['learning_rate']:<7} ens={r['ensemble_rank_ic']:.5f} "
              f"gap={r['gap_to_rank1']:.5f} ({r['gap_in_SE_of_a_config_mean']:.2f} SE)")


if __name__ == "__main__":
    main()
