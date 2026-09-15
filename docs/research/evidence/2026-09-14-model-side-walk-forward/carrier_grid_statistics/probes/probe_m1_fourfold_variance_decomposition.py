#!/usr/bin/env python3
"""Read-only probe: M1's grid across all four folds, as the replication §BL could not run yet.

§BL decomposed M2's F1 grid (24 configs x 3 seeds = 72 observations) and found that a one-way ANOVA
on the raw seed values gives p=0.98 -- "the configs are indistinguishable" -- while treating seed as a
block gives F(23,46)=8.76, p=3.2e-10. §BL.6 then noted that because all four folds share the same three
seeds, F1..F4 could be pooled in one blocked model to halve the standard error, but that had to wait for
M2 to finish.

It does not have to wait for M1. M1-full already has complete search records for all four folds
(27 configs x 4 folds x 3 seeds = 324 observations), on the same 15-factor order and the same four
dataset ids M2 consumes. So this probe runs the pooled analysis now, on a carrier that is finished, and
uses it to answer the question §BL could not even pose:

  Is the winning config stable across folds at all?

That question decides whether pooling is legitimate. If config x fold interaction is negligible, a single
config choice generalises and pooling buys real power. If it is large, each fold genuinely needs its own
selection and a pooled "best config" is an average over four different answers.

Two blocking factors (fold and seed) are removed, and the config x fold interaction is tested explicitly
rather than assumed away. Every sum of squares is cross-checked against the residual sum of squares of an
explicit least-squares fit of the same design matrix, because §BL shipped two arithmetic errors that only
an additivity/SS check caught.

Safety: reads only JSON search records the training runs already wrote. No database access, no weight
loading, no repository module imported or executed, nothing written into any artifact directory.
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


def load(workspace: Path):
    """Return {fold: [record, ...]} for every fold with complete per-seed results."""
    out = {}
    for fold_dir in sorted(glob.glob(str(workspace / "search" / "F*"))):
        fold = Path(fold_dir).name
        recs = []
        for p in sorted(glob.glob(fold_dir + "/*.json")):
            d = json.load(open(p, encoding="utf-8"))
            r = d.get("result", {})
            sv = r.get("seed_validation_rank_ic") or {}
            if sv and r.get("ensemble_validation_rank_ic") is not None:
                recs.append({"config": d["config"], "config_index": d.get("config_index"),
                             "ensemble": r["ensemble_validation_rank_ic"], "seeds": sv,
                             "worst_seed": r.get("worst_seed"), "path": p})
        if recs:
            out[fold] = recs
    return out


def design_rss(Y: np.ndarray, terms: list[np.ndarray]) -> float:
    """Residual sum of squares of an explicit least-squares fit, as an independent check on the SS.

    ``terms`` are dummy-variable blocks shaped like Y with one extra trailing axis (Y is C-ordered, so
    reshaping to (n, -1) keeps each observation aligned with its own row of dummies).
    """
    y = Y.ravel()
    X = np.column_stack([np.ones(y.size)] + [t.reshape(y.size, -1) for t in terms])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(((y - X @ beta) ** 2).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", type=Path, default=ART / "m1_full_v1")
    ap.add_argument("--out", type=Path, default=ART / "carrier_grid_statistics")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    by_fold = load(args.workspace)
    folds = sorted(by_fold)
    assert len(folds) == 4, f"expected 4 folds, got {folds}"
    seeds = sorted(by_fold[folds[0]][0]["seeds"].keys())
    n_cfg = len(by_fold[folds[0]])
    assert all(len(by_fold[f]) == n_cfg for f in folds), "unbalanced config count across folds"
    # the config identity must be the same list in the same order in every fold, or "config" is not a factor
    sig = {f: tuple(sorted(json.dumps(r["config"], sort_keys=True) for r in by_fold[f])) for f in folds}
    assert len(set(sig.values())) == 1, "the config set differs across folds; pooling would be meaningless"
    cfg_keys = sorted({json.dumps(r["config"], sort_keys=True) for r in by_fold[folds[0]]})
    a, b, c = len(cfg_keys), len(folds), len(seeds)

    Y = np.full((a, b, c), np.nan)
    ens = np.full((a, b), np.nan)
    worst = {}
    for j, f in enumerate(folds):
        for r in by_fold[f]:
            i = cfg_keys.index(json.dumps(r["config"], sort_keys=True))
            for k, s in enumerate(seeds):
                Y[i, j, k] = r["seeds"][s]
            ens[i, j] = r["ensemble"]
            worst[(i, j)] = str(r["worst_seed"])
    assert not np.isnan(Y).any(), "incomplete cell; the design is not a complete block"
    n = Y.size
    gm = float(Y.mean())

    # ---- marginal means
    m_cfg = Y.mean(axis=(1, 2))     # over folds and seeds
    m_fold = Y.mean(axis=(0, 2))    # over configs and seeds
    m_seed = Y.mean(axis=(0, 1))    # over configs and folds
    m_cfg_fold = Y.mean(axis=2)     # over seeds

    e_cfg = m_cfg - gm
    e_fold = m_fold - gm
    e_seed = m_seed - gm
    e_int = m_cfg_fold - m_cfg[:, None] - m_fold[None, :] + gm
    e_seed3 = m_seed - gm
    # the model is config + fold + seed + config x fold. The interaction absorbs the config and
    # fold main effects, so the fitted value for cell (i,j) is just its seed-averaged mean, and
    # the residual is Y minus that minus the seed effect. (An earlier draft subtracted the three
    # marginal means and added the cell mean back, which is not this model's residual; the
    # least-squares assertion below is what makes that class of mistake unable to ship.)
    resid = Y - m_cfg_fold[:, :, None] - e_seed3[None, None, :]

    SS_cfg = b * c * float((e_cfg ** 2).sum())
    SS_fold = a * c * float((e_fold ** 2).sum())
    SS_seed = a * b * float((e_seed ** 2).sum())
    SS_int = c * float((e_int ** 2).sum())
    SS_res = float((resid ** 2).sum())
    SS_tot = float(((Y - gm) ** 2).sum())

    df_cfg, df_fold, df_seed = a - 1, b - 1, c - 1
    df_int = (a - 1) * (b - 1)
    df_res = n - 1 - df_cfg - df_fold - df_seed - df_int

    # ---- HARD ASSERTIONS. §BL shipped two wrong decompositions; neither was visible in the F value.
    add_err = abs((SS_cfg + SS_fold + SS_seed + SS_int + SS_res) - SS_tot)
    assert add_err < 1e-10 * SS_tot, f"additivity broken: {add_err:.3e} on {SS_tot:.3e}"
    for v in (e_cfg, e_fold, e_seed):
        assert abs(float(v.sum())) < 1e-12, "an effect vector is not centred"
    assert abs(float(e_int.sum())) < 1e-10 and abs(float(resid.sum())) < 1e-8
    # independent route: least squares on an explicit design matrix must reproduce the residual SS
    ii, jj, kk = np.indices((a, b, c))
    d_cfg = np.stack([(ii == t).astype(float) for t in range(1, a)], -1)
    d_fold = np.stack([(jj == t).astype(float) for t in range(1, b)], -1)
    d_seed = np.stack([(kk == t).astype(float) for t in range(1, c)], -1)
    d_int = np.stack([((ii == p) & (jj == q)).astype(float)
                      for p in range(1, a) for q in range(1, b)], -1)
    rss_full = design_rss(Y, [d_cfg, d_fold, d_seed, d_int])
    assert abs(rss_full - SS_res) < 1e-8 * max(1.0, SS_res), \
        f"least-squares RSS {rss_full:.6e} != SS_res {SS_res:.6e}"
    rss_main = design_rss(Y, [d_cfg, d_fold, d_seed])
    assert abs((rss_main - SS_res) - SS_int) < 1e-8 * max(1.0, SS_tot), "interaction SS not reproduced"

    def ftest(SS, df, MS_res_, df_res_):
        F = (SS / df) / MS_res_
        return F, float(stats.f.sf(F, df, df_res_))

    MS_res = SS_res / df_res
    F_cfg, p_cfg = ftest(SS_cfg, df_cfg, MS_res, df_res)
    F_fold, p_fold = ftest(SS_fold, df_fold, MS_res, df_res)
    F_seed, p_seed = ftest(SS_seed, df_seed, MS_res, df_res)
    F_int, p_int = ftest(SS_int, df_int, MS_res, df_res)

    # the naive route, for the record: one-way over configs on all 324 raw values, nothing blocked
    F_naive, p_naive = stats.f_oneway(*[Y[i].ravel() for i in range(a)])
    # and the half-way route §BL used on M2: seed blocked, fold NOT
    Ysf = Y - e_seed[None, None, :]
    F_half, p_half = stats.f_oneway(*[Ysf[i].ravel() for i in range(a)])

    sig_res = math.sqrt(MS_res)
    se_cfg = sig_res / math.sqrt(b * c)          # a config mean averages over 4 folds x 3 seeds
    se_cfg_1fold = sig_res / math.sqrt(c)        # within a single fold
    sigA2 = (SS_cfg / df_cfg - MS_res) / (b * c)

    # ---- is the winner the same config in every fold?
    per_fold_winner = {}
    for j, f in enumerate(folds):
        i = int(np.argmax(ens[:, j]))
        per_fold_winner[f] = {
            "config": json.loads(cfg_keys[i]), "config_index": by_fold[f][0]["config_index"],
            "ensemble_rank_ic": float(ens[i, j]),
            "runner_up_gap": float(ens[i, j] - np.sort(ens[:, j])[-2]),
            "n_configs": int(a),
        }
    winners = {json.dumps(v["config"], sort_keys=True) for v in per_fold_winner.values()}
    # rank agreement between folds, on the ensemble values
    rho = {}
    for x in range(b):
        for y in range(x + 1, b):
            rho[f"{folds[x]}_vs_{folds[y]}"] = float(
                stats.spearmanr(ens[:, x], ens[:, y]).statistic)
    rho_vals = list(rho.values())
    # per-fold axis tests, blocked on seed only (fold-specific, so no pooling assumption)
    axes = sorted({k for r in by_fold[folds[0]] for k in r["config"]})
    axis_pooled, axis_per_fold = {}, {}
    # drop both blocking factors; the grand mean cancels in a between-level comparison
    Ysd = Y - e_seed[None, None, :] - e_fold[None, :, None]
    for ax in axes:
        levels = sorted({json.loads(k)[ax] for k in cfg_keys}, key=str)
        lab = np.array([str(json.loads(k)[ax]) for k in cfg_keys])
        groups = [Ysd[lab == str(lv)].ravel() for lv in levels]
        F_ax, p_ax = stats.f_oneway(*groups)
        means = {str(lv): float(Y[lab == str(lv)].mean()) for lv in levels}
        axis_pooled[ax] = {"levels": [str(lv) for lv in levels],
                           "mean_over_all_values": means,
                           "best_level": max(means, key=means.get),
                           "spread_best_minus_worst": float(max(means.values()) - min(means.values())),
                           "F": float(F_ax), "p": float(p_ax)}
        per = {}
        for j, f in enumerate(folds):
            Yf = Y[:, j, :] - e_seed[None, :]
            gf = [Yf[lab == str(lv)].ravel() for lv in levels]
            Fj, pj = stats.f_oneway(*gf)
            mj = {str(lv): float(Y[lab == str(lv), j, :].mean()) for lv in levels}
            per[f] = {"best_level": max(mj, key=mj.get), "F": float(Fj), "p": float(pj),
                      "spread": float(max(mj.values()) - min(mj.values()))}
        axis_per_fold[ax] = per

    # worst-seed census: is "the bad seed" a property of the seed, or of seed x fold?
    seed_x_fold = {f: {sd: float(Y[:, j, k].mean()) for k, sd in enumerate(seeds)}
                   for j, f in enumerate(folds)}

    # ---- BLOCKING ON seed x fold (12 blocks) instead of seed and fold separately.
    # The seed ranking is not the same in every fold, so a seed main effect is the wrong block: it
    # averages away a swing that is large within each fold. This is the model §BL.6 would have to use
    # before pooling M2's four folds, and M1 is where that can be checked without waiting for M2.
    blk = np.zeros((a, b, c))
    cell = Y.reshape(a, b * c)
    blkmean = cell.mean(axis=0)                     # one mean per (fold, seed) block
    e_blk = blkmean - gm
    SS_blk = a * float((e_blk ** 2).sum())
    SS_cfg_b = (b * c) * float((e_cfg ** 2).sum())
    # two-way residual: subtract BOTH the config mean and the block mean, then add the grand mean
    # back once. `cell - blkmean` alone is the within-block deviation, which still carries the whole
    # config effect -- the additivity assertion below is what caught that.
    fit_blk = e_cfg[:, None] + e_blk[None, :] + gm
    resid_blk = (cell - fit_blk).reshape(a, b, c)
    SS_res_b = float((resid_blk ** 2).sum())
    df_blk = b * c - 1
    df_res_b = n - 1 - df_cfg - df_blk
    assert abs((SS_cfg_b + SS_blk + SS_res_b) - SS_tot) < 1e-10 * SS_tot, "seedxfold blocking not additive"
    # independent check on this decomposition too: config dummies + 11 of the 12 (fold, seed) blocks
    blk_id = jj * c + kk
    d_blk = np.stack([(blk_id == t).astype(float) for t in range(1, b * c)], -1)
    rss_blk = design_rss(Y, [d_cfg, d_blk])
    assert abs(rss_blk - SS_res_b) < 1e-8 * max(1.0, SS_res_b), \
        f"seedxfold least-squares RSS {rss_blk:.6e} != SS_res {SS_res_b:.6e}"
    MS_res_b = SS_res_b / df_res_b
    F_cfg_b = (SS_cfg_b / df_cfg) / MS_res_b
    p_cfg_b = float(stats.f.sf(F_cfg_b, df_cfg, df_res_b))
    F_blk = (SS_blk / df_blk) / MS_res_b
    p_blk = float(stats.f.sf(F_blk, df_blk, df_res_b))
    seed_x_fold_swing = {f: float(max(seed_x_fold[f].values()) - min(seed_x_fold[f].values()))
                         for f in folds}
    seed_rank_by_fold = {f: [s for s, _ in sorted(seed_x_fold[f].items(), key=lambda kv: -kv[1])]
                         for f in folds}

    # ---- THE STATISTIC THE FRAMEWORK ACTUALLY SELECTS ON. Everything above is on per-seed values,
    # but select_model_config ranks configs by the ENSEMBLE rank IC, one value per config per fold.
    # Ensembling is not the mean of the seed ICs, so the two analyses are about different estimands
    # and can disagree. This measures whether they do.
    Ea, Eb = ens.shape
    gmE = float(ens.mean())
    e_cfgE = ens.mean(axis=1) - gmE
    e_foldE = ens.mean(axis=0) - gmE
    residE = ens - e_cfgE[:, None] - e_foldE[None, :] - gmE
    SS_cfgE = Eb * float((e_cfgE ** 2).sum())
    SS_foldE = Ea * float((e_foldE ** 2).sum())
    SS_resE = float((residE ** 2).sum())
    SS_totE = float(((ens - gmE) ** 2).sum())
    assert abs((SS_cfgE + SS_foldE + SS_resE) - SS_totE) < 1e-10 * SS_totE, "ensemble SS not additive"
    df_cfgE, df_foldE, df_resE = Ea - 1, Eb - 1, (Ea - 1) * (Eb - 1)
    MS_resE = SS_resE / df_resE
    F_cfgE = (SS_cfgE / df_cfgE) / MS_resE
    p_cfgE = float(stats.f.sf(F_cfgE, df_cfgE, df_resE))
    F_foldE = (SS_foldE / df_foldE) / MS_resE
    p_foldE = float(stats.f.sf(F_foldE, df_foldE, df_resE))
    ensE_d = ens - e_foldE[None, :]
    F_cfgE_ind, p_cfgE_ind = stats.f_oneway(*[row for row in ensE_d])
    se_cfgE = math.sqrt(MS_resE / Eb)

    axis_ens = {}
    for ax in axes:
        levels = sorted({json.loads(k)[ax] for k in cfg_keys}, key=str)
        lab = np.array([str(json.loads(k)[ax]) for k in cfg_keys])
        groups = [ensE_d[lab == str(lv)].ravel() for lv in levels]
        F_ax, p_ax = stats.f_oneway(*groups)
        means = {str(lv): float(ens[lab == str(lv)].mean()) for lv in levels}
        axis_ens[ax] = {"levels": [str(lv) for lv in levels], "mean_of_ensemble_values": means,
                        "best_level": max(means, key=means.get),
                        "spread_best_minus_worst": float(max(means.values()) - min(means.values())),
                        "F": float(F_ax), "p": float(p_ax),
                        "per_seed_best_level_for_contrast": axis_pooled[ax]["best_level"],
                        "the_two_statistics_agree":
                            max(means, key=means.get) == axis_pooled[ax]["best_level"]}
    # per-fold level means ON THE ENSEMBLE values, and the top-5 census, because the pooled level
    # means can hide a per-fold reversal -- which is exactly what the four winners at hidden_dim=128
    # turned out to be sitting on top of.
    axis_ens_per_fold = {}
    for ax in axes:
        levels = sorted({json.loads(k)[ax] for k in cfg_keys}, key=str)
        lab = np.array([str(json.loads(k)[ax]) for k in cfg_keys])
        per = {}
        for j, f in enumerate(folds):
            m = {str(lv): float(ens[lab == str(lv), j].mean()) for lv in levels}
            order = sorted(ens[:, j], reverse=True)
            top5 = [str(json.loads(cfg_keys[int(np.argsort(-ens[:, j])[:5][t])])[ax]) for t in range(5)]
            per[f] = {"level_means_on_ensemble_values": m,
                      "best_level": max(m, key=m.get),
                      "spread": float(max(m.values()) - min(m.values())),
                      "top5_census_of_this_axis": top5}
        bests = [per[f]["best_level"] for f in folds]
        axis_ens_per_fold[ax] = {
            "per_fold": per,
            "best_level_by_fold": {f: per[f]["best_level"] for f in folds},
            "n_distinct_best_levels_across_folds": len(set(bests)),
            "ordering_flips_across_folds": len(set(bests)) > 1,
            "pooled_best_level_for_contrast": axis_ens[ax]["best_level"],
        }
    winner_hidden_census = {}
    for f, v in per_fold_winner.items():
        for k, val in v["config"].items():
            winner_hidden_census.setdefault(k, {}).setdefault(str(val), 0)
            winner_hidden_census[k][str(val)] += 1

    # ---- WHY did all four per-fold winners have hidden_dim=128 when both statistics say 64 is the
    # better level on average? Selecting an argmax over noisy estimates is biased toward whichever
    # candidates are NOISIEST, not whichever are best. This measures each config's within-config seed
    # dispersion and asks whether the winners were the high-variance ones.
    cfg_sd = np.full((a, b), np.nan)
    for i in range(a):
        for j in range(b):
            cfg_sd[i, j] = float(np.std(Y[i, j, :], ddof=1))
    cfg_rng = np.full((a, b), np.nan)
    for i in range(a):
        for j in range(b):
            cfg_rng[i, j] = float(Y[i, j, :].max() - Y[i, j, :].min())
    is_winner = np.zeros((a, b), dtype=bool)
    for j in range(b):
        is_winner[int(np.argmax(ens[:, j])), j] = True
    sd_by_axis = {}
    for ax in axes:
        levels = sorted({json.loads(k)[ax] for k in cfg_keys}, key=str)
        lab = np.array([str(json.loads(k)[ax]) for k in cfg_keys])
        sd_by_axis[ax] = {
            str(lv): {"mean_within_config_seed_sd": float(cfg_sd[lab == str(lv)].mean()),
                      "mean_within_config_seed_range": float(cfg_rng[lab == str(lv)].mean()),
                      "mean_ensemble_value": float(ens[lab == str(lv)].mean()),
                      "mean_per_seed_value": float(Y[lab == str(lv)].mean())}
            for lv in levels}
    rho_sd_win, p_sd_win = stats.mannwhitneyu(
        cfg_sd[is_winner], cfg_sd[~is_winner], alternative="two-sided")
    argmax_bias = {
        "question": ("the four per-fold winners all had hidden_dim=128, while both the per-seed and the "
                     "ensemble axis tests put the better mean at hidden_dim=64; is argmax selection "
                     "picking the noisier configs rather than the better ones?"),
        "winners_n": int(is_winner.sum()), "non_winners_n": int((~is_winner).sum()),
        "mean_within_config_seed_sd_of_winners": float(cfg_sd[is_winner].mean()),
        "mean_within_config_seed_sd_of_non_winners": float(cfg_sd[~is_winner].mean()),
        "ratio": float(cfg_sd[is_winner].mean() / cfg_sd[~is_winner].mean()),
        "mannwhitney_U": float(rho_sd_win), "p_two_sided": float(p_sd_win),
        "point_biserial_corr_sd_vs_is_winner": float(np.corrcoef(
            cfg_sd.ravel(), is_winner.ravel().astype(float))[0, 1]),
        "by_axis": sd_by_axis,
        "reading": ("if the winners' within-config seed dispersion exceeds the non-winners', the "
                    "per-fold argmax is partly a selection for variance. That does not make the "
                    "selected model worse in expectation, but it does mean the identity of the winner "
                    "carries less information about the axis levels than the axis tests do."),
    }

    worst_census = {}
    for j, f in enumerate(folds):
        cnt = {s: 0 for s in seeds}
        for i in range(a):
            cnt[worst[(i, j)]] = cnt.get(worst[(i, j)], 0) + 1
        worst_census[f] = cnt

    result = {
        "question": ("Is M1's winning config stable across the four folds, and does the blocked variance "
                     "decomposition §BL ran on M2's F1 replicate on a carrier that has already finished "
                     "all four folds?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False,
        "repository_module_executed": False,
        "carrier": {"family": "RESIDUAL_MLP", "workspace": str(args.workspace), "folds": folds,
                    "seeds": seeds, "n_configs": a, "n_observations": n,
                    "grid": "hidden_dim{64,128,256} x num_blocks{1,2,3} x dropout{0.0,0.1,0.2}, a full 3^3 factorial",
                    "factor_order": json.load(
                        open(by_fold[folds[0]][0]["path"], encoding="utf-8"))
                        ["bindings"]["factor_order"]},
        "grand_mean": gm,
        "design": ("randomized complete block design with TWO blocking factors (fold, seed) and one "
                   "observation per config x fold x seed cell; the config x fold interaction is tested "
                   "explicitly rather than assumed away"),
        "variance_decomposition": {
            "SS_total": SS_tot, "additivity_error": add_err,
            "least_squares_RSS_reproduces_SS_res": True,
            "config": {"SS": SS_cfg, "df": df_cfg, "F": F_cfg, "p": p_cfg, "share": SS_cfg / SS_tot},
            "fold": {"SS": SS_fold, "df": df_fold, "F": F_fold, "p": p_fold, "share": SS_fold / SS_tot},
            "seed": {"SS": SS_seed, "df": df_seed, "F": F_seed, "p": p_seed, "share": SS_seed / SS_tot},
            "config_x_fold": {"SS": SS_int, "df": df_int, "F": F_int, "p": p_int, "share": SS_int / SS_tot},
            "residual": {"SS": SS_res, "df": df_res, "share": SS_res / SS_tot},
        },
        "the_three_routes": {
            "naive_one_way_over_configs_nothing_blocked": {"F": float(F_naive), "p": float(p_naive)},
            "seed_blocked_but_fold_not_what_BL_did_on_one_fold": {"F": float(F_half), "p": float(p_half)},
            "both_blocked_correct": {"F": F_cfg, "p": p_cfg},
            "reading": ("the same 324 numbers give three different p values for the same question; the "
                        "fold main effect is a second common level shift that, like the seed effect, "
                        "cancels in config-to-config comparison but destroys a one-way test's power"),
        },
        "effect_sizes": {
            "residual_sd": sig_res,
            "SE_of_a_config_mean_pooled_over_4_folds": se_cfg,
            "SE_of_a_config_mean_within_one_fold": se_cfg_1fold,
            "power_gain_from_pooling": se_cfg_1fold / se_cfg,
            "estimated_true_between_config_sd": (math.sqrt(sigA2) if sigA2 > 0 else 0.0),
            "fold_means": {f: float(m_fold[j]) for j, f in enumerate(folds)},
            "fold_swing": float(m_fold.max() - m_fold.min()),
            "seed_means": {s: float(m_seed[k]) for k, s in enumerate(seeds)},
            "seed_swing": float(m_seed.max() - m_seed.min()),
            "seed_swing_as_share_of_grand_mean": float((m_seed.max() - m_seed.min()) / gm),
        },
        "is_the_winner_stable_across_folds": {
            "per_fold_winner": per_fold_winner,
            "n_distinct_winners": len(winners),
            "distinct_winner_configs": [json.loads(w) for w in sorted(winners)],
            "config_rank_agreement_between_folds_spearman": rho,
            "rank_agreement_median": float(np.median(rho_vals)),
            "rank_agreement_min": float(min(rho_vals)),
            "rank_agreement_max": float(max(rho_vals)),
        },
        "axis_tests_blocked_on_seed_and_fold": axis_pooled,
        "axis_tests_per_fold_blocked_on_seed_only": axis_per_fold,
        "blocking_on_seed_x_fold_instead_of_seed_and_fold": {
            "why": ("the seed ranking is not the same in every fold, so a seed main effect averages "
                    "away a swing that is large within each fold; this is the model §BL.6 must use "
                    "before pooling M2's four folds"),
            "n_blocks": b * c,
            "block_effect": {"SS": SS_blk, "df": df_blk, "F": F_blk, "p": p_blk, "share": SS_blk / SS_tot},
            "config_effect": {"SS": SS_cfg_b, "df": df_cfg, "F": F_cfg_b, "p": p_cfg_b,
                              "share": SS_cfg_b / SS_tot},
            "residual": {"SS": SS_res_b, "df": df_res_b, "share": SS_res_b / SS_tot},
            "config_p_with_seed_as_a_main_effect": p_cfg,
            "config_p_with_seed_x_fold_as_blocks": p_cfg_b,
            "seed_swing_within_each_fold": seed_x_fold_swing,
            "seed_swing_of_the_pooled_main_effect": float(m_seed.max() - m_seed.min()),
            "seed_ranking_best_to_worst_by_fold": seed_rank_by_fold,
        },
        "the_statistic_the_framework_selects_on": {
            "note": ("select_model_config ranks configs by ensemble rank IC, one value per config per "
                     "fold, not by per-seed IC; the ensemble is scored on averaged predictions and is "
                     "not the mean of the seed ICs, so the two analyses have different estimands"),
            "n_ensemble_values": int(ens.size), "grand_mean": gmE,
            "config_effect": {"SS": SS_cfgE, "df": df_cfgE, "F": F_cfgE, "p": p_cfgE,
                              "share": SS_cfgE / SS_totE},
            "fold_effect": {"SS": SS_foldE, "df": df_foldE, "F": F_foldE, "p": p_foldE,
                            "share": SS_foldE / SS_totE},
            "residual": {"SS": SS_resE, "df": df_resE, "share": SS_resE / SS_totE},
            "independent_check_one_way_on_fold_demeaned_rows": {"F": float(F_cfgE_ind), "p": float(p_cfgE_ind)},
            "SE_of_a_config_ensemble_mean_over_4_folds": se_cfgE,
            "axis_tests_on_ensemble_values_blocked_on_fold": axis_ens,
            "census_of_the_four_per_fold_winners_by_axis": winner_hidden_census,
            "axis_level_means_per_fold_on_ensemble_values": axis_ens_per_fold,
        },
        "argmax_selection_bias_check": argmax_bias,
        "worst_seed_census_by_fold": worst_census,
        "seed_mean_by_fold": seed_x_fold,
        "m2_f1_reference_from_BL": {
            "note": "for comparison only; M2 F1 is a different carrier and a single fold",
            "seed_share": 0.7882198746695456, "config_share": 0.1724296992746318,
            "residual_share": 0.03935042605582192, "F_config": 8.763803422612282,
            "p_config": 3.242826958234832e-10, "naive_one_way_p": 0.9835017916353112,
        },
    }
    out = args.out / "m1_fourfold_variance_decomposition.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    print(f"design: {a} configs x {b} folds x {c} seeds = {n} observations, grand mean {gm:.6f}")
    print(f"additivity error {add_err:.3e}; least-squares RSS reproduces SS_res")
    print("\nvariance decomposition:")
    for k in ("config", "fold", "seed", "config_x_fold", "residual"):
        v = result["variance_decomposition"][k]
        f_ = v.get("F"); p_ = v.get("p")
        print(f"  {k:<14} share {100*v['share']:6.2f}%  df {v['df']:>3}  "
              + (f"F={f_:9.3f} p={p_:.3e}" if f_ is not None else ""))
    print("\nthree routes to the same question:")
    for k, v in result["the_three_routes"].items():
        if isinstance(v, dict):
            print(f"  {k:<52} F={v['F']:8.3f} p={v['p']:.3e}")
    w = result["is_the_winner_stable_across_folds"]
    print(f"\ndistinct winners across the four folds: {w['n_distinct_winners']}")
    for f, v in w["per_fold_winner"].items():
        print(f"  {f}: {v['config']} ens={v['ensemble_rank_ic']:.5f} gap_to_runner_up={v['runner_up_gap']:.5f}")
    print(f"  cross-fold rank agreement (Spearman): median {w['rank_agreement_median']:+.4f}, "
          f"min {w['rank_agreement_min']:+.4f}, max {w['rank_agreement_max']:+.4f}")
    print("\naxis tests (blocked on seed AND fold):")
    for ax, v in axis_pooled.items():
        print(f"  {ax:<11} best={v['best_level']:<5} spread={v['spread_best_minus_worst']:.5f} "
              f"F={v['F']:8.3f} p={v['p']:.3e}")
    print("\nper-fold best level of each axis:")
    for ax, per in axis_per_fold.items():
        print(f"  {ax:<11} " + "  ".join(f"{f}:{v['best_level']}(p={v['p']:.3f})" for f, v in per.items()))
    print("\n--- blocking on seed x fold (12 blocks) instead of seed and fold separately ---")
    sb = result["blocking_on_seed_x_fold_instead_of_seed_and_fold"]
    print(f"  block effect: share {100*sb['block_effect']['share']:.2f}%  F={sb['block_effect']['F']:.3f} p={sb['block_effect']['p']:.3e}")
    print(f"  config effect: share {100*sb['config_effect']['share']:.2f}%  F={sb['config_effect']['F']:.3f} p={sb['config_effect']['p']:.3e}")
    print(f"  config p: seed-as-main-effect {sb['config_p_with_seed_as_a_main_effect']:.4f} vs seedxfold-as-blocks {sb['config_p_with_seed_x_fold_as_blocks']:.4f}")
    print(f"  seed swing WITHIN each fold: " + ", ".join(f"{k}={v:.5f}" for k, v in sb["seed_swing_within_each_fold"].items()))
    print(f"  seed swing of the POOLED main effect: {sb['seed_swing_of_the_pooled_main_effect']:.5f}")
    print(f"  seed ranking best->worst by fold: " + json.dumps(sb["seed_ranking_best_to_worst_by_fold"]))
    print("\n--- the statistic the framework actually selects on (ensemble values, 27 x 4) ---")
    se_ = result["the_statistic_the_framework_selects_on"]
    for k in ("config_effect", "fold_effect", "residual"):
        v = se_[k]
        print(f"  {k:<14} share {100*v['share']:6.2f}%  df {v['df']:>3}  " + (f"F={v['F']:9.3f} p={v['p']:.3e}" if "F" in v else ""))
    for ax, v in se_["axis_tests_on_ensemble_values_blocked_on_fold"].items():
        agree = "AGREE" if v["the_two_statistics_agree"] else "**DISAGREE**"
        print(f"  {ax:<11} ensemble best={v['best_level']:<5} spread={v['spread_best_minus_worst']:.5f} "
              f"F={v['F']:7.3f} p={v['p']:.4f}   per-seed best={v['per_seed_best_level_for_contrast']:<5} {agree}")
    print(f"  census of the four per-fold winners by axis: {json.dumps(se_['census_of_the_four_per_fold_winners_by_axis'])}")
    for ax, v in se_["axis_level_means_per_fold_on_ensemble_values"].items():
        flip = "FLIPS" if v["ordering_flips_across_folds"] else "stable"
        print(f"  {ax}: per-fold best level {json.dumps(v['best_level_by_fold'])} -> {flip}; "
              f"pooled best {v['pooled_best_level_for_contrast']}")
        for f, pf in v["per_fold"].items():
            print(f"      {f}: " + "  ".join(f"{k}={m:.5f}" for k, m in
                  sorted(pf["level_means_on_ensemble_values"].items(), key=lambda kv: float(kv[0])))
                  + f"   spread={pf['spread']:.5f}  top5={pf['top5_census_of_this_axis']}")
    print("\n--- argmax selection bias check ---")
    ab = result["argmax_selection_bias_check"]
    print(f"  winners' mean within-config seed sd = {ab['mean_within_config_seed_sd_of_winners']:.5f}")
    print(f"  non-winners'                        = {ab['mean_within_config_seed_sd_of_non_winners']:.5f}")
    print(f"  ratio = {ab['ratio']:.3f}   Mann-Whitney p = {ab['p_two_sided']:.4f}")
    for ax, lv in ab["by_axis"].items():
        print(f"  {ax}:")
        for k, v in lv.items():
            print(f"     {k:<5} seed_sd={v['mean_within_config_seed_sd']:.5f} "
                  f"range={v['mean_within_config_seed_range']:.5f} "
                  f"ens_mean={v['mean_ensemble_value']:.5f} perseed_mean={v['mean_per_seed_value']:.5f}")
    print("\nworst-seed census by fold:", json.dumps(worst_census))
    print(f"\nSE of a config mean: pooled {se_cfg:.5f} vs within one fold {se_cfg_1fold:.5f} "
          f"-> pooling buys {se_cfg_1fold/se_cfg:.2f}x")


if __name__ == "__main__":
    main()
