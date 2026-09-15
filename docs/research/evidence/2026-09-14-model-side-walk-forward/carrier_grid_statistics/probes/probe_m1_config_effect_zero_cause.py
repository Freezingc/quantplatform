#!/usr/bin/env python3
"""Read-only probe: why is M1's configuration effect exactly zero while the other three carriers' are not?

§BO.1 reports that after blocking by fold and seed, three of the four gridded carriers show a highly
significant configuration effect (M0-base p=4.11e-07, M0-refined p=1.96e-08, M2 F1 p=3.24e-10) while
M1 does not (F=0.5668, p=0.9566), and §BO.1's second table prints M1's "true between-config sd" as
0.000000 because the variance component came out <= 0 and was truncated. §BO.9 lists exactly this as
the missing piece: no cause was given for M1's insensitivity. This probe closes that item.

Three candidate explanations are tested, and the probe is built so each one can fail:

  H1  M1's grid is too narrow to move the metric.
      Tested by reading the grid axes off the records themselves. If M1's capacity range were narrower
      than the carriers that DO show an effect, H1 would survive.

  H2  M1's zero is an underpowered estimate, not evidence of insensitivity.
      Tested two ways: (a) an exact upper confidence bound on the config sd from the F pivot
      MS_cfg/MS_res ~ (1 + b*c*sigma_c^2/sigma_res^2) * F(df_cfg, df_res), compared against the tree
      carriers' observed config sd; (b) a residual-bootstrap simulation that keeps M1's real fold,
      seed, config-x-fold and pure-error structure and injects a synthetic config effect of known sd,
      then reports P(sigA2 <= 0) and P(p < 0.05).

  H3  M1 searches only capacity and regularisation axes, while every carrier that does show an effect
      also searches an optimisation axis (learning_rate) or an architecture axis (family). If the
      config effect lives mostly on those axes, M1's grid omits the axis that matters.
      Tested by an orthogonal per-axis main-effect decomposition. All four grids are full balanced
      factorials, so per-axis SS computed from marginal level means is exact and the axis SS sum to
      <= SS_config, with the remainder being axis-interaction SS.

Cross-check requirement: this probe must first reproduce §BO.1's published M1 numbers
(F=0.5668, p=0.9566, config share 1.40%, residual sd 0.01067, sigA2 <= 0) from the same formulas
before any new claim is computed, because §BL shipped two arithmetic errors and §BN three more.

Safety: reads only JSON that finished runs already wrote. No database access, no weight or checkpoint
loaded, no repository module imported or executed, nothing written into any artifact directory, no
signal sent to any process. M2 (pid 3588838) is still training while this runs.
"""
from __future__ import annotations

import glob
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
OUT = Path(__file__).resolve().parent.parent / "measurements" / "m1_config_effect_zero.json"

CARRIERS = (
    ("M0-base", "m0_v2", "LightGBM"),
    ("M0-refined", "m0_v3", "LightGBM"),
    ("M1", "m1_full_v1", "ResidualMLP"),
    ("M2", "m2_v1", "TemporalGRU/TCN"),
)

# §BO.1 published values, used as a reproduction gate rather than as input.
PUBLISHED = {
    "M1": {"config_F": 0.5668, "config_p": 0.9566, "config_share": 0.0140, "resid_sd": 0.01067},
    "M0-base": {"config_F": 3.6392453927090602, "config_p": 4.1053225286178405e-07},
    "M0-refined": {"config_F": 4.1420, "config_p": 1.96e-08},
    "M2": {"config_F": 8.763803422612282, "config_p": 3.242826958234832e-10},
}
# §BO.1 second table: "true between-config sd"
PUBLISHED_SIGA = {"M0-base": 0.002756, "M0-refined": 0.002673, "M1": 0.0, "M2": 0.003709}


def load_grid(ws: Path):
    """{fold: [record,...]} from either storage layout; same two-layout handling as §BO's instrument."""
    out = {}
    for d in sorted(glob.glob(str(ws / "search" / "F*"))):
        recs = []
        for p in sorted(glob.glob(d + "/*.json")):
            j = json.load(open(p, encoding="utf-8"))
            r = j.get("result", {})
            if r.get("seed_validation_rank_ic") and r.get("ensemble_validation_rank_ic") is not None:
                recs.append({"config": j["config"], "ensemble": r["ensemble_validation_rank_ic"],
                             "seeds": r["seed_validation_rank_ic"]})
        if recs:
            out[Path(d).name] = recs
    if out:
        return out, "search/<fold>/c*.json"
    rp = ws / "report.json"
    if not rp.is_file():
        return {}, "absent"
    rep = json.load(open(rp, encoding="utf-8"))
    for fd in rep.get("folds", []):
        recs = []
        for sr in fd.get("search_results", []):
            if sr.get("seed_validation_rank_ic") and sr.get("ensemble_validation_rank_ic") is not None:
                recs.append({"config": sr.get("config", {}), "ensemble": sr["ensemble_validation_rank_ic"],
                             "seeds": sr["seed_validation_rank_ic"]})
        if recs:
            out[fd.get("fold_id", f"F{len(out)+1}")] = recs
    return out, "report.json folds[].search_results[]"


def build_tensor(by_fold):
    """Return (Y[a,b,c], ens[a,b], cfg_keys, folds, seeds, axis_levels)."""
    folds = sorted(by_fold)
    seeds = sorted(by_fold[folds[0]][0]["seeds"].keys())
    cfg_keys = sorted({json.dumps(r["config"], sort_keys=True) for r in by_fold[folds[0]]})
    a, b, c = len(cfg_keys), len(folds), len(seeds)
    Y = np.full((a, b, c), np.nan)
    ens = np.full((a, b), np.nan)
    parsed = [json.loads(k) for k in cfg_keys]
    for j, f in enumerate(folds):
        for r in by_fold[f]:
            i = cfg_keys.index(json.dumps(r["config"], sort_keys=True))
            for k, s in enumerate(seeds):
                Y[i, j, k] = float(r["seeds"][s])
            ens[i, j] = float(r["ensemble"])
    assert not np.isnan(Y).any(), "incomplete block"
    axes = {}
    for key in parsed[0]:
        vals = sorted({p[key] for p in parsed}, key=lambda v: (isinstance(v, str), v))
        axes[key] = {"levels": vals, "index": [vals.index(p[key]) for p in parsed]}
    return Y, ens, parsed, folds, seeds, axes


def decompose(Y: np.ndarray) -> dict:
    """The §BO blocked decomposition, same formulas, with the same three assertions."""
    a, b, c = Y.shape
    n = Y.size
    gm = float(Y.mean())
    m_cfg = Y.mean(axis=(1, 2)); m_fold = Y.mean(axis=(0, 2))
    m_seed = Y.mean(axis=(0, 1)); m_cf = Y.mean(axis=2)
    e_cfg = m_cfg - gm; e_fold = m_fold - gm; e_seed = m_seed - gm
    e_int = m_cf - m_cfg[:, None] - m_fold[None, :] + gm
    resid = Y - m_cf[:, :, None] - e_seed[None, None, :]
    SS = {"config": b * c * float((e_cfg ** 2).sum()),
          "fold": a * c * float((e_fold ** 2).sum()),
          "seed": a * b * float((e_seed ** 2).sum()),
          "config_x_fold": c * float((e_int ** 2).sum()),
          "residual": float((resid ** 2).sum())}
    SS_tot = float(((Y - gm) ** 2).sum())
    # Within-(config,fold) variation splits EXACTLY into the seed main effect plus the residual:
    # resid = Y - m_cf - e_seed is orthogonal to anything constant along k because e_seed is centred,
    # so SS_within = SS_seed + SS_res with df ab(c-1) = (c-1) + (ab-1)(c-1). This is the statistic
    # that decides whether blocking can remove a carrier's run-to-run noise at all.
    SS_within = float((((Y - m_cf[:, :, None]) ** 2)).sum())
    SS_seed_only = a * b * float((e_seed ** 2).sum())
    assert abs(SS_within - (SS_seed_only + SS["residual"])) < 1e-9 * SS_within, "within-cell split not exact"
    assert a * b * (c - 1) == (c - 1) + (a * b - 1) * (c - 1), "within-cell df bookkeeping"
    df = {"config": a - 1, "fold": b - 1, "seed": c - 1,
          "config_x_fold": (a - 1) * (b - 1), "residual": (a * b - 1) * (c - 1)}
    err = abs(sum(SS.values()) - SS_tot)
    assert err < 1e-9 * SS_tot, f"additivity broken {err:.3e}"
    for v in (e_cfg, e_fold, e_seed):
        assert abs(float(v.sum())) < 1e-11, "effect vector not centred"
    assert df["residual"] == n - 1 - df["config"] - df["fold"] - df["seed"] - df["config_x_fold"], "df bookkeeping"
    MS_res = SS["residual"] / df["residual"]
    F_cfg = (SS["config"] / df["config"]) / MS_res
    p_cfg = float(stats.f.sf(F_cfg, df["config"], df["residual"]))
    sigA2 = (SS["config"] / df["config"] - MS_res) / (b * c)
    return {"gm": gm, "a": a, "b": b, "c": c, "n": n, "SS": SS, "df": df, "SS_tot": SS_tot,
            "MS_res": float(MS_res), "resid_sd": float(math.sqrt(MS_res)),
            "config_F": float(F_cfg), "config_p": p_cfg,
            "config_share": SS["config"] / SS_tot,
            "sigA2_raw": float(sigA2), "sigA2_truncated": float(max(sigA2, 0.0)),
            "sigma_c_truncated": float(math.sqrt(max(sigA2, 0.0))),
            "SS_within_config_fold": SS_within, "SS_seed_main": SS_seed_only,
            "within_cell_variation_explained_by_seed_identity": SS_seed_only / SS_within,
            "within_cell_variation_left_after_blocking_seed": SS["residual"] / SS_within,
            "e_cfg": e_cfg, "e_fold": e_fold, "e_seed": e_seed, "e_int": e_int, "resid": resid,
            "MS": {k: SS[k] / df[k] for k in SS if df[k] > 0}}


def axis_decomposition(Y: np.ndarray, axes: dict, dec: dict) -> dict:
    """Orthogonal per-axis main-effect SS. Exact because every grid is a full balanced factorial."""
    a, b, c = Y.shape
    out = {}
    ss_sum = 0.0
    df_sum = 0
    for name, info in axes.items():
        idx = np.asarray(info["index"])
        levels = info["levels"]
        L = len(levels)
        if L < 2:
            continue
        # balance check: a full factorial gives every level of every axis the same number of configs
        counts = np.array([(idx == t).sum() for t in range(L)])
        assert counts.size == L and (counts == counts[0]).all(), f"{name}: grid is not a full factorial"
        n_obs_per_level = int(counts[0]) * b * c
        assert n_obs_per_level * L == Y.size, f"{name}: design not balanced on this axis"
        lvl_means = np.array([Y[idx == t].mean() for t in range(L)])
        SS_ax = float(n_obs_per_level * ((lvl_means - dec["gm"]) ** 2).sum())
        df_ax = L - 1
        F_ax = (SS_ax / df_ax) / dec["MS_res"]
        out[name] = {"levels": [str(v) for v in levels], "level_means": [float(v) for v in lvl_means],
                     "n_obs_per_level": n_obs_per_level,
                     "level_spread_max_minus_min": float(lvl_means.max() - lvl_means.min()),
                     "SS": SS_ax, "df": df_ax, "F": float(F_ax),
                     "p": float(stats.f.sf(F_ax, df_ax, dec["df"]["residual"])),
                     "share_of_SS_tot": SS_ax / dec["SS_tot"],
                     "share_of_SS_config": SS_ax / dec["SS"]["config"] if dec["SS"]["config"] > 0 else None}
        ss_sum += SS_ax
        df_sum += df_ax
    out["_axis_total"] = {"SS": float(ss_sum), "df": df_sum,
                          "SS_config": dec["SS"]["config"], "df_config": dec["df"]["config"],
                          "SS_axis_interaction_remainder": float(dec["SS"]["config"] - ss_sum),
                          "df_axis_interaction_remainder": dec["df"]["config"] - df_sum}
    assert ss_sum <= dec["SS"]["config"] + 1e-9 * dec["SS_tot"], "axis SS exceed config SS: not orthogonal"
    return out


def sigma_c_upper_bound(F_obs: float, df_cfg: int, df_res: int, MS_res: float, reps: int, alpha: float = 0.05):
    """Exact upper confidence bound on the true config sd from the F pivot.

    MS_cfg/MS_res ~ (1 + reps*sigma_c^2/sigma_res^2) * F(df_cfg, df_res), so with
    lam = reps*sigma_c^2/sigma_res^2 the bound solves F_obs/(1+lam_U) = F_alpha(df_cfg, df_res).
    """
    F_lo = float(stats.f.ppf(alpha, df_cfg, df_res))
    lam_U = F_obs / F_lo - 1.0
    if lam_U <= 0:
        return {"F_alpha_lower_quantile": F_lo, "lambda_upper": float(lam_U),
                "sigma_c_upper": 0.0, "note": "bound is non-positive; even the upper bound is 0"}
    var_U = lam_U * MS_res / reps
    return {"F_alpha_lower_quantile": F_lo, "lambda_upper": float(lam_U),
            "sigma2_c_upper": float(var_U), "sigma_c_upper": float(math.sqrt(var_U))}


def bootstrap_config_effect(Y: np.ndarray, dec: dict, sigma_c: float, n_sim: int, rng) -> dict:
    """Keep M1's real fold/seed/interaction/pure-error structure; inject a synthetic config effect.

    Y_sim = gm + A_i + e_fold_j + e_seed_k + e_int_ij + resampled(resid)
    A_i ~ N(0, sigma_c^2) (or all-zero when sigma_c == 0), redrawn per simulation.
    Returns P(sigA2 <= 0) and P(p < 0.05) under that true sigma_c.
    """
    a, b, c = Y.shape
    flat_resid = dec["resid"].ravel()
    le_zero = 0
    le_sig = 0
    siga2s = []
    for _ in range(n_sim):
        A = rng.normal(0.0, sigma_c, size=a) if sigma_c > 0 else np.zeros(a)
        A = A - A.mean()
        rs = rng.choice(flat_resid, size=flat_resid.size, replace=True).reshape(a, b, c)
        Ysim = (dec["gm"] + A[:, None, None] + dec["e_fold"][None, :, None]
                + dec["e_seed"][None, None, :] + dec["e_int"][:, :, None] + rs)
        d = decompose(Ysim)
        siga2s.append(d["sigA2_raw"])
        if d["sigA2_raw"] <= 0:
            le_zero += 1
        if d["config_p"] < 0.05:
            le_sig += 1
    arr = np.asarray(siga2s)
    return {"true_sigma_c": sigma_c, "n_sim": n_sim,
            "P_sigA2_le_zero": le_zero / n_sim, "P_p_lt_0.05": le_sig / n_sim,
            "mean_estimated_sigma_c": float(np.sqrt(np.clip(arr, 0, None)).mean()),
            "median_estimated_sigma_c": float(np.median(np.sqrt(np.clip(arr, 0, None)))),
            "q90_estimated_sigma_c": float(np.quantile(np.sqrt(np.clip(arr, 0, None)), 0.90))}


def bisect_sigma_c(Y: np.ndarray, dec: dict, key: str, target: float, lo: float, hi: float,
                   n_sim: int, rng, iters: int = 14) -> dict:
    """Simulation-based inversion of the estimator, because the analytic F-pivot bound degenerates.

    The analytic bound solves F_obs/(1+lam) = F_alpha, but M1's observed F (0.5668) sits BELOW the
    5% quantile of the null (0.5801), so lam_U comes out negative and the formula returns 0 -- an
    uninformative answer. Inverting the simulation instead gives two numbers that are usable:
      key="P_sigA2_le_zero", target=0.05 -> largest true sigma_c still compatible with having
                                           observed a truncated zero, i.e. a 95% upper bound.
      key="P_p_lt_0.05",     target=0.80 -> the minimum config sd this design detects at 80% power.
    """
    f = lambda s: bootstrap_config_effect(Y, dec, s, n_sim, rng)[key]
    flo, fhi = f(lo), f(hi)
    if (flo - target) * (fhi - target) > 0:
        return {"key": key, "target": target, "bracket": [lo, hi], "P_at_bracket": [flo, fhi],
                "sigma_c": None, "note": "not bracketed"}
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        fm = f(mid)
        if (flo - target) * (fm - target) <= 0:
            hi, fhi = mid, fm
        else:
            lo, flo = mid, fm
    return {"key": key, "target": target, "sigma_c": 0.5 * (lo + hi), "n_sim_per_eval": n_sim,
            "bracket": [lo, hi], "P_at_bracket": [flo, fhi]}


def noise_vs_signal_counterfactual(per: dict) -> dict:
    """Attribute the F collapse on M1's strongest axis to noise level versus to effect size.

    F_axis = (SS_axis/df_axis)/MS_res, and SS_axis scales with the square of the level spread, so
    both substitutions are exact rescalings of a quantity already computed -- no refitting.
    """
    m1 = per["M1"]
    best = max(((k, v) for k, v in m1["axis_main_effects"].items() if not k.startswith("_")),
               key=lambda kv: kv[1]["F"])
    bname, bax = best
    msr_m1 = m1["resid_sd"] ** 2
    out = {"m1_strongest_axis": bname, "m1_axis_F_observed": bax["F"], "m1_axis_p": bax["p"],
           "m1_axis_level_spread": bax["level_spread_max_minus_min"],
           "m1_resid_sd": m1["resid_sd"], "substitutions": []}
    for lab, _, _ in CARRIERS:
        if lab == "M1" or "skipped" in per[lab]:
            continue
        o = per[lab]
        # that carrier's strongest axis, for a like-for-like comparison
        obest = max(((k, v) for k, v in o["axis_main_effects"].items() if not k.startswith("_")),
                    key=lambda kv: kv[1]["F"])
        msr_o = o["resid_sd"] ** 2
        # The observed F gap between M1's strongest axis and this carrier's strongest decomposes
        # multiplicatively into MS_res ratio x spread ratio^2 x a remainder. The remainder is the
        # df/shape term: M1 searches 3-level axes (2 df) while the trees' strongest axis is 2-level
        # (1 df), so the same per-level spread buys a smaller MS. Log shares make the attribution
        # additive and are reported so "noise dominates" can be checked rather than asserted.
        F_ratio = obest[1]["F"] / bax["F"]
        noise_term = msr_m1 / msr_o
        spread_term = (obest[1]["level_spread_max_minus_min"] / bax["level_spread_max_minus_min"]) ** 2
        shape_term = F_ratio / (noise_term * spread_term)
        lt = math.log(F_ratio)
        out["substitutions"].append({
            "other_carrier": lab, "other_strongest_axis": obest[0], "other_axis_F": obest[1]["F"],
            "other_axis_level_spread": obest[1]["level_spread_max_minus_min"],
            "other_resid_sd": o["resid_sd"],
            "m1_axis_F_if_it_had_the_other_carriers_noise": bax["F"] * msr_m1 / msr_o,
            "m1_axis_F_if_it_had_the_other_carriers_spread": bax["F"] * (
                obest[1]["level_spread_max_minus_min"] / bax["level_spread_max_minus_min"]) ** 2,
            "m1_axis_F_if_it_had_both": bax["F"] * msr_m1 / msr_o * (
                obest[1]["level_spread_max_minus_min"] / bax["level_spread_max_minus_min"]) ** 2,
            "noise_ratio_ms_res": msr_m1 / msr_o,
            "spread_ratio": obest[1]["level_spread_max_minus_min"] / bax["level_spread_max_minus_min"],
            "F_gap_total_ratio": float(F_ratio),
            "shape_df_term": float(shape_term),
            "log_share_noise": float(math.log(noise_term) / lt),
            "log_share_signal_spread": float(math.log(spread_term) / lt),
            "log_share_axis_shape_df": float(math.log(shape_term) / lt),
        })
        assert abs(sum([math.log(noise_term) / lt, math.log(spread_term) / lt,
                        math.log(shape_term) / lt]) - 1.0) < 1e-9, "log shares do not sum to 1"
    return out


def seed_scaling(Y: np.ndarray, dec: dict, sigma_c: float, c_list, n_sim: int, rng) -> list:
    """What M1's design would detect if the budget went to MORE SEEDS instead of more configs.

    Blocking only removes a nuisance factor that is a reproducible shift. M1's seed main effect is
    0.88% of variance while its residual sd is 0.01067, so its run-to-run noise is not seed-labelled
    and no blocking scheme can remove it; the only lever is averaging over more seeds, which shrinks
    SE(config mean) = sqrt(MS_res/(b*c)). This synthesises c' seeds by resampling M1's own pooled
    residuals and reports the 80%-power minimum detectable config sd at each c'.
    """
    a, b, c = Y.shape
    flat_resid = dec["resid"].ravel()
    MS_seed = dec["MS"]["seed"]
    sig2_seed = max((MS_seed - dec["MS_res"]) / (a * b), 0.0)
    out = []
    for cp in c_list:
        lo, hi = 0.0, 0.02

        def power(s, cp=cp):
            hit = 0
            for _ in range(n_sim):
                A = rng.normal(0.0, s, size=a) if s > 0 else np.zeros(a)
                A = A - A.mean()
                sd = rng.normal(0.0, math.sqrt(sig2_seed), size=cp) if sig2_seed > 0 else np.zeros(cp)
                sd = sd - sd.mean()
                rs = rng.choice(flat_resid, size=a * b * cp, replace=True).reshape(a, b, cp)
                Ysim = (dec["gm"] + A[:, None, None] + dec["e_fold"][None, :, None]
                        + sd[None, None, :] + dec["e_int"][:, :, None] + rs)
                if decompose(Ysim)["config_p"] < 0.05:
                    hit += 1
            return hit / n_sim

        for _ in range(12):
            mid = 0.5 * (lo + hi)
            if power(mid) < 0.80:
                lo = mid
            else:
                hi = mid
        mde = 0.5 * (lo + hi)
        out.append({"n_seeds": int(cp), "min_detectable_sigma_c_at_80pct_power": float(mde),
                    "se_of_config_mean": float(math.sqrt(dec["MS_res"] / (b * cp))),
                    "analytic_scaling_from_c3": None})
    base = next((o for o in out if o["n_seeds"] == c), None)
    if base:
        for o in out:
            o["analytic_scaling_from_c3"] = base["min_detectable_sigma_c_at_80pct_power"] * math.sqrt(c / o["n_seeds"])
    return out


def main() -> None:
    rng = np.random.default_rng(20260916)
    per = {}
    for label, ws_name, model in CARRIERS:
        ws = ART / ws_name
        by_fold, layout = load_grid(ws)
        if not by_fold:
            per[label] = {"skipped": "no grid"}
            continue
        Y, ens, parsed, folds, seeds, axes = build_tensor(by_fold)
        dec = decompose(Y)
        ax = axis_decomposition(Y, axes, dec)
        # within-(config,fold) seed spread: the pure run-to-run noise, independent of the ANOVA table
        seed_spread = float(np.std(Y - Y.mean(axis=2, keepdims=True), ddof=1))
        cfg_means = Y.mean(axis=(1, 2))
        per[label] = {
            "workspace": ws_name, "model_family": model, "layout": layout,
            "n_configs": dec["a"], "n_folds": dec["b"], "n_seeds": dec["c"], "n_obs": dec["n"],
            "grid_axes": {k: [str(x) for x in v["levels"]] for k, v in axes.items()},
            "grid_capacity_span": {
                "hidden_dim_ratio": (max(axes["hidden_dim"]["levels"]) / min(axes["hidden_dim"]["levels"])
                                     if "hidden_dim" in axes and len(axes["hidden_dim"]["levels"]) > 1 else None),
                "searches_learning_rate": "learning_rate" in axes,
                "searches_architecture_family": "family" in axes,
                "searches_depth": any(k in axes for k in ("num_blocks", "max_depth")),
            },
            "config_F": dec["config_F"], "config_p": dec["config_p"],
            "config_share_of_variance": dec["config_share"],
            "df_config": dec["df"]["config"], "df_residual": dec["df"]["residual"],
            "resid_sd": dec["resid_sd"], "within_config_fold_seed_spread_sd": seed_spread,
            "SS_within_config_fold": dec["SS_within_config_fold"],
            "SS_seed_main": dec["SS_seed_main"],
            "within_cell_variation_explained_by_seed_identity": dec["within_cell_variation_explained_by_seed_identity"],
            "within_cell_variation_left_after_blocking_seed": dec["within_cell_variation_left_after_blocking_seed"],
            "sigA2_raw": dec["sigA2_raw"], "sigA2_truncated": dec["sigA2_truncated"],
            "sigma_c_truncated": dec["sigma_c_truncated"],
            "config_mean_range": float(cfg_means.max() - cfg_means.min()),
            "config_mean_sd_observed": float(cfg_means.std(ddof=1)),
            "se_of_a_config_mean_pooled": float(math.sqrt(dec["MS_res"] / (dec["b"] * dec["c"]))),
            "signal_to_noise_per_config_mean": float(cfg_means.std(ddof=1) / math.sqrt(dec["MS_res"] / (dec["b"] * dec["c"]))),
            "axis_main_effects": ax,
        }
        if dec["b"] > 1:
            per[label]["sigma_c_upper_bound_95"] = sigma_c_upper_bound(
                dec["config_F"], dec["df"]["config"], dec["df"]["residual"], dec["MS_res"], dec["b"] * dec["c"])

    # ---- reproduction gate on M1 before any new claim
    m1 = per["M1"]
    rep = {"config_F": (m1["config_F"], PUBLISHED["M1"]["config_F"]),
           "config_p": (m1["config_p"], PUBLISHED["M1"]["config_p"]),
           "config_share": (m1["config_share_of_variance"], PUBLISHED["M1"]["config_share"]),
           "resid_sd": (m1["resid_sd"], PUBLISHED["M1"]["resid_sd"])}
    repro = {k: {"this_probe": v[0], "published": v[1], "abs_diff": abs(v[0] - v[1])} for k, v in rep.items()}
    assert abs(m1["config_F"] - PUBLISHED["M1"]["config_F"]) < 5e-4, "M1 config F not reproduced"
    assert abs(m1["config_p"] - PUBLISHED["M1"]["config_p"]) < 5e-4, "M1 config p not reproduced"
    assert abs(m1["resid_sd"] - PUBLISHED["M1"]["resid_sd"]) < 5e-5, "M1 residual sd not reproduced"
    assert m1["sigA2_raw"] <= 0, "M1 sigA2 expected <= 0 (the published 0.000000 is a truncation)"
    for lab in ("M0-base", "M0-refined", "M2"):
        assert abs(per[lab]["config_F"] - PUBLISHED[lab]["config_F"]) < 5e-3 * PUBLISHED[lab]["config_F"] + 5e-4, \
            f"{lab} config F not reproduced"
        assert abs(per[lab]["sigma_c_truncated"] - PUBLISHED_SIGA[lab]) < 5e-4, f"{lab} sigma_c not reproduced"

    # ---- H2 simulation on M1's own noise structure
    by_fold, _ = load_grid(ART / "m1_full_v1")
    Y1, _, _, _, _, _ = build_tensor(by_fold)
    d1 = decompose(Y1)
    tree_sigma_c = [PUBLISHED_SIGA["M0-base"], PUBLISHED_SIGA["M0-refined"]]
    sims = [bootstrap_config_effect(Y1, d1, s, 400, rng)
            for s in [0.0] + tree_sigma_c + [PUBLISHED_SIGA["M2"], 0.005, 0.008, 0.01067]]
    # simulation-based inversion, because the analytic bound degenerates on M1 (see bisect_sigma_c)
    ub = bisect_sigma_c(Y1, d1, "P_sigA2_le_zero", 0.05, 0.0, 0.002756, 250, rng)
    mde = bisect_sigma_c(Y1, d1, "P_p_lt_0.05", 0.80, 0.0, 0.008, 250, rng)
    cf = noise_vs_signal_counterfactual(per)
    scaling = seed_scaling(Y1, d1, 0.00208, [3, 6, 12, 24], 120, rng)

    doc = {
        "question": ("§BO.9 open item: why is M1's configuration effect exactly zero while the tree carriers' "
                     "are not? Is M1 genuinely hyperparameter-insensitive, or is that zero an artefact of "
                     "power, of the axes M1 searched, or of M1's noise level?"),
        "reproduction_gate": repro,
        "per_carrier": per,
        "h2_simulation_on_m1_noise": sims,
        "h2_simulation_based_inversion": {"upper_bound_sigma_c_at_5pct_truncation": ub,
                                          "min_detectable_sigma_c_at_80pct_power": mde},
        "h3_noise_vs_signal_counterfactual": cf,
        "seed_scaling_for_m1": scaling,
        "read_only": True, "database_access": False, "weights_loaded": False, "training_performed": False,
        "repository_module_imported_or_executed": False,
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("reproduction gate (this probe vs §BO.1 published):")
    for k, v in repro.items():
        print(f"  {k:14s} {v['this_probe']:.6f} vs {v['published']:.6f}  diff {v['abs_diff']:.2e}")
    print()
    print(f"{'carrier':12s} {'cfgF':>8s} {'cfg p':>10s} {'share':>8s} {'resid sd':>9s} "
          f"{'sigma_c':>9s} {'sig/noise':>9s} {'lr?':>4s} {'arch?':>6s}")
    for lab, _, _ in CARRIERS:
        p = per[lab]
        if "skipped" in p:
            continue
        print(f"{lab:12s} {p['config_F']:8.4f} {p['config_p']:10.3e} {p['config_share_of_variance']*100:7.2f}% "
              f"{p['resid_sd']:9.5f} {p['sigma_c_truncated']:9.6f} {p['signal_to_noise_per_config_mean']:9.3f} "
              f"{str(p['grid_capacity_span']['searches_learning_rate']):>4s} "
              f"{str(p['grid_capacity_span']['searches_architecture_family']):>6s}")
    print()
    print("M1 sigma_c 95% upper bound:", json.dumps(per["M1"].get("sigma_c_upper_bound_95"), ensure_ascii=False))
    print("tree carriers' observed sigma_c:", tree_sigma_c)
    print()
    print("H2 simulation on M1's real noise structure (400 sims each):")
    for s in sims:
        print(f"  true sigma_c={s['true_sigma_c']:.6f}  P(sigA2<=0)={s['P_sigA2_le_zero']:.3f}  "
              f"P(p<0.05)={s['P_p_lt_0.05']:.3f}  median est sigma_c={s['median_estimated_sigma_c']:.6f}")
    print()
    print("H2 inversion (analytic bound degenerates because M1's F=0.5668 < F_0.05=0.5801):")
    print("  95% upper bound on M1's true sigma_c :", json.dumps(ub, ensure_ascii=False))
    print("  min detectable sigma_c at 80% power  :", json.dumps(mde, ensure_ascii=False))
    print()
    print("H3 counterfactual on M1's strongest axis "
          f"({cf['m1_strongest_axis']}, F={cf['m1_axis_F_observed']:.3f}, p={cf['m1_axis_p']:.4g}, "
          f"spread={cf['m1_axis_level_spread']:.5f}):")
    for s in cf["substitutions"]:
        print(f"  vs {s['other_carrier']:11s} ({s['other_strongest_axis']}, F={s['other_axis_F']:.1f}): "
              f"M1 axis F with their noise={s['m1_axis_F_if_it_had_the_other_carriers_noise']:.2f}, "
              f"with their spread={s['m1_axis_F_if_it_had_the_other_carriers_spread']:.2f}, "
              f"with both={s['m1_axis_F_if_it_had_both']:.2f} "
              f"(MS_res ratio {s['noise_ratio_ms_res']:.2f}, spread ratio {s['spread_ratio']:.3f})")
        print(f"      F gap {s['F_gap_total_ratio']:.2f}x = noise {s['noise_ratio_ms_res']:.2f} "
              f"x spread^2 {s['spread_ratio']**2:.2f} x shape/df {s['shape_df_term']:.2f}  |  log shares: "
              f"noise {s['log_share_noise']*100:.1f}%, signal {s['log_share_signal_spread']*100:.1f}%, "
              f"axis shape/df {s['log_share_axis_shape_df']*100:.1f}%")
    print()
    print("can blocking remove each carrier's run-to-run noise?")
    print(f"  {'carrier':12s} {'seed explains':>14s} {'left after blocking':>20s} {'resid sd':>9s}")
    for lab, _, _ in CARRIERS:
        p = per[lab]
        if "skipped" in p:
            continue
        print(f"  {lab:12s} {p['within_cell_variation_explained_by_seed_identity']*100:13.1f}% "
              f"{p['within_cell_variation_left_after_blocking_seed']*100:19.1f}% {p['resid_sd']:9.5f}")
    print()
    print("seed scaling for M1 (budget to more seeds rather than more configs):")
    for s in scaling:
        print(f"  c={s['n_seeds']:3d}  MDE(80% power)={s['min_detectable_sigma_c_at_80pct_power']:.6f}  "
              f"SE(config mean)={s['se_of_config_mean']:.6f}  "
              f"analytic 1/sqrt(c) scaling={s['analytic_scaling_from_c3']}")
    print()
    print("per-axis main effects (F, p, level spread):")
    for lab, _, _ in CARRIERS:
        p = per[lab]
        if "skipped" in p:
            continue
        print(f"  {lab}:")
        for name, v in p["axis_main_effects"].items():
            if name.startswith("_"):
                continue
            print(f"    {name:20s} levels={v['levels']} spread={v['level_spread_max_minus_min']:.5f} "
                  f"F={v['F']:7.3f} p={v['p']:.3e} share_of_SS_config="
                  f"{(v['share_of_SS_config']*100 if v['share_of_SS_config'] is not None else float('nan')):6.2f}%")
        t = p["axis_main_effects"]["_axis_total"]
        print(f"    {'[axis total]':20s} SS={t['SS']:.3e} df={t['df']}  "
              f"interaction remainder SS={t['SS_axis_interaction_remainder']:.3e} df={t['df_axis_interaction_remainder']}")
    print()
    print("wrote", OUT)


if __name__ == "__main__":
    main()
