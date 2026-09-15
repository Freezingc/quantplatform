#!/usr/bin/env python3
"""Read-only probe: the same blocked variance decomposition, run on every carrier that has a grid.

§BL ran it on M2's F1 (one fold). §BN ran it on M1's four folds and found the config effect
undetectable there, the seed effect reversing direction between folds, and four unrelated winners.
§BN.8 then left open the question that actually decides how much any of this generalises: do the
LightGBM tree carriers behave like the neural ones? The 14:57 entry had already noticed that the
tree carriers' within-config-seed-spread / between-config-range ratio is 0.555 and 0.677 while both
neural carriers sit at 1.721 and 1.722, and explicitly did not explain it.

This probe runs one instrument over every carrier whose per-seed grid results are on disk, so the
comparison is like-for-like rather than four separately-written analyses:

  M0-base     m0_v2       LightGBM   24 configs x 4 folds x 3 seeds = 288
  M0-refined  m0_v3       LightGBM   24 configs x 4 folds x 3 seeds = 288
  M1          m1_full_v1  MLP        27 configs x 4 folds x 3 seeds = 324
  M2          m2_v1       GRU/TCN    24 configs x 1 fold  x 3 seeds =  72  (F2 still running)

M0R is excluded: it has one config per fold, so there is no grid to decompose.

Two storage layouts are handled, because the carriers do not agree on one: the neural carriers write
search/<fold>/c*.json while the tree carriers keep the same records inside report.json under
folds[].search_results[]. Reading them through one loader is what makes the cross-carrier numbers
comparable at all.

Every decomposition is asserted three ways, because §BL shipped two arithmetic errors and §BN three
more, and in all five cases the F value looked plausible while the sums of squares did not add up:
exact additivity, every effect vector centred, and the residual SS reproduced by an explicit
least-squares fit of the same design matrix.

Safety: reads only JSON the finished runs already wrote. No database access, no weight or checkpoint
loaded, no repository module imported or executed, nothing written into any artifact directory.
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

CARRIERS = (
    ("M0-base", "m0_v2", "LightGBM"),
    ("M0-refined", "m0_v3", "LightGBM"),
    ("M1", "m1_full_v1", "ResidualMLP"),
    ("M2", "m2_v1", "TemporalGRU/TCN"),
)


def design_rss(Y: np.ndarray, terms: list[np.ndarray]) -> float:
    """RSS of an explicit least-squares fit; the independent check on every SS below."""
    y = Y.ravel()
    X = np.column_stack([np.ones(y.size)] + [t.reshape(y.size, -1) for t in terms])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(((y - X @ beta) ** 2).sum())


def load_grid(ws: Path):
    """Return {fold: [{config, ensemble, seeds, worst_seed}, ...]} from either storage layout."""
    out = {}
    for d in sorted(glob.glob(str(ws / "search" / "F*"))):
        recs = []
        for p in sorted(glob.glob(d + "/*.json")):
            j = json.load(open(p, encoding="utf-8"))
            r = j.get("result", {})
            if r.get("seed_validation_rank_ic") and r.get("ensemble_validation_rank_ic") is not None:
                recs.append({"config": j["config"], "ensemble": r["ensemble_validation_rank_ic"],
                             "seeds": r["seed_validation_rank_ic"], "worst_seed": r.get("worst_seed")})
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
                recs.append({"config": sr.get("config", {}),
                             "ensemble": sr["ensemble_validation_rank_ic"],
                             "seeds": sr["seed_validation_rank_ic"],
                             "worst_seed": sr.get("worst_seed")})
        if recs:
            out[fd.get("fold_id", f"F{len(out)+1}")] = recs
    return out, "report.json folds[].search_results[]"


def analyse(label: str, ws: Path, model: str) -> dict:
    by_fold, layout = load_grid(ws)
    if not by_fold:
        return {"carrier": label, "workspace": str(ws), "model_family": model, "skipped": "no grid found"}
    folds = sorted(by_fold)
    seeds = sorted(by_fold[folds[0]][0]["seeds"].keys())
    assert all(sorted(by_fold[f][0]["seeds"].keys()) == seeds
               for f in folds for _ in (0,)), f"{label}: seed set differs across folds"
    cfg_keys = sorted({json.dumps(r["config"], sort_keys=True) for r in by_fold[folds[0]]})
    a = len(cfg_keys)
    assert all(len({json.dumps(r["config"], sort_keys=True) for r in by_fold[f]}) == a for f in folds), \
        f"{label}: config count differs across folds"
    b, c = len(folds), len(seeds)

    Y = np.full((a, b, c), np.nan)
    ens = np.full((a, b), np.nan)
    worst = {}
    for j, f in enumerate(folds):
        for r in by_fold[f]:
            i = cfg_keys.index(json.dumps(r["config"], sort_keys=True))
            for k, s in enumerate(seeds):
                Y[i, j, k] = float(r["seeds"][s])
            ens[i, j] = float(r["ensemble"])
            worst[(i, j)] = str(r["worst_seed"])
    assert not np.isnan(Y).any(), f"{label}: incomplete block"
    n = Y.size
    gm = float(Y.mean())

    m_cfg = Y.mean(axis=(1, 2)); m_fold = Y.mean(axis=(0, 2))
    m_seed = Y.mean(axis=(0, 1)); m_cf = Y.mean(axis=2)
    e_cfg = m_cfg - gm; e_fold = m_fold - gm; e_seed = m_seed - gm
    e_int = m_cf - m_cfg[:, None] - m_fold[None, :] + gm
    e_seed3 = e_seed
    resid = Y - m_cf[:, :, None] - e_seed3[None, None, :]

    SS_cfg = b * c * float((e_cfg ** 2).sum())
    SS_fold = a * c * float((e_fold ** 2).sum())
    SS_seed = a * b * float((e_seed ** 2).sum())
    SS_int = c * float((e_int ** 2).sum())
    SS_res = float((resid ** 2).sum())
    SS_tot = float(((Y - gm) ** 2).sum())
    df_cfg, df_fold, df_seed = a - 1, b - 1, c - 1
    df_int = (a - 1) * (b - 1)
    df_res = (a * b - 1) * (c - 1)

    # ---- three assertions; all five arithmetic errors across §BL and §BN were caught by these
    add_err = abs((SS_cfg + SS_fold + SS_seed + SS_int + SS_res) - SS_tot)
    assert add_err < 1e-9 * SS_tot, f"{label}: additivity broken {add_err:.3e}"
    for v in (e_cfg, e_fold, e_seed):
        assert abs(float(v.sum())) < 1e-11, f"{label}: an effect vector is not centred"
    assert abs(float(resid.sum())) < 1e-7, f"{label}: residual not centred"
    ii, jj, kk = np.indices((a, b, c))
    d_cfg = np.stack([(ii == t).astype(float) for t in range(1, a)], -1)
    d_fold = np.stack([(jj == t).astype(float) for t in range(1, b)], -1) if b > 1 else np.zeros((a, b, c, 0))
    d_seed = np.stack([(kk == t).astype(float) for t in range(1, c)], -1)
    d_int = (np.stack([((ii == p_) & (jj == q_)).astype(float)
                       for p_ in range(1, a) for q_ in range(1, b)], -1) if b > 1
             else np.zeros((a, b, c, 0)))
    assert abs(design_rss(Y, [d_cfg, d_fold, d_seed, d_int]) - SS_res) < 1e-7 * max(1.0, SS_res), \
        f"{label}: least-squares RSS does not reproduce SS_res"
    assert df_res == n - 1 - df_cfg - df_fold - df_seed - df_int, f"{label}: df bookkeeping"

    MS_res = SS_res / df_res

    def ft(SS, df):
        if df <= 0:
            return {"SS": SS, "df": df, "F": None, "p": None, "share": SS / SS_tot}
        F = (SS / df) / MS_res
        return {"SS": SS, "df": df, "F": float(F), "p": float(stats.f.sf(F, df, df_res)), "share": SS / SS_tot}

    main = {"config": ft(SS_cfg, df_cfg), "fold": ft(SS_fold, df_fold), "seed": ft(SS_seed, df_seed),
            "config_x_fold": ft(SS_int, df_int), "residual": {"SS": SS_res, "df": df_res, "share": SS_res / SS_tot}}

    # naive and half-blocked routes, for the same three-way comparison §BN made on M1
    F_naive, p_naive = stats.f_oneway(*[Y[i].ravel() for i in range(a)])
    Ys = Y - e_seed[None, None, :]
    F_sb, p_sb = stats.f_oneway(*[Ys[i].ravel() for i in range(a)])

    # seed x fold as b*c blocks
    cell = Y.reshape(a, b * c)
    blkmean = cell.mean(axis=0)
    e_blk = blkmean - gm
    SS_blk = a * float((e_blk ** 2).sum())
    fit_blk = e_cfg[:, None] + e_blk[None, :] + gm
    SS_res_b = float(((cell - fit_blk) ** 2).sum())
    df_blk = b * c - 1
    df_res_b = n - 1 - df_cfg - df_blk
    assert abs((SS_cfg + SS_blk + SS_res_b) - SS_tot) < 1e-9 * SS_tot, f"{label}: seedxfold not additive"
    blk_id = jj * c + kk
    d_blk = np.stack([(blk_id == t).astype(float) for t in range(1, b * c)], -1)
    assert abs(design_rss(Y, [d_cfg, d_blk]) - SS_res_b) < 1e-7 * max(1.0, SS_res_b), \
        f"{label}: seedxfold RSS mismatch"
    MS_res_b = SS_res_b / df_res_b
    F_cfg_b = (SS_cfg / df_cfg) / MS_res_b
    p_cfg_b = float(stats.f.sf(F_cfg_b, df_cfg, df_res_b))

    sigA2 = (SS_cfg / df_cfg - MS_res) / (b * c)
    se_pool = math.sqrt(MS_res / (b * c))
    se_fold = math.sqrt(MS_res / c)

    # winners, rank agreement, seed direction
    winners = {}
    for j, f in enumerate(folds):
        i = int(np.argmax(ens[:, j]))
        srt = np.sort(ens[:, j])
        winners[f] = {"config": json.loads(cfg_keys[i]), "ensemble_rank_ic": float(ens[i, j]),
                      "gap_to_runner_up": float(srt[-1] - srt[-2]) if a > 1 else None,
                      "gap_in_SE": float((srt[-1] - srt[-2]) / se_fold) if a > 1 else None}
    rho = {}
    if b > 1 and a > 2:
        for x in range(b):
            for y in range(x + 1, b):
                rho[f"{folds[x]}_vs_{folds[y]}"] = float(stats.spearmanr(ens[:, x], ens[:, y]).statistic)
    seed_by_fold = {f: {s: float(Y[:, j, k].mean()) for k, s in enumerate(seeds)}
                    for j, f in enumerate(folds)}
    worst_census = {}
    for j, f in enumerate(folds):
        cnt = {s: 0 for s in seeds}
        for i in range(a):
            cnt[worst[(i, j)]] = cnt.get(worst[(i, j)], 0) + 1
        worst_census[f] = cnt

    # ensemble statistic (what select_model_config actually ranks on)
    gmE = float(ens.mean())
    e_cfgE = ens.mean(axis=1) - gmE; e_foldE = ens.mean(axis=0) - gmE
    residE = ens - e_cfgE[:, None] - e_foldE[None, :] - gmE
    SS_cfgE = b * float((e_cfgE ** 2).sum()); SS_foldE = a * float((e_foldE ** 2).sum())
    SS_resE = float((residE ** 2).sum()); SS_totE = float(((ens - gmE) ** 2).sum())
    assert abs((SS_cfgE + SS_foldE + SS_resE) - SS_totE) < 1e-9 * SS_totE, f"{label}: ensemble SS not additive"
    df_cfgE, df_foldE = a - 1, b - 1
    df_resE = (a - 1) * (b - 1)
    ens_block = {"config": None, "fold": None, "residual_share": SS_resE / SS_totE,
                 "grand_mean": gmE, "n_values": int(ens.size)}
    if df_resE > 0:
        MS_resE = SS_resE / df_resE
        for nm, SS, df in (("config", SS_cfgE, df_cfgE), ("fold", SS_foldE, df_foldE)):
            if df > 0:
                F = (SS / df) / MS_resE
                ens_block[nm] = {"share": SS / SS_totE, "df": df, "F": float(F),
                                 "p": float(stats.f.sf(F, df, df_resE))}
        ens_block["SE_of_a_config_ensemble_mean"] = math.sqrt(MS_resE / b)

    # axis tests, blocked on seed and fold, on both statistics
    axes = sorted({k for r in by_fold[folds[0]] for k in r["config"]})
    Ysf = Y - e_seed[None, None, :] - e_fold[None, :, None]
    ensE_d = ens - e_foldE[None, :] if b > 1 else ens - gmE
    axis_out = {}
    for ax in axes:
        levels = sorted({json.loads(k)[ax] for k in cfg_keys}, key=str)
        if len(levels) < 2:
            axis_out[ax] = {"constant_at": str(levels[0]), "note": "not searched; fixed in every config"}
            continue
        lab = np.array([str(json.loads(k)[ax]) for k in cfg_keys])
        gp = [Ysf[lab == str(lv)].ravel() for lv in levels]
        Fp, pp = stats.f_oneway(*gp)
        ge = [ensE_d[lab == str(lv)].ravel() for lv in levels]
        Fe, pe = stats.f_oneway(*ge)
        mp = {str(lv): float(Y[lab == str(lv)].mean()) for lv in levels}
        me = {str(lv): float(ens[lab == str(lv)].mean()) for lv in levels}
        per_fold_best = {f: max(({str(lv): float(ens[lab == str(lv), j].mean()) for lv in levels}).items(),
                                key=lambda kv: kv[1])[0]
                         for j, f in enumerate(folds)}
        axis_out[ax] = {
            "levels": [str(lv) for lv in levels],
            "per_seed_blocked": {"best_level": max(mp, key=mp.get), "level_means": mp,
                                 "spread": float(max(mp.values()) - min(mp.values())),
                                 "F": float(Fp), "p": float(pp)},
            "ensemble_blocked": {"best_level": max(me, key=me.get), "level_means": me,
                                 "spread": float(max(me.values()) - min(me.values())),
                                 "F": float(Fe), "p": float(pe)},
            "the_two_statistics_agree_on_best_level": max(mp, key=mp.get) == max(me, key=me.get),
            "best_level_by_fold_on_ensemble": per_fold_best,
            "ordering_flips_across_folds": len(set(per_fold_best.values())) > 1,
        }

    return {
        "carrier": label, "workspace": str(ws), "model_family": model, "record_layout": layout,
        "design": {"n_configs": a, "n_folds": b, "folds": folds, "n_seeds": c, "seeds": seeds,
                   "n_observations": n, "grid": " x ".join(
                       f"{ax}{{{','.join(sorted({str(json.loads(k)[ax]) for k in cfg_keys}, key=str))}}}"
                       for ax in axes)},
        "grand_mean": gm,
        "main_model_config_fold_seed_plus_interaction": main,
        "additivity_error": add_err,
        "routes_to_the_config_question": {
            "nothing_blocked": {"F": float(F_naive), "p": float(p_naive)},
            "seed_blocked_only": {"F": float(F_sb), "p": float(p_sb)},
            "seed_and_fold_blocked": {"F": main["config"]["F"], "p": main["config"]["p"]},
            "seed_x_fold_as_blocks": {"F": float(F_cfg_b), "p": p_cfg_b},
        },
        "effect_sizes": {
            "residual_sd": math.sqrt(MS_res),
            "SE_of_a_config_mean_pooled_over_folds": se_pool,
            "SE_of_a_config_mean_within_one_fold": se_fold,
            "estimated_true_between_config_sd": (math.sqrt(sigA2) if sigA2 > 0 else 0.0),
            "estimated_true_between_config_variance_raw": sigA2,
            "fold_swing": float(m_fold.max() - m_fold.min()),
            "seed_swing_pooled_main_effect": float(m_seed.max() - m_seed.min()),
            "seed_swing_within_each_fold": {f: float(max(v.values()) - min(v.values()))
                                            for f, v in seed_by_fold.items()},
            "config_swing_on_ensemble_within_fold_median": float(
                np.median([ens[:, j].max() - ens[:, j].min() for j in range(b)])),
        },
        "seed_direction": {
            "seed_mean_by_fold": seed_by_fold,
            "best_to_worst_by_fold": {f: [s for s, _ in sorted(v.items(), key=lambda kv: -kv[1])]
                                      for f, v in seed_by_fold.items()},
            "n_distinct_seed_orderings_across_folds": len({
                tuple(s for s, _ in sorted(v.items(), key=lambda kv: -kv[1]))
                for v in seed_by_fold.values()}),
            "worst_seed_census_by_fold": worst_census,
        },
        "winner_stability": {
            "per_fold_winner": winners,
            "n_distinct_winners": len({json.dumps(v["config"], sort_keys=True) for v in winners.values()}),
            "cross_fold_rank_agreement_spearman": rho,
            "rank_agreement_median": (float(np.median(list(rho.values()))) if rho else None),
            "median_gap_to_runner_up_in_SE": float(np.median(
                [v["gap_in_SE"] for v in winners.values() if v["gap_in_SE"] is not None])) if a > 1 else None,
        },
        "ensemble_statistic": ens_block,
        "axis_tests": axis_out,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", type=Path, default=ART)
    ap.add_argument("--out", type=Path, default=ART / "carrier_grid_statistics")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    per = {}
    for label, ws_name, model in CARRIERS:
        ws = args.artifacts / ws_name
        if not ws.is_dir():
            per[label] = {"carrier": label, "skipped": f"workspace {ws} absent"}
            continue
        per[label] = analyse(label, ws, model)
        d = per[label]
        if "skipped" in d:
            print(f"\n### {label}: skipped ({d['skipped']})")
            continue
        r = d["routes_to_the_config_question"]
        m = d["main_model_config_fold_seed_plus_interaction"]
        print(f"\n### {label} ({d['model_family']}) {d['design']['n_configs']} cfg x "
              f"{d['design']['n_folds']} folds x {d['design']['n_seeds']} seeds = {d['design']['n_observations']} obs"
              f"   [{d['record_layout']}]")
        print(f"    grid: {d['design']['grid']}")
        print(f"    shares: " + "  ".join(
            f"{k}={100*v['share']:.2f}%" for k, v in m.items() if isinstance(v, dict) and "share" in v))
        print(f"    config p by route: nothing={r['nothing_blocked']['p']:.4f} "
              f"seed={r['seed_blocked_only']['p']:.4f} "
              f"seed+fold={r['seed_and_fold_blocked']['p'] if r['seed_and_fold_blocked']['p'] is not None else float('nan'):.4f} "
              f"seedxfold={r['seed_x_fold_as_blocks']['p']:.4f}")
        e = d["effect_sizes"]
        print(f"    true between-config sd = {e['estimated_true_between_config_sd']:.6f} "
              f"(raw variance {e['estimated_true_between_config_variance_raw']:.3e}); residual sd = {e['residual_sd']:.5f}")
        print(f"    SE of a config mean: pooled {e['SE_of_a_config_mean_pooled_over_folds']:.5f}, "
              f"within one fold {e['SE_of_a_config_mean_within_one_fold']:.5f}")
        sd = d["seed_direction"]
        print(f"    seed swing: pooled main effect {e['seed_swing_pooled_main_effect']:.5f} vs within-fold "
              + ", ".join(f"{k}={v:.5f}" for k, v in e["seed_swing_within_each_fold"].items()))
        print(f"    distinct seed orderings across folds: {sd['n_distinct_seed_orderings_across_folds']}"
              f"   best->worst: {json.dumps(sd['best_to_worst_by_fold'])}")
        w = d["winner_stability"]
        print(f"    distinct winners: {w['n_distinct_winners']} of {d['design']['n_folds']} folds; "
              f"median gap to runner-up {w['median_gap_to_runner_up_in_SE'] if w['median_gap_to_runner_up_in_SE'] is not None else float('nan'):.2f} SE; "
              f"cross-fold rank Spearman median "
              f"{w['rank_agreement_median'] if w['rank_agreement_median'] is not None else float('nan'):+.4f}")
        print("    axes:")
        for ax, v in d["axis_tests"].items():
            if "constant_at" in v:
                print(f"      {ax:<18} CONSTANT at {v['constant_at']} (not searched)")
                continue
            agree = "agree" if v["the_two_statistics_agree_on_best_level"] else "DISAGREE"
            print(f"      {ax:<18} per-seed best={v['per_seed_blocked']['best_level']:<6} "
                  f"p={v['per_seed_blocked']['p']:.4f} | ensemble best={v['ensemble_blocked']['best_level']:<6} "
                  f"p={v['ensemble_blocked']['p']:.4f} | {agree} | per-fold "
                  f"{'FLIPS' if v['ordering_flips_across_folds'] else 'stable'} "
                  f"{json.dumps(v['best_level_by_fold_on_ensemble'])}")

    # cross-carrier summary
    rows = []
    for label, ws_name, model in CARRIERS:
        d = per.get(label, {})
        if "design" not in d:
            continue
        m = d["main_model_config_fold_seed_plus_interaction"]
        r = d["routes_to_the_config_question"]
        rows.append({
            "carrier": label, "model_family": model,
            "n_configs": d["design"]["n_configs"], "n_folds": d["design"]["n_folds"],
            "n_observations": d["design"]["n_observations"],
            "config_share_of_variance": m["config"]["share"],
            "config_F": m["config"]["F"], "config_p": m["config"]["p"],
            "config_p_nothing_blocked": r["nothing_blocked"]["p"],
            "config_p_seed_x_fold_blocks": r["seed_x_fold_as_blocks"]["p"],
            "config_x_fold_share": m["config_x_fold"]["share"],
            "fold_share": m["fold"]["share"], "seed_share": m["seed"]["share"],
            "residual_share": m["residual"]["share"],
            "estimated_true_between_config_sd": d["effect_sizes"]["estimated_true_between_config_sd"],
            "n_distinct_winners": d["winner_stability"]["n_distinct_winners"],
            "cross_fold_rank_agreement_median": d["winner_stability"]["rank_agreement_median"],
            "n_distinct_seed_orderings": d["seed_direction"]["n_distinct_seed_orderings_across_folds"],
            "seed_swing_pooled": d["effect_sizes"]["seed_swing_pooled_main_effect"],
            "ensemble_config_p": (d["ensemble_statistic"]["config"] or {}).get("p"),
        })

    result = {
        "question": ("Does the config effect, the seed effect and the stability of the per-fold winner "
                     "behave the same way in the LightGBM tree carriers as in the neural carriers, and "
                     "does the 14:57 entry's 0.555/0.677 vs 1.721/1.722 split show up as a structural "
                     "difference rather than a coincidence?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "read_only": True, "database_access": False, "weights_loaded": False,
        "repository_module_executed": False,
        "assertions": ["SS additive to 1e-9 relative", "every effect vector centred",
                       "residual SS reproduced by explicit least-squares RSS (both blocking schemes)",
                       "df bookkeeping consistent"],
        "excluded": {"M0R": "one config per fold, so there is no grid to decompose"},
        "cross_carrier_summary": rows,
        "per_carrier": per,
    }
    out = args.out / "all_carrier_grid_variance.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"\nwrote {out}")
    print("\n=== cross-carrier summary ===")
    hdr = ("carrier", "model", "cfg", "fld", "obs", "cfg%", "cfg p", "p none", "p sxf",
           "cfgxfold%", "fld%", "seed%", "winners", "rho", "seed orders")
    print("  ".join(f"{h:>11}" for h in hdr))
    for r_ in rows:
        print("  ".join(f"{v:>11}" for v in (
            r_["carrier"][:11], r_["model_family"][:11], r_["n_configs"], r_["n_folds"],
            r_["n_observations"], f"{100*r_['config_share_of_variance']:.2f}",
            "-" if r_["config_p"] is None else f"{r_['config_p']:.4f}",
            f"{r_['config_p_nothing_blocked']:.4f}", f"{r_['config_p_seed_x_fold_blocks']:.4f}",
            f"{100*r_['config_x_fold_share']:.2f}", f"{100*r_['fold_share']:.2f}",
            f"{100*r_['seed_share']:.2f}",
            f"{r_['n_distinct_winners']}/{r_['n_folds']}",
            "-" if r_["cross_fold_rank_agreement_median"] is None else f"{r_['cross_fold_rank_agreement_median']:+.3f}",
            r_["n_distinct_seed_orderings"])))


if __name__ == "__main__":
    main()
