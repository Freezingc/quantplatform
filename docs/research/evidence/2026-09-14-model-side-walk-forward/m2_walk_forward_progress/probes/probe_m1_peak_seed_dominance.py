#!/usr/bin/env python3
"""Read-only probe for §CN: on PEAK EPOCH, the seed effect is larger than the fold effect, and the fold
pattern is itself seed-dependent -- which removes the statistical footing of §CK's effect B.

WHY THIS PROBE EXISTS. §CK.2 published effect B: apply M1's fold ratios (F3/F2 = 0.8202,
F4/F3 = 0.8887) to M2's measured F2 epochs/run of 28.8056, giving 139.0 h remaining, below §CI's
published optimistic bound of 166.6 h. §CK.2 already said this must be cited as "an unexcluded branch,
never a prediction", because M2's own F1->F2 elasticity (2.5121x on peak-mean) is nearly twice M1's
(1.3110x), so transferring ratios across architectures has no evidence behind it. This probe finds a
SECOND, independent reason the transfer is unsound -- and this one is inside M1 itself, so it needs no
cross-architecture assumption.

WHAT IS MEASURED, and the four results

(1) config x fold interaction on peak epoch is NOT significant. F = 0.8872, p = 0.727, 17.18% of SS on
    78 df -- indistinguishable from noise once divided by its df. This is the same model structure
    §BN.1 used on validation RankIC, where config x fold scored F = 0.3894, p = 1.0000. So on BOTH
    dependent variables the fold pattern is not config-specific. That closes the candidate this pack
    proposed at the end of §CM: "does F3's bimodality correspond to some class of configs converging
    either very fast or very slowly?" -- it does not, because configs do not differ in their fold
    profiles. §CK.2's licence to read M1's four-fold pattern as a fold effect stands (the config set
    is byte-identical across folds), and this adds that even without that licence the interaction would
    not be detectable.

(2) The SEED main effect on peak epoch is 16.28% of SS with F = 32.79, p = 3.78e-13 -- larger than the
    fold main effect (7.17%, F = 9.63, p = 5.5e-6). In raw epochs: the seed swing is 8.352 (seed 01
    5.435, 02 10.361, 03 13.787) while the fold main-effect range is 6.309. So WHICH SEED a run used
    moves its peak epoch further than WHICH FOLD it ran on. Contrast validation RankIC, where §BN.1
    measured seed at only 0.8798% (F = 4.635, p = 0.0107) against fold at 74.53%: the ordering is
    INVERTED between the two dependent variables. Peak LOCATION is seed-driven; peak QUALITY is
    fold-driven. This is a clean dissociation and it matters for the paper: any statement of the form
    "runs on fold X converge after about N epochs" has seed-induced uncertainty of the same order as
    the fold effect it is describing.

(3) fold x seed interaction is 14.52% of SS, F = 13.03, p = 1.68e-12 -- MORE significant and larger
    than the fold main effect it qualifies. The fold profile is not a property of the fold alone:
    per-seed argmax is F4 for seed 20260801 and F2 for seeds 20260802 and 20260803, so the
    "rise-then-fall peaking at F2" shape §CK.2 published holds for 2 of 3 seeds and REVERSES for the
    third. Per-seed fold ratios, on the epochs/run basis effect B actually used:
        F2/F1: 0.8406 (seed 01) / 1.4381 (02) / 1.1399 (03)   spread 0.597
        F3/F2: 0.9725 / 0.6715 / 0.8943                        spread 0.301
        F4/F3: 1.4576 / 0.8126 / 0.6511                        spread 0.807
    Two of the three ratios are above 1.0 for seed 01 and below 1.0 for seed 03 -- the same fold pair
    moves in opposite directions depending on the seed.

(4) CONSEQUENCE FOR EFFECT B, which is a correction to §CK.5 rather than a new estimate. Applying
    effect B's own arithmetic per seed instead of pooled:
        seed 20260801 -> M2 F3 28.01, F4 40.83 -> 204.4 h (8.5 d)
        seed 20260802 -> M2 F3 19.34, F4 15.72 -> 113.8 h (4.7 d)
        seed 20260803 -> M2 F3 25.76, F4 16.77 -> 132.9 h (5.5 d)
        pooled (§CK's published figure)        -> 139.0 h (5.8 d)
    The per-seed spread is 113.8 to 204.4 h. The 204.4 h branch is ABOVE §CI's published optimistic
    bound of 166.6 h, and above §CK's own low end. So §CK.5's statement that effect B "gives 139.0 h,
    below §CI's optimistic bound" is not robust: it is the pooled average of three seed-specific
    estimates, one of which exceeds that bound. The honest revision is that the pooled 139.0 h is a
    point inside a seed-induced range whose upper end re-enters §CI's interval, so effect B no longer
    widens the interval downward on the strength of M1's fold pattern alone. Bootstrap CIs confirm the
    pooled ratios are themselves imprecise: F3/F2 = 0.8202 with 95% CI [0.7161, 0.9345] excludes 1.0,
    but F4/F3 = 0.8887 with CI [0.7699, 1.0264] COVERS 1.0, i.e. that half of effect B is not
    distinguishable from "flat".

WHAT THIS DOES AND DOES NOT CHANGE IN ALREADY-PUBLISHED CONTENT
  * §CI's four scenarios are untouched: they never used M1's fold ratios, and their inputs (measured
    rows via §CK/A15, M2's own epochs) are unaffected. §CK.5's combined interval low end must be
    re-read: the low end rested on effect B, which this probe shows is seed-fragile, so the defensible
    low end reverts to §CI's own optimistic scenario recomputed on measured rows -- 173.4 h -- and
    139.0 h survives only as "one seed's fold ratios, pooled-free, not robust".
  * §CK.2's four-fold epochs/run table (20.1235 / 23.2716 / 19.0864 / 16.9630) is arithmetically
    correct and stays; what changes is how much weight it can bear. It is a pooled mean over 3 seeds
    whose interaction with fold is highly significant, so its "non-monotone, peaking at F2" reading is
    a statement about the pooled data and not about every seed. §CK.2 did state the n=4 p-value
    caveat for its Spearman; it did not have the per-seed decomposition, which is what this probe adds.
  * §CM.4's order flip between mean and median (F3 vs F4) is untouched and independent of seeds.

Only checkpoint directory names are read, to obtain each run's peak epoch. No manifest values, no
checkpoint.pt, no weight or tensor, no parquet at all, no repository module imported or executed, no
database access, no network, no writes to any artifact directory, no signal to any process. M2
(pid 3588838) is still training; this probe touches only m1_full_v1, whose run finished long ago.
"""
from __future__ import annotations

import glob
import json
import re
import statistics as st
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "measurements" / "m1_peak_seed_dominance.json"

PATIENCE = 10
CAP = 100
# §CI's cost model constants, carried unchanged so effect B's arithmetic is identical to §CK.5's
K_GRID = 0.00219327
K_FINAL = 0.00020381721864733542
FINAL_UPLIFT = 1.45
GAPS_AND_OOS_H = 4 * 5.0 + 2.0
EXACT_ROWS = {"F3": 3650743, "F4": 3955048}   # §CK/A15, re-confirmed by §CL as exact
M2_F2_EPOCHS_PER_RUN = 28.8056                # §CI measured, re-confirmed by §CJ's cross-check
CI_OPTIMISTIC_PUBLISHED_H = 166.6
CK_EFFECT_B_POOLED_H = 139.0
CI_OPTIMISTIC_ON_MEASURED_ROWS_H = 173.4      # §CK.1's effect A row
BOOT_N = 20000
BOOT_SEED = 20260918


def load_peaks(ws: str, car: str) -> dict[tuple[str, str, str], int]:
    """{(config, fold, seed): peak_epoch} for grid runs, from directory names only."""
    run_re = re.compile(r"^(?P<run>.+)-epoch-(?P<epoch>\d+)$")
    runs: dict[str, list[int]] = defaultdict(list)
    for d in glob.glob(str(ART / ws / "checkpoints" / "*-epoch-*")):
        m = run_re.match(Path(d).name)
        if m:
            runs[m.group("run")].append(int(m.group("epoch")))
    cell: dict[tuple[str, str, str], int] = {}
    for r, e in runs.items():
        g = re.match(rf"^{car}-(f\d)-(c\d+)-(\d+)-seed-(\d+)$", r)
        if g:
            fold, cfg, tag, seed = g.group(1), g.group(2), g.group(3), g.group(4)
            assert tag == seed, f"run id tag {tag} != seed {seed} for {r}"
            cell[(cfg, fold, seed)] = max(e)
    return cell


def anova(Y: np.ndarray, factors: dict[str, np.ndarray], terms: list[tuple[str, list[str]]]) -> dict:
    """Type-I-free orthogonal decomposition by cell means, matching §BN.1's structure.

    Every term is the SS of the joint grouping minus the SS of its own main effects, which is exact
    for a balanced design (this one is: 27 configs x 4 folds x 3 seeds, no missing cell, verified by
    an assertion in main). df are computed from group counts, residual df by subtraction, and the
    additivity of all SS against SST is asserted to machine precision rather than assumed.
    """
    gm = Y.mean()
    SST = float(((Y - gm) ** 2).sum())
    dfT = len(Y) - 1

    def ss_joint(names: list[str]) -> tuple[float, int]:
        cols = [factors[n] for n in names]
        labels = np.array(["|".join(str(c[i]) for c in cols) for i in range(len(Y))])
        # sorted(), not set(). Floating-point addition is order-dependent, and Python's set iteration
        # order for strings varies between processes (hash randomisation), so iterating a set made the
        # recorded sums differ in their last few digits from run to run -- enough to fail this pack's
        # byte-identical-reproducibility convention. Same defect class as §CM.6(iii)'s `uptime -s`,
        # caught the same way: re-running and diffing. Sorting makes the summation order deterministic.
        levels = sorted(set(labels.tolist()))
        out = 0.0
        for lv in levels:
            m = labels == lv
            out += float(m.sum()) * (Y[m].mean() - gm) ** 2
        return out, len(levels) - 1

    rows: dict[str, dict[str, float]] = {}
    used_df = 0
    used_ss = 0.0
    main_ss: dict[str, float] = {}
    main_df: dict[str, int] = {}
    for name, comps in terms:
        if len(comps) == 1:
            ss, df = ss_joint(comps)
            main_ss[name] = ss
            main_df[name] = df
        else:
            ss_joint_all, df_joint = ss_joint(comps)
            ss = ss_joint_all - sum(main_ss[c] for c in comps)
            df = df_joint - sum(main_df[c] for c in comps)
        rows[name] = {"ss": ss, "df": df, "pct_of_SST": round(100 * ss / SST, 4)}
        used_df += df
        used_ss += ss
    df_res = dfT - used_df
    ss_res = SST - used_ss
    ms_res = ss_res / df_res
    for name, r in rows.items():
        r["ms"] = r["ss"] / r["df"]
        r["F"] = round(r["ms"] / ms_res, 4)
        r["p"] = float(stats.f.sf(r["F"], r["df"], df_res))
    residual = {"ss": ss_res, "df": df_res, "pct_of_SST": round(100 * ss_res / SST, 4),
                "ms": ms_res}
    return {"terms": rows, "residual": residual, "SST": SST,
            "additivity_error": abs(SST - (used_ss + ss_res)),
            "df_total": dfT, "df_accounted": used_df + df_res}


def validate_against_BN1() -> dict:
    """The instrument must reproduce a decomposition this pack has ALREADY published.

    §BN.1 ran this exact model structure (config + fold + seed + config x fold) on M1's per-seed
    validation RankIC and published fold 74.5273% / F 261.744, config 1.3987% / F 0.5668 / p 0.9566,
    seed 0.8798% / F 4.6350 / p 0.0107, config x fold 2.8831% / F 0.3894 / p 1.0000, residual
    20.3110%, with grand mean 0.076318. Re-running the same code path on that dependent variable and
    getting those numbers back to four decimals is far stronger evidence that the ANOVA is right than
    any synthetic control, because the target is a published answer computed by a different probe
    (probe_m1_fourfold_variance_decomposition.py) rather than a value this probe expects.
    """
    cell: dict[tuple[str, str, str], float] = {}
    for f in ("F1", "F2", "F3", "F4"):
        for rec in sorted((ART / "m1_full_v1" / "search" / f).glob("c*.json")):
            d = json.load(open(rec, encoding="utf-8"))
            cfg = f"c{d['config_index']}"
            for seed, ic in d["result"]["seed_validation_rank_ic"].items():
                cell[(cfg, f.lower(), seed)] = float(ic)
    cfgs = sorted({k[0] for k in cell}, key=lambda c: int(c[1:]))
    folds = ["f1", "f2", "f3", "f4"]
    seeds = sorted({k[2] for k in cell})
    assert len(cell) == len(cfgs) * 4 * len(seeds), "RankIC design not balanced"
    Y = np.array([cell[(c, f, sd)] for c in cfgs for f in folds for sd in seeds])
    C = np.array([c for c in cfgs for _ in folds for _ in seeds])
    F = np.array([f for _ in cfgs for f in folds for _ in seeds])
    S = np.array([sd for _ in cfgs for _ in folds for sd in seeds])
    mdl = anova(Y, {"config": C, "fold": F, "seed": S},
                [("fold", ["fold"]), ("config", ["config"]), ("seed", ["seed"]),
                 ("config_x_fold", ["config", "fold"])])
    published = {
        "fold": (74.5273, 261.744), "config": (1.3987, 0.5668),
        "seed": (0.8798, 4.6350), "config_x_fold": (2.8831, 0.3894),
        "residual": (20.3110, None),
    }
    rows = {}
    all_ok = True
    for name, (pct, fval) in published.items():
        got_pct = (mdl["residual"]["pct_of_SST"] if name == "residual"
                   else mdl["terms"][name]["pct_of_SST"])
        got_f = None if fval is None else mdl["terms"][name]["F"]
        ok = abs(got_pct - pct) < 1e-3 and (fval is None or abs(got_f - fval) < 1e-3)
        all_ok &= ok
        rows[name] = {"BN1_published_pct": pct, "recomputed_pct": round(got_pct, 4),
                      "BN1_published_F": fval, "recomputed_F": got_f, "matches": ok}
    return {"per_term": rows, "grand_mean_published": 0.076318,
            "grand_mean_recomputed": round(float(Y.mean()), 6),
            "grand_mean_matches": abs(float(Y.mean()) - 0.076318) < 1e-6,
            "ALL_MATCH": all_ok and abs(float(Y.mean()) - 0.076318) < 1e-6,
            "reading": ("reproducing a decomposition another probe already published, on a different "
                        "dependent variable, to four decimals -- this is the primary validation of the "
                        "ANOVA, and it is a check against an external answer rather than an expectation "
                        "baked into this probe")}


def permutation_calibration(Y: np.ndarray, factors: dict, terms, n_perm: int = 300) -> dict:
    """Calibration of the F statistic under the null, instead of a single flaky permutation draw.

    The first version of this probe used ONE permutation and asserted that all four terms had
    p > 0.01. That control was itself defective: under the null, four independent terms each have a 1%
    chance of landing below 0.01, so the control fails spuriously about 4% of the time -- it is flaky,
    and a flaky control is worse than none because it teaches the reader to ignore failures. It failed
    on exactly that: seed came out p = 0.0037 on the first draw. The correct check is a calibration:
    permute many times and verify the null distribution behaves as theory says, namely E[F] = 1 for
    every term and a false-positive rate at alpha = 0.05 near 0.05. Both are stable to measure.
    """
    rng = np.random.default_rng(BOOT_SEED)
    Fs = defaultdict(list)
    for _ in range(n_perm):
        Yp = Y.copy()
        rng.shuffle(Yp)
        m = anova(Yp, factors, terms)
        for name, r in m["terms"].items():
            Fs[name].append(r["F"])
    out = {}
    for name, vals in Fs.items():
        mean_f = float(np.mean(vals))
        # df for this term is fixed by the design; take it from the last permutation's model
        dfn = m["terms"][name]["df"]
        dfd = m["residual"]["df"]
        crit = float(stats.f.ppf(0.95, dfn, dfd))
        fpr = float(np.mean([v > crit for v in vals]))
        out[name] = {
            "n_permutations": len(vals),
            "null_mean_F": round(mean_f, 4),
            "null_mean_F_within_15pct_of_1": bool(abs(mean_f - 1) < 0.15),
            "false_positive_rate_at_alpha_0.05": round(fpr, 4),
            "fpr_within_expected_band": bool(0.0 <= fpr <= 0.12),
            "critical_F_at_alpha_0.05": round(crit, 4),
        }
    return {"per_term": out,
            "all_terms_calibrated": all(
                v["null_mean_F_within_15pct_of_1"] and v["fpr_within_expected_band"]
                for v in out.values())}


def boot_ratio(va: np.ndarray, vb: np.ndarray, rng) -> tuple[float, float, float]:
    idx = np.arange(len(va))
    out = np.empty(BOOT_N)
    for i in range(BOOT_N):
        k = rng.choice(idx, size=len(idx), replace=True)
        out[i] = vb[k].mean() / va[k].mean()
    lo, hi = np.percentile(out, [2.5, 97.5])
    return float(vb.mean() / va.mean()), float(lo), float(hi)


def remaining_h(e3: float, e4: float) -> float:
    return (K_GRID * e3 * EXACT_ROWS["F3"] / 3600
            + K_FINAL * e3 * EXACT_ROWS["F3"] * FINAL_UPLIFT / 3600
            + K_GRID * e4 * EXACT_ROWS["F4"] / 3600
            + K_FINAL * e4 * EXACT_ROWS["F4"] * FINAL_UPLIFT / 3600
            + GAPS_AND_OOS_H)


def main() -> None:
    cell = load_peaks("m1_full_v1", "m1")
    cfgs = sorted({k[0] for k in cell}, key=lambda c: int(c[1:]))
    folds = ["f1", "f2", "f3", "f4"]
    seeds = sorted({k[2] for k in cell})
    assert len(cell) == len(cfgs) * len(folds) * len(seeds), "design is not balanced"

    Y = np.array([cell[(c, f, s)] for c in cfgs for f in folds for s in seeds], dtype=float)
    C = np.array([c for c in cfgs for _ in folds for _ in seeds])
    F = np.array([f for _ in cfgs for f in folds for _ in seeds])
    S = np.array([s for _ in cfgs for _ in folds for s in seeds])
    factors = {"config": C, "fold": F, "seed": S}

    model_a = anova(Y, factors, [("fold", ["fold"]), ("config", ["config"]), ("seed", ["seed"]),
                                 ("config_x_fold", ["config", "fold"])])
    model_b = anova(Y, factors, [("fold", ["fold"]), ("seed", ["seed"]),
                                 ("fold_x_seed", ["fold", "seed"]), ("config", ["config"]),
                                 ("config_x_fold", ["config", "fold"])])

    def mean_by(level_arr, value):
        out = {}
        for lv in sorted(set(level_arr)):
            out[lv] = float(Y[level_arr == lv].mean())
        return out

    fold_means = mean_by(F, Y)
    seed_means = mean_by(S, Y)
    seed_swing = max(seed_means.values()) - min(seed_means.values())
    fold_range = max(fold_means.values()) - min(fold_means.values())

    per_seed_fold_mean = {s: {f: float(Y[(F == f) & (S == s)].mean()) for f in folds} for s in seeds}
    per_seed_argmax = {s: max(per_seed_fold_mean[s], key=per_seed_fold_mean[s].get) for s in seeds}

    # fold ratios, per seed and pooled, on the epochs/run basis effect B used
    ratios: dict[str, dict] = {}
    rng = np.random.default_rng(BOOT_SEED)
    for fa, fb in (("f1", "f2"), ("f2", "f3"), ("f3", "f4")):
        per_seed = {}
        for s in seeds:
            a = np.mean([cell[(c, fa, s)] + PATIENCE for c in cfgs])
            b = np.mean([cell[(c, fb, s)] + PATIENCE for c in cfgs])
            per_seed[s] = round(float(b / a), 4)
        va = np.array([cell[(c, fa, s)] + PATIENCE for c in cfgs for s in seeds], dtype=float)
        vb = np.array([cell[(c, fb, s)] + PATIENCE for c in cfgs for s in seeds], dtype=float)
        r, lo, hi = boot_ratio(va, vb, rng)
        ratios[f"{fb}/{fa}"] = {
            "per_seed": per_seed, "spread_across_seeds": round(max(per_seed.values())
                                                              - min(per_seed.values()), 4),
            "pooled": round(r, 4), "pooled_bootstrap_95ci": [round(lo, 4), round(hi, 4)],
            "pooled_ci_covers_1": bool(lo <= 1.0 <= hi),
            "per_seed_direction": {s: ("above 1" if v > 1 else "below 1" if v < 1 else "exactly 1")
                                   for s, v in per_seed.items()},
        }

    # effect B applied per seed versus pooled
    effect_b = {}
    for s in seeds:
        r32 = ratios["f3/f2"]["per_seed"][s]
        r43 = ratios["f4/f3"]["per_seed"][s]
        e3 = M2_F2_EPOCHS_PER_RUN * r32
        e4 = e3 * r43
        h = remaining_h(e3, e4)
        effect_b[f"seed_{s}"] = {"F3_over_F2": r32, "F4_over_F3": r43,
                                 "M2_F3_epochs_per_run": round(e3, 2),
                                 "M2_F4_epochs_per_run": round(e4, 2),
                                 "total_remaining_h": round(h, 1), "days": round(h / 24, 1),
                                 "above_CI_optimistic_published": h > CI_OPTIMISTIC_PUBLISHED_H}
    r32p, r43p = ratios["f3/f2"]["pooled"], ratios["f4/f3"]["pooled"]
    e3p = M2_F2_EPOCHS_PER_RUN * r32p
    e4p = e3p * r43p
    hp = remaining_h(e3p, e4p)
    effect_b["POOLED_as_published_in_CK5"] = {
        "F3_over_F2": r32p, "F4_over_F3": r43p, "M2_F3_epochs_per_run": round(e3p, 2),
        "M2_F4_epochs_per_run": round(e4p, 2), "total_remaining_h": round(hp, 1),
        "days": round(hp / 24, 1), "reproduces_CK5_139_0": abs(hp - CK_EFFECT_B_POOLED_H) < 0.6,
        "above_CI_optimistic_published": hp > CI_OPTIMISTIC_PUBLISHED_H}
    hs = [effect_b[f"seed_{s}"]["total_remaining_h"] for s in seeds]

    doc = {
        "question": ("on peak epoch, how much of the variation is fold, how much is seed, does the fold "
                     "pattern depend on the seed, and does §CK.5's effect B survive being computed per "
                     "seed instead of pooled?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "design": {"carrier": "M1 (RESIDUAL_MLP)", "workspace": "m1_full_v1",
                   "n_configs": len(cfgs), "n_folds": len(folds), "n_seeds": len(seeds),
                   "n_observations": len(Y), "balanced_no_missing_cell": True,
                   "dependent_variable": "peak epoch = max saved checkpoint epoch (§CJ.2 consequence 1)"},
        "model_A_same_structure_as_BN1": model_a,
        "model_B_adds_fold_x_seed": model_b,
        "BN1_reference_on_validation_rank_ic": {
            "fold_pct": 74.5273, "fold_F": 261.744, "fold_p": 2.6e-71,
            "config_pct": 1.3987, "config_F": 0.5668, "config_p": 0.9566,
            "seed_pct": 0.8798, "seed_F": 4.6350, "seed_p": 0.0107,
            "config_x_fold_pct": 2.8831, "config_x_fold_F": 0.3894, "config_x_fold_p": 1.0000,
            "note": "quoted from §BN.1 so the two dependent variables can be compared line by line"},
        "dissociation": {
            "seed_pct_on_peak": model_a["terms"]["seed"]["pct_of_SST"],
            "fold_pct_on_peak": model_a["terms"]["fold"]["pct_of_SST"],
            "seed_pct_on_rank_ic": 0.8798, "fold_pct_on_rank_ic": 74.5273,
            "seed_swing_epochs": round(seed_swing, 4),
            "fold_main_effect_range_epochs": round(fold_range, 4),
            "seed_swing_exceeds_fold_range": bool(seed_swing > fold_range),
            "ratio": round(seed_swing / fold_range, 4),
            "fold_means_epochs_peak": {k: round(v, 4) for k, v in fold_means.items()},
            "seed_means_epochs_peak": {k: round(v, 4) for k, v in seed_means.items()},
            "reading": ("peak LOCATION is seed-driven and peak QUALITY is fold-driven: the ordering "
                        "of these two sources is inverted between the two dependent variables. Any "
                        "sentence of the form 'runs on fold X converge after about N epochs' carries "
                        "seed-induced uncertainty of the same order as the fold effect it describes."),
        },
        "fold_x_seed_interaction": {
            "pct_of_SST": model_b["terms"]["fold_x_seed"]["pct_of_SST"],
            "F": model_b["terms"]["fold_x_seed"]["F"], "p": model_b["terms"]["fold_x_seed"]["p"],
            "fold_main_effect_pct": model_b["terms"]["fold"]["pct_of_SST"],
            "interaction_exceeds_main_effect": bool(
                model_b["terms"]["fold_x_seed"]["pct_of_SST"]
                > model_b["terms"]["fold"]["pct_of_SST"]),
            "per_seed_fold_means": {s: {f: round(v, 3) for f, v in d.items()}
                                    for s, d in per_seed_fold_mean.items()},
            "per_seed_argmax_fold": per_seed_argmax,
            "n_seeds_whose_argmax_is_F2": sum(1 for v in per_seed_argmax.values() if v == "f2"),
            "shape_per_seed": {},
            "reading": ("the fold profile is not a property of the fold alone. §CK.2's 'rise then fall, "
                        "peaking at F2' holds for 2 of 3 seeds and reverses for the third, whose argmax "
                        "is F4."),
        },
        "fold_ratios": ratios,
        "effect_B_per_seed_versus_pooled": {
            "per_seed_and_pooled": effect_b,
            "per_seed_range_h": [min(hs), max(hs)],
            "per_seed_spread_h": round(max(hs) - min(hs), 1),
            "n_seeds_above_CI_optimistic": sum(
                1 for s in seeds if effect_b[f"seed_{s}"]["above_CI_optimistic_published"]),
            "correction_to_CK5": (
                "§CK.5 said effect B gives 139.0 h, below §CI's published optimistic bound of 166.6 h. "
                "That is the POOLED figure; computed per seed the same arithmetic gives "
                f"{min(hs):.1f} to {max(hs):.1f} h, and the seed-01 branch "
                f"({effect_b['seed_20260801']['total_remaining_h']} h) is ABOVE 166.6 h. So 'effect B "
                "lies below §CI's optimistic bound' is not robust -- it holds for 2 of 3 seeds. The "
                f"defensible low end reverts to §CI's optimistic scenario on measured rows "
                f"({CI_OPTIMISTIC_ON_MEASURED_ROWS_H} h), with 139.0 h surviving only as one pooled "
                "point inside a seed-induced range."),
            "half_of_effect_B_not_distinguishable_from_flat": {
                "F3_over_F2_pooled_ci": ratios["f3/f2"]["pooled_bootstrap_95ci"],
                "F3_over_F2_excludes_1": not ratios["f3/f2"]["pooled_ci_covers_1"],
                "F4_over_F3_pooled_ci": ratios["f4/f3"]["pooled_bootstrap_95ci"],
                "F4_over_F3_covers_1": ratios["f4/f3"]["pooled_ci_covers_1"],
                "reading": ("the F4/F3 half of effect B has a bootstrap CI covering 1.0, so that half "
                            "is not distinguishable from 'flat', which is §CI's own optimistic scenario"),
            },
        },
        "config_x_fold_interaction_not_significant": {
            "pct_of_SST": model_a["terms"]["config_x_fold"]["pct_of_SST"],
            "df": model_a["terms"]["config_x_fold"]["df"],
            "F": model_a["terms"]["config_x_fold"]["F"],
            "p": model_a["terms"]["config_x_fold"]["p"],
            "config_main_p": model_a["terms"]["config"]["p"],
            "same_in_BN1_on_rank_ic": {"pct": 2.8831, "F": 0.3894, "p": 1.0000},
            "candidate_closed": ("§CM's closing suggestion -- that F3's bimodality might correspond to a "
                                 "class of configs converging either very fast or very slowly -- is "
                                 "REJECTED: configs do not differ in their fold profiles on peak epoch "
                                 "(p = 0.727), just as they do not on validation RankIC (§BN.1, "
                                 "p = 1.0000)."),
        },
        "what_this_does_not_change": [
            "§CI's four scenarios: they never used M1's fold ratios, and their inputs are unaffected",
            "§CK.2's four-fold epochs/run table: arithmetically correct, but it is a pooled mean over "
            "3 seeds whose fold interaction is highly significant, so 'non-monotone, peaking at F2' is "
            "a statement about pooled data, not about every seed",
            "§CM.4's mean-versus-median order flip between F3 and F4: independent of seeds",
            "§CJ's cap scan: peak epochs are what they are; 0 truncated is unaffected",
            "§CL's panel identity and A15's exact row counts: unrelated to peak epochs",
        ],
        "negative_controls": {},
    }

    shape = {}
    for s in seeds:
        v = [per_seed_fold_mean[s][f] for f in folds]
        shape[s] = ("rise_then_fall_peak_F2" if v[1] > v[0] and v[1] > v[2] and v[2] > v[3]
                    else "monotone_decline" if all(v[i] > v[i + 1] for i in range(3))
                    else "monotone_rise" if all(v[i] < v[i + 1] for i in range(3))
                    else "other")
    doc["fold_x_seed_interaction"]["shape_per_seed"] = shape

    # --- validation and calibration
    bn1 = validate_against_BN1()
    calib = permutation_calibration(
        Y, {"config": C, "fold": F, "seed": S},
        [("fold", ["fold"]), ("config", ["config"]), ("seed", ["seed"]),
         ("config_x_fold", ["config", "fold"])])
    doc["validation_and_controls"] = {
        "primary_validation_reproduces_BN1": bn1,
        "permutation_calibration": calib,
        "why_the_first_control_was_replaced": (
            "the first version permuted the response ONCE and asserted all four terms had p > 0.01. "
            "Under the null each term has a 1% chance of falling below 0.01, so that control fails "
            "spuriously about 4% of the time -- it is flaky, and a flaky control is worse than none "
            "because it trains the reader to ignore failures. It did fail, on seed at p = 0.0037. "
            "Replacing a single draw with a 300-draw calibration (null mean F must be within 15% of "
            "the theoretical 1.0, and the false-positive rate at alpha = 0.05 must sit in 0.00-0.12) "
            "measures the same property stably."),
        "internal_consistency": {
            "additivity_errors": {"model_A": model_a["additivity_error"],
                                  "model_B": model_b["additivity_error"]},
            "df_accounted_equals_total": (model_a["df_accounted"] == model_a["df_total"]
                                         and model_b["df_accounted"] == model_b["df_total"]),
            "design_balanced_no_missing_cell": len(cell) == len(cfgs) * len(folds) * len(seeds),
        },
        "effect_B_pooled_reproduces_CK5": effect_b["POOLED_as_published_in_CK5"][
            "reproduces_CK5_139_0"],
        "ALL_VALIDATIONS_PASS": bool(
            bn1["ALL_MATCH"] and calib["all_terms_calibrated"]
            and model_a["additivity_error"] < 1e-6 and model_b["additivity_error"] < 1e-6
            and model_a["df_accounted"] == model_a["df_total"]
            and model_b["df_accounted"] == model_b["df_total"]
            and effect_b["POOLED_as_published_in_CK5"]["reproduces_CK5_139_0"]),
    }
    doc.update({"read_only": True, "database_access": False, "weights_loaded": False,
                "checkpoint_pt_opened": False, "parquet_read": False, "training_performed": False,
                "repository_module_imported_or_executed": False, "processes_signalled": []})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

    print(f"design: {len(cfgs)} configs x {len(folds)} folds x {len(seeds)} seeds = {len(Y)} obs")
    for label, mdl in (("A (BN.1 structure)", model_a), ("B (+fold x seed)", model_b)):
        print(f"\n=== model {label} on PEAK EPOCH ===")
        print(f"  {'source':16s} {'SS%':>8s} {'df':>4s} {'F':>9s} {'p':>11s}")
        for name, r in mdl["terms"].items():
            print(f"  {name:16s} {r['pct_of_SST']:>7.4f}% {r['df']:>4d} {r['F']:>9.4f} "
                  f"{r['p']:>11.3e}")
        print(f"  {'residual':16s} {mdl['residual']['pct_of_SST']:>7.4f}% {mdl['residual']['df']:>4d}")
        print(f"  additivity error {mdl['additivity_error']:.3e}, df {mdl['df_accounted']}"
              f"/{mdl['df_total']}")
    d = doc["dissociation"]
    print(f"\n=== dissociation ===")
    print(f"  peak epoch : seed {d['seed_pct_on_peak']}% vs fold {d['fold_pct_on_peak']}%")
    print(f"  rank IC    : seed {d['seed_pct_on_rank_ic']}% vs fold {d['fold_pct_on_rank_ic']}%")
    print(f"  seed swing {d['seed_swing_epochs']} epochs vs fold range "
          f"{d['fold_main_effect_range_epochs']} -> ratio {d['ratio']}")
    print(f"  per-seed argmax fold: {per_seed_argmax}")
    print(f"  per-seed shape: {shape}")
    print(f"\n=== effect B per seed vs pooled ===")
    for k, v in effect_b.items():
        print(f"  {k:28s} F3/F2={v['F3_over_F2']:.4f} F4/F3={v['F4_over_F3']:.4f} -> "
              f"{v['total_remaining_h']:>6.1f} h  above CI optimistic 166.6: "
              f"{v['above_CI_optimistic_published']}")
    print(f"  per-seed range {min(hs):.1f}-{max(hs):.1f} h, spread {max(hs)-min(hs):.1f} h")
    print(f"  F3/F2 pooled CI {ratios['f3/f2']['pooled_bootstrap_95ci']} excludes 1: "
          f"{not ratios['f3/f2']['pooled_ci_covers_1']}")
    print(f"  F4/F3 pooled CI {ratios['f4/f3']['pooled_bootstrap_95ci']} covers 1: "
          f"{ratios['f4/f3']['pooled_ci_covers_1']}")
    print(f"\n=== config x fold interaction: F={model_a['terms']['config_x_fold']['F']} "
          f"p={model_a['terms']['config_x_fold']['p']:.4f} -> "
          f"{'NOT significant' if model_a['terms']['config_x_fold']['p']>0.05 else 'significant'}")
    v = doc["validation_and_controls"]
    print("\n=== primary validation: reproduce §BN.1's published decomposition on RankIC ===")
    print(f"  grand mean {v['primary_validation_reproduces_BN1']['grand_mean_recomputed']} "
          f"vs published {v['primary_validation_reproduces_BN1']['grand_mean_published']}")
    for name, r in v["primary_validation_reproduces_BN1"]["per_term"].items():
        print(f"  {name:14s} pct {r['recomputed_pct']:>8.4f} vs {r['BN1_published_pct']:>8.4f} | "
              f"F {str(r['recomputed_F']):>9s} vs {str(r['BN1_published_F']):>9s} "
              f"{'MATCH' if r['matches'] else '*** MISMATCH ***'}")
    print(f"  ALL_MATCH: {v['primary_validation_reproduces_BN1']['ALL_MATCH']}")
    print("\n=== permutation calibration (300 draws, null mean F should be ~1) ===")
    for name, r in v["permutation_calibration"]["per_term"].items():
        print(f"  {name:14s} null mean F {r['null_mean_F']:>7.4f} "
              f"({'ok' if r['null_mean_F_within_15pct_of_1'] else '*** OFF ***'}) | "
              f"FPR@0.05 {r['false_positive_rate_at_alpha_0.05']:.4f} "
              f"({'ok' if r['fpr_within_expected_band'] else '*** OFF ***'})")
    print(f"  all terms calibrated: {v['permutation_calibration']['all_terms_calibrated']}")
    print(f"  effect B pooled reproduces §CK.5's 139.0: {v['effect_B_pooled_reproduces_CK5']}")
    print(f"  ALL_VALIDATIONS_PASS: {v['ALL_VALIDATIONS_PASS']}")
    assert v["ALL_VALIDATIONS_PASS"], "a validation or control failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
