#!/usr/bin/env python3
"""Read-only probe for §CP: the seed effect on peak epoch is overwhelmingly CONFIG-INDEPENDENT, which
constrains the mechanism far more tightly than §CO.5's underpowered test could.

WHY §CO.5'S TEST HAD NO POWER, AND WHAT TO USE INSTEAD
§CO.5 tried to separate the two channels by which seed reaches a run -- A, the date order within an
epoch, and B, the initial weights -- using a binary indicator (do two configs produce the IDENTICAL
ordering of three seeds?). With only 6 possible orderings that indicator throws away almost all the
information, and the test came out p = 0.814: no power. The correct statistic is already available in
the same design: the config x seed INTERACTION in the ANOVA. Channel A is identical for every config
within a fold, so it can produce a seed main effect but NO config x seed interaction. Channel B is
config-dependent (different parameter counts consume different amounts of the seeded RNG stream), so it
must show up as config x seed interaction. The two channels therefore make opposite predictions about
one term, measured on 52 df rather than a 6-outcome indicator.

RESULT
config x seed: 9.2441% of SST, 52 df, F = 0.9441, p = 0.585. The F is BELOW 1, and the observed
interaction SS (2,161) is 5.6% BELOW the pure-noise expectation (52 x MS_residual = 2,289) -- the
numerator chi-square is 49.1 against its null mean of 52, i.e. 0.29 sd below, entirely ordinary.
The 95% upper confidence bound on the non-centrality is lambda <= 20.01, giving interaction SS <= 881,
i.e. <= 3.77% of SST, i.e. an rms contribution of <= 1.65 epochs per observation.
Against that, the SEED MAIN EFFECT is 16.2848% of SST (F = 43.24, p = 1.1e-15) with a swing of 8.352
epochs (§CN.3). So at least (16.28 - 3.77)/16.28 = 76.8% of the seed variance in peak epoch is shared
across all 27 configs, and no config-specific component larger than 3.77% of SST (rms 1.65 epochs) can
be hiding.

POWER, so "not significant" is not doing the work
With df1 = 52, df2 = 156 and alpha = 0.05 the critical F is 1.4260. This design has 50% power at an
interaction of 4.27% of SST (rms 1.76 epochs), 80% at 7.04% (rms 2.25) and 95% at 10.05% (rms 2.69).
The seed main effect is 16.28%, so an interaction anywhere near the size of the main effect would have
been detected with power essentially 1. The null is therefore informative, not an absence of evidence.

CORROBORATION: the pattern is exactly channel A's signature, on two terms at once
Channel A depends on (seed, that fold's date set): DateBalancedSampler sorts dates by
sha256(f"{seed}|{date.isoformat()}") (sampler.py:27-32) and rotates by epoch % len(ordered)
(:33-34), constructed as DateBalancedSampler(tuple(batches), seed=seed) at torch_loop.py:43, where
`batches` are that fold's training dates. So A predicts BOTH a seed main effect AND a fold x seed
interaction (because the date set differs by fold), while predicting NO config x seed interaction.
Observed: seed 16.2848% (F = 43.24), fold x seed 14.5190% (F = 12.85, p = 9.0e-12), config x seed
F = 0.94. All three match.

INTERNAL CROSS-CHECK against §CN, exact to four decimals
§CN's model B (fold, seed, fold x seed, config, config x fold) reported residual 38.6182% on 208 df.
This probe's model adds config x seed and reports residual 29.3741% on 156 df. The differences are
38.6182 - 29.3741 = 9.2441% and 208 - 156 = 52 -- exactly the config x seed term's SS share and df.
And every term the two models share reproduces §CN's published value exactly: fold 7.1718%, seed
16.2848%, fold x seed 14.5190%, config 6.2222%, config x fold 17.1840%. So this is not a different
decomposition; it is §CN's model with one more term partitioned out of its residual.

A STRUCTURAL CAVEAT THAT MUST TRAVEL WITH EVERY NUMBER HERE
The design has exactly one observation per (config, fold, seed) cell, so it is fully saturated: the
residual IS the config x fold x seed three-way interaction (26 x 3 x 2 = 156 df, matching df_res), and
there is no pure error term. Every F is therefore measured RELATIVE TO the three-way term, not to
within-cell replication. That is legitimate here precisely because the three-way term behaves like
noise (config x fold F = 1.17, config x seed F = 0.94, both ~1), but it means these F values bound
effects relative to three-way variation and cannot be read as bounds relative to measurement error.

WHAT THIS DOES AND DOES NOT ESTABLISH ABOUT THE MECHANISM
Established: the seed effect on peak epoch is config-independent to within 3.77% of SST. Any proposed
mechanism must explain a seed effect that acts nearly identically on all 27 configs -- which rules out
any mechanism whose strength or direction depends on the config.
NOT established: that the sampler is the mechanism. This probe measures variance structure, never runs
the sampler, never constructs a model, and never counts RNG draws. Channel B is not excluded by this
test either, for a reason worth stating: configs that share an architecture prefix consume the same
initial segment of the seeded RNG stream, so weight init is only PARTLY config-dependent -- two configs
differing only in num_blocks draw identical values for their shared first block. So channel B could
contribute config-independent variance too. What the test does establish is that whatever the mechanism,
it must be predominantly config-independent, and among the two channels only the sampler's date order is
config-independent BY CONSTRUCTION. That is a constraint on the answer, not the answer. Per ledger B1
no unverified mechanism is asserted.

CONTROLS
(i) permutation: shuffling the response must drive every F to ~1 and every p to ~1, averaged over 300
    draws (a single draw is flaky, the defect §CN.6(ii) records).
(ii) injection: adding a synthetic config x seed interaction of a KNOWN size (scaled to 10% of SST,
    which the power table says is detectable with 95% power) must make that term significant while
    leaving the others near their null. Without this, "F = 0.94" could not be distinguished from an
    instrument that reports ~1 for everything -- the exact failure §CO.6 records for Kendall's W.
(iii) reproduction: §CN's published shares for all five shared terms must come back exactly. The
    residual is deliberately EXCLUDED from that equality check -- §CN's model lacks the config x seed
    term so its residual must exceed this one by exactly that term -- and the check that replaces it
    asserts the residual difference equals the new term's SS share and df.
(iv) all injection comparisons are in ABSOLUTE SS units, because injecting variance enlarges SST and
    every share-of-SST figure then falls by dilution even when the underlying SS is unchanged.
(v) additivity and df accounting must be exact.

Only checkpoint directory names are read. No parquet (not even footers), no checkpoint.pt, no weight or
tensor, no repository module imported or executed (sampler.py / torch_loop.py / cross_section_search.py
/ seeds.py are cited as text), no database, no network, no writes to any artifact directory, no signal
to any process. This probe does not touch /tmp/walk_forward_cache_v1, so it has no atime side effect.
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
from scipy.optimize import brentq

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "measurements" / "m1_config_seed_interaction.json"

FOLDS = ["f1", "f2", "f3", "f4"]
SEEDS = ["20260801", "20260802", "20260803"]
ALPHA = 0.05
N_PERM = 300
RNG_SEED = 20260919

# §CN's published model-B shares, used as the reproduction target
CN_PUBLISHED = {
    "fold": 7.1718, "seed": 16.2848, "fold_x_seed": 14.5190,
    "config": 6.2222, "config_x_fold": 17.1840, "residual": 38.6182,
}
CN_PUBLISHED_RESIDUAL_DF = 208


def load_peaks(ws: str, car: str) -> dict[tuple[str, str, str], int]:
    run_re = re.compile(r"^(?P<run>.+)-epoch-(?P<epoch>\d+)$")
    runs: dict[str, list[int]] = defaultdict(list)
    for d in glob.glob(str(ART / ws / "checkpoints" / "*-epoch-*")):
        m = run_re.match(Path(d).name)
        if m:
            runs[m.group("run")].append(int(m.group("epoch")))
    out: dict[tuple[str, str, str], int] = {}
    for r, e in runs.items():
        g = re.match(rf"^{car}-(f\d)-(c\d+)-(\d+)-seed-(\d+)$", r)
        if g:
            assert g.group(3) == g.group(4), f"tag/seed mismatch in {r}"
            out[(g.group(2), g.group(1), g.group(4))] = max(e)
    return out


def decompose(Y: np.ndarray, C: np.ndarray, F: np.ndarray, S: np.ndarray) -> dict:
    """Orthogonal cell-mean decomposition over the saturated 27 x 4 x 3 design.

    Terms are built up in a fixed order and each interaction is the joint grouping's SS minus its own
    main effects, which is exact for a balanced design with no missing cell (asserted by the caller).
    With one observation per cell the residual equals the config x fold x seed three-way interaction,
    so its df is 26 x 3 x 2 = 156 and there is no pure error term -- stated in the output because every
    F here is relative to three-way variation, not to replication.
    """
    gm = float(Y.mean())
    SST = float(((Y - gm) ** 2).sum())
    n = len(Y)
    df_total = n - 1
    factors = {"config": C, "fold": F, "seed": S}

    def ss_joint(names: list[str]) -> tuple[float, int]:
        cols = [factors[x] for x in names]
        # Labels are built once and then grouped by integer code via np.unique, so this is O(n log n)
        # rather than O(levels x n) -- the permutation control calls this 300 times per term.
        # sorted(set(...)) order is what np.unique already gives, and the summation runs over the
        # ascending group sums, so the floating-point order is deterministic across processes
        # (the defect §CM.6(iii) records for set()-iteration order).
        labels = np.array(["|".join(str(c[i]) for c in cols) for i in range(n)])
        _, inverse, counts = np.unique(labels, return_inverse=True, return_counts=True)
        sums = np.bincount(inverse, weights=Y, minlength=len(counts))
        means = sums / counts
        out = float((counts * (means - gm) ** 2).sum())
        return out, len(counts) - 1

    terms = [("fold", ["fold"]), ("seed", ["seed"]), ("config", ["config"]),
             ("fold_x_seed", ["fold", "seed"]), ("config_x_seed", ["config", "seed"]),
             ("config_x_fold", ["config", "fold"])]
    rows: dict[str, dict] = {}
    main_ss: dict[str, float] = {}
    main_df: dict[str, int] = {}
    used_ss = 0.0
    used_df = 0
    for name, comps in terms:
        if len(comps) == 1:
            ss, df = ss_joint(comps)
            main_ss[name], main_df[name] = ss, df
        else:
            sj, dj = ss_joint(comps)
            ss = sj - sum(main_ss[c] for c in comps)
            df = dj - sum(main_df[c] for c in comps)
        rows[name] = {"ss": ss, "df": df, "pct_of_SST": round(100 * ss / SST, 4)}
        used_ss += ss
        used_df += df
    df_res = df_total - used_df
    ss_res = SST - used_ss
    ms_res = ss_res / df_res
    for name, r in rows.items():
        r["ms"] = r["ss"] / r["df"]
        r["F"] = round(r["ms"] / ms_res, 4)
        r["p"] = float(stats.f.sf(r["F"], r["df"], df_res))
    rows["residual"] = {"ss": ss_res, "df": df_res, "pct_of_SST": round(100 * ss_res / SST, 4),
                        "ms": ms_res, "F": None, "p": None}
    return {"terms": rows, "SST": SST, "n": n, "grand_mean": round(gm, 6),
            "additivity_error": abs(SST - (used_ss + ss_res)),
            "df_total": df_total, "df_accounted": used_df + df_res,
            "residual_is_three_way_interaction_df": 26 * 3 * 2,
            "MS_residual": ms_res}


def power_analysis(df1: int, df2: int, ms_res: float, SST: float, n: int,
                   F_obs: float) -> dict:
    """Power of the config x seed test, the 95% upper bound on the interaction, and the noise check."""
    F_crit = float(stats.f.ppf(1 - ALPHA, df1, df2))
    lam_hi = brentq(lambda lam: stats.ncf.cdf(F_obs, df1, df2, lam) - ALPHA, 1e-9, 1000)
    ss_hi = lam_hi * ms_res
    rows = {}
    for target in (0.5, 0.8, 0.95):
        lam = brentq(lambda l, t=target: 1 - stats.ncf.cdf(F_crit, df1, df2, l) - t, 1e-6, 2000)
        ss = lam * ms_res
        rows[f"{int(target*100)}pct"] = {
            "lambda": round(lam, 2), "interaction_ss": round(ss, 1),
            "pct_of_SST": round(100 * ss / SST, 4),
            "rms_epochs_per_observation": round(float(np.sqrt(ss / n)), 3)}
    return {
        "critical_F_at_alpha_0.05": round(F_crit, 4),
        "observed_F": F_obs,
        "observed_numerator_chi2": round(F_obs * df1, 2),
        "null_mean_numerator_chi2": df1,
        "null_sd_numerator_chi2": round(float(np.sqrt(2 * df1)), 2),
        "sd_below_null_mean": round((df1 - F_obs * df1) / float(np.sqrt(2 * df1)), 3),
        "pure_noise_expected_interaction_ss": round(df1 * ms_res, 1),
        "upper_95pct_confidence_bound": {
            "lambda": round(float(lam_hi), 2), "interaction_ss": round(ss_hi, 1),
            "pct_of_SST": round(100 * ss_hi / SST, 4),
            "rms_epochs_per_observation": round(float(np.sqrt(ss_hi / n)), 3)},
        "detectable_interaction_sizes": rows,
        "reading": ("the observed interaction SS is below the pure-noise expectation and the numerator "
                    "chi-square sits within a third of a sd of its null mean, so F = 0.94 is an "
                    "ordinary null rather than a suspiciously small value"),
    }


def permutation_calibration(Y, C, F, S, n_perm: int = N_PERM) -> dict:
    """Average null behaviour over many permutations, not one flaky draw (§CN.6(ii)'s lesson)."""
    rng = np.random.default_rng(RNG_SEED)
    Fs = defaultdict(list)
    for _ in range(n_perm):
        Yp = Y.copy()
        rng.shuffle(Yp)
        d = decompose(Yp, C, F, S)
        for name, r in d["terms"].items():
            if r["F"] is not None:
                Fs[name].append(r["F"])
        df2 = d["terms"]["residual"]["df"]
        dfs = {name: r["df"] for name, r in d["terms"].items() if r["F"] is not None}
    out = {}
    for name, vals in sorted(Fs.items()):
        df1 = dfs[name]
        crit = float(stats.f.ppf(1 - ALPHA, df1, df2))
        out[name] = {"null_mean_F": round(float(np.mean(vals)), 4),
                     "null_mean_F_within_15pct_of_1": bool(abs(np.mean(vals) - 1) < 0.15),
                     "false_positive_rate_at_alpha_0.05": round(
                         float(np.mean([v > crit for v in vals])), 4),
                     "fpr_in_expected_band": bool(0.0 <= float(np.mean([v > crit for v in vals])) <= 0.12),
                     "n_permutations": len(vals)}
    return {"per_term": out,
            "all_terms_calibrated": all(
                v["null_mean_F_within_15pct_of_1"] and v["fpr_in_expected_band"]
                for v in out.values())}


def injection_control(Y, C, F, S, target_pct: float = 10.0) -> dict:
    """Inject a config x seed interaction of known size; the test MUST detect it.

    Without this, "F = 0.94 on the real data" could not be distinguished from an instrument that
    reports ~1 for every input -- the failure mode §CO.6 records for Kendall's W, where a reversed
    matrix orientation produced W = 0 identically and one of the two controls was blind to it.

    The injection adds, to each cell, a fixed offset that depends only on (config, seed) and sums to
    zero over both margins, so it creates interaction WITHOUT perturbing the config or seed main
    effects. Its scale is calibrated so the resulting interaction SS is the target fraction of SST.
    """
    cfgs = sorted(set(C.tolist()), key=lambda c: int(c[1:]))
    rng = np.random.default_rng(RNG_SEED + 1)
    # centred random offsets on the (config, seed) grid
    off = rng.normal(size=(len(cfgs), len(SEEDS)))
    off -= off.mean(axis=0, keepdims=True)      # zero column means -> no seed main effect
    off -= off.mean(axis=1, keepdims=True)      # zero row means    -> no config main effect
    idx_c = {c: i for i, c in enumerate(cfgs)}
    idx_s = {s: i for i, s in enumerate(SEEDS)}
    pattern = np.array([off[idx_c[c], idx_s[s]] for c, s in zip(C, S)])
    # calibrate: interaction SS of (alpha*pattern) in this design equals alpha^2 * SS(pattern|cells)
    gm_p = pattern.mean()
    ss_pattern = float(((pattern - gm_p) ** 2).sum())
    SST = float(((Y - Y.mean()) ** 2).sum())
    scale = np.sqrt(target_pct / 100 * SST / ss_pattern)
    Y2 = Y + scale * pattern
    d2 = decompose(Y2, C, F, S)
    t = d2["terms"]["config_x_seed"]
    # ABSOLUTE SS units, never percentages. Injecting variance enlarges SST, so every share-of-SST
    # figure falls by dilution even when the underlying SS is unchanged: the first draft of this
    # control compared percentages and reported the seed main effect "moving" from 16.2848% to
    # 14.958%, when in SS units it went 3807 -> 3847, i.e. +1.0%. A control that compares two
    # quantities with different denominators is not measuring what it claims.
    base = decompose(Y, C, F, S)["terms"]
    expected_injected_ss = float(scale ** 2 * ss_pattern)
    return {
        "injected_interaction_target_pct_of_original_SST": target_pct,
        "scale_applied_epochs": round(float(scale), 4),
        "expected_injected_ss_units": round(expected_injected_ss, 1),
        "interaction_ss_before": round(base["config_x_seed"]["ss"], 1),
        "interaction_ss_after": round(t["ss"], 1),
        "interaction_ss_increase": round(t["ss"] - base["config_x_seed"]["ss"], 1),
        "recovered_fraction_of_expected_increase": round(
            (t["ss"] - base["config_x_seed"]["ss"]) / expected_injected_ss, 4),
        "recovered_pct_of_SST_after": t["pct_of_SST"],
        "recovered_F": t["F"], "recovered_p": t["p"],
        "F_before": base["config_x_seed"]["F"],
        "detected_at_alpha_0.05": bool(t["p"] < ALPHA),
        "seed_main_effect_ss_before": round(base["seed"]["ss"], 1),
        "seed_main_effect_ss_after": round(d2["terms"]["seed"]["ss"], 1),
        "seed_main_effect_ss_change_pct": round(
            100 * (d2["terms"]["seed"]["ss"] - base["seed"]["ss"]) / base["seed"]["ss"], 2),
        "config_main_effect_ss_before": round(base["config"]["ss"], 1),
        "config_main_effect_ss_after": round(d2["terms"]["config"]["ss"], 1),
        "config_main_effect_ss_change_pct": round(
            100 * (d2["terms"]["config"]["ss"] - base["config"]["ss"]) / base["config"]["ss"], 2),
        "reading": ("the injection is margin-centred so it must not move the seed or config main "
                    "effects in SS units; verifying that they stay put is part of the control, because "
                    "an instrument that leaks injected interaction into a main effect would "
                    "misattribute variance on the real data too. All comparisons are in absolute SS."),
    }


def main() -> None:
    cell = load_peaks("m1_full_v1", "m1")
    cfgs = sorted({k[0] for k in cell}, key=lambda c: int(c[1:]))
    assert len(cfgs) == 27 and len(cell) == 27 * 4 * 3, "expected a complete 27x4x3 design"
    Y = np.array([cell[(c, f, s)] for c in cfgs for f in FOLDS for s in SEEDS], dtype=float)
    C = np.array([c for c in cfgs for _ in FOLDS for _ in SEEDS])
    F = np.array([f for _ in cfgs for f in FOLDS for _ in SEEDS])
    S = np.array([s for _ in cfgs for _ in FOLDS for s in SEEDS])
    assert len(Y) == len(C) == len(F) == len(S) == 324

    d = decompose(Y, C, F, S)
    terms = d["terms"]
    cs = terms["config_x_seed"]
    pa = power_analysis(cs["df"], terms["residual"]["df"], d["MS_residual"], d["SST"], d["n"], cs["F"])
    noise_ss = cs["df"] * d["MS_residual"]
    pa["observed_interaction_ss"] = round(cs["ss"], 1)
    pa["observed_interaction_ss_below_noise_expectation_pct"] = round(
        100 * (1 - cs["ss"] / noise_ss), 2)

    seed_pct = terms["seed"]["pct_of_SST"]
    ub_pct = pa["upper_95pct_confidence_bound"]["pct_of_SST"]
    config_independent_share = (seed_pct - ub_pct) / seed_pct

    inj = injection_control(Y, C, F, S)

    perm = permutation_calibration(Y, C, F, S)

    # The residual is EXCLUDED from the equality check on purpose. §CN's model B lacks the
    # config x seed term, so its residual must exceed this one by exactly that term's SS share and df
    # -- comparing it for equality would fail by construction. The first draft of this control did
    # exactly that and reported "reproduces_CN_exactly: False" while all five shared terms matched to
    # four decimals. The check that replaces it is stronger: the residual difference must EQUAL the
    # new term, which is asserted below.
    reproduce = {}
    for name, pub in CN_PUBLISHED.items():
        if name == "residual":
            continue
        got = terms[name]["pct_of_SST"]
        reproduce[name] = {"CN_published_pct": pub, "recomputed_pct": got,
                           "matches": bool(abs(got - pub) < 1e-3)}
    residual_diff = round(CN_PUBLISHED["residual"] - terms["residual"]["pct_of_SST"], 4)
    df_diff = CN_PUBLISHED_RESIDUAL_DF - terms["residual"]["df"]

    doc = {
        "question": ("is the seed effect on peak epoch config-specific (channel B, weight init) or "
                     "config-independent (channel A, the sampler's date order), and with what power "
                     "can this design answer that?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "design": {"carrier": "M1 (RESIDUAL_MLP)", "workspace": "m1_full_v1",
                   "n_configs": len(cfgs), "n_folds": 4, "n_seeds": 3, "n_observations": d["n"],
                   "balanced_complete": True, "observations_per_cell": 1,
                   "saturated_no_pure_error": True,
                   "residual_equals_config_x_fold_x_seed_df": terms["residual"]["df"],
                   "expected_three_way_df": d["residual_is_three_way_interaction_df"]},
        "decomposition": terms,
        "grand_mean": d["grand_mean"], "SST": round(d["SST"], 2),
        "additivity_error": d["additivity_error"], "df_accounted": d["df_accounted"],
        "df_total": d["df_total"],
        "reproduces_CN_model_B": {
            "per_shared_term": reproduce,
            "residual_pct_CN": CN_PUBLISHED["residual"],
            "residual_pct_here": terms["residual"]["pct_of_SST"],
            "residual_difference_pct": residual_diff,
            "residual_df_CN": CN_PUBLISHED_RESIDUAL_DF,
            "residual_df_here": terms["residual"]["df"],
            "residual_df_difference": df_diff,
            "difference_equals_config_x_seed_term": (
                abs(residual_diff - cs["pct_of_SST"]) < 1e-3 and df_diff == cs["df"]),
            "all_shared_terms_match": all(v["matches"] for v in reproduce.values()),
            "reading": ("§CN's model B is this model minus the config x seed term, so its residual must "
                        "exceed this one by exactly that term's SS share and df. It does, to four "
                        "decimals, and all five shared terms reproduce §CN exactly -- so this is the "
                        "same decomposition with one more term partitioned out, not a different model."),
        },
        "config_x_seed_test": {
            "pct_of_SST": cs["pct_of_SST"], "df": cs["df"], "F": cs["F"], "p": cs["p"],
            "MS_interaction": round(cs["ms"], 4), "MS_residual": round(d["MS_residual"], 4),
            "F_below_1": bool(cs["F"] < 1.0),
            "MS_interaction_below_MS_residual": bool(cs["ms"] < d["MS_residual"]),
        },
        "power_analysis": pa,
        "verdict_on_config_independence": {
            "seed_main_effect_pct_of_SST": seed_pct,
            "seed_main_effect_swing_epochs": 8.3519,
            "config_x_seed_upper_95pct_bound_pct_of_SST": ub_pct,
            "config_x_seed_upper_95pct_bound_rms_epochs":
                pa["upper_95pct_confidence_bound"]["rms_epochs_per_observation"],
            "min_share_of_seed_variance_that_is_config_independent_pct": round(
                100 * config_independent_share, 1),
            "statement": (f"at least {round(100 * config_independent_share, 1)}% of the seed variance "
                          f"in peak epoch is shared across all 27 configs, and no config-specific "
                          f"component larger than {ub_pct}% of SST "
                          f"(rms {pa['upper_95pct_confidence_bound']['rms_epochs_per_observation']} "
                          "epochs) can be present"),
        },
        "channel_predictions_versus_observed": {
            "channel_A_sampler_date_order": {
                "source": ("sampler.py:27-32 sorts dates by sha256 of the seed joined to the date; "
                           ":33-34 rotates by epoch modulo the date count; constructed at "
                           "torch_loop.py:43 from that fold's training dates"),
                "depends_on": "(seed, that fold's date set) -- identical for every config in a fold",
                "predicts_seed_main_effect": True,
                "predicts_fold_x_seed_interaction": True,
                "predicts_config_x_seed_interaction": False,
            },
            "channel_B_initial_weights": {
                "source": ("cross_section_search.py:178-179 calls configure_reproducibility(seed), "
                           "whose torch.manual_seed is at seeds.py:46, then constructs the model"),
                "depends_on": "config, through how much of the seeded RNG stream its parameter count consumes",
                "predicts_seed_main_effect": False,
                "predicts_fold_x_seed_interaction": False,
                "predicts_config_x_seed_interaction": True,
                "PARTIAL_CAVEAT": ("configs sharing an architecture prefix consume the same initial "
                                   "segment of the seeded stream, so two configs differing only in "
                                   "num_blocks draw identical values for their shared first block. "
                                   "Channel B is therefore only PARTLY config-dependent and could "
                                   "contribute some config-independent variance; this probe does not "
                                   "exclude it."),
            },
            "observed": {
                "seed_main_effect": f"{terms['seed']['pct_of_SST']}% (F {terms['seed']['F']})",
                "fold_x_seed": f"{terms['fold_x_seed']['pct_of_SST']}% (F {terms['fold_x_seed']['F']}, "
                               f"p {terms['fold_x_seed']['p']:.2e})",
                "config_x_seed": f"{cs['pct_of_SST']}% (F {cs['F']}, p {cs['p']:.3f})",
                "matches_channel_A_signature": bool(
                    terms["seed"]["p"] < 1e-6 and terms["fold_x_seed"]["p"] < 1e-6
                    and cs["F"] < 1.2),
            },
            "what_is_established": (
                "the seed effect on peak epoch is config-independent to within "
                f"{ub_pct}% of SST, so any mechanism must act nearly identically on all 27 configs; "
                "of the two channels only the sampler's date order is config-independent BY "
                "CONSTRUCTION, which makes it the better-supported candidate"),
            "what_is_NOT_established": (
                "that the sampler IS the mechanism. This probe measures variance structure only; it "
                "never runs the sampler, never constructs a model, never counts RNG draws. Channel B "
                "is not excluded, for the prefix-sharing reason above. Per ledger B1 no unverified "
                "mechanism is asserted."),
        },
        "injection_control": inj,
        "permutation_calibration": perm,
        "controls_pass": {},
        "read_only": True, "database_access": False, "weights_loaded": False,
        "checkpoint_pt_opened": False, "parquet_read": False, "training_performed": False,
        "repository_module_imported_or_executed": False, "processes_signalled": [],
        "atime_side_effect": False,
    }
    doc["controls_pass"] = {
        "reproduces_CN_exactly": doc["reproduces_CN_model_B"]["all_shared_terms_match"],
        "CN_residual_difference_is_the_new_term":
            doc["reproduces_CN_model_B"]["difference_equals_config_x_seed_term"],
        "additivity_exact": d["additivity_error"] < 1e-6,
        "df_accounted_equals_total": d["df_accounted"] == d["df_total"],
        "residual_is_the_three_way_interaction":
            terms["residual"]["df"] == d["residual_is_three_way_interaction_df"],
        "injection_detected": inj["detected_at_alpha_0.05"],
        "injection_recovered_expected_ss_within_25pct": abs(
            inj["recovered_fraction_of_expected_increase"] - 1.0) < 0.25,
        "injection_did_not_move_seed_main_effect_ss": abs(
            inj["seed_main_effect_ss_change_pct"]) < 5.0,
        "injection_did_not_move_config_main_effect_ss": abs(
            inj["config_main_effect_ss_change_pct"]) < 15.0,
        "permutation_calibrated": perm["all_terms_calibrated"],
    }
    doc["controls_pass"]["ALL_PASS"] = all(
        v for k, v in doc["controls_pass"].items() if k != "ALL_PASS")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

    print(f"design: {len(cfgs)} configs x 4 folds x 3 seeds = {d['n']} obs, "
          f"1 per cell (saturated, residual = three-way {terms['residual']['df']} df)")
    print(f"\n=== decomposition (additivity {d['additivity_error']:.2e}, df {d['df_accounted']}"
          f"/{d['df_total']}) ===")
    print(f"  {'source':16s} {'SS%':>8s} {'df':>4s} {'MS':>10s} {'F':>9s} {'p':>11s}")
    for name in ("fold", "seed", "config", "fold_x_seed", "config_x_seed", "config_x_fold"):
        r = terms[name]
        tag = "   <== DISCRIMINATOR" if name == "config_x_seed" else ""
        print(f"  {name:16s} {r['pct_of_SST']:>7.4f}% {r['df']:>4d} {r['ms']:>10.4f} "
              f"{r['F']:>9.4f} {r['p']:>11.3e}{tag}")
    r = terms["residual"]
    print(f"  {'residual':16s} {r['pct_of_SST']:>7.4f}% {r['df']:>4d} {r['ms']:>10.4f}")
    rc = doc["reproduces_CN_model_B"]
    print(f"\n=== reproduces §CN model B: {rc['all_shared_terms_match']}; "
          f"residual difference {rc['residual_difference_pct']}% / {rc['residual_df_difference']} df "
          f"== config_x_seed term: {rc['difference_equals_config_x_seed_term']} ===")
    print(f"\n=== power of the config x seed test ===")
    print(f"  critical F {pa['critical_F_at_alpha_0.05']}, observed F {pa['observed_F']} "
          f"(numerator chi2 {pa['observed_numerator_chi2']} vs null mean {pa['null_mean_numerator_chi2']}"
          f", {pa['sd_below_null_mean']} sd below)")
    print(f"  observed interaction SS {pa['observed_interaction_ss']} vs pure-noise expectation "
          f"{pa['pure_noise_expected_interaction_ss']} -> "
          f"{pa['observed_interaction_ss_below_noise_expectation_pct']}% BELOW noise")
    for k, v in pa["detectable_interaction_sizes"].items():
        print(f"  {k:6s} power at {v['pct_of_SST']:6.2f}% SST = rms {v['rms_epochs_per_observation']:.2f} epochs")
    ub = pa["upper_95pct_confidence_bound"]
    print(f"  95% upper bound: {ub['pct_of_SST']}% SST = rms {ub['rms_epochs_per_observation']} epochs")
    v = doc["verdict_on_config_independence"]
    print(f"\n=== verdict ===")
    print(f"  seed main effect {v['seed_main_effect_pct_of_SST']}% SST "
          f"(swing {v['seed_main_effect_swing_epochs']} epochs); config x seed <= "
          f"{v['config_x_seed_upper_95pct_bound_pct_of_SST']}% SST")
    print(f"  => at least {v['min_share_of_seed_variance_that_is_config_independent_pct']}% of the "
          f"seed variance is config-INDEPENDENT")
    co = doc["channel_predictions_versus_observed"]
    print(f"\n  channel A signature matched: {co['observed']['matches_channel_A_signature']}")
    print(f"  observed: seed {co['observed']['seed_main_effect']} | fold x seed "
          f"{co['observed']['fold_x_seed']} | config x seed {co['observed']['config_x_seed']}")
    print(f"\n=== injection control (must DETECT a known interaction; all figures in SS units) ===")
    print(f"  injected {inj['injected_interaction_target_pct_of_original_SST']}% of original SST "
          f"= {inj['expected_injected_ss_units']} SS units (scale {inj['scale_applied_epochs']} epochs)")
    print(f"  interaction SS {inj['interaction_ss_before']} -> {inj['interaction_ss_after']} "
          f"(+{inj['interaction_ss_increase']}, i.e. "
          f"{inj['recovered_fraction_of_expected_increase']:.3f}x the injected amount)")
    print(f"  F {inj['F_before']} -> {inj['recovered_F']}, p {inj['recovered_p']:.2e}, "
          f"detected={inj['detected_at_alpha_0.05']}")
    print(f"  seed main effect SS {inj['seed_main_effect_ss_before']} -> "
          f"{inj['seed_main_effect_ss_after']} ({inj['seed_main_effect_ss_change_pct']:+.2f}%, must stay put)")
    print(f"  config main effect SS {inj['config_main_effect_ss_before']} -> "
          f"{inj['config_main_effect_ss_after']} ({inj['config_main_effect_ss_change_pct']:+.2f}%)")
    print(f"\n=== permutation calibration ({N_PERM} draws) ===")
    for name, r2 in perm["per_term"].items():
        print(f"  {name:16s} null mean F {r2['null_mean_F']:>7.4f} "
              f"({'ok' if r2['null_mean_F_within_15pct_of_1'] else '*** OFF ***'}) | "
              f"FPR {r2['false_positive_rate_at_alpha_0.05']:.4f}")
    print(f"\n=== controls ===")
    for k, ok in doc["controls_pass"].items():
        print(f"  {'OK ' if ok else '*** FAIL ***'} {k}")
    assert doc["controls_pass"]["ALL_PASS"], "a control failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
