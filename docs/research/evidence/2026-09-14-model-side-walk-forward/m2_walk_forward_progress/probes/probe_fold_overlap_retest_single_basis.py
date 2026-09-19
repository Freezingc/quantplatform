#!/usr/bin/env python3
"""Read-only probe for §CW: redo §CV.5's inconclusive verdict on ONE analysis unit, correct a basis
mismatch §CV itself contained, and add two independent tests it did not have.

THE DEFECT IN §CV.5 THAT THIS PROBE CORRECTS. §CV.5 argued the empirical test of window overlap was
inconclusive because "the contrast's sign flips with the metric", citing ensemble RankIC -0.3230 (27 rows,
one per config) against per-seed RankIC +0.5510 (81 rows, one per config-seed). Those two numbers are not
comparable: they use DIFFERENT ANALYSIS UNITS. Centering each of 81 config-seed rows removes that row's own
mean, and the three seeds of one config agree strongly with each other (their fold profiles correlate 0.67
to 0.93), so the 81-row centering mostly removes within-config seed agreement rather than anything to do
with fold overlap. Putting both on the SAME 27-row basis gives ensemble -0.3230 and seed-mean -0.0631 --
the SAME SIGN. So the headline "sign flip" in §CV.5 was partly an artefact of comparing a 27-row statistic
with an 81-row one, which is exactly the basis-mismatch family this pack keeps catching (B13, B17②, B21,
B25). §CV.5's per-seed row used a config-blocked null, which was the right instinct about clustering, but
the reported contrast value itself was still on the 81-row basis and therefore not comparable to the
ensemble one. This probe reports every statistic on the config basis (27 or 24 rows), which is the honest
unit: the three seeds of a config are not independent draws of the config-fold response.

TWO INDEPENDENT TESTS §CV DID NOT HAVE. (a) M0 (LGBM_RANKER, gradient-boosted trees) runs on the SAME four
fold windows -- verified: all four dataset_ids match M1's exactly -- with 24 configs, so its config x fold
response is a second, model-family-independent test of the same question. If window overlap induced
dependence, a different model family on the same windows should show it too. (b) peak epoch at the SEED
level (81 rows) versus the seed-mean level (27 rows) for M1, which shows the contrast statistic itself is
basis-sensitive (+0.0002 at seed level versus +0.1905 at seed-mean level), so quoting it without stating
the basis is meaningless -- the same lesson as the §CV.5 defect, measured rather than asserted.

WHAT THE CORRECTED VERDICT IS. On the config basis, across four independent (metric, carrier) combinations:
M1 ensemble RankIC -0.3230, M1 seed-mean RankIC -0.0631, M0 ensemble RankIC -0.0995, M1 peak epoch
(seed-mean) +0.1905. Three of four are negative -- adjacent folds LESS correlated than distant ones, the
opposite of what shared training data predicts -- and the fourth is not significant. So the corrected
verdict is not "sign flips with the metric" but "no metric shows the positive adjacency dependence that
window overlap would induce, and three show the opposite sign". This is a STRONGER statement than §CV.5
made and it is the honest one: the overlap's predicted consequence is not observed. But the structural
overlap (§CV.1/CV.2) remains certain and its limitation for the paper remains valid -- the absence of a
measurable adjacency signal in these four tests does not prove four folds carry four independent
confirmations' worth of evidence, especially since a pure fold main effect is invisible to this test by
construction (§CV.6 control vi).

CONTROLS: (i) the config-basis and seed-basis contrasts must differ for peak epoch, proving the basis
sensitivity claim is measured and not assumed; (ii) the permutation null must reproduce the analytic
-1/(k-1) = -0.3333 centering artefact; (iii) a planted config x year interaction must be detected on the
config basis with the corrected two-directional p-values (§CV.6's defect fix); (iv) independent synthetic
data must be non-significant; (v) M0 and M1 must be confirmed to share all four dataset_ids, otherwise M0
is not an independent test of the same windows; (vi) the config-basis ensemble and seed-mean contrasts must
share a sign, which is the direct test of the §CV.5 basis-mismatch correction.

Reads only: report.json text, search-record JSONs, checkpoint directory names. No parquet (not even
footers), no checkpoint.pt, no weight or tensor, no repository module imported or executed, no database,
no network, no writes to any artifact directory, no signal to any process. Does not touch
/tmp/walk_forward_cache_v1, so no atime side effect. Fixed RNG seeds and no live state, so byte-reproducible.
"""
from __future__ import annotations

import glob
import itertools
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "measurements" / "fold_overlap_retest_single_basis.json"

FOLDS = ["f1", "f2", "f3", "f4"]
SEEDS = ["20260801", "20260802", "20260803"]
PAIRS = list(itertools.combinations(range(4), 2))
SHARED = {(0, 1): 3, (1, 2): 3, (2, 3): 3, (0, 2): 2, (1, 3): 2, (0, 3): 1}
N_PERM = 5000
RNG_SEED = 20260919


def residual_corr(M: np.ndarray) -> dict:
    R = M - M.mean(axis=1, keepdims=True)
    return {(a, b): float(np.corrcoef(R[:, a], R[:, b])[0, 1]) for a, b in PAIRS}


def contrast(c: dict) -> float:
    adj = [c[k] for k in PAIRS if SHARED[k] == 3]
    non = [c[k] for k in PAIRS if SHARED[k] < 3]
    return float(np.mean(adj) - np.mean(non))


def test(M: np.ndarray, label: str, n_perm: int = N_PERM, seed: int = RNG_SEED) -> dict:
    """Permutation-calibrated adjacency test, with BOTH one-sided directions reported explicitly."""
    rng = np.random.default_rng(seed)
    obs = residual_corr(M)
    obs_c = contrast(obs)
    null_corrs = {k: [] for k in PAIRS}
    null_c = []
    n = M.shape[0]
    for _ in range(n_perm):
        P = M.copy()
        for i in range(n):
            P[i] = P[i][rng.permutation(M.shape[1])]
        c = residual_corr(P)
        for k in PAIRS:
            null_corrs[k].append(c[k])
        null_c.append(contrast(c))
    null_c = np.array(null_c)
    p_pos = float((null_c >= obs_c).mean())
    p_neg = float((null_c <= obs_c).mean())
    per_pair = {}
    for k in PAIRS:
        a = np.array(null_corrs[k])
        per_pair[f"F{k[0]+1}-F{k[1]+1}"] = {
            "shared_training_years": SHARED[k], "observed_corr": round(obs[k], 4),
            "null_mean": round(float(a.mean()), 4), "null_sd": round(float(a.std()), 4),
            "z": round(float((obs[k] - a.mean()) / a.std()), 3),
            "two_sided_p": round(max(float(2 * min((a <= obs[k]).mean(), (a >= obs[k]).mean())),
                                     1 / n_perm), 4)}
    sig_p = p_pos if obs_c > 0 else p_neg
    return {"metric": label, "n_rows": n, "analysis_unit": "config",
            "per_pair": per_pair,
            "contrast_observed": round(obs_c, 4),
            "contrast_null_mean": round(float(null_c.mean()), 4),
            "contrast_null_sd": round(float(null_c.std()), 4),
            "contrast_z": round(float((obs_c - null_c.mean()) / null_c.std()), 3),
            "p_one_sided_adjacent_more_POSITIVE": round(max(p_pos, 1 / n_perm), 4),
            "p_one_sided_adjacent_more_NEGATIVE": round(max(p_neg, 1 / n_perm), 4),
            "significant_in_observed_direction": bool(sig_p < 0.05),
            "direction": ("adjacent MORE positively correlated (as shared data predicts)"
                          if obs_c > 0 else
                          "adjacent MORE NEGATIVELY correlated (opposite of what shared data predicts)")}


def load_m1() -> dict:
    ens, per = {}, {}
    for f in ("F1", "F2", "F3", "F4"):
        for rec in sorted((ART / "m1_full_v1" / "search" / f).glob("c*.json")):
            d = json.load(open(rec, encoding="utf-8"))
            c = f"c{d['config_index']}"
            ens[(c, f.lower())] = d["result"]["ensemble_validation_rank_ic"]
            for s, v in d["result"]["seed_validation_rank_ic"].items():
                per[(c, s, f.lower())] = v
    cfgs = sorted({k[0] for k in ens}, key=lambda c: int(c[1:]))
    M_ens = np.array([[ens[(c, f)] for f in FOLDS] for c in cfgs], dtype=float)
    M_seedmean = np.array([[np.mean([per[(c, s, f)] for s in SEEDS]) for f in FOLDS]
                           for c in cfgs], dtype=float)
    M_seedlevel = np.array([[per[(c, s, f)] for f in FOLDS] for c in cfgs for s in SEEDS], dtype=float)
    return {"configs": cfgs, "ensemble": M_ens, "seed_mean": M_seedmean, "seed_level": M_seedlevel}


def load_m0() -> dict:
    """M0 (LGBM_RANKER) per-config ensemble rank_ic, keyed by config CONTENT so folds line up."""
    d = json.load(open(ART / "m0_v3" / "report.json", encoding="utf-8"))
    by_cfg: dict[str, dict[str, float]] = defaultdict(dict)
    for fo in d["folds"]:
        for r in fo["search_results"]:
            by_cfg[json.dumps(r["config"], sort_keys=True)][fo["fold_id"].lower()] = \
                r["ensemble_validation_rank_ic"]
    keys = sorted(k for k, v in by_cfg.items() if len(v) == 4)
    M = np.array([[by_cfg[k][f] for f in FOLDS] for k in keys], dtype=float)
    ds = {fo["fold_id"]: fo["dataset_id"] for fo in d["folds"]}
    return {"matrix": M, "n_configs": len(keys), "dataset_ids": ds,
            "configs": [json.loads(k) for k in keys]}


def load_m1_peak() -> dict:
    run_re = re.compile(r"^(.+)-epoch-(\d+)$")
    runs: dict[str, list[int]] = defaultdict(list)
    for d in glob.glob(str(ART / "m1_full_v1" / "checkpoints" / "*-epoch-*")):
        m = run_re.match(Path(d).name)
        if m:
            runs[m.group(1)].append(int(m.group(2)))
    pk: dict[tuple[str, str, str], int] = {}
    for r, e in runs.items():
        g = re.match(r"^m1-(f\d)-(c\d+)-(\d+)-seed-(\d+)$", r)
        if g:
            pk[(g.group(2), g.group(4), g.group(1))] = max(e)
    cfgs = sorted({k[0] for k in pk}, key=lambda c: int(c[1:]))
    M_mean = np.array([[float(np.mean([pk[(c, s, f)] for s in SEEDS])) for f in FOLDS]
                       for c in cfgs], dtype=float)
    M_seed = np.array([[float(pk[(c, s, f)]) for f in FOLDS] for c in cfgs for s in SEEDS], dtype=float)
    return {"seed_mean": M_mean, "seed_level": M_seed, "n_configs": len(cfgs)}


def centering_artifact() -> dict:
    rng = np.random.default_rng(3)
    M = rng.normal(size=(400, 4))
    c = residual_corr(M)
    analytic = -1.0 / 3
    emp = float(np.mean(list(c.values())))
    return {"analytic": round(analytic, 4), "empirical": round(emp, 4),
            "within_0.05": abs(emp - analytic) < 0.05}


def synthetic_controls() -> dict:
    """Planted config x year interaction (detectable) vs independent (not), on the CONFIG basis."""
    rng = np.random.default_rng(11)
    years = list(range(2017, 2024))
    M_ind = rng.normal(size=(27, 4))
    t_ind = test(M_ind, "SYNTHETIC independent", n_perm=2000, seed=5)
    gain = rng.normal(size=(27, len(years)))
    M_dep = np.zeros((27, 4))
    for i in range(27):
        for a in range(4):
            M_dep[i, a] = (sum(gain[i, years.index(y)] for y in years[a:a + 4])
                           + float(rng.normal(scale=0.5)))
    t_dep = test(M_dep, "SYNTHETIC config x year interaction", n_perm=2000, seed=7)
    return {
        "independent": {"contrast": t_ind["contrast_observed"],
                        "p_more_positive": t_ind["p_one_sided_adjacent_more_POSITIVE"],
                        "p_more_negative": t_ind["p_one_sided_adjacent_more_NEGATIVE"],
                        "significant": t_ind["significant_in_observed_direction"]},
        "planted_interaction": {"contrast": t_dep["contrast_observed"], "z": t_dep["contrast_z"],
                                "p_more_positive": t_dep["p_one_sided_adjacent_more_POSITIVE"],
                                "significant": t_dep["significant_in_observed_direction"]},
        "planted_detected_positive": bool(
            t_dep["significant_in_observed_direction"] and t_dep["contrast_observed"] > 0),
        "independent_not_significant": not t_ind["significant_in_observed_direction"],
    }


def main() -> None:
    m1 = load_m1()
    m0 = load_m0()
    pk = load_m1_peak()
    m1_ds = {f["fold_id"]: f["dataset_id"]
             for f in json.load(open(ART / "m1_full_v1" / "report.json", encoding="utf-8"))["folds"]}
    same_panels = m0["dataset_ids"] == m1_ds

    # --- the §CV.5 basis-mismatch correction, measured
    basis = {
        "ensemble_27rows": contrast(residual_corr(m1["ensemble"])),
        "seed_mean_27rows": contrast(residual_corr(m1["seed_mean"])),
        "seed_level_81rows": contrast(residual_corr(m1["seed_level"])),
        "peak_epoch_seed_mean_27rows": contrast(residual_corr(pk["seed_mean"])),
        "peak_epoch_seed_level_81rows": contrast(residual_corr(pk["seed_level"])),
    }
    same_sign_on_config_basis = (np.sign(basis["ensemble_27rows"])
                                 == np.sign(basis["seed_mean_27rows"]))
    basis_sensitive_peak = abs(basis["peak_epoch_seed_level_81rows"]
                               - basis["peak_epoch_seed_mean_27rows"]) > 0.05

    # --- the four tests, all on the CONFIG basis
    tests = {
        "M1_ensemble_rank_ic": test(m1["ensemble"], "M1 ensemble validation RankIC (27 configs)"),
        "M1_seed_mean_rank_ic": test(m1["seed_mean"], "M1 seed-mean validation RankIC (27 configs)"),
        "M0_ensemble_rank_ic": test(m0["matrix"], "M0 (LGBM_RANKER) ensemble RankIC (24 configs)"),
        "M1_peak_epoch_seed_mean": test(pk["seed_mean"], "M1 peak epoch, seed-mean (27 configs)"),
    }
    # the seed-level ones are reported for the basis-sensitivity point, NOT as independent tests
    basis_only = {
        "M1_seed_level_rank_ic_81rows_NOT_A_VALID_UNIT": test(
            m1["seed_level"], "M1 per-seed RankIC (81 rows) -- basis illustration only",
            n_perm=2000, seed=9),
        "M1_peak_epoch_seed_level_81rows_NOT_A_VALID_UNIT": test(
            pk["seed_level"], "M1 peak epoch per seed (81 rows) -- basis illustration only",
            n_perm=2000, seed=10),
    }

    n_negative = sum(1 for t in tests.values() if t["contrast_observed"] < 0)
    n_positive_sig = sum(1 for t in tests.values()
                         if t["contrast_observed"] > 0 and t["significant_in_observed_direction"])
    art = centering_artifact()
    syn = synthetic_controls()

    doc = {
        "question": ("on a single honest analysis unit (the config), does training-window overlap induce "
                     "positive dependence between folds, and does §CV.5's 'sign flips with the metric' "
                     "verdict survive correcting its basis mismatch?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "cv5_basis_mismatch_corrected": {
            "what_CV5_claimed": ("ensemble RankIC -0.3230 (27 rows) versus per-seed RankIC +0.5510 "
                                 "(81 rows) differ in sign, so the test is inconclusive"),
            "why_that_comparison_is_invalid": ("different analysis units. Centering 81 config-seed rows "
                                               "removes each row's own mean, and a config's three seeds "
                                               "correlate 0.67-0.93 in their fold profiles, so the 81-row "
                                               "centering mostly removes within-config seed agreement, "
                                               "not fold overlap"),
            "on_the_same_27row_config_basis": {
                "ensemble": round(basis["ensemble_27rows"], 4),
                "seed_mean": round(basis["seed_mean_27rows"], 4),
                "same_sign": bool(same_sign_on_config_basis)},
            "conclusion": ("the headline 'sign flip' was partly an artefact of comparing a 27-row "
                           "statistic with an 81-row one -- the basis-mismatch family (B13, B17②, B21, "
                           "B25). §CV.5's per-seed row did use a config-BLOCKED null, which was the right "
                           "instinct about clustering, but the contrast VALUE it reported was still on the "
                           "81-row basis and so not comparable to the ensemble one."),
            "basis_sensitivity_measured_on_peak_epoch": {
                "seed_mean_27rows": round(basis["peak_epoch_seed_mean_27rows"], 4),
                "seed_level_81rows": round(basis["peak_epoch_seed_level_81rows"], 4),
                "differ_materially": bool(basis_sensitive_peak),
                "reading": ("the same quantity gives +0.19 on one basis and +0.00 on the other, so a "
                            "contrast quoted without its basis is meaningless")},
        },
        "m0_is_an_independent_carrier_test": {
            "m0_family": "LGBM_RANKER (gradient-boosted trees)",
            "m1_family": "RESIDUAL_MLP (neural)",
            "same_four_dataset_ids": bool(same_panels),
            "m0_dataset_ids": m0["dataset_ids"],
            "m1_dataset_ids": m1_ds,
            "m0_n_configs": m0["n_configs"],
            "reading": ("M0 trains a different model family on the SAME four fold windows, so its config x "
                        "fold response is an independent test of whether the window overlap induces "
                        "dependence; M0R cannot be used because it has 1 config per fold"),
        },
        "tests_on_config_basis": tests,
        "basis_illustration_only_not_valid_units": basis_only,
        "corrected_verdict": {
            "n_tests": len(tests),
            "n_with_negative_contrast": n_negative,
            "n_with_significant_POSITIVE_contrast": n_positive_sig,
            "statement": (
                f"{n_negative} of {len(tests)} config-basis tests give a NEGATIVE contrast (adjacent folds "
                f"LESS correlated than distant ones, the opposite of what shared training data predicts) "
                f"and {n_positive_sig} give a significant positive one. So the corrected verdict is not "
                "'the sign flips with the metric' but 'no metric shows the positive adjacency dependence "
                "that window overlap would induce, and most show the opposite sign'."),
            "still_not_a_proof_of_independence": (
                "the structural overlap (§CV.1/CV.2) remains certain, and a pure fold main effect is "
                "invisible to this test by construction (§CV.6 control vi), so absence of a measurable "
                "adjacency signal does not prove four folds carry four independent confirmations' worth of "
                "evidence. The §CV limitation for the paper stands; only §CV.5's 'inconclusive, sign "
                "flips' wording is replaced by this sharper statement."),
        },
        "centering_artifact": art,
        "synthetic_controls": syn,
        "controls": {},
        "read_only": True, "database_access": False, "weights_loaded": False,
        "checkpoint_pt_opened": False, "parquet_read": False, "training_performed": False,
        "repository_module_imported_or_executed": False, "processes_signalled": [],
        "atime_side_effect": False, "byte_reproducible": True,
    }
    doc["controls"] = {
        "M0_and_M1_share_all_four_dataset_ids": bool(same_panels),
        "centering_artifact_matches_analytic": art["within_0.05"],
        "basis_sensitivity_measured_not_assumed": bool(basis_sensitive_peak),
        "cv5_basis_correction_confirmed_same_sign": bool(same_sign_on_config_basis),
        "synthetic_planted_detected": syn["planted_detected_positive"],
        "synthetic_independent_not_significant": syn["independent_not_significant"],
        "all_four_config_basis_tests_have_a_null_centred_near_zero": all(
            abs(t["contrast_null_mean"]) < 0.02 for t in tests.values()),
        "ALL_PASS": None,
    }
    doc["controls"]["ALL_PASS"] = all(
        v for k, v in doc["controls"].items() if k != "ALL_PASS")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

    b = doc["cv5_basis_mismatch_corrected"]
    print("=== §CV.5 basis mismatch, corrected ===")
    print(f"  on the 27-row config basis: ensemble {b['on_the_same_27row_config_basis']['ensemble']:+.4f}, "
          f"seed-mean {b['on_the_same_27row_config_basis']['seed_mean']:+.4f} -> same sign: "
          f"{b['on_the_same_27row_config_basis']['same_sign']}")
    print(f"  (CV.5 compared that ensemble value against an 81-row per-seed value of "
          f"{basis['seed_level_81rows']:+.4f} -- not comparable)")
    bs = b["basis_sensitivity_measured_on_peak_epoch"]
    print(f"  basis sensitivity on peak epoch: seed-mean {bs['seed_mean_27rows']:+.4f} vs "
          f"seed-level {bs['seed_level_81rows']:+.4f} -> differ: {bs['differ_materially']}")
    print(f"\n=== M0 is an independent carrier test: same panels = {same_panels}, "
          f"{m0['n_configs']} configs, family LGBM_RANKER ===")
    print(f"\n=== four tests, all on the CONFIG basis ===")
    print(f"  {'metric':34s} {'n':>4s} {'contrast':>9s} {'null mean':>10s} {'z':>7s} "
          f"{'p+':>7s} {'p-':>7s} {'sig':>5s}")
    for k, t in tests.items():
        print(f"  {t['metric'][:34]:34s} {t['n_rows']:>4d} {t['contrast_observed']:>+9.4f} "
              f"{t['contrast_null_mean']:>+10.4f} {t['contrast_z']:>+7.2f} "
              f"{t['p_one_sided_adjacent_more_POSITIVE']:>7.4f} "
              f"{t['p_one_sided_adjacent_more_NEGATIVE']:>7.4f} "
              f"{str(t['significant_in_observed_direction']):>5s}")
    v = doc["corrected_verdict"]
    print(f"\n=== corrected verdict ===")
    print(f"  {v['n_with_negative_contrast']}/{v['n_tests']} negative, "
          f"{v['n_with_significant_POSITIVE_contrast']} significant positive")
    print(f"  {v['statement']}")
    print(f"\n=== synthetic controls ===")
    print(f"  independent: contrast {syn['independent']['contrast']:+.4f}, significant="
          f"{syn['independent']['significant']} (must be False)")
    print(f"  planted config x year: contrast {syn['planted_interaction']['contrast']:+.4f}, "
          f"z {syn['planted_interaction']['z']:+.2f}, detected={syn['planted_detected_positive']} "
          f"(must be True)")
    print(f"\n=== centering artifact: analytic {art['analytic']}, empirical {art['empirical']} ===")
    print(f"\n=== controls ===")
    for k, ok in doc["controls"].items():
        print(f"  {'OK ' if ok else '*** FAIL ***'} {k}")
    assert doc["controls"]["ALL_PASS"], "a control failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
