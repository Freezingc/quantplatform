#!/usr/bin/env python3
"""Read-only probe for §CV: the four walk-forward folds are NOT four independent datasets -- they are
four rolling 4-year windows with stride 1, sharing 75% of their training years pairwise, and one year
(2020) appears in all four. This is mandated by source, was never recorded in the pack, and it bounds how
much weight the pack's "stable on all four folds" criterion can carry.

THE STRUCTURAL FACT (certain, from source and directory names, no parquet read).
datasets/splits.py:58-65 builds the folds as
    for index, train_start_year in enumerate(range(2017, 2021), start=1):
        AnnualFold(train_years=tuple(range(train_start_year, train_start_year + 4)), ...)
i.e. a 4-year training window advancing by 1 year per fold. splits.py:121 then asserts
len(outer.train_years) != 4 raises, so the 4-year width is enforced. Consequences, verified against the
dataset directories' year= partition names (which is all this probe reads from the artefacts):
    F1 train 2017-2020, F2 2018-2021, F3 2019-2022, F4 2020-2023
    adjacent folds share 3 of 4 training years (75%, Jaccard 0.600)
    4 folds x 4 years = 16 fold-years covering only 7 distinct years -> reuse factor 16/7 = 2.286
    per-year fold membership: 2017:1, 2018:2, 2019:3, 2020:4, 2021:3, 2022:2, 2023:1
So 2020 is inside every fold's training set, and 2017 and 2023 are each in exactly one. The four folds
are four correlated views of seven years of data, not four independent replications.

WHY THIS MATTERS FOR THE PACK. §BO.9 set its own reliability standard as "only p < 1e-4 AND stable on
all four folds count as reliable", ledger D4 tells the paper writer to cite that standard, and
§BN/§CN/§CP all run ANOVAs with fold as a 4-level factor. None of them recorded that the four levels are
not independent samples. "Stable across all four folds" is therefore weaker evidence than it reads: with
75% pairwise training overlap, a result driven by one peculiar year can appear in three or four folds at
once. This section states the bound; it does NOT claim any published conclusion is wrong.

A CALIBRATION TRAP THAT WOULD MISLEAD ANYONE COMPUTING FOLD CORRELATIONS. Centering each config's four
fold values forces their sum to zero, which under complete independence already induces
E[corr] = -1/(k-1) = -1/3 = -0.3333 for k = 4 folds. A raw fold-correlation matrix therefore looks
"strongly negatively correlated" when there is no dependence at all. The permutation null used here
(shuffling fold labels within each config row, which preserves the centering structure) returns -0.327,
matching the analytic value, so every correlation in this probe is reported against that null rather than
against zero. Anyone comparing fold correlations to 0 is off by a third of a correlation unit.

THE EMPIRICAL TEST IS INCONCLUSIVE, AND THAT IS REPORTED AS THE RESULT. If window overlap induced
positive dependence, adjacent folds (3 shared years) should show HIGHER residual correlation than distant
folds (1-2 shared years). Testing the contrast (mean adjacent corr - mean non-adjacent corr) against the
permutation null on three metrics gives three answers that do not agree in sign:
    M1 ensemble validation RankIC (27 configs): contrast -0.3230, z = -2.74, one-sided p = 0.0024
        -> adjacent folds are MORE NEGATIVELY correlated, the opposite of what shared data predicts
    M1 per-seed validation RankIC (81 rows, config-blocked null): contrast +0.5510, z = +5.81
        -> adjacent folds are MORE POSITIVELY correlated, as shared data would predict
    M1 peak epoch, grid, mean over seeds (27 configs): contrast +0.1905, z = +1.02, not significant
So the sign of the effect flips with the metric, and the only metric with a blocked null available points
one way while the ensemble points the other. With 4 folds there are just 6 pairs and a 3-versus-3
contrast, so this test has little power and cannot settle the question. The honest conclusion is that
whether the overlap materially induces dependence is NOT DETERMINABLE from four folds; the structural
overlap is certain, its statistical consequence is not measurable here. Per ledger B1 no mechanism or
magnitude is asserted beyond what is measured.

CONTROLS: (i) the permutation null must reproduce the analytic -1/3 induced correlation (it returns
-0.327), which validates the null construction itself; (ii) the null must be centred at ~0 for the
contrast statistic, since the contrast is a difference of two averages of the same induced artifact;
(iii) a synthetic matrix with NO fold structure (independent normal per cell) must produce a
non-significant contrast, proving the test can say "no dependence"; (iv) a synthetic matrix WITH planted dependence must produce a
significant POSITIVE contrast, proving the test can detect the dependence the real data might have had.
Control (iv) failed on the first draft and the failure is instructive: the planted model used a SHARED
year effect, but a year effect common to all configs is a fold MAIN effect, constant across rows, so
per-row centering annihilates it and no test of this kind could ever detect it. What window overlap can
actually turn into fold correlation is a CONFIG x YEAR interaction -- config i responds differently to
each year, and folds sharing years inherit shared config-specific components. That is also precisely the
mechanism that weakens "stable on all four folds": a config suited to 2020 looks good in every fold
containing 2020, which is all four. The corrected control plants that structure and is detected
decisively. Controls (iii) and (iv) together make the inconclusive verdict informative rather than merely
absent: the instrument works in both directions, so "the three real metrics disagree" is a property of the
data, not of a broken test.

Reads only: splits.py as text, dataset directory names (year= partitions), M1 search-record JSONs, and M1
checkpoint directory names. No parquet (not even footers), no checkpoint.pt, no weight or tensor, no
repository module imported or executed, no database, no network, no writes to any artifact directory, no
signal to any process. Does not touch /tmp/walk_forward_cache_v1, so no atime side effect. Deterministic
and byte-reproducible (fixed RNG seeds; no live state is read).
"""
from __future__ import annotations

import glob
import itertools
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "measurements" / "fold_window_overlap_and_reuse.json"

FOLDS = ["f1", "f2", "f3", "f4"]
SEEDS = ["20260801", "20260802", "20260803"]
RNG_SEED = 20260919
N_PERM = 5000
# fold pair -> number of shared training years, from the design
SHARED = {(0, 1): 3, (1, 2): 3, (2, 3): 3, (0, 2): 2, (1, 3): 2, (0, 3): 1}
PAIRS = list(itertools.combinations(range(4), 2))


def design_from_source() -> dict:
    """Read the fold-window construction off splits.py as text, and confirm the 4-year assert."""
    path = REPO / "src/research_v2/datasets/splits.py"
    lines = path.read_text(encoding="utf-8").split("\n")
    loop = {n: l.strip() for n, l in enumerate(lines, 1)
            if "for index, train_start_year in enumerate" in l}
    window = {n: l.strip() for n, l in enumerate(lines, 1)
              if "train_years=tuple(range(train_start_year" in l}
    guard = {n: l.strip() for n, l in enumerate(lines, 1)
             if 'if len(outer.train_years) != 4' in l}
    return {"file": str(path.relative_to(REPO)), "loop_over_start_years": loop,
            "window_expression": window, "four_year_assert": guard,
            "conclusion": ("train_years = range(start, start+4) with start in range(2017,2021): a 4-year "
                           "window advancing 1 year per fold, and the 4-year width is asserted at "
                           "inner_folds(), so the overlap is mandated by design, not incidental")}


def windows_from_directories() -> dict:
    """Year partitions per fold, read from dataset DIRECTORY NAMES only (no parquet opened)."""
    rep = json.load(open(ART / "m1_full_v1" / "report.json", encoding="utf-8"))
    fold2ds = {f["fold_id"]: f["dataset_id"] for f in rep["folds"]}
    base = ART / "m1_full_v1" / "datasets"
    out = {}
    for fid, ds in sorted(fold2ds.items()):
        d = base / ds
        tr = sorted(p.name.replace("year=", "") for p in (d / "split=train").iterdir()
                    if p.name.startswith("year="))
        va = sorted(p.name.replace("year=", "") for p in (d / "split=validation").iterdir()
                    if p.name.startswith("year="))
        te = sorted(p.name.replace("year=", "") for p in (d / "split=test").iterdir()
                    if p.name.startswith("year="))
        out[fid] = {"dataset_id": ds[:16], "train_years": tr, "validation_years": va,
                    "test_years": te, "n_train_years": len(tr)}
    # cross-check the directories against the source-derived design
    expected = {f"F{i+1}": [str(y) for y in range(2017 + i, 2021 + i)] for i in range(4)}
    return {"per_fold": out,
            "matches_source_design": all(out[k]["train_years"] == expected[k] for k in expected),
            "source_design_expected": expected}


def overlap_structure(windows: dict) -> dict:
    per_fold = windows["per_fold"]
    pair_rows = []
    for (a, b) in PAIRS:
        ya = set(per_fold[f"F{a+1}"]["train_years"]); yb = set(per_fold[f"F{b+1}"]["train_years"])
        inter = sorted(ya & yb); union = sorted(ya | yb)
        pair_rows.append({"pair": f"F{a+1}-F{b+1}", "shared_years": inter, "n_shared": len(inter),
                          "jaccard": round(len(inter) / len(union), 4),
                          "pct_of_either_window": round(100 * len(inter) / len(ya), 1),
                          "fold_distance": b - a})
    all_years = Counter(y for f in per_fold.values() for y in f["train_years"])
    n_fold_years = sum(f["n_train_years"] for f in per_fold.values())
    return {
        "pairs": pair_rows,
        "per_year_fold_membership": dict(sorted(all_years.items())),
        "n_fold_years_total": n_fold_years,
        "n_distinct_years": len(all_years),
        "data_reuse_factor": round(n_fold_years / len(all_years), 4),
        "adjacent_share_pct": round(100 * 3 / 4, 1),
        "year_in_all_four_folds": sorted(y for y, c in all_years.items() if c == 4),
        "years_in_exactly_one_fold": sorted(y for y, c in all_years.items() if c == 1),
        "reading": ("the four folds are four overlapping views of seven distinct years, not four "
                    "independent datasets; one year is inside every fold's training set"),
    }


def residual_corr(M: np.ndarray) -> dict:
    R = M - M.mean(axis=1, keepdims=True)
    return {(a, b): float(np.corrcoef(R[:, a], R[:, b])[0, 1]) for a, b in PAIRS}


def contrast(c: dict) -> float:
    adj = [c[k] for k in PAIRS if SHARED[k] == 3]
    non = [c[k] for k in PAIRS if SHARED[k] < 3]
    return float(np.mean(adj) - np.mean(non))


def run_test(M: np.ndarray, label: str, blocked: bool = False, blocks: np.ndarray | None = None,
             n_perm: int = N_PERM, seed: int = RNG_SEED) -> dict:
    """Permutation-calibrated test of whether adjacent folds are more correlated than distant ones.

    The null shuffles fold labels so that any true fold-specific association is destroyed while the
    per-row centering artifact (-1/(k-1)) is preserved. With blocked=True the SAME fold permutation is
    applied to every row of a block, which is required when rows are clustered (3 seeds of one config).
    """
    rng = np.random.default_rng(seed)
    obs = residual_corr(M)
    obs_c = contrast(obs)
    null_corrs = {k: [] for k in PAIRS}
    null_c = []
    n_rows = M.shape[0]
    for _ in range(n_perm):
        P = M.copy()
        if blocked and blocks is not None:
            for b in np.unique(blocks):
                idx = np.where(blocks == b)[0]
                perm = rng.permutation(M.shape[1])
                P[idx] = P[idx][:, perm]
        else:
            for i in range(n_rows):
                P[i] = P[i][rng.permutation(M.shape[1])]
        c = residual_corr(P)
        for k in PAIRS:
            null_corrs[k].append(c[k])
        null_c.append(contrast(c))
    null_c = np.array(null_c)
    per_pair = {}
    for k in PAIRS:
        a = np.array(null_corrs[k])
        z = (obs[k] - a.mean()) / a.std()
        p = 2 * min((a <= obs[k]).mean(), (a >= obs[k]).mean())
        per_pair[f"F{k[0]+1}-F{k[1]+1}"] = {
            "shared_training_years": SHARED[k], "observed_corr": round(obs[k], 4),
            "null_mean": round(float(a.mean()), 4), "null_sd": round(float(a.std()), 4),
            "z": round(float(z), 3), "two_sided_p": round(max(float(p), 1 / n_perm), 4)}
    # ONE-SIDED p VALUES, BOTH DIRECTIONS, with the labels matching the arithmetic.
    #
    # THE FIRST DRAFT HAD THESE BACKWARDS, and the planted-dependence control is what caught it. It
    # computed p = P(null <= observed) and labelled it "adjacent more positive". For a large positive
    # observed contrast that probability is ~1, so a strongly detected effect was reported as p = 1.0000
    # and the control "failed" while the instrument was in fact working perfectly. The correct one-sided
    # p for the alternative "adjacent folds are MORE positively correlated than the null" is
    # P(null >= observed). Both directions are now reported so no reader has to trust a label, and the
    # detection predicates use the direction that matches the planted sign.
    p_more_positive = float((null_c >= obs_c).mean())
    p_more_negative = float((null_c <= obs_c).mean())
    return {
        "metric": label, "n_rows": n_rows, "blocked_null": blocked,
        "per_pair": per_pair,
        "contrast_observed": round(obs_c, 4),
        "contrast_null_mean": round(float(null_c.mean()), 4),
        "contrast_null_sd": round(float(null_c.std()), 4),
        "contrast_z": round(float((obs_c - null_c.mean()) / null_c.std()), 3),
        "p_one_sided_adjacent_more_POSITIVE": round(max(p_more_positive, 1 / n_perm), 4),
        "p_one_sided_adjacent_more_NEGATIVE": round(max(p_more_negative, 1 / n_perm), 4),
        "p_value_direction_note": ("P(null >= obs) tests 'adjacent more positively correlated'; "
                                   "P(null <= obs) tests 'adjacent more negatively correlated'. An "
                                   "earlier draft swapped these, which made a decisively detected "
                                   "planted effect read as p = 1.0."),
        "direction": ("adjacent MORE positively correlated (as shared data would predict)"
                      if obs_c > 0 else
                      "adjacent MORE NEGATIVELY correlated (opposite of what shared data predicts)"),
        "significant_in_the_direction_observed": bool(
            (p_more_positive if obs_c > 0 else p_more_negative) < 0.05),
    }


def synthetic_controls() -> dict:
    """(iii) no-dependence synthetic must be non-significant; (iv) planted dependence must be detected.

    CONTROL (iv) WAS WRONG ON THE FIRST DRAFT, AND FIXING IT IS THE MOST INFORMATIVE PART OF THIS PROBE.
    The first version planted a shared YEAR effect: fold value = sum of the effects of its four training
    years + noise, with the year effects common to all configs. That produced contrast -0.14, p = 0.15 --
    the test FAILED TO DETECT its own planted dependence. The reason is structural, not bad luck: a year
    effect common to every config is a FOLD MAIN EFFECT, constant across rows, so after per-row centering
    it contributes zero across-row variance and cannot create any fold-to-fold correlation. Centering
    removes exactly that component by construction.

    What overlap CAN turn into fold correlation is a CONFIG x YEAR interaction: config i has its own
    response to each year, and two folds sharing years inherit shared config-specific components. That is
    also the mechanism that makes "stable on all four folds" weaker evidence than it reads -- a config
    that happens to suit 2020 looks good in EVERY fold containing 2020, which is all four of them. The
    corrected control plants exactly that, with the real design's overlap (years[a:a+4] over 2017-2023),
    and it is detected decisively (contrast about +0.72 against a null sd near 0.12).
    """
    rng = np.random.default_rng(11)
    years = list(range(2017, 2024))

    # (iii) independent cells: no fold structure at all
    M_ind = rng.normal(size=(27, 4))
    t_ind = run_test(M_ind, "SYNTHETIC independent (no fold structure)", n_perm=2000, seed=5)

    # (iv-a) the WRONG planted model, kept in the output as a recorded negative: a shared year effect is a
    # fold main effect and is annihilated by per-row centering, so it is undetectable BY DESIGN.
    eff = rng.normal(size=len(years))
    M_shared_year = np.zeros((27, 4))
    for i in range(27):
        for a in range(4):
            M_shared_year[i, a] = sum(eff[years.index(y)] for y in years[a:a + 4]) \
                + rng.normal(scale=0.5)
    t_shared = run_test(M_shared_year, "SYNTHETIC shared year effect (fold main effect only)",
                        n_perm=2000, seed=6)

    # (iv-b) the CORRECT planted model: config x year interaction, overlapping as in the real design
    gain = rng.normal(size=(27, len(years)))
    M_dep = np.zeros((27, 4))
    for i in range(27):
        for a in range(4):
            M_dep[i, a] = (sum(gain[i, years.index(y)] for y in years[a:a + 4])
                           + float(rng.normal(scale=0.5)))
    t_dep = run_test(M_dep, "SYNTHETIC config x year interaction (overlap can induce dependence)",
                     n_perm=2000, seed=7)

    return {
        "independent_synthetic": {"contrast": t_ind["contrast_observed"], "z": t_ind["contrast_z"],
                                  "p_more_positive": t_ind["p_one_sided_adjacent_more_POSITIVE"],
                                  "p_more_negative": t_ind["p_one_sided_adjacent_more_NEGATIVE"]},
        "shared_year_effect_synthetic_UNDETECTABLE_BY_DESIGN": {
            "contrast": t_shared["contrast_observed"], "z": t_shared["contrast_z"],
            "p_more_positive": t_shared["p_one_sided_adjacent_more_POSITIVE"],
            "p_more_negative": t_shared["p_one_sided_adjacent_more_NEGATIVE"],
            "why_undetectable": ("a year effect common to all configs is constant across rows, i.e. a "
                                 "fold main effect; per-row centering removes it, so it cannot create "
                                 "fold-to-fold correlation and no test of this kind could detect it"),
        },
        "config_year_interaction_synthetic": {
            "contrast": t_dep["contrast_observed"], "z": t_dep["contrast_z"],
            "p_more_positive": t_dep["p_one_sided_adjacent_more_POSITIVE"],
            "p_more_negative": t_dep["p_one_sided_adjacent_more_NEGATIVE"]},
        "independent_is_nonsignificant": bool(
            min(t_ind["p_one_sided_adjacent_more_POSITIVE"],
                t_ind["p_one_sided_adjacent_more_NEGATIVE"]) > 0.05),
        "planted_dependence_is_detected": bool(
            t_dep["p_one_sided_adjacent_more_POSITIVE"] < 0.05
            and t_dep["contrast_observed"] > 0),
        "shared_year_effect_is_NOT_detected_as_expected": not bool(
            t_dep["p_one_sided_adjacent_more_POSITIVE"] < 0.05
            and t_shared["contrast_observed"] > 0
            and min(t_shared["p_one_sided_adjacent_more_POSITIVE"],
                    t_shared["p_one_sided_adjacent_more_NEGATIVE"]) < 0.05),
        "reading": ("the instrument detects overlap-driven dependence when the dependence is of the kind "
                    "overlap can produce (config x year interaction) and reports no dependence on "
                    "independent data, so the real-data verdict of 'inconclusive, sign flips with the "
                    "metric' is a property of the data rather than a broken test. It also shows that a "
                    "pure fold main effect is invisible to this test by construction, which bounds what "
                    "the test could ever have shown."),
    }


def centering_artifact_check() -> dict:
    """The permutation null must reproduce the analytic -1/(k-1) induced correlation."""
    rng = np.random.default_rng(3)
    M = rng.normal(size=(400, 4))
    c = residual_corr(M)
    means = [v for v in c.values()]
    analytic = -1.0 / (4 - 1)
    return {"analytic_minus_one_over_k_minus_1": round(analytic, 4),
            "empirical_mean_pairwise_corr_of_independent_data": round(float(np.mean(means)), 4),
            "within_0.05_of_analytic": abs(float(np.mean(means)) - analytic) < 0.05,
            "reading": ("per-row centering of k=4 fold values forces the row sum to zero, which induces "
                        "E[corr] = -1/3 under complete independence; fold correlations must never be "
                        "compared against 0")}


def load_metrics() -> dict:
    ic = {}
    for f in ("F1", "F2", "F3", "F4"):
        for rec in sorted((ART / "m1_full_v1" / "search" / f).glob("c*.json")):
            d = json.load(open(rec, encoding="utf-8"))
            r = d["result"]
            ic[(f"c{d['config_index']}", f.lower())] = r["ensemble_validation_rank_ic"]
            for s, v in r["seed_validation_rank_ic"].items():
                ic[(f"c{d['config_index']}", s, f.lower())] = v
    cfgs = sorted({k[0] for k in ic if len(k) == 2}, key=lambda c: int(c[1:]))
    M_ens = np.array([[ic[(c, f)] for f in FOLDS] for c in cfgs], dtype=float)

    sic = {k: v for k, v in ic.items() if len(k) == 3}
    rows = sorted({(k[0], k[1]) for k in sic}, key=lambda t: (int(t[0][1:]), t[1]))
    M_seed = np.array([[sic[(c, s, f)] for f in FOLDS] for c, s in rows], dtype=float)
    blocks = np.array([cfgs.index(c) for c, _ in rows])

    run_re = re.compile(r"^(?P<run>.+)-epoch-(?P<epoch>\d+)$")
    runs: dict[str, list[int]] = defaultdict(list)
    for d in glob.glob(str(ART / "m1_full_v1" / "checkpoints" / "*-epoch-*")):
        m = run_re.match(Path(d).name)
        if m:
            runs[m.group("run")].append(int(m.group("epoch")))
    pk: dict[tuple[str, str], list[int]] = defaultdict(list)
    for r, e in runs.items():
        g = re.match(r"^m1-(f\d)-(c\d+)-\d+-seed-(\d+)$", r)
        if g:
            pk[(g.group(2), g.group(1))].append(max(e))
    M_pk = np.array([[float(np.mean(pk[(c, f)])) for f in FOLDS] for c in cfgs], dtype=float)
    return {"configs": cfgs, "M_ensemble": M_ens, "M_per_seed": M_seed, "seed_blocks": blocks,
            "M_peak": M_pk, "n_seed_rows": len(rows)}


def main() -> None:
    design = design_from_source()
    windows = windows_from_directories()
    overlap = overlap_structure(windows)
    art = centering_artifact_check()
    syn = synthetic_controls()

    met = load_metrics()
    t_ens = run_test(met["M_ensemble"], "M1 ensemble validation RankIC (27 configs)")
    t_seed = run_test(met["M_per_seed"], "M1 per-seed validation RankIC (81 rows)",
                      blocked=True, blocks=met["seed_blocks"])
    t_peak = run_test(met["M_peak"], "M1 peak epoch, grid, mean over 3 seeds (27 configs)")

    signs = {"ensemble": np.sign(t_ens["contrast_observed"]),
             "per_seed": np.sign(t_seed["contrast_observed"]),
             "peak_epoch": np.sign(t_peak["contrast_observed"])}
    all_same_sign = len(set(signs.values())) == 1

    doc = {
        "question": ("are the four walk-forward folds independent datasets, and does their training-window "
                     "overlap materially induce dependence between folds?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "design_from_source": design,
        "windows_from_directories": windows,
        "overlap_structure": overlap,
        "centering_artifact": art,
        "empirical_tests": {"ensemble_rank_ic": t_ens, "per_seed_rank_ic": t_seed,
                            "peak_epoch": t_peak,
                            "contrast_signs": {k: float(v) for k, v in signs.items()},
                            "all_three_agree_in_sign": bool(all_same_sign),
                            "verdict": (
                                "INCONCLUSIVE: the contrast's sign flips with the metric (ensemble "
                                f"{t_ens['contrast_observed']:+.4f}, per-seed "
                                f"{t_seed['contrast_observed']:+.4f}, peak epoch "
                                f"{t_peak['contrast_observed']:+.4f}), so four folds cannot settle "
                                "whether the overlap induces dependence. The structural overlap is "
                                "certain; its statistical consequence is not measurable here."
                                if not all_same_sign else
                                "the three metrics agree in sign; see the per-metric tables")},
        "synthetic_controls": syn,
        "implication_for_the_pack": {
            "criterion_affected": ("§BO.9's self-set reliability standard 'only p<1e-4 AND stable on all "
                                   "four folds count as reliable', cited to paper writers by ledger D4, "
                                   "and the fold-as-4-level-factor ANOVAs in §BN/§CN/§CP"),
            "what_must_be_stated": ("'stable across all four folds' is not four independent "
                                    "confirmations: adjacent folds share 75% of training years, the reuse "
                                    "factor is 2.286, and 2020 is in every fold's training set, so a "
                                    "result driven by one peculiar year can appear in three or four folds "
                                    "at once. The effective number of independent folds is less than 4."),
            "does_not_claim": ("no published conclusion is shown to be wrong by this; it is a bound on "
                               "how much weight the four-fold criterion can carry, and a limitation to "
                               "state in the paper"),
        },
        "read_only": True, "database_access": False, "weights_loaded": False,
        "checkpoint_pt_opened": False, "parquet_read": False, "training_performed": False,
        "repository_module_imported_or_executed": False, "processes_signalled": [],
        "atime_side_effect": False, "byte_reproducible": True,
    }
    doc["controls"] = {
        "windows_match_the_source_design": windows["matches_source_design"],
        "all_four_windows_are_4_years": all(
            f["n_train_years"] == 4 for f in windows["per_fold"].values()),
        "centering_artifact_matches_analytic": art["within_0.05_of_analytic"],
        "synthetic_independent_is_nonsignificant": syn["independent_is_nonsignificant"],
        "synthetic_planted_dependence_is_detected": syn["planted_dependence_is_detected"],
        "synthetic_fold_main_effect_is_invisible_as_expected":
            syn["shared_year_effect_is_NOT_detected_as_expected"],
        "null_contrast_centred_near_zero": all(
            abs(t["contrast_null_mean"]) < 0.02 for t in (t_ens, t_seed, t_peak)),
        "ALL_PASS": None,
    }
    doc["controls"]["ALL_PASS"] = all(
        v for k, v in doc["controls"].items() if k != "ALL_PASS")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1, default=float)

    print("=== design from source ===")
    for k, v in design.items():
        if k != "file":
            print(f"  {k}: {v}")
    print(f"\n=== windows from directory names (match source design: "
          f"{windows['matches_source_design']}) ===")
    for f, v in windows["per_fold"].items():
        print(f"  {f}: train {v['train_years'][0]}-{v['train_years'][-1]} "
              f"({v['n_train_years']} yrs) | val {v['validation_years']} | test {v['test_years']}")
    o = overlap
    print(f"\n=== overlap structure ===")
    for p in o["pairs"]:
        print(f"  {p['pair']}: shared {p['n_shared']} yrs {p['shared_years']} "
              f"Jaccard {p['jaccard']} ({p['pct_of_either_window']}% of a window)")
    print(f"  fold-years {o['n_fold_years_total']} over {o['n_distinct_years']} distinct years "
          f"-> reuse factor {o['data_reuse_factor']}")
    print(f"  per-year membership: {o['per_year_fold_membership']}")
    print(f"  in all four folds: {o['year_in_all_four_folds']} | in exactly one: "
          f"{o['years_in_exactly_one_fold']}")
    a = art
    print(f"\n=== centering artifact ===")
    print(f"  analytic -1/(k-1) = {a['analytic_minus_one_over_k_minus_1']}; empirical "
          f"{a['empirical_mean_pairwise_corr_of_independent_data']} "
          f"(matches: {a['within_0.05_of_analytic']})")
    print(f"\n=== empirical tests (contrast = mean adjacent corr - mean non-adjacent corr) ===")
    for label, t in (("ensemble RankIC", t_ens), ("per-seed RankIC", t_seed), ("peak epoch", t_peak)):
        print(f"  {label:18s} contrast {t['contrast_observed']:+.4f} | null {t['contrast_null_mean']:+.4f}"
              f" +/- {t['contrast_null_sd']:.4f} | z {t['contrast_z']:+.2f}")
        print(f"    {'':18s} p(adjacent MORE positive) {t['p_one_sided_adjacent_more_POSITIVE']:.4f}"
              f" | p(adjacent MORE negative) {t['p_one_sided_adjacent_more_NEGATIVE']:.4f}"
              f" | significant in observed direction: {t['significant_in_the_direction_observed']}")
        print(f"    {'':18s} -> {t['direction']}")
    s = syn
    print(f"\n=== synthetic controls ===")
    ind = s["independent_synthetic"]
    print(f"  independent (no fold structure): contrast {ind['contrast']:+.4f}, "
          f"p+ {ind['p_more_positive']:.4f} / p- {ind['p_more_negative']:.4f} -> nonsignificant: "
          f"{s['independent_is_nonsignificant']}")
    sh = s["shared_year_effect_synthetic_UNDETECTABLE_BY_DESIGN"]
    print(f"  shared year effect (fold main effect): contrast {sh['contrast']:+.4f}, "
          f"p+ {sh['p_more_positive']:.4f} / p- {sh['p_more_negative']:.4f} -> invisible as expected: "
          f"{s['shared_year_effect_is_NOT_detected_as_expected']}")
    pd_ = s["config_year_interaction_synthetic"]
    print(f"  config x year interaction (what overlap CAN induce): contrast {pd_['contrast']:+.4f}, "
          f"z {pd_['z']:+.2f}, p+ {pd_['p_more_positive']:.4f} -> detected: "
          f"{s['planted_dependence_is_detected']}")
    print(f"\n=== VERDICT ===")
    print(f"  {doc['empirical_tests']['verdict']}")
    print(f"\n=== controls ===")
    for k, v in doc["controls"].items():
        print(f"  {'OK ' if v else '*** FAIL ***'} {k}")
    assert doc["controls"]["ALL_PASS"], "a control failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
