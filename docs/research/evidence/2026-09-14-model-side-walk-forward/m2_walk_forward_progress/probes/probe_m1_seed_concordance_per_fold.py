#!/usr/bin/env python3
"""Read-only probe for §CO: §CN.3's blockability verdict used the wrong basis, and correcting it
changes the verdict.

WHAT §CN.3 CLAIMED AND WHY IT IS WRONG
§CN.3 counted, over all 27 configs x 4 folds = 108 (config, fold) cells POOLED, which seed had the
longest peak; the modal ordering 03>02>01 covered 49.1% of cells, and from that it concluded seed is
"NOT a clean blocking factor for peak epoch", contrasting this with §BN.2, where seed x fold blocking
IS valid on validation RankIC. The contrast is not valid, because the two figures are on different
bases. §BN.2's blocking structure is PER FOLD (seed x fold = 12 blocks, and its table reports the
worst_seed census separately for F1, F2, F3, F4). §CN.3 pooled the four folds into one count -- the
very operation §BN.2 warns against, since §BN.2 established that the seed DIRECTION flips across
folds on RankIC, so pooling cancels real within-fold agreement. §CN.3 even cited §BN.2 in the same
paragraph whose statistic violated §BN.2's lesson.

THE CORRECTED MEASUREMENT
Kendall's W of concordance, computed PER FOLD with the 3 seeds as items and the 27 configs as raters,
on both dependent variables, alongside the pooled figure so the damage from pooling is visible:

    basis      W(peak)     p        W(RankIC)    p
    F1         0.7514   1.55e-09    0.7833   6.54e-10
    F2         0.6245   4.76e-08    0.6214   5.17e-08
    F3         0.1307   2.94e-02    0.3512   7.63e-05
    F4         0.2315   1.93e-03    0.4705   3.04e-06
    POOLED     0.1444   1.69e-07    0.0442   8.41e-03   <- the wrong basis

Pooling destroys concordance on BOTH variables (peak 0.75/0.62 -> 0.14; RankIC 0.78/0.62 -> 0.04),
which is itself the demonstration that the pooled figure measures fold-to-fold direction reversal
rather than within-fold seed agreement. The same shows in the modal-share terms: per fold the modal
ordering covers 77.8% / 63.0% / 55.6% / 29.6% of the 27 configs, against §CN.3's pooled 49.1%.

THE CORRECTED VERDICT is not the negation of §CN.3's but a sharpening of it, and it differs by fold:
  * On F1 and F2, seed IS a usable blocking factor for peak epoch, about as usable as for RankIC
    (W 0.75 vs 0.78, and 0.62 vs 0.62 -- the second pair is indistinguishable to two decimals).
  * On F3 and F4, seed concordance on peak epoch is much weaker than on RankIC (0.13 vs 0.35, and
    0.23 vs 0.47). The modal ordering is significantly above chance on ALL four folds once tied cells
    are excluded (79.2% / 65.4% / 38.9% / 33.3%, binomial p = 2.95e-11 / 3.97e-08 / 2.06e-02 /
    3.54e-02), so the seed effect is real everywhere; what varies is how consistently it orders the
    seeds within a fold. So §CN.3's conclusion fails for F1/F2 and holds in spirit for F3/F4, and a
    single verdict for "peak epoch" was never available -- the answer is fold-specific.

    NOTE ON TIES: peak epoch is a small integer, so 16 of the 108 (config, fold) cells have two or
    three seeds tied (RankIC, being continuous, has none). np.argsort breaks ties inconsistently --
    argsort([5,5,3])[::-1] is (1,0,2) while argsort([-5,-5,-3]) is (0,1,2) -- so a modal-ordering
    count that keeps tied cells depends on which of two algebraically equivalent expressions was
    written, and it moved the F1 share between 70.4% and 79.2%. Tied cells are therefore EXCLUDED and
    the excluded count is reported per fold, which is also why the shares here differ from the
    77.8/63.0/55.6/29.6% a tie-inclusive count gives.

A NEW FINDING THAT FALLS OUT: the seed direction on peak epoch and on RankIC AGREE on F1 and F2
(both 03 > 02 > 01) and DISAGREE on F3 and F4 (peak 03>02>01 vs RankIC 01>03>02 on F3; peak
01>03>02 vs RankIC 01>02>03 on F4). So the dissociation §CN.3 reported is not merely that seed moves
peak location more than it moves peak quality -- on two of four folds the seed that trains LONGEST is
not the seed that scores BEST, and on F3 the ordering is even reversed in its middle. That is a
sharper statement than §CN.3 made and it matters for interpreting any single-seed training-dynamics
observation.

Only checkpoint directory names (for peaks) and already-written search records (for RankIC) are read.
No parquet at all, no checkpoint.pt, no weight or tensor, no repository module imported or executed,
no database access, no network, no writes to any artifact directory, no signal to any process. M1's
run finished long ago, so nothing here is a partial snapshot; M2 (pid 3588838) is untouched.
"""
from __future__ import annotations

import glob
import json
import re
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "measurements" / "m1_seed_concordance_per_fold.json"

FOLDS = ["f1", "f2", "f3", "f4"]
SEEDS = ["20260801", "20260802", "20260803"]
CHANCE = 1.0 / 6.0


def kendall_w(ranks: np.ndarray) -> dict:
    """Kendall's coefficient of concordance.

    ranks has shape (n_items, n_raters); each COLUMN is one rater's ranking of the items. Getting this
    orientation right matters: the first attempt at this analysis built the matrix the other way round
    (one row per config, rankdata applied across that row's 3 seeds), which makes every row a
    permutation of (1,2,3), so every row sums to 6, so S = 0 and W = 0 identically -- and it reported
    W = 0.0000 with p = 1.0 for all four folds on both variables, which looked like a finding
    ("no agreement anywhere") rather than an indexing error. Items must be the rows.
    """
    n, k = ranks.shape
    row_sums = ranks.sum(axis=1)
    S = float(((row_sums - row_sums.mean()) ** 2).sum())
    W = 12.0 * S / (k ** 2 * (n ** 3 - n))
    chi2 = k * (n - 1) * W
    return {"W": round(W, 4), "chi2": round(chi2, 4), "p": float(stats.chi2.sf(chi2, n - 1)),
            "n_items": n, "n_raters": k}


def kendall_w_negative_controls() -> dict:
    """The W instrument must reach ~1 on unanimous input and ~0 on random input."""
    unanimous = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=float)
    rng = np.random.default_rng(7)
    random_ranks = np.array([rng.permutation(3) + 1 for _ in range(200)]).T.astype(float)
    w_un = kendall_w(unanimous)["W"]
    w_rnd = kendall_w(random_ranks)["W"]
    # and the orientation error must be detectable: transposing unanimous gives W = 1 too (3 raters,
    # 3 items, still unanimous), so instead check the SHAPE guard that caught the original bug
    wrong_orientation = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]], dtype=float)
    return {
        "unanimous_ranking_W_must_be_1": w_un,
        "unanimous_ok": abs(w_un - 1.0) < 1e-9,
        "random_200_raters_W_must_be_near_0": w_rnd,
        "random_ok": w_rnd < 0.05,
        "transposed_unanimous_W": kendall_w(wrong_orientation)["W"],
        "note_on_orientation": ("every row a permutation of (1,2,3) forces all row sums equal and hence "
                                "W = 0 regardless of the data; that is the failure mode the first "
                                "attempt hit and why items must be rows"),
        "ALL_OK": bool(abs(w_un - 1.0) < 1e-9 and w_rnd < 0.05),
    }


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
            assert g.group(3) == g.group(4)
            out[(g.group(2), g.group(1), g.group(4))] = max(e)
    return out


def load_rank_ic(ws: str) -> dict[tuple[str, str, str], float]:
    out: dict[tuple[str, str, str], float] = {}
    for f in ("F1", "F2", "F3", "F4"):
        d = ART / ws / "search" / f
        if not d.is_dir():
            continue
        for rec in sorted(d.glob("c*.json")):
            r = json.load(open(rec, encoding="utf-8"))
            for seed, v in r["result"]["seed_validation_rank_ic"].items():
                out[(f"c{r['config_index']}", f.lower(), seed)] = float(v)
    return out


def ordering_of(values: list[float]) -> tuple[int, ...] | None:
    """Indices of the 3 seeds ordered from LARGEST value to smallest, or None if any tie.

    Ties are excluded rather than broken, because np.argsort does not break them consistently:
    argsort([5,5,3])[::-1] is (1,0,2) while argsort([-5,-5,-3]) is (0,1,2), so a modal-ordering count
    that keeps tied cells depends on which of two algebraically equivalent expressions was written.
    16 of M1's 108 (config, fold) cells have a tie on peak epoch (RankIC has none, being continuous),
    so this is not a corner case -- it moved the F1 modal share between 70.4% and 79.2% depending on
    the tie order. Excluding tied cells makes the statistic well-defined, and the excluded count is
    reported so the exclusion is visible rather than silent.
    """
    if len(set(values)) != len(values):
        return None
    return tuple(int(x) for x in np.argsort(values)[::-1])   # LARGEST first, both variables


def concordance(cfgs: list[str], folds: list[str], getter, label: str) -> dict:
    """Per-fold and pooled seed concordance for one dependent variable.

    ONE convention for both variables: ranks and orderings always run from the LARGEST value down, so
    for peak epoch rank 1 is the LONGEST run and for RankIC rank 1 is the BEST score. The first draft
    of this probe applied sign=-1 for peak and sign=+1 for RankIC and then reversed argsort for both,
    which made the two variables' ordering labels mean opposite things -- undetectable in W (which is
    invariant to reversing all ranks) but fatal to any sentence comparing the two modal orderings. The
    convention is now stated once and used identically.
    """
    per_fold = {}
    for f in folds:
        vals = {s: [getter((c, f, s)) for c in cfgs] for s in SEEDS}
        # Kendall's W: items = 3 seeds (rows), raters = 27 configs (columns), rank within each rater
        cols = [stats.rankdata([-getter((c, f, s)) for s in SEEDS], method="average") for c in cfgs]
        w = kendall_w(np.array(cols).T)
        # modal ordering on TIE-FREE cells only
        orders = [ordering_of([getter((c, f, s)) for s in SEEDS]) for c in cfgs]
        tie_free = [o for o in orders if o is not None]
        n_tied = len(orders) - len(tie_free)
        counts = Counter(tie_free)
        modal, n_modal = counts.most_common(1)[0]
        binom = stats.binomtest(n_modal, len(tie_free), CHANCE, alternative="greater")
        per_fold[f] = {
            "kendall_w": w,
            "n_configs": len(cfgs),
            "n_cells_excluded_for_ties": n_tied,
            "n_tie_free_cells": len(tie_free),
            "ordering_distribution": [
                {"order_largest_first": [SEEDS[i][-2:] for i in pat], "n_configs": cnt,
                 "share_of_tie_free_pct": round(100 * cnt / len(tie_free), 1)}
                for pat, cnt in counts.most_common()],
            "modal_order_largest_first": [SEEDS[i][-2:] for i in modal],
            "modal_n_configs": n_modal,
            "modal_share_pct": round(100 * n_modal / len(tie_free), 1),
            "chance_share_pct": round(100 * CHANCE, 1),
            "binomial_p_modal_exceeds_chance": float(binom.pvalue),
            "modal_share_significant_at_0.05": bool(binom.pvalue < 0.05),
            "seed_means_largest_first": {
                SEEDS[i][-2:]: round(float(st.mean(vals[SEEDS[i]])), 4) for i in range(len(SEEDS))},
            "direction_largest_first": [
                SEEDS[i][-2:] for i in np.argsort(
                    [-float(st.mean(vals[s])) for s in SEEDS])],
        }

    cells = [(c, f) for c in cfgs for f in folds]
    cols = [stats.rankdata([-getter((c, f, s)) for s in SEEDS], method="average") for c, f in cells]
    pooled_w = kendall_w(np.array(cols).T)
    orders = [ordering_of([getter((c, f, s)) for s in SEEDS]) for c, f in cells]
    tie_free = [o for o in orders if o is not None]
    counts = Counter(tie_free)
    modal, n_modal = counts.most_common(1)[0]
    return {
        "variable": label,
        "convention": "rank 1 and 'largest first' always mean the BIGGEST value (longest peak / best IC)",
        "per_fold": per_fold,
        "POOLED_wrong_basis": {
            "kendall_w": pooled_w,
            "n_cells": len(cells),
            "n_cells_excluded_for_ties": len(cells) - len(tie_free),
            "modal_order_largest_first": [SEEDS[i][-2:] for i in modal],
            "modal_n_cells": n_modal,
            "modal_share_pct": round(100 * n_modal / len(tie_free), 1),
            "binomial_p_modal_exceeds_chance": float(
                stats.binomtest(n_modal, len(tie_free), CHANCE, alternative="greater").pvalue),
            "why_wrong": ("pooling folds whose seed directions differ cancels real within-fold "
                          "agreement; this is exactly the operation §BN.2 warns against when it "
                          "establishes seed x fold -- not seed alone -- as the correct blocking"),
        },
    }


def quantization_control(cfgs: list[str], folds: list[str], peak: dict, ic: dict,
                         n_draws: int = 200) -> dict:
    """Is the low W on peak epoch for F3/F4 an artefact of peak being a small integer?

    W measures rank concordance, and a coarse response produces ties, which depress it. Peak epoch is
    a small integer (18-26 distinct values per fold) while RankIC is continuous, so a low W on peak
    could in principle be a measurement artefact rather than a property of the seed effect. The
    control: quantize the CONTINUOUS variable down to the number of distinct levels the peak variable
    has in that fold and see whether its W falls toward the peak W. If quantization were the cause,
    quantized RankIC would lose most of its concordance on exactly the folds where peak loses it. It
    does not, so the gap between peak and RankIC on F3/F4 is a real property of the seed effect on peak
    epoch rather than of integer-valuedness.

    Each draw adds negligible jitter before binning so the digitisation boundary is not systematically
    aligned with the data, and cells that become tied after quantization are excluded exactly as they
    are for peak, so the two variables are treated identically.
    """
    rng = np.random.default_rng(11)
    per_fold = {}
    for f in folds:
        pv = [peak[(c, f, sd)] for c in cfgs for sd in SEEDS]
        n_levels = len(set(pv))
        wp = kendall_w(np.array(
            [stats.rankdata([-peak[(c, f, sd)] for sd in SEEDS]) for c in cfgs]).T)["W"]
        wi = kendall_w(np.array(
            [stats.rankdata([-ic[(c, f, sd)] for sd in SEEDS]) for c in cfgs]).T)["W"]
        draws, excluded = [], []
        for _ in range(n_draws):
            vals = np.array([ic[(c, f, sd)] for c in cfgs for sd in SEEDS])
            edges = np.quantile(vals, np.linspace(0, 1, n_levels + 1)[1:-1])
            q = np.digitize(vals + rng.normal(0, 1e-9, vals.size), edges) + 1
            qd = {(cfgs[i], SEEDS[j]): float(q[i * len(SEEDS) + j])
                  for i in range(len(cfgs)) for j in range(len(SEEDS))}
            cols, n_ex = [], 0
            for c in cfgs:
                v = [qd[(c, sd)] for sd in SEEDS]
                if len(set(v)) < len(SEEDS):
                    n_ex += 1
                    continue
                cols.append(stats.rankdata([-x for x in v]))
            excluded.append(n_ex)
            if cols:
                draws.append(kendall_w(np.array(cols).T)["W"])
        wq = float(np.mean(draws)) if draws else float("nan")
        per_fold[f] = {
            "n_distinct_peak_values": n_levels,
            "W_peak": wp, "W_rank_ic_raw": wi, "W_rank_ic_quantized_mean": round(wq, 4),
            "quantization_cost_W": round(wi - wq, 4),
            "n_draws": len(draws),
            "mean_cells_excluded_after_quantization": round(float(np.mean(excluded)), 2),
            "gap_peak_vs_rank_ic": round(wi - wp, 4),
            # The gap only needs explaining where it is material. On F2 peak and RankIC agree to two
            # decimals (gap -0.0031, i.e. peak is marginally HIGHER), so there is nothing to explain
            # and any fraction is meaningless -- dividing a tiny cost by a near-zero gap yields a huge
            # ratio out of nothing. The first draft of this predicate asked
            # `abs(cost) > abs(remaining gap)`, which is exactly that degenerate comparison and it
            # flagged F2 for "explaining" a gap that does not exist. The same failure class as §CN.6(ii)'s
            # flaky control: a test whose verdict is dominated by a quantity that is zero by
            # construction. So materiality is gated first, and the reported quantity is the FRACTION of
            # a material gap that quantization accounts for.
            "gap_is_material": bool((wi - wp) > 0.05),
            "fraction_of_material_gap_explained_by_quantization": (
                round((wi - wq) / (wi - wp), 4) if (wi - wp) > 0.05 else None),
        }
    explained = {f: v["fraction_of_material_gap_explained_by_quantization"]
                 for f, v in per_fold.items() if v["gap_is_material"]}
    return {
        "per_fold": per_fold,
        "folds_with_a_material_gap": sorted(explained),
        "max_fraction_of_gap_explained_by_quantization": (
            max(explained.values()) if explained else None),
        "reading": ("quantizing RankIC to peak's level count changes its W by at most a couple of "
                    "hundredths on every fold, while the peak-versus-RankIC gap on the two folds where "
                    "it is material (F3, F4) is 0.22 and 0.24. Quantization therefore accounts for only "
                    "a small fraction of those gaps, so they are a real property of the seed effect on "
                    "peak epoch and not a coarse-response artefact. Without this control, 'W is low on "
                    "peak for F3/F4' could not be distinguished from 'W is low because peak is an "
                    "integer'."),
        "quantization_explains_a_material_gap": bool(
            explained and max(explained.values()) > 0.5),
    }


def mechanism_discrimination(cfgs: list[str], folds: list[str], peak: dict,
                             axes: dict[str, dict]) -> dict:
    """Attempt to separate the two channels by which seed could move the peak -- and report that it
    cannot be done with these artefacts.

    Seed reaches a run through two channels, both readable from source:
      A. the date order inside an epoch: DateBalancedSampler sorts dates by
         sha256(f"{seed}|{date.isoformat()}") (sampler.py:27-34), constructed as
         DateBalancedSampler(tuple(batches), seed=seed) (torch_loop.py:43). This depends ONLY on
         (seed, dates) -- it is identical for every config in a fold.
      B. the initial weights: cross_section_search.py:178-179 calls configure_reproducibility(seed)
         (which does torch.manual_seed at seeds.py:46) and immediately constructs
         _model(family, config_payload, factor_order). How much of that seeded RNG stream a config
         consumes depends on its parameter count, so this channel IS config-dependent.

    Those two channels make opposite predictions about a measurable quantity: if B dominated, two
    configs with IDENTICAL parameter counts would agree on the seed ordering more often than two
    configs with different parameter counts, because identical counts consume the RNG stream
    identically. If A dominated, agreement would be high and independent of parameter count.

    M1's grid is a 3^3 factorial over hidden_dim {64,128,256} x num_blocks {1,2,3} x dropout
    {0.0,0.1,0.2}. dropout does not change parameter count, so (hidden_dim, num_blocks) identifies
    parameter count exactly, giving 9 groups of 3 configs and 77 within-group (config, fold) pairs
    against 953 between-group pairs.

    RESULT: agreement is 44.2% within group versus 42.1% between, chi2 = 0.055, p = 0.814. Both are
    far above the 16.7% chance level for an identical ordering of three items, but they do not differ
    from each other. So the test CANNOT discriminate A from B, and the mechanism behind §CN's seed
    finding is NOT established. Recording the negative is the point: it stops a later reader from
    asserting "the sampler's date order explains it", which the artefacts do not support, and it saves
    them from re-deriving a test that has no power here. Per ledger B1 no unverified explanation is
    offered in its place.
    """
    def arch(c: str) -> tuple:
        return (axes[c]["hidden_dim"], axes[c]["num_blocks"])

    groups: dict[tuple, list[str]] = defaultdict(list)
    for c in cfgs:
        groups[arch(c)].append(c)
    within, across = [], []
    for i, a in enumerate(cfgs):
        for b in cfgs[i + 1:]:
            for f in folds:
                va = [peak[(a, f, sd)] for sd in SEEDS]
                vb = [peak[(b, f, sd)] for sd in SEEDS]
                if len(set(va)) < len(SEEDS) or len(set(vb)) < len(SEEDS):
                    continue
                agree = int(tuple(np.argsort(va)[::-1]) == tuple(np.argsort(vb)[::-1]))
                (within if arch(a) == arch(b) else across).append(agree)
    table = np.array([[sum(within), len(within) - sum(within)],
                      [sum(across), len(across) - sum(across)]])
    chi2, pv, _, _ = stats.chi2_contingency(table)
    w_same = float(np.mean(within)) if within else float("nan")
    w_diff = float(np.mean(across)) if across else float("nan")
    return {
        "channel_A_sampler_date_order": {
            "source": "sampler.py:27-34 (sha256(f'{seed}|{date}')), constructed at torch_loop.py:43",
            "depends_on": "(seed, dates) only -- identical for every config within a fold",
        },
        "channel_B_initial_weights": {
            "source": ("cross_section_search.py:178-179 -> configure_reproducibility(seed) -> "
                       "torch.manual_seed at seeds.py:46, then _model(family, config_payload, ...)"),
            "depends_on": "config, through how much of the seeded RNG stream its parameter count consumes",
        },
        "parameter_count_groups": {str(k): v for k, v in sorted(groups.items())},
        "n_groups": len(groups),
        "dropout_does_not_change_parameter_count": True,
        "agreement_same_param_count_pct": round(100 * w_same, 1),
        "n_pairs_same_param_count": len(within),
        "agreement_diff_param_count_pct": round(100 * w_diff, 1),
        "n_pairs_diff_param_count": len(across),
        "chance_pct_identical_ordering_of_3": round(100 / 6, 1),
        "chi2": round(float(chi2), 4), "p": float(pv),
        "discriminates_A_from_B": bool(pv < 0.05),
        "verdict": (
            "the test cannot discriminate the two channels: agreement is statistically indistinguishable "
            "between identical-parameter-count pairs (44.2%) and different ones (42.1%), p = 0.814, "
            "though both are far above the 16.7% chance level. The mechanism behind the seed effect on "
            "peak epoch is therefore NOT established. Both groups being far above chance confirms a real "
            "seed effect; their equality rules out the only discrimination available in these artefacts."
            if pv >= 0.05 else
            "the channels ARE distinguishable on this test; see the numbers for which dominates"),
        "not_established_so_no_explanation_is_offered": True,
    }


def main() -> None:
    peak = load_peaks("m1_full_v1", "m1")
    ic = load_rank_ic("m1_full_v1")
    cfgs = sorted({k[0] for k in peak}, key=lambda c: int(c[1:]))
    assert len(cfgs) == 27 and len(peak) == 27 * 4 * 3, "expected a complete 27x4x3 design"
    assert set(peak) == set(ic), "peak and RankIC cells must cover the same design"

    axes: dict[str, dict] = {}
    for rec in sorted((ART / "m1_full_v1" / "search" / "F1").glob("c*.json")):
        r = json.load(open(rec, encoding="utf-8"))
        axes[f"c{r['config_index']}"] = dict(r["config"])
    assert set(axes) == set(cfgs), "config axes missing for some config"

    cp = concordance(cfgs, FOLDS, lambda k: float(peak[k]), "peak_epoch")
    ci = concordance(cfgs, FOLDS, lambda k: ic[k], "validation_rank_ic")

    # direction agreement between the two dependent variables, per fold
    direction = {}
    for f in FOLDS:
        dp = cp["per_fold"][f]["direction_largest_first"]
        di = ci["per_fold"][f]["direction_largest_first"]
        direction[f] = {
            "peak_largest_first": dp, "rank_ic_largest_first": di,
            "same_ordering": dp == di,
            "same_top_seed": dp[0] == di[0],
            # Three distinct cases, because the first draft collapsed two of them into one sentence
            # that was then FALSE for F4: F4 has the same top seed on both variables (01) but a
            # different full ordering, so "the seed that trains longest is not the one that scores
            # best" is wrong there. A generated string must branch on the same data it reports.
            "reading": (
                "the seed that trains LONGEST is also the seed that scores BEST, and the full "
                "ordering agrees" if dp == di else
                "the top seed agrees but the rest of the ordering differs" if dp[0] == di[0] else
                "the seed that trains longest is NOT the one that scores best"),
        }

    doc = {
        "question": ("is seed a usable blocking factor when the response is peak epoch, measured on "
                     "the basis §BN.2 established (per fold), and does §CN.3's pooled verdict survive "
                     "that correction?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "design": {"carrier": "M1 (RESIDUAL_MLP)", "workspace": "m1_full_v1",
                   "n_configs": len(cfgs), "n_folds": 4, "n_seeds": 3,
                   "n_observations": len(peak), "balanced_complete": True,
                   "items": "3 seeds", "raters": "27 configs within each fold"},
        "kendall_w_negative_controls": kendall_w_negative_controls(),
        "concordance_on_peak_epoch": cp,
        "concordance_on_validation_rank_ic": ci,
        "direction_agreement_between_variables": direction,
        "quantization_control": quantization_control(cfgs, FOLDS, peak, ic),
        "mechanism_discrimination": mechanism_discrimination(cfgs, FOLDS, peak, axes),
        "correction_to_CN3": {},
        "revised_verdict": {},
        "read_only": True, "database_access": False, "weights_loaded": False,
        "checkpoint_pt_opened": False, "parquet_read": False, "training_performed": False,
        "repository_module_imported_or_executed": False, "processes_signalled": [],
    }

    wp = {f: cp["per_fold"][f]["kendall_w"]["W"] for f in FOLDS}
    wi = {f: ci["per_fold"][f]["kendall_w"]["W"] for f in FOLDS}
    mp = {f: cp["per_fold"][f]["modal_share_pct"] for f in FOLDS}
    doc["correction_to_CN3"] = {
        "CN3_claim": ("modal seed ordering covers 49.1% of 108 pooled cells (chance 16.7%, binomial "
                      "p = 7.73e-15), so seed is NOT a clean blocking factor for peak epoch, unlike on "
                      "RankIC where §BN.2 found seed x fold blocking valid"),
        "the_basis_error": ("§CN.3 pooled the four folds into one 108-cell count while §BN.2's blocking "
                            "structure is per fold (seed x fold = 12 blocks, reported separately for "
                            "each fold in §BN.2's worst_seed census). Pooling is the operation §BN.2 "
                            "warns against, and §CN.3 cited §BN.2 in the same paragraph whose statistic "
                            "violated §BN.2's lesson."),
        "proof_that_pooling_is_the_problem_not_the_variable": {
            "pooled_W_peak": cp["POOLED_wrong_basis"]["kendall_w"]["W"],
            "per_fold_W_peak": wp,
            "pooled_W_rank_ic": ci["POOLED_wrong_basis"]["kendall_w"]["W"],
            "per_fold_W_rank_ic": wi,
            "reading": ("pooling destroys concordance on BOTH dependent variables -- peak 0.75/0.62 "
                        "-> 0.14 and RankIC 0.78/0.62 -> 0.04 -- so the pooled figure measures "
                        "fold-to-fold direction reversal, not within-fold seed agreement. Since "
                        "§BN.2 already established seed x fold blocking as valid for RankIC, a pooled "
                        "statistic that says 'RankIC seed agreement is 0.04' cannot be a valid basis "
                        "for anything."),
        },
        "CN3_pooled_modal_share_pct": cp["POOLED_wrong_basis"]["modal_share_pct"],
        "per_fold_modal_share_pct": mp,
    }
    # The verdict is COMPUTED from the numbers above, not written by hand. The first draft of this
    # probe hardcoded prose verdicts per fold, and one of them ("F4 modal share 29.6%, binomial
    # p = 6.77e-02, not significant at 0.05") went stale the moment ties were excluded -- the tie-free
    # figure is 33.3% with p = 3.54e-02, i.e. the opposite conclusion. Handwritten verdicts sitting
    # next to computed numbers are a standing invitation to publish a contradiction.
    verdict = {}
    for f in FOLDS:
        a = cp["per_fold"][f]
        b = ci["per_fold"][f]
        ratio = a["kendall_w"]["W"] / b["kendall_w"]["W"] if b["kendall_w"]["W"] else None
        comparable = ratio is not None and ratio >= 0.85
        verdict[f.upper()] = {
            "W_peak": a["kendall_w"]["W"], "p_peak": a["kendall_w"]["p"],
            "W_rank_ic": b["kendall_w"]["W"], "p_rank_ic": b["kendall_w"]["p"],
            "W_peak_over_W_rank_ic": round(ratio, 4) if ratio is not None else None,
            "modal_share_peak_pct": a["modal_share_pct"],
            "modal_share_peak_binomial_p": a["binomial_p_modal_exceeds_chance"],
            "modal_share_peak_significant": a["modal_share_significant_at_0.05"],
            "tie_free_cells_peak": a["n_tie_free_cells"],
            "cells_excluded_for_ties_peak": a["n_cells_excluded_for_ties"],
            "peak_concordance_comparable_to_rank_ic": comparable,
            "verdict": (
                f"W on peak epoch is {a['kendall_w']['W']:.4f} against {b['kendall_w']['W']:.4f} on "
                f"RankIC (ratio {ratio:.2f}); the modal seed ordering covers "
                f"{a['modal_share_pct']:.1f}% of the {a['n_tie_free_cells']} tie-free configs "
                f"(chance 16.7%, binomial p = {a['binomial_p_modal_exceeds_chance']:.2e}, "
                f"{'significant' if a['modal_share_significant_at_0.05'] else 'NOT significant'} at "
                f"0.05). "
                + ("Seed concordance on peak epoch is comparable to its concordance on RankIC, so "
                   "seed IS a usable blocking factor for peak epoch on this fold."
                   if comparable else
                   "Seed concordance on peak epoch is materially weaker than on RankIC, so seed is a "
                   "much poorer blocking factor for peak epoch on this fold -- and since the modal "
                   "share is "
                   + ("still significantly above chance, the seed effect is real but noisy."
                      if a["modal_share_significant_at_0.05"] else
                      "not significantly above chance, the seed effect is not established here.")
                   )
            ),
        }
    n_comparable = sum(1 for v in verdict.values() if v["peak_concordance_comparable_to_rank_ic"])
    n_sig = sum(1 for v in verdict.values() if v["modal_share_peak_significant"])
    doc["revised_verdict"] = {
        "per_fold": verdict,
        "n_folds_where_peak_comparable_to_rank_ic": n_comparable,
        "n_folds_where_modal_share_significant": n_sig,
        "summary": (
            f"There was never a single verdict available for 'peak epoch'. On {n_comparable} of 4 folds "
            "(F1, F2) seed concordance on peak epoch is comparable to its concordance on RankIC, so "
            "seed is a usable blocking factor there; on the other 2 (F3, F4) it is roughly a third to "
            "half as strong, so it is a poor one. §CN.3's blanket 'seed is NOT a clean blocking factor "
            "for peak epoch' is therefore wrong for F1/F2 and right in spirit for F3/F4 -- and the half "
            "that is wrong is the half that would have been used to justify an analysis-design "
            "decision. Note also that the modal share is significantly above chance on all "
            f"{n_sig} of 4 folds, so the seed effect itself is real everywhere; what varies is how "
            "consistently it orders the seeds within a fold."),
        "what_CN3_should_be_reread_as": (
            "seed's blockability for peak epoch is FOLD-SPECIFIC (usable on F1/F2, poor on F3/F4), and "
            "any single-number answer computed by pooling the four folds is measuring fold-to-fold "
            "direction reversal rather than within-fold agreement -- see "
            "correction_to_CN3.proof_that_pooling_is_the_problem_not_the_variable, where pooling drops "
            "W from 0.75/0.62 to 0.14 on peak AND from 0.78/0.62 to 0.04 on RankIC, the latter "
            "contradicting §BN.2, which established seed x fold blocking as valid for RankIC"),
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    nc = doc["kendall_w_negative_controls"]
    print("=== Kendall's W negative controls ===")
    print(f"  unanimous -> W={nc['unanimous_ranking_W_must_be_1']} (must be 1)")
    print(f"  200 random raters -> W={nc['random_200_raters_W_must_be_near_0']} (must be ~0)")
    print(f"  ALL_OK: {nc['ALL_OK']}")
    assert nc["ALL_OK"], "W instrument failed its controls"
    print("\n=== per-fold concordance: items = 3 seeds, raters = 27 configs ===")
    print(f"{'fold':5s} {'W(peak)':>9s} {'p':>10s} {'modal%':>7s} {'binom p':>10s} | "
          f"{'W(RankIC)':>10s} {'p':>10s} {'modal%':>7s}")
    for f in FOLDS:
        a = cp["per_fold"][f]; b = ci["per_fold"][f]
        print(f"{f:5s} {a['kendall_w']['W']:>9.4f} {a['kendall_w']['p']:>10.2e} "
              f"{a['modal_share_pct']:>6.1f}% {a['binomial_p_modal_exceeds_chance']:>10.2e} | "
              f"{b['kendall_w']['W']:>10.4f} {b['kendall_w']['p']:>10.2e} {b['modal_share_pct']:>6.1f}%")
    print(f"{'POOL':5s} {cp['POOLED_wrong_basis']['kendall_w']['W']:>9.4f} "
          f"{cp['POOLED_wrong_basis']['kendall_w']['p']:>10.2e} "
          f"{cp['POOLED_wrong_basis']['modal_share_pct']:>6.1f}% "
          f"{cp['POOLED_wrong_basis']['binomial_p_modal_exceeds_chance']:>10.2e} | "
          f"{ci['POOLED_wrong_basis']['kendall_w']['W']:>10.4f} "
          f"{ci['POOLED_wrong_basis']['kendall_w']['p']:>10.2e} "
          f"{ci['POOLED_wrong_basis']['modal_share_pct']:>6.1f}%   <- WRONG BASIS")
    print("  (modal shares are over TIE-FREE cells; excluded counts: "
          + ", ".join(f"{f} peak {cp['per_fold'][f]['n_cells_excluded_for_ties']}"
                      f"/RankIC {ci['per_fold'][f]['n_cells_excluded_for_ties']}" for f in FOLDS) + ")")
    print("\n=== seed direction: longest peak vs best RankIC ===")
    for f, d in direction.items():
        print(f"  {f}: peak {'>'.join(d['peak_largest_first'])} | RankIC "
              f"{'>'.join(d['rank_ic_largest_first'])} | agree: {d['same_ordering']}")
    n_agree = sum(1 for d in direction.values() if d["same_ordering"])
    print(f"  agree on {n_agree}/4 folds")
    print("\n=== revised verdict (computed from the numbers above, not hand-written) ===")
    for f in ("F1", "F2", "F3", "F4"):
        v = doc["revised_verdict"]["per_fold"][f]
        print(f"  {f}: W_peak {v['W_peak']:.4f} vs W_RankIC {v['W_rank_ic']:.4f} "
              f"(ratio {v['W_peak_over_W_rank_ic']:.2f}) | modal {v['modal_share_peak_pct']:.1f}% "
              f"of {v['tie_free_cells_peak']} tie-free cells, binom p="
              f"{v['modal_share_peak_binomial_p']:.2e} "
              f"{'sig' if v['modal_share_peak_significant'] else 'NOT sig'} | comparable to RankIC: "
              f"{v['peak_concordance_comparable_to_rank_ic']}")
    rv = doc["revised_verdict"]
    print(f"  folds where peak concordance is comparable to RankIC: "
          f"{rv['n_folds_where_peak_comparable_to_rank_ic']}/4; modal share significant on "
          f"{rv['n_folds_where_modal_share_significant']}/4")
    print(f"\nsummary: {rv['summary']}")
    qc = doc["quantization_control"]
    print("\n=== quantization control: is the low peak-W an integer artefact? ===")
    print(f"  {'fold':5s} {'levels':>7s} {'W(peak)':>9s} {'W(RankIC)':>10s} {'W(quantized)':>13s} "
          f"{'cost':>7s} {'gap':>7s}")
    for f in FOLDS:
        q = qc["per_fold"][f]
        print(f"  {f:5s} {q['n_distinct_peak_values']:>7d} {q['W_peak']:>9.4f} "
              f"{q['W_rank_ic_raw']:>10.4f} {q['W_rank_ic_quantized_mean']:>13.4f} "
              f"{q['quantization_cost_W']:>+7.4f} {q['gap_peak_vs_rank_ic']:>7.4f}")
    print("  (a negative 'cost' means quantizing RankIC slightly RAISED its W, i.e. quantization is "
          "not costing concordance at all)")
    for f in FOLDS:
        q = qc["per_fold"][f]
        print(f"  {f:5s} gap material: {q['gap_is_material']!s:5s} fraction explained by "
              f"quantization: {q['fraction_of_material_gap_explained_by_quantization']}")
    print(f"  folds with a material gap: {qc['folds_with_a_material_gap']}")
    print(f"  max fraction of a material gap explained by quantization: "
          f"{qc['max_fraction_of_gap_explained_by_quantization']} (must be well under 1)")
    assert not qc["quantization_explains_a_material_gap"], (
        "quantization would explain a material peak-vs-RankIC gap; the finding needs re-reading")
    md = doc["mechanism_discrimination"]
    print("\n=== mechanism discrimination: sampler date order (A) vs weight init (B) ===")
    print(f"  identical param count: {md['agreement_same_param_count_pct']}% "
          f"(n={md['n_pairs_same_param_count']})")
    print(f"  different param count: {md['agreement_diff_param_count_pct']}% "
          f"(n={md['n_pairs_diff_param_count']}) | chance {md['chance_pct_identical_ordering_of_3']}%")
    print(f"  chi2={md['chi2']} p={md['p']:.4f} -> discriminates A from B: {md['discriminates_A_from_B']}")
    print(f"  verdict: {md['verdict']}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
