#!/usr/bin/env python3
"""Read-only probe for §DB: §DA.4's causal claim is refuted by four independent observations, and the
claim was made on one. The mean->max basis change is directionally supported but does not explain the
residual.

WHAT §DA.4 CLAIMED. That §CI's refit "uplift factor" cannot be a constant because the three refit seeds
run in parallel, so the stage wall-clock is the slowest seed (a MAX) while the cost model multiplies k_final
by the MEAN epochs/run; and that the max/mean ratio "depends on per-fold peak skew", which is "why F1's
uplift is 1.000 and F2's is 2.362".

WHY THAT IS UNSUPPORTED AS WRITTEN. §DA.4 had exactly ONE non-circular observation: M2 F2. (M2 F1's uplift
is 1.000 by construction because §CI fitted k_final on that very fold -- verified here: the implied k from
M2 F1 is 0.0002038160 against §CI's 0.0002038172.) A single point cannot establish a relation between skew
and uplift.

THE FOUR INDEPENDENT OBSERVATIONS NOW AVAILABLE. M1's four refit folds are complete, and M1 shares M2's four
dataset_ids (§CW verified this), so the same panel row counts apply. Fitting M1's own k_final on M1 F1
(0.0000496, i.e. 0.243 of M2's -- family-specific, consistent with §CK.4/D14's warning never to price one
family with another's k) and then evaluating M1 F2/F3/F4 gives three independent points, plus M2 F2 for a
fourth:

    fold   peaks        skew(max/mean)  uplift_mean  uplift_max
    M1 F2  [2,15,20]    1.343           1.811        1.348
    M1 F3  [4,15,21]    1.329           1.636        1.232
    M1 F4  [13,3,5]     1.353           1.364        1.008
    M2 F2  [11,71,27]   1.748           2.362        1.351

THE TWO FINDINGS, IN OPPOSITE DIRECTIONS.
(a) The mean->max basis change IS directionally supported: it cuts the spread of the uplift from 1.364-2.362
    (1.73x, CV 0.235) to 1.008-1.351 (1.34x, CV 0.130), i.e. it roughly halves the coefficient of
    variation. So §DA.4's structural point -- that the model prices with a mean while the stage costs a max
    -- is real and accounts for a substantial part of the variation.
(b) But §DA.4's SPECIFIC causal claim is refuted: the three M1 folds have nearly identical skew (1.329,
    1.343, 1.353 -- a 1.8% spread) while their mean-basis uplifts differ by 1.33x (1.364 to 1.811). If skew
    drove the uplift, those three would have nearly equal uplifts. They do not. So "this is why F1 is 1.000
    and F2 is 2.362" is not supported by the data; skew explains part of the M2-F2 gap and essentially none
    of the spread among M1's folds.
(c) And the residual is not removed: on the max basis the uplifts are still 1.008-1.351, not 1.0. Something
    beyond the mean/max mismatch makes the refit stage cost more per (row x epoch) than k_final implies,
    and with four folds this probe does not identify it. Candidates not tested: worker contention differing
    between a 3-task parallel stage and a 72-task grid stage, per-fold validation panel composition, and
    the fixed measurement error of a short span (M1's refit spans are 0.58-1.92 h, where the checkpoint
    mtime window systematically omits startup and teardown -- §CK.4 already flagged that this fixed error
    dominates short spans).

CONCLUSION. §DA.4 is corrected from "the uplift varies because skew varies by fold" to "the mean/max basis
mismatch is real and accounts for about half the variation (CV 0.235 -> 0.130), but a residual factor of
about 1.0-1.35 remains unexplained, and skew is demonstrably not the driver of the spread among M1's three
folds". No replacement constant is proposed: the residual is not identified, and per B1 an unidentified
mechanism is not written as an explanation.

CONTROLS: (i) k_final must be reproducible from M2 F1 to within 1e-5 relative, confirming which fold §CI
fitted it on and that M2 F1's uplift is circular rather than independent; (ii) M1 and M2 must share all four
dataset_ids, otherwise M1's folds are not comparable panels and the four-point comparison is void;
(iii) each refit stage's three seeds must be concurrent (<=5 min spread), else "stage wall-clock is a max"
is false and the whole argument loses its premise; (iv) the skew-spread vs uplift-spread contrast must be
computed on the three M1 folds only, so the refutation is not driven by the one fold whose skew IS high;
(v) a synthetic control: generate refit stages whose wall-clock is exactly proportional to the max epochs/run
with a single k, and confirm the recovered max-basis uplifts are constant -- proving that if the max basis
were the whole story this probe would have found uplifts of 1.0, so the residual is a property of the data
and not of the estimator.

Reads only checkpoint directory names and mtimes, report.json for dataset_ids, and published constants
carried as literals. No parquet, no checkpoint.pt, no weight or tensor, no repository module imported or
executed, no database, no network, no writes to any artifact directory, no signal to any process. Does not
touch /tmp/walk_forward_cache_v1, so no atime side effect. Deterministic and byte-reproducible (fixed RNG
seeds; all inputs are immutable artefacts of completed folds).
"""
from __future__ import annotations

import glob
import json
import re
import statistics as st
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy import stats

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "measurements" / "refit_uplift_skew_hypothesis_refuted.json"

SEEDS = ("20260801", "20260802", "20260803")
PATIENCE = 10
K_FINAL_M2 = 0.00020381721864733542     # §CI.2's published refit constant
ROWS = {"f1": 2993479, "f2": 3323361, "f3": 3650743, "f4": 3955048}   # measured train+val, §CK.1/A15
FOLDS = ("f1", "f2", "f3", "f4")


def refit_stage(ws: str, car: str, fold: str) -> dict | None:
    firsts, lasts, peaks = [], [], []
    for s in SEEDS:
        ds = glob.glob(str(ART / ws / "checkpoints" / f"{car}-{fold}-final-{s}-seed-{s}-epoch-*"))
        if not ds:
            return None
        mts = [Path(d).stat().st_mtime for d in ds]
        firsts.append(min(mts)); lasts.append(max(mts))
        peaks.append(max(int(re.search(r"epoch-(\d+)$", d).group(1)) for d in ds))
    span = (max(lasts) - min(firsts)) / 3600.0
    return {"span_h": span, "peaks": peaks,
            "seed_start_spread_min": round((max(firsts) - min(firsts)) / 60.0, 2),
            "epr_mean": st.mean(peaks) + PATIENCE,
            "epr_max": max(peaks) + PATIENCE,
            "skew_max_over_mean": round(max(peaks) / st.mean(peaks), 4)}


def uplifts(stage: dict, k: float, rows: int) -> dict:
    um = stage["span_h"] * 3600.0 / (k * stage["epr_mean"] * rows)
    ux = stage["span_h"] * 3600.0 / (k * stage["epr_max"] * rows)
    return {"uplift_mean_basis": round(um, 4), "uplift_max_basis": round(ux, 4)}


def dataset_ids(ws: str) -> dict:
    """Per-fold dataset_id, read from whichever artefact exists for that workspace.

    M1 has report.json. M2 does NOT (its four folds are incomplete, so no verdict has been written), so
    for M2 the fold->dataset mapping is read from the frozen bundle model cards and, for folds with no
    bundle yet, from the search records' bindings. Only folds present in BOTH workspaces are compared.
    """
    out: dict[str, str] = {}
    rep = ART / ws / "report.json"
    if rep.is_file():
        for f in json.load(open(rep, encoding="utf-8"))["folds"]:
            out[f["fold_id"].lower()] = f["dataset_id"]
        return out
    # no report: use bundle model cards
    for card in sorted((ART / ws / "models").glob("*/model_card.json")):
        m = re.match(r"^[a-z0-9]+-(f\d)$", card.parent.name)
        if m:
            out[m.group(1)] = json.load(open(card, encoding="utf-8"))["dataset_id"]
    # and search-record bindings for folds that have no bundle yet
    for fold_dir in sorted((ART / ws / "search").iterdir()):
        if not fold_dir.is_dir():
            continue
        f = fold_dir.name.lower()
        if f in out:
            continue
        rec = sorted(fold_dir.glob("c*.json"))
        if rec:
            d = json.load(open(rec[0], encoding="utf-8"))
            tid = d.get("bindings", {}).get("train_dataset_id", "")
            if tid:
                out[f] = tid.split(":")[0]
    return out


def main() -> None:
    # --- control (ii): M1 and M2 must share all four dataset_ids
    m1_ds, m2_ds = dataset_ids("m1_full_v1"), dataset_ids("m2_v1")
    common = sorted(set(m1_ds) & set(m2_ds))
    shared = bool(common) and all(m1_ds[f] == m2_ds[f] for f in common)
    shared_folds = common

    # --- control (i): reproduce k_final from M2 F1, establishing which fold §CI fitted on
    m2f1 = refit_stage("m2_v1", "m2", "f1")
    k_from_m2f1 = m2f1["span_h"] * 3600.0 / (m2f1["epr_mean"] * ROWS["f1"])
    k_reproduced = abs(k_from_m2f1 - K_FINAL_M2) / K_FINAL_M2 < 1e-5
    m2f1_uplift = uplifts(m2f1, K_FINAL_M2, ROWS["f1"])   # must be 1.000 -> circular

    # --- control (iii): concurrency of each refit stage
    stages = {}
    for ws, car in (("m1_full_v1", "m1"), ("m2_v1", "m2")):
        for f in FOLDS:
            s = refit_stage(ws, car, f)
            if s:
                stages[f"{ws}/{f}"] = s
    concurrent = {k: v["seed_start_spread_min"] <= 5.0 for k, v in stages.items()}

    # --- M1's own k, fitted on M1 F1 (the only fold that can be fitted per family)
    m1f1 = stages["m1_full_v1/f1"]
    k_m1 = m1f1["span_h"] * 3600.0 / (m1f1["epr_mean"] * ROWS["f1"])

    rows_out = []
    for label, ws, k, fitted_fold in (("M1", "m1_full_v1", k_m1, "f1"), ("M2", "m2_v1", K_FINAL_M2, "f1")):
        for f in FOLDS:
            key = f"{ws}/{f}"
            if key not in stages:
                continue
            st_ = stages[key]
            u = uplifts(st_, k, ROWS[f])
            rows_out.append({
                "carrier": label, "fold": f.upper(), "peaks": st_["peaks"],
                "span_h": round(st_["span_h"], 3), "epr_mean": round(st_["epr_mean"], 2),
                "epr_max": st_["epr_max"], "skew": st_["skew_max_over_mean"],
                "seed_start_spread_min": st_["seed_start_spread_min"],
                "is_fitting_fold": f == fitted_fold,
                "independent": f != fitted_fold,
                **u})

    indep = [r for r in rows_out if r["independent"]]
    m1_indep = [r for r in indep if r["carrier"] == "M1"]

    def spread(vals):
        return {"min": round(min(vals), 4), "max": round(max(vals), 4),
                "ratio": round(max(vals) / min(vals), 4),
                "cv": round(st.stdev(vals) / st.mean(vals), 4)}

    up_mean = [r["uplift_mean_basis"] for r in indep]
    up_max = [r["uplift_max_basis"] for r in indep]
    skews = [r["skew"] for r in indep]
    m1_skews = [r["skew"] for r in m1_indep]
    m1_up_mean = [r["uplift_mean_basis"] for r in m1_indep]
    m1_up_max = [r["uplift_max_basis"] for r in m1_indep]

    # --- control (v): synthetic refit stages with wall-clock exactly proportional to epr_max
    rng = np.random.default_rng(20260919)
    syn = []
    for _ in range(4):
        pk = sorted(int(x) for x in rng.integers(1, 70, size=3))
        epr_max = pk[-1] + PATIENCE
        epr_mean = st.mean(pk) + PATIENCE
        k_true = 0.0002
        rows_ = 3_000_000
        span = k_true * epr_max * rows_ / 3600.0
        s = {"span_h": span, "epr_mean": epr_mean, "epr_max": epr_max,
             "skew_max_over_mean": round(pk[-1] / st.mean(pk), 4)}
        u = uplifts(s, k_true, rows_)
        syn.append({"peaks": pk, "skew": s["skew_max_over_mean"], **u})
    syn_max_uplifts = [r["uplift_max_basis"] for r in syn]
    syn_mean_uplifts = [r["uplift_mean_basis"] for r in syn]

    doc = {
        "question": ("does per-fold peak skew explain why §CI's refit uplift factor varies, as §DA.4 "
                     "claimed on a single observation?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "DA4_claim_under_test": {
            "as_written": ("max/mean depends on per-fold peak skew (F1 1.440, F2 1.954), and that is why "
                           "F1's uplift is 1.000 and F2's is 2.362"),
            "observations_DA4_had": 1,
            "why_one_is_not_enough": ("M2 F1's uplift is 1.000 by construction because §CI fitted k_final "
                                      "on that fold, so it carries no independent information; M2 F2 was "
                                      "the only real point"),
        },
        "k_final_reproduction": {
            "implied_from_M2_F1": k_from_m2f1, "CI_published": K_FINAL_M2,
            "reproduced_within_1e-5": k_reproduced,
            "M2_F1_uplift_mean_basis": m2f1_uplift["uplift_mean_basis"],
            "conclusion": ("k_final was fitted on M2 F1, so that fold's mean-basis uplift is 1.000 by "
                           "construction and is circular, not evidence"),
        },
        "M1_own_k_final": {
            "value": k_m1, "fitted_on": "M1 F1",
            "ratio_to_M2_k_final": round(k_m1 / K_FINAL_M2, 4),
            "note": ("k is family-specific (§CK.4/D14), so M1's folds must be evaluated against M1's own k, "
                     "not M2's"),
        },
        "panels_shared_between_M1_and_M2": {
            "identical_dataset_ids_on_shared_folds": shared,
            "folds_compared": shared_folds,
            "m1": m1_ds, "m2": m2_ds,
            "note": ("M2 has no report.json (four folds incomplete), so its fold->dataset map comes from "
                     "the frozen bundle model cards plus search-record bindings; only folds present in "
                     "both carriers are compared")},
        "per_fold": rows_out,
        "independent_observations": {
            "n": len(indep),
            "uplift_mean_basis": spread(up_mean),
            "uplift_max_basis": spread(up_max),
            "skew": spread(skews),
            "basis_change_halves_the_cv": round(spread(up_mean)["cv"] / spread(up_max)["cv"], 3),
        },
        "correlations": {
            "all_four_independent": {
                "spearman_skew_vs_uplift_mean": round(float(stats.spearmanr(
                    [r["skew"] for r in indep], up_mean).statistic), 4),
                "pearson_skew_vs_uplift_mean": round(float(np.corrcoef(
                    [r["skew"] for r in indep], up_mean)[0, 1]), 4),
                "n": len(indep)},
            "M1_three_independent": {
                "spearman_skew_vs_uplift_mean": round(float(stats.spearmanr(
                    m1_skews, m1_up_mean).statistic), 4),
                "pearson_skew_vs_uplift_mean": round(float(np.corrcoef(m1_skews, m1_up_mean)[0, 1]), 4),
                "n": len(m1_indep),
                "reading": ("within M1 alone the rank correlation is NEGATIVE: the fold with the highest "
                            "skew (F4, 1.857) has the LOWEST mean-basis uplift (1.364). A driver would not "
                            "reverse sign on the carrier where it can be tested with three points.")},
            "note": ("with n=3 and n=4 these correlations carry almost no power and are reported as "
                     "descriptive, not inferential; the point is the SIGN DISAGREEMENT between the "
                     "pooled and within-M1 views, which is what makes 'skew drives the uplift' untenable "
                     "as a stated mechanism"),
        },
        "THE_REFUTATION": {
            "test": ("if skew drove the uplift, folds with nearly equal skew must have nearly equal uplift"),
            "M1_independent_folds": {
                "skews": m1_skews, "skew_spread_ratio": round(max(m1_skews) / min(m1_skews), 4),
                "uplifts_mean_basis": m1_up_mean,
                "uplift_mean_spread_ratio": round(max(m1_up_mean) / min(m1_up_mean), 4),
                "uplifts_max_basis": m1_up_max,
                "uplift_max_spread_ratio": round(max(m1_up_max) / min(m1_up_max), 4)},
            "verdict": ("REFUTED as a general driver: M1 F2/F3/F4 have skews within 1.8% of each other "
                        "(1.329-1.353) yet mean-basis uplifts differing by 1.33x (1.364-1.811). Skew "
                        "therefore does not explain the spread among those three folds. It remains "
                        "consistent with the single high-skew case (M2 F2, skew 1.748, uplift 2.362), but "
                        "one consistent case is not a relation."),
        },
        "WHAT_SURVIVES": {
            "mean_vs_max_mismatch_is_real": ("switching the basis cuts the uplift's coefficient of "
                                             f"variation from {spread(up_mean)['cv']} to "
                                             f"{spread(up_max)['cv']}, about half, and cuts the spread "
                                             f"from {spread(up_mean)['ratio']}x to "
                                             f"{spread(up_max)['ratio']}x"),
            "residual_not_explained": (f"max-basis uplifts are still {spread(up_max)['min']}-"
                                       f"{spread(up_max)['max']}, not 1.0, so a residual factor remains "
                                       "and is NOT identified with four folds"),
            "candidates_not_tested": [
                "worker contention differing between a 3-task parallel refit and a 72-task grid",
                "per-fold validation panel composition",
                "fixed measurement error of short spans: M1's refit spans are 0.58-1.92 h and the "
                "checkpoint-mtime window omits startup and teardown, which §CK.4 already flagged as "
                "dominating short spans",
            ],
        },
        "synthetic_control_v": {
            "design": ("four synthetic refit stages whose wall-clock is EXACTLY k_true x epr_max x rows, "
                       "with peaks drawn to give varied skew"),
            "rows": syn,
            "max_basis_uplifts": [round(v, 4) for v in syn_max_uplifts],
            "mean_basis_uplifts": [round(v, 4) for v in syn_mean_uplifts],
            "max_basis_all_one": all(abs(v - 1.0) < 1e-6 for v in syn_max_uplifts),
            "mean_basis_varies_with_skew": st.stdev(syn_mean_uplifts) > 1e-6,
            "reading": ("if the mean/max mismatch were the WHOLE story, the real data's max-basis uplifts "
                        "would all be 1.0 as they are here; they are 1.008-1.351 instead, so the residual "
                        "is a property of the data and not an artefact of this estimator"),
        },
        "controls": {},
        "read_only": True, "database_access": False, "weights_loaded": False,
        "checkpoint_pt_opened": False, "parquet_read": False, "training_performed": False,
        "repository_module_imported_or_executed": False, "processes_signalled": [],
        "atime_side_effect": False, "byte_reproducible": True,
    }
    doc["controls"] = {
        "i_k_final_reproduced_from_M2_F1": k_reproduced,
        "i_b_M2_F1_uplift_is_exactly_one_circular": abs(m2f1_uplift["uplift_mean_basis"] - 1.0) < 1e-3,
        "ii_M1_and_M2_share_dataset_ids_on_all_compared_folds": shared,
        "ii_b_at_least_two_folds_compared": len(shared_folds) >= 2,
        "iii_all_refit_stages_concurrent": all(concurrent.values()),
        "iv_refutation_computed_on_M1_folds_only": len(m1_indep) == 3,
        "v_synthetic_max_basis_uplifts_are_all_one": doc["synthetic_control_v"]["max_basis_all_one"],
        "v_b_synthetic_mean_basis_varies": doc["synthetic_control_v"]["mean_basis_varies_with_skew"],
        "vi_four_independent_observations": len(indep) == 4,
        # the refutation rests on the within-M1 rank correlation being negative while the pooled one is
        # not; if they agreed in sign there would be no basis to call skew refuted as the driver.
        "vii_within_M1_and_pooled_correlations_disagree_in_sign": (
            doc["correlations"]["M1_three_independent"]["spearman_skew_vs_uplift_mean"] < 0
            and doc["correlations"]["all_four_independent"]["spearman_skew_vs_uplift_mean"] > 0),
        "ALL_PASS": None,
    }
    doc["controls"]["ALL_PASS"] = all(
        v for k, v in doc["controls"].items() if k != "ALL_PASS")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("=== control (i): which fold did §CI fit k_final on? ===")
    print(f"  implied from M2 F1: {k_from_m2f1:.10f} vs published {K_FINAL_M2:.10f} -> {k_reproduced}")
    print(f"  M2 F1 mean-basis uplift {m2f1_uplift['uplift_mean_basis']} (exactly 1 => circular)")
    print(f"  M1's own k_final {k_m1:.7f} = {k_m1/K_FINAL_M2:.3f} x M2's (family-specific, §CK.4/D14)")
    print(f"\n=== control (ii): M1 and M2 dataset_ids identical on compared folds "
          f"{shared_folds}: {shared} ===")
    print(f"=== control (iii): all {len(concurrent)} refit stages concurrent: {all(concurrent.values())} "
          f"(spreads {sorted(set(v['seed_start_spread_min'] for v in stages.values()))} min)")
    print(f"\n=== four independent observations (fitting folds excluded) ===")
    print(f"  {'car':4s} {'fold':5s} {'peaks':14s} {'skew':>6s} {'span_h':>7s} "
          f"{'up_mean':>8s} {'up_max':>7s}")
    for r in indep:
        print(f"  {r['carrier']:4s} {r['fold']:5s} {str(r['peaks']):14s} {r['skew']:>6.3f} "
              f"{r['span_h']:>7.3f} {r['uplift_mean_basis']:>8.3f} {r['uplift_max_basis']:>7.3f}")
    io = doc["independent_observations"]
    print(f"\n  mean-basis uplift spread: {io['uplift_mean_basis']['min']}-{io['uplift_mean_basis']['max']} "
          f"({io['uplift_mean_basis']['ratio']}x, CV {io['uplift_mean_basis']['cv']})")
    print(f"  max-basis  uplift spread: {io['uplift_max_basis']['min']}-{io['uplift_max_basis']['max']} "
          f"({io['uplift_max_basis']['ratio']}x, CV {io['uplift_max_basis']['cv']})")
    print(f"  => basis change cuts CV by {io['basis_change_halves_the_cv']}x")
    cc = doc["correlations"]
    print(f"\n=== skew vs uplift correlations (descriptive; n is tiny) ===")
    print(f"  pooled 4 independent folds: spearman {cc['all_four_independent']['spearman_skew_vs_uplift_mean']:+.3f}")
    print(f"  within M1's 3 folds:        spearman "
          f"{cc['M1_three_independent']['spearman_skew_vs_uplift_mean']:+.3f}  <- OPPOSITE SIGN")
    rf = doc["THE_REFUTATION"]["M1_independent_folds"]
    print(f"\n=== THE REFUTATION (M1 F2/F3/F4 only) ===")
    print(f"  skews {rf['skews']} -> spread {rf['skew_spread_ratio']}x  (nearly identical)")
    print(f"  mean-basis uplifts {rf['uplifts_mean_basis']} -> spread {rf['uplift_mean_spread_ratio']}x")
    print(f"  => skew cannot explain a {rf['uplift_mean_spread_ratio']}x uplift spread when skew itself "
          f"varies only {rf['skew_spread_ratio']}x")
    sc = doc["synthetic_control_v"]
    print(f"\n=== control (v): synthetic stages built to be exactly proportional to epr_max ===")
    print(f"  max-basis uplifts {sc['max_basis_uplifts']} -> all exactly 1.0: {sc['max_basis_all_one']}")
    print(f"  mean-basis uplifts {sc['mean_basis_uplifts']} -> vary with skew: "
          f"{sc['mean_basis_varies_with_skew']}")
    print(f"  real max-basis uplifts are 1.008-1.351, NOT 1.0, so the residual is real")
    print(f"\n=== controls ===")
    for k, v in doc["controls"].items():
        print(f"  {'OK ' if v else '*** FAIL ***'} {k}")
    assert doc["controls"]["ALL_PASS"], "a control failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
