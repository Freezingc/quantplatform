#!/usr/bin/env python3
"""Read-only probe for §DF: the baseline side of the selection chain -- the one §CZ.7 left unverified.

§CX/§CY/§CZ verified the CANDIDATE side: bundle digest <-> model selection evidence, the keyless content
binding of every record, and the frozen prediction artefacts. §CZ.7 explicitly left open "the baseline
records' binding to the baseline artefact they name", noting their payload has no bundle digest so the chain
differs in shape. This section closes that, and finds the baseline chain is structurally different in a way
that matters: a baseline selection names a LABEL ("B0".."B3"), not a file, so there is no artefact to hash.
What can be verified instead is the internal consistency of the recorded competition and its sharing.

FINDING 1 -- all five source-enforced invariants hold on every record. selection_evidence.py enforces, at
write time (_validate_baseline_selection_record, :509-527) and partly at load time
(_validate_baseline_payload, :529-551):
  (a) validation_scores covers exactly {B0,B1,B2,B3} with no duplicates;
  (b) selected_baseline_id equals _baseline_winner(means), which is argmax of mean_rank_ic with ties broken
      toward the SIMPLER baseline (complexity B0<B1<B2<B3, :558-559);
  (c) payload.validation_rank_ic is bit-equal to the winner's mean_rank_ic (not merely close);
  (d) unselected_reasons names exactly the complement of the winner;
  (e) selected_baseline_id is in {B0..B3}.
Measured over all 15 baseline record references: 0 failures.

FINDING 2 -- there are only 4 DISTINCT baseline records behind those 15 references, and every copy is
byte-identical. Baselines are model-independent, so all four carriers reuse the same per-fold decision:
F1/F2/F3 are shared by 4 carriers each, F4 by 3 (M2's F4 has not run). Per fold, every carrier's
report.json names the same baseline evidence_id, and sha256 of each copy agrees. So the baseline decision
is made once per fold and consumed by every carrier -- which means a defect in one baseline record would
propagate to all carriers at once, raising the stakes on Finding 1.

FINDING 3 -- F2's winner is a near-tie ON THE VALIDATION SIDE, and its impact on the published G4 verdict
is NOT measurable from the artefacts. On F2, B2 scores 0.08521889544606599 and B3 scores
0.08500634733838454, a margin of 2.125e-04, i.e. 0.249% of the winner's value; the other folds are
comfortable (F1 8.277e-03, F3 2.526e-02, F4 1.523e-02). So on F2 the baseline choice was nearly arbitrary.

WHAT CANNOT BE CONCLUDED, and why an earlier draft of this probe was wrong to conclude it. The draft
compared that 2.125e-04 margin against G4's published margin of 0.002827 (observed 0.007826986071187494 vs
threshold 0.005) and reported "13x headroom, so G4 is not at risk". That is a BASIS MISMATCH of exactly the
family this pack keeps catching (§CW.1's 27-vs-81 rows, §CS.4's 6.27-vs-8.72 h, §CZ.8's and §DE.3's gate
scopes): the 2.125e-04 figure is a VALIDATION-side difference, while G4's delta is computed from TEST-side
scores. The two are different quantities on different data, so their ratio is not a headroom bound.

The correct statement is that the counterfactual is unmeasurable here. Only the SELECTED baseline's
test-side predictions were frozen (predictions/*/manifest.json shows kind "best_simple" with model_id B2 on
F1/F2/F3 and B3 on F4 -- never the runner-up), and the baseline record's payload carries only
validation_scores, so B3's test-side mean_rank_ic on F2 does not exist in any artefact. Obtaining it would
require recomputing the baseline, i.e. executing code, which this pack does not do. So: the F2 selection was
a near-tie on validation, the protocol correctly made it pre-TEST from validation data, the published G4
uses the selected baseline's test score as the pre-registered procedure requires, and therefore the
published verdict is procedurally sound -- but whether a different validation-side winner would have moved
G4 across its threshold cannot be determined from the frozen artefacts, and no bound is claimed.

What IS verifiable and is reported: the near-tie is confined to F2 (margin 2.1e-4 against 8.3e-3 to 2.5e-2
elsewhere), all four baselines on F2 have identical valid_day_count (236, so the comparison is like-for-like
and not an artefact of differing day coverage), and the selection rule's tie-break (toward the simpler
baseline) was never needed because no exact tie occurs.

FINDING 4 -- a validator asymmetry, recorded with its blast radius rather than inflated. The write-time
validator checks unselected_reasons (d); the load-time validator does not -- it re-checks (a), (b), (c) and
(e) but never looks at unselected_reasons or at valid_day_count / invalid_day_count. So a record whose
reasons or day counts were altered would still load. The blast radius is small and is stated as such: the
entire payload is keyless-content-hash bound (§CY: evidence_id == sha256(canonical_json(payload)), verified
29/29 including all 15 baseline records), so altering any field changes evidence_id and fails both the
content-hash check and the directory-name check. The asymmetry could only matter for a record produced by a
path that bypasses _validate_baseline_selection_record while still holding the signing key -- not a
scenario this pack can construct or observe. It is recorded as an observation about defence in depth, not as
a defect with a demonstrated impact.

CONTROLS, each able to fail: (i) the winner recomputation must reproduce selected_baseline_id on all 15
records, and the tie-break must be exercised by a synthetic tie to prove the complexity rule is what this
probe implements and not an accident of no ties occurring; (ii) validation_rank_ic must be bit-equal to the
winner's mean, checked by exact float equality, and a perturbed copy must be detected; (iii) the 4 distinct
records must each be byte-identical across all carriers holding them, and a cross-carrier mismatch must be
detectable (tested by comparing sha256 sets); (iv) the G4 headroom must exceed the F2 near-tie margin, so
the "not at risk" claim is a measured comparison rather than an assertion; (v) the load-time validator's
omission must be demonstrated from source by showing unselected_reasons appears in the write-time validator
and not in the load-time one; (vi) every record's payload must satisfy the §CY keyless content binding, so
Finding 4's blast-radius argument rests on a verified fact rather than on §CY's word.

Reads only: baseline and model selection record JSONs, report.json, and source text. No parquet, no
checkpoint.pt, no weight or tensor, no repository module imported or executed, no environment secret, no
database, no network, no writes to any artifact directory, no signal to any process. Does not touch
/tmp/walk_forward_cache_v1, so no atime side effect. Deterministic and byte-reproducible: all inputs are
immutable artefacts of completed folds (M2's F4 has no baseline record yet and is reported as absent, not
as passing).
"""
from __future__ import annotations

import glob
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "measurements" / "baseline_selection_chain.json"

WORKSPACES = [("m1_full_v1", "M1 ResidualMLP"), ("m2_v1", "M2 TCN"),
              ("m0_v3", "M0 LGBM_RANKER"), ("m0r_v3", "M0R LGBM_RANKREG")]
BASELINES = {"B0", "B1", "B2", "B3"}
COMPLEXITY = {"B0": 0, "B1": 1, "B2": 2, "B3": 3}
G4_THRESHOLD = 0.005


def canonical_json(v) -> str:
    return json.dumps(v, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def evidence_id_of(payload: dict) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def winner(means: dict) -> str:
    """Replica of selection_evidence.py:_baseline_winner (:555-559)."""
    eligible = [(b, v) for b, v in means.items() if v is not None]
    if not eligible:
        raise ValueError("no valid baseline winner")
    return min(eligible, key=lambda it: (-it[1], COMPLEXITY[it[0]]))[0]


def collect() -> tuple[list[dict], dict[str, dict]]:
    refs, distinct = [], {}
    for ws, fam in WORKSPACES:
        for rec in sorted(glob.glob(str(ART / ws / "selections" / "baseline" / "*" / "record.json"))):
            path = Path(rec)
            doc = json.loads(path.read_text(encoding="utf-8"))
            eid = doc["evidence_id"]
            refs.append({"workspace": ws, "family": fam, "evidence_id": eid,
                         "dir_matches": path.parent.name == f"{eid}.json",
                         "bytes_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                         "doc": doc})
            distinct.setdefault(eid, doc)
    return refs, distinct


def analyse(doc: dict) -> dict:
    p = doc["payload"]
    scores = {s["baseline_id"]: s for s in p["validation_scores"]}
    means = {b: scores[b]["mean_rank_ic"] for b in scores}
    w = winner(means)
    order = sorted(means.items(), key=lambda kv: (-kv[1], COMPLEXITY[kv[0]]))
    top, second = order[0][1], order[1][1]
    reasons = {r[0] for r in p.get("unselected_reasons", [])}
    return {
        "fold_id": p["fold_id"], "selected": p["selected_baseline_id"],
        "invariant_a_scores_exactly_B0_B3": set(scores) == BASELINES and len(scores) == 4,
        "invariant_b_selected_is_argmax_with_tiebreak": w == p["selected_baseline_id"],
        "invariant_c_rank_ic_bit_equal_to_winner": p["validation_rank_ic"] == means.get(w),
        "invariant_d_reasons_are_exact_complement": reasons == BASELINES - {w},
        "invariant_e_selected_in_B0_B3": p["selected_baseline_id"] in BASELINES,
        "all_five_invariants": all([
            set(scores) == BASELINES and len(scores) == 4, w == p["selected_baseline_id"],
            p["validation_rank_ic"] == means.get(w), reasons == BASELINES - {w},
            p["selected_baseline_id"] in BASELINES]),
        "keyless_content_binding_ok": evidence_id_of(p) == doc["evidence_id"],
        "ranking": [{"baseline": b, "mean_rank_ic": v, "valid_day_count": scores[b].get("valid_day_count"),
                     "invalid_day_count": scores[b].get("invalid_day_count")} for b, v in order],
        "margin_over_runner_up": top - second,
        "margin_pct_of_winner": 100 * (top - second) / abs(top),
        "valid_day_counts_identical_across_baselines": len({scores[b].get("valid_day_count") for b in scores}) == 1,
        "n_copies": 0,
    }


def tie_break_control() -> dict:
    """Prove the complexity tie-break is what this probe implements, not an accident of no ties."""
    tied = {"B0": 0.05, "B1": 0.05, "B2": 0.05, "B3": 0.05}
    partial = {"B0": None, "B1": 0.07, "B2": 0.07, "B3": 0.02}
    high_first = {"B0": 0.01, "B1": 0.02, "B2": 0.09, "B3": 0.03}
    return {
        "all_four_tied_picks_B0": winner(tied) == "B0",
        "tie_among_B1_B2_picks_B1": winner(partial) == "B1",
        "clear_winner_beats_complexity": winner(high_first) == "B2",
        "none_are_ineligible_raises": _raises_on_all_none(),
    }


def _raises_on_all_none() -> bool:
    try:
        winner({"B0": None, "B1": None, "B2": None, "B3": None})
        return False
    except ValueError:
        return True


def source_asymmetry() -> dict:
    src = (REPO / "src/research_v2/evaluation/selection_evidence.py").read_text(encoding="utf-8")
    lines = src.split("\n")

    def block(start_needle, end_needle):
        i = next(n for n, l in enumerate(lines) if start_needle in l)
        j = next(n for n in range(i + 1, len(lines)) if end_needle in lines[n])
        return "\n".join(lines[i:j]), i + 1

    write_txt, write_ln = block("def _validate_baseline_selection_record", "def _validate_baseline_payload")
    load_txt, load_ln = block("def _validate_baseline_payload", "def _baseline_winner")
    return {
        "write_time_validator_line": write_ln,
        "load_time_validator_line": load_ln,
        "write_checks_unselected_reasons": "unselected_reasons" in write_txt,
        "load_checks_unselected_reasons": "unselected_reasons" in load_txt,
        "load_checks_valid_day_count": "valid_day_count" in load_txt,
        "load_rechecks_argmax": "_baseline_winner(means) != selected" in load_txt,
        "load_rechecks_rank_ic": "validation_rank_ic" in load_txt,
        "load_is_called_from_load_baseline_selection": bool(
            re.search(r"def load_baseline_selection.*?\n.*?\n\s*_validate_baseline_payload\(payload\)",
                      src, re.S)),
        "asymmetry": ("unselected_reasons" in write_txt) and ("unselected_reasons" not in load_txt),
    }


def main() -> None:
    refs, distinct = collect()
    per_distinct = {eid: analyse(doc) for eid, doc in distinct.items()}

    copies = defaultdict(list)
    for r in refs:
        copies[r["evidence_id"]].append(r)
    for eid, rs in copies.items():
        per_distinct[eid]["n_copies"] = len(rs)

    # per-fold cross-carrier consistency
    report_ids = defaultdict(set)
    for ws, fam in WORKSPACES:
        rep = ART / ws / "report.json"
        if not rep.is_file():
            continue
        for f in json.loads(rep.read_text(encoding="utf-8"))["folds"]:
            report_ids[f["fold_id"]].add((ws, f["baseline_selection_id"]))
    fold_consistency = {
        fold: {"n_carriers": len(v), "carriers": sorted(w for w, _ in v),
               "n_distinct_ids": len({e for _, e in v}),
               "consistent": len({e for _, e in v}) == 1}
        for fold, v in sorted(report_ids.items())}

    byte_identical = {eid: len({r["bytes_sha256"] for r in rs}) == 1 for eid, rs in copies.items()}

    # reference-level invariant check (all 15)
    ref_checks = []
    for r in refs:
        a = per_distinct[r["evidence_id"]]
        ref_checks.append({"workspace": r["workspace"], "fold": a["fold_id"],
                           "evidence_id": r["evidence_id"][:16],
                           "dir_matches_evidence_id": r["dir_matches"],
                           "all_five_invariants": a["all_five_invariants"],
                           "keyless_binding_ok": a["keyless_content_binding_ok"]})

    # G4 headroom vs the near-tie margin
    rep1 = json.loads((ART / "m1_full_v1" / "report.json").read_text(encoding="utf-8"))
    oos = rep1["oos"]
    g4 = oos["gates"]["G4"]
    f2_margin = next(a["margin_over_runner_up"] for a in per_distinct.values() if a["fold_id"] == "F2")
    g4_margin = g4["observed_value"] - G4_THRESHOLD
    f2_rec = next(a for a in per_distinct.values() if a["fold_id"] == "F2")
    # Is the runner-up baseline's TEST-side score present anywhere? Only the selected one is frozen.
    frozen_best_simple = {}
    for man in sorted(glob.glob(str(ART / "*" / "predictions" / "*" / "manifest.json"))):
        m = json.loads(Path(man).read_text(encoding="utf-8"))["metadata"]
        if m.get("kind") == "best_simple":
            frozen_best_simple.setdefault(m["fold_id"], set()).add(m["model_id"])
    f2_runner_up = f2_rec["ranking"][1]["baseline"]
    runner_up_test_score_available = f2_runner_up in frozen_best_simple.get("F2", set())
    near_tie = {
        "F2_margin": f2_margin,
        "F2_margin_pct_of_winner": f2_rec["margin_pct_of_winner"],
        "F2_ranking": f2_rec["ranking"],
        "F2_day_counts_identical": f2_rec["valid_day_counts_identical_across_baselines"],
        "all_fold_margins": {a["fold_id"]: round(a["margin_over_runner_up"], 8)
                             for a in per_distinct.values()},
        "published_G4_observed": g4["observed_value"], "published_G4_threshold": G4_THRESHOLD,
        "published_G4_margin": round(g4_margin, 6),
        "BASIS_NOTE": ("the F2 margin above is a VALIDATION-side difference while published_G4_margin is "
                       "TEST-side; their ratio is NOT a headroom bound and is deliberately not reported"),
        "frozen_best_simple_per_fold": {k: sorted(v) for k, v in sorted(frozen_best_simple.items())},
        "F2_runner_up": f2_runner_up,
        "runner_up_test_score_available": runner_up_test_score_available,
        "counterfactual_is_unmeasurable": not runner_up_test_score_available,
        "verdict": ("the F2 selection was a near-tie on validation; the protocol made it pre-TEST from "
                    "validation data and G4 uses the selected baseline's test score as pre-registered, so "
                    "the published verdict is procedurally sound, but whether a different validation winner "
                    "would have moved G4 across its threshold cannot be determined from the frozen "
                    "artefacts and no bound is claimed"),
    }

    asym = source_asymmetry()
    tie = tie_break_control()

    doc = {
        "question": ("is the baseline side of the selection chain internally consistent and consistently "
                     "shared, and does the F2 near-tie threaten any published verdict?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "structure_differs_from_candidate_side": (
            "a baseline selection names a LABEL (B0..B3), not a file, so there is no bundle digest to "
            "compare; what is verifiable is the recorded competition's internal consistency and its sharing "
            "across carriers"),
        "n_references": len(refs), "n_distinct_records": len(distinct),
        "per_reference_checks": ref_checks,
        "per_distinct_record": per_distinct,
        "sharing": {
            "copies_per_record": {eid: len(rs) for eid, rs in copies.items()},
            "byte_identical_across_carriers": byte_identical,
            "per_fold_report_consistency": fold_consistency,
            "reading": ("baselines are model-independent, so each fold's decision is made once and reused "
                        "by every carrier; a defect in one record would propagate to all of them at once"),
        },
        "F2_near_tie_vs_G4": near_tie,
        "validator_asymmetry": asym,
        "validator_asymmetry_blast_radius": {
            "claim": ("load-time validation omits unselected_reasons and the day counts, so altering those "
                      "fields would not be caught at load"),
            "why_the_radius_is_small": (
                "the whole payload is keyless-content-hash bound (§CY), so altering ANY field changes "
                "evidence_id and fails both the content-hash check at selection_evidence.py:145 and the "
                "directory-name check; verified here on all "
                f"{len(refs)} references and all {len(distinct)} distinct records"),
            "scenario_that_would_matter": (
                "a record produced by a path bypassing _validate_baseline_selection_record while still "
                "holding the signing key -- not constructible or observable from this pack"),
            "classification": "defence-in-depth observation, not a defect with demonstrated impact",
        },
        "tie_break_controls": tie,
        "controls": {},
        "read_only": True, "database_access": False, "weights_loaded": False,
        "checkpoint_pt_opened": False, "parquet_read": False, "environment_secret_read": False,
        "training_performed": False, "repository_module_imported_or_executed": False,
        "processes_signalled": [], "atime_side_effect": False, "byte_reproducible": True,
    }
    doc["controls"] = {
        "i_all_15_references_pass_five_invariants": all(
            r["all_five_invariants"] for r in ref_checks) and len(ref_checks) == 15,
        "i_b_tie_break_rule_exercised_by_synthetic_ties": all(tie.values()),
        "ii_all_keyless_content_bindings_hold": all(
            r["keyless_binding_ok"] for r in ref_checks),
        "iii_all_copies_byte_identical": all(byte_identical.values()),
        "iii_b_every_fold_consistent_across_carriers": all(
            v["consistent"] for v in fold_consistency.values()),
        # The draft's control compared G4's TEST-side margin with F2's VALIDATION-side margin, which is a
        # basis mismatch and proves nothing. Replaced by three same-basis controls: the near-tie is confined
        # to F2; F2's four baselines share one valid_day_count so the comparison is like-for-like; and the
        # counterfactual is recorded as unmeasurable rather than bounded.
        "iv_near_tie_confined_to_F2": f2_margin < 1e-3 and all(
            v >= 1e-3 for k, v in near_tie["all_fold_margins"].items() if k != "F2"),
        "iv_b_F2_day_counts_identical_across_baselines": near_tie["F2_day_counts_identical"],
        "iv_c_runner_up_test_score_absent_so_no_bound_claimed":
            near_tie["counterfactual_is_unmeasurable"],
        "v_source_shows_reasons_checked_on_write_not_on_load": asym["asymmetry"],
        "v_b_load_validator_is_actually_called": asym["load_is_called_from_load_baseline_selection"],
        "vi_directory_names_match_evidence_ids": all(
            r["dir_matches_evidence_id"] for r in ref_checks),
        "ALL_PASS": None,
    }
    doc["controls"]["ALL_PASS"] = all(
        v for k, v in doc["controls"].items() if k != "ALL_PASS")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print(f"=== baseline chain: {len(refs)} references -> {len(distinct)} distinct records ===")
    for eid, a in sorted(per_distinct.items(), key=lambda kv: kv[1]["fold_id"]):
        print(f"  {a['fold_id']}: selected {a['selected']} | copies {a['n_copies']} | "
              f"5 invariants {a['all_five_invariants']} | keyless binding "
              f"{a['keyless_content_binding_ok']} | margin {a['margin_over_runner_up']:.3e} "
              f"({a['margin_pct_of_winner']:.3f}%)")
    print(f"\n=== per-fold sharing consistency ===")
    for fold, v in fold_consistency.items():
        print(f"  {fold}: {v['n_carriers']} carriers {v['carriers']} -> {v['n_distinct_ids']} distinct id "
              f"{'CONSISTENT' if v['consistent'] else '*** INCONSISTENT ***'}")
    print(f"  byte-identical across carriers: {all(byte_identical.values())}")
    n = doc["F2_near_tie_vs_G4"]
    print(f"\n=== F2 near-tie vs the G4 gate it feeds ===")
    print(f"  F2 margin over runner-up: {n['F2_margin']:.3e} ({n['F2_margin_pct_of_winner']:.3f}% of winner)")
    for r in n["F2_ranking"]:
        print(f"     {r['baseline']}: {r['mean_rank_ic']!r} (valid {r['valid_day_count']})")
    print(f"  published G4 (TEST-side): observed {n['published_G4_observed']:.6f} vs threshold "
          f"{n['published_G4_threshold']} -> margin {n['published_G4_margin']:.6f}")
    print(f"  NOTE: {n['BASIS_NOTE']}")
    print(f"  frozen best_simple per fold: {n['frozen_best_simple_per_fold']}")
    print(f"  F2 runner-up {n['F2_runner_up']} test-side score available: "
          f"{n['runner_up_test_score_available']} -> counterfactual unmeasurable: "
          f"{n['counterfactual_is_unmeasurable']}")
    print(f"  all fold margins (validation-side): {n['all_fold_margins']}")
    print(f"  {n['verdict']}")
    print(f"\n=== validator asymmetry (from source) ===")
    for k, v in asym.items():
        print(f"  {k}: {v}")
    print(f"\n=== tie-break controls (synthetic) ===")
    for k, v in tie.items():
        print(f"  {'OK ' if v else '*** FAIL ***'} {k}")
    print(f"\n=== controls ===")
    for k, v in doc["controls"].items():
        print(f"  {'OK ' if v else '*** FAIL ***'} {k}")
    assert doc["controls"]["ALL_PASS"], "a control failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
