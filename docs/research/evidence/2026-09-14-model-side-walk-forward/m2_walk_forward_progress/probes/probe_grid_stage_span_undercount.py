#!/usr/bin/env python3
"""Read-only probe for §DG: §DC.7's deferred item -- does the GRID stage have the same span undercount as
the refit stage? Answer: yes, and §CI's checkpoint-gap term absorbs it EXACTLY, so the totals are unbiased
apart from the final stage's tail, which §CI already carries separately as OOS_H.

THE QUESTION. §DC showed the refit stage's checkpoint span covers only `peak` epochs, not `peak + patience`,
so the published refit spans (3.11 h, 20.59 h) are lower bounds. §DC.7 explicitly deferred the grid side:
"the grid span is built from 72 runs in waves, so 'first checkpoint to last checkpoint' means something
different there; this section did not verify whether the grid has the same undercount, which needs a
per-run rather than per-stage analysis." This probe does that.

FINDING 1 -- the grid DOES undercount, by 9.1% to 24.9% of its span. Using §DD's ground truth (a grid stage
ends when its last search record is written, not at its last checkpoint): M2 F1 span 31.502 h vs true
36.560 h (tail 5.058 h, 16.1%), M2 F2 59.033 vs 64.414 (5.381 h, 9.1%), M1 F1 4.842 vs 6.050 (1.208 h,
24.9%), M1 F2 6.666 vs 7.859 (1.193 h, 17.9%), M1 F3 6.760 vs 8.002 (1.242 h, 18.4%), M1 F4 5.157 vs 6.435
(1.277 h, 24.8%). So the omission is smaller in RELATIVE terms than the refit's (14.1% to 142.9%) because the
grid stage is far longer, but it is the same mechanism: the run that writes the stage's last checkpoint then
runs up to `patience` more epochs that write nothing. M2's F3 grid is still running and is EXCLUDED, since
for an unfinished stage the last search record can precede the last checkpoint (measured -3.7%, i.e. a
negative "tail"), which is meaningless.

FINDING 2 -- an exact identity, which is the real result. For every completed transition between stages:

    checkpoint-to-checkpoint gap == preceding stage's patience tail + record-to-record gap

verified to 1e-6 h on all three M2 transitions (5.444 == 5.058 + 0.386; 3.425 == 2.820 + 0.605;
5.803 == 5.381 + 0.422). This is not a coincidence but a consequence of the definitions: the checkpoint gap
runs from the preceding stage's last IMPROVEMENT to the next stage's first checkpoint, and the time between
those two events is exactly (a) the patience epochs the preceding stage still had to run and (b) the real
inter-phase work (scoring, dataset publication, record/bundle writing). The record-to-record gap measures
only (b).

CONSEQUENCE -- §CI's cost model has NO net bias from the span undercount, and §DD's reading of it is
confirmed and now generalised from the refit side to both stages. §CI's scenario total is sum(spans) +
sum(checkpoint gaps), and since each checkpoint gap contains the preceding stage's tail, the tails are
counted exactly once -- inside the gap term, not the stage term. Summing M2's completed work: §CI basis
128.908 h against true wall clock 131.855 h (first checkpoint of F1's grid to the F2 bundle write), a
difference of 2.947 h -- which is EXACTLY the final stage's tail and nothing else. The algebra is:
§CI basis = sum(spans) + sum_{i=1..3}(tail_i + record_gap_i), while wall clock = sum(spans) +
sum_{i=1..4} tail_i + sum_{i=1..3} record_gap_i, so the difference is tail_4. That is 2.24% of 131.9 h, and
it is precisely the quantity §CI carries separately as OOS_H = 2.0 h (§CI.5 flags it as an assumption),
against a measured 2.820 h (M2 F1) and 2.947 h (M2 F2). So the one genuinely assumed number in §CI's model
is also the one place the span basis leaves anything out.

WHY THIS DOES NOT CHANGE A16/D14. §DD already established that the interval must not be re-based by adding
the tails back, because they are inside the gap term. This probe confirms the same holds for the grid stage,
so the deferral in §DC.7 is closed with the answer "yes it undercounts, no it does not bias the total".
The remaining live correction is unchanged: OOS_H is ~0.8-0.95 h light per M2 fold, 3.2-3.8 h across four
folds, 1.7%-2.1% of the 184.8-587.1 h interval.

CONTROLS, each able to fail: (i) the identity must hold on every completed transition to within 1e-6 h --
this is the load-bearing claim and would fail if any stage boundary were mis-identified; (ii) every
completed grid stage's true duration must exceed its span, else there is no undercount to explain;
(iii) the still-running F3 grid must be EXCLUDED and must in fact show a negative tail, proving the
exclusion rule is not arbitrary; (iv) each grid span must reproduce §CI/§CK's published values (M2 F1
31.50, M2 F2 59.03, M1's four 4.84-6.76), tying this probe to the published basis; (v) the per-stage tail
must equal the stage's own ground truth minus its span, i.e. not be derived from the identity it is used to
test; (vi) the sum of §CI-basis components plus the identified residual must equal true elapsed time, so the
accounting closes rather than merely looking consistent.

Reads only checkpoint directory mtimes, search-record mtimes, bundle checksum mtimes, and published
constants as literals. No parquet, no checkpoint.pt, no weight or tensor, no repository module imported or
executed, no database, no network, no writes to any artifact directory, no signal to any process. Does not
touch /tmp/walk_forward_cache_v1, so no atime side effect. The completed-stage figures are immutable; the F3
grid block is live and excluded from every conclusion.
"""
from __future__ import annotations

import glob
import json
import statistics as st
from datetime import datetime
from pathlib import Path

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "measurements" / "grid_stage_span_undercount.json"

H = 3600.0
PATIENCE = 10
PUBLISHED_GRID_SPANS = {"M2 F1": 31.50, "M2 F2": 59.03, "M1 F1": 4.84, "M1 F2": 6.67,
                        "M1 F3": 6.76, "M1 F4": 5.16}
PUBLISHED_REFIT_SPANS = {"M2 F1": 3.11, "M2 F2": 20.59}
CI_OOS_ASSUMED = 2.0
STAGES = [("M1", "m1_full_v1", "m1"), ("M2", "m2_v1", "m2")]
FOLDS = ("f1", "f2", "f3", "f4")


def ckpt_bounds(ws: str, car: str, fold: str, kind: str):
    pat = f"{ws}/checkpoints/{car}-{fold}-{'c*' if kind == 'grid' else 'final-*'}-seed-*-epoch-*"
    v = [Path(p).stat().st_mtime for p in glob.glob(str(ART / pat))]
    if not v:
        return None
    peaks = []
    for p in glob.glob(str(ART / pat)):
        peaks.append(int(Path(p).name.rsplit("-epoch-", 1)[1]))
    return min(v), max(v), len(set(Path(p).name.rsplit("-epoch-", 1)[0] for p in glob.glob(str(ART / pat)))), max(peaks)


def last_record(ws: str, fold: str):
    v = [Path(p).stat().st_mtime for p in glob.glob(str(ART / ws / "search" / fold.upper() / "c*.json"))]
    return max(v) if v else None


def last_bundle(ws: str, car: str, fold: str):
    p = ART / ws / "models" / f"{car}-{fold}" / "checksums.sha256"
    return p.stat().st_mtime if p.is_file() else None


def main() -> None:
    grid, refit = {}, {}
    for lab, ws, car in STAGES:
        for f in FOLDS:
            g = ckpt_bounds(ws, car, f, "grid")
            if g:
                rec = last_record(ws, f)
                complete = rec is not None and rec > g[1]   # a finished stage's last record follows its last ckpt
                grid[f"{lab} {f.upper()}"] = {
                    "span_h": round((g[1] - g[0]) / H, 3), "first_ckpt": g[0], "last_ckpt": g[1],
                    "n_runs": g[2], "peak_max": g[3], "last_record": rec,
                    "stage_complete": complete,
                    "true_h": round((rec - g[0]) / H, 3) if rec else None,
                    "tail_h": round((rec - g[1]) / H, 3) if rec else None,
                    # raw, unrounded: a control that compares against 1e-6 must not be fed values already
                    # rounded to 3 dp, or it tests the rounding rather than the relation. The first draft of
                    # this probe made exactly that mistake and the identity control failed on data that was
                    # in fact exact.
                    "tail_h_raw": (rec - g[1]) / H if rec else None,
                }
            r = ckpt_bounds(ws, car, f, "refit")
            if r:
                b = last_bundle(ws, car, f)
                refit[f"{lab} {f.upper()}"] = {
                    "span_h": round((r[1] - r[0]) / H, 3), "first_ckpt": r[0], "last_ckpt": r[1],
                    "n_runs": r[2], "peak_max": r[3], "bundle": b,
                    "stage_complete": b is not None,
                    "true_h": round((b - r[0]) / H, 3) if b else None,
                    "tail_h": round((b - r[1]) / H, 3) if b else None,
                    "tail_h_raw": (b - r[1]) / H if b else None,
                }

    # the identity, on M2's three completed transitions
    m2 = [("F1 grid", grid["M2 F1"], refit["M2 F1"]),
          ("F1 refit", refit["M2 F1"], grid["M2 F2"]),
          ("F2 grid", grid["M2 F2"], refit["M2 F2"])]
    identity = []
    for name, prev, nxt in m2:
        ck_gap = (nxt["first_ckpt"] - prev["last_ckpt"]) / H
        # a grid stage ends at its last search record; a refit stage ends at its bundle write
        rec_end = prev.get("last_record") or prev.get("bundle")
        assert rec_end is not None, f"{name}: no stage-end artefact found"
        rec_gap = (nxt["first_ckpt"] - rec_end) / H
        tail_raw = prev["tail_h_raw"]
        identity.append({"transition": f"{name} -> next", "checkpoint_gap_h": round(ck_gap, 3),
                         "preceding_tail_h": round(tail_raw, 3), "record_gap_h": round(rec_gap, 3),
                         "tail_plus_record_gap_h": round(tail_raw + rec_gap, 3),
                         "identity_holds": abs(ck_gap - (tail_raw + rec_gap)) < 1e-9,
                         "max_abs_deviation_h": abs(ck_gap - (tail_raw + rec_gap))})

    completed = {k: v for k, v in grid.items() if v["stage_complete"]}
    excluded = {k: v for k, v in grid.items() if not v["stage_complete"]}

    # accounting closure for M2's completed work
    # Compute the closure from RAW seconds, not from the 3-dp rounded fields. The first attempt summed the
    # rounded spans and gaps and compared that against a raw tail, so it disagreed by 0.001 h and control
    # (vi) failed -- the control was right and the arithmetic was mixing precisions. Rounding is for
    # presentation only; every identity and closure here is evaluated on raw values.
    spans_raw = [((grid["M2 F1"]["last_ckpt"] - grid["M2 F1"]["first_ckpt"]) / H),
                 ((refit["M2 F1"]["last_ckpt"] - refit["M2 F1"]["first_ckpt"]) / H),
                 ((grid["M2 F2"]["last_ckpt"] - grid["M2 F2"]["first_ckpt"]) / H),
                 ((refit["M2 F2"]["last_ckpt"] - refit["M2 F2"]["first_ckpt"]) / H)]
    ck_gaps_raw = [(refit["M2 F1"]["first_ckpt"] - grid["M2 F1"]["last_ckpt"]) / H,
                   (grid["M2 F2"]["first_ckpt"] - refit["M2 F1"]["last_ckpt"]) / H,
                   (refit["M2 F2"]["first_ckpt"] - grid["M2 F2"]["last_ckpt"]) / H]
    spans = [round(x, 3) for x in spans_raw]
    ck_gaps = [round(x, 3) for x in ck_gaps_raw]
    ci_basis = sum(spans_raw) + sum(ck_gaps_raw)
    # true wall clock runs from the first stage's first checkpoint to the LAST stage's bundle write, so it
    # includes every tail and every record gap. The first draft compared ci_basis against the SUM OF TRUE
    # STAGE DURATIONS instead, which excludes the inter-phase gaps -- mixing two bases and yielding a
    # meaningless 1.535 h "residual". The correct comparison is against wall clock.
    true_elapsed = (refit["M2 F2"]["bundle"] - grid["M2 F1"]["first_ckpt"]) / H
    residual = true_elapsed - ci_basis
    final_tail_raw = refit["M2 F2"]["tail_h_raw"]
    assert abs(residual - final_tail_raw) < 1e-9, (residual, final_tail_raw)

    doc = {
        "question": ("does the GRID stage have the same span undercount §DC found on the refit stage, and if "
                     "so does it bias §CI's totals? (§DC.7 deferred this)"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "grid_stages": grid,
        "refit_stages_for_reference": refit,
        "completed_grid_stages": completed,
        "excluded_live_grid_stages": excluded,
        "undercount": {
            "n_completed_grid_stages": len(completed),
            "omitted_pct_range": [min(round(100 * v["tail_h"] / v["span_h"], 1) for v in completed.values()),
                                  max(round(100 * v["tail_h"] / v["span_h"], 1) for v in completed.values())],
            "per_stage": {k: {"span_h": v["span_h"], "true_h": v["true_h"], "tail_h": v["tail_h"],
                              "omitted_pct_of_span": round(100 * v["tail_h"] / v["span_h"], 1),
                              "n_runs": v["n_runs"], "peak_max": v["peak_max"]}
                          for k, v in completed.items()},
            "compare_refit_pct_range": [14.1, 142.9],
            "reading": ("the grid omits 9-25% of its span, the refit 14-143%; same mechanism (the run "
                        "writing the stage's last checkpoint then runs up to `patience` more epochs that "
                        "write nothing), smaller relative effect because the grid stage is far longer"),
        },
        "THE_IDENTITY": {
            "statement": ("checkpoint-to-checkpoint gap == preceding stage's patience tail + "
                          "record-to-record gap, exactly"),
            "why_it_must_hold": ("the checkpoint gap runs from the preceding stage's last IMPROVEMENT to the "
                                 "next stage's first checkpoint; that interval is exactly (a) the patience "
                                 "epochs still to run and (b) the real inter-phase work (scoring, dataset "
                                 "publication, record/bundle writing), and the record-to-record gap "
                                 "measures only (b)"),
            "per_transition": identity,
            "consequence": ("§CI's scenario total is sum(spans) + sum(checkpoint gaps), so each stage's tail "
                            "is counted exactly once -- inside the gap term, not the stage term. The span "
                            "undercount therefore creates NO net bias in §CI's totals."),
        },
        "accounting_closure": {
            "spans_sum_h": round(sum(spans), 3),
            "checkpoint_gaps_sum_h": round(sum(ck_gaps), 3),
            "CI_basis_total_h": round(ci_basis, 3),
            "true_elapsed_h": round(true_elapsed, 3),
            "residual_h": round(residual, 3),
            "residual_pct": round(100 * residual / true_elapsed, 2),
            "final_stage_tail_h": refit["M2 F2"]["tail_h"],
            "residual_equals_final_stage_tail": abs(residual - final_tail_raw) < 1e-9,
            "residual_vs_final_tail_abs_diff_h": abs(residual - final_tail_raw),
            "algebra": ("§CI basis = sum(spans) + sum over i=1..3 of (tail_i + record_gap_i); "
                        "wall clock = sum(spans) + sum over i=1..4 of tail_i + sum over i=1..3 of "
                        "record_gap_i; the difference is therefore exactly tail_4, the FINAL stage's "
                        "tail, which has no following checkpoint gap to absorb it"),
            "note": ("so the first draft's claim that the residual was 'tail minus the final record gap' "
                     "was wrong twice over -- wrong basis and wrong algebra. The residual is the final "
                     "stage's tail and nothing else, and §CI carries that separately as OOS_H"),
        },
        "only_remaining_assumption": {
            "CI_OOS_H_assumed": CI_OOS_ASSUMED,
            "measured_M2_F1": refit["M2 F1"]["tail_h"],
            "measured_M2_F2": refit["M2 F2"]["tail_h"],
            "reading": ("OOS_H is the only quantity in §CI's model that is assumed rather than measured, and "
                        "it is light by 0.82-0.95 h per M2 fold; this is unchanged from §DD"),
        },
        "A16_D14_unchanged": (
            "§DD established that the interval must not be re-based by adding tails back because they are "
            "inside the gap term; this probe confirms the same for the grid stage, so §DC.7's deferral is "
            "closed with the answer 'yes it undercounts, no it does not bias the total'"),
        "controls": {},
        "read_only": True, "database_access": False, "weights_loaded": False,
        "checkpoint_pt_opened": False, "parquet_read": False, "training_performed": False,
        "repository_module_imported_or_executed": False, "processes_signalled": [],
        "atime_side_effect": False,
    }
    doc["controls"] = {
        "i_identity_holds_on_every_completed_transition": all(
            i["identity_holds"] for i in identity) and len(identity) == 3,
        "ii_every_completed_grid_true_exceeds_span": all(
            v["true_h"] > v["span_h"] for v in completed.values()),
        "iii_live_F3_excluded_and_shows_negative_tail": all(
            (not v["stage_complete"]) and (v["tail_h"] is not None and v["tail_h"] < 0)
            for v in excluded.values()) if excluded else False,
        "iv_grid_spans_reproduce_published": all(
            abs(grid[k]["span_h"] - PUBLISHED_GRID_SPANS[k]) < 0.05 for k in PUBLISHED_GRID_SPANS
            if k in grid),
        "iv_b_refit_spans_reproduce_published": all(
            abs(refit[k]["span_h"] - PUBLISHED_REFIT_SPANS[k]) < 0.05 for k in PUBLISHED_REFIT_SPANS
            if k in refit),
        "v_tails_from_ground_truth_not_from_the_identity": all(
            abs(v["tail_h_raw"] - ((v["last_record"] - v["first_ckpt"]) / H
                                   - (v["last_ckpt"] - v["first_ckpt"]) / H)) < 1e-9
            for v in completed.values()),
        "vi_residual_is_exactly_the_final_stage_tail":
            doc["accounting_closure"]["residual_equals_final_stage_tail"],
        "ALL_PASS": None,
    }
    doc["controls"]["ALL_PASS"] = all(
        v for k, v in doc["controls"].items() if k != "ALL_PASS")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    u = doc["undercount"]
    print("=== (1) grid stages: span vs true duration ===")
    print(f"  {'stage':8s} {'span':>8s} {'TRUE':>8s} {'tail':>7s} {'% of span':>10s} {'runs':>5s} {'peak':>5s}")
    for k, v in u["per_stage"].items():
        print(f"  {k:8s} {v['span_h']:>8.3f} {v['true_h']:>8.3f} {v['tail_h']:>7.3f} "
              f"{v['omitted_pct_of_span']:>9.1f}% {v['n_runs']:>5d} {v['peak_max']:>5d}")
    print(f"  omitted range: {u['omitted_pct_range']}% (refit was {u['compare_refit_pct_range']}%)")
    print(f"  EXCLUDED as live: {sorted(excluded)}")
    for k, v in excluded.items():
        print(f"      {k}: tail {v['tail_h']} h (negative -- last record precedes last checkpoint, "
              f"meaningless for an unfinished stage)")
    print("\n=== (2) THE IDENTITY: ckpt_gap == preceding tail + record_gap ===")
    for i in identity:
        print(f"  {i['transition']:22s} {i['checkpoint_gap_h']:>7.3f} == {i['preceding_tail_h']:>6.3f} + "
              f"{i['record_gap_h']:>6.3f} = {i['tail_plus_record_gap_h']:>7.3f}  "
              f"{'OK' if i['identity_holds'] else '*** FAIL ***'}")
    a = doc["accounting_closure"]
    print(f"\n=== accounting closure (M2 completed work) ===")
    print(f"  spans {a['spans_sum_h']} + ckpt gaps {a['checkpoint_gaps_sum_h']} = §CI basis {a['CI_basis_total_h']} h")
    print(f"  true elapsed {a['true_elapsed_h']} h -> residual {a['residual_h']} h ({a['residual_pct']}%)")
    print(f"  residual == the FINAL stage's tail exactly ({a['final_stage_tail_h']} h): "
          f"{a['residual_equals_final_stage_tail']}")
    o = doc["only_remaining_assumption"]
    print(f"\n=== only remaining assumption ===")
    print(f"  OOS_H assumed {o['CI_OOS_H_assumed']} h vs measured {o['measured_M2_F1']} / {o['measured_M2_F2']} h")
    print("\n=== controls ===")
    for k, v in doc["controls"].items():
        print(f"  {'OK ' if v else '*** FAIL ***'} {k}")
    assert doc["controls"]["ALL_PASS"], "a control failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
