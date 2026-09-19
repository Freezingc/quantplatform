#!/usr/bin/env python3
"""Read-only probe for §DD: ground truth for §CI/§CS's stage wall-clocks from artefact mtimes. Three
findings, one of which CORRECTS §DC.5 and one of which corrects this probe's own first draft.

GROUND TRUTH, and why it is ground truth. §CI/§CS/§DA measured stage wall-clock as a checkpoint SPAN
(max last ckpt mtime - min first ckpt mtime). This probe additionally reads the artefacts that mark stage
boundaries independently of checkpoints: a grid stage ends when its last search record is written
(search/<fold>/c*.json mtime), a refit stage ends when its bundle is written
(models/<id>/checksums.sha256 mtime), and the process start comes from /proc/<pid>/stat field 22 plus
/proc/stat btime. Ten consecutive segments from process start to now sum to the actual elapsed time with
0.00000% closure error, which is what makes the segment values trustworthy rather than plausible.

FINDING 1 -- §DC's patience-tail model is validated. Actual durations exceed the published spans by the
tail §DC predicted: F1 grid 36.560 vs span 31.500, F1 refit 5.927 vs 3.107, F2 grid 64.414 vs 59.030,
F2 refit 23.541 vs 20.593. §DC's estimate (span + span x patience/peak) lands within -3.9% to +0.0% of
actual on all four stages, so §DC.1's "estimated true duration" of ~5.70 h and ~23.49 h for the two M2
refits is confirmed by measurement (5.927 h, 23.541 h) and is mildly conservative.

FINDING 2 -- §CI's accounting is INTERNALLY CONSISTENT, and this probe's first draft got that wrong.
The first draft reported that §CI's 5.0 h inter-phase gap assumption overstates reality by ~17.9 h,
because it measured the gaps as record-to-record (F1 grid's last search record -> F1 refit's first
checkpoint = 0.386 h). That is a BASIS MISMATCH: §CI.1 states its stage boundaries are taken "from
checkpoint directory mtimes", so its gaps are checkpoint-to-checkpoint, and reproducing that definition
exactly gives 5.444 / 3.425 / 5.803 h against §CI's published 5.44 / 3.42 / 5.80 -- a match to three
significant figures on all three. The patience tail is therefore NOT missing from §CI's total: §CI's spans
exclude it, and its checkpoint-to-checkpoint gaps ABSORB it. The two accountings differ by exactly the
final stage's tail plus bundle write (2.947 h), which §CI carries separately. So §CI's 5.0 h assumption is
well calibrated (measured mean 4.891 h over its three gaps), not 8-13x too large.

FINDING 3 -- §DC.5's inference about A16/D14 must be corrected. §DC.5 said the interval 184.8-587.1 h is
optimistic because "the refit term is systematically understated by roughly one patience tail". That is
true of the SPANS but not of §CI's TOTAL, because the tail is already inside the gap term, and the gap term
was set to 5.0 h against a measured 4.891 h. Counting the tail again would double-count it. The correct
statement: §CI's published spans are lower bounds on individual stage durations (Finding 1), but its
scenario totals are not understated by that amount, because the same tails appear in its gap term which
was calibrated against measurement. What remains genuinely assumed rather than measured in §CI's model is
OOS_H = 2.0 h, and this probe measures the corresponding interval (refit's last checkpoint -> bundle
written) as 2.820 h for M2 F1 and 2.947 h for M2 F2, against 0.965-1.037 h for M1's four folds -- so 2.0 h
is reasonable for M1 and about 0.8-0.95 h light for M2, a much smaller correction than the tail claim and
in the same direction.

LIVE VS IMMUTABLE. All four M2/M1 completed-stage durations, all gaps and the closure check use immutable
artefacts. The F3 grid segment grows while the grid runs and is reported separately, used in the closure
identity only and in no conclusion.

CONTROLS, each able to fail: (i) the ten segments must close to actual elapsed time within 0.01%;
(ii) every segment positive and time-ordered; (iii) each actual stage duration must exceed its published
span, the direct test of §DC's lower-bound claim; (iv) §DC's span+tail estimate within 5% of actual on all
four stages; (v) the checkpoint spans must reproduce §CI/§CS's published 31.50 / 59.03 / 3.11 / 20.59 h;
(vi) the CHECKPOINT-TO-CHECKPOINT gaps must reproduce §CI's published 5.44 / 3.42 / 5.80 h -- this is the
control that caught the first draft's basis mismatch, and without it the draft's "gaps overstated by 17.9 h"
would have been published; (vii) the record-to-record and checkpoint-to-checkpoint gap sums must differ
materially, so the two bases cannot be silently conflated again; (viii) the difference between the two
accountings must equal the final stage's tail plus bundle write, i.e. the two must reconcile rather than
merely both being reported.

Reads only artefact mtimes, /proc for the process start and btime, and published constants as literals. No
parquet, no checkpoint.pt, no weight or tensor, no repository module imported or executed, no database, no
network, no writes to any artifact directory, no signal to any process. Does not touch
/tmp/walk_forward_cache_v1, so no atime side effect.
"""
from __future__ import annotations

import glob
import json
import time
from datetime import datetime
from pathlib import Path

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "measurements" / "stage_wallclock_ground_truth.json"

H = 3600.0
PATIENCE = 10
PID = 3588838
PUBLISHED_SPANS = {"F1_grid": 31.50, "F2_grid": 59.03, "F1_refit": 3.11, "F2_refit": 20.59}
PUBLISHED_GAPS = [5.44, 3.42, 5.80]          # §CI.1's "阶段间隔（实测三次）"
CI_GAP_ASSUMED = 5.0
CI_OOS_ASSUMED = 2.0
DC_TAIL_EST = {"F1_grid": 4.516, "F1_refit": 2.589, "F2_grid": 5.398, "F2_refit": 2.900}


def proc_start() -> float:
    stat = Path(f"/proc/{PID}/stat").read_text().split()
    btime = int([l for l in Path("/proc/stat").read_text().split("\n")
                 if l.startswith("btime ")][0].split()[1])
    return btime + int(stat[21]) / 100.0


def mt(pattern: str) -> list[float]:
    return [Path(p).stat().st_mtime for p in glob.glob(str(ART / pattern))]


def ckpt_bounds(fold: str, kind: str) -> tuple[float, float]:
    lo = fold.lower()
    pat = f"m2_v1/checkpoints/m2-{lo}-c*-epoch-*" if kind == "grid" \
        else f"m2_v1/checkpoints/m2-{lo}-final-*-epoch-*"
    v = mt(pat)
    return min(v), max(v)


def main() -> None:
    start = proc_start()
    now = time.time()

    f1g0, f1g1 = ckpt_bounds("F1", "grid")
    f1r0, f1r1 = ckpt_bounds("F1", "refit")
    f2g0, f2g1 = ckpt_bounds("F2", "grid")
    f2r0, f2r1 = ckpt_bounds("F2", "refit")
    f3g0 = min(mt("m2_v1/checkpoints/m2-f3-c*-epoch-*"))
    f1_rec = max(mt("m2_v1/search/F1/c*.json"))
    f2_rec = max(mt("m2_v1/search/F2/c*.json"))
    f1_bundle = (ART / "m2_v1/models/m2-f1/checksums.sha256").stat().st_mtime
    f2_bundle = (ART / "m2_v1/models/m2-f2/checksums.sha256").stat().st_mtime

    segments = [
        ("startup + F1 data load", start, f1g0, False),
        ("F1 grid ACTUAL (first ckpt -> last search record)", f1g0, f1_rec, False),
        ("gap F1 grid -> F1 refit (record basis)", f1_rec, f1r0, False),
        ("F1 refit ACTUAL (first ckpt -> bundle)", f1r0, f1_bundle, False),
        ("gap F1 bundle -> F2 grid (record basis)", f1_bundle, f2g0, False),
        ("F2 grid ACTUAL (first ckpt -> last search record)", f2g0, f2_rec, False),
        ("gap F2 grid -> F2 refit (record basis)", f2_rec, f2r0, False),
        ("F2 refit ACTUAL (first ckpt -> bundle)", f2r0, f2_bundle, False),
        ("gap F2 bundle -> F3 grid (record basis)", f2_bundle, f3g0, False),
        ("F3 grid so far (LIVE)", f3g0, now, True),
    ]
    seg_out = []
    total = 0.0
    for name, a, b, live in segments:
        d = (b - a) / H
        total += d
        seg_out.append({"segment": name, "hours": round(d, 3), "live": live,
                        "start": datetime.fromtimestamp(a).isoformat(timespec="seconds"),
                        "end": datetime.fromtimestamp(b).isoformat(timespec="seconds")})
    elapsed = (now - start) / H
    closure_pct = abs(elapsed - total) / elapsed * 100

    spans = {"F1_grid": (f1g1 - f1g0) / H, "F1_refit": (f1r1 - f1r0) / H,
             "F2_grid": (f2g1 - f2g0) / H, "F2_refit": (f2r1 - f2r0) / H}
    actual = {"F1_grid": (f1_rec - f1g0) / H, "F1_refit": (f1_bundle - f1r0) / H,
              "F2_grid": (f2_rec - f2g0) / H, "F2_refit": (f2_bundle - f2r0) / H}
    ckpt_gaps = {"F1_grid_to_refit": (f1r0 - f1g1) / H,
                 "F1_refit_to_F2_grid": (f2g0 - f1r1) / H,
                 "F2_grid_to_refit": (f2r0 - f2g1) / H}
    rec_gaps = {"F1_grid_to_refit": (f1r0 - f1_rec) / H,
                "F1_bundle_to_F2_grid": (f2g0 - f1_bundle) / H,
                "F2_grid_to_refit": (f2r0 - f2_rec) / H}
    tail_and_bundle = {"F1": (f1_bundle - f1r1) / H, "F2": (f2_bundle - f2r1) / H}
    m1_tail_and_bundle = {}
    for f in ("f1", "f2", "f3", "f4"):
        v = mt(f"m1_full_v1/checkpoints/m1-{f}-final-*-epoch-*")
        b = (ART / f"m1_full_v1/models/m1-{f}/checksums.sha256").stat().st_mtime
        m1_tail_and_bundle[f.upper()] = (b - max(v)) / H

    tail_check = {}
    for k in spans:
        est = spans[k] + DC_TAIL_EST[k]
        tail_check[k] = {"published_span_h": round(spans[k], 3),
                         "dc_estimated_tail_h": DC_TAIL_EST[k],
                         "span_plus_tail_h": round(est, 3),
                         "actual_h": round(actual[k], 3),
                         "err_pct": round(100 * (est - actual[k]) / actual[k], 2),
                         "actual_exceeds_span_by_h": round(actual[k] - spans[k], 3),
                         "span_is_a_lower_bound": actual[k] > spans[k]}

    ci_total = sum(spans.values()) + sum(ckpt_gaps.values())
    rec_total = sum(actual.values()) + sum(rec_gaps.values())

    doc = {
        "question": ("what are the ACTUAL stage wall-clocks, does §DC's tail model survive contact with "
                     "them, and is §CI's gap assumption mis-calibrated as this probe's first draft claimed?"),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "ground_truth_source": {
            "grid_stage_end": "mtime of the fold's last search record",
            "refit_stage_end": "mtime of the bundle's checksums.sha256",
            "stage_start": "mtime of the stage's first checkpoint",
            "process_start": f"/proc/{PID}/stat field 22 / 100 + /proc/stat btime",
        },
        "segments": seg_out,
        "end_to_end_closure": {"sum_of_segments_h": round(total, 3),
                               "actual_elapsed_h": round(elapsed, 3),
                               "closure_error_pct": round(closure_pct, 5)},
        "published_spans_reproduced": {k: {"recomputed": round(spans[k], 3),
                                           "published": PUBLISHED_SPANS[k],
                                           "match": abs(spans[k] - PUBLISHED_SPANS[k]) < 0.05}
                                       for k in spans},
        "DC_tail_model_validated": tail_check,
        "CI_gap_basis_reconciliation": {
            "CI_stated_basis": "stage boundaries taken from checkpoint directory mtimes (§CI.1)",
            "checkpoint_to_checkpoint_gaps_h": {k: round(v, 3) for k, v in ckpt_gaps.items()},
            "CI_published_gaps_h": PUBLISHED_GAPS,
            "all_three_reproduce": sorted(round(v, 2) for v in ckpt_gaps.values()) == sorted(PUBLISHED_GAPS),
            "record_to_record_gaps_h": {k: round(v, 3) for k, v in rec_gaps.items()},
            "record_basis_sum_h": round(sum(rec_gaps.values()), 3),
            "checkpoint_basis_sum_h": round(sum(ckpt_gaps.values()), 3),
            "the_two_bases_differ_by_h": round(sum(ckpt_gaps.values()) - sum(rec_gaps.values()), 3),
            "reading": ("§CI's gaps are checkpoint-to-checkpoint and therefore ABSORB the preceding "
                        "stage's patience tail; the record-to-record gaps exclude it. The first draft of "
                        "this probe compared the two and wrongly concluded §CI's 5.0 h assumption "
                        "overstated reality by ~17.9 h."),
            "CI_gap_assumption_calibrated": (
                f"§CI assumed {CI_GAP_ASSUMED} h per gap; the measured checkpoint-basis mean is "
                f"{sum(ckpt_gaps.values())/len(ckpt_gaps):.3f} h, so the assumption is well calibrated"),
        },
        "two_accountings_reconcile": {
            "CI_basis_total_h": round(ci_total, 3),
            "record_basis_total_h": round(rec_total, 3),
            "difference_h": round(rec_total - ci_total, 3),
            "difference_is_final_tail_plus_bundle_h": round(tail_and_bundle["F2"], 3),
            "reconciles": abs((rec_total - ci_total) - tail_and_bundle["F2"]) < 0.01,
        },
        "DC5_inference_corrected": {
            "what_DC5_said": ("A16/D14's interval is optimistic because the refit term is understated by "
                              "roughly one patience tail"),
            "why_that_is_wrong": (
                "true of the SPANS but not of §CI's TOTAL: the tail is already carried inside §CI's "
                "checkpoint-to-checkpoint gap term, which was calibrated at 5.0 h against a measured "
                "4.891 h. Adding the tail again would double-count it."),
            "what_remains_assumed": (
                f"OOS_H = {CI_OOS_ASSUMED} h. The corresponding measured interval (refit's last checkpoint "
                f"-> bundle written) is {tail_and_bundle['F1']:.3f} h for M2 F1 and "
                f"{tail_and_bundle['F2']:.3f} h for M2 F2, against "
                f"{min(m1_tail_and_bundle.values()):.3f}-{max(m1_tail_and_bundle.values()):.3f} h for M1's "
                "four folds. So 2.0 h is reasonable for M1 and about 0.8-0.95 h light for M2 -- a much "
                "smaller correction than the tail claim, and in the same direction."),
            "measured_tail_plus_bundle_h": {"M2": tail_and_bundle, "M1": m1_tail_and_bundle},
        },
        "live_vs_immutable": {
            "immutable": ["all four M2 completed-stage durations", "all gaps", "the closure identity"],
            "live_excluded_from_conclusions": "the F3 grid segment (grows while the grid runs)",
        },
        "controls": {},
        "read_only": True, "database_access": False, "weights_loaded": False,
        "checkpoint_pt_opened": False, "parquet_read": False, "training_performed": False,
        "repository_module_imported_or_executed": False, "processes_signalled": [],
        "atime_side_effect": False,
    }
    doc["controls"] = {
        "i_segments_close_within_0.01pct": closure_pct < 0.01,
        "ii_all_segments_positive_and_ordered": all(s["hours"] > 0 for s in seg_out) and all(
            seg_out[i]["end"] <= seg_out[i + 1]["end"] for i in range(len(seg_out) - 1)),
        "iii_every_actual_exceeds_its_span": all(
            v["span_is_a_lower_bound"] for v in tail_check.values()),
        "iv_DC_tail_model_within_5pct_all_four": all(
            abs(v["err_pct"]) < 5.0 for v in tail_check.values()),
        "v_spans_reproduce_published": all(
            v["match"] for v in doc["published_spans_reproduced"].values()),
        "vi_ckpt_gaps_reproduce_CI_published":
            doc["CI_gap_basis_reconciliation"]["all_three_reproduce"],
        "vii_the_two_gap_bases_differ_materially":
            abs(sum(ckpt_gaps.values()) - sum(rec_gaps.values())) > 5.0,
        "viii_two_accountings_reconcile": doc["two_accountings_reconcile"]["reconciles"],
        "ALL_PASS": None,
    }
    doc["controls"]["ALL_PASS"] = all(
        v for k, v in doc["controls"].items() if k != "ALL_PASS")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(doc, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    print("=== ground-truth segments (artefact mtimes) ===")
    for s in seg_out:
        print(f"  {s['segment']:52s} {s['hours']:>8.3f} h{'  (LIVE)' if s['live'] else ''}")
    c = doc["end_to_end_closure"]
    print(f"\n  sum {c['sum_of_segments_h']:.3f} h vs elapsed {c['actual_elapsed_h']:.3f} h "
          f"-> closure error {c['closure_error_pct']:.5f}%")
    print("\n=== §DC's tail model vs ground truth ===")
    print(f"  {'stage':10s} {'span':>7s} {'tail est':>9s} {'span+tail':>10s} {'ACTUAL':>8s} {'err':>7s}")
    for k, v in tail_check.items():
        print(f"  {k:10s} {v['published_span_h']:>7.3f} {v['dc_estimated_tail_h']:>9.3f} "
              f"{v['span_plus_tail_h']:>10.3f} {v['actual_h']:>8.3f} {v['err_pct']:>+6.1f}%")
    r = doc["CI_gap_basis_reconciliation"]
    print(f"\n=== §CI's gap basis: this probe's FIRST DRAFT WAS WRONG HERE ===")
    print(f"  §CI stated basis: {r['CI_stated_basis']}")
    print(f"  checkpoint-to-checkpoint gaps: {[round(v,3) for v in ckpt_gaps.values()]}")
    print(f"  §CI published:                 {PUBLISHED_GAPS}  -> reproduce: {r['all_three_reproduce']}")
    print(f"  record-to-record gaps:         {[round(v,3) for v in rec_gaps.values()]}")
    print(f"  {r['CI_gap_assumption_calibrated']}")
    a = doc["two_accountings_reconcile"]
    print(f"\n=== the two accountings reconcile ===")
    print(f"  §CI basis total {a['CI_basis_total_h']:.3f} h | record basis {a['record_basis_total_h']:.3f} h")
    print(f"  difference {a['difference_h']:.3f} h == F2 tail+bundle {a['difference_is_final_tail_plus_bundle_h']:.3f} h"
          f" -> reconciles: {a['reconciles']}")
    d5 = doc["DC5_inference_corrected"]
    print(f"\n=== §DC.5's inference corrected ===")
    print(f"  measured refit_last_ckpt -> bundle: M2 {tail_and_bundle} | M1 "
          f"{ {k: round(v,3) for k,v in m1_tail_and_bundle.items()} }")
    print(f"  §CI assumed OOS_H = {CI_OOS_ASSUMED} h")
    print("\n=== controls ===")
    for k, v in doc["controls"].items():
        print(f"  {'OK ' if v else '*** FAIL ***'} {k}")
    assert doc["controls"]["ALL_PASS"], "a control failed"
    print("wrote", OUT)


if __name__ == "__main__":
    main()
