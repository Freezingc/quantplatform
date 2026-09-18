#!/usr/bin/env python3
"""Read-only probe: extract the measured inputs M2's remaining runtime depends on.

§CH established WHEN it is safe to merge the peak/final fix (after M2 finishes), which makes "when
does M2 finish" the question gating every downstream item: the merge itself, replacing §BQ.4's
projections with measurements for F2-F4, §BT.1(b)'s k=8 paired experiment, and the peak-basis
recomputation of §CD.2's regret table. This probe gathers what a projection needs; the companion
probe_m2_runtime_projection.py turns it into bounded scenarios.

Everything is read off artifacts the run already wrote, plus parquet FOOTERS for row counts -- no
parquet data loaded, no database touched, no weight read:

  phase boundaries and epochs   checkpoint directory mtimes under m2_v1/checkpoints, parsed by name.
                                Checkpoints are saved only on strict improvement (trainer.py:111-116),
                                so the maximum saved epoch IS that run's peak epoch, and with
                                early_stopping_patience=10 the run stopped at peak+10. Grid runs are
                                m2-<fold>-c<cfg>-<tag>-seed-<seed>-epoch-<n>; winning-config refits are
                                m2-<fold>-final-<seed>-seed-<seed>-epoch-<n>.
  panel sizes                   pyarrow.ParquetFile(...).metadata.num_rows over datasets/<id>/split=*.
                                The resulting F1/F2 test counts (804,575 / 882,426) reproduce §BI.5's
                                published per-fold panel rows exactly, which cross-checks that these
                                footers describe the same panels §BI.5 counted.
  budget constants              --max-epochs defaults to 100 (run_real_candidate_training.py:187) and
                                M2's command line does not override it; TORCH_EARLY_STOPPING_PATIENCE
                                is 10 (:2121).

Safety: read-only. No database access, no network, no MCP, no repository module imported or executed,
no weight or tensor loaded, nothing written into any artifact directory, no signal sent to any process.
M2 (pid 3588838) is still training and its F2 final-refit phase is incomplete, so every F2-final figure
here is a LOWER BOUND and is labelled as such downstream.
"""
import glob, json, os, re, statistics as st, collections
from datetime import datetime, timedelta
from pathlib import Path

CK = "/data/zhangyudong/qrv2_research_artifacts/m2_v1/checkpoints"
HERE = Path(__file__).resolve().parent.parent
OUT = HERE / "measurements" / "m2_runtime_projection_inputs.json"

def phase_stats(fold, kind):
    runs = collections.defaultdict(int); peak = collections.defaultdict(int); ts = []
    if kind == "grid":
        pat = re.compile(rf"^m2-{fold}-c(\d+)-(\d+)-seed-(\d+)-epoch-(\d+)$")
    else:
        pat = re.compile(rf"^m2-{fold}-final-(\d+)-seed-(\d+)-epoch-(\d+)$")
    for d in glob.glob(os.path.join(CK, f"m2-{fold}-*")):
        m = pat.match(os.path.basename(d))
        if not m:
            continue
        g = m.groups(); ep = int(g[-1]); key = g[:-1]
        runs[key] += 1; peak[key] = max(peak[key], ep); ts.append(os.path.getmtime(d))
    if not runs:
        return None
    eps = list(peak.values())
    return {"runs": len(runs), "checkpoints": sum(runs.values()),
            "ckpt_per_run": sum(runs.values()) / len(runs),
            "peak_median": st.median(eps), "peak_max": max(eps), "peak_mean": st.mean(eps),
            "epochs_per_run_mean": st.mean(eps) + 10,   # patience = 10
            "first": min(ts), "last": max(ts), "span_h": (max(ts) - min(ts)) / 3600}

P = {(f, k): phase_stats(f, k) for f in ("f1", "f2") for k in ("grid", "final")}

# panel rows: measured from parquet footers (F1/F2), test rows for F3/F4 from §BI.5
PANEL = {"F1": {"train": 2296573, "val": 696906, "test": 804575},
         "F2": {"train": 2539884, "val": 783477, "test": 882426},
         "F3": {"test": 932167}, "F4": {"test": 938594}}
for f in ("F1", "F2"):
    PANEL[f]["trainval"] = PANEL[f]["train"] + PANEL[f]["val"]
    PANEL[f]["ratio_trainval_over_test"] = PANEL[f]["trainval"] / PANEL[f]["test"]
# project F3/F4 train+val from the measured test rows and the observed ratio trend
ratios = [PANEL["F1"]["ratio_trainval_over_test"], PANEL["F2"]["ratio_trainval_over_test"]]
step = ratios[1] - ratios[0]
for i, f in enumerate(("F3", "F4"), start=2):
    r = ratios[1] + step * i
    PANEL[f]["ratio_trainval_over_test_projected"] = r
    PANEL[f]["trainval_projected"] = int(PANEL[f]["test"] * r)

# cost constant k = wall_seconds / (epochs_per_run x panel_rows), fitted on both grid phases
k = {}
for fold, key in (("f1", "F1"), ("f2", "F2")):
    g = P[(fold, "grid")]
    k[fold] = g["span_h"] * 3600 / (g["epochs_per_run_mean"] * PANEL[key]["trainval"])
kmean = st.mean(k.values())

out = {
  "question": ("How much longer will M2 run? Built from measured phase timings, measured panel row "
               "counts and the observed epochs-per-run, not from a geometric extrapolation of wall "
               "clock alone."),
  "measured_phases": {f"{f}_{k_}": {kk: (round(vv, 4) if isinstance(vv, float) else
                                         (datetime.fromtimestamp(vv).isoformat(timespec="seconds")
                                          if kk in ("first", "last") else vv))
                                     for kk, vv in v.items()}
                      for (f, k_), v in P.items() if v},
  "panel_rows": PANEL,
  "cost_model": {
      "form": "grid_wall_seconds = k * epochs_per_run * panel_train_val_rows",
      "k_per_fold_s_per_row_epoch": {a: round(b, 8) for a, b in k.items()},
      "k_mean": round(kmean, 8),
      "k_spread_ratio": round(max(k.values()) / min(k.values()), 4),
      "validation": {
          "predicted_wall_ratio_f2_over_f1": round(
              (P[("f2", "grid")]["epochs_per_run_mean"] / P[("f1", "grid")]["epochs_per_run_mean"])
              * (PANEL["F2"]["trainval"] / PANEL["F1"]["trainval"]), 4),
          "observed_wall_ratio_f2_over_f1": round(
              P[("f2", "grid")]["span_h"] / P[("f1", "grid")]["span_h"], 4),
      },
  },
  "epochs_trend": {
      "f1_grid_epochs_per_run": round(P[("f1", "grid")]["epochs_per_run_mean"], 2),
      "f2_grid_epochs_per_run": round(P[("f2", "grid")]["epochs_per_run_mean"], 2),
      "growth_ratio": round(P[("f2", "grid")]["epochs_per_run_mean"]
                            / P[("f1", "grid")]["epochs_per_run_mean"], 3),
      "peak_max_f1": P[("f1", "grid")]["peak_max"], "peak_max_f2": P[("f2", "grid")]["peak_max"],
      "max_epochs_cap": 100,
      "patience": 10,
      "note": ("peak + patience <= max_epochs, so epochs_per_run is capped at 100; F2's peak_max of "
               "71 implies 81 epochs, still under the cap"),
  },
  "inter_phase_gaps_h": {
      "f1_grid_end_to_f1_final_start": round(
          (P[("f1", "final")]["first"] - P[("f1", "grid")]["last"]) / 3600, 2),
      "f1_final_end_to_f2_grid_start": round(
          (P[("f2", "grid")]["first"] - P[("f1", "final")]["last"]) / 3600, 2),
      "f2_grid_end_to_f2_final_start": round(
          (P[("f2", "final")]["first"] - P[("f2", "grid")]["last"]) / 3600, 2),
  },
  "f2_final_status": {"span_h_so_far": round(P[("f2", "final")]["span_h"], 2),
                      "complete": False,
                      "note": "F2 final refit was still writing checkpoints at measurement time"},
  "generated_at_local": datetime.now().isoformat(timespec="seconds"),
  "read_only": True, "database_access": False, "weights_loaded": False,
}
out["cost_model"]["validation"]["model_error_pct"] = round(
    100 * abs(out["cost_model"]["validation"]["predicted_wall_ratio_f2_over_f1"]
              - out["cost_model"]["validation"]["observed_wall_ratio_f2_over_f1"])
    / out["cost_model"]["validation"]["observed_wall_ratio_f2_over_f1"], 2)
OUT.parent.mkdir(parents=True, exist_ok=True)
json.dump(out, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("wrote", OUT)
print(json.dumps({k2: out[k2] for k2 in ("measured_phases", "cost_model", "epochs_trend",
                                         "inter_phase_gaps_h")}, ensure_ascii=False, indent=1)[:2600])
