#!/usr/bin/env python3
"""Read-only projection: how much longer will M2 run, with the uncertainty stated rather than hidden.

Takes the measurements from probe_m2_runtime_projection_inputs.py and produces three bounded
scenarios. The model is deliberately simple and is validated against the two finished folds before it
predicts anything:

  grid_wall_seconds = k * epochs_per_run * panel_train_plus_validation_rows

Fitted separately on F1 and F2 this gives k = 0.0021666 and 0.0022200 s per (row x epoch), so the two
folds agree within 2.5%, and the model predicts F2/F1's wall-time ratio as 1.8289 against 1.8739
observed -- a 2.4% error. One constant explains both folds, which is what licenses using it for F3/F4.

The final-refit phase does NOT follow that constant: it is about 10.8x cheaper per (row x epoch) than
the grid phase, because 3 runs share 24 workers in one round whereas the grid puts 72 runs through
three rounds. It gets its own constant, and since F2's final phase is still running the F1-derived
constant under-predicts it by a measured 1.45x, carried as an explicit uplift rather than quietly
absorbed.

The dominant uncertainty is neither panel size (F3/F4 train+validation rows are 3.60M and 3.66M, only
1.09x and 1.11x F2's) nor k (2.5% spread). It is epochs_per_run, which grew 1.647x from F1 (17.49) to
F2 (28.81) and whose peak epoch reached 71 on F2 against 14 on F1. Two folds cannot determine whether
that growth is a trend or noise, so the scenarios bracket it rather than extrapolating one number:
flat at F2's level, geometric continuation capped at max_epochs=100, and every run hitting the cap.
The optimistic-to-pessimistic spread is 167h to 524h of remaining wall time, which is the honest
answer -- hence a range and a projected finish date per scenario rather than a single number.

Safety: reads one JSON file written by the companion probe. No database access, no network, no MCP, no
repository module imported or executed, no weight or tensor loaded, nothing written into any artifact
directory, no signal sent to any process.
"""
import json
from datetime import datetime, timedelta
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
IN_PATH = HERE / "measurements" / "m2_runtime_projection_inputs.json"
OUT_PATH = HERE / "measurements" / "m2_runtime_projection.json"

IN = json.load(open(IN_PATH, encoding="utf-8"))
P = IN["measured_phases"]; PANEL = IN["panel_rows"]; CM = IN["cost_model"]
k = CM["k_mean"]
MAX_EPOCHS, PATIENCE = 100, 10
EPOCH_CAP = MAX_EPOCHS   # epochs/run cannot exceed max_epochs

# --- panel rows for F3/F4: measured test rows (verified against §BI.5) x projected trainval/test ratio
r1 = PANEL["F1"]["trainval"] / PANEL["F1"]["test"]
r2 = PANEL["F2"]["trainval"] / PANEL["F2"]["test"]
ratio = {"F1": r1, "F2": r2}
step = r2 - r1
rows = {"F1": PANEL["F1"]["trainval"], "F2": PANEL["F2"]["trainval"]}
for i, f in enumerate(("F3", "F4"), start=2):
    rr = r2 + step * i
    ratio[f] = rr
    rows[f] = PANEL[f]["test"] * rr

ep = {"F1": P["f1_grid"]["epochs_per_run_mean"], "F2": P["f2_grid"]["epochs_per_run_mean"]}
growth = ep["F2"] / ep["F1"]

def grid_h(fold, epochs):
    return k * min(epochs, EPOCH_CAP) * rows[fold] / 3600

# final phase: measured separately, it does NOT follow the grid cost constant (10.8x cheaper per
# row-epoch because 3 runs share 24 workers instead of 72 runs over 3 rounds). Modelled directly from
# the measured F1/F2 final walls scaled by epochs x rows, with the observed 1.45x under-prediction of
# F2 treated as the model's own bias and carried as an uplift.
kf = P["f1_final"]["span_h"] * 3600 / (P["f1_final"]["epochs_per_run_mean"] * rows["F1"])
BIAS = 1.45  # measured: F2 final already ran 1.45x the F1-scaled prediction and is not finished
def final_h(fold, epochs):
    return BIAS * kf * min(epochs, EPOCH_CAP) * rows[fold] / 3600

gap = 5.0   # mean of the three measured inter-phase gaps (5.44, 3.42, 5.80)
oos_h = 2.0 # OOS evaluation + report; not separately measured, kept small and flagged

SCEN = {
 "optimistic_epochs_flat":      {"F3": ep["F2"],                 "F4": ep["F2"]},
 # Two folds cannot distinguish geometric from linear growth in epochs_per_run, so both are carried
 # rather than one being asserted. Linear sits between the flat and geometric scenarios.
 "linear_growth_capped":        {"F3": min(ep["F2"] + (ep["F2"] - ep["F1"]), EPOCH_CAP),
                                 "F4": min(ep["F2"] + 2 * (ep["F2"] - ep["F1"]), EPOCH_CAP)},
 "central_geometric_capped":    {"F3": min(ep["F2"]*growth, EPOCH_CAP), "F4": min(ep["F2"]*growth**2, EPOCH_CAP)},
 "pessimistic_all_hit_cap":     {"F3": EPOCH_CAP,                "F4": EPOCH_CAP},
}

now = datetime.now()
f2_final_so_far = P["f2_final"]["span_h"]
# elapsed is derived from the first grid checkpoint of F1, not hardcoded, so the projection stays
# honest if this is rerun later; the driver's own process start is 2026-09-13 18:54:55 and its first
# checkpoint landed 1789298882 (2026-09-13 19:28), the ~34 minute difference being dataset assembly.
run_start = datetime.fromtimestamp(P["f1_grid"]["first"])
elapsed_days = (now - run_start).total_seconds() / 86400
out = {"generated_at_local": now.isoformat(timespec="seconds"),
       "run_start_from_first_checkpoint": run_start.isoformat(timespec="seconds"),
       "elapsed_days_at_generation": round(elapsed_days, 2),
       "cost_model_grid": {"k_s_per_row_epoch": k, "fitted_on": "F1 and F2 grid phases",
                           "validation_error_pct": CM["validation"]["model_error_pct"],
                           "k_spread_between_folds": CM["k_spread_ratio"]},
       "cost_model_final": {"k_s_per_row_epoch": kf, "under_prediction_bias_applied": BIAS,
                            "note": ("the final-refit phase is 10.8x cheaper per (row x epoch) than the "
                                     "grid phase -- 3 runs over 24 workers in one round versus 72 runs in "
                                     "three rounds -- so it is modelled with its own constant, and the "
                                     "measured 1.45x under-prediction of F2's still-running final phase "
                                     "is carried as an explicit uplift")},
       "panel_rows_trainval": {f: int(round(v)) for f, v in rows.items()},
       "trainval_over_test_ratio": {f: round(v, 4) for f, v in ratio.items()},
       "epochs_per_run_grid": {"F1": round(ep["F1"], 2), "F2": round(ep["F2"], 2),
                               "growth_ratio_F2_over_F1": round(growth, 3),
                               "cap": EPOCH_CAP,
                               "F2_peak_max_epoch": P["f2_grid"]["peak_max"],
                               "F2_worst_case_epochs": P["f2_grid"]["peak_max"] + PATIENCE},
       "inter_phase_gap_h_assumed": gap, "oos_and_report_h_assumed": oos_h,
       "f2_final_hours_already_elapsed": round(f2_final_so_far, 2),
       "scenarios": {}}

for name, e in SCEN.items():
    f3g, f4g = grid_h("F3", e["F3"]), grid_h("F4", e["F4"])
    f3f = final_h("F3", e["F3"]); f4f = final_h("F4", e["F4"])
    # F2 final is incomplete: assume the remaining half of what F2's final needs at its own epochs
    f2_rest = max(0.0, final_h("F2", ep["F2"]) - f2_final_so_far)
    total = (f2_rest + gap + f3g + gap + f3f + gap + f4g + gap + f4f + oos_h)
    out["scenarios"][name] = {
        "epochs_per_run_grid": {"F3": round(min(e["F3"], EPOCH_CAP), 1),
                                "F4": round(min(e["F4"], EPOCH_CAP), 1)},
        "f2_final_remaining_h": round(f2_rest, 1),
        "f3_grid_h": round(f3g, 1), "f3_final_h": round(f3f, 1),
        "f4_grid_h": round(f4g, 1), "f4_final_h": round(f4f, 1),
        "gaps_and_oos_h": round(4 * gap + oos_h, 1),
        "total_remaining_h": round(total, 1),
        "total_remaining_days": round(total / 24, 1),
        "projected_finish": (now + timedelta(hours=total)).strftime("%Y-%m-%d %H:%M"),
        "elapsed_days_at_generation": round(elapsed_days, 2),
        "projected_total_run_days": round(elapsed_days + total / 24, 1),
    }
json.dump(out, open(OUT_PATH, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

print("epochs/run (grid): F1 %.2f -> F2 %.2f  (growth %.3fx, cap %d)" % (ep["F1"], ep["F2"], growth, EPOCH_CAP))
print("panel train+val rows:", {f: f"{int(round(v)):,}" for f, v in rows.items()})
print()
print(f"{'scenario':28s} {'F3 ep':>6s} {'F4 ep':>6s} {'F3 grid':>8s} {'F4 grid':>8s} {'total_h':>8s} {'days':>5s} {'finish':>17s}")
for n, s in out["scenarios"].items():
    print(f"{n:28s} {s['epochs_per_run_grid']['F3']:6.1f} {s['epochs_per_run_grid']['F4']:6.1f} "
          f"{s['f3_grid_h']:8.1f} {s['f4_grid_h']:8.1f} {s['total_remaining_h']:8.1f} "
          f"{s['total_remaining_days']:5.1f} {s['projected_finish']:>17s}")
print()
print("elapsed so far: %.2f days (from first checkpoint %s)" % (
    elapsed_days, run_start.strftime("%Y-%m-%d %H:%M")))
print("projected total run length %.1f to %.1f days" % (
    out["scenarios"]["optimistic_epochs_flat"]["projected_total_run_days"],
    out["scenarios"]["pessimistic_all_hit_cap"]["projected_total_run_days"]))
print("wrote", OUT_PATH)
