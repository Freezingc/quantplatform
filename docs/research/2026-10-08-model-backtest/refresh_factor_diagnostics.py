#!/usr/bin/env python3
"""Recompute the factor section of optimization_levers_results.json with per-date IC.

The first run of `optimization_levers.py` computed each factor's IC as ONE correlation
pooled over every (date, stock) row in a fold -- `ic_days` came back as 1 per fold instead
of ~242. A pooled correlation mixes cross-sectional signal with cross-date drift in both
the factor and the market, so those numbers were invalid and were discarded rather than
reported. The function has since been rewritten to rank within each signal date, correlate
within each signal date, and average across dates.

This script reruns only that section and writes it back, so the JSON on disk matches what
a fresh full run of the corrected `optimization_levers.py` would produce. Every other key
is carried over byte-for-byte.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from optimization_levers import ARTIFACTS, FACTOR_MAP, factor_diagnostics

HERE = Path(__file__).resolve().parent
TARGET = HERE / "optimization_levers_results.json"

data = json.loads(TARGET.read_text())
old = data.get("factor_diagnostics", {}).get("all_folds", {})
print(f"discarding {len(old)} invalid pooled-IC rows")
data["factor_diagnostics"] = factor_diagnostics(ARTIFACTS)
TARGET.write_text(json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False, default=str) + "\n")

rows = data["factor_diagnostics"]["all_folds"]
print(f"{'pos':<5} {'factor':<26} {'dens':<7} {'IC':>8} {'ICsd':>7} {'IR':>7} {'t':>7} {'IC>0':>6} "
      f"{'top50pp':>8} {'NaN':>6}")
for f in sorted(rows, key=lambda k: -abs(rows[k]["ic_mean"]) if math.isfinite(rows[k]["ic_mean"]) else 0):
    v = rows[f]
    print(f"{f:<5} {FACTOR_MAP[f]:<26} {v['density']:<7} {v['ic_mean']:>+8.4f} {v['ic_std']:>7.4f} "
          f"{v['ic_ir_daily']:>+7.3f} {v['ic_t_stat']:>+7.2f} {v['ic_positive_share']*100:>5.1f}% "
          f"{v['top50_spread_pp']:>+8.2f} {v['nan_rate']*100:>5.1f}%")
print(f"\nic_days per factor (should be ~960): "
      f"{ {f: rows[f]['ic_days'] for f in sorted(rows)[:3]} }")
print(f"wrote {TARGET}")
