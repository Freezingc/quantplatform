#!/usr/bin/env python3
"""Read-only: the YesLab store holds several mutually different GP_006 values per key. Which one is ours?

`probe_replay_fidelity_census.py` reported yeslab_375 at Spearman 0.9271 against GP_006. That number is
not interpretable, and the reason is measured here rather than assumed: on 2024-02-28 the store holds
17,909 GP_006 rows for 6,005 distinct codes -- 2.98 rows per code -- and 4,736 of those codes carry TWO
DIFFERENT values, with no version, batch or timestamp column anywhere in the table schema
(`ts_code SYMBOL, trade_date DATE, factor_name STRING, value DOUBLE`). The census therefore joined our
one frozen value against a mixture of two store values and correlated the result.

The controls show this is specific to GP_006: AttentionSpillover, GP_095, 60日成交量去均值相对偏离 and
羊群效应_CSAD波动率21 hold exactly one row per code, and GP_015 / 行业内收益标准化绝对值 hold two rows per
code but with IDENTICAL values, so duplicating them cannot move a rank correlation.

This probe asks the question that is actually answerable: for each code, does our frozen value equal one
of the store's values for that key? It reports exact-match rates and, where a code is ambiguous, which
variant we match -- so "our replay reproduces one of the store's publications" can be distinguished from
"our replay reproduces neither".

SAFETY: SELECT-only on dfs://YesLab and dfs://QuantResearchV2Test. No calculator executed, no writes,
no artifact mutation, no weights.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

ART = Path("/data/zhangyudong/qrv2_research_artifacts")
REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
TARGET_DB = "dfs://QuantResearchV2Test"
CASES = (
    ("factor_data8", "GP_006", "yeslab_375", "TARGET ambiguous store, census Spearman 0.9271"),
    ("factor_data8", "GP_015", "yeslab_384", "CONTROL two identical rows per code"),
    ("factor_data8", "GP_003", "yeslab_372", "CONTROL one row per code"),
    ("factor_data10", "AttentionSpillover", "yeslab_471", "CONTROL one row per code, census 0.8698"),
)
DAYS = ("2024-02-28", "2024-11-05", "2025-06-16")
TOL = 1e-9


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=ART / "replay_vs_store")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import new_holder, query
    holder = new_holder()

    result = {
        "question": (
            "The store holds several different values for GP_006 on one key. Does our frozen value equal "
            "one of them, and if so which -- or does it equal neither?"
        ),
        "generated_at_local": datetime.now().isoformat(timespec="seconds"),
        "safety": "SELECT-only; no calculator executed; nothing written",
        "tolerance_for_exact_match": TOL,
        "days": list(DAYS),
        "cases": {},
    }
    for table, name, fid, role in CASES:
        per_day = {}
        for ds in DAYS:
            dl = ds.replace("-", ".")
            ok_s, st = query(holder,
                f"select ts_code, value from loadTable('dfs://YesLab', '{table}') "
                f"where factor_name = '{name}' and trade_date = date('{dl}')")
            ok_f, fz = query(holder,
                f"select ts_code, value from loadTable('{TARGET_DB}', 'factor_daily') "
                f"where factor_id = '{fid}' and signal_date = date('{dl}')")
            if not ok_s or not ok_f or not isinstance(st, pd.DataFrame) or not isinstance(fz, pd.DataFrame):
                per_day[ds] = {"error": "query failed"}
                continue
            st["ts_code"] = st["ts_code"].astype(str)
            fz["ts_code"] = fz["ts_code"].astype(str)
            st["value"] = pd.to_numeric(st["value"], errors="coerce")
            fz["value"] = pd.to_numeric(fz["value"], errors="coerce")
            st = st.dropna(subset=["value"])
            fz = fz.dropna(subset=["value"])
            variants = st.groupby("ts_code")["value"].apply(lambda s: sorted(set(s.tolist())))
            n_var = variants.map(len)
            fz_v = fz.set_index("ts_code")["value"]
            common = [c for c in fz_v.index if c in variants.index]
            match_any = sum(
                1 for c in common
                if any(abs(fz_v[c] - v) <= TOL * max(1.0, abs(v)) for v in variants[c])
            )
            match_first = sum(
                1 for c in common
                if abs(fz_v[c] - variants[c][0]) <= TOL * max(1.0, abs(variants[c][0]))
            )
            match_last = sum(
                1 for c in common
                if abs(fz_v[c] - variants[c][-1]) <= TOL * max(1.0, abs(variants[c][-1]))
            )
            amb = [c for c in common if len(variants[c]) > 1]
            entry = {
                "store_rows": int(len(st)),
                "store_distinct_codes": int(st["ts_code"].nunique()),
                "store_rows_per_code": float(len(st) / max(int(st["ts_code"].nunique()), 1)),
                "codes_with_more_than_one_distinct_value": int((n_var > 1).sum()),
                "max_distinct_values_on_one_code": int(n_var.max()) if len(n_var) else 0,
                "frozen_rows": int(len(fz)),
                "codes_in_both": len(common),
                "frozen_equals_some_store_value": match_any,
                "frozen_equals_some_store_value_share": match_any / len(common) if common else None,
                "frozen_equals_the_lowest_variant": match_first,
                "frozen_equals_the_highest_variant": match_last,
                "ambiguous_codes": len(amb),
                "frozen_matches_on_ambiguous_codes": sum(
                    1 for c in amb
                    if any(abs(fz_v[c] - v) <= TOL * max(1.0, abs(v)) for v in variants[c])
                ),
            }
            if amb:
                low = pd.Series({c: variants[c][0] for c in amb})
                high = pd.Series({c: variants[c][-1] for c in amb})
                ours = pd.Series({c: fz_v[c] for c in amb})
                entry["on_ambiguous_codes_spearman"] = {
                    "vs_lowest_variant": float(stats.spearmanr(ours, low.reindex(ours.index)).statistic),
                    "vs_highest_variant": float(stats.spearmanr(ours, high.reindex(ours.index)).statistic),
                    "lowest_vs_highest": float(stats.spearmanr(low, high.reindex(low.index)).statistic),
                }
                entry["on_ambiguous_codes_median_relative_spread_of_the_two_variants"] = float(
                    ((high - low).abs() / high.abs().clip(lower=1e-12)).median()
                )
            per_day[ds] = entry
        result["cases"][fid] = {"source_name": name, "table": table, "role": role, "per_day": per_day}
        print(f"  {fid:12} {name[:20]:22} " + " | ".join(
            f"{d}: rows/code={e.get('store_rows_per_code', 0):.2f} amb={e.get('ambiguous_codes', 0)} "
            f"match_any={e.get('frozen_equals_some_store_value_share')}"
            for d, e in per_day.items() if "error" not in e), flush=True)

    out = args.out / "gp006_ambiguous_store.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str) + "\n",
                   encoding="utf-8")
    print(f"wrote {out}")
    for fid, c in result["cases"].items():
        print(f"\n{fid} ({c['source_name']}) -- {c['role']}")
        for d, e in c["per_day"].items():
            if "error" in e:
                print(f"   {d} {e}"); continue
            print(f"   {d} store {e['store_rows']} rows / {e['store_distinct_codes']} codes "
                  f"({e['store_rows_per_code']:.3f} per code), {e['codes_with_more_than_one_distinct_value']} "
                  f"codes with >1 distinct value (max {e['max_distinct_values_on_one_code']}); "
                  f"frozen {e['frozen_rows']}, in both {e['codes_in_both']}, "
                  f"frozen equals SOME store value on {e['frozen_equals_some_store_value_share']:.4f}")
            if "on_ambiguous_codes_spearman" in e:
                s = e["on_ambiguous_codes_spearman"]
                print(f"        ambiguous={e['ambiguous_codes']} matched={e['frozen_matches_on_ambiguous_codes']} "
                      f"spearman vs lowest={s['vs_lowest_variant']:+.4f} vs highest={s['vs_highest_variant']:+.4f} "
                      f"lowest_vs_highest={s['lowest_vs_highest']:+.4f} "
                      f"median relative spread={e['on_ambiguous_codes_median_relative_spread_of_the_two_variants']:.4f}")


if __name__ == "__main__":
    main()
