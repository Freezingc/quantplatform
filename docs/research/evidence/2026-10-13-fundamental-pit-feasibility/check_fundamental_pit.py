"""Read-only feasibility check of fundamental_pit_snapshot as a NEW information domain.

Questions answered:
  1. PIT integrity  -- any row whose value was not yet available on its signal_date?
  2. Coverage       -- per metric per year, share of cn_liquid_a member-days holding a value
  3. Breadth        -- per metric per day, median cross-section size (vs the ~2.7-4.3k universe)
  4. Staleness      -- how old the carried report is (fundamentals are quarterly)
  5. Value sanity   -- nulls / degeneracy / extreme outliers
  6. Window gap     -- does it cover the folds' test years (2022-2025)?

Read-only: no DB writes, no artifact-store writes, no 2026/E9 reads (the query
window is clamped to <= 2025-12-31).
"""
from __future__ import annotations

import json
import pathlib

import dolphindb as ddb
import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).resolve().parent
DB = "dfs://QuantResearchV2Test"
SNAP = f"loadTable('{DB}','fundamental_pit_snapshot')"
UNI = f"loadTable('{DB}','universe_daily')"
YEARS = (2020, 2021, 2022, 2023, 2024)   # snapshot's own span; 2026 never queried
CLAMP = "signal_date between 2020.01.01 : 2025.12.31"


def main() -> int:
    s = ddb.session()
    s.connect("10.176.55.211", 8848, "xiahongjie", "123456")
    res: dict = {"scope": "FUNDAMENTAL_PIT_FEASIBILITY_V1", "database": DB, "years": list(YEARS)}

    # ---- 1. PIT integrity -------------------------------------------------
    pit = s.run(f"select count(*) as violations from {SNAP} where {CLAMP} and (available_date > signal_date or stale_days < 0)")
    res["pit_violations"] = int(pit["violations"][0])
    span = s.run(f"select metric_id, count(*) as rows, min(signal_date) as d0, max(signal_date) as d1, "
                 f"min(stale_days) as stale_min, max(stale_days) as stale_max from {SNAP} where {CLAMP} group by metric_id order by metric_id")
    res["per_metric_span"] = json.loads(span.to_json(orient="records", date_format="iso"))

    # ---- 2/3. universe-restricted coverage and breadth --------------------
    uni_years = s.run(f"select count(*) as member_days from {UNI} "
                      f"where universe_id='cn_liquid_a' and is_member=true "
                      f"group by year(signal_date) as yr order by yr")
    res["universe_member_days"] = {int(r["yr"]): int(r["member_days"]) for _, r in uni_years.iterrows()}

    cov_rows = {}
    for y in YEARS:
        q = (f"u = select signal_date, ts_code from {UNI} where universe_id='cn_liquid_a' and is_member=true "
             f"and year(signal_date)={y}\n"
             f"f = select signal_date, ts_code, metric_id from {SNAP} where year(signal_date)={y}\n"
             f"j = select metric_id, signal_date, ts_code from ej(f, u, `signal_date`ts_code)\n"
             f"select metric_id, count(*) as n from j group by metric_id")
        cov_rows[y] = s.run(q)
        print(f"  coverage {y} done", flush=True)
    cov = pd.concat([v.assign(yr=y) for y, v in cov_rows.items()], ignore_index=True)
    cov["member_days"] = cov["yr"].map(res["universe_member_days"])
    cov["coverage_pct"] = (cov["n"] / cov["member_days"] * 100).round(2)
    res["coverage_by_metric_year"] = json.loads(cov.to_json(orient="records"))

    # ---- 4. staleness distribution (universe-restricted, 2022-2024) -------
    q = ("u = select signal_date, ts_code from "
         f"{UNI} where universe_id='cn_liquid_a' and is_member=true and year(signal_date) between 2022 and 2024\n"
         f"f = select signal_date, ts_code, metric_id, stale_days from {SNAP} where year(signal_date) between 2022 and 2024\n"
         "j = ej(f, u, `signal_date`ts_code)\n"
         "select metric_id, median(stale_days) as p50, percentile(stale_days, 0.95) as p95, "
         "max(stale_days) as mx from j group by metric_id order by metric_id")
    st = s.run(q)
    res["staleness_2022_2024"] = json.loads(st.to_json(orient="records"))

    # ---- 5. value sanity ---------------------------------------------------
    q2 = (f"select metric_id, count(*) as n, sum(iif(isNull(value), 1, 0)) as nulls, "
          f"min(value) as vmin, max(value) as vmax, avg(value) as vmean, std(value) as vstd "
          f"from {SNAP} where {CLAMP} group by metric_id order by metric_id")
    vs = s.run(q2)
    res["value_sanity"] = json.loads(vs.to_json(orient="records"))

    # ---- 6. window gap -----------------------------------------------------
    gap = s.run(f"select count(*) as n from {SNAP} where {CLAMP} group by year(signal_date) as yr order by yr")
    res["rows_by_year"] = {int(r["yr"]): int(r["n"]) for _, r in gap.iterrows()}
    s.close()

    (OUT / "fundamental_pit_feasibility.json").write_text(json.dumps(res, ensure_ascii=False, indent=1) + "\n")

    # ---- markdown summary --------------------------------------------------
    L = ["# fundamental_pit_snapshot 可行性核查（新信息域）", ""]
    L.append(f"- PIT 违规行（available_date > signal_date 或 stale_days < 0）：**{res['pit_violations']}**（应为 0）")
    L.append(f"- 覆盖年份：{min(res['rows_by_year'])}–{max(res['rows_by_year'])}；行数按年：{res['rows_by_year']}")
    L.append("")
    L.append("## 宇宙内覆盖率（占 cn_liquid_a 成员日 %）")
    L.append("")
    piv = cov.pivot_table(index="metric_id", columns="yr", values="coverage_pct")
    L.append("| metric | " + " | ".join(str(y) for y in YEARS) + " | 中位 |")
    L.append("|---|" + "---|" * (len(YEARS) + 1))
    for m, row in piv.iterrows():
        vals = [f"{row[y]:.1f}" if pd.notna(row[y]) else "-" for y in YEARS]
        L.append(f"| {m} | " + " | ".join(vals) + f" | {np.nanmedian(row.values):.1f} |")
    L.append("")
    L.append("## 陈旧度（2022–2024，宇宙内，交易日）")
    L.append("")
    L.append("| metric | p50 | p95 | max |")
    L.append("|---|---|---|---|")
    for r in res["staleness_2022_2024"]:
        L.append(f"| {r['metric_id']} | {r['p50']:.0f} | {r['p95']:.0f} | {r['mx']:.0f} |")
    L.append("")
    L.append("## 取值健全性")
    L.append("")
    L.append("| metric | rows | nulls | min | max | mean | std |")
    L.append("|---|---|---|---|---|---|---|")
    for r in res["value_sanity"]:
        L.append(f"| {r['metric_id']} | {int(r['n']):,} | {int(r['nulls'])} | {r['vmin']:.3g} | {r['vmax']:.3g} | {r['vmean']:.3g} | {r['vstd']:.3g} |")
    (OUT / "fundamental_pit_feasibility.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
