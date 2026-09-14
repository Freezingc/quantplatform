"""Did upstream always have those STAR rows, or were they backfilled after we materialized?

probe_star_quarantine_reason.py found NO quarantine rows for yeslab_357 / yeslab_471 on either
dropped day, or anywhere in 2023-06..2023-11. That is informative only together with the fact that
the clean/quarantine split is total: `deduplicate_factor_observations` sends every dropped key to
`factor_quarantine` with a reason. No quarantine row plus no `factor_daily` row means the
materialization never saw those keys at all -- it did not see them and reject them.

Combined with probe_star_missing_chain.py (upstream HAS finite values for all 318 STAR codes today),
that leaves one ordering question: our materialization ran at some moment; upstream's STAR coverage
for pre-2023-10 either existed then or did not. This reads the timestamps both sides keep:

  * `factor_daily.created_at` / `data_version` for the rows we did publish, before and after the
    October breakpoint -- one run or two?
  * the upstream table's own columns, to see whether YesLab records when a row was written at all
  * `factor_quarantine` row count overall, so "no quarantine rows" cannot be an empty-table artifact

SAFETY: SELECT-only. No writes, no weights, no artifact mutation. Run from research_v2/.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
sys.path.insert(0, "src")

import pandas as pd  # noqa: E402
from walk_forward_common import TARGET_DATABASE, new_holder, query  # noqa: E402

OUT = Path("/data/zhangyudong/qrv2_research_artifacts/star_backfill_timing_probe.json")
STAR_PREFIXES = ("688", "689")


def _run(holder: dict, sql: str) -> pd.DataFrame:
    ok, frame = query(holder, sql)
    if not ok or not isinstance(frame, pd.DataFrame):
        raise RuntimeError(f"query failed: {sql}")
    return frame


def main() -> None:
    holder = new_holder()
    result: dict = {}

    print("=== factor_quarantine: is the table populated at all? ===", flush=True)
    total = _run(holder, f"select count(*) as n from loadTable('{TARGET_DATABASE}','factor_quarantine')")
    n_quarantine = int(total["n"].iloc[0])
    result["factor_quarantine_total_rows"] = n_quarantine
    print(f"  total rows: {n_quarantine}", flush=True)
    if n_quarantine:
        span = _run(
            holder,
            "select min(observation_date) as lo, max(observation_date) as hi, "
            "count(*) as n2 "
            f"from loadTable('{TARGET_DATABASE}','factor_quarantine')",
        )
        result["factor_quarantine_span"] = {str(k): str(span[k].iloc[0]) for k in span.columns}
        print(f"  span: {result['factor_quarantine_span']}", flush=True)
        factors = _run(
            holder,
            "select distinct(factor_id) as fid "
            f"from loadTable('{TARGET_DATABASE}','factor_quarantine')",
        )
        result["factor_quarantine_distinct_factors"] = int(len(factors))
        print(f"  distinct factors quarantined: {len(factors)}", flush=True)
        reasons = _run(
            holder,
            "select reason, count(*) as n "
            f"from loadTable('{TARGET_DATABASE}','factor_quarantine') group by reason order by n desc",
        )
        result["factor_quarantine_reasons"] = {
            str(r.reason): int(r.n) for r in reasons.itertuples(index=False)
        }
        print(f"  reasons: {result['factor_quarantine_reasons']}", flush=True)

    print("\n=== upstream table columns (does YesLab record a write time?) ===", flush=True)
    result["upstream_columns"] = {}
    for table in ("factor_data8", "factor_data10"):
        sample = _run(holder, f"select top 3 * from loadTable('dfs://YesLab', '{table}')")
        result["upstream_columns"][table] = list(sample.columns)
        print(f"  {table}: {list(sample.columns)}", flush=True)

    print("\n=== factor_daily: one materialization run or two? ===", flush=True)
    result["factor_daily"] = {}
    fd_columns = list(
        _run(holder, f"select top 1 * from loadTable('{TARGET_DATABASE}','factor_daily')").columns
    )
    result["factor_daily_columns"] = fd_columns
    print(f"  factor_daily columns: {fd_columns}", flush=True)
    wanted = ["signal_date", "ts_code", "value", "data_version"]
    for extra in ("created_at", "run_id", "source_watermark_id", "available_date"):
        if extra in fd_columns:
            wanted.append(extra)
    select_list = ", ".join(wanted)
    for factor_id in ("yeslab_357", "yeslab_471", "yeslab_381"):
        rows = _run(
            holder,
            f"select {select_list} "
            f"from loadTable('{TARGET_DATABASE}','factor_daily') "
            f"where factor_id='{factor_id}' "
            "and signal_date between date('2023.07.01') and date('2023.10.31')",
        )
        if rows.empty:
            result["factor_daily"][factor_id] = {"rows": 0}
            print(f"  {factor_id}: no rows in 2023-07..2023-10", flush=True)
            continue
        rows["month"] = pd.to_datetime(rows["signal_date"]).dt.strftime("%Y-%m")
        is_star = rows["ts_code"].astype(str).str.startswith(STAR_PREFIXES)
        per_month = {}
        for month, group in rows.assign(_star=is_star).groupby("month"):
            stats = {
                "rows": int(len(group)),
                "star_rows": int(group["_star"].sum()),
                "data_versions": sorted(group["data_version"].astype(str).str[:12].unique()),
            }
            for extra in ("created_at", "run_id", "source_watermark_id"):
                if extra in group.columns:
                    values = group[extra].astype(str)
                    stats[f"{extra}_min"] = str(values.min())
                    stats[f"{extra}_max"] = str(values.max())
            per_month[str(month)] = stats
        result["factor_daily"][factor_id] = {"rows": int(len(rows)), "per_month": per_month}
        for month, stats in sorted(per_month.items()):
            extras = " ".join(
                f"{key}={stats[key]}"
                for key in sorted(stats)
                if key.endswith("_min") or key.endswith("_max")
            )
            print(
                f"  {factor_id} {month}: rows={stats['rows']:>6} star={stats['star_rows']:>5} "
                f"versions={stats['data_versions']} {extras}",
                flush=True,
            )
        del rows

    print("\n=== upstream STAR rows on a pre-breakpoint day, all columns ===", flush=True)
    result["upstream_star_sample"] = {}
    for table, name in (
        ("factor_data8", "行业内收益标准化绝对值"),
        ("factor_data10", "AttentionSpillover"),
    ):
        rows = _run(
            holder,
            "select * from loadTable('dfs://YesLab', "
            f"'{table}') where factor_name = '{name}' "
            "and trade_date between date('2023.07.05') and date('2023.07.05')",
        )
        code = rows["ts_code"].astype(str)
        star = rows.loc[code.str.startswith(STAR_PREFIXES)]
        value = pd.to_numeric(star["value"], errors="coerce")
        entry = {
            "columns": list(rows.columns),
            "rows": int(len(rows)),
            "star_rows": int(len(star)),
            "star_finite": int(value.notna().sum()),
            "star_inf": int((value.abs() == float("inf")).sum()) if value.notna().any() else 0,
            "star_value_min": float(value.min()) if value.notna().any() else None,
            "star_value_max": float(value.max()) if value.notna().any() else None,
            "sample": [
                {str(col): str(row[col]) for col in rows.columns}
                for _, row in star.head(3).iterrows()
            ],
        }
        result["upstream_star_sample"][table] = entry
        print(
            f"  {table}/{name}: star_rows={entry['star_rows']} finite={entry['star_finite']} "
            f"inf={entry['star_inf']} range=[{entry['star_value_min']}, {entry['star_value_max']}]",
            flush=True,
        )
        for row in entry["sample"][:2]:
            print(f"          {row}", flush=True)
        del rows

    OUT.write_text(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    print(f"\nwrote {OUT}", flush=True)


if __name__ == "__main__":
    main()
