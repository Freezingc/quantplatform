"""Scan all 969 OOS days for valid-label rows that carry NONE of the nine dense factors.

probe_allnine_absent_origin.py found that the binding day 2024-12-20 is bad for a reason that is neither the
industry-label problem nor the suspension signature: 65 securities have a valid label -- so they are in the
floor's denominator -- but no row for ANY factor_id in factor_daily, while the upstream store holds 473 of
520 (security, factor) pairs for them. Neither option 1 nor option 2 can reach those rows, which is why that
day binds both. This measures how widespread the mode is: for every day in the window it buckets the
valid-label rows by how many of the nine dense factors they carry, and reports the count carrying zero.

Scope note: "zero" here means zero of the DENSE NINE, which is what the floor's numerator counts. On
2024-12-20 the same securities were separately confirmed to have no row for any factor_id at all, so on that
day the two notions coincide; this scan does not re-verify that equivalence on every day.

SAFETY: SELECT-only against dfs://QuantResearchV2Test. No writes except --out. No weights.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
DENSE9 = ["yeslab_206", "yeslab_357", "yeslab_425", "yeslab_381", "yeslab_443_reg30_max20_v1",
          "yeslab_464", "yeslab_471", "yeslab_484", "yeslab_96"]


def sql_list(v) -> str:
    return ", ".join("'" + str(x).replace("'", "''") + "'" for x in v)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    h = new_holder()
    lpins, fpins = current_versions(h, "label_daily"), current_versions(h, "factor_daily")
    ids = sql_list(DENSE9)

    # lpins keys are YYYY-MM partitions; compare on the month prefix, otherwise "2022-01" < "2022-01-01"
    # lexicographically and the first month of the window is silently dropped
    all_days = sorted({d for d in lpins if args.start[:7] <= d <= args.end[:7]})
    print(f"  {len(all_days)} published months from {all_days[0]} to {all_days[-1]}", flush=True)

    rows = []
    for year in sorted({d[:4] for d in all_days}):
        parts = [p for p in all_days if p.startswith(year)]
        # DolphinDB date literals are YYYY.MM.DD; date('...') is rejected with "Failed to convert the string to DATE"
        lo = parts[0].replace("-", ".") + ".01"
        hi = parts[-1][:4] + ".12.31"
        ll = sql_list(sorted({lpins[p] for p in parts}))
        fl = sql_list(sorted({fpins[p] for p in parts}))
        # per-day histogram of "how many of the nine does each valid-label row carry", done in the database
        script = (
            f"lab = select signal_date, ts_code from loadTable('{TARGET_DATABASE}','label_daily') "
            f"where signal_date between {lo} and {hi} and label_valid=1 and data_version in [{ll}];"
            f"fac = select signal_date, ts_code, factor_id from loadTable('{TARGET_DATABASE}','factor_daily') "
            f"where signal_date between {lo} and {hi} and factor_id in [{ids}] and data_version in [{fl}];"
            "dk = select signal_date, ts_code, factor_id from fac group by signal_date, ts_code, factor_id;"
            "cnt = select signal_date, ts_code, count(*) as c from dk group by signal_date, ts_code;"
            "j = select signal_date, ts_code, long(0) as c from lab;"
            "u = unionAll(select signal_date, ts_code, long(c) as c from cnt, j);"
            "m = select max(c) as c from u group by signal_date, ts_code;"
            "select signal_date, c, count(*) as n from m group by signal_date, c")
        ok, res = query(h, script)
        if not ok or not isinstance(res, pd.DataFrame):
            raise SystemExit(f"query failed for {year}: {res}")
        res["day"] = pd.to_datetime(res["signal_date"]).dt.strftime("%Y-%m-%d")
        piv = res.pivot_table(index="day", columns="c", values="n", aggfunc="sum", fill_value=0)
        for c in range(0, len(DENSE9) + 1):
            if c not in piv:
                piv[c] = 0
        piv = piv.sort_index()
        for day, r in piv.iterrows():
            V = int(r.sum()); N = int(r[len(DENSE9)])
            rows.append({"day": day, "valid_label_rows": V, "intersection_rows": N,
                         "coverage": N / V if V else None,
                         "zero_of_nine": int(r[0]),
                         **{f"carrying_{c}_of_9": int(r[c]) for c in range(1, len(DENSE9))}})
        print(f"  {year}: {len(piv)} days, zero-of-nine total {int(piv[0].sum())}", flush=True)

    df = pd.DataFrame(rows).sort_values("day").reset_index(drop=True)
    z = df.zero_of_nine
    out = {
        "question": "How many valid-label rows carry NONE of the nine dense factors, per day, across the whole OOS window?",
        "window": [args.start, args.end], "days": len(df),
        "zero_of_nine": {
            "days_with_any": int((z > 0).sum()), "days_with_none": int((z == 0).sum()),
            "total_rows": int(z.sum()), "min": int(z.min()), "median": float(z.median()),
            "p90": float(z.quantile(0.9)), "max": int(z.max()),
            "worst_day": str(df.loc[z.idxmax(), "day"]),
            "share_of_all_valid_label_rows": float(z.sum() / df.valid_label_rows.sum()),
        },
        "top_15_days": df.nlargest(15, "zero_of_nine")[
            ["day", "valid_label_rows", "intersection_rows", "coverage", "zero_of_nine"]].to_dict("records"),
        "by_year": {str(y): {"days": int(len(g)), "zero_of_nine_total": int(g.zero_of_nine.sum()),
                             "zero_of_nine_median": float(g.zero_of_nine.median()),
                             "zero_of_nine_max": int(g.zero_of_nine.max())}
                    for y, g in df.assign(y=df.day.str[:4]).groupby("y")},
        "answer": None, "per_day": rows,
        "database_writes": False, "weights_written": False,
    }
    o = out["zero_of_nine"]
    out["answer"] = (
        f"{o['days_with_any']} of {len(df)} days have at least one valid-label row carrying none of the nine "
        f"({o['total_rows']} rows in total, {o['share_of_all_valid_label_rows']:.3%} of all valid-label rows); "
        f"the median day has {o['median']:.0f} and the worst is {o['worst_day']} with {o['max']}.")
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n" + out["answer"])
    print(pd.DataFrame(out["top_15_days"]).to_string(index=False))
    print(json.dumps(out["by_year"], ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
