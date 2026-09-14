"""Per-day STAR valid-label row counts for every pre-cutover OOS day, in one aggregate query per year.

The union-versus-replace projection needs the STAR gain per day. Measuring it day by day for 450 days
would pull millions of rows; grouping by (signal_date, three-character prefix) inside SQL returns about
thirteen thousand and answers the same question. Pre-cutover the STAR gain equals the STAR valid-label
row count, because a STAR row cannot be complete without an industry label and the cache has none
before 2023-09-25 -- verified without exception on all 50 days measured individually.

SAFETY: SELECT-only against dfs://QuantResearchV2Test. No writes anywhere except --out. No weights.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
STAR = ("688", "689")


def sql_list(values) -> str:
    return ", ".join("'" + str(v).replace("'", "''") + "'" for v in values)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2023-09-24")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "scripts"))
    sys.path.insert(0, str(REPO / "src"))
    import pandas as pd
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query

    holder = new_holder()
    lpins = current_versions(holder, "label_daily")
    per_day: dict[str, dict] = {}
    for year in range(int(args.start[:4]), int(args.end[:4]) + 1):
        ys = max(args.start, f"{year}-01-01"); ye = min(args.end, f"{year}-12-31")
        parts = [f"{year}-{m:02d}" for m in range(1, 13) if f"{year}-{m:02d}" <= ye[:7] and f"{year}-{m:02d}" >= ys[:7]]
        miss = [p for p in parts if p not in lpins]
        if miss:
            raise SystemExit(f"unpublished partitions for {year}: {miss}")
        ll = sql_list(sorted({lpins[p] for p in parts}))
        ok, df = query(holder,
            f"lab = select signal_date, substr(ts_code,0,3) as pfx from loadTable('{TARGET_DATABASE}','label_daily') "
            f"where signal_date >= {ys.replace('-','.')} and signal_date <= {ye.replace('-','.')} "
            f"and label_valid=1 and data_version in [{ll}];"
            "select signal_date, pfx, count(*) as n from lab group by signal_date, pfx")
        if not ok or not isinstance(df, pd.DataFrame):
            raise SystemExit(f"query failed for {year}: {df}")
        df["day"] = pd.to_datetime(df["signal_date"]).dt.strftime("%Y-%m-%d")
        df["pfx"] = df["pfx"].astype(str)
        for day, g in df.groupby("day"):
            per_day[day] = {"valid_label_rows": int(g["n"].sum()),
                            "star_valid_label_rows": int(g.loc[g["pfx"].isin(STAR), "n"].sum())}
        print(f"  {year}: {len(df)} prefix-groups, {df['day'].nunique()} days", flush=True)

    out = {
        "question": "How many valid-label rows are STAR rows, per day, across the whole pre-cutover OOS window?",
        "database": TARGET_DATABASE, "window": [args.start, args.end], "days": len(per_day),
        "star_min": min(v["star_valid_label_rows"] for v in per_day.values()),
        "star_max": max(v["star_valid_label_rows"] for v in per_day.values()),
        "valid_label_rows_match_the_instrument_on": None,
        "per_day": per_day, "database_writes": False, "weights_written": False,
    }
    inst = json.loads(Path("/tmp/dense_intersection_2022_2025_v2.json").read_text())
    ref = {r["day"]: r["valid_label_rows"] for r in inst["per_day"]}
    out["valid_label_rows_match_the_instrument_on"] = sum(
        1 for d, v in per_day.items() if ref.get(d) == v["valid_label_rows"])
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_day"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
