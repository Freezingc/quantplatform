"""Classify the securities that upstream carries but the replay panel does not, one by one.

finding.md section 15 inferred from sample names (000018.SZ and similar) that the 544-1,153 upstream-only
securities per day are long-suspended or delisted names, and recorded that this was inference, not
verification. It also recorded that the cause of the strict-subset relation had not been confirmed from
code. Both are closed here.

Cause, from source: fetch_full_tushare_shsz_market (audit_formula_replay_tushare.py:311-350) fills each
date from api().daily(trade_date=...), Tushare's daily endpoint, which returns a row ONLY for securities
that actually traded that day. The upstream store carries a row per listed security per date regardless.
So the replay panel is the set of securities with a trade, and the difference is by construction.

This then classifies each upstream-only security on the two extreme days -- the largest gap (2022-03-15,
1,153) and the smallest (2025-08-08, 544) -- using list_date and delist_date from the vendor's
stock_basic across all three listing states:

  already delisted      delist_date <= the day
  not yet listed        list_date  >  the day
  listed and trading    neither -- so it had no daily bar that day, i.e. SUSPENDED

SAFETY: SELECT-only against dfs://YesLab, local JSON reads, one read-only vendor call for listing dates
(cached). No writes except --out. No weights, no repository file touched.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
CACHE = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/tushare_cache")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--list-cache", type=Path, required=True)
    ap.add_argument("--table", default="factor_data10")
    ap.add_argument("--factor-name", default="GP_095")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    ld = pd.read_json(args.list_cache, orient="table")
    if "ts_code" not in ld.columns:            # the cache may carry it as the index
        ld = ld.reset_index()
    ld["list_p"] = pd.to_datetime(ld["list_date"].astype(str), format="%Y%m%d", errors="coerce")
    ld["delist_p"] = pd.to_datetime(ld["delist_date"].astype(str).str.replace(r"\.0$", "", regex=True),
                                    format="%Y%m%d", errors="coerce")
    info = ld.set_index(ld["ts_code"].astype(str))

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import new_holder, query
    holder = new_holder()

    rows = []
    for day in sorted(args.days):
        f = CACHE / f"daily_{day.replace('-', '')}.json"
        if not f.is_file():
            rows.append({"day": day, "replay_panel_cached": False}); continue
        d = json.loads(f.read_text())
        recs = d["data"] if isinstance(d, dict) and "data" in d else d
        replay = {str(r["ts_code"]) for r in recs}
        dl = day.replace("-", ".")
        # value must be selected and filtered: the store carries a row for every security in its master
        # list on every date, with value NULL where it could not compute one. Comparing against the
        # unfiltered row set counts not-yet-listed and long-delisted names as "upstream-only", which is
        # meaningless for a panel-composition argument.
        ok, up = query(holder,
            f"select ts_code, value from loadTable('dfs://YesLab','{args.table}') "
            f"where factor_name = '{args.factor_name}' and trade_date = date('{dl}')")
        if not ok or not isinstance(up, pd.DataFrame):
            rows.append({"day": day, "error": str(up)[:160]}); continue
        up["ts_code"] = up["ts_code"].astype(str)
        up["value"] = pd.to_numeric(up["value"], errors="coerce")
        upstream_all = set(up["ts_code"])
        upstream = set(up.loc[up["value"].notna(), "ts_code"])
        extra = sorted(upstream - replay)
        ts = pd.Timestamp(day)
        sub = info.reindex(extra)
        delisted = sub["delist_p"].notna() & (sub["delist_p"] <= ts)
        not_listed = sub["list_p"].notna() & (sub["list_p"] > ts)
        unknown = sub["list_p"].isna()
        suspended = ~(delisted | not_listed | unknown)
        rows.append({
            "day": day, "replay_panel_cached": True,
            "replay_panel": len(replay),
            "upstream_rows_all": len(upstream_all), "upstream_rows_with_a_value": len(upstream),
            "upstream_only_with_a_value": len(extra),
            "replay_only_against_all_rows": len(replay - upstream_all),
            "replay_only_against_valued_rows": len(replay - upstream),
            "sample_replay_only_against_valued": sorted(replay - upstream)[:10],
            "already_delisted": int(delisted.sum()),
            "not_yet_listed": int(not_listed.sum()),
            "listed_but_no_bar_that_day_suspended": int(suspended.sum()),
            "not_found_in_stock_basic": int(unknown.sum()),
            "star_among_upstream_only": sum(1 for c in extra if c.startswith(("688", "689"))),
            "sample_delisted": [c for c in extra if bool(delisted.get(c, False))][:8],
            "sample_suspended": [c for c in extra if bool(suspended.get(c, False))][:8],
        })
        print(f"  {day}: valued-upstream-only {len(extra)} (all-rows {len(upstream_all)}, valued {len(upstream)}, "
              f"replay {len(replay)}, replay-only-vs-valued {len(replay - upstream)}) -> "
              f"delisted {int(delisted.sum())}, "
              f"not-yet-listed {int(not_listed.sum())}, suspended {int(suspended.sum())}, "
              f"unknown {int(unknown.sum())}", flush=True)

    measured = [r for r in rows if r.get("upstream_only_with_a_value") is not None]
    tot = sum(r["upstream_only_with_a_value"] for r in measured)
    out = {
        "question": "What are the securities that upstream carries but the replay panel does not?",
        "cause_confirmed_from_source": {
            "file": "research_v2/scripts/audit_formula_replay_tushare.py",
            "function": "fetch_full_tushare_shsz_market", "lines": "311-350",
            "mechanism": "each date is filled from api().daily(trade_date=...), Tushare's daily endpoint, "
                         "which returns a row ONLY for securities that traded that day; the upstream store "
                         "carries a row per listed security per date regardless. The replay panel is "
                         "therefore the traded set, and the subset relation is by construction.",
        },
        "database_read": "dfs://YesLab", "table": args.table, "factor_name": args.factor_name,
        "days": measured and [r["day"] for r in measured] or [],
        "comparison_basis": "upstream rows WITH A NON-NULL value, not all rows; the store keeps a row "
                            "for every master-list security on every date and NULLs the ones it could not "
                            "compute, so an unfiltered comparison counts not-yet-listed and long-delisted "
                            "names as differences",
        "classification_totals": {
            "upstream_only_with_a_value": tot,
            "replay_only_against_all_rows_total": sum(r["replay_only_against_all_rows"] for r in measured),
            "replay_only_against_valued_rows_total": sum(r["replay_only_against_valued_rows"] for r in measured),
            "already_delisted": sum(r["already_delisted"] for r in measured),
            "not_yet_listed": sum(r["not_yet_listed"] for r in measured),
            "listed_but_no_bar_that_day_suspended": sum(r["listed_but_no_bar_that_day_suspended"] for r in measured),
            "not_found_in_stock_basic": sum(r["not_found_in_stock_basic"] for r in measured),
        },

        "verdict": None, "per_day": rows,
        "database_writes": False, "weights_written": False,
    }
    c = out["classification_totals"]
    explained = c["already_delisted"] + c["not_yet_listed"] + c["listed_but_no_bar_that_day_suspended"]
    out["verdict"] = (
        f"{explained} of {c['upstream_only_with_a_value']} valued upstream-only securities are explained "
        f"by listing status or by having no trade that day ({explained / max(1, c['upstream_only_with_a_value']):.4f}); "
        f"{c['not_found_in_stock_basic']} are not in stock_basic at all. Against ALL upstream rows the "
        f"replay-only count is {c['replay_only_against_all_rows_total']}; against VALUED rows it is "
        f"{c['replay_only_against_valued_rows_total']}, which is the comparison that matters."
    ) if c["upstream_only_with_a_value"] else "no measurable days"
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_day"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
