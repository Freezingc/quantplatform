"""Which securities are in the replay's market panel but not upstream's, and vice versa?

finding.md section 14 named the mechanism for yeslab_464's "scale right, ordering wrong" shape: its
calculator contains nine global cross-sectional operators, including _v14 = _v13.max(axis=1), a cs_max that
collapses the whole cross-section into one market-wide series which then serves as the regression regressor
for EVERY security; plus cs_zscore and cs_rank, which pin the scale by construction while letting the
ordering move with panel composition. The panel sizes were measured to differ (4,737 on 2022-01-04 rising
to 5,329 by 2023-12-29, against roughly 5,250-5,400 upstream). Section 14.5 recorded that only the SIZE
difference had been measured, not the COMPOSITION. This measures composition.

If the two panels differ by a few hundred names, cs_max can select a different extremum and every
security's regressor shifts, which is the mechanism. If they are the same set, the mechanism is dead and
464 needs another explanation.

Replay panel: the cached per-date market fetch the materializer reads
(research_v2/artifacts/tushare_cache/daily_<YYYYMMDD>.json). Upstream panel: the securities holding a
GP_095 value on that date in dfs://YesLab/factor_data10.

SAFETY: SELECT-only against dfs://YesLab, local JSON reads. No writes except --out. No weights.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

CACHE = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/tushare_cache")
REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
GP095 = ("factor_data10", "GP_095")
STAR = ("688", "689")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import new_holder, query
    holder = new_holder()
    table, name = GP095

    rows = []
    for day in sorted(args.days):
        f = CACHE / f"daily_{day.replace('-', '')}.json"
        if not f.is_file():
            rows.append({"day": day, "replay_panel_cached": False}); continue
        d = json.loads(f.read_text())
        recs = d["data"] if isinstance(d, dict) and "data" in d else d
        replay = {str(r["ts_code"]) for r in recs}
        dl = day.replace("-", ".")
        ok, up = query(holder,
            f"select ts_code, value from loadTable('dfs://YesLab','{table}') "
            f"where factor_name = '{name}' and trade_date = date('{dl}')")
        if not ok or not isinstance(up, pd.DataFrame):
            rows.append({"day": day, "replay_panel_cached": True, "upstream_query_failed": str(up)[:160]}); continue
        up["ts_code"] = up["ts_code"].astype(str)
        up["value"] = pd.to_numeric(up["value"], errors="coerce")
        upstream = set(up.loc[up["value"].notna(), "ts_code"])
        upstream_all = set(up["ts_code"])
        only_replay = replay - upstream_all
        only_upstream = upstream_all - replay
        rows.append({
            "day": day, "replay_panel_cached": True,
            "replay_panel": len(replay),
            "upstream_rows": len(up), "upstream_with_a_value": len(upstream),
            "in_replay_only": len(only_replay), "in_upstream_only": len(only_upstream),
            "in_both": len(replay & upstream_all),
            "jaccard": len(replay & upstream_all) / len(replay | upstream_all) if (replay | upstream_all) else None,
            "in_replay_only_star": sum(1 for c in only_replay if c.startswith(STAR)),
            "in_upstream_only_star": sum(1 for c in only_upstream if c.startswith(STAR)),
            "sample_in_replay_only": sorted(only_replay)[:12],
            "sample_in_upstream_only": sorted(only_upstream)[:12],
        })
        print(f"  {day}: replay {len(replay)}, upstream {len(upstream_all)}, "
              f"replay-only {len(only_replay)}, upstream-only {len(only_upstream)}", flush=True)

    measured = [r for r in rows if r.get("jaccard") is not None]
    out = {
        "question": "Do the replay's market panel and the upstream GP_095 panel hold the same securities?",
        "replay_panel_source": str(CACHE / "daily_<YYYYMMDD>.json"),
        "upstream_panel_source": f"dfs://YesLab/{table} where factor_name = '{name}'",
        "days_requested": sorted(args.days), "days_measurable": len(measured),
        "days_without_a_cached_replay_panel": [r["day"] for r in rows if not r.get("replay_panel_cached")],
        "identical_panels_on_any_day": any(r["in_replay_only"] == 0 and r["in_upstream_only"] == 0 for r in measured),
        "jaccard_range": [min((r["jaccard"] for r in measured), default=None),
                           max((r["jaccard"] for r in measured), default=None)],
        "median_symmetric_difference": (sorted((r["in_replay_only"] + r["in_upstream_only"] for r in measured))
                                        [len(measured) // 2] if measured else None),
        "verdict": None, "per_day": rows,
        "database_writes": False, "weights_written": False,
    }
    if measured:
        out["verdict"] = (
            "PANELS DIFFER -- cs_max and the other global cross-sectional operators in "
            "GP_Factor_095_Calculator see a different cross-section than production did, which is the "
            "mechanism section 14 proposed. This confirms composition, not merely size."
            if not out["identical_panels_on_any_day"] else
            "PANELS IDENTICAL on at least one day -- the composition mechanism is not supported there"
        )
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_day"}, ensure_ascii=False, indent=1))
    for r in measured:
        print(f"  {r['day']}  replay {r['replay_panel']:5d}  upstream {r['upstream_with_a_value']:5d}  "
              f"replay-only {r['in_replay_only']:4d} (star {r['in_replay_only_star']:3d})  "
              f"upstream-only {r['in_upstream_only']:4d} (star {r['in_upstream_only_star']:3d})  J={r['jaccard']:.4f}")
        print(f"      replay-only sample {r['sample_in_replay_only'][:6]}")
        print(f"      upstream-only sample {r['sample_in_upstream_only'][:6]}")


if __name__ == "__main__":
    main()
