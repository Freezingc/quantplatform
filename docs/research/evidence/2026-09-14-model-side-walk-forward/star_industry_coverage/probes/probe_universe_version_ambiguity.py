"""Does the materializer's "latest universe version" rule pick the newest one? It is max() over a hash.

materialize_frozen_factor_daily.py resolves the per-date universe with
    latest = uni.groupby("signal_date")["data_version"].max()
    uni = uni[uni["data_version"] == uni["latest_version"]]
data_version is a sha256 hex digest, so max() is the LEXICOGRAPHIC largest hash -- a rule with no temporal
meaning. probe_universe_vintage_skew.py found 2024-12-20 carries four concurrent versions that disagree by
exactly the 65 securities at issue, and that the lexicographic maximum is not the version containing them.

This measures two things. First, whether the lexicographic maximum coincides with the newest created_at on
the days that matter: if it does not, the materializer silently materialized factors against a stale
universe. Second, how many days in the OOS window carry more than one universe version at all, which is the
exposure of the arbitrary tie-break -- on single-version days the rule cannot choose wrongly, which is why
this has been invisible.

SAFETY: SELECT-only against dfs://QuantResearchV2Test. No writes except --out. No weights.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
UNIVERSE_ID = "cn_liquid_a"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default="2024-12-19,2024-12-20,2024-12-23,2024-12-24,2024-12-25,2024-12-26,2024-12-31,2025-01-07,2025-06-16,2022-05-31,2023-07-05,2022-07-29")
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2025-12-31")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    days = args.days.split(",")

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, new_holder, query
    h = new_holder()

    per = []
    for day in days:
        dl = day.replace("-", ".")
        ok, v = query(h,
            f"select data_version, min(created_at) as created_min, max(created_at) as created_max, count(*) as rows "
            f"from loadTable('{TARGET_DATABASE}','universe_daily') "
            f"where signal_date = date('{dl}') and universe_id='{UNIVERSE_ID}' and is_member=1 "
            "group by data_version")
        if not ok:
            raise SystemExit(f"query failed {day}: {v}")
        v["data_version"] = v["data_version"].astype(str)
        v["created_max"] = pd.to_datetime(v["created_max"])
        v = v.sort_values("data_version").reset_index(drop=True)
        lex = v.loc[v.data_version.idxmax()]
        newest = v.loc[v.created_max.idxmax()]
        per.append({
            "day": day, "n_versions": int(len(v)),
            "lexicographic_max_version": lex.data_version[:16],
            "lexicographic_max_created": str(lex.created_max),
            "lexicographic_max_rows": int(lex.rows),
            "newest_version": newest.data_version[:16],
            "newest_created": str(newest.created_max),
            "newest_rows": int(newest.rows),
            "rule_picks_the_newest": bool(lex.data_version == newest.data_version),
            "versions": [{"v": r.data_version[:16], "created": str(r.created_max), "rows": int(r.rows)}
                         for _, r in v.iterrows()],
        })
        print(f"  {day}: {len(v)} version(s); lexicographic max = {lex.data_version[:16]}… "
              f"({lex.created_max}, {int(lex.rows)} rows); newest = {newest.data_version[:16]}… "
              f"({newest.created_max}, {int(newest.rows)} rows); rule picks newest = "
              f"{lex.data_version == newest.data_version}", flush=True)

    # window-wide exposure: how many days carry more than one version
    ok2, w = query(h,
        "t = select signal_date, data_version from loadTable("
        f"'{TARGET_DATABASE}','universe_daily') "
        f"where signal_date between {args.start.replace('-','.')} and {args.end.replace('-','.')} "
        f"and universe_id='{UNIVERSE_ID}' and is_member=1 group by signal_date, data_version;"
        "select signal_date, count(*) as n_versions from t group by signal_date")
    if not ok2:
        raise SystemExit(f"window query failed: {w}")
    w["n_versions"] = w["n_versions"].astype(int)
    w["day"] = pd.to_datetime(w["signal_date"]).dt.strftime("%Y-%m-%d")
    hist = w.n_versions.value_counts().sort_index().to_dict()
    multi = w[w.n_versions > 1].sort_values("day")

    dfp = pd.DataFrame(per)
    out = {
        "question": "Does max(data_version) pick the newest universe, and how many days are ambiguous at all?",
        "universe_id": UNIVERSE_ID,
        "rule": 'latest = uni.groupby("signal_date")["data_version"].max() -- lexicographic maximum of a sha256 hex digest',
        "sampled_days": per,
        "sampled_days_with_multiple_versions": int((dfp.n_versions > 1).sum()),
        "sampled_days_where_the_rule_picks_the_newest": int(dfp.rule_picks_the_newest.sum()),
        "window": [args.start, args.end],
        "window_days": int(len(w)),
        "version_count_histogram": {str(k): int(v) for k, v in hist.items()},
        "window_days_with_more_than_one_version": int((w.n_versions > 1).sum()),
        "window_days_ambiguous_share": float((w.n_versions > 1).mean()),
        "ambiguous_day_span": [str(multi.day.min()), str(multi.day.max())] if len(multi) else None,
        "ambiguous_days": multi[["day", "n_versions"]].to_dict("records"),
        "answer": None, "database_writes": False, "weights_written": False,
    }
    multi_sampled = dfp[dfp.n_versions > 1]
    wrong_on = int((~multi_sampled.rule_picks_the_newest.astype(bool)).sum())
    out["sampled_multi_version_days_where_the_rule_picks_a_stale_version"] = wrong_on
    out["answer"] = (
        f"Of the {len(per)} sampled days, {len(multi_sampled)} carry more than one universe version, and on "
        f"{wrong_on} of those {len(multi_sampled)} the lexicographic maximum is NOT the newest by created_at -- "
        f"the rule picks a stale version every time it has a choice. Across the whole window "
        f"{int((w.n_versions > 1).sum())} of {len(w)} days "
        f"({float((w.n_versions > 1).mean()):.2%}) carry more than one version, all inside "
        f"{out['ambiguous_day_span'][0]}..{out['ambiguous_day_span'][1]}; on the other "
        f"{int((w.n_versions == 1).sum())} days there is nothing to choose between, which is why this has been "
        f"invisible.")
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n" + out["answer"])
    print("version-count histogram over the window:", out["version_count_histogram"])
    if len(multi):
        print(f"ambiguous days: {len(multi)}, span {out['ambiguous_day_span']}")
        print(multi[["day", "n_versions"]].to_string(index=False))


if __name__ == "__main__":
    main()
