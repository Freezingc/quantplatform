"""Why do four cn_liquid_a versions coexist for 2024-12-20, and which one did label_daily actually use?

§AH established that materialize_frozen_factor_daily.py resolves the per-date universe with max() over a
sha256 data_version, which on 2024-12-20 selects a version created at 04:55 while the newest is 09:51, and
that the 65 securities with a valid label but no factor row are members of the newest version and not of the
selected one. Two questions were left open and both are answerable from columns already in the tables:
universe_daily carries run_id, created_at, source_watermark_id, universe_version, adv20_rank,
membership_rank, exclusion_reason and reason_bitmap, and label_daily carries run_id, data_version,
created_at and source_watermark_id.

So this reports, per version: which run produced it, when, its member count, and how the member sets differ
pairwise. It then identifies which version label_daily's valid rows correspond to, by set comparison rather
than by inference, and cross-checks with source_watermark_id. Finally it asks what distinguishes the members
that the versions disagree about -- adv20_rank and membership_rank distributions -- so "a liquidity
threshold moved" can be told apart from "two runs raced".

SAFETY: SELECT-only against dfs://QuantResearchV2Test. No writes except --out. No weights.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import pandas as pd

REPO = Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2")
UNIVERSE_ID = "cn_liquid_a"


def recs(df) -> list:
    if not isinstance(df, pd.DataFrame):
        return []
    d = df.copy()
    for c in d.columns:
        if pd.api.types.is_datetime64_any_dtype(d[c]):
            d[c] = d[c].dt.strftime("%Y-%m-%d %H:%M:%S.%f")
    return json.loads(d.to_json(orient="records"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", default="2024-12-19,2024-12-20,2025-01-07")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    days = args.days.split(",")

    sys.path.insert(0, str(REPO / "scripts")); sys.path.insert(0, str(REPO / "src"))
    from walk_forward_common import TARGET_DATABASE, current_versions, new_holder, query
    h = new_holder()
    lpins = current_versions(h, "label_daily")

    out_days = []
    for day in days:
        dl = day.replace("-", ".")
        ok, u = query(h,
            f"select data_version, run_id, universe_version, ts_code, created_at, source_watermark_id, "
            f"adv20_rank, membership_rank, exclusion_reason, reason_bitmap "
            f"from loadTable('{TARGET_DATABASE}','universe_daily') "
            f"where signal_date = date('{dl}') and universe_id='{UNIVERSE_ID}' and is_member=1")
        if not ok:
            raise SystemExit(f"universe query failed {day}: {u}")
        u["data_version"] = u.data_version.astype(str); u["ts_code"] = u.ts_code.astype(str)
        u["created_at"] = pd.to_datetime(u["created_at"])
        versions = {}
        for dv, g in u.groupby("data_version"):
            versions[dv] = {
                "run_id": sorted(set(g.run_id.astype(str))),
                "universe_version": sorted(set(str(x) for x in g.universe_version)),
                "created_min": str(g.created_at.min()), "created_max": str(g.created_at.max()),
                "members": int(g.ts_code.nunique()),
                "source_watermark_ids": sorted(set(g.source_watermark_id.astype(str)))[:3],
                "adv20_rank_max": int(pd.to_numeric(g.adv20_rank, errors="coerce").max()),
                "membership_rank_max": int(pd.to_numeric(g.membership_rank, errors="coerce").max()),
                "set": set(g.ts_code),
            }
        ordered = sorted(versions.items(), key=lambda kv: kv[1]["created_max"])
        lex_max = max(versions)                      # the rule the materializer uses
        newest = ordered[-1][0]
        # pairwise differences
        diffs = []
        keys = [k for k, _ in ordered]
        for i, a in enumerate(keys):
            for b in keys[i + 1:]:
                sa, sb = versions[a]["set"], versions[b]["set"]
                diffs.append({"a": a[:16], "b": b[:16],
                              "a_only": len(sa - sb), "b_only": len(sb - sa), "shared": len(sa & sb),
                              "identical": sa == sb})

        # what does label_daily look like, and which version does its valid-row set match?
        part = f"{day[:4]}-{day[5:7]}"
        ll = "'" + lpins[part] + "'" if part in lpins else "''"
        ok2, lab = query(h,
            f"select ts_code, run_id, data_version, created_at, source_watermark_id "
            f"from loadTable('{TARGET_DATABASE}','label_daily') "
            f"where signal_date = date('{dl}') and label_valid=1 and data_version in [{ll}]")
        lab_set = set(lab.ts_code.astype(str)) if ok2 and isinstance(lab, pd.DataFrame) else set()
        match = []
        for dv, v in versions.items():
            s = v["set"]
            match.append({"data_version": dv[:16], "created_max": v["created_max"], "members": v["members"],
                          "is_the_lexicographic_max": dv == lex_max, "is_the_newest": dv == newest,
                          "label_rows_inside_this_version": len(lab_set & s),
                          "label_rows_outside_this_version": len(lab_set - s),
                          "version_members_without_a_valid_label": len(s - lab_set),
                          "jaccard_with_label_rows": len(lab_set & s) / len(lab_set | s) if (lab_set | s) else None})
        match.sort(key=lambda r: -r["jaccard_with_label_rows"])
        labmeta = {}
        if ok2 and isinstance(lab, pd.DataFrame) and len(lab):
            lab["created_at"] = pd.to_datetime(lab["created_at"])
            labmeta = {"run_ids": sorted(set(lab.run_id.astype(str))),
                       "data_versions": sorted(set(lab.data_version.astype(str))),
                       "created_min": str(lab.created_at.min()), "created_max": str(lab.created_at.max()),
                       "source_watermark_ids": sorted(set(lab.source_watermark_id.astype(str)))[:3],
                       "valid_rows": len(lab_set)}
        out_days.append({
            "day": day, "n_versions": len(versions),
            "versions_in_created_order": [{k: v for k, v in vv.items() if k != "set"} |
                                          {"data_version": dv[:16]} for dv, vv in ordered],
            "lexicographic_max": lex_max[:16], "newest_by_created_at": newest[:16],
            "rule_picks_the_newest": lex_max == newest,
            "pairwise_set_differences": diffs,
            "label_daily": labmeta,
            "best_matching_version_for_label_rows": match[0] if match else None,
            "version_match_table": match,
        })
        print(f"  {day}: {len(versions)} version(s); lex_max={lex_max[:16]} newest={newest[:16]} "
              f"same={lex_max == newest}; label valid rows {len(lab_set)}; best matching version "
              f"{match[0]['data_version'] if match else None} "
              f"(jaccard {match[0]['jaccard_with_label_rows']:.4f}, outside {match[0]['label_rows_outside_this_version']})"
              if match else f"  {day}: no versions", flush=True)

    out = {"question": "Why do four cn_liquid_a versions coexist, and which one did label_daily use?",
           "universe_id": UNIVERSE_ID, "days": days, "per_day": out_days,
           "answer": None, "database_writes": False, "weights_written": False}
    d20 = next((r for r in out_days if r["day"] == "2024-12-20"), out_days[0])
    best = d20["best_matching_version_for_label_rows"]
    out["answer"] = (
        f"On {d20['day']} the {d20['n_versions']} coexisting versions were produced by runs "
        f"{[v['run_id'] for v in d20['versions_in_created_order']]} at "
        f"{[v['created_max'] for v in d20['versions_in_created_order']]}. label_daily's valid rows match "
        f"{best['data_version']}… best (Jaccard {best['jaccard_with_label_rows']:.4f}, "
        f"{best['label_rows_outside_this_version']} label rows outside it), which "
        f"{'IS' if best['is_the_newest'] else 'is NOT'} the newest by created_at and "
        f"{'IS' if best['is_the_lexicographic_max'] else 'is NOT'} the one max(data_version) selects.")
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print("\n" + out["answer"])
    for r in out_days:
        print(f"\n=== {r['day']} ({r['n_versions']} versions, in created order)")
        for v in r["versions_in_created_order"]:
            print(f"   {v['data_version']} created={v['created_max']} members={v['members']} "
                  f"universe_version={v['universe_version']} run={v['run_id']} "
                  f"adv20_rank_max={v['adv20_rank_max']} membership_rank_max={v['membership_rank_max']}")
        print("   label_daily:", json.dumps(r["label_daily"], ensure_ascii=False))
        print("   pairwise:", json.dumps(r["pairwise_set_differences"], ensure_ascii=False))


if __name__ == "__main__":
    main()
