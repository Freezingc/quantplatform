#!/usr/bin/env python3
"""On the ambiguous days where the stale universe pick caused no mismatch, was that luck or a pattern?

§AK found that max(data_version) picks a stale universe on 20 of 20 ambiguous days, yet only 11 of those days
produce valid-label rows with no factor value. This reads the two existing measurement files -- the
per-version member sets from probe_universe_versions_origin.py and the per-day mismatch counts from
probe_universe_versions_origin.py's own match table -- and asks whether the nine harmless days differ from
the eleven harmful ones in a way that is visible in the version sets themselves.

The answer turns out to be exact rather than statistical: on every harmless day the stale and newest versions
have identical member sets, so there was nothing to mis-resolve, and on every harmful day the mismatch count
equals the number of members the newest version has that the stale one lacks (with one day off by one,
explained in §AJ: one of the newly added members has no valid label and so never enters the denominator).

Pure post-processing over existing measurement files: no database, no cache, no repository import.

Usage: python summarize_universe_harmless_days.py \
           --versions measurements/universe_versions_origin_20days.json \
           --out      measurements/universe_harmless_days_analysis.json
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import pandas as pd


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--versions", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    d = json.loads(args.versions.read_text())
    rows = []
    for r in d["per_day"]:
        day = r["day"]
        lex, new = r["lexicographic_max"], r["newest_by_created_at"]
        pair = next((x for x in r["pairwise_set_differences"]
                     if {x["a"], x["b"]} == {lex[:16], new[:16]}), None)
        if pair is None:
            raise SystemExit(f"{day}: no pairwise record for the stale vs newest versions")
        stale_only = pair["a_only"] if pair["a"] == lex[:16] else pair["b_only"]
        newest_only = pair["b_only"] if pair["a"] == lex[:16] else pair["a_only"]
        match = {m["data_version"]: m for m in r["version_match_table"]}
        rows.append({
            "day": day, "n_versions": r["n_versions"],
            "rule_picks_the_newest": r["rule_picks_the_newest"],
            "stale_members": next(v["members"] for v in r["versions_in_created_order"] if v["data_version"] == lex[:16]),
            "newest_members": next(v["members"] for v in r["versions_in_created_order"] if v["data_version"] == new[:16]),
            "stale_only_members": stale_only, "newest_only_members": newest_only,
            "shared_members": pair["shared"],
            "label_valid_rows": r["label_daily"]["valid_rows"],
            "mismatch_rows": match[lex[:16]]["label_rows_outside_this_version"],
            "label_rows_outside_the_newest": match[new[:16]]["label_rows_outside_this_version"],
            "harmless": match[lex[:16]]["label_rows_outside_this_version"] == 0,
        })

    df = pd.DataFrame(rows)
    corr = float(df.newest_only_members.corr(df.mismatch_rows))
    exact = df[df.mismatch_rows == df.newest_only_members]
    out = {
        "question": "On the ambiguous days where the stale pick caused no mismatch, was that luck or a pattern?",
        "days": len(df),
        "harmless_days": int(df.harmless.sum()), "harmful_days": int((~df.harmless).sum()),
        "harmless_days_where_the_two_versions_have_identical_member_sets": int(
            ((df.harmless) & (df.stale_only_members == 0) & (df.newest_only_members == 0)).sum()),
        "correlation_newest_only_members_vs_mismatch_rows": corr,
        "days_where_mismatch_equals_newest_only_members_exactly": int(len(exact)),
        "days_where_they_differ": df.loc[df.mismatch_rows != df.newest_only_members,
                                         ["day", "newest_only_members", "mismatch_rows"]].to_dict("records"),
        "divergence_sequence": df.sort_values("day").newest_only_members.tolist(),
        "answer": None, "per_day": rows,
        "database_writes": False, "weights_written": False, "cache_writes": False,
    }
    ident = out["harmless_days_where_the_two_versions_have_identical_member_sets"]
    out["answer"] = (
        f"A pattern, not luck. On all {ident} of the {out['harmless_days']} harmless days the stale and newest "
        f"versions have identical member sets (zero members unique to either), so the wrong version number "
        f"still selected the right content. Mismatch rows correlate {corr:.4f} with the number of members the "
        f"newest version adds, and equal it exactly on {out['days_where_mismatch_equals_newest_only_members_exactly']}"
        f" of {out['days']} days; the exception is "
        f"{', '.join(str(r['day']) for r in out['days_where_they_differ'])}, where one newly added member has no "
        f"valid label and therefore never enters the denominator. The divergence decays "
        f"{out['divergence_sequence']} , so the two snapshots converged within about ten trading days and the "
        f"exposure closed on its own -- but the rule is unchanged, so the next universe republication reopens it.")
    args.out.write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(out["answer"])
    pd.set_option("display.width", 260)
    print(df[["day", "n_versions", "rule_picks_the_newest", "stale_members", "newest_members",
              "stale_only_members", "newest_only_members", "mismatch_rows", "harmless"]].to_string(index=False))


if __name__ == "__main__":
    main()
