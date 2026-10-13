"""Per-year coverage of each source panel over the labelled cn_liquid_a universe.

Why this matters for admission: margin_detail only covers two-margin-eligible names, so its
cross-section is a non-random, time-varying subset of the universe (the eligible list grew every
year of the window).  A factor whose coverage drifts is not comparable across folds, and it can
never satisfy the production score-health gate (coverage in [0.9, 1.0]) as a standalone score --
only as a sparse model feature.  moneyflow/cyq_perf are near-full coverage and serve as the
control.  Also reports per-year non-null share of each candidate column, so a field that decays
inside its own panel (e.g. rqye before short-selling resumed) is visible.

Read-only; reuses the published-data_version context from screen_daily_domains.load_context.
"""
from __future__ import annotations

import json
import pathlib
import sys

import pandas as pd

OUT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from screen_daily_domains import build_candidates, load_context, KEYS  # noqa: E402


def main() -> int:
    uni, lab = load_context()
    ctx = uni.merge(lab, on=KEYS, how="inner")
    ctx = ctx.loc[ctx["label_valid"].fillna(False).astype(bool)].copy()
    ctx["year"] = ctx["signal_date"].dt.year
    denom = ctx.groupby("year").size().rename("labelled_member_days")
    print(denom.to_string(), flush=True)

    cands = build_candidates()
    rows = []
    for source, frame in cands.items():
        # A left merge always fills the key columns, so presence has to be marked explicitly;
        # counting non-null ts_code would report 1.0 for every source (the v1 bug).
        marked = frame.assign(_present=True)
        m = ctx.loc[:, ["year"] + KEYS].merge(marked, on=KEYS, how="left")
        present = m.groupby("year")["_present"].apply(lambda s: int(s.notna().sum())).rename("matched_rows")
        for year in denom.index:
            row = {"source": source, "year": int(year), "labelled_member_days": int(denom.loc[year]),
                   "matched_rows": int(present.get(year, 0)),
                   "key_coverage": float(present.get(year, 0) / int(denom.loc[year]))}
            for col in [c for c in frame.columns if c not in KEYS]:
                g = m.loc[m["year"] == year, col]
                row[f"nonnull::{col}"] = float(g.notna().mean()) if len(g) else None
            rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "analysis" / "coverage_drift.csv", index=False)

    print("\n=== key coverage by source and year ===")
    print(df.pivot(index="source", columns="year", values="key_coverage").round(4).to_string())
    print("\n=== non-null share of each candidate column, by year ===")
    for source, g in df.groupby("source"):
        cols = [c for c in g.columns if c.startswith("nonnull::")]
        print(f"\n-- {source}")
        print(g.set_index("year")[cols].rename(columns=lambda c: c.split("::")[1]).round(4).to_string())
    (OUT / "analysis" / "coverage_drift.json").write_text(
        json.dumps({"labelled_member_days": {str(k): int(v) for k, v in denom.items()},
                    "rows": json.loads(df.to_json(orient="records"))}, ensure_ascii=False, indent=1))
    print("\nwrote analysis/coverage_drift.csv and analysis/coverage_drift.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
