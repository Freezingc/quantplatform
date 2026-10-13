"""Redundancy and effective-dimension check for the daily-domain IC-screen survivors.

Window 2025, the same convention as the 2026-10-13 alpha393 recheck, so the two are directly
comparable.  Three questions are answered per survivor:

  1. pairwise vs the frozen v2 fifteen  -- daily |Spearman| of within-day percentile ranks
     (median / p90 / share >= 0.80 / share >= 0.90); >= 0.90 BLOCK, >= 0.80 MONITOR.
  2. pairwise vs the OTHER survivors    -- the v1 script omitted this, yet the margin family
     (buy_intensity, repay_intensity, net_fin_ratio, rzye_chg5) is algebraically entangled:
     net_fin_ratio = buy_intensity - repay_intensity and rzye_chg5 is its 5-day smoothing.
     Without this leg a domain can look like it adds four dimensions while adding one.
  3. set-incremental retention given dense9 -- residual RankIC after projecting the survivor's
     daily percentile rank on the dense nine's percentile ranks, over days where all nine are
     finite (within-day coverage reported, since the six sparse factors are NaN on down days).
     The 2026-10-13 method note found pairwise gates alone do not protect effective dimension
     (alpha393 passed every pairwise check at <= 0.66 yet retained only 10.2%), so retention
     >= 0.30 is reported alongside, not instead of, the pairwise verdict.

A greedy pass then counts how many survivors actually add a dimension: strongest |ICIR| first,
admitted only if its median |rho| stays below the 0.90 block cap against BOTH the frozen fifteen
and the survivors already admitted.

Alignment per source (PIT): moneyflow/cyq_perf publish on the evening of T, before the T+1 open
entry the label uses -> lag0; margin_detail publishes on the morning of T+1 ~08:30 -> lag1 is the
conservative choice and is the decision basis here (lag0 is defensible, 08:30 < 09:30 open, and
the screen reports both; the margin signal clears the bar under either).

Read-only: no DB writes, no artifact-store writes, no 2026/E9 reads (window ends 2025-12-31).
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import pandas as pd
import dolphindb as ddb

OUT = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(OUT))
from screen_daily_domains import build_candidates, load_context, KEYS, MIN_CS  # noqa: E402

DB = "dfs://QuantResearchV2Test"
SAFE_LAG = {"moneyflow": 0, "cyq_perf": 0, "margin_detail": 1}
# Mirrors build_candidates() in screen_daily_domains; main() asserts the two agree so a
# renamed/added candidate cannot silently map to the wrong source (and wrong safe lag).
CANDIDATE_COLUMNS = {
    "moneyflow": ["net_mf_amount", "net_lg", "net_elg", "net_sm",
                  "net_lg_ratio", "net_elg_ratio", "net_sm_ratio", "net_mf_ratio"],
    "cyq_perf": ["winner_rate", "cost_dispersion", "cost_iqr_rel", "high_dist", "cost_50pct"],
    "margin_detail": ["buy_intensity", "repay_intensity", "net_fin_ratio",
                      "short_to_long", "rzye_chg5", "rzye"],
}
# Derived from the frozen v2 manifest at run time (see frozen_sets); this literal is
# only the documented expectation, asserted against the manifest to catch drift.
DENSE9_EXPECTED = {"yeslab_96", "yeslab_206", "yeslab_357", "yeslab_381", "yeslab_425",
                   "yeslab_443_reg30_max20_v1", "yeslab_464", "yeslab_471", "yeslab_484"}
FROZEN_V2 = pathlib.Path(
    "/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/diagnostics/factor_set_frozen_v2.json"
)
WINDOW = ("2025.01.01", "2025.12.31")
PAIR_BLOCK, PAIR_MONITOR = 0.90, 0.80
RETENTION_BAR = 0.30
SV = "sv::"  # prefix keeping survivor columns distinct from factor_id columns in the wide frame


def frozen_sets() -> tuple[list[str], list[str], str]:
    """Return (all15, dense9, data_version) straight from the frozen v2 manifest."""
    manifest = json.loads(FROZEN_V2.read_text())
    all15 = [str(f) for f in manifest["factors"]]
    dense9 = [str(f) for f in manifest["baseline_eligible"]]
    if set(dense9) != DENSE9_EXPECTED:
        raise RuntimeError(f"baseline_eligible drifted from the documented dense9: {sorted(dense9)}")
    if int(manifest["count"]) != len(all15):
        raise RuntimeError("frozen v2 manifest count disagrees with its factor list")
    return all15, dense9, str(manifest["data_version"])


def source_of(candidate: str) -> str:
    hits = [s for s, cols in CANDIDATE_COLUMNS.items() if candidate in cols]
    if len(hits) != 1:
        raise RuntimeError(f"candidate {candidate!r} maps to {hits}; names must be unique across sources")
    return hits[0]


def select_survivors(threshold: float) -> pd.DataFrame:
    """One row per candidate, preferring the PIT-safe alignment for its source.

    The screen reports every candidate under both lag0 and lag1, so a candidate can clear the
    threshold twice.  Redundancy is only meaningful for the alignment that could actually be
    admitted, i.e. SAFE_LAG; fall back to the strongest |ICIR| row when the safe lag did not
    clear the threshold, and flag that in the output rather than hiding it.
    """
    frame = pd.read_csv(OUT / "analysis" / "ic_screen_summary.csv")
    frame = frame[frame["icir"].abs() >= threshold].copy()
    if frame.empty:
        return frame.assign(source=pd.Series(dtype=str), is_safe_lag=pd.Series(dtype=bool))
    frame["source"] = frame["candidate"].map(source_of)
    frame["is_safe_lag"] = [int(row.lag) == SAFE_LAG[row.source] for row in frame.itertuples(index=False)]
    frame = frame.sort_values(["candidate", "is_safe_lag", "icir"],
                              key=lambda s: s.abs() if s.name == "icir" else s,
                              ascending=[True, False, False])
    chosen = frame.drop_duplicates(subset=["candidate"], keep="first").reset_index(drop=True)
    off = chosen.loc[~chosen["is_safe_lag"], "candidate"].tolist()
    if off:
        print(f"  NOTE: safe-lag alignment did not clear the threshold for {off}; "
              f"using their strongest-|ICIR| row and flagging it", flush=True)
    return chosen


def align(frame: pd.DataFrame, cand: str, cal: np.ndarray, lag: int) -> pd.DataFrame:
    src = frame.loc[:, KEYS + [cand]].dropna(subset=[cand]).copy()
    if lag:
        pos = np.searchsorted(cal, src["signal_date"].to_numpy(), side="left")
        tgt = pos + lag
        keep = (tgt < len(cal)) & (cal[np.minimum(pos, len(cal) - 1)] == src["signal_date"].to_numpy())
        src = src.loc[keep]
        src["signal_date"] = cal[tgt[keep]]
    if src.duplicated(KEYS).any():
        raise RuntimeError(f"aligned candidate {cand} has duplicated keys")
    return src


def daily_abs_rho(frame: pd.DataFrame, a: str, b: str) -> dict | None:
    """Median/p90/tail-share of daily |Spearman| between two columns, on days with enough overlap."""
    m = frame.dropna(subset=[a, b])
    if m.empty:
        return None
    m = m.loc[m.groupby("signal_date")[a].transform("size") >= MIN_CS]
    if m.empty:
        return None
    ranked = m.assign(ra=m.groupby("signal_date")[a].rank(pct=True).to_numpy(),
                      rb=m.groupby("signal_date")[b].rank(pct=True).to_numpy())
    rho = ranked.groupby("signal_date").apply(
        lambda x: np.corrcoef(x["ra"], x["rb"])[0, 1], include_groups=False).dropna()
    if rho.empty:
        return None
    ar = rho.abs()
    return {"days": int(len(ar)), "median_abs_rho": float(ar.median()), "mean_abs_rho": float(ar.mean()),
            "p90_abs_rho": float(ar.quantile(0.90)),
            "share_ge_080": float((ar >= PAIR_MONITOR).mean()), "share_ge_090": float((ar >= PAIR_BLOCK).mean())}


def verdict_from(median_abs_rho: float | None) -> str:
    v = median_abs_rho or 0.0
    return "BLOCK" if v >= PAIR_BLOCK else "MONITOR" if v >= PAIR_MONITOR else "PASS"


def residual_retention(sub: pd.DataFrame, target: str, cond_cols: list[str]) -> dict:
    raw, res, cov, days = [], [], [], 0
    for _, g in sub.groupby("signal_date"):
        gg = g.loc[np.isfinite(g[cond_cols].to_numpy(dtype=float)).all(axis=1)]
        if len(gg) < MIN_CS:
            continue
        ry = pd.Series(gg[target].to_numpy(dtype=float)).rank(pct=True).to_numpy()
        rr = pd.Series(gg["label"].to_numpy(dtype=float)).rank(pct=True).to_numpy()
        rx = np.apply_along_axis(lambda c: pd.Series(c).rank(pct=True).to_numpy(), 0,
                                 gg[cond_cols].to_numpy(dtype=float))
        A = np.column_stack([np.ones(len(rx)), rx])
        beta, *_ = np.linalg.lstsq(A, ry, rcond=None)
        resid = ry - A @ beta
        if np.std(resid) < 1e-12 or np.std(rr) < 1e-12:
            continue
        days += 1
        raw.append(float(np.corrcoef(ry, rr)[0, 1]))
        res.append(float(np.corrcoef(resid, rr)[0, 1]))
        cov.append(len(gg) / len(g))
    if not days:
        return {"days": 0}
    mr, ms = float(np.mean(raw)), float(np.mean(res))
    return {"days": days, "mean_raw_rank_ic": mr, "mean_residual_rank_ic": ms,
            "retention": (ms / mr) if abs(mr) > 1e-12 else None,
            "mean_within_day_coverage": float(np.mean(cov)),
            "meets_retention_bar": bool(abs(ms / mr) >= RETENTION_BAR) if abs(mr) > 1e-12 else False}


def main(threshold: float = 0.25) -> int:
    all15, dense9, frozen_dv = frozen_sets()
    survivors = select_survivors(threshold)
    print(f"survivors (|ICIR| >= {threshold}, one row per candidate): {len(survivors)}", flush=True)
    results = {"scope": "DAILY_DOMAIN_REDUNDANCY_V2", "window": "2025", "threshold_abs_icir": threshold,
               "safe_lag": SAFE_LAG, "dense9": dense9, "all15": all15,
               "frozen_v2_data_version": frozen_dv,
               "pair_block": PAIR_BLOCK, "pair_monitor": PAIR_MONITOR, "retention_bar": RETENTION_BAR,
               "context_provenance": "universe_daily/label_daily filtered to the published data_version "
                                     "per YYYY-MM logical partition (walk_forward_common.filter_current)",
               "candidates": []}
    if survivors.empty:
        results["note"] = "NO_SURVIVORS_AT_THRESHOLD"
        (OUT / "analysis" / "redundancy_survivors.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
        print("no survivors at this threshold; nothing to redundancy-check")
        return 0

    uni, lab = load_context()
    ctx = uni.merge(lab, on=KEYS, how="inner")
    if ctx.duplicated(KEYS).any():
        raise RuntimeError("context has duplicated (signal_date, ts_code) keys")
    ctx = ctx[(ctx["signal_date"] >= "2025-01-01") & (ctx["signal_date"] <= "2025-12-31")].rename(
        columns={"raw_return": "label"})
    results["context_rows_2025"] = int(len(ctx))
    cal = np.sort(ctx["signal_date"].unique())
    cands = build_candidates()
    for source, expected in CANDIDATE_COLUMNS.items():
        actual = [c for c in cands[source].columns if c not in KEYS]
        if sorted(actual) != sorted(expected):
            raise RuntimeError(f"CANDIDATE_COLUMNS[{source}] is stale: panel has {actual}")

    # One wide frame: label + every survivor (at its chosen alignment) + all fifteen frozen factors.
    wide = ctx.loc[:, KEYS + ["label", "label_valid"]].copy()
    chosen = {}
    for row in survivors.itertuples(index=False):
        col = SV + row.candidate
        al = align(cands[row.source], row.candidate, cal, int(row.lag))
        wide = wide.merge(al.rename(columns={row.candidate: col}), on=KEYS, how="left")
        chosen[row.candidate] = col

    s = ddb.session(); s.connect("10.176.55.211", 8848, "xiahongjie", "123456")
    for fid in all15:
        df = s.run(f"select signal_date, ts_code, value from loadTable('{DB}','factor_daily') "
                   f"where factor_id=`{fid} and signal_date between {WINDOW[0]} : {WINDOW[1]}")
        df["signal_date"] = pd.to_datetime(df["signal_date"])
        if df.duplicated(KEYS).any():
            raise RuntimeError(f"factor_daily returned duplicated keys for {fid}; needs a version filter")
        non_null = int(df["value"].notna().sum())
        wide = wide.merge(df.rename(columns={"value": fid}), on=KEYS, how="left")
        print(f"  loaded {fid}: {non_null:,} non-null values in 2025", flush=True)
    s.close()
    labelled = wide.loc[wide["label_valid"].fillna(False).astype(bool)].copy()
    print(f"wide frame: {len(wide):,} rows, {len(labelled):,} with a valid label", flush=True)

    for row in survivors.itertuples(index=False):
        col = chosen[row.candidate]
        entry = {"source": row.source, "candidate": row.candidate, "lag": int(row.lag),
                 "safe_lag_for_source": SAFE_LAG[row.source], "used_safe_lag": bool(row.is_safe_lag),
                 "screen_icir": float(row.icir), "screen_mean_rank_ic": float(row.mean_rank_ic),
                 "rows_2025_non_null": int(labelled[col].notna().sum()),
                 "coverage_of_labelled_days": float(labelled[col].notna().mean())}
        pw = {fid: daily_abs_rho(labelled, col, fid) for fid in all15}
        entry["pairwise_vs_frozen15"] = {k: v for k, v in pw.items() if v}
        worst = max(entry["pairwise_vs_frozen15"].items(),
                    key=lambda kv: kv[1]["median_abs_rho"], default=(None, {"median_abs_rho": None}))
        entry["worst_frozen_partner"] = worst[0]
        entry["worst_frozen_median_abs_rho"] = worst[1]["median_abs_rho"]
        entry["pairwise_vs_frozen_verdict"] = verdict_from(worst[1]["median_abs_rho"])

        vs_surv = {}
        for other in survivors["candidate"]:
            if other == row.candidate:
                continue
            r = daily_abs_rho(labelled, col, chosen[other])
            if r:
                vs_surv[other] = r
        entry["pairwise_vs_other_survivors"] = vs_surv
        w2 = max(vs_surv.items(), key=lambda kv: kv[1]["median_abs_rho"],
                 default=(None, {"median_abs_rho": None}))
        entry["worst_survivor_partner"] = w2[0]
        entry["worst_survivor_median_abs_rho"] = w2[1]["median_abs_rho"]
        entry["pairwise_vs_survivor_verdict"] = verdict_from(w2[1]["median_abs_rho"])

        entry["set_incremental_given_dense9"] = residual_retention(labelled.dropna(subset=[col]), col, dense9)
        entry["overall_verdict"] = (
            "BLOCK" if "BLOCK" in (entry["pairwise_vs_frozen_verdict"], entry["pairwise_vs_survivor_verdict"])
            else "MONITOR" if "MONITOR" in (entry["pairwise_vs_frozen_verdict"], entry["pairwise_vs_survivor_verdict"])
            else "PASS")
        results["candidates"].append(entry)
        ret = entry["set_incremental_given_dense9"].get("retention")
        print(f"  {row.source}/{row.candidate} lag{row.lag}: vs frozen {entry['worst_frozen_partner']}="
              f"{(entry['worst_frozen_median_abs_rho'] or 0):.3f} [{entry['pairwise_vs_frozen_verdict']}] | "
              f"vs survivors {entry['worst_survivor_partner']}="
              f"{(entry['worst_survivor_median_abs_rho'] or 0):.3f} [{entry['pairwise_vs_survivor_verdict']}] | "
              f"retention={'n/a' if ret is None else round(ret, 3)}", flush=True)

    # Greedy effective-dimension count: strongest screen |ICIR| first, admitted only if it stays
    # under the block cap against the frozen fifteen AND the survivors already admitted.
    order = sorted(results["candidates"], key=lambda e: -abs(e["screen_icir"]))
    admitted: list[str] = []
    for e in order:
        blockers = []
        if e["pairwise_vs_frozen_verdict"] == "BLOCK":
            blockers.append(f"frozen:{e['worst_frozen_partner']}={e['worst_frozen_median_abs_rho']:.3f}")
        for kept in admitted:
            r = e["pairwise_vs_other_survivors"].get(kept)
            if r and r["median_abs_rho"] >= PAIR_BLOCK:
                blockers.append(f"admitted:{kept}={r['median_abs_rho']:.3f}")
        e["greedy_admitted"] = not blockers
        e["greedy_blockers"] = blockers
        if not blockers:
            admitted.append(e["candidate"])
    results["greedy_effective_dimension"] = {
        "rule": f"strongest |screen ICIR| first; admit if median |rho| < {PAIR_BLOCK} vs frozen15 and vs admitted",
        "admitted": admitted, "count": len(admitted),
        "rejected": {e["candidate"]: e["greedy_blockers"] for e in order if e["greedy_blockers"]},
        "note": "greedy uses the BLOCK cap (0.90) so it counts near-duplicates; the 0.80 monitor "
                "cap is reported per candidate but not applied here",
    }
    print(f"\ngreedy effective new dimensions from this screen: {len(admitted)} -> {admitted}", flush=True)

    (OUT / "analysis" / "redundancy_survivors.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))
    print("wrote analysis/redundancy_survivors.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(float(sys.argv[1]) if len(sys.argv) > 1 else 0.25))
