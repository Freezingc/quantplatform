"""IC screen for three daily non-price-volume domains against the project's own conventions.

Label forward_open_5d:v1 (label_valid), universe cn_liquid_a, daily cross-sectional Spearman,
ICIR = mean/std of the daily IC series (NON-annualised), minimum cross-section 30 -- identical
to research_v2.evaluation.rank_ic.compute_daily_rank_ic, which is imported rather than reimplemented.

Two alignments are reported because publication timing differs:
  lag0 : source row of trade_date T joined to signal_date T   (moneyflow / cyq_perf publish on the
         evening of T, i.e. before the T+1 open entry the label uses -> PIT-safe)
  lag1 : source row of T-1 joined to signal_date T            (conservative; margin_detail publishes
         on the morning of T+1 ~08:30, so its T row is only arguably available at the T+1 open)

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
RAW = OUT / "raw"
sys.path.insert(0, "/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/src")
from research_v2.evaluation.rank_ic import compute_daily_rank_ic  # noqa: E402

DB = "dfs://QuantResearchV2Test"
YEARS = (2022, 2023, 2024, 2025)
KEYS = ["signal_date", "ts_code"]
MIN_CS = 30


def build_candidates() -> dict[str, pd.DataFrame]:
    """Return {source: frame with signal_date, ts_code and derived candidate columns}."""
    out = {}
    mf = pd.concat([pd.read_parquet(RAW / f"moneyflow_{y}.parquet") for y in YEARS], ignore_index=True)
    amt = ["buy_sm_amount", "sell_sm_amount", "buy_md_amount", "sell_md_amount",
           "buy_lg_amount", "sell_lg_amount", "buy_elg_amount", "sell_elg_amount"]
    total = mf[amt].abs().sum(axis=1).replace(0, np.nan)
    mf["net_lg"] = (mf["buy_lg_amount"] + mf["buy_elg_amount"]) - (mf["sell_lg_amount"] + mf["sell_elg_amount"])
    mf["net_elg"] = mf["buy_elg_amount"] - mf["sell_elg_amount"]
    mf["net_sm"] = mf["buy_sm_amount"] - mf["sell_sm_amount"]
    mf["net_lg_ratio"] = mf["net_lg"] / total
    mf["net_elg_ratio"] = mf["net_elg"] / total
    mf["net_sm_ratio"] = mf["net_sm"] / total
    mf["net_mf_ratio"] = mf["net_mf_amount"] / total
    mf["signal_date"] = pd.to_datetime(mf["trade_date"], format="%Y%m%d")
    out["moneyflow"] = mf.loc[:, ["signal_date", "ts_code", "net_mf_amount", "net_lg", "net_elg", "net_sm",
                                  "net_lg_ratio", "net_elg_ratio", "net_sm_ratio", "net_mf_ratio"]]

    cy = pd.concat([pd.read_parquet(RAW / f"cyq_perf_{y}.parquet") for y in YEARS], ignore_index=True)
    cy["signal_date"] = pd.to_datetime(cy["trade_date"], format="%Y%m%d")
    cy["cost_dispersion"] = (cy["cost_95pct"] - cy["cost_5pct"]) / cy["cost_50pct"].replace(0, np.nan)
    cy["cost_iqr_rel"] = (cy["cost_85pct"] - cy["cost_15pct"]) / cy["weight_avg"].replace(0, np.nan)
    cy["high_dist"] = cy["his_high"] / cy["cost_50pct"].replace(0, np.nan)
    out["cyq_perf"] = cy.loc[:, ["signal_date", "ts_code", "winner_rate", "cost_dispersion",
                                 "cost_iqr_rel", "high_dist", "cost_50pct"]]

    mg = pd.concat([pd.read_parquet(RAW / f"margin_detail_{y}.parquet") for y in YEARS], ignore_index=True)
    mg["signal_date"] = pd.to_datetime(mg["trade_date"], format="%Y%m%d")
    mg = mg.sort_values(["ts_code", "signal_date"])
    mg["buy_intensity"] = mg["rzmre"] / mg["rzye"].replace(0, np.nan)
    mg["repay_intensity"] = mg["rzche"] / mg["rzye"].replace(0, np.nan)
    mg["net_fin_ratio"] = (mg["rzmre"] - mg["rzche"]) / mg["rzye"].replace(0, np.nan)
    mg["short_to_long"] = mg["rqye"] / mg["rzye"].replace(0, np.nan)
    mg["rzye_chg5"] = mg.groupby("ts_code")["rzye"].transform(lambda s: s.pct_change(5))
    out["margin_detail"] = mg.loc[:, ["signal_date", "ts_code", "buy_intensity", "repay_intensity",
                                      "net_fin_ratio", "short_to_long", "rzye_chg5", "rzye"]]
    return out


def published_versions(session, object_name: str) -> dict[str, str]:
    """Mirror of research_v2 walk_forward_common.current_versions.

    ``published_version`` may hold several rows per logical partition; production
    sorts by ``published_at`` and lets the last one win, so we do the same.
    """
    frame = session.run(
        f"select logical_partition, data_version, published_at "
        f"from loadTable('{DB}','published_version') where object_name='{object_name}'"
    )
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        return {}
    frame = frame.sort_values("published_at", kind="mergesort")
    return {str(row.logical_partition): str(row.data_version) for row in frame.itertuples(index=False)}


def filter_current(frame: pd.DataFrame, date_column: str, versions: dict[str, str]) -> pd.DataFrame:
    """Mirror of research_v2 walk_forward_common.filter_current.

    Both ``universe_daily`` and ``label_daily`` are versioned per YYYY-MM logical
    partition, so an unfiltered read returns several historical revisions of the
    same (signal_date, ts_code) business key.  Keeping only the published
    ``data_version`` is what every walk-forward script does; without it the
    cross-join inflates and ``compute_daily_rank_ic`` rejects duplicate keys.
    """
    partitions = pd.to_datetime(frame[date_column]).dt.strftime("%Y-%m")
    expected = partitions.map(versions)
    return frame.loc[expected.notna() & (frame["data_version"] == expected)].copy()


def load_context():
    s = ddb.session()
    s.connect("10.176.55.211", 8848, "xiahongjie", "123456")
    uni = s.run(f"select signal_date, ts_code, data_version from loadTable('{DB}','universe_daily') "
                f"where universe_id='cn_liquid_a' and universe_version=1 and is_member=true "
                f"and year(signal_date) between 2022 and 2025")
    lab = s.run(f"select signal_date, ts_code, raw_return, label_valid, data_version "
                f"from loadTable('{DB}','label_daily') "
                f"where label_id='forward_open_5d' and label_version=1 and year(signal_date) between 2022 and 2025")
    versions = {name: published_versions(s, name) for name in ("universe_daily", "label_daily")}
    s.close()
    for df in (uni, lab):
        df["signal_date"] = pd.to_datetime(df["signal_date"])
    raw_uni, raw_lab = len(uni), len(lab)
    uni = filter_current(uni, "signal_date", versions["universe_daily"]).drop(columns=["data_version"])
    lab = filter_current(lab, "signal_date", versions["label_daily"]).drop(columns=["data_version"])
    print(f"version filter: universe_daily {raw_uni:,} -> {len(uni):,} rows | "
          f"label_daily {raw_lab:,} -> {len(lab):,} rows", flush=True)
    for name, df in (("universe_daily", uni), ("label_daily", lab)):
        duplicates = int(df.duplicated(KEYS).sum())
        if duplicates:
            raise RuntimeError(
                f"{name} retains {duplicates} duplicated (signal_date, ts_code) keys after the "
                f"published-version filter; refusing to screen on an ambiguous context"
            )
        if df.empty:
            raise RuntimeError(f"{name} is empty after the published-version filter")
    return uni, lab


def screen(frame: pd.DataFrame, cand: str, ctx: pd.DataFrame, lag: int) -> dict:
    src = frame.loc[:, KEYS + [cand]].dropna(subset=[cand]).copy()
    if lag:
        # shift by `lag` TRADING days using the labelled universe calendar (not calendar days)
        cal = np.sort(ctx["signal_date"].unique())
        pos = np.searchsorted(cal, src["signal_date"].to_numpy(), side="left")
        tgt = pos + lag
        keep = (tgt < len(cal)) & (pos < len(cal)) & (cal[np.minimum(pos, len(cal) - 1)] == src["signal_date"].to_numpy())
        src = src.loc[keep]
        src["signal_date"] = cal[tgt[keep]]
    merged = ctx.merge(src, on=KEYS, how="inner")
    if merged.empty:
        return {"candidate": cand, "lag": lag, "rows": 0}
    d = compute_daily_rank_ic(merged.loc[:, KEYS + [cand]].rename(columns={cand: "score"}),
                              merged.loc[:, KEYS + ["raw_return", "label_valid"]].rename(columns={"raw_return": "raw_return_5d"}),
                              minimum_cross_section=MIN_CS, include_invalid=True, require_score_health=False)
    valid = d.loc[d["rank_ic_valid"] & d["rank_ic"].notna()]
    if len(valid) < 20:
        return {"candidate": cand, "lag": lag, "rows": int(len(merged)), "valid_days": int(len(valid))}
    ic = valid["rank_ic"].to_numpy(dtype=float)
    yearly = {str(y): {"mean_rank_ic": float(g["rank_ic"].mean()),
                       "icir": float(g["rank_ic"].mean() / g["rank_ic"].std(ddof=1)) if g["rank_ic"].std(ddof=1) > 0 else None,
                       "days": int(len(g))}
              for y, g in valid.assign(y=pd.to_datetime(valid["signal_date"]).dt.year).groupby("y")}
    return {"candidate": cand, "lag": lag, "rows": int(len(merged)), "valid_days": int(len(valid)),
            "coverage_of_universe_member_days": float(len(merged) / len(ctx)),
            "mean_rank_ic": float(ic.mean()), "rank_ic_std": float(ic.std(ddof=1)),
            "icir": float(ic.mean() / ic.std(ddof=1)),
            "direction": int(np.sign(ic.mean())) if ic.mean() != 0 else 0,
            "directional_day_share": float((np.sign(ic.mean()) * ic > 0).mean()),
            "yearly": yearly}


def main() -> int:
    uni, lab = load_context()
    ctx = uni.merge(lab, on=KEYS, how="inner")
    print(f"context: universe member-days={len(uni):,} labelled={len(ctx):,}")
    cands = build_candidates()
    results = {"scope": "DAILY_DOMAIN_IC_SCREEN_V1", "window": "2022-2025", "label": "forward_open_5d:v1",
               "universe": "cn_liquid_a", "icir_convention": "mean/std of daily RankIC, NON-annualised",
               "context_rows": int(len(ctx)), "sources": {}}
    for source, frame in cands.items():
        cols = [c for c in frame.columns if c not in KEYS]
        per = []
        for cand in cols:
            for lag in (0, 1):
                r = screen(frame, cand, ctx, lag)
                per.append(r)
                if r.get("icir") is not None:
                    print(f"  {source:14s} {cand:18s} lag{lag} ic={r['mean_rank_ic']:+.4f} icir={r['icir']:+.3f} "
                          f"days={r['valid_days']} cov={r.get('coverage_of_universe_member_days',0):.2f}", flush=True)
        results["sources"][source] = per
    (OUT / "analysis" / "ic_screen.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))

    rows = [r for src in results["sources"].values() for r in src if r.get("icir") is not None]
    df = pd.DataFrame(rows).sort_values("icir", key=lambda s: s.abs(), ascending=False)
    df.to_csv(OUT / "analysis" / "ic_screen_summary.csv", index=False)
    print("\n=== |ICIR| 排名前 20 ===")
    print(df.head(20)[["candidate", "lag", "mean_rank_ic", "icir", "valid_days",
                       "coverage_of_universe_member_days", "directional_day_share"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
