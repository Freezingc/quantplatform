"""Read-only redundancy re-check of yeslab_393 (GP_024) against the v2 frozen 15.

Why: 393 was rejected ONLY for "full-day similarity to formal Alpha376 = 0.9447", and 376 is
not part of the v2 training feature space. Its annual evidence is 5/5 on every strict gate
(signed IC, directional share, |ICIR| >= 0.50). So the rejection reason may be void under v2.

Data:
  - 393 (+376 as a methodology control) 2025 values from the sanctioned similarity panel
    research_v2/artifacts/diagnostics/gp_candidate_similarity_2025_values.parquet
    (YesLab normalization / dedup / PIT alignment, written by materialize_factor_values_similarity.py)
  - the v2 fifteen 2025 values from factor_daily (isolated canonical materialization)
  - labels: label_daily forward_open_5d:v1 (label_valid) for 2025

CAVEAT (recorded, not hidden): this is a TRIAGE-grade comparison because 393 comes from the
source panel while the fifteen come from the canonical store. Promotion would still require a
canonical replay definition for 393 (it has none today) and isolated materialization.

Read-only: no DB writes, no artifact-store writes, no 2026/E9 reads.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pyarrow.dataset as pad
import dolphindb as ddb

OUT = pathlib.Path(__file__).resolve().parent
PANEL = pathlib.Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/diagnostics/gp_candidate_similarity_2025_values.parquet")
DB = "dfs://QuantResearchV2Test"
V2 = json.loads(pathlib.Path("/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2/artifacts/diagnostics/factor_set_frozen_v2.json").read_text())["factors"]
KEYS = ["signal_date", "ts_code"]
CONTROL = "yeslab_376"          # documented similarity to 393: 0.9447022272
DOC_SIM_376 = 0.9447022272
MIN_CS = 30


def load_panel(fid: str) -> pd.DataFrame:
    d = pad.dataset(str(PANEL), format="parquet")
    t = d.to_table(filter=pad.field("factor_id") == fid, columns=["signal_date", "ts_code", "value"])
    df = t.to_pandas()
    df["signal_date"] = pd.to_datetime(df["signal_date"])
    return df


def main() -> int:
    s = ddb.session()
    s.connect("10.176.55.211", 8848, "xiahongjie", "123456")

    a393 = load_panel("yeslab_393").rename(columns={"value": "a393"})
    a376 = load_panel(CONTROL).rename(columns={"value": "a376"})
    print(f"panel: 393 rows={len(a393):,} days={a393['signal_date'].nunique()} | 376 rows={len(a376):,}")

    lab = s.run(f"select signal_date, ts_code, raw_return from loadTable('{DB}','label_daily') "
                f"where label_id='forward_open_5d' and label_version=1 and label_valid=true "
                f"and signal_date between 2025.01.01 : 2025.12.31").rename(columns={"raw_return": "label"})
    lab["signal_date"] = pd.to_datetime(lab["signal_date"])
    print(f"labels 2025: {len(lab):,}")

    v2_frames = {}
    for fid in V2:
        q = (f"select signal_date, ts_code, value from loadTable('{DB}','factor_daily') "
             f"where factor_id='{fid}' and signal_date between 2025.01.01 : 2025.12.31")
        df = s.run(q)
        df["signal_date"] = pd.to_datetime(df["signal_date"])
        v2_frames[fid] = df.rename(columns={"value": fid})
        print(f"  factor_daily {fid}: {len(df):,} rows", flush=True)
    s.close()

    base = a393.merge(a376, on=KEYS, how="inner").merge(lab, on=KEYS, how="left")
    results = {"scope": "ALPHA393_REDUNDANCY_RECHECK_V1", "window": "2025-01-02..2025-12-31",
               "panel_rows_393": int(len(a393)), "panel_days": int(a393["signal_date"].nunique()),
               "labels_rows": int(len(lab)), "pairs": {}}

    def daily_stats(frame: pd.DataFrame, col: str) -> dict:
        w = frame.dropna(subset=["a393", col])
        w = w[w.groupby("signal_date")["a393"].transform("size") >= MIN_CS]
        ra = w.groupby("signal_date")["a393"].rank(pct=True)
        rb = w.groupby("signal_date")[col].rank(pct=True)
        tmp = w.assign(ra=ra.to_numpy(), rb=rb.to_numpy())
        g = tmp.groupby("signal_date").apply(
            lambda x: np.corrcoef(x["ra"], x["rb"])[0, 1] if len(x) >= MIN_CS else np.nan,
            include_groups=False)
        rho = g.dropna()
        ar = rho.abs()
        return {"days": int(len(rho)), "mean_rho": float(rho.mean()), "median_abs_rho": float(ar.median()),
                "p90_abs_rho": float(ar.quantile(0.90)), "share_ge_080": float((ar >= 0.80).mean()),
                "share_ge_090": float((ar >= 0.90).mean()), "max_abs_rho": float(ar.max()),
                "rows": int(len(w))}

    # control: 393 vs 376 must reproduce the documented 0.9447
    results["control_393_vs_376"] = daily_stats(base, "a376")
    results["control_393_vs_376"]["documented_median_abs_spearman"] = DOC_SIM_376
    results["control_393_vs_376"]["reproduces_document"] = bool(
        abs(results["control_393_vs_376"]["median_abs_rho"] - DOC_SIM_376) < 0.02)

    for fid in V2:
        m = base.merge(v2_frames[fid], on=KEYS, how="inner")
        results["pairs"][fid] = daily_stats(m, fid)

    med = {k: v["median_abs_rho"] for k, v in results["pairs"].items()}
    results["worst_partner"] = max(med, key=med.get)
    results["worst_median_abs_rho"] = max(med.values())
    results["thresholds"] = {"block_joint_admission": 0.90, "requires_monitoring": 0.80}
    results["verdict"] = ("BLOCK" if results["worst_median_abs_rho"] >= 0.90 else
                          "MONITOR" if results["worst_median_abs_rho"] >= 0.80 else "PASS")

    # incremental information: 393 residualised on the fifteen incumbents (per-day OLS on ranks)
    cols = KEYS + ["a393", "label"] + list(V2)
    wide = base.rename(columns={})
    for fid in V2:
        wide = wide.merge(v2_frames[fid], on=KEYS, how="left")
    wide = wide.loc[:, cols]
    sub = wide.dropna(subset=["a393", "label"])
    days, raw_ic, res_ic, cov = [], [], [], []
    for d, g in sub.groupby("signal_date"):
        X = g[list(V2)].to_numpy(dtype=float)
        finite = np.isfinite(X).all(axis=1)
        gg = g.loc[finite]
        if len(gg) < MIN_CS:
            continue
        Xg = gg[list(V2)].to_numpy(dtype=float)
        y = gg["a393"].to_numpy(dtype=float)
        r = gg["label"].to_numpy(dtype=float)
        rx = np.apply_along_axis(lambda c: pd.Series(c).rank(pct=True).to_numpy(), 0, Xg)
        ry = pd.Series(y).rank(pct=True).to_numpy()
        rr = pd.Series(r).rank(pct=True).to_numpy()
        A = np.column_stack([np.ones(len(rx)), rx])
        beta, *_ = np.linalg.lstsq(A, ry, rcond=None)
        resid = ry - A @ beta
        if np.std(resid) < 1e-12 or np.std(rr) < 1e-12:
            continue
        days.append(str(d.date()))
        raw_ic.append(float(np.corrcoef(ry, rr)[0, 1]))
        res_ic.append(float(np.corrcoef(resid, rr)[0, 1]))
        cov.append(float(finite.mean()))
    results["incremental"] = {
        "days": len(days),
        "mean_raw_rank_ic": float(np.mean(raw_ic)) if raw_ic else None,
        "mean_residual_rank_ic_given_v2_15": float(np.mean(res_ic)) if res_ic else None,
        "retention": float(np.mean(res_ic) / np.mean(raw_ic)) if raw_ic and abs(np.mean(raw_ic)) > 1e-9 else None,
        "mean_all15_finite_coverage": float(np.mean(cov)) if cov else None,
        "note": "residual = per-day OLS of 393's percentile rank on the fifteen incumbents' percentile ranks",
        "daily": {"signal_date": days, "raw_ic": raw_ic, "residual_ic": res_ic, "coverage": cov},
    }
    (OUT / "alpha393_redundancy_recheck.json").write_text(json.dumps(results, ensure_ascii=False, indent=1))

    L = ["# yeslab_393 (GP_024) 对 v2 十五因子冗余复检（2025 窗口，只读）", ""]
    c = results["control_393_vs_376"]
    L.append(f"**方法对照**：393 vs 376 median|ρ| = {c['median_abs_rho']:.4f}，文档记载 {DOC_SIM_376:.4f} → "
             f"{'复现一致' if c['reproduces_document'] else '不一致'}")
    L.append("")
    L.append(f"**判定：{results['verdict']}**（最相似对手 {results['worst_partner']}，median|ρ| = {results['worst_median_abs_rho']:.4f}；"
             f"阈值：≥0.90 阻断联合准入，≥0.80 需监控）")
    L.append("")
    L.append("| v2 因子 | 交集行数 | 天数 | mean ρ | median|ρ| | p90|ρ| | ≥0.80 占比 | ≥0.90 占比 |")
    L.append("|---|---|---|---|---|---|---|---|")
    for fid, v in sorted(results["pairs"].items(), key=lambda kv: -kv[1]["median_abs_rho"]):
        L.append(f"| {fid} | {v['rows']:,} | {v['days']} | {v['mean_rho']:+.3f} | {v['median_abs_rho']:.4f} | "
                 f"{v['p90_abs_rho']:.4f} | {v['share_ge_080']:.1%} | {v['share_ge_090']:.1%} |")
    inc = results["incremental"]
    L.append("")
    L.append(f"**增量信息**：393 原始日 RankIC 均值 {inc['mean_raw_rank_ic']:+.5f}；对 v2 十五因子逐日投影后的残差 RankIC 均值 "
             f"{inc['mean_residual_rank_ic_given_v2_15']:+.5f}（保留率 {inc['retention']:.3f}，{inc['days']} 天，"
             f"十五因子全有限覆盖 {inc['mean_all15_finite_coverage']:.1%}）")
    L.append("")
    L.append("**口径说明（不可忽略）**：393 取自 YesLab 规范化相似度面板（2025），十五因子取自 factor_daily 隔离规范物化；"
             "两者来源不同，故本结果是**分诊(triage)级**证据。若判定为可准入，仍需先为 393 建立规范重放定义（目前没有）并做隔离物化，"
             "再在隔离口径上复算一次冗余。")
    (OUT / "alpha393_redundancy_recheck.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
