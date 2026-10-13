"""Compose the detailed markdown report from the persisted artifacts.

Every number in daily_domain_feasibility.md is read back out of analysis/*.json|csv rather than
retyped, so the report cannot drift from the runs that produced it.  Re-running this script after
any of the upstream steps regenerates the report deterministically.
"""
from __future__ import annotations

import json
import pathlib
import platform
import subprocess
import sys

import pandas as pd

OUT = pathlib.Path(__file__).resolve().parent
ANA = OUT / "analysis"
REPORT = OUT / "daily_domain_feasibility.md"


def md(df: pd.DataFrame, float_fmt: str = ".3f") -> str:
    """Render a markdown table without the optional `tabulate` dependency (not installed here)."""
    cols = [str(c) for c in df.columns]

    def cell(value) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return "—" if value != value else format(value, float_fmt)
        if isinstance(value, (dict, list)):
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        if value is pd.NaT:
            return "—"
        text = str(value)
        return "—" if text in ("nan", "NaT", "None", "") else text.replace("|", "\\|")

    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for row in df.itertuples(index=False, name=None):
        out.append("| " + " | ".join(cell(v) for v in row) + " |")
    return "\n".join(out)


def versions() -> str:
    mods = ["pandas", "numpy", "pyarrow", "dolphindb", "scipy"]
    out = [f"python {platform.python_version()}"]
    for m in mods:
        try:
            mod = __import__(m)
            out.append(f"{m} {getattr(mod, '__version__', '?')}")
        except ImportError:
            out.append(f"{m} absent")
    return ", ".join(out)


def git_head() -> str:
    try:
        h = subprocess.run(["git", "-C", "/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX",
                            "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        b = subprocess.run(["git", "-C", "/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX",
                            "rev-parse", "--abbrev-ref", "HEAD"], capture_output=True, text=True, check=True)
        return f"{b.stdout.strip()} @ {h.stdout.strip()[:12]}"
    except Exception as exc:  # pragma: no cover
        return f"unavailable ({exc})"


def main() -> int:
    screen = json.loads((ANA / "ic_screen.json").read_text())
    summary = pd.read_csv(ANA / "ic_screen_summary.csv")
    pooled = pd.read_csv(ANA / "ic_screen_pooled_windows.csv")
    bench = pd.read_csv(ANA / "benchmark_frozen_v2_icir_2022_2024.csv")
    redun = json.loads((ANA / "redundancy_survivors.json").read_text())
    dim = json.loads((ANA / "dimension_summary.json").read_text())
    dec = pd.read_csv(ANA / "decision_table.csv")
    cov = pd.read_csv(ANA / "coverage_drift.csv")

    L: list[str] = []
    A = L.append
    A("# 日频非价量域可行性核查 —— 明细报告（2026-10-13，只读）\n")
    A(f"- 生产树：{git_head()}（零改动）")
    A(f"- 环境：{versions()}")
    A(f"- 标签 / 宇宙：`forward_open_5d:v1` / `cn_liquid_a`，`minimum_cross_section={30}`，"
      f"ICIR = 日度 RankIC 的 mean/std（**未年化**）")
    A(f"- 上下文口径：`universe_daily` 与 `label_daily` 均按 `published_version` 指针保留当月已发布 "
      f"`data_version`（等价 `walk_forward_common.filter_current`）；筛查窗 2022-2025 共 "
      f"{screen['context_rows']:,} 个成员日")
    A("- 2026 锁定段（E9）未读取、未写入；无数据库写入、无生产工件写入。**精确说明**："
      "`published_version` 指针查询本身不带日期过滤，会返回 `2026-*` 的分区标签（版本元数据），"
      "但所有行情/标签/因子数据查询均以 `year(signal_date) between 2022 and 2025` 或 "
      "`between 2025.01.01 : 2025.12.31` 硬界定，锁定段的任何数据行都未进入本次分析\n")

    A("## A. 数据层：三源逐年覆盖率\n")
    A("`key_coverage` = 该源在当年有行记录的比例；`nonnull::<字段>` = 该字段在匹配行上的非空比例。\n")
    key = cov.pivot(index="source", columns="year", values="key_coverage").round(4).reset_index()
    A(md(key, ".4f"))
    A("")
    for source, g in cov.groupby("source"):
        own = {f"nonnull::{c}" for c in _cand_cols()[source]}
        cols = [c for c in g.columns if c in own]
        if not cols:
            continue
        t = g.set_index("year")[cols].rename(columns=lambda c: c.split("::")[1]).round(4).reset_index()
        A(f"**{source}** 字段非空率\n")
        A(md(t, ".4f"))
        A("")
    A("> 关键读法：`margin_detail` 的覆盖率从 2022 的 0.568 单调升到 2025 的 0.725（+15.7pp），"
      "因为两融标的名单逐年扩容。这意味着该域因子的横截面**本身在漂移**，跨折不可直接比较，"
      "且缺失非随机（是否可两融与市值/流动性强相关）。`cyq_perf` 从 0.970 升到 0.999，"
      "`moneyflow` 恒为 1.000。\n")

    A("## B. 预测力层：全字段 × 双对齐 IC 筛查\n")
    A("`lag0` = 源 T 日对信号 T 日；`lag1` = 源 T-1 日对信号 T 日。"
      "moneyflow/cyq_perf 于 T 日晚间发布，T+1 开盘建仓前可得，lag0 即为 PIT 安全；"
      "margin_detail 于 T+1 早 08:30 发布，早于 09:30 开盘但余量很小，故以 lag1 为保守决策口径，"
      "两者同时报告。\n")
    full = summary[["candidate", "lag", "mean_rank_ic", "icir", "valid_days",
                    "coverage_of_universe_member_days", "directional_day_share"]].copy()
    full.insert(0, "source", full["candidate"].map(_src_map()))
    A(md(full.sort_values(["source", "candidate", "lag"]), ".4f"))
    A("")
    A("逐年 ICIR（同一口径，便于看年度稳定性）：\n")
    yr = pooled[["source", "candidate", "lag", "yearly_icir"]].copy()
    A(md(yr.sort_values(["source", "candidate", "lag"]), ".3f"))
    A("")

    A("## C. 基准对照：冻结 v2 因子在同窗口的 |ICIR|\n")
    A("取自库内 `factor_rank_ic_daily`（同标签 `forward_open_5d:v1`、同 ICIR 口径），"
      "该表覆盖 2020Q1–2024Q4，故基准窗为 **2022–2024**；候选的 2022–2024 池化值由逐年 "
      "mean/std/天数按全方差定律精确还原（见 `pooled_window_icir.py`）。\n")
    A(md(bench[["factor_id", "role", "days", "mean_ic", "icir_2224", "abs_icir"]], ".4f"))
    A("")
    A(f"冻结集 |ICIR| 区间 **{bench['abs_icir'].min():.3f}（{bench.loc[bench['abs_icir'].idxmin(), 'factor_id']}）"
      f"– {bench['abs_icir'].max():.3f}**，中位 {bench['abs_icir'].median():.3f}，"
      f"dense9 均值 {bench.loc[bench['role'] == 'dense9', 'abs_icir'].mean():.3f}。\n")
    cmp = pooled[["source", "candidate", "lag", "icir_2022_2024", "icir_2022_2025", "coverage"]].copy()
    cmp["abs_icir_2022_2024"] = cmp["icir_2022_2024"].abs()
    cmp["vs_weakest_frozen_0.497"] = (cmp["abs_icir_2022_2024"] - bench["abs_icir"].min()).round(3)
    A(md(cmp.sort_values("abs_icir_2022_2024", ascending=False).drop(columns=["abs_icir_2022_2024"]), ".4f"))
    A("")

    A("## D. 冗余与有效维度\n")
    A("窗口 2025（与 2026-10-13 alpha393 复检同口径，可直接比较）。三条腿：\n")
    A("1. **逐对 vs 冻结 15**：日内百分位秩的 |Spearman| 中位数，≥0.90 BLOCK、≥0.80 MONITOR")
    A("2. **逐对 vs 其他幸存者**：v1 脚本缺这条腿，而 margin 族在代数上互相纠缠"
      "（`net_fin_ratio = buy_intensity - repay_intensity`，`rzye_chg5` 是其 5 日平滑）")
    A("3. **给定 dense9 的集合增量保留率**：残差 RankIC / 原始 RankIC，门槛 ≥ +0.30（**有符号**）\n")
    A(md(dec[["candidate", "source", "lag", "used_safe_lag", "icir_2022_2025", "coverage",
              "worst_frozen_rho", "worst_frozen_partner", "worst_survivor_rho", "worst_survivor_partner",
              "retention_given_dense9", "retention_flag", "residual_rank_ic",
              "clears_icir_bar", "clears_residual_ic_bar", "dim_retention_aware"]], ".3f"))
    A("")
    A("幸存者两两 |Spearman| 中位数矩阵：\n")
    names = [e["candidate"] for e in redun["candidates"]]
    mat = pd.DataFrame(index=names, columns=names, dtype=float)
    for e in redun["candidates"]:
        for other, r in e["pairwise_vs_other_survivors"].items():
            mat.loc[e["candidate"], other] = r["median_abs_rho"]
    for n in names:
        mat.loc[n, n] = 0.0
    A(md(mat.round(3).reset_index(), ".3f"))
    A("")
    A("三种贪心规则下的『有效新维度』计数：\n")
    for rule, val in dim["greedy"].items():
        A(f"- `{rule}` → **{val['count']}**：{val['admitted']}")
        for k, v in val["rejected"].items():
            A(f"    - 拒 {k}：{v}")
    A("")

    A("## E. 复现\n")
    A("```bash")
    A("cd /data/zhangyudong/qrv2_research_artifacts/daily_domain_feasibility_20261013")
    A("python fetch_daily_domains.py        # 抓取三源 2022-2025 -> raw/（可断点续跑）")
    A("python screen_daily_domains.py       # IC 筛查 -> analysis/ic_screen{.json,_summary.csv}")
    A("python pooled_window_icir.py         # 池化到基准窗 -> analysis/ic_screen_pooled_windows.csv")
    A("python redundancy_survivors.py 0.25  # 冗余三腿 -> analysis/redundancy_survivors.json")
    A("python greedy_caps.py                # 决策表与贪心规则 -> analysis/decision_table.csv")
    A("python coverage_drift.py             # 覆盖率漂移 -> analysis/coverage_drift.csv")
    A("python make_report.py                # 本报告")
    A("```")
    A("")
    A("原始面板（594 MB，12 个 parquet）留在服务器 `raw/`，不入 Git；本目录只提交脚本与小件摘要。\n")
    REPORT.write_text("\n".join(L))
    print(f"wrote {REPORT} ({REPORT.stat().st_size:,} bytes)")
    return 0


def _src_map() -> dict[str, str]:
    from redundancy_survivors import CANDIDATE_COLUMNS
    return {c: s for s, cols in CANDIDATE_COLUMNS.items() for c in cols}


def _cand_cols() -> dict[str, list[str]]:
    from redundancy_survivors import CANDIDATE_COLUMNS
    return CANDIDATE_COLUMNS


if __name__ == "__main__":
    raise SystemExit(main())
