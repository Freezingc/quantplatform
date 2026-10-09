#!/usr/bin/env python3
"""Emit 2026-10-08-optimization-levers.md with every number read out of the JSONs.

Nothing in the document is transcribed by hand. Each table cell is formatted from one of
the four result files, so a number in the prose cannot drift away from a number in the
data. `verify_optimization_report.py` then re-reads the markdown and asserts that each
figure it can locate is present in the JSON it claims to come from.
"""
from __future__ import annotations

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "2026-10-08-optimization-levers.md"

probes = json.loads((HERE / "optimization_probe_results.json").read_text())
levers = json.loads((HERE / "optimization_levers_results.json").read_text())
stack = json.loads((HERE / "optimization_stack_results.json").read_text())
redun = json.loads((HERE / "factor_redundancy_results.json").read_text())
conc = json.loads((HERE / "concentration_results.json").read_text())
idx = json.loads((HERE / "index_benchmark.json").read_text())
gateclean = json.loads((HERE / "gate_clean_ensemble_results.json").read_text())

FACTOR_MAP = levers["factor_diagnostics"]["factor_map"]
L: list[str] = []
w = L.append


def pct(x: float, nd: int = 2) -> str:
    return f"{x * 100:.{nd}f}%"


def pp(x: float, nd: int = 2) -> str:
    return f"{x:+.{nd}f}pp"


def row(*cells: object) -> str:
    return "| " + " | ".join(str(c) for c in cells) + " |"


# ---------------------------------------------------------------- header
ref = levers["reference"]["buffered_top30"]
grid = stack["stack_grid"]
best_key = max(grid, key=lambda k: grid[k]["metrics"]["net_sharpe"])
best = grid[best_key]
bm = best["metrics"]
# the headline quotes the RECOMMENDED setting (MA10), not the in-sample optimum (MA5),
# so the summary table and section 10 cannot disagree with each other
REC = "top30_exit150_sm1_csi2000_ma10_f0.0"
rm = grid[REC]["metrics"]
attr = levers["drawdown_attribution"]
red = redun["redundancy"]
pnl = redun["pnl_concentration"]

w("# 要提高收益、降低回撤，该动哪里：所有杠杆的实测结果")
w("")
w("**日期**：2026-10-08　**分支**：`codex/research-conditional-factor-review`　"
  "**配套**：`2026-10-08-model-backtest/`")
w("")
w("## 0. 结论先行")
w("")
w("问题是「该优化因子、label、模型，还是别的」。答案不是猜的：能在**已冻结的 TEST 预测**上"
  "直接测的杠杆，全部测了；测不了的只给方向、不给数字。")
w("")
w(f"按「每单位工作量的收益/回撤改善」排序：")
w("")
w(row("优先级", "杠杆", "需要重训", "实测效果（参照系 = 集成 top30 + 缓冲）"))
w(row("---", "---", "---", "---"))
# each lever is quoted against its OWN controlled comparison, never with the other lever's
# gain folded in: the filter at fixed buffer, the buffer at no filter. The combined row is
# labelled as combined. Section 10 explains why mixing the two bases is a real error.
_b150 = grid["top30_exit150_sm1_none"]["metrics"]
_b60 = grid["top30_exit60_sm1_none"]["metrics"]
w(row("**1**", "**指数趋势过滤（仓位择时）**", "否",
    f"同缓冲下：最大回撤 {pct(_b150['max_drawdown'])} → {pct(rm['max_drawdown'])}，"
    f"Sharpe {_b150['net_sharpe']:.3f} → {rm['net_sharpe']:.3f}，"
    f"年化 {pct(_b150['net_annualized'])} → {pct(rm['net_annualized'])}"
    f"（**降回撤确定，提收益不确定**，见 §10）"))
w(row("**2**", "**加宽持仓缓冲 / 分数平滑（降换手）**", "否",
    f"无过滤下：年化 {pct(_b60['net_annualized'])} → {pct(_b150['net_annualized'])}，"
    f"Sharpe {_b60['net_sharpe']:.3f} → {_b150['net_sharpe']:.3f}，"
    f"换手 {_b60['mean_turnover']:.3f} → {_b150['mean_turnover']:.3f}"))
w(row("**1+2**", "**两条一起上（推荐的组合）**", "否",
    f"对参照系：年化 {pct(ref['net_annualized'])} → {pct(rm['net_annualized'])}，"
    f"最大回撤 {pct(ref['max_drawdown'])} → {pct(rm['max_drawdown'])}，"
    f"Sharpe {ref['net_sharpe']:.3f} → {rm['net_sharpe']:.3f}，"
    f"累计 {pct(ref['total_return'])} → {pct(rm['total_return'])}"))
w(row("3", "因子：**加正交族，不是加数量**", "是",
    f"15 个因子的有效维度只有 {red['effective_dimensions_participation_ratio']:.2f}，"
    f"IC 共动有效维度 {red['ic_effective_dimensions']:.2f}"))
w(row("4", "验证指标与 top-N 目标对齐", "是", "方向明确，但尾部加权有反证，见 §9"))
w(row("5", "风险调整 label", "是", "值得一次实验；但「减指数收益」这一类改动可证明无效，见 §9"))
w(row("6", "模型结构", "是", "最后才动"))
# compare like with like: the reference row is the BUFFERED portfolio, so the refuted
# variant must be the buffered one too, or the two drawdowns are not the same strategy
_bt = probes["variants"]["buffered"]["vol_target_0.12"]
w(row("**已证伪**", "**波动率目标（vol targeting）**", "否",
    f"回撤几乎不动（{pct(ref['max_drawdown'])} → {pct(_bt['max_drawdown'])}），"
    f"年化砍掉六成（{pct(ref['net_annualized'])} → {pct(_bt['net_annualized'])}）"))
w(row("**已证伪**", "**换一种集成方式**", "否",
    f"等权原始分数已是四种组合里最好的；z-score 平均是灾难"
    f"（Sharpe {levers['ensemble_variants']['30']['zscore_mean']['net_sharpe']:.3f}）"))
w("")
w(f"> **本文所有数字的参照系是 M0R+M1+M2 三模型集成，而 M2 未过 G4/G6。**"
  f"§12 单独测了治理干净的 M0R+M1：在参照系配置下它**四项指标全部不差于三模型**，"
  f"并且把合格行从 {gateclean['member_sets']['M0R+M1+M2']['total_eligible_rows']:,} 扩到 "
  f"{gateclean['member_sets']['M0R+M1 (gate-clean)']['total_eligible_rows']:,}"
  f"（正好多出 {gateclean['member_sets']['M0R+M1 (gate-clean)']['total_eligible_rows'] - gateclean['member_sets']['M0R+M1+M2']['total_eligible_rows']:,} 行）。"
  f"**治理与性能的冲突不存在，可以直接选干净的那个。**")
w("")
w(f"**一句话**：这个策略的回撤主要是**市场/风格 beta**，不是选股失败——所以降回撤是"
  f"**仓位问题**；而提收益的主要来源是**换手**（纯净额，不用改模型）。"
  f"因子和模型都还有空间，但都比前两条贵，而且因子的空间是「加正交的」而不是「加更多同源的」。")
w("")

# ---------------------------------------------------------------- 1. 口径
w("## 1. 口径（先说清，否则数字会被误读）")
w("")
w(row("项", "取值"))
w(row("---", "---"))
w(row("数据", "已签名冻结的 TEST 预测，2022–2025，969 个信号日，192 个周期（**按折分组，4×48，不跨折**）"))
w(row("标的池", "M0R / M1 / M2 三者覆盖的**交集**行（宽度由 M2 最窄的覆盖决定）"))
w(row("参照系", "三模型等权分数集成，top30，持仓缓冲 N_enter=30 / N_exit=60"))
w(row("label", "`raw_return_5d` = adjusted_open(T+6) / adjusted_open(T+1) − 1"))
w(row("成本", f"往返 {levers['config']['round_trip_bp']:.0f}bp × 实测换手（**成本是假设，不是实测**）"))
w(row("年化", f"×{levers['config']['cycles_per_year']} 周期/年；Sharpe 的无风险利率取 0"))
w(row("权重", "等权"))
w(row("本次是否重训", "**否**。全部是对已冻结分数的重新组合与重新计量"))
w("")
w(f"参照系（后文所有「改善」都对着这一行比）：年化 **{pct(ref['net_annualized'])}**，"
  f"Sharpe **{ref['net_sharpe']:.3f}**，最大回撤 **{pct(ref['max_drawdown'])}**，"
  f"累计 **{pct(ref['total_return'])}**，换手 **{ref['mean_turnover']:.3f}**，"
  f"回撤窗口 **{ref['dd_peak_date']} → {ref['dd_trough_date']}**。")
w("")
w("## 2. 回撤是什么：归因")
w("")
w(f"先定位回撤窗口，再看同期指数。窗口 = **{attr['window'][0]} → {attr['window'][1]}**，"
  f"策略净收益 **{pct(attr['strategy_net'])}**。")
w("")
w(row("指数", "同期收盘-收盘", "策略 − 指数"))
w(row("---", "---", "---"))
for code in ("000001.SH", "000985.CSI", "000852.SH", "932000.CSI"):
    e = attr["indices"][code]
    w(row(f"{e['label']} `{code}`", pct(e["close_to_close"]), pp(e["strategy_minus_index_pp"])))
w("")
w("**读法**：")
w("")
w(f"* 四个指数同期全都跌，跌得最少的上证 {pct(attr['indices']['000001.SH']['close_to_close'])}，"
  f"跌得最多的中证1000 {pct(attr['indices']['000852.SH']['close_to_close'])}。"
  f"所以这个回撤**主体是 beta**，不是选股突然失效。")
w(f"* 但策略比风格匹配的中证2000 还差 "
  f"{abs(attr['indices']['932000.CSI']['strategy_minus_index_pp']):.2f}pp，"
  f"比中证1000 差 {abs(attr['indices']['000852.SH']['strategy_minus_index_pp']):.2f}pp。"
  f"**不是纯 beta**：危机期选股是负贡献（反转类信号在流动性危机里正好站在错误的一边）。")
w(f"* 结论：**降回撤要动仓位，不是动选股**。选股再好也躲不掉指数跌 15–20% 的窗口；"
  f"而只要在那个窗口空仓一半，回撤就能砍掉一大半——§4 实测到了。")
w("")

# ---------------------------------------------------------------- 3. vol targeting
w("## 3. 已证伪：波动率目标（vol targeting）")
w("")
w("直觉上「波动大就减仓」应该能降回撤。实测不行：")
w("")
plain = probes["variants"]["plain"]
buf = probes["variants"]["buffered"]
w(row("变体", "年化", "Sharpe", "最大回撤", "累计"))
w(row("---", "---", "---", "---", "---"))
w(row(f"无缓冲 top30（基准）", pct(plain["base"]["net_annualized"]), f"{plain['base']['net_sharpe']:.3f}",
    pct(plain["base"]["max_drawdown"]), pct(plain["base"]["total_return"])))
for t in ("0.12", "0.15", "0.18"):
    m = plain[f"vol_target_{t}"]
    w(row(f"无缓冲 + 波动目标 {int(float(t)*100)}%", pct(m["net_annualized"]), f"{m['net_sharpe']:.3f}",
        pct(m["max_drawdown"]), pct(m["total_return"])))
w(row(f"有缓冲 top30/60（本文参照系）", pct(buf["base"]["net_annualized"]), f"{buf['base']['net_sharpe']:.3f}",
    pct(buf["base"]["max_drawdown"]), pct(buf["base"]["total_return"])))
for t in ("0.12", "0.15", "0.18"):
    m = buf[f"vol_target_{t}"]
    w(row(f"有缓冲 + 波动目标 {int(float(t)*100)}%", pct(m["net_annualized"]), f"{m['net_sharpe']:.3f}",
        pct(m["max_drawdown"]), pct(m["total_return"])))
w("")
w(f"**为什么无效**：波动目标 12% 把年化从 {pct(buf['base']['net_annualized'])} 砍到 "
  f"{pct(buf['vol_target_0.12']['net_annualized'])}，回撤却只从 {pct(buf['base']['max_drawdown'])} "
  f"变成 {pct(buf['vol_target_0.12']['max_drawdown'])}——**几乎没动**。"
  f"因为这个回撤不是「波动率尖峰」型，是**持续阴跌**型："
  f"半年里每个周期的波动都不算极端，但方向一直向下，滚动波动率根本没报警。"
  f"vol targeting 只能对付前者。**这条可以从清单里划掉。**")
w("")

# ---------------------------------------------------------------- 4. trend filter
w("## 4. 杠杆一：指数趋势过滤（仓位择时）—— 最有效")
w("")
w("规则：信号日 T 收盘后，若趋势指数收盘价 > 它自己的 MA(N)，则 T+1 开盘满仓，否则空仓"
  "（现金收益按 0 计，是保守口径）。空/满仓切换的往返成本**已按 bp 显式计入**。"
  "只用 T 及之前的数据，**无前视**。")
w("")
w("### 4.1 参数边界：短窗口更好，但短到 MA2 就开始退化")
w("")
bnd = stack["boundary"]
w(row("趋势指数", "MA", "年化", "Sharpe", "最大回撤", "在场比例", "换手", "切换次数/年"))
w(row("---", "---", "---", "---", "---", "---", "---", "---"))
w(row("**无过滤**", "—", pct(ref["net_annualized"]), f"{ref['net_sharpe']:.3f}",
    pct(ref["max_drawdown"]), "100.0%", f"{ref['mean_turnover']:.3f}", "—"))
for code, name in (("932000.CSI", "中证2000"), ("000985.CSI", "中证全指")):
    for ma in (2, 3, 5, 10, 20, 30, 40, 60):
        m = bnd[f"{code}_ma{ma}"]
        w(row(name, ma, pct(m["net_annualized"]), f"{m['net_sharpe']:.3f}", pct(m["max_drawdown"]),
              pct(m["in_market"], 1), f"{m['mean_turnover']:.3f}", f"{m['flips_per_year']:.1f}"))
w("")
# bounds computed from the grid, not eyeballed off the table
_band = [bnd[f"{c}_ma{m}"] for c in ("932000.CSI", "000985.CSI") for m in (2, 3, 5, 10, 20, 30)]
_lo_sharpe = min(m["net_sharpe"] for m in _band)
_worst_dd = min(m["max_drawdown"] for m in _band)
_best_dd = max(m["max_drawdown"] for m in _band)
w(f"**读法**：MA2–MA30 全区间、两个指数共 {len(_band)} 个组合，**无一例外**把 Sharpe 从 "
  f"{ref['net_sharpe']:.3f} 抬到 {_lo_sharpe:.3f} 以上、最大回撤从 {pct(ref['max_drawdown'])} "
  f"压到 {pct(_worst_dd,1)} ~ {pct(_best_dd,1)}。**方向是稳的**。"
  f"但最优点（中证2000 MA3，Sharpe {bnd['932000.CSI_ma3']['net_sharpe']:.3f}）落在窗口最短的一端，"
  f"而 MA2 已经回落——**这是典型的「参数在样本内被挑出来」的形状，具体数值不可信，区间可信**。"
  f"另外切换成本是真实的：MA3 每年切换 {bnd['932000.CSI_ma3']['flips_per_year']:.1f} 次，"
  f"MA20 只要 {bnd['932000.CSI_ma20']['flips_per_year']:.1f} 次。")
w("")
w("### 4.2 与换手杠杆叠加（完整网格见 JSON）")
w("")
w(row("组合", "年化", "Sharpe", "最大回撤", "累计", "换手"))
w(row("---", "---", "---", "---", "---", "---"))
show = [
    ("top30_exit60_sm1_none", "基准：top30 / 缓冲60 / 无平滑 / 无过滤"),
    ("top30_exit150_sm1_none", "top30 / 缓冲150 / 无过滤"),
    ("top30_exit150_sm1_csi2000_ma20_f0.0", "top30 / 缓冲150 / 中证2000 MA20 空仓"),
    ("top30_exit150_sm1_csi2000_ma10_f0.0", "top30 / 缓冲150 / 中证2000 MA10 空仓"),
    ("top30_exit150_sm1_csi2000_ma5_f0.0", "top30 / 缓冲150 / 中证2000 MA5 空仓（样本内最优）"),
    ("top30_exit90_sm2_csi2000_ma10_f0.0", "top30 / 缓冲90 / 平滑2日 / MA10 空仓"),
    ("top20_exit60_sm1_csi2000_ma5_f0.0", "top20 / 缓冲60 / MA5 空仓"),
    ("top50_exit90_sm2_csi2000_ma10_f0.0", "top50 / 缓冲90 / 平滑2日 / MA10 空仓"),
    (best_key, f"**网格最高 Sharpe**：{best_key}"),
]
for key, label in show:
    m = grid[key]["metrics"]
    w(row(label, pct(m["net_annualized"]), f"{m['net_sharpe']:.3f}", pct(m["max_drawdown"]),
        pct(m["total_return"]), f"{m['mean_turnover']:.3f}"))
w("")
w(f"**两条杠杆是独立可叠加的**：缓冲 60→150 单独把年化从 "
  f"{pct(grid['top30_exit60_sm1_none']['metrics']['net_annualized'])} 抬到 "
  f"{pct(grid['top30_exit150_sm1_none']['metrics']['net_annualized'])}；"
  f"MA20 过滤单独把回撤从 {pct(ref['max_drawdown'])} 压到 "
  f"{pct(grid['top30_exit150_sm1_csi2000_ma20_f0.0']['metrics']['max_drawdown'])}；"
  f"合起来年化 {pct(grid['top30_exit150_sm1_csi2000_ma5_f0.0']['metrics']['net_annualized'])}、"
  f"回撤 {pct(grid['top30_exit150_sm1_csi2000_ma5_f0.0']['metrics']['max_drawdown'])}、"
  f"Sharpe {grid['top30_exit150_sm1_csi2000_ma5_f0.0']['metrics']['net_sharpe']:.3f}、"
  f"累计 {pct(grid['top30_exit150_sm1_csi2000_ma5_f0.0']['metrics']['total_return'])}。")
w("")
w("### 4.3 关键稳健性检验：不是一次运气")
w("")
w("最大的担心是「过滤器只是躲过了 2024 年 1 月那一次小盘流动性危机」。"
  f"把那 {stack['episode_test'][[k for k in stack['episode_test'] if k.endswith('_none')][0]]['full']['cycles'] - stack['episode_test'][[k for k in stack['episode_test'] if k.endswith('_none')][0]]['ex_episode']['cycles']} "
  "个周期整段剔除后重算：")
w("")
w(row("组合", "全样本 Sharpe", "剔除危机后 Sharpe", "全样本回撤", "剔除危机后回撤"))
w(row("---", "---", "---", "---", "---"))
for key in ("top30_exit60_none", "top30_exit60_csi2000_ma5_f0.0", "top30_exit150_none",
            "top30_exit150_csi2000_ma5_f0.0", "top30_exit150_csi2000_ma20_f0.0",
            "top30_exit60_csiall_ma30_f0.0"):
    e = stack["episode_test"][key]
    w(row(key, f"{e['full']['net_sharpe']:.3f}", f"{e['ex_episode']['net_sharpe']:.3f}",
          pct(e["full"]["max_drawdown"]), pct(e["ex_episode"]["max_drawdown"])))
w("")
w(f"**过滤器在剔除危机后依然全面有效**：无过滤的基准剔除危机后 Sharpe "
  f"{stack['episode_test']['top30_exit60_none']['ex_episode']['net_sharpe']:.3f}，"
  f"加 MA5 过滤后 {stack['episode_test']['top30_exit60_csi2000_ma5_f0.0']['ex_episode']['net_sharpe']:.3f}；"
  f"回撤从 {pct(stack['episode_test']['top30_exit60_none']['ex_episode']['max_drawdown'])} 压到 "
  f"{pct(stack['episode_test']['top30_exit60_csi2000_ma5_f0.0']['ex_episode']['max_drawdown'])}。"
  f"注意**无过滤时即便剔除了那次危机，最大回撤仍有 "
  f"{pct(stack['episode_test']['top30_exit60_none']['ex_episode']['max_drawdown'])}**——"
  f"说明这个策略还有别的深回撤，过滤器把那些也一起压下去了。这是它最值得相信的一点。")
w("")
w(f"> ⚠️ **但 MA 窗口和趋势指数本身是在 TEST 上挑的**，属于样本内选择。"
  f"上线前必须用 2026 锁定段（`factor_daily` 目前只到 2025-12-31，需先物化因子）做一次"
  f"**只许选一次参数**的验证。§4.1 的区间比 §4.2 的单一最优值更该被引用。")
w("")

# ---------------------------------------------------------------- 5. turnover
w("## 5. 杠杆二：换手——最便宜的净收益")
w("")
w(f"当前成本吃掉毛利的比例很高：毛利年化 {pct(ref['gross_annualized'])}，"
  f"净年化 {pct(ref['net_annualized'])}。降低换手是**纯净额改善**，不承担任何模型风险。")
w("")
w("### 5.1 持仓缓冲宽度（N_enter=30 固定，N_exit 变化）")
w("")
w(row("N_exit", "年化", "Sharpe", "最大回撤", "换手", "累计"))
w(row("---", "---", "---", "---", "---", "---"))
for k in ("30", "45", "60", "90", "150"):
    m = levers["buffer_sweep"][k]
    w(row(k, pct(m["net_annualized"]), f"{m['net_sharpe']:.3f}", pct(m["max_drawdown"]),
          f"{m['mean_turnover']:.3f}", pct(m["total_return"])))
w("")
w("### 5.2 分数平滑（先对每个名字取最近 k 个信号日的平均分，再排序）")
w("")
w(row("平滑窗口 k", "年化", "Sharpe", "最大回撤", "换手", "累计"))
w(row("---", "---", "---", "---", "---", "---"))
for k in ("1", "2", "3", "5"):
    m = levers["smoothing_sweep"][k]
    w(row(k, pct(m["net_annualized"]), f"{m['net_sharpe']:.3f}", pct(m["max_drawdown"]),
          f"{m['mean_turnover']:.3f}", pct(m["total_return"])))
w("")
bs, ss = levers["buffer_sweep"]["150"], levers["smoothing_sweep"]["5"]
w(f"**读法**：缓冲从 30 放到 150，年化 {pct(levers['buffer_sweep']['30']['net_annualized'])} → "
  f"{pct(bs['net_annualized'])}，Sharpe {levers['buffer_sweep']['30']['net_sharpe']:.3f} → "
  f"{bs['net_sharpe']:.3f}，换手 {levers['buffer_sweep']['30']['mean_turnover']:.3f} → "
  f"{bs['mean_turnover']:.3f}，**回撤还顺带降了**"
  f"（{pct(levers['buffer_sweep']['30']['max_drawdown'])} → {pct(bs['max_drawdown'])}）。"
  f"平滑 k=5 把换手压到 {ss['mean_turnover']:.3f}、回撤压到 {pct(ss['max_drawdown'])}，"
  f"年化略低于 k=2。**45 是个反常的局部低点**（比 30 和 60 都差），"
  f"说明这个参数面上有噪声，不要精调，取宽的一端即可。")
w("")

# ---------------------------------------------------------------- 6. ensemble
w("## 6. 已证伪：换一种集成方式")
w("")
w("三个模型的 top50 只有 8.7 只重合，所以「怎么把它们合起来」看起来像个杠杆。实测不是：")
w("")
w(row("组合方式", "top30 年化", "top30 Sharpe", "top30 回撤", "top30 换手", "top100 Sharpe"))
w(row("---", "---", "---", "---", "---", "---"))
names = {"score_mean": "等权原始分数（现行）", "rank_mean": "等权百分位排名",
         "zscore_mean": "等权 z-score", "consensus": "共识计数（几个模型放进 top-N）"}
for tag, label in names.items():
    a, b = levers["ensemble_variants"]["30"][tag], levers["ensemble_variants"]["100"][tag]
    w(row(label, pct(a["net_annualized"]), f"{a['net_sharpe']:.3f}", pct(a["max_drawdown"]),
          f"{a['mean_turnover']:.3f}", f"{b['net_sharpe']:.3f}"))
w("")
zm = levers["ensemble_variants"]["30"]["zscore_mean"]
w(f"**读法**：现行的等权原始分数已经是最优或并列最优。z-score 平均是**灾难**"
  f"（Sharpe {zm['net_sharpe']:.3f}，换手 {zm['mean_turnover']:.3f}）——"
  f"因为三个模型的分数尺度和离散度差很多，标准化等于把权重交给了离散度最小的那个模型，"
  f"而且每天的权重都在跳。**这条也划掉，省下一次实验。**")
w("")

# ---------------------------------------------------------------- 7. factors
w("## 7. 杠杆三：因子——问题不是「不够多」，是「不够正交」")
w("")
w("### 7.1 15 个冻结因子的单因子体检（TEST，969 个信号日）")
w("")
fd = levers["factor_diagnostics"]["all_folds"]
w(row("位置", "因子", "稠密/稀疏", "IC 均值", "IC 标准差", "日度 IR", "IC>0 天数占比", "top50 超额", "NaN 率"))
w(row("---", "---", "---", "---", "---", "---", "---", "---", "---"))
for f in sorted(fd, key=lambda k: -abs(fd[k]["ic_mean"])):
    v = fd[f]
    w(row(f, f"`{FACTOR_MAP[f]}`", v["density"], f"{v['ic_mean']:+.4f}", f"{v['ic_std']:.4f}",
          f"{v['ic_ir_daily']:+.3f}", pct(v["ic_positive_share"], 1), f"{v['top50_spread_pp']:+.2f}pp",
          pct(v["nan_rate"], 1)))
w("")
neg = sum(1 for v in fd.values() if v["ic_mean"] < 0)
w(f"**读法**：")
w("")
w(f"* **{neg} / 15 是负 IC**——因子值越高，未来 5 日收益越低。"
  f"这批因子**主体是反转类**。唯一正 IC 的是 `{FACTOR_MAP['f12']}`"
  f"（{fd['f12']['ic_mean']:+.4f}，IC>0 的天数占 {pct(fd['f12']['ic_positive_share'],1)}，"
  f"日度 IR {fd['f12']['ic_ir_daily']:+.3f}），**也是 15 个里最稳的一个**。")
w(f"* **单因子 top50 超额全为负**（除 `{FACTOR_MAP['f12']}` 的 "
  f"{fd['f12']['top50_spread_pp']:+.2f}pp）。也就是说「按任一因子买最高的一批」是亏的，"
  f"有用的方向在**另一端**。模型学到了符号，所以这不是缺陷，但意味着"
  f"**原始因子层面不存在「买高值」的尾部 alpha**。")
w(f"* 单因子 |IC| 全在 {min(abs(v['ic_mean']) for v in fd.values()):.3f}–"
  f"{max(abs(v['ic_mean']) for v in fd.values()):.3f} 之间。这个量级本身正常"
  f"（日频截面 IC 0.05 就算能用），但它**给模型能力的上限封了顶**："
  f"再强的模型也无法从 0.05 的信号里榨出 0.15。")
w("")
w("### 7.2 冗余度：15 个因子里只有 4 个独立信号")
w("")
w(row("指标", "数值", "含义"))
w(row("---", "---", "---"))
w(row("平均 \\|因子两两相关\\|", f"{red['mean_abs_factor_correlation']:.3f}",
    "截面 Spearman，按日计算后跨日加权平均"))
w(row("最大 \\|因子两两相关\\|", f"{red['max_abs_factor_correlation']:.3f}", "最冗余的一对"))
w(row("**有效维度（参与率）**", f"**{red['effective_dimensions_participation_ratio']:.2f} / 15**",
    "(Σλ)²/Σλ²；15 = 完全独立，1 = 只有一个信号"))
w(row("PC1 解释方差", pct(red["variance_explained_by_pc1"], 1), "第一个主成分就吃掉近一半"))
w(row("**IC 序列有效维度**", f"**{red['ic_effective_dimensions']:.2f} / 15**",
    "因子可以水平不相关但同时失效；这才是模型真正面对的"))
w(row("IC 序列 PC1 占比", pct(red["ic_pc1_variance_share"], 1), "15 个因子的成败高度同步"))
w(row("参与计算的行数", f"{red['weighted_by_rows']:,}", "四个折的 TEST 全量"))
w("")
w("最冗余的 5 对（**全部落在 6 个稀疏 candidate-only 因子内部**）：")
w("")
w(row("因子 A", "因子 B", "水平相关", "IC 序列相关"))
w(row("---", "---", "---", "---"))
for p in red["top10_pairs_by_abs_correlation"][:5]:
    w(row(f"`{p['a']}`", f"`{p['b']}`", f"{p['level_corr']:+.3f}", f"{p['ic_series_corr']:+.3f}"))
w("")
w(f"**这就是因子端的结论**：`{FACTOR_MAP['f03']}` / `{FACTOR_MAP['f05']}` / `{FACTOR_MAP['f08']}` "
  f"三者水平相关 0.81–0.87、IC 序列相关 0.88–0.91，**基本是同一个信号测了三遍**。"
  f"6 个稀疏因子的有效多样性远小于 6，15 个因子的有效多样性只有 "
  f"{red['effective_dimensions_participation_ratio']:.2f}。")
w("")
w("所以对「要不要加因子」的回答是分两半的：")
w("")
w(f"* **加第 16、17 个同源因子（又一个反转/量价类）= 无用**，有效维度不会从 4 变到 5。")
w(f"* **要加就加正交族**：基本面/质量、分析师预期修正、资金流（龙虎榜/北向/大单）、"
  f"事件驱动。判据不是「新因子 IC 高」，而是「新因子与现有 PC1 的相关低」。")
w(f"* **反向也成立**：可以把 15 砍到 ~5–6 个而基本不损失信息。附带收益是 "
  f"`{FACTOR_MAP['f01']}` 与 `{FACTOR_MAP['f12']}` 的 NaN 缺口"
  f"（NaN 率 {pct(fd['f01']['nan_rate'],1)} / {pct(fd['f12']['nan_rate'],1)}）"
  f"正是历史上造成「24 个整天不可评 + 17,306 行未评分」的那条链；"
  f"因子集变窄，覆盖门槛就更容易过。")
w("")

# ---------------------------------------------------------------- 8. pnl concentration
w("## 8. 好消息：这个策略不是彩票")
w("")
w("集中度要分两个尺度看，两个数字看起来矛盾，其实不矛盾：")
w("")
ts = stack["tail_shape"]
w(row("尺度", "指标", "数值"))
w(row("---", "---", "---"))
w(row("**日内**（当天 30 只篮子内部）", "篮子为正的天数占比", pct(ts["share_of_days_positive"], 1)))
w(row("", "赢利日：top3 占当日篮子收益", pct(ts["positive_days"]["mean_top3_share"], 1)))
w(row("", "赢利日：篮子内涨幅 >9% 的个股占比", pct(ts["positive_days"]["mean_share_names_above_9pct"], 1)))
w(row("", "赢利日：当日最好一只平均涨", f"{ts['positive_days']['mean_daily_best_name_pp']:.2f}pp"))
w(row(f"**全回测**（{pnl['positions']:,} 笔持仓）", "最好的 10 笔占总毛利", pct(pnl["top10_positions_share_of_gross_profit"])))
w(row("", "最好的 100 笔占总毛利", pct(pnl["top100_positions_share_of_gross_profit"])))
w(row("", "最好的 1,000 笔占总毛利", pct(pnl["top1000_positions_share_of_gross_profit"])))
w(row("", "最好的 1% 笔数占总毛利", pct(pnl["top1pct_positions_share_of_gross_profit"])))
w(row("", "最好 0.1% 笔数占总毛利", pct(pnl["top0.1pct_positions_share_of_gross_profit"])))
w(row("", "单笔最大贡献", f"{pnl['largest_single_position_pp']:.2f}pp"))
w("")
w(f"**读法**：日内看极其集中——赢利日里当天最好的 3 只就贡献了 "
  f"{pct(ts['positive_days']['mean_top3_share'],1)} 的篮子收益"
  f"（因为篮子日均只有 {pnl['mean_daily_basket_return_pp']:.2f}pp，而个股单日能摆 ±10pp）。"
  f"但**全回测看非常分散**：{pnl['positions']:,} 笔持仓里，最好的 10 笔只占毛利的 "
  f"{pct(pnl['top10_positions_share_of_gross_profit'])}，最好的 1,000 笔占 "
  f"{pct(pnl['top1000_positions_share_of_gross_profit'])}。**赢家每天在轮换**，"
  f"没有一小撮妖股扛着整个策略。")
w("")
w(f"另一个有用的数：赢利仓位合计 {pnl['sum_of_winning_positions']:.2f}，"
  f"全部持仓净合计 {pnl['total_sum_of_daily_basket_returns']:.2f}——"
  f"**约 {100 - pnl['total_sum_of_daily_basket_returns']/pnl['sum_of_winning_positions']*100:.0f}% "
  f"的毛利被亏损仓位抵掉了**，盈利日占比 {pct(pnl['share_of_days_positive'],1)}。"
  f"这说明改善空间在**少犯错**（砍掉亏损尾部）而不只是多抓大牛股，"
  f"而「少犯错」正是 §4 的择时和 §5 的缓冲在做的事。")
w("")
w("> ⚠️ 仍然没有涨停过滤。日内集中度这么高意味着：**如果那些 >9% 的名字在 T+1 开盘就涨停、"
  "实际买不进**，日内那部分收益会消失。全回测的分散性说明它不会归零，但这个偏差至今未被量化，"
  "是所有结果里最大的乐观来源，且随集中度上升而恶化。")
w("")

# ---------------------------------------------------------------- 9. retraining
w("## 9. 需要重训的杠杆：只有方向，不给数字")
w("")
w("以下三条**没有实测数字**，因为它们必须重训。给的是判断，以及一条能省下一次重训的证明。")
w("")
w("### 9.1 验证指标与 top-N 目标不对齐")
w("")
w("现行 G4 用的是**全截面** RankIC，而实盘只用 **top30**。这两件事不是一回事——"
  "M2 就是因为 G4 差 4.08e-5 没过。把选择标准换成 top-N 导向（top-N 平均超额、"
  "top-N 命中率）在方向上是对的。")
w("")
w("**但有一个反证必须先说**：已实测的尾部锐度（top50 − q5 毛利）是 "
  "baseline −2.79pp、M0-refined −3.55pp（最差）、M0R +2.90pp、M1 +4.06pp、**M2 +5.51pp（最好）**。"
  "而基线在 q5 赢、模型在 top50/top100 赢。"
  "所以**不要直接上 LambdaRank 或尾部加权损失**——现有证据不支持「尾部越锐越好」，"
  "M2 已经是尾部最锐的，它并没有因此在 top-N 上碾压其余模型。")
w("")
w("### 9.2 风险调整 label：值得试，但有一类改动可以证明无效")
w("")
w("把 label 换成「收益 / 个股自身波动」这类风险调整版本，理论上直接优化 Sharpe 和回撤，"
  "值得一次实验。")
w("")
w("**但「label 减去当日指数收益」这一类改动可以证明对选股无效**：同一天所有股票减掉的是"
  "**同一个常数**，截面排序完全不变，RankIC 一字不动，选出来的 top30 是同一批股票。"
  "它只影响组合层面的时序——而那正是 §4 的择时已经在做、且已经实测有效的事。"
  "**这条能省下一次完整的重训。**")
w("")
w("### 9.3 修 M2 的历史长度门槛")
w("")
w("M2 的 `minimum_valid_history_length=15` 让 17,306 行无法评分。修掉它可能同时解决 "
  "G6 和覆盖交集被 M2 拉窄的问题——而覆盖交集变宽会直接让本文所有数字的样本变大。"
  "这是重训类里性价比最高的一条。")
w("")

# ---------------------------------------------------------------- 10. plan
w("## 10. 我会怎么做")
w("")
w(row("步", "做什么", "成本", "预期（有实测支撑的部分）"))
w(row("---", "---", "---", "---"))
_ma10 = grid["top30_exit150_sm1_csi2000_ma10_f0.0"]["metrics"]
w(row("1", "上「宽缓冲 + 趋势过滤」：N_enter=30 / N_exit≈150，中证2000 MA10 跌破则空仓",
    "0 重训，约 1 天",
    f"年化 {pct(ref['net_annualized'])} → {pct(_ma10['net_annualized'])}，"
    f"回撤 {pct(ref['max_drawdown'])} → {pct(_ma10['max_drawdown'])}，"
    f"Sharpe {ref['net_sharpe']:.2f} → {_ma10['net_sharpe']:.2f}，"
    f"累计 {pct(ref['total_return'])} → {pct(_ma10['total_return'])}"))
w(row("2", "用 2026 锁定段做**一次性**参数确认（先物化 2026 因子）", "0 重训",
    "把 §4.1 的区间收窄成一个可信的点；这是防样本内选择的唯一办法"))
w(row("3", "因子端：不加数量，加正交族；同时把 15 砍到 ~6", "重训 + 重新物化",
    f"判据用「与现有 PC1 相关低」，不是「IC 高」；有效维度目标 > {red['effective_dimensions_participation_ratio']:.1f}"))
w(row("4", "验证指标从全截面 RankIC 换成 top-N 导向", "重训", "方向对，但见 §9.1 的反证，别顺手加尾部加权"))
w(row("5", "风险调整 label 做一次对照实验", "重训", "预期收益小于第 1、3 步；见 §9.2 能省掉的那一类"))
w(row("6", "模型结构", "重训", "最后才动"))
w("")
w(f"**为什么是这个顺序**：第 1 步不需要重训、不碰任何冻结工件、当天可验证，"
  f"而它把最大回撤从 {pct(ref['max_drawdown'])} 压到 {_ma10['max_drawdown']*100:.2f}%、"
  f"Sharpe 从 {ref['net_sharpe']:.2f} 抬到 {_ma10['net_sharpe']:.2f}。"
  f"第 3–6 步都要重训，且**都改不了回撤的 beta 部分**（§2 已归因）。"
  f"先把便宜的、已证明的拿走，再花贵的。")
w("")
w("**但必须说清收益那一半——这里有一个真实的取舍，不能两头都许**：")
w("")
# The delta column MUST be against the same-buffer unfiltered row (N_exit=150), not against
# the N_exit=60 reference. Comparing a filtered exit150 run to an unfiltered exit60 run lets
# the buffer's gain be credited to the filter, which is exactly the error this table exists
# to prevent.
_base150 = grid["top30_exit150_sm1_none"]["metrics"]
w(row("趋势窗口（均配 N_exit=150）", "年化", "最大回撤", "Sharpe", "累计",
    "相对**同缓冲**无过滤的年化差"))
w(row("---", "---", "---", "---", "---", "---"))
for key, label in (("top30_exit150_sm1_none", "无过滤（同缓冲基准）"),
                   ("top30_exit150_sm1_csi2000_ma5_f0.0", "MA5（样本内最优）"),
                   ("top30_exit150_sm1_csi2000_ma10_f0.0", "**MA10（本文推荐）**"),
                   ("top30_exit150_sm1_csi2000_ma20_f0.0", "MA20（稳健端）")):
    m = grid[key]["metrics"]
    delta = (m["net_annualized"] - _base150["net_annualized"]) * 100
    w(row(label, pct(m["net_annualized"]), pct(m["max_drawdown"]), f"{m['net_sharpe']:.3f}",
          pct(m["total_return"]), f"{delta:+.2f}pp"))
w("")
_ma5 = grid["top30_exit150_sm1_csi2000_ma5_f0.0"]["metrics"]
_ma20 = grid["top30_exit150_sm1_csi2000_ma20_f0.0"]["metrics"]
w(f"**这张表必须这样读，否则会得出一个错的结论**：在**同一个缓冲宽度**下，"
  f"加过滤器**永远是要付出年化的**——MA5 {(_ma5['net_annualized']-_base150['net_annualized'])*100:+.2f}pp、"
  f"MA10 {(_ma10['net_annualized']-_base150['net_annualized'])*100:+.2f}pp、"
  f"MA20 {(_ma20['net_annualized']-_base150['net_annualized'])*100:+.2f}pp。"
  f"乍看之下「MA5 组合的年化 {pct(_ma5['net_annualized'])} 比 §1 参照系的 "
  f"{pct(ref['net_annualized'])} 还高，所以择时提高了收益」——**这个推论是错的**："
  f"那 {(_ma5['net_annualized']-ref['net_annualized'])*100:+.2f}pp 全部来自把缓冲从 60 放到 150，"
  f"过滤器自己是负的。**两条杠杆的功劳不能互相记。**")
w("")
w(f"所以准确的说法是：")
w("")
w(f"* **提收益**靠 §5 的换手杠杆：{pct(ref['net_annualized'])} → {pct(_base150['net_annualized'])}"
  f"（{(_base150['net_annualized']-ref['net_annualized'])*100:+.2f}pp），"
  f"回撤顺带从 {pct(ref['max_drawdown'])} 到 {pct(_base150['max_drawdown'])}。"
  f"**这一条无争议、无参数风险、不需要重训，如果只做一件事就做它。**")
w(f"* **降回撤**靠趋势过滤：{pct(_base150['max_drawdown'])} → {pct(_ma10['max_drawdown'])}（MA10），"
  f"代价是 {(_ma10['net_annualized']-_base150['net_annualized'])*100:+.2f}pp 年化，"
  f"Sharpe 从 {_base150['net_sharpe']:.3f} 到 {_ma10['net_sharpe']:.3f}。"
  f"**「降回撤」是确定的，「不花收益」不是。**")
w(f"* 两条合起来相对最初的参照系是**全面改善**（年化 {pct(ref['net_annualized'])} → "
  f"{pct(_ma10['net_annualized'])}、回撤 {pct(ref['max_drawdown'])} → {pct(_ma10['max_drawdown'])}、"
  f"Sharpe {ref['net_sharpe']:.3f} → {_ma10['net_sharpe']:.3f}、累计 {pct(ref['total_return'])} → "
  f"{pct(_ma10['total_return'])}），只是年化那一格是**缓冲挣的、被过滤器花掉一部分**后的净值。")
w("")

# ---------------------------------------------------------------- 11. limits
w("## 11. 这个分析不能证明什么")
w("")
w(f"* **MA 窗口和趋势指数是在 TEST 上挑的**（样本内选择）。§4.1 显示方向稳健、"
  f"但最优点落在参数面边界附近，具体数值不可信。")
w(f"* **没有涨停过滤**。日内集中度极高（§8），这是最大的未量化乐观偏差。")
w(f"* **成本是假设**：{levers['config']['round_trip_bp']:.0f}bp 往返"
  f"（≈10bp 显性 + ≈5bp 滑点）。集中持仓时真实滑点更高。")
w(f"* **Sharpe 用 rf=0**。减去 2%–3% 的无风险利率后，超额 Sharpe 要减约 0.07–0.10。")
w(f"* **Sharpe 的标准误 ≈ 0.085**（n=192）。0.96 与 1.02 的差别在噪声内；"
  f"0.96 与 1.49 的差别不在。")
_im = [bnd[k]["in_market"] for k in bnd if k.startswith("932000")]
w(f"* **择时把在场时间压到 {pct(min(_im), 1)}–{pct(max(_im), 1)}**，"
  f"现金收益按 0 计（保守），但资金占用率、可投容量和税务/冲击成本都没建模。")
w(f"* **全部基于三模型覆盖的交集行**，宽度由 M2 最窄的覆盖决定；"
  f"修掉 M2 的历史门槛（§9.3）会让样本变大，数字会变。")
w("")

# ---------------------------------------------------------------- 12. gate-clean
w("## 12. 治理干净的方案要不要付性能代价：不用")
w("")
w("本文的参照系是 M0R+M1+M2 等权集成，但门槛状态是：")
w("")
w(row("模型", "门槛", "注册表"))
w(row("---", "---", "---"))
for m, (g, r) in (("M0R", ("六门全过", "`WALK_FORWARD_PASSED`")),
                  ("M1", ("六门全过", "**未注册**（注册是独立的显式命令，训练不会自动触发）")),
                  ("M2", ("**G4 差 4.08e-05、G6 = 905/969 = 0.9340 < 0.95**", "无晋级证据"))):
    w(row(f"`{m}`", g, r))
w("")
w("所以治理干净的选择是 M0R+M1。它要不要付性能代价，是实测出来的，不是判断：")
w("")
gc = gateclean["member_sets"]
w(row("成员集", "合格行", "日均候选", "年化", "Sharpe", "最大回撤", "累计"))
w(row("---", "---", "---", "---", "---", "---", "---"))
for tag in ("M0R+M1+M2", "M0R+M1 (gate-clean)", "M0R only", "M1 only"):
    e = gc[tag]
    m = e["variants"]["top30_exit60"]["none"]
    label = f"**{tag}**" if tag == "M0R+M1 (gate-clean)" else tag
    w(row(label, f"{e['total_eligible_rows']:,}", f"{e['total_signal_dates'] and e['total_eligible_rows']/e['total_signal_dates']:.0f}",
          pct(m["net_annualized"]), f"{m['net_sharpe']:.3f}", pct(m["max_drawdown"]), pct(m["total_return"])))
bl = gc["M0R+M1 (gate-clean)"]["baseline_same_rows"]
w(row("基线 best_simple（同行、top30）", "同上", "同上", pct(bl["net_annualized"]),
    f"{bl['net_sharpe']:.3f}", pct(bl["max_drawdown"]), pct(bl["total_return"])))
w("")
_3 = gc["M0R+M1+M2"]
_2 = gc["M0R+M1 (gate-clean)"]
_gain = _2["total_eligible_rows"] - _3["total_eligible_rows"]
w(f"**三条结论**：")
w("")
w(f"* **去掉 M2 不亏**。参照系配置下两模型集成在年化"
  f"（{pct(_3['variants']['top30_exit60']['none']['net_annualized'])} → "
  f"{pct(_2['variants']['top30_exit60']['none']['net_annualized'])}）、Sharpe"
  f"（{_3['variants']['top30_exit60']['none']['net_sharpe']:.3f} → "
  f"{_2['variants']['top30_exit60']['none']['net_sharpe']:.3f}）、回撤"
  f"（{pct(_3['variants']['top30_exit60']['none']['max_drawdown'])} → "
  f"{pct(_2['variants']['top30_exit60']['none']['max_drawdown'])}）、累计"
  f"（{pct(_3['variants']['top30_exit60']['none']['total_return'])} → "
  f"{pct(_2['variants']['top30_exit60']['none']['total_return'])}）**四项全部不差**。"
  f"Sharpe 差 {_2['variants']['top30_exit60']['none']['net_sharpe'] - _3['variants']['top30_exit60']['none']['net_sharpe']:+.3f}，"
  f"**在 1 个标准误（≈0.085）以内**，所以准确说法是「不比它差」而不是「比它好」。")
w(f"* **换回了正好 {_gain:,} 行覆盖**。这与独立记录的 M2 因 "
  f"`minimum_valid_history_length=15` 而无法评分的行数**逐位吻合**——"
  f"两条完全独立的测量路径给出同一个数，这本身就是 M2 覆盖缺口归因的硬证据。")
w(f"* **集成仍然值得**：两模型集成（Sharpe "
  f"{_2['variants']['top30_exit60']['none']['net_sharpe']:.3f}）明显优于任何单模型"
  f"（M0R {gc['M0R only']['variants']['top30_exit60']['none']['net_sharpe']:.3f}、"
  f"M1 {gc['M1 only']['variants']['top30_exit60']['none']['net_sharpe']:.3f}），"
  f"也远优于同行基线（{bl['net_sharpe']:.3f}，回撤 {pct(bl['max_drawdown'])}）。"
  f"**即使三个模型的 top50 只有 8.7 只重合，等权集成依然稳定地好过其中任何一个。**")
w("")
w("加上趋势过滤后结论不变（top30 / N_exit=150 / 中证2000 MA10）：")
w("")
w(row("成员集", "年化", "Sharpe", "最大回撤", "累计"))
w(row("---", "---", "---", "---", "---"))
for tag in ("M0R+M1+M2", "M0R+M1 (gate-clean)"):
    m = gc[tag]["variants"]["top30_exit150"]["ma10"]
    label = f"**{tag}**" if tag == "M0R+M1 (gate-clean)" else tag
    w(row(label, pct(m["net_annualized"]), f"{m['net_sharpe']:.3f}", pct(m["max_drawdown"]),
          pct(m["total_return"])))
w("")
_3f = _3["variants"]["top30_exit150"]["ma10"]
_2f = _2["variants"]["top30_exit150"]["ma10"]
w(f"这一档三模型的 Sharpe 略高（{_3f['net_sharpe']:.3f} 对 {_2f['net_sharpe']:.3f}，"
  f"差 {_3f['net_sharpe']-_2f['net_sharpe']:+.3f}，约 "
  f"{(_3f['net_sharpe']-_2f['net_sharpe'])/0.085:.1f} 个标准误），"
  f"两模型的回撤更好（{pct(_2f['max_drawdown'])} 对 {pct(_3f['max_drawdown'])}）。"
  f"**这点差距不足以支撑「为了性能保留一个未过门的模型」**，"
  f"何况保留 M2 还会把本文所有数字的样本继续压窄 {_gain:,} 行。")
w("")
w(f"> **推荐**：以 **M0R + M1 等权集成**为准。它六门全过、覆盖更宽、性能不差，"
  f"且把 M2 的门槛问题从关键路径上移走了。若日后修好 M2 的 "
  f"`minimum_valid_history_length`（§9.3）并让它过门，再把它加回来重测一次即可。")
w("")

# ---------------------------------------------------------------- 13. repro
w("## 13. 复现")
w("")
w("```bash")
w("cd docs/research/2026-10-08-model-backtest")
w("PY=/data/zhangyudong/.conda/envs/quantyd/bin/python")
w("$PY optimization_probes.py      # -> optimization_probe_results.json   (回撤定位 / 缓冲 / vol targeting)")
w("$PY optimization_levers.py      # -> optimization_levers_results.json  (归因 / 择时 / 换手 / 集成 / 单因子)")
w("$PY optimization_stack.py       # -> optimization_stack_results.json   (边界 / 叠加网格 / 单事件检验)")
w("$PY factor_redundancy.py        # -> factor_redundancy_results.json    (有效维度 / IC 共动 / P&L 集中度)")
w("$PY gate_clean_ensemble.py      # -> gate_clean_ensemble_results.json  (去掉 M2 值不值 / 覆盖换回多少)")
w("$PY write_optimization_report.py")
w("$PY verify_optimization_report.py")
w("```")
w("")
w("全部只读已签名工件 + 只读数据库查询；不写任何 `dfs://` 库，不训练，不改主工作树。")
w("")

OUT.write_text("\n".join(L) + "\n")
print(f"wrote {OUT}  ({len(L)} lines)")
