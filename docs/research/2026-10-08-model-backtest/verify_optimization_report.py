#!/usr/bin/env python3
"""Assert that 2026-10-08-optimization-levers.md is well-formed and its numbers are real.

Two independent checks, because this project has already shown the model fabricated numbers:

  1. STRUCTURE -- balanced ** pairs per line, no stray emphasis markers, every markdown
     table internally consistent (header, separator and body rows all the same width),
     no leftover python format placeholders.
  2. NUMBERS -- a list of (claim, expected string, JSON path) triples. Each expected string
     is recomputed from the JSON right here, then asserted to be literally present in the
     document. The script reports how many cells it actually compared, so a vacuous pass
     is visible.

Exits non-zero on any failure.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DOC = HERE.parent / "2026-10-08-optimization-levers.md"

probes = json.loads((HERE / "optimization_probe_results.json").read_text())
levers = json.loads((HERE / "optimization_levers_results.json").read_text())
stack = json.loads((HERE / "optimization_stack_results.json").read_text())
redun = json.loads((HERE / "factor_redundancy_results.json").read_text())
gateclean = json.loads((HERE / "gate_clean_ensemble_results.json").read_text())

text = DOC.read_text()
lines = text.split("\n")
errors: list[str] = []

# ---------------------------------------------------------------- 1. structure
def cells(line: str) -> list[str]:
    """Markdown table cells, splitting only on UNESCAPED pipes (\\| is a literal)."""
    parts = re.split(r"(?<!\\)\|", line)
    return [c.strip() for c in parts[1:-1]]


for n, line in enumerate(lines, 1):
    if line.count("**") % 2:
        errors.append(f"L{n}: unbalanced ** ({line.count('**')} markers): {line[:100]}")
    # a leading list marker is not emphasis; neither is an escaped literal pipe
    stripped = re.sub(r"^\s*[*+-]\s+", "", line).replace("**", "").replace("\\|", "")
    if re.search(r"(?<!\*)\*(?!\*)", stripped):
        errors.append(f"L{n}: stray single-asterisk emphasis: {line[:100]}")
    if re.search(r"\{[a-z_]+\[|\{[a-z_]+:|\.3f\}|\.2f\}", line):
        errors.append(f"L{n}: unformatted python placeholder leaked: {line[:100]}")
    # only a NaN in a VALUE is a bug; "NaN 率" as a column header or "NaN 缺口" in prose is not
    if line.startswith("|"):
        for cell in cells(line):
            if re.fullmatch(r"-?nan|nan%|nanpp|\+?nan\.?\d*", cell, re.IGNORECASE):
                errors.append(f"L{n}: NaN value in table cell: {line[:100]}")
    elif re.search(r"\bnan%|\bnanpp", line):
        errors.append(f"L{n}: NaN value in prose: {line[:100]}")

# table consistency: consecutive | lines must agree on column count
tables, current = [], []
for n, line in enumerate(lines, 1):
    if line.startswith("|"):
        current.append((n, len(cells(line))))
    elif current:
        tables.append(current)
        current = []
if current:
    tables.append(current)
for table in tables:
    widths = {w for _, w in table}
    if len(widths) != 1:
        errors.append(f"table starting L{table[0][0]}: inconsistent column counts {sorted(widths)}")
    if len(table) < 2 or not re.fullmatch(r"[\s:|-]+", lines[table[1][0] - 1].strip("|")):
        errors.append(f"table starting L{table[0][0]}: second line is not a separator row")

# ---------------------------------------------------------------- 2. numbers
def pct(x: float, nd: int = 2) -> str:
    return f"{x * 100:.{nd}f}%"


ref = levers["reference"]["buffered_top30"]
grid = stack["stack_grid"]
bnd = stack["boundary"]
ep = stack["episode_test"]
fd = levers["factor_diagnostics"]["all_folds"]
rd = redun["redundancy"]
pn = redun["pnl_concentration"]
attr = levers["drawdown_attribution"]
ts = stack["tail_shape"]

claims: list[tuple[str, str]] = [
    # reference / baseline of every comparison
    ("参照系年化", pct(ref["net_annualized"])),
    ("参照系 Sharpe", f"{ref['net_sharpe']:.3f}"),
    ("参照系回撤", pct(ref["max_drawdown"])),
    ("参照系累计", pct(ref["total_return"])),
    ("参照系换手", f"{ref['mean_turnover']:.3f}"),
    ("参照系毛利年化", pct(ref["gross_annualized"])),
    ("回撤窗口起", attr["window"][0]),
    ("回撤窗口止", attr["window"][1]),
    # attribution
    ("策略窗口净收益", pct(attr["strategy_net"])),
    ("上证窗口", pct(attr["indices"]["000001.SH"]["close_to_close"])),
    ("中证全指窗口", pct(attr["indices"]["000985.CSI"]["close_to_close"])),
    ("中证1000窗口", pct(attr["indices"]["000852.SH"]["close_to_close"])),
    ("中证2000窗口", pct(attr["indices"]["932000.CSI"]["close_to_close"])),
    # vol targeting refutation
    ("vol12 年化", pct(probes["variants"]["plain"]["vol_target_0.12"]["net_annualized"])),
    ("vol12 回撤", pct(probes["variants"]["plain"]["vol_target_0.12"]["max_drawdown"])),
    ("buffered vol12 年化", pct(probes["variants"]["buffered"]["vol_target_0.12"]["net_annualized"])),
    ("buffered vol12 回撤", pct(probes["variants"]["buffered"]["vol_target_0.12"]["max_drawdown"])),
    # trend filter boundary
    ("MA3 Sharpe", f"{bnd['932000.CSI_ma3']['net_sharpe']:.3f}"),
    ("MA3 年化", pct(bnd["932000.CSI_ma3"]["net_annualized"])),
    ("MA20 Sharpe", f"{bnd['932000.CSI_ma20']['net_sharpe']:.3f}"),
    ("MA60 Sharpe", f"{bnd['932000.CSI_ma60']['net_sharpe']:.3f}"),
    ("MA3 切换/年", f"{bnd['932000.CSI_ma3']['flips_per_year']:.1f}"),
    ("MA20 切换/年", f"{bnd['932000.CSI_ma20']['flips_per_year']:.1f}"),
    # stack grid
    ("基准 grid 年化", pct(grid["top30_exit60_sm1_none"]["metrics"]["net_annualized"])),
    ("缓冲150 无过滤 年化", pct(grid["top30_exit150_sm1_none"]["metrics"]["net_annualized"])),
    ("缓冲150 无过滤 Sharpe", f"{grid['top30_exit150_sm1_none']['metrics']['net_sharpe']:.3f}"),
    ("缓冲150+MA20 年化", pct(grid["top30_exit150_sm1_csi2000_ma20_f0.0"]["metrics"]["net_annualized"])),
    ("缓冲150+MA20 回撤", pct(grid["top30_exit150_sm1_csi2000_ma20_f0.0"]["metrics"]["max_drawdown"])),
    ("缓冲150+MA20 Sharpe", f"{grid['top30_exit150_sm1_csi2000_ma20_f0.0']['metrics']['net_sharpe']:.3f}"),
    ("缓冲150+MA5 年化", pct(grid["top30_exit150_sm1_csi2000_ma5_f0.0"]["metrics"]["net_annualized"])),
    ("缓冲150+MA5 回撤", pct(grid["top30_exit150_sm1_csi2000_ma5_f0.0"]["metrics"]["max_drawdown"])),
    ("缓冲150+MA5 Sharpe", f"{grid['top30_exit150_sm1_csi2000_ma5_f0.0']['metrics']['net_sharpe']:.3f}"),
    ("缓冲150+MA5 累计", pct(grid["top30_exit150_sm1_csi2000_ma5_f0.0"]["metrics"]["total_return"])),
    # episode test
    ("剔除危机 基准 Sharpe", f"{ep['top30_exit60_none']['ex_episode']['net_sharpe']:.3f}"),
    ("剔除危机 MA5 Sharpe", f"{ep['top30_exit60_csi2000_ma5_f0.0']['ex_episode']['net_sharpe']:.3f}"),
    ("剔除危机 基准 回撤", pct(ep["top30_exit60_none"]["ex_episode"]["max_drawdown"])),
    ("剔除危机 MA5 回撤", pct(ep["top30_exit60_csi2000_ma5_f0.0"]["ex_episode"]["max_drawdown"])),
    # turnover levers
    ("缓冲30 年化", pct(levers["buffer_sweep"]["30"]["net_annualized"])),
    ("缓冲150 换手", f"{levers['buffer_sweep']['150']['mean_turnover']:.3f}"),
    ("平滑5 换手", f"{levers['smoothing_sweep']['5']['mean_turnover']:.3f}"),
    ("平滑5 回撤", pct(levers["smoothing_sweep"]["5"]["max_drawdown"])),
    ("平滑2 年化", pct(levers["smoothing_sweep"]["2"]["net_annualized"])),
    # ensemble refutation
    ("score_mean Sharpe", f"{levers['ensemble_variants']['30']['score_mean']['net_sharpe']:.3f}"),
    ("zscore Sharpe", f"{levers['ensemble_variants']['30']['zscore_mean']['net_sharpe']:.3f}"),
    ("zscore 换手", f"{levers['ensemble_variants']['30']['zscore_mean']['mean_turnover']:.3f}"),
    # factor diagnostics
    ("f12 IC", f"{fd['f12']['ic_mean']:+.4f}"),
    ("f12 IC>0 占比", pct(fd["f12"]["ic_positive_share"], 1)),
    ("f12 IR", f"{fd['f12']['ic_ir_daily']:+.3f}"),
    ("f12 top50", f"{fd['f12']['top50_spread_pp']:+.2f}pp"),
    ("f01 NaN", pct(fd["f01"]["nan_rate"], 1)),
    # redundancy
    ("有效维度", f"{rd['effective_dimensions_participation_ratio']:.2f}"),
    ("IC 有效维度", f"{rd['ic_effective_dimensions']:.2f}"),
    ("平均因子相关", f"{rd['mean_abs_factor_correlation']:.3f}"),
    ("最大因子相关", f"{rd['max_abs_factor_correlation']:.3f}"),
    ("PC1 方差", pct(rd["variance_explained_by_pc1"], 1)),
    ("IC PC1 占比", pct(rd["ic_pc1_variance_share"], 1)),
    ("参与行数", f"{rd['weighted_by_rows']:,}"),
    # pnl concentration
    ("持仓笔数", f"{pn['positions']:,}"),
    ("top10 占毛利", pct(pn["top10_positions_share_of_gross_profit"])),
    ("top100 占毛利", pct(pn["top100_positions_share_of_gross_profit"])),
    ("top1000 占毛利", pct(pn["top1000_positions_share_of_gross_profit"])),
    ("top1% 占毛利", pct(pn["top1pct_positions_share_of_gross_profit"])),
    ("top0.1% 占毛利", pct(pn["top0.1pct_positions_share_of_gross_profit"])),
    ("单笔最大", f"{pn['largest_single_position_pp']:.2f}pp"),
    ("日均篮子", f"{pn['mean_daily_basket_return_pp']:.2f}pp"),
    ("盈利日占比", pct(pn["share_of_days_positive"], 1)),
    ("赢利仓位合计", f"{pn['sum_of_winning_positions']:.2f}"),
    ("净合计", f"{pn['total_sum_of_daily_basket_returns']:.2f}"),
    # tail shape
    ("赢利日 top3 占比", pct(ts["positive_days"]["mean_top3_share"], 1)),
    ("赢利日 >9% 占比", pct(ts["positive_days"]["mean_share_names_above_9pct"], 1)),
    ("赢利日最好一只", f"{ts['positive_days']['mean_daily_best_name_pp']:.2f}pp"),
    ("篮子为正天数占比", pct(ts["share_of_days_positive"], 1)),
]

# every most-redundant pair must appear too
for pair in rd["top10_pairs_by_abs_correlation"][:5]:
    claims.append((f"冗余对 {pair['a']}~{pair['b']} 水平", f"{pair['level_corr']:+.3f}"))
    claims.append((f"冗余对 {pair['a']}~{pair['b']} IC", f"{pair['ic_series_corr']:+.3f}"))

# ---- section 12: the gate-clean comparison
gc = gateclean["member_sets"]
_g3, _g2 = gc["M0R+M1+M2"], gc["M0R+M1 (gate-clean)"]
for tag, e in gc.items():
    v = e["variants"]["top30_exit60"]["none"]
    claims.append((f"{tag} 合格行", f"{e['total_eligible_rows']:,}"))
    claims.append((f"{tag} top30/exit60 年化", pct(v["net_annualized"])))
    claims.append((f"{tag} top30/exit60 Sharpe", f"{v['net_sharpe']:.3f}"))
    claims.append((f"{tag} top30/exit60 回撤", pct(v["max_drawdown"])))
    claims.append((f"{tag} top30/exit60 累计", pct(v["total_return"])))
    # the overlay table in the document only carries the two ensembles, so only those
    # two are asserted here; claiming rows the document does not contain is a false pass
    if tag in ("M0R+M1+M2", "M0R+M1 (gate-clean)"):
        f10 = e["variants"]["top30_exit150"]["ma10"]
        claims.append((f"{tag} exit150+MA10 Sharpe", f"{f10['net_sharpe']:.3f}"))
        claims.append((f"{tag} exit150+MA10 回撤", pct(f10["max_drawdown"])))
_bl = _g2["baseline_same_rows"]
claims += [
    ("同行基线年化", pct(_bl["net_annualized"])),
    ("同行基线 Sharpe", f"{_bl['net_sharpe']:.3f}"),
    ("同行基线回撤", pct(_bl["max_drawdown"])),
    ("同行基线累计", pct(_bl["total_return"])),
    ("覆盖换回行数", f"{_g2['total_eligible_rows'] - _g3['total_eligible_rows']:,}"),
]

compared = 0
for label, expected in claims:
    if expected in text:
        compared += 1
    else:
        errors.append(f"MISSING {label}: expected literal '{expected}' not found in the document")

# ---------------------------------------------------------------- report
print(f"document        : {DOC}")
print(f"lines           : {len(lines)}")
print(f"tables checked  : {len(tables)}")
print(f"numeric claims  : {len(claims)}  (matched {compared})")
if errors:
    print(f"\nFAILED with {len(errors)} problem(s):")
    for e in errors[:60]:
        print(f"  - {e}")
    sys.exit(1)
print("\nOK: structure clean and every numeric claim traced back to a JSON file.")
