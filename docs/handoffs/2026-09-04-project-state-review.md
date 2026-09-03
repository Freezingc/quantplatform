# 项目状态复查文件（Reviewable State Snapshot）

> 生成时间：2026-09-04 01:50 CST（Asia/Shanghai）
> 仓库：`/data/zhangyudong/FDU/Quant-Platform-Prod_CODEX` · 分支 `codex/publish-quant-platform` · 生成时 HEAD `5bc59bd`（本文件随下一次提交入库）
> 远端：`https://github.com/Freezingc/quantplatform.git`（网络不稳定，推送可能需重试）
> 环境：`conda activate quantyd` · `PYTHONPATH=research_v2/src:research_v2/scripts` · DolphinDB `dfs://QuantResearchV2Test` @ 10.176.55.211:8848（用户 xiahongjie）
> 用途：对「因子质量改进 → 准入冻结 → 因子值物化 → 模型训练」全流程当前状态做一次可复查快照。所有关键结论在第 7 节附复核命令，复查者可直接重跑验证。

## 0. 一句话状态

冻结因子集（17 个因子）已决策并入库；冻结集因子值正在物化进 `factor_daily`（2020–2023 完成，2024 进行中，2025 未开始）；物化完成后进入模型训练准备。**模型训练尚未开始；2026 锁定测试段保持未读。**

## 1. 不变约束（复查红线）

1. 2026 锁定测试段：严禁读取、严禁用于任何训练/调参/因子选择。
2. 未经批准不改正式因子、不发布新因子值、不训练模型。
3. 股票宇宙仅 SH/SZ（排除北交所 BJ）；训练与评估一律直读官方 `universe_daily` 的 `cn_liquid_a`（不做静态复现，存在 ~17% 漂移，见决策简报 4w）。
4. 不修改已有数据库表结构（可新建表）。
5. 不复用历史 YesLab 因子值，只用隔离规范版本。

## 2. 全流程位置

```
因子质量检验（全部完成）
  → 不达标因子补救修复（第一优先级完成：yeslab_206 修复版、443 修复变体等）
  → 五期证据（2017/2018/2019 训练期前 + 2020-2024 训练 + 2025 验证，官方口径）
  → 四维分级（训练预测力/样本外/体制稳健/独特性）：S4 / A10 / B6 / C2 / D1
  → 准入决策（决策点 D1-D4 全部确认）
  → 冻结因子集（17 因子，写入 factor_set_frozen）          ✅ 2026-09-03 完成
  → 冻结集因子值物化进 factor_daily（2020-2025）            ◐ 进行中（当前阶段）
  → 模型训练准备 → 训练                                      ○ 未开始
```

## 3. 冻结因子集（2026-09-03 入库，17 因子）

| 级 | 因子 |
|---|---|
| S（4） | yeslab_357、yeslab_471、yeslab_406、yeslab_390 |
| A（8） | yeslab_206、yeslab_464、yeslab_389、yeslab_405、yeslab_96、yeslab_375、yeslab_372、yeslab_384 |
| B（4） | yeslab_425、yeslab_381、yeslab_484、yeslab_398 |
| 443 修复变体（1） | yeslab_443_reg30_max20_v1（GP_074 reg30_max20，替换规范 443；决策点 D1=A） |

- 已执行的互斥约束：114↔484、357↔363、379↔375、370↔372、379↔384（363/379/370/114 因互斥不入）。
- 关注对同时准入、**训练前需单独组合评审**：381↔405、398↔381。
- 数据库契约：`dfs://QuantResearchV2Test.factor_set_frozen`（17 行，frozen_date=2026-09-03，evidence_commit=`a2103372…`）。
- 文件契约：`research_v2/artifacts/diagnostics/factor_set_frozen_v1.json`。

## 4. 物化进度（快照 @ 2026-09-04T01:48:27）

- 目标表：`dfs://QuantResearchV2Test.factor_daily`（迁移 016 新建；键：factor_id + signal_date + ts_code，含 data_version）。
- 总行数：**53,444,828**；重复键组：**0**（幂等机制保证；可用 `--check-dups` 全量复核）。
- 主进程：PID 3307676 运行中；日志 `research_v2/artifacts/diagnostics/factor_daily_materialization/full_run2.log`。
- 进度：2020/2021/2022/2023 各 **17/17 完成**；2024 = **15/17**（剩 484、96）；2025 = **0/17** 未开始。
- 快照文件（可随时重跑刷新）：同目录 `materialization_status.md` / `materialization_status.json`。

各因子 × 年份行数（0 表示尚未物化）：

| factor | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|
| yeslab_206 | 649,977 | 699,386 | 791,213 | 873,248 | 924,987 | 0 |
| yeslab_357 | 662,595 | 691,359 | 760,514 | 829,679 | 931,696 | 0 |
| yeslab_372 | 326,271 | 348,104 | 382,102 | 411,966 | 449,844 | 0 |
| yeslab_375 | 314,139 | 341,459 | 370,737 | 421,524 | 466,508 | 0 |
| yeslab_381 | 665,416 | 714,588 | 803,668 | 881,874 | 931,656 | 0 |
| yeslab_384 | 335,489 | 365,238 | 392,001 | 439,395 | 481,430 | 0 |
| yeslab_389 | 336,211 | 359,191 | 394,254 | 427,920 | 462,798 | 0 |
| yeslab_390 | 336,792 | 359,873 | 394,784 | 428,372 | 463,364 | 0 |
| yeslab_398 | 662,905 | 712,018 | 801,521 | 880,545 | 930,514 | 0 |
| yeslab_405 | 665,416 | 714,588 | 803,668 | 881,874 | 931,656 | 0 |
| yeslab_406 | 320,429 | 349,001 | 374,468 | 423,216 | 466,905 | 0 |
| yeslab_425 | 664,016 | 713,175 | 802,484 | 881,143 | 931,022 | 0 |
| yeslab_443_reg30_max20_v1 | 653,073 | 702,308 | 793,402 | 875,048 | 926,148 | 0 |
| yeslab_464 | 662,905 | 712,018 | 801,521 | 880,545 | 930,514 | 0 |
| yeslab_471 | 657,201 | 685,940 | 755,929 | 826,871 | 929,224 | 0 |
| yeslab_484 | 666,468 | 715,643 | 804,566 | 882,426 | 0 | 0 |
| yeslab_96 | 661,111 | 710,206 | 799,982 | 879,593 | 0 | 0 |

注：

- 物化器幂等（按 factor+signal_date+data_version 去重）；中断后重跑自动跳过已完成因子-年，不会写重。
- yeslab_471（AttentionSpillover）为 CPU 密集因子，是主循环中最慢的环节。
- 2022 曾因中途杀进程出现 9/17 缺写，已用补跑脚本修复至 17/17（见 `backfill_2022.log`），修复后全库 0 重复。

## 5. 训练口径（下一阶段用，源自模型训练交接文档）

- 特征：`factor_daily` 中 17 个冻结因子（隔离规范版本）。
- 标签：`label_daily` 的 `forward_open_5d:v1`（5 交易日前向复权收益，原始值为主）。
- 宇宙：官方 `universe_daily.cn_liquid_a`（直读）。
- 分割：2020-2024 训练 / 2025 验证 / 2026 锁定测试（严禁触碰）；按年滚动。
- 体制处理（决策点 D3=a）：「大盘股主导」体制 ON 时对 437/431/443 修复变体降权；先在验证段 OOS 证实有效再固化。
- 候选模型：Residual MLP、Set Attention、Temporal GRU/TCN；树模型（LightGBM 等）作对照基线。
- Walk-forward 基建：`scripts/run_real_walk_forward.py`（criterion-303，folds F01-F06）。

## 6. 关键证据文件（复查入口）

| 类别 | 路径 |
|---|---|
| 决策简报（4a-4aa 全证据） | `docs/handoffs/2026-09-01-admission-decision-brief.md` |
| 准入决策清单（D1-D4 已填） | `docs/handoffs/2026-09-03-admission-decision-sheet.md` |
| 模型训练交接 | `docs/handoffs/2026-09-03-model-training-handoff.md` |
| 因子质量阶段交接（早期） | `docs/handoffs/2026-08-19-factor-quality-handoff.md` |
| 冻结集契约 JSON | `research_v2/artifacts/diagnostics/factor_set_frozen_v1.json` |
| 物化快照（自动刷新） | `research_v2/artifacts/diagnostics/factor_daily_materialization/materialization_status.{md,json}` |
| 物化器（幂等） | `research_v2/scripts/materialize_frozen_factor_daily.py` |
| 状态刷新脚本 | `research_v2/scripts/check_materialization_status.py` |
| 因子计算逻辑（用户提供） | `factor_calculators/` |

## 7. 复核命令清单（复查者可直接重跑）

```bash
cd /data/zhangyudong/FDU/Quant-Platform-Prod_CODEX

# 1) 物化进程是否存活
pgrep -af materialize_frozen_factor_daily

# 2) 最新进度日志
tail -n 20 research_v2/artifacts/diagnostics/factor_daily_materialization/full_run2.log

# 3) 刷新物化快照（快；重复检查默认关闭，幂等已保证无重复）
conda activate quantyd
python research_v2/scripts/check_materialization_status.py
# 全量重复校验（慢）：
python research_v2/scripts/check_materialization_status.py --check-dups

# 4) DolphinDB 权威计数（xiahongjie@10.176.55.211:8848）
python - <<'PY'
import dolphindb as ddb
s = ddb.session(); s.connect("10.176.55.211", 8848, "xiahongjie", "123456")
db = "dfs://QuantResearchV2Test"
print(s.run(f"select count(*) as n from loadTable('{db}','factor_daily')"))
print(s.run(f"select count(*) as n from loadTable('{db}','factor_daily') group by factor_id, year(signal_date) as yr order by factor_id, yr"))
print(s.run(f"select factor_id, decision, grade from loadTable('{db}','factor_set_frozen') order by factor_id"))
s.close()
PY

# 5) 代码与文档版本
git log --oneline -8
git status --short
```

**物化完成验收标准**：2020–2025 每年因子数 = 17，重复键组 = 0，无残留物化进程；各因子 2025 行数应与 2024 同量级（受交易日数与宇宙大小影响，允许小幅差异）。

## 8. 待办（TODO，按顺序）

1. ⏳ 等待物化完成：2024 剩 484、96 → 2025 全部 17 因子。
2. 跑第 7 节命令 3 确认 6 年 × 17 因子、0 重复；抽查值完整性（无整列 NaN、无异常截断）。
3. 更新 `2026-09-03-model-training-handoff.md`：标记 `factor_daily` 就绪；提交并推送。
4. 关注对组合评审：381↔405、398↔381（两者已同时准入，训练前必须完成）。
5. 训练准备：按第 5 节口径组装特征 × 标签矩阵，核对 17 因子在 `cn_liquid_a` 上的覆盖率无缺口。
6. 按候选模型训练与对照；体制降权先在 2025 验证段验证再固化。

## 9. 风险与注意事项

- GitHub 网络不稳定：推送失败需重试（历史已多次出现）。
- 不要对同一年份并行启动第二个物化器：幂等机制能防重复，但会浪费 CPU 并可能争抢资源；471 属 CPU 密集型。
- 若主进程中断：直接重跑 `/tmp/launch_materialize_full.sh` 对应年份即可，已完成因子-年自动跳过（2022 补跑已验证该流程）。
- 训练启动后仍需保留准入监控要求（见决策清单 A 表「监控要求」列，如 96 的 2020 年度门槛监控、381/398/425 的关注对监控等）。
- 本文件为时点快照：物化进度类数字以 `materialization_status.md` 实时刷新结果为准。
