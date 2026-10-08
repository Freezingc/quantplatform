# dense9 / M0R 最小重放（2026-10-08）

> 目标：补回遗失的 dense9 三种子集成测试预测，并给出 all15−dense9 的日级增量区间。
> 性质：**已知历史摘要后的探索性重放**，不是新方法搜索、不是预注册、也不是独立市场确认。
> 约束遵守：未改生产/数据库/旧模型与签名；未读 2026/E9；未跑 LambdaRank、M1/M2；未新增配置或种子；输出全部落在新目录。

## 0. 结论

- **dense9 已补齐**：4 折 × 3 种子 = **12 次拟合**，与原实验**逐位一致**——每折 trees 完全相同（[3, 14, 27] / [142, 158, 204] / [54, 32, 52] / [187, 133, 156]），pooled 的 cand/base/delta/icir/hac_t/valid_days/yearly/gates **全部相等**（cand `0.09405814016074272`）。
- **all15 − dense9 日级增量**（主口径，score-health 门控，945 日）：均值 **+0.00031782**，HAC(lag=5) t = 0.293，circular block bootstrap（折内 block=10，5000 次，seed=20260912）95% CI = **[-0.002110, +0.002223]**，**跨零**。
- 原点差 `0.00031782123177895305` 被精确复现（差 2.9e-17）。
- 读数：在更好的载体（rankreg）下，六个稀疏因子相对九个稠密因子的边际贡献**仍在噪声内**——与原 `_verdict` 的判定一致，但这是**重放确认**，不是新证据。

## 1. 对上一份审计的追加更正（不覆盖旧审计）

上一份 `2026-10-08-prediction-persistence-audit` 报告里有两处措辞需要追加纠正：

1. **"驱动脚本已丢失"不准确**：原 rank-regression 消融脚本与汇总**一直在 GitHub 上**，位于提交 `d89b46d9cb0da5ed97d2174323b347452049bf2d`（分支 `codex/research-conditional-factor-review`）：
   - `docs/research/evidence/2026-09-10-conditional-factor/experiment/run_rankreg_subsets.py`，blob `3d930399c36d7b208d10f3291206432d6f606dd0`，SHA-256 `41931618246abfe050437fafa93f896faba757615992ea9ab90b51790b8ee560`
   - `.../rankreg_subsets.json`，blob `faea5bc0adc62bbaec3bbb960978f537d4b47075`，SHA-256 `10618f69af351a1a81e8e2c8014b4d9ab864248d5a7093d8449c6b205ebcbe22`
   本目录 `source/` 内为逐字节副本，两个哈希均已复核（`source/SOURCES.sha256`）。
2. **"唯一存活证据只有 LambdaRank 表格"应修正为**：*预测与权重*确实未持久化（这一点成立），但 dense9 是**在该脚本内存中投影九列**得到的，缺少九列 dataset manifest 不能证明 dense9 从未运行；其数值证据（rankreg 的 all15/dense9/sparse6 三个 cell）在 `rankreg_subsets.json` 中完整存在，本次重放已把 dense9 预测补回。

## 2. 执行

| 项 | 值 |
|---|---|
| 运行目录 | `/data/zhangyudong/qrv2_research_artifacts/dense9_rankreg_replay_20261008` |
| 原脚本/JSON | SHA-256 逐字节核验通过（见 §1） |
| 重放驱动 | `replay_dense9_rankreg.py`，sha256 `0c362a806d956322…`；最小 diff `source/replay_driver.diff`（sha256 `3116385d3112eb27…`） |
| 改动面 | 仅输出持久化、路径参数、`SUBSETS={'dense9': DENSE9}`、移除跨子集 `_verdict`；**数据准备/训练/评分/评测逻辑未改** |
| 数据 | 原 `m0r_v2/report.json` 的四个 dataset_id；train/validation/test 分片哈希全部记录在 `replay_contract.json` |
| 列序 | `f00,f01,f04,f09,f10,f11,f12,f13,f14`（只投影九列，不物化新因子、不删行） |
| 模型/参数 | `LGBMRankRegBaseline`；seeds [20260801, 20260802, 20260803]；500 轮 / patience 50 / min_cs 30 / allow_nan=True；num_leaves=15,max_depth=4,lr=0.05,mcs=200,ff=0.85,l2=0.0,threads=16 |
| 环境 | Python 3.11.13, LightGBM 4.6.0, pandas/numpy 2.3.3/2.3.3, pyarrow 21.0.0 |
| 训练实现 | 与 source_commit 逐字节相同（`lgbm_rankreg.py` 仅一次提交 c43e038 且为其祖先）；每次 fit 新建 `lgb.Dataset`（模块内记录的确定性要求） |
| 资源与上限 | `/usr/bin/time -v` + `timeout --kill-after=60 7200`；实测 wall **11:49.68**，退出码 **0**，peak RSS **11,100,232 KB** |
| 首次拟合前 HEAD | `2502014df94a`（codex/publish-quant-platform） |

**停止规则遵守**：首折三种子完成后核对了预测键、九列列序与 trees（`trees=[3,14,27]` 与参照一致）；预测总耗时未越 2 小时上限，故按契约跑完全部 12 次拟合，未减种子、未改参数；trees 与原参照不一致时未做任何"调参对齐"。

## 3. 逐位复现证据

| 折 | trees（重放） | trees（原参照） | 一致 | paired delta（重放） | paired delta（原） | 一致 |
|---|---|---|---|---|---|---|
| F1 | [3, 14, 27] | [3, 14, 27] | ✓ | 0.010248951627842742 | 0.010248951627842742 | ✓ |
| F2 | [142, 158, 204] | [142, 158, 204] | ✓ | 0.001642643390710713 | 0.001642643390710713 | ✓ |
| F3 | [54, 32, 52] | [54, 32, 52] | ✓ | 0.01697752431167221 | 0.01697752431167221 | ✓ |
| F4 | [187, 133, 156] | [187, 133, 156] | ✓ | 0.005791179814624084 | 0.005791179814624084 | ✓ |

| pooled 字段 | 重放 | 原值 | 相等 |
|---|---|---|---|
| cand | 0.09405814016074272 | 0.09405814016074272 | ✓ |
| base | 0.0852268662737014 | 0.0852268662737014 | ✓ |
| delta | 0.00883127388704133 | 0.00883127388704133 | ✓ |
| icir | 0.771236977475296 | 0.771236977475296 | ✓ |
| hac_t | 12.96563007235339 | 12.96563007235339 | ✓ |
| valid_days | 945 | 945 | ✓ |
| all_passed | True | True | ✓ |
| yearly | （见 `fits/fit_summary.json`） | 同 | ✓ |
| gates | 同 | 同 | ✓ |

耗时对比：重放 701.7s vs 原 647.1s（同一量级；差值来自当前机器负载与保存产物）。

## 4. 增量区间

| 口径 | 有效日 | mean IC all15 | mean IC dense9 | mean diff | HAC t | 95% CI | 跨零 |
|---|---|---|---|---|---|---|---|
| 主口径：三方配对 + score-health 门控 | 945 | 0.09437596 | 0.09405814 | +0.00031782 | 0.293 | [-0.002110, +0.002223] | 是 |
| 辅助：三方配对（未门控） | 969 | 0.09659048 | 0.09633089 | +0.00025959 | 0.245 | [-0.002056, +0.002068] | 是 |
| 支持性旧口径：各自与基线配对（门控） | 945 | 0.09437596 | 0.09405814 | +0.00031782 | 0.293 | [-0.002110, +0.002223] | 是 |
| 支持性旧口径（未门控） | 969 | 0.09659048 | 0.09633089 | +0.00025959 | 0.245 | [-0.002056, +0.002068] | 是 |

- **两种口径在本例中完全一致**：the three-way and legacy calibers coincide here: all15 and dense9 share identical test key sets and the baseline is a strict subset, so every pairing yields the same rows
- 排除日：**24 天**（主口径下 score-health 不达标），逐日原因在 `analysis/increment_interval.json` 的 `invalid_or_excluded_days`。
- 年度明细：`analysis/increment_interval.json` 的 `yearly_mean_diff` / `yearly_mean_ic`。
- 描述性分辨率 ±0.002：本次点差（+0.00032）**小于分辨率**，且 CI 跨零——按契约，这不改变任何停止规则，也不构成"CAF 准入优于简单规则"或净收益的证据。

## 5. 交付物与服务器路径

本目录（小件，已入独立研究分支）：

- `replay_contract.json`（首次拟合前落盘的执行契约：哈希、折、参数、统计契约、E9 排除）
- `source/`：原脚本与 JSON 的逐字节副本、`SOURCES.sha256`、`replay_driver.diff`
- `replay_dense9_rankreg.py`（重放驱动）
- `fits/fit_summary.json`（12 次拟合的 trees/best_iteration/耗时/booster 与逐种子预测哈希）
- `analysis/`：`increment_interval.json`、`pairing_provenance.json`、`dense9_replay_results.json`、`daily_ic_three_way.csv`、`daily_ic_three_way_ungated.csv`、`paired_daily_head.csv`、`export_paired_daily.py`
- `logs/`：`replay_run.log`、`fits_progress.jsonl`（追加式）、`time_v.txt`、`inputs_<fold>_<role>.json`

留在服务器（大件，未上传）：

- `analysis/paired_daily.csv`（逐股票日配对表，**393 MB**；列：signal_date, ts_code, all15, dense9, baseline, raw_return_5d, label_valid, fold_id, test_year）
- `predictions/<fold>/scores.parquet`（dense9 每折集成预测）与 `fits/<fold>_seed<seed>/{booster.txt,test_percentile.parquet}`

## 6. 限制

- 本重放**不是**历史逐位复现的证明吗？恰恰相反：trees 与全部 pooled 字段都逐位相等，因此按契约这与"固定配置重放"相比是更强的结果；但驱动有持久化改动，故严格意义上仍是**固定配置重放（且数值逐位一致）**。
- 未做 all15 重拟合，也未用冻结实现重新打分；all15 直接复用已归档 M0R 预测（哈希见 §5 与 `pairing_provenance.json`）。
- 引用的 `2026-10-08-remote-pairing-extraction-task.md` **在本服务器上不存在**；三方契约按任务文本实现，推导已记录在 `replay_contract.json.open_items`。
- 结果不改变任何既有结论之外的东西：它只回答"dense9 预测能否补回、all15−dense9 日级增量多大"。
