# 预测持久化只读审计（2026-10-08）

> 范围：只读检查 `/data/zhangyudong/qrv2_research_artifacts/`、`/tmp/es_metric_diag/`、当时研究工作树。
> 约束遵守：未启动任何训练；未写数据库、旧工件、训练规则；E9 锁定 2026 段未触碰；输出仅落在新目录与本证据分支。

## 0. 结论

| 目标 | 状态 |
|---|---|
| all15 三种子集成测试预测 | ✅ **PERSISTED_AND_VERIFIED**（三个家族 m0_v3 / m0r_v2 / m0r_v3；24/24 预测 SHA-256 与 manifest 一致；独立重算与归档**逐位一致**） |
| dense9 三种子集成测试预测 | ❌ **PREDICTIONS_NOT_PERSISTED**（仅有汇总表；无数据集、无模型、无预测） |
| `/tmp/es_metric_diag/`（消融工作目录） | ❌ **已不存在**（目录已清理，消融脚本随之丢失） |

**因此第 3 步「后处理导出四类输入」当前只能覆盖 3/4**：all15 候选（M0 与 M0R 两个载体）、best_simple 基线、test 标签均在；**dense9 缺失**，且无法由后处理得到——只能重跑消融（脚本已不在，需重建）。未自动重训，也未换用 M1/M2。

## 1. all15 持久化清单

数据集（4 折，15 特征 f00–f14；三家族共用同一 dataset_id）：

| 折 | 测试年 | dataset_id | 全 split 行数 | test parquet SHA-256 |
|---|---|---|---|---|
| F1 | 2022 | `226a8dfc497c2de6…` | 3,798,054 | `e9a98223dfa91017…` |
| F2 | 2023 | `eb9bba1b5a0bd66d…` | 4,205,787 | `4cd3d4b9bc27668f…` |
| F3 | 2024 | `9766e69f342511c5…` | 4,582,910 | `015c51ea13d5aec8…` |
| F4 | 2025 | `6f9478d7645c8c74…` | 4,893,642 | `f29f5c46b8a6f8ab…` |

模型与预测（每折 3 个种子 checkpoint + 验证集集成预测 + 测试集候选预测）：

| 家族 | 折 | model_id | 种子 | 候选预测 prediction_id | 候选预测 SHA-256 | selection_id |
|---|---|---|---|---|---|---|
| m0_v3 | F3 | m0-f3 | 20260801/02/03 | `07d663b75ef2746e…` | `54f6e7b8a0a3fa37…` | `5c7b013902a8…` |
| m0_v3 | F2 | m0-f2 | 20260801/02/03 | `0f2564f5947f4339…` | `d50fda23b8146d40…` | `2f43cb5e472a…` |
| m0_v3 | F4 | m0-f4 | 20260801/02/03 | `2f8aaeb70752ba2f…` | `7bc28bc2ec570b08…` | `305bc084e592…` |
| m0_v3 | F1 | m0-f1 | 20260801/02/03 | `3bbae89e20c12f87…` | `a4fd74abf9f2c646…` | `45d140f7fd1c…` |
| m0r_v2 | F4 | m0r-f4 | 20260801/02/03 | `8a870cadd1ed54b3…` | `b4ce8c6a0d4af2fc…` | `179e6821d305…` |
| m0r_v2 | F1 | m0r-f1 | 20260801/02/03 | `98dc1610117ab6e8…` | `b881c8bf5f1a6eb5…` | `1348cbd593c7…` |
| m0r_v2 | F3 | m0r-f3 | 20260801/02/03 | `b69fa46adb984d34…` | `a50b48d713389c57…` | `b824fdb6ae08…` |
| m0r_v2 | F2 | m0r-f2 | 20260801/02/03 | `ed752887b6614ae1…` | `aa811c62dad1a25a…` | `82a52d182f29…` |
| m0r_v3 | F4 | m0r-f4 | 20260801/02/03 | `8a870cadd1ed54b3…` | `b4ce8c6a0d4af2fc…` | `179e6821d305…` |
| m0r_v3 | F1 | m0r-f1 | 20260801/02/03 | `98dc1610117ab6e8…` | `b881c8bf5f1a6eb5…` | `1348cbd593c7…` |
| m0r_v3 | F3 | m0r-f3 | 20260801/02/03 | `b69fa46adb984d34…` | `a50b48d713389c57…` | `b824fdb6ae08…` |
| m0r_v3 | F2 | m0r-f2 | 20260801/02/03 | `ed752887b6614ae1…` | `aa811c62dad1a25a…` | `82a52d182f29…` |

基线（G4 标尺，best_simple，直接来自 baseline_v2）：

| 折 | 路径 | SHA-256 | 行数 |
|---|---|---|---|
| F1 | `/data/zhangyudong/qrv2_research_artifacts/baseline_v2/baseline_oos/F1.parquet` | `00a9a9db01fa7980…` | 747,461 |
| F2 | `/data/zhangyudong/qrv2_research_artifacts/baseline_v2/baseline_oos/F2.parquet` | `8722f9843ca2bab4…` | 820,829 |
| F3 | `/data/zhangyudong/qrv2_research_artifacts/baseline_v2/baseline_oos/F3.parquet` | `5bbc55291db103e6…` | 924,499 |
| F4 | `/data/zhangyudong/qrv2_research_artifacts/baseline_v2/baseline_oos/F4.parquet` | `74024a9e462ff1b5…` | 925,174 |

## 2. 完整性验证

- 预测产物：24/24（3 家族 × 8 个）`scores.parquet` 实际 SHA-256 == manifest 声明值。
- 模型产物：每家族 40 个文件，其中 36 个有 `checksums.sha256` 声明，**0 处不一致**。
- 三种子集成语义（代码 + 产物双证）：`REQUIRED_SEEDS = (20260801, 20260802, 20260803)`；发布路径 `_m0_seed_mean_scores(loaded, test, factor_order)` 对三 seed 的百分位分数按 `(signal_date, ts_code)` 取均值；模型目录内 `validation_predictions.parquet` 即 `build_seed_ensemble(...)` 的集成验证预测。

## 3. 独立重算：证明 all15 与消融输入同源（而非数值碰巧相等）

用**生产评测器** `evaluate_frozen_oos` 从归档预测 + 归档标签重算（报告数字不参与计算）：

| 载体 | pooled mean RankIC | pooled delta | HAC t | 有效日 | 与归档的绝对差 |
|---|---|---|---|---|---|
| M0 / LGBM_RANKER | 0.089429 | +0.004202 | 12.383 | 945 | mean 0.0e+00 / delta 0.0e+00 / HAC 0.0e+00 |
| M0R / LGBM_RANKREG | 0.094376 | +0.009149 | 12.905 | 945 | mean 0.0e+00 / delta 0.0e+00 / HAC 0.0e+00 |

逐折 delta（与消融文档 §3 的 all15 行对照）：

| 折（测试年） | m0_v3（M0，调参） | 文档 all15 调参行 | m0r_v2（M0R，单配置） |
|---|---|---|---|
| F1 (2022) | +0.00793 | +0.00793 | +0.01249 |
| F2 (2023) | -0.00270 | -0.00270 | +0.00319 |
| F3 (2024) | +0.00902 | +0.00902 | +0.01392 |
| F4 (2025) | +0.00192 | +0.00192 | +0.00645 |

- 池化配对行数：**3,417,963**（候选 × 基线交集，F1–F4 合计）——与消融文档自述分母完全一致，说明消融的 all15 行消费的就是这批归档预测。
- G6 口径复核：969 个有效标签日中 945 个候选可评分 = 0.9752，与文档 G6 一致。

## 4. dense9 状态：PREDICTIONS_NOT_PERSISTED

搜索范围与结果：

1. `qrv2_research_artifacts/**/datasets/*/manifest.json` 全量特征数扫描：**53 × 15 特征、1 × 14 特征、0 × 9 特征**——不存在 dense9 数据集存储。
2. 各家族 `models/`：无 dense9 模型 bundle（3 家族 × 4 折 × 3 种子，全部标注 `raw_factor_count: 15`）。
3. 各家族 `predictions/`：所有 candidate manifest 的 `dataset_id` 均指向 15 特征数据集。
4. `/tmp/es_metric_diag/`：**不存在**。
5. 主仓库 + 全部 worktree 内 `grep dense9`：仅 3 份 handoff 文档 + 1 处代码注释，**无消融驱动脚本**。

唯一存活证据：`docs/handoffs/2026-09-08-m0-attribution-and-carrier-findings.md` §3 汇总表（dense9 调参 +0.00347 / dense9 固定配置 +0.00177 / all15 固定配置 +0.00109 / sparse6 −0.02122）。

## 5. 机器可读交付物

本目录（小件，已入研究分支）：

- `audit_inventory.json`：文件级清单（路径 + SHA-256 + 声明比对）+ 重算结果 + dense9 搜索记录
- `verification_summary.json`：精简核对摘要（供程序消费）
- `daily_rank_ic/*.parquet`：候选与基线**逐日 IC 表**（969 行/族；含 `rank_ic`、`rank_ic_valid`、`coverage`、`prediction_unique_count`、`prediction_std`、`fold_id`）

本机大面板（未上传，路径即挂载点）：

- `/data/zhangyudong/qrv2_research_artifacts/prediction_persistence_audit_20261008/keys/<family>_<fold>_keys.parquet`：逐日股票键集合（`signal_date, ts_code, in_candidate, in_baseline, raw_return_5d, label_valid`；F1–F4 合计 3,557,762 键/族）
- `/data/zhangyudong/qrv2_research_artifacts/prediction_persistence_audit_20261008/audit_predictions.py`：本次审计脚本（只读，可复跑）

## 6. 环境与版本

| 项 | 值 |
|---|---|
| Python | 3.11.13（conda env `quantyd`） |
| LightGBM | 4.6.0 |
| pandas / numpy / pyarrow | 2.3.3 / 2.3.3 / 21.0.0 |
| 模型代码版本（产物自带） | `lgbm-ranker-1`（M0）、`lgbm-rankreg-1`（M0R） |
| 生产窗口内 HEAD | `bd021ff`（2026-09-09 04:56，两次独立 M0R 全量跑逐字节一致的记录提交） |

## 7. 本次未做的事

- 未启动训练、未重训、未换用 M1/M2。
- 未写数据库、未写旧工件目录（审计输出全部在新目录）。
- 未触碰 E9 锁定 2026 段（评测器 `role=TEST`，测试年仅 2022–2025）。
- 未修改生产分支；交付物落在独立研究分支。

## 8. 若要把 dense9 补齐，需要的最小动作

1. 重建消融脚本（9 稠密因子 = 357/471/206/464/96/425/381/484/443变体；从 15 特征数据集按列子集取样，其余流程与 M0/M0R 相同）。
2. 三种子（20260801/02/03）+ 同集成规则，逐折训练并在**同一 test 键集合**上产出候选预测。
3. 用同一生产评测器与 `baseline_v2/baseline_oos/*.parquet` 配对评估，产出与 all15 同格式日表 + 预测产物，落新目录（不改旧工件）。
4. 成本量级：与 M0R 全量相当（单配置 × 4 折 × 3 种子；分钟~小时级，远低于 M1 的 42.7 天）。
