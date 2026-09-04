# Real-Data Candidate Model Training & Evaluation (M1→M2→M3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在标准 303 已闭合的真实数据六折基础上，按 E 计划契约训练并评价
M1（Residual MLP）→ M2（GRU/TCN）→ M3（Set Attention）复杂候选，产出
`WALK_FORWARD_PASSED` 判定所需的全部证据（G1–G6 + 相对 fold-matched 基线
ΔRankIC ≥ 0.005）。

**Architecture:** 复用 `run_real_walk_forward.py` 的数据集装配路径（版本快照 +
折缓存）构建候选训练驱动；基线侧先补一次「折匹配基线预测」工件落盘（criterion-303
的胜者基线在每折 TEST 上的预测），候选评价通过 `evaluate_model_walk_forward`
与 fold-matched 基线对齐计算 G1–G6 与 Δ。训练编排全部走已验收的
`research_v2.training`（sampler/torch_loop/artifacts/model_bundle）与
`research_v2.evaluation.model_evaluator`。

**Tech Stack:** Python 3.11、PyTorch 2.6、scikit-learn 无关、DolphinDB SDK 3.0、
pandas/pyarrow、pytest；GPU：4×RTX 4090（单卡即可跑 smoke，全量搜索建议 1–2 卡）。

**Spec:** [`2026-08-01-quant-research-v2-design.md`](../specs/2026-08-01-quant-research-v2-design.md)；
E 计划：[`2026-08-01-quant-research-v2-e-models-evaluation-plan.md`](2026-08-01-quant-research-v2-e-models-evaluation-plan.md)
（E7 预算、E8 评价、E8.4 判据）；基线证据：
`docs/reports/2026-09-04-real-walk-forward-criterion-303.json`。

## Global Constraints

- 执行顺序 M1 → M2 → M3 不可跳跃（E 计划第 16 行）；M4 不在本计划（E9 未授权）。
- 搜索预算：每家族 ≤30 configs × 3 seeds × F01–F06（E7/E9.6 同预算）。
- 门控：G1 mean RankIC ≥0.030、G2 HAC t ≥2.0、G3 正年占比 ≥0.70、
  G4 候选减 fold-matched 基线 ΔRankIC ≥0.005、G5 最大年正贡献占比 ≤0.50、
  G6 有效得分日占比 ≥0.95；任何缺 fold、缺 seed、score collapse、年度集中均失败。
- 训练只用 TRAIN；验证集只做选择；TEST 每折仅在冻结选择后评价一次；
  2026 FINAL locked 永不触碰（`DatasetBuildRequest` locked 检查兜底）。
- 因子集 = 执行时已冻结的准入因子集（当前 14 个：
  yeslab_376,377,380,381,387,388,391,396,400,405,425,432,433,442；
  若人工准入决策扩集，以冻结清单为准并在报告记录）。
- 已知契约张力：`configs/models/*.yaml` 的 `raw_factor_count_min: 15`
  高于当前 14 因子；模型代码硬约束为 1–60（合法）。若冻结集仍为 14，
  需先经批准把 yaml 下限改为 14，再进行训练（不在本计划内擅自修改）。
- 产物目录：`/data/zhangyudong/qrv2_research_artifacts/`（git 树外，可重跑复现）。

---

### Task 1: 折匹配基线预测工件（baseline OOS prediction artifacts）

**Files:**
- Modify: `research_v2/scripts/run_real_walk_forward.py`
- Test: `research_v2/tests/unit/scripts/test_run_real_walk_forward_report.py`

**Interfaces:**
- Consumes: `run_walk_forward_baselines` 返回的 `BaselineWalkForwardResult`
  （`.fold_results[fold_id].test_evaluation.predictions.frame`）。
- Produces: `save_baseline_oos_artifacts(result, root: Path) -> dict[str, str]`，
  每折写 `<root>/F0x.parquet`（columns: signal_date, ts_code, score, dataset_id），
  返回 `{fold_id: 文件路径}`；主流程在写报告 JSON 前调用，并把路径映射写入
  `report["baseline_oos_artifacts"]`。

- [ ] **1.1 写失败测试**（`test_run_real_walk_forward_report.py` 追加）：
  用既有 `_result()` 构造器生成 `BaselineWalkForwardResult`（需先给
  `BaselineFoldResult.test_evaluation` 造最小 `PredictionFrame`），
  断言 `save_baseline_oos_artifacts` 在 `tmp_path` 生成 6 个折文件且
  列集合为 `{signal_date, ts_code, score, dataset_id}`。
- [ ] **1.2 运行确认失败**：`pytest tests/unit/scripts/test_run_real_walk_forward_report.py -q` → FAIL（函数不存在）。
- [ ] **1.3 实现** `save_baseline_oos_artifacts`（纯函数：遍历
  `result.fold_results`，取 `test_evaluation.predictions.frame`，
  `to_parquet(index=False)`；缺失任一普通折即抛 `ValueError`）。
- [ ] **1.4 接入主流程**：`main()` 中 `report = build_walk_forward_report(...)`
  之后、写文件之前调用；`report["baseline_oos_artifacts"] = paths`。
- [ ] **1.5 测试通过 + ruff + `MYPYPATH=src mypy scripts/run_real_walk_forward.py`
  无新增错误**，提交。
- [ ] **1.6 重跑基线生成工件**（约 17 小时；可与后续任务并行启动）：

```bash
cd research_v2
setsid nohup bash -c '/data/zhangyudong/.conda/envs/quantyd/bin/python \
  scripts/run_real_walk_forward.py --folds F01,F02,F03,F04,F05,F06 \
  --report /data/zhangyudong/qrv2_research_artifacts/baseline_oos_report.json \
  > /data/zhangyudong/qrv2_research_artifacts/baseline_rerun.log 2>&1' &
```

Expected：报告 JSON 含 `baseline_oos_artifacts` 六个路径；数值与
`2026-09-04-real-walk-forward-criterion-303.json` 一致（同数据版本时
`all_passed`、`oos.*` 全等），否则停止并排查版本漂移。

---

### Task 2: 候选训练驱动骨架（真实数据装配 + 家族分发）

**Files:**
- Create: `research_v2/scripts/run_real_candidate_training.py`
- Test: `research_v2/tests/unit/scripts/test_run_real_candidate_training.py`

**Interfaces:**
- Consumes: Task 1 工件路径映射；`research_v2.datasets.builder.build_point_in_time_dataset`；
  `research_v2.training.{sampler,batches,torch_loop,artifacts,model_bundle}`；
  `research_v2.evaluation.model_walk_forward.evaluate_model_walk_forward`。
- Produces: `assemble_candidate_folds(holder, resolver, fold_ids, factor_members, versions)`
  → `dict[str, CandidateFoldDatasets]`（train/validation/test 三帧，列与基线驱动一致）；
  `load_baseline_predictions(paths: Mapping[str, str]) -> Mapping[str, pd.DataFrame]`。

- [ ] **2.1 失败测试**：CLI 参数契约（`--family {M1,M2,M3}`、`--smoke`、
  `--fold-subset`、`--baseline-artifacts <dir>`、`--report <path>`）；
  家族顺序守卫（`--family M2` 且 `--require-order` 时拒绝，除非提供
  `--prior-families-evidence`）；`load_baseline_predictions` 缺折即抛错。
- [ ] **2.2 运行确认失败** → **2.3 最小实现**（复用 `run_real_walk_forward.py`
  的 `_connect/_query/_current_versions/_extract_dataset/fold_cache_key`，
  抽为共享模块 `scripts/walk_forward_common.py` 供两个驱动 import，
  避免复制；基线驱动同步改为从该模块导入并全量回归）。
- [ ] **2.4 测试通过 + ruff/mypy** → **2.5 提交**。

---

### Task 3: M1（Residual MLP）真实数据训练与评价

**Files:**
- Modify: `research_v2/scripts/run_real_candidate_training.py`（M1 分支）
- Test: `research_v2/tests/unit/scripts/test_run_real_candidate_training.py`

**Interfaces:**
- Consumes: Task 2 装配的折数据集；`configs/models/residual_mlp.yaml` 搜索空间
  （3×3×3=27 configs ≤30 预算）；`REQUIRED_SEEDS` 三 seed。
- Produces: 每折选择记录 + 冻结 bundle + TEST 预测；
  `evaluate_model_walk_forward` 输入齐全后输出 `ModelWalkForwardResult`。

- [ ] **3.1 失败测试（合成 smoke）**：`--family M1 --smoke` 在合成小帧上走通
  搜索→选择→TEST 评价→报告 JSON（只验证编排，不作为业绩证据；参照
  `tests/integration/models/test_model_walk_forward.py` 的构造方式）。
- [ ] **3.2 实现 M1 分支**：日期均衡 sampler + `TorchDateTrainingLoop`，
  验证集早停选择，`CheckpointStore` + `ModelArtifactBundleStore` 落盘到
  `/data/zhangyudong/qrv2_research_artifacts/m1/`；TEST 预测仅在
  选择记录写入后生成（与基线驱动同一 fail-closed 顺序）。
- [ ] **3.3 报告生成**：`build_candidate_report(result, baseline_frames)` —
  G1–G6（含 G4 Δ）+ 相对 B0/B1/B2/B3 各自增量 + 参数量 + 训练时长 +
  最差 seed + 因子集与数据版本记录；判定 `WALK_FORWARD_PASSED` / `FAILED`。
- [ ] **3.4 全量运行**（1–2 卡，预计 6–24 小时）：

```bash
setsid nohup bash -c '/data/zhangyudong/.conda/envs/quantyd/bin/python \
  scripts/run_real_candidate_training.py --family M1 \
  --baseline-artifacts /data/zhangyudong/qrv2_research_artifacts/baseline_oos \
  --report /data/zhangyudong/qrv2_research_artifacts/m1_report.json \
  > /data/zhangyudong/qrv2_research_artifacts/m1.log 2>&1' &
```

- [ ] **3.5 判定归档**：通过则渲染判定文档到
  `docs/reports/<date>-m1-real-candidate-walk-forward.md`（复用判定器模式），
  更新主计划 10.2；失败则归档失败报告并分析（不得用 2026 数据补救）。

---

### Task 4: M2 → M3 顺序执行

**Files:** 同 Task 3（`--family M2` / `--family M3` 分支，时序家族走
`TemporalTrainingBatch` + history_length=20 契约，横截面家族走
`FeatureBatch`）。

- [ ] **4.1 M2 分支 + 合成 smoke**（GRU/TCN 双家族内选择，4×2×3×2=48 组合
  超出 30 预算 → 按 E7「受限搜索」先以验证集 RankIC 粗筛到 ≤30 再精评；
  粗筛规则写入报告）。
- [ ] **4.2 M2 全量运行 + 判定归档**（同 3.4/3.5）。
- [ ] **4.3 M3 分支 + 合成 smoke**（32 组合 ≤30 不成立时同 4.1 粗筛规则）。
- [ ] **4.4 M3 全量运行 + 判定归档**。

---

### Task 5: 10.2 汇总与交接

**Files:**
- Create: `docs/reports/<date>-real-candidate-models-comparison.md`
- Modify: `docs/superpowers/plans/2026-08-01-quant-research-v2-master-plan.md`（10.2）

- [ ] **5.1 汇总**：M1–M3 各自门控结果、相对基线与相对彼此增量、
  最優候选的年度分布与最差 seed；若无候选通过，明确记录「无复杂候选
  通过，简单基线保留」（主计划风险登记「基线过强」条款）。
- [ ] **5.2 更新主计划 10.2 勾选状态与日期注记** → 提交推送。
- [ ] **5.3 若有 `WALK_FORWARD_PASSED` 候选**：记录其进入影子审批的
  前置清单（F6.1 部署审批、60 有效日影子运行、监控与回滚演练），
  移交人工审批；不擅自启动 SHADOW。

---

## Self-Review 记录

- Spec 覆盖：E7 预算（Global Constraints + 3.1/4.1）、E8.1–E8.4（Task 2/3 接口与
  3.3 报告字段）、E8.5 合成 smoke（3.1/4.1/4.3）、E8.6 测试门（各任务测试步骤）、
  10.2 判据（Task 5）。
- 无占位符：所有步骤含命令或代码位置；M2/M3 为同驱动的家族分支执行任务，
  契约差异（时序批/粗筛）已分别写明。
- 类型一致性：`CandidateFoldDatasets`、`save_baseline_oos_artifacts`、
  `build_candidate_report` 在任务间签名一致。
- 已知决策依赖：因子集冻结（人工准入决策）与 `raw_factor_count_min` 调整
  （若仍 14 因子）为执行前置，已在 Global Constraints 标注。
