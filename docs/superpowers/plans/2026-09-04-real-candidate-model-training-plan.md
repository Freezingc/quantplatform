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
- **规格偏差记录（种子数）**：设计规格 §训练规范要求「初筛至少 3 个种子，最终候选 5 个种子」（2026-08-01-quant-research-v2-design.md 第 469 行）；本计划统一采用 3 种子契约（`REQUIRED_SEEDS`），理由：种子集是训练→bundle→推理→监控的单一 fail-closed 契约（`model_bundle` 落盘校验、`inference/predictor` 与 `artifact_loader` 的集合相等检查、`score_health` 逐种子一致性均按 `REQUIRED_SEEDS` 硬校验），仅最终阶段改 5 种子将造成双层契约并波及 F 计划推理/监控路径，全链改 5 种子则使搜索预算增加约 67%，超出 E7 预算。该偏差为已决策项，待下一轮设计修订时一并裁决（扩契约或修订规格），本轮不擅自变更。
- **规格偏差记录（训练规范其余条款）**：对照设计规格 §训练规范（第 461-468 行）：
  1) 「warmup 加 cosine decay」未实现——驱动采用固定学习率（M1 默认 0.003，M2/M3 按配置网格）配合 validation RankIC 早停（patience 10）；
  2) 「日期按月度块打乱」实现为 `DateBalancedSampler` 的逐 epoch 确定性全日期哈希置换（保日期完整、日期等权，混洗强度高于月度块，意图保持）；
  3) 「混合精度」未启用——训练按已决策的 CPU fp32 路径执行。
  4) 损失权重固定为规格规范默认值（`LossWeights` ic=0.60 / pair=0.30 / huber=0.10，和为 1 硬校验；IC Loss 为分数与 return_rank 的 Pearson 相关——均与规格第 455-459 行一致），未做「少量离散候选由 Validation 选择」的权重搜索：E7 预算全部用于架构网格；未来若调权重必须走 Validation 离散候选机制（防 test 调参红线）；
  以上均为既有已测试契约（1089 测试覆盖），本轮运行前不变更；若候选未过 G1，学习率调度是下一轮经批准再评估的首选杠杆。
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

- [x] **1.1 写失败测试**（`test_run_real_walk_forward_report.py` 追加）：
  用既有 `_result()` 构造器生成 `BaselineWalkForwardResult`（需先给
  `BaselineFoldResult.test_evaluation` 造最小 `PredictionFrame`），
  断言 `save_baseline_oos_artifacts` 在 `tmp_path` 生成 6 个折文件且
  列集合为 `{signal_date, ts_code, score, dataset_id}`。
- [x] **1.2 运行确认失败**：`pytest tests/unit/scripts/test_run_real_walk_forward_report.py -q` → FAIL（函数不存在）。
- [x] **1.3 实现** `save_baseline_oos_artifacts`（纯函数：遍历
  `result.fold_results`，取 `test_evaluation.predictions.frame`，
  `to_parquet(index=False)`；缺失任一普通折即抛 `ValueError`）。
- [x] **1.4 接入主流程**：`main()` 中 `report = build_walk_forward_report(...)`
  之后、写文件之前调用；`report["baseline_oos_artifacts"] = paths`。
- [x] **1.5 测试通过 + ruff + `MYPYPATH=src mypy scripts/run_real_walk_forward.py`
  无新增错误**，提交。
- [x] **1.6 重跑基线生成工件**（约 17 小时；可与后续任务并行启动）：

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

- [x] **2.1 失败测试**：CLI 参数契约（`--family {M1,M2,M3}`、`--smoke`、
  `--fold-subset`、`--baseline-artifacts <dir>`、`--report <path>`）；
  家族顺序守卫（`--family M2` 且 `--require-order` 时拒绝，除非提供
  `--prior-families-evidence`）；`load_baseline_predictions` 缺折即抛错。
- [x] **2.2 运行确认失败** → **2.3 最小实现**（复用 `run_real_walk_forward.py`
  的 `_connect/_query/_current_versions/_extract_dataset/fold_cache_key`，
  抽为共享模块 `scripts/walk_forward_common.py` 供两个驱动 import，
  避免复制；基线驱动同步改为从该模块导入并全量回归）。
- [x] **2.4 测试通过 + ruff/mypy** → **2.5 提交**。

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

- [x] **3.1 失败测试（合成 smoke）**：`--family M1 --smoke` 在合成小帧上走通
  搜索→选择→TEST 评价→报告 JSON（只验证编排，不作为业绩证据；参照
  `tests/integration/models/test_model_walk_forward.py` 的构造方式）。
- [x] **3.2 实现 M1 分支**：日期均衡 sampler + `TorchDateTrainingLoop`，
  验证集早停选择，`CheckpointStore` + `ModelArtifactBundleStore` 落盘到
  `/data/zhangyudong/qrv2_research_artifacts/m1/`；TEST 预测仅在
  选择记录写入后生成（与基线驱动同一 fail-closed 顺序）。
- [x] **3.3 报告生成**：`build_candidate_report(result, baseline_frames)` —
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

      **实测吞吐与本项估算的偏差（2026-09-05 18:17，M1 全量运行中只读观测，未变更任何运行参数）**：
      本项估算「1–2 卡，6–24 小时」以 GPU 为前提，但驱动在 4 处硬编码
      `device=torch.device("cpu")`（`run_real_candidate_training.py:382/738/1115/1527`），
      实际按 Global Constraints 已裁决的「CPU fp32」路径执行 → 估算前提与实现口径不一致。
      拟合总量：每折 27 configs × 3 seeds（`REQUIRED_SEEDS`）+ 3 次最终拟合 = 84，
      × F01–F06 = **504 次拟合**（M1 单家族）。`--max-epochs` 默认 100（启动脚本未覆盖），
      早停 patience=10 → 每次拟合**至少** 11 个 epoch。
      实测（F01/config0/seed 20260801）：17:35:17 启动 → 17:48:17 落 epoch-1 检查点
      （含数据装配与 batch 构建）；至 18:17:21 无第二个检查点（epoch 2 起未再改进），
      故稳态单 epoch > 2.65 分钟 → 单次拟合 ≥ 29 分钟 → **M1 ≥ 504 × 29 min ≈ 244 小时 ≈ 10 天（下界）**；
      若部分 config 在更晚 epoch 才改进，实际更长。M2/M3（GRU/TCN，CPU 上更贵）不低于此量级。
      **实测精确化（2026-09-05 18:35，检查点时间戳直接测量，取代上面的下界推算）**：
      F01/config0/seed 20260801 的 epoch-1 检查点 17:48:17、epoch-8 检查点 18:35:39
      （validation RankIC 0.1029 → 0.1057，epoch 8 再次改进 → 早停计时重置，该拟合至少跑到 epoch 18）
      → **稳态单 epoch = 47.4 min / 7 = 6.77 分钟**。据此：
      单次拟合（best@8 → 停于 epoch 18）≈ 18 × 6.77 min ≈ **2.03 小时**；
      M1 全家族 504 次拟合 ≈ **42.6 天**；若 config 在更晚 epoch 改进（停于 25/30）则为 **59/71 天**。
      注意 config0 = `hidden_dim 64 / num_blocks 1 / dropout 0.0` 是 27 格中**最小**的一格，
      故 6.77 min/epoch 是网格均值的**下界**；`hidden 256 / blocks 3` 显著更贵。
      结论：本项「6–24 小时」估算与实测相差约 **40–70 倍**，M1→M3 全链在 CPU 上是**数月**量级，
      而非数天。该结论同时适用于冻结集链（同一驱动、同一 CPU 口径、拟合次数同量级）。
      资源现状：4×RTX 4090 利用率 0%；104 核中本进程仅实现 ~4.3 核有效占用
      （进程 232 线程、torch 默认 intra-op 52 线程——逐日小 batch 下线程同步开销主导，
      非算力饱和）。
      **设备/线程实测基准（2026-09-05 19:01，独立进程只读探针 `/tmp/bench_device.py`，
      复用驱动自身 `_prepare_frame`/`build_static_training_batches`/`_m1_model_factory` 与
      `run_validation_governed_training`，跑 1 个 epoch；数据取 M1 已发布的 F01 数据集
      `split=train/year=2014`（395,643 行 / 245 日）+ `validation/year=2019` 子集
      （152,921 行 / 60 日）；模型 = config0 `hidden 64/blocks 1/dropout 0.0`，10,369 参数）**：

      | 设备 / 线程 | 单 epoch 秒 | ms/train 日 | checkpoint sha256（前 8 位） |
      |---|---|---|---|
      | CPU threads=1 | 81.76 | 333.7 | `1b1f74b7` |
      | CPU threads=4 | 81.80 | 333.9 | `d72bc345` |
      | CPU threads=8 | 81.56 | 332.9 | `38f78b86` |
      | **CUDA:1（RTX 4090）** | **238.62** | **973.9** | `9735f01b` |

      交叉校验：在跑的 M1 稳态 epoch = 6.77 min = 406 s / ~1220 train 日 = **333 ms/日**，
      与 CPU 基准逐位吻合（两个测量互相印证）。四条结论：
      1) **GPU 比 CPU 慢 2.9 倍**——10,369 参数 + 逐日 ~1600 行 batch 属**派发/启动开销受限**，
         不是 FLOP 受限；「启用 4 张 4090」对 M1 是**反向优化**，不能作为提速杠杆。
      2) **线程数对速度中性**（1/4/8 线程 81.56–81.80 s，差异 <0.3%）→ 单次拟合无法靠多线程加速，
         但**可多路并发、每路 1 线程**（104 核主机）。并发扩展性已实测标定（solo/16 路/48 路
         同一工作量对照，`OMP_NUM_THREADS=1`、独立 checkpoint 目录、0 失败）：
         单次拟合降级 **16 路 1.14×、48 路 1.30×** → 有效吞吐 **16 路 14.0×、48 路 37.0×**；
         按降级趋势外推 ~96 路约 1.5× 降级 → 有效约 **64×**。
         据此 M1 的 504 次拟合 ≈ 504 × 2.12 h / 37 ≈ **29 小时**（48 路），对比串行 44.6 天。
         **并发上限的约束是内存而非核数**：单进程 RSS ≈ 3.4 GB（一折全帧），worker 约需 3 GB，
         473 GB 可用 → 内存约容 150 路、核数约 100 路，故 **48–64 路为稳妥工作点**
         （48 路全帧约 145 GB）。
         前提是并发化必须保持并列（tie）时的 canonical 选择顺序不变（E 计划已有
         `sha256(model_family|fold|config_json)` 排序约定可复用）。
      3) **线程数改变数值结果**：同 seed/config/数据在 threads=1/4/8 下产生 **3 个不同的
         checkpoint sha256**。而 torch 训练路径**既不钉住也不记录**线程数
         （全仓仅 `baselines/lgbm_ranker.py:374` 钉了 `num_threads: 1`；
         `training/seeds.py::configure_reproducibility` 记录 seed/torch_version/cuda 设备名，
         不含线程数；checkpoint manifest 亦不含）→ **候选模型数值当前依赖一个未钉住、
         未记录的主机属性**（`torch.get_num_threads()` 在本机 = 核数/2 = 52）。
         这是 E3.5/主计划 §4.3 可复现性契约的实质缺口：同一命令在不同核数主机上会给出不同结果。
         任何并发化改造**必须同时显式钉住线程数并写入证据**，否则会把该缺口放大 100 倍。
      4) **分家族基准已补齐（2026-09-05 19:38，threads=1，同一探针方法）**——GPU 结论**依家族相反**：
         M1 最小格（10,369 参数）CPU 333.7 vs CUDA 973.9 ms/日（慢 2.92×）；
         M1 最大格（403,969 参数）388.2 vs 777.5（慢 2.00×）；
         M3 SetAttention 最大格（418,177 参数）430.6 vs 899.3（慢 2.09×）；
         M2 TemporalGRU（60,801 参数）1301.3 vs 828.7（**快 1.57×**）；
         M2 TemporalTCN（54,913 参数）1518.7 vs 866.9（**快 1.75×**）。
         **决定性推论**：M1 参数量增 **39 倍**（10,369→403,969），CPU 每日成本只增 **1.16 倍**
         （333.7→388.2 ms）→ 成本几乎全是**逐日派发/Python 开销**，与模型 FLOPs 基本无关；
         故「换更快设备」不是杠杆，**并发才是**。只有时序家族（对 20 日历史做循环/卷积）在 GPU 获益。
         另：每个家族 CPU 与 CUDA 的 checkpoint sha256 **均不同**（M1 最大格
         `42a20ea5d079e7c8` vs `bd5b7628a7f188f0`；M3 `0b03c21dc85b083a` vs `539859815385cec2`；
         M2 GRU `b786aa50a9fd3f34` vs `64a35584e8ef76ac`；M2 TCN `2159a6e5ecd31faf` vs `1684f87eebbd8e43`）
         → **device 与线程数同属影响数值的参数，必须一并钉住并写入证据**。
      5) **全链串行成本推算**（F01 train ~1220 日、每拟合最少 18 epoch）：
         M1 = 504 拟合 × 2.04 h ≈ **42.7 天**；M2 = 450 拟合 × 8.60 h ≈ **161.3 天**；
         M3 = 306 拟合 × 2.63 h ≈ **33.5 天**；合计 **1260 拟合 ≈ 5,700 h ≈ 238 天**。
         自校验：推算的 M1 单 epoch 6.8 min 与在跑 M1 实测 6.77 min 吻合；且为**下界**
         （F02–F06 的 train 窗口年份更晚、行数比 F01 的 2014–2018 多约 40%）。
         并发化后（按实测降级标定，非零降级假设）：48 路 ≈ **6.5 天**、外推 ~96 路 ≈ **3.7 天**。
         单拟合时长已由在跑 M1 实测确认：fit#1 的 epoch-1 检查点（17:48:17）到 seed 20260802 的
         epoch-1 检查点（19:55:40）相隔 127.4 min = fit#1 的 epochs 2–18 加 fit#2 的 epoch 1
         = 18 个 epoch 当量 → **单次拟合 2.12 h**（推算值 2.03 h，偏差 4%）；据此
         M1 = 504 × 2.12 h = **1,070 h ≈ 44.6 天**、全链 **≈ 5,745 h ≈ 239 天**。
         注：6.77 min/epoch 取自无背景负载窗口，127.4 min 窗口后半段叠加了本线程自己的
         dry-run 与基准（约 0.7–1.7 核），故 2.12 h 略偏保守，诚实区间为 **2.03–2.12 h/拟合**。
         并发扩展性证据：`/data/zhangyudong/qrv2_research_artifacts/concurrency_scaling_benchmark.txt`。
      证据全文：`/data/zhangyudong/qrv2_research_artifacts/device_thread_benchmark.txt`（M1 线程/设备）、
      `/data/zhangyudong/qrv2_research_artifacts/family_device_benchmark.txt`（分家族矩阵与全链推算）；
      决策简报：`docs/handoffs/2026-09-05-compute-and-coverage-decision-brief.md`。
      附带运维风险：config 搜索循环**无折/配置级续跑**（`_existing_evidence_id` 仅在
      baseline/model 证据写入处去重，不跳过搜索），任何中断都从 F01/config0/seed1 重来；
      在数十小时的运行长度下该风险被显著放大。
      处置：**未做任何变更**（不改 device、不改线程数、不改 max_epochs、不打断运行中的链）——
      以上任一项都会改变数值口径或已裁决偏差，属人工裁决项；本节仅记录实测事实供裁决。

- [ ] **3.5 判定归档**：通过则渲染判定文档到
  `docs/reports/<date>-m1-real-candidate-walk-forward.md`（复用判定器模式），
  更新主计划 10.2；失败则归档失败报告并分析（不得用 2026 数据补救）。

---

### Task 4: M2 → M3 顺序执行

**Files:** 同 Task 3（`--family M2` / `--family M3` 分支，时序家族走
`TemporalTrainingBatch` + history_length=20 契约，横截面家族走
`FeatureBatch`）。

- [x] **4.1 M2 分支 + 合成 smoke**（GRU/TCN 双家族内选择，4×2×3×2=48 组合
  超出 30 预算 → 按 E7「受限搜索」先以验证集 RankIC 粗筛到 ≤30 再精评；
  粗筛规则写入报告）。
- [ ] **4.2 M2 全量运行 + 判定归档**（同 3.4/3.5）。
- [x] **4.3 M3 分支 + 合成 smoke**（32 组合 ≤30 不成立时同 4.1 粗筛规则）。
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

## 勾选审计与实测偏差记录（2026-09-06 00:30）

本计划 21 个复选框此前**全部未勾选**，而 Task 1–3.3 与 4.1/4.3 的交付物其实早已存在并在
生产使用（M1 全量正在运行）。本次逐项对照当前仓库状态核验后勾选 14 项，并记录三处
**实现与计划文本的偏差**——偏差本身不是缺陷，但计划文本若不更正就会误导后续执行。

**勾选依据**：
- 1.1/1.2：`tests/unit/scripts/test_run_real_walk_forward_report.py`（6 测试），含
  `test_save_baseline_oos_artifacts_writes_per_fold_parquet:167`、
  `test_save_baseline_oos_artifacts_requires_all_folds:183`（后者固化「必须六折齐全」）。
- 1.3：`scripts/run_real_walk_forward.py:51 def save_baseline_oos_artifacts`。
- 1.4：同文件 `main()` 中 `:194` 调用，工件根为 `args.report.parent / "baseline_oos"`。
- 1.5：上述测试与 `test_run_real_candidate_training.py` 合计 **30 passed**；ruff 清；
  mypy 仅剩 `walk_forward_common.py:20` 既有 dolphindb stub 报错。
- 1.6：`/data/zhangyudong/qrv2_research_artifacts/baseline_oos/F01..F06.parquet` 与
  `baseline_oos_report.json` 均在；`/tmp/baseline_rerun_verification.txt` 首行
  `CONSISTENT: re-run reproduces the criterion-303 evidence exactly`；
  `/tmp/baseline_schema_gate.txt` 首行 `SCHEMA PASS`。
- 2.1–2.5：`run_real_candidate_training.py::parse_args` 含 `--family/--smoke/--report/
  --baseline-artifacts/--baseline-report/--workspace/--max-configs/--max-epochs/
  --prior-report`；`tests/unit/scripts/test_run_real_candidate_training.py` 覆盖 smoke 路径。
- 3.1：`train_m1_smoke`（`:447`）。3.2：`run_full_m1`（`:1281`）→ `_run_full_cross_section`
  （`:952`），搜索空间 `m1_config_grid`（`:814`，3×3×3 = 27 ≤ E7 预算 30）。
- 4.1：`run_full_m2`（`:1349`）+ `m2_config_grid`（`:1305`）+ `train_m2_smoke`（`:652`）。
- 4.3：`run_full_m3`（`:1293`）+ `m3_config_grid`（`:1257`）+ `train_m3_smoke`（`:465`）。

**偏差 1（3.3）**：计划要求 `build_candidate_report(result, baseline_frames)`，
实现中**没有这个具名函数**——报告在 `_run_full_cross_section` 的返回字典里内联构造。
字段与计划一致（`family`/`mode`/`folds`/`oos`，其中 `oos` 含 `valid_day_count`、
`mean_rank_ic`、`icir`、`hac_t_statistic`、`baseline_mean_rank_ic`、`delta_rank_ic`、
`yearly_mean_rank_ic`、`gates`）。**本计划 Self-Review 中「`build_candidate_report`
在任务间签名一致」一句因此不成立，应予更正。**

**偏差 2（4.1）**：计划文本写「4×2×3×2=48 组合」，实现的 `m2_config_grid` 是
family(2) × hidden_dim(2) × dropout(3) × learning_rate(2) = **24 组合**（其 docstring
明写「24 <= 30」）。48 会超出 E7 的 ≤30 预算，故 24 是符合预算的正确实现，
**计划文本的算式有误**。

**偏差 3（4.3）**：计划文本写「32 组合 ≤30 不成立时同 4.1 粗筛规则」，实现的
`m3_config_grid` 固定 `dropout=0.1`、`learning_rate=0.0003`，只搜索
hidden_dim(2) × num_heads(2) × inducing_points(2) × num_blocks(2) = **16 组合**，
其 docstring 已说明这是为符合 E7 预算的既定缩减。

**偏差 4（3.4，已另行详记）**：计划写「1–2 卡，预计 6–24 小时」，实测驱动硬编码
`device=cpu`、稳态 **6.77 min/epoch**、**2.03–2.12 h/拟合**、M1 共 **504 次拟合**
→ **42.7–44.6 天**；M1→M3 全链 **≈239 天（下界）**。GPU 经 5 个格点实测对 M1/M3
**反而更慢**（2.0–2.9 倍），仅对 M2 快 1.57–1.75 倍；线程数对速度中性但改变数值。
详见 Task 3.4 的实测记录与 `docs/handoffs/2026-09-05-compute-and-coverage-decision-brief.md`。

仍开放：**3.4**（M1 全量运行**已于 2026-09-06 01:34:05 崩溃**，见下）、**3.5 / 4.2 / 4.4 /
5.1 / 5.2 / 5.3**。

**3.4 的 M1 全量运行崩溃与根因修复（2026-09-06 02:35，commit `0cbbc88`）**
进程 1137022 于 17:35 启动、**01:34:05 退出（7h58m）**，watcher 写 `/tmp/m1_full_failed_at.txt`；
错误 `ValueError: seed predictions contain duplicate business keys`（`training/ensemble.py:56`）。
先前记录的「13 个 checkpoint = 13/504 次拟合」是**误读**：13 个 checkpoint 全部属于
**F01 config0 的第一次拟合**（seed 20260801 epoch 1/8；20260802 epoch 1/2/3/5/6；
20260803 epoch 1/2/7/8/10/11），即三个 seed 均已早停完成，随后在**合并阶段**崩溃。
**根因**：驱动内 **6 处**直接遍历 `batch.inputs.ts_codes[0]` 而**无视 `stock_mask`**；
`build_static_training_batches` 把窄于「该段最宽横截面」的每个日期用**空证券码**补齐
（`codes = tuple(...) + ("",) * (width - count)`），真实段参差（F01/2019 validation 每日
2,366–2,469 只），故每个较窄日吐出多行 `ts_code=""` → `(signal_date,"")` 重复。
`FeatureBatch.__post_init__` 本身就强制该 padding 契约（有效位唯一非空、padding 位必须空串），
是**消费方未遵守**。smoke 夹具每日股票数恒定、从不 padding，故既有测试测不出来。
**更严重的一半**：`_percentile_rank_series` 在 `width`（含 padding）而非有效数上排名，
`width - count` 逐日不同 → 百分位被逐日变化的量扭曲；这些百分位正是**选 config 的
validation RankIC** 的输入，且 6 处中有 2 处产出**喂给 G1–G6 的冻结 TEST 预测**。
**修复**：6 处全部改为委托 `predictions_to_frame`（`models/base.py`，inference 预测器与
`training/validation.py` 已在用的同一契约：只保留 `stock_mask` 位、拒绝有效位非有限分数、
日内 `rank(method="average", pct=True)`）；多 seed 平均改为**按业务键 join** 且
`validate="one_to_one"`（该函数按分数排序，两个 seed 行序一般不同，位置相加会混股票）；
`_percentile_rank_series` 无调用者，删除。新增 3 项**参差夹具**回归测试（两日 40/31 只、
9 个 padding 槽；断言 padding 被丢弃、排名在日内有效股票上、seed 平均按键对齐而非按位置）。
**验证**：全量 **1190 passed / 45 skipped**（原 1178/45，+12 = 本项 3 + 标定工具 9）；
`ruff check src tests` 清（并修掉 commit `05d04fa` 遗留的 F401，见 `4022727`）、`mypy src` 155 文件清。
**未触碰 `run_real_walk_forward.py`**，故 Task 2.4 的逐位复现核验不受影响。
**对 3.4 与裁决 2/3 的影响**：这 8 小时产物不可用（未签发报告、未产出 evidence），
且**修复前重启必然同样崩溃**；修复对冻结集候选链（Task 5.x）同样必需，故非沉没成本。

**同类缺陷排查（2026-09-06 03:50）：该缺陷类已根除，无第二处实例。**
逐个核对所有按位置索引 `batch.inputs.ts_codes` 的消费方：
`training/validation.py:70→77`（位置取自 `nonzero(stock_mask)`）、
`training/losses.py:66→80`（同）、`models/base.py:predictions_to_frame`（同）——
**三者都遵守 mask**；`inference/builder.py:100/103、263/266` 构造的是全 1 mask（该路径不 padding）；
修复后的驱动 6 处全部委托 `predictions_to_frame`。全仓已无第二处手搓 `argsort().argsort()` 排名
（唯一的 `np.argsort` 在 `monitoring/score_health.py:235`，作用于已过滤的 `finite_scores`，与 padding 无关）。
`temporal_batches.py:157` 用同一套 `+ ("",) * (width - count)` padding 契约，其消费方现同样委托库函数。

**3.4 重启前必须先处理的一件事（2026-09-06 21:15 只读核实，未改任何代码或工件）：
照原命令重启会在第一次拟合的第一个检查点上直接崩，崩在那 8 小时自己的产物上。**

三件事各自成立、合起来是硬阻塞：

1. **检查点 id 是完全确定的，不含任何运行标识。**
   `run_real_candidate_training.py:1032` 是
   `checkpoint_prefix=f"{family.lower()}-{fold_id.lower()}-c{config_index}-{seed}"`，
   `training/runner.py:62` 再拼 `checkpoint_id = f"{checkpoint_prefix}-seed-{seed}-epoch-{epoch}"`。
   **无时间戳、无 run_id、无内容哈希**，故重启后 F01/config0/seed 20260801 的第一个检查点
   **必然**仍是 `m1-f01-c0-20260801-seed-20260801-epoch-1`。
2. **`CheckpointStore.save` 拒绝覆盖。**
   `training/artifacts.py:44-45`：`if destination.exists(): raise FileExistsError(
   "checkpoint is immutable and already exists: ...")`。
   **这是正确的不可变性契约**，问题不在它。
3. **那 13 个目录还在。** `/data/zhangyudong/qrv2_research_artifacts/m1/checkpoints/` 下
   三个 seed 的 `epoch-1` 都存在（epoch 1 必然是一次改进，故必然被写）。

**故重启到同一 `--workspace` 会在第一次拟合就 `FileExistsError`**——
而且是在 8 小时之前那批**数值已作废**的检查点上。

**那批检查点在两个独立维度上都是陈旧的，且没有任何字段能自动发现这一点**：

- **数值陈旧**：它们产于 commit `0cbbc88` 修复之前，其 `validation_rank_ic`
  （如 `m1-f01-c0-20260801-seed-20260801-epoch-8` 存的 **0.10573690469159619**）
  正是该 commit 自述里"在 `width`（含 padding）而非有效数上排名"的**被扭曲的百分位**算出来的。
  **而这个数就是挑选获胜 config 的信号**，故从它续跑等于从一个错的选择信号续跑。
- **schema 陈旧**：盘上 manifest 的 `metadata` 只有 **6 个字段**
  （`epoch`、`seed`、`torch_version`、`train_dataset_id`、`validation_dataset_id`、
  `validation_rank_ic`），而 `training/runner.py:60-77` 现在**总是写 9 个**——
  多出 `torch_intraop_threads`、`torch_interop_threads`、`model_device`，
  由 commit `d708864`（**09-05 21:34**）与 `7f2e007`（**09-05 21:40**）加入，
  **即在那次运行期间（17:35 → 01:34）加入**，故这批检查点是 21:34 之前的 runner 写的。
- **manifest 里没有任何驱动/契约版本字段**，故**没有自动检查能识别它们陈旧**；
  上面两条都是本轮人工比对代码与盘上工件得出的。

**处置（两个都可以，第二个在实质上更该做）**：
(i) 把 `--workspace` 指向一个新目录（计划的启动命令**没有传 `--workspace`**，
而 `parser.add_argument("--workspace", type=Path, default=None)`，
故实际路径由启动脚本给定，重启时显式换一个即可）；
(ii) 移走或删除那 13 个目录。**倾向 (ii)**：它们数值已作废，留着只会在将来某次
"看起来能续跑"的时候被误用；**且 (i) 不解决根因**——
下一次出现 `0cbbc88` 这类修复时，同样的陷阱会在新目录里重新长出来。

**由此得到一条对裁决 2 (a) 的"续跑粒度"改造的硬要求（这是从证据推出的，不是设计偏好，
故记录它不预判裁决 2）**：
> **续跑的键必须包含驱动/契约版本**（例如驱动文件的 sha256，或一个显式的
> `scoring_contract_version`），否则一次像 `0cbbc88` 那样**改变数值语义**的修复，
> 会让同一个 artifact root 里**静默混入两种数值口径的检查点**，
> 而不可变性契约会阻止新的覆盖旧的、manifest 又无从分辨。
> **本轮的 13 个目录就是这个缺口的实例**，只是它们碰巧崩在了合并阶段，
> 所以还没有机会与修复后的检查点混在一起。

**这条与裁决 2 的四个选项无关**：(a)/(b)/(c)/(d) 都要重启，重启都会撞上它。
**故它不是决定门住的工作，而是重启前必须做的一步。**
**本轮只核实、只记录，未删除任何工件、未改动 `--workspace` 默认值、未改代码。**
**生产路径本来就是正确范例**：`inference/predictor.py` 正是 `predictions_to_frame` +
`combine_seed_percentile_ranks`（按键 merge、`validate="one_to_one"`），
即本次修复是让候选驱动**向项目既有契约收敛**，而非另造一套。
**一处并存的第二套实现，经核对数值等价、故不重构**：驱动的 `_validation_rank_ic` 与库的
`evaluate_validation_batches` 都算「验证集 RankIC」。差异是前者用 `percentile_rank`、后者用
`raw_score`；但 `percentile_rank` 是日内 `rank(method="average", pct=True)`，为 `raw_score` 的
**保序且保并列**变换，故 Spearman 逐日相同。标签来源亦等价：库取批次内 `target_raw`
（已过 `label_valid` 与有限值过滤），驱动取原验证帧并按键 inner join，而批次键集正是该帧的
有效标签子集，故 join 后键集一致。`minimum_cross_section` 两侧均为 30。
**不重构的理由**：它会在可能重启 M1 之前改动 config 选择路径的来源，而等价性论证已成立；
记录在此以便后续如需统一时有据可依。
勾选仅改动 `- [ ]` → `- [x]`，已用 diff 核验**除复选框外零字符变化**。

## 勾选审计与实测偏差记录（2026-09-07 16:00）：新增 M0（LightGBM LambdaRank）候选家族

**起因（owner 指示，2026-09-07）**：因子端初步冻结约 15 个高质量因子（最终名单待 owner 通知）。
鉴于因子数量有限，**首版候选不采用过于复杂的深度学习模型，改用机器学习模型**；
但执行仍遵循本计划的纪律（TDD、smoke、G1–G6、工件与证据链）。
**这是对本计划「M1 Residual MLP 先行」顺序的一次 owner 授权的偏离**：
新增家族 **M0**（无前置，`FAMILY_ORDER_PREREQUISITES["M0"]=()`），M1→M2→M3 的既有顺序不变。

**实现（复用既有接缝，不另造一套）**：

- 模型直接复用**已评审的 B3 基线** `baselines/lgbm_ranker.py` 的 `LGBMRankerBaseline`
  （LambdaRank、按交易日分组、label_gain 0–4）——**候选与基线用同一个训练实现**，
  差别只在候选走「逐折受限搜索 × 3 种子集成 × bundle × 签名证据链 × 冻结预测 × G1–G6」。
- 基线模块**最小扩展**（`LGBMHyperparameters` 增加 `feature_fraction/bagging_fraction/bagging_freq`，
  `fit()` 增加可选 `seed`；`_training_parameters` 仅在非默认时追加采样键）：
  **B3 的参数逐位不变**（基线 7 测试全过）。新增 2 条基线测试：
  同种子复现确定性、feature_fraction<1 时三种子确实发散（**否则三种子契约形同虚设**；
  第二条曾因 2 特征夹具退化为「种子 801/803 恰好撞同一特征子集序列」而假失败，
  改成 4 特征夹具后三种子全部发散）。
- 驱动新增：家族表 M0、`m0_config_grid`（**24 ≤ 30**，`feature_fraction=0.85` 使种子契约非虚设）、
  `_m0_fit/_m0_percentile_frame/_m0_seed_mean_scores/_m0_signature/_m0_save_booster/_m0_load_booster/_m0_write_bundle`、
  `train_m0_smoke`、`run_full_m0`。**下游全部复用、模型无关**：
  `publish_fold_dataset`、签名证据链、`FrozenPredictionStore`、`evaluate_model_walk_forward`（读冻结预测算 G1–G6）。
  **与 torch 家族的两处刻意不同**：(1) 无 `FoldPreprocessor`——树对逐特征单调变换不变，
  bundle 记 `preprocessing={"state_hash":"identity"}` 使契约显式；(2) TEST 预测从**已验证 bundle**
  读回 booster（`lgb.Booster(model_file=bundle/seed-N.pt)`），不读训练内存——
  与 `load_inference_artifact` 的「从冻结工件预测」纪律同义。
- **smoke 刻意不带早停**：合成数据对 ranker 过于可分，早停会让模型塌缩成 1 棵树，
  其分数的每日唯一值 <10，**恰好触发 OOS 分数健康门**——那是数据的退化、不是机制的缺陷；
  早停分支本身由基线自带测试覆盖，全量运行始终带验证帧早停。smoke 用固定 64 轮、无验证集。

**验证（全部实测）**：`--family M0 --smoke` 端到端走通（valid_day_count=8、六门齐算；
G3/G4/G5 在合成数据上 FAIL 属预期，因「基线」是真 alpha 而 M0 只见带噪特征）。
`pytest tests -q` → **1709 passed / 45 skipped**（1705 + 本轮 4 条）；
`ruff check src tests` 清、`ruff check scripts/run_real_candidate_training.py` 清；
`mypy src` 158 文件清；脚本 mypy 仅剩**既有** `walk_forward_common.py:20` dolphindb stub 报错（非本轮引入）。

**当前未做 / 等待输入（按依赖排序，非承诺）**：

1. **因子名单**（owner 待定，约 15 个）：watch-pair 评审建议移除 405/398（**待授权**，W1/裁决 4 (iii)），
   契约 `factor_set_frozen_v1.json` 现仍为 17。名单定后只需改 `FROZEN_FACTOR_SET_V1` 元组 + 契约
   （保持一致）；模型 `raw_factor_count=len(factor_order)` 自动跟随；数据集缓存键含因子集，自动重建。
   `configs/models/*.yaml` 的 `raw_factor_count_min: 15`：最终 ≥15 不动、<15 需改一行。
2. **基线须按新因子集重算**：现 `baseline_oos/` 工件是 **legacy 14 因子源**（OOS mean RankIC 0.0864、
   逐折赢家 B3）；G4 要求候选比**同一因子集**的 fold-matched 基线高 0.005，故换集后须以
   `--factor-source frozen` 重跑基线（参照 Task 1.6，约 17h）**再跑 M0 全量**，否则 G4 失真。
3. **M0 全量运行**：`--family M0 --factor-source frozen`，估计每折 24 configs × 3 种子 + 3 终拟合
   ≈ 数百次快速拟合，CPU 单线程数小时量级（远低于 M1 的 42.7 天）。
4. **R8 未动**：M1 崩溃遗留的 13 个检查点仍在（`m1-f01-*`）；M0 用独立前缀 `m0-*`，**不与之碰撞**；
   M1 链是否重启（裁决 2/3 + R8）仍开放，与 M0 无依赖。
