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
- [x] **3.4 全量运行**（1–2 卡，预计 6–24 小时）：
      **【已完成 2026-09-13 17:42】** 工作区 `m1_full_v1`，全 27 格 × 4 折 × 3 seed，
      41 路并行、钉 1 线程，**六道门全过**，G4 = **+0.007826986071187494**。
      **本项标注的"1–2 卡，预计 6–24 小时"错了约 40 倍**：实际是 **CPU 41 核、41.0 小时**
      （09-12 02:05 → 09-13 17:42），GPU 对本家族是**反向优化**（本节 2026-09-05 19:01 设备基准：
      M1 最小格 CPU 333.7 对 CUDA 973.9 ms/日，慢 2.92 倍）。偏差全过程见本节六次修正的记录。

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

      **续跑风险已缓解（2026-09-09，`504aa89` / `79f6a8f` / `97eb150`；上面「未做任何变更」的
      裁决项——device、线程数、max_epochs——一个都没动，本节只增加能力，不改数值口径）**：
      每次全量运行现在都把「每（折, 配置）的搜索结果」写入 `workspace/search/<fold>/`
      （临时文件写入后 rename，中断不会留下半条记录）；`--resume` 时才**采纳**，默认路径逐字不变。
      采纳条件覆盖全部影响数值的输入：内容寻址的 train/validation dataset id、因子序、家族与
      code version、配置内容哈希、**逐配置的有效预算**（LightGBM 侧含 `_m0_hyperparameters`
      填的库内默认值——配置哈希看不见它们，而 `model_code_version` 是人工标签、默认值变了它不会变）、
      以及运行时描述符（torch 版本与 intra/inter-op 线程数：线程数不敏感于速度但改变 checkpoint
      摘要，故必须绑定；**钉住**线程数仍属裁决 2，此处只观测与绑定）。
      折级采纳同样落地：bundle 重哈希、签名 model 证据的 `model_bundle_digest` 与该重哈希比对、
      三个 dataset id 与 baseline 证据 id 比对、card 与 feature_order 比对、两个签名预测工件必须在索引中、
      且每个配置的搜索记录都可采纳——报告的 `search_results` 与 `best_config` 由这些记录**按原网格顺序
      用同一个 `max` 重建**，因此续跑运行的报告是新跑一遍会写出的那份报告，而不是它的近似。
      判据可检验：两次独立全量跑（`m0r_v2` 与 `m0r_v3`）的 `report.json` **逐字节相同**（5,723 字节），
      所以「续跑后报告逐字节一致」是能证伪的验收条件。

      **该判据已实测，并且第一次尝试就把它证伪了（2026-09-10）**：真实 M0R 全量跑写下搜索记录后，
      同一工作区加 `--resume` 再跑，四折**全部整折采纳、零次拟合**，两份报告语义全等
      （`json.load(a) == json.load(b)` 为 True，整个 `oos` 块与六道门逐项相同）却**逐字节不同**：
      差异在 `best_config` 的**键序**——新跑是网格序（`num_leaves, max_depth, …`），续跑是字母序。
      根因是记录落盘用了 `sort_keys=True`；这对原语自身的每一处比较都不可见（dict 相等忽略顺序，
      记录文件名绑定的是另行规范化的 `config_hash`），所以 33 条单元测试全绿，只有端到端能发现。
      已修（`26dd5bd`：不再排序文档，config 按插入顺序存取），并补上钉住键序的测试
      （把 `sort_keys` 放回去会精确杀死它）。**处置原则：改代码去满足判据，而不是放宽判据。**
      修复后复验（全新工作区新跑 C → 同一工作区 `--resume` 跑 D）：两份报告 md5 相同，
      且与 `m0r_v2`、`m0r_v3` 也相同——**五份独立全量跑的 `report.json` 同为
      `834fb523f0177b3d47c5e53797e07d69`**，唯一的例外正是修复前那次续跑（`201ff62d…`）。
      另：新跑派生的 model_id 与已晋级的 `sha256:32315c89…` 相同，即续跑埋点本身不改变任何数值。
      采纳只跳过**重算**，不跳过任何校验：走查评价仍自行加载并校验 bundle、选择证据与预测工件。
      未采纳时的处置是**开跑前拒绝**（bundle 不可变、同折不允许两个 candidate 预测），
      而不是几小时后崩在 bundle 写入上。

      **上面 3.4 的发射命令已作废（2026-09-12 01:40 备好发射包，02:20 按实测并发降级重算时间表）：
      它用的是冻结集链之前的路径（`baseline_oos` 而非 `baseline_v2/baseline_oos`）、没有 `--workspace`、
      没有 `--factor-source`、没有 `--parallel-workers`，且「预计 6–24 小时」与实测相差约 40–70 倍。
      按下面的发射包发射。**

      **M1 全网格发射包（单次拟合成本与并发降级因子均为实测，不是外推）**

      **前置：M1 在 `FAMILY_ORDER_PREREQUISITES` 里没有任何前置**（`"M1": ()`），所以全网格运行
      **不需要 `--prior-report`**。需要的是裁决 2：`--torch-threads` 取哪个值。下面的命令按**钉 1 线程**写，
      理由是实测的——同一折同配置同三 seed 下，串行 52 线程与并行 1 线程的单次拟合墙钟是
      **1.46 h 对 1.48 h（+1.4%）**，即静态家族的拟合几乎不吃 intra-op 线程（逐日小 batch 下同步开销主导，
      与本节 2026-09-05 19:01 设备基准的结论 2「线程数对速度中性」一致），而钉住换来的是身份可移植与
      「在任意宿主上可被精确审计」——后者已在 `m1_parallel_v1` 四折 3,553,163 行上实测为 `max|diff| 0.0`。

      **单次拟合成本（实测，2026-09-12 00:55–01:40，只读 `m1_v2` 已发布的真实 F1 数据集，1 线程，
      走驱动自己的 `_train_cross_section_seed` 与 `_m1_model_factory`，`max_epochs=1`）：**

      | 网格单元 | 参数量 | 单 epoch | 相对最小单元 |
      |---|---|---|---|
      | `hidden 64 / blocks 1`（config0） | 10,497 | **318.4 s** | 1.00 |
      | `hidden 128 / blocks 2` | 70,657 | **339.3 s** | 1.07 |
      | `hidden 256 / blocks 3` | 404,481 | **409.8 s** | 1.29 |

      **参数量差 38.5 倍，单 epoch 只差 1.29 倍**——成本几乎全在逐日循环的固定开销上（F1 有 967 个训练日 +
      237 个验证日），不在矩阵乘上；这与本节 2026-09-05 19:38 那条独立测量（参数量增 39 倍、
      CPU 每日成本只增 1.16 倍）互相印证。这**定量地回答了本节早先那句警告**：config0 确实是 27 格里
      最小的一格、因此是下界，但这个下界只松 **10%（网格均值 348.9 s 对 318.4 s，+9.6%）到 29%（最贵格）**，
      不是按参数量外推会得到的 4 倍。过两端格点的直线是
      `epoch_seconds ≈ 310.1 + 0.12983 × (hidden × blocks)`，三个实测点的偏差 0.0% / 1.2% / 0.0%；
      代入 `hidden × blocks` 的网格均值 298.67 得 **27 格的平均单 epoch = 348.9 s（5.81 min）**。

      **每折的时长**（27 配置 × 3 seed = 81 次拟合；epoch 数取载体运行实测的 16–26 之间、均值 20，
      即最佳 epoch 6–14 加 patience 10）：单次拟合 ≈ 20 × 348.9 s = **1.94 h**
      （交叉核对：config0 实测 17–19 epoch × 318.4 s = 1.50–1.68 h，与载体运行的 1.46–1.48 h 一致）。

      **时间表必须乘上并发降级因子——这是 01:40 那版草稿的实质错误**：它按零降级算，把 41 路写成
      23.3 h ≈ 1.0 天，而实测降级会把它推到 27.8 h ≈ 1.2 天（低估 19%）。降级是**已实测**的：
      `concurrency_scaling_benchmark.txt`（solo / 16 路 / 48 路同一工作量对照，`OMP_NUM_THREADS=1`、
      独立 checkpoint 目录、0 失败）给出 **1 路 1.00×、16 路 1.14×、48 路 1.30×**，线性化得
      `degradation(W) ≈ 1.06 + 0.005 W`（该式在 96 路给出 1.54×，与原基准「~96 路约 1.5×」的外推一致）。
      轮数与粒度择优不是手算的，下表 `gran` / 粗 / 细三列由驱动自己的
      `search_granularity` 与 `granularity_makespans` 在 2026-09-12 02:20 直接调用得出（打平取粗）：

      | W | gran | 粗粒度 | 细粒度 | 轮数 | 降级 | 搜索 | 最终拟合 | 单折 | 四折 |
      |---|---|---|---|---|---|---|---|---|---|
      | 19 | seed | 6 | 5 | 5 | 1.16× | 11.2 h | 2.0 h | 13.2 h | 53.0 h ≈ 2.2 天 |
      | 24 | seed | 6 | 4 | 4 | 1.18× | 9.2 h | 2.0 h | 11.2 h | 44.8 h ≈ 1.9 天 |
      | 27 | config | 3 | 3 | 3 | 1.20× | 7.0 h | 2.0 h | 9.0 h | 36.0 h ≈ 1.5 天 |
      | **41** | **seed** | 3 | **2** | **2** | **1.27×** | 4.9 h | 2.0 h | 6.9 h | **27.8 h ≈ 1.2 天** |
      | 81 | seed | 3 | **1** | 1 | 1.47× | 2.8 h | 2.0 h | 4.9 h | 19.5 h ≈ 0.8 天 |

      「最终拟合」一列按 3 个 seed 在飞、降级取 1.05× 算（2.04 h）——它**不随 W 缩小**，所以 W 越大
      它占单折的比例越高（81 路时已占 42%）。**81 是自然天花板**：81 = 27 × 3 恰好用尽种子级并行，
      再加 worker 也无轮可减。推荐 **41 路**：约 1.2 天、占 104 核中的 41，且 41 → 81 只再省 8.3 h
      却把降级从 1.27× 推到 1.47×。

      **与 M2 不同，M1 的约束是核数而不是内存，这一点已用当前宿主实测确认（2026-09-12 02:20，
      调用驱动自己的 `worker_memory_budget`，parent 取帧释放后实测的 17.07 GiB）：**
      批次张量只有 **0.27 GiB**（train 0.212 + validation 0.058），派发给 worker 的 pickle 载荷
      **0.381 GiB**，批次构造 **9.9 s**（M2 的对应数字是 4.16 GiB 与 400 s）。按每 worker 2.5 GiB 声明：
      `MemAvailable = 515.8 GiB`、可提交上限 `0.9 × 515.8 = 464.2 GiB`，**41 路需 119.6 GiB、
      81 路需 219.6 GiB，两者 `fits=True`，`safe_workers = 178`**——远超 104 核，即护栏到 81 路都不会拒，
      核数与轮数才是真约束。注意 2.5 GiB 这个声明值本身是由 0.381 GiB 载荷 + 0.27 GiB 批次 + torch 运行时
      估的、**不是实测 RSS**；首折要用 `ps -o rss` 打在 spawned worker 上校正（见遥测项 2）。

      ```bash
      cd /data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2
      KEY=$(cat /data/zhangyudong/qrv2_research_artifacts/secrets/selection_evidence_hmac.key)
      setsid nohup env QRV2_SELECTION_EVIDENCE_HMAC_KEY="$KEY" QRV2_PREDICTION_EVIDENCE_HMAC_KEY="$KEY" \
        QRV2_MODEL_REGISTRY_ROOT=/data/zhangyudong/qrv2_research_artifacts/model_registry \
        /data/zhangyudong/.conda/envs/quantyd/bin/python scripts/run_real_candidate_training.py \
        --family M1 \
        --baseline-artifacts /data/zhangyudong/qrv2_research_artifacts/baseline_v2/baseline_oos \
        --baseline-report /data/zhangyudong/qrv2_research_artifacts/baseline_v2/report.json \
        --workspace /data/zhangyudong/qrv2_research_artifacts/m1_full_v1 \
        --report /data/zhangyudong/qrv2_research_artifacts/m1_full_v1/report.json \
        --factor-source frozen_v2 \
        --parallel-workers 41 --torch-threads 1 --worker-memory-gb 2.5 --resume \
        > /data/zhangyudong/qrv2_research_artifacts/m1_full_v1.log 2>&1 &
      ```

      **`cd` 不是可选的**：驱动在 `research_v2/scripts/` 下，而命令里的
      `scripts/run_real_candidate_training.py` 是相对路径，从仓库根粘贴会在第一秒就死于
      `can't open file`（M2 发射包同一问题，已一并修正）。

      **不带 `--max-configs`**：默认即全 27 格。`--resume` 从第一次就带上（新工作区采纳为空、
      行为等同不带；一旦中断，重启即采纳已完成的配置与折）。工作区必须**新开**：`m1_v2` 与
      `m1_parallel_v1` 里已有 `m1-f1..f4` 的不可变 bundle，在同工作区跑全网格会在第一折就被
      「bundle 不可变」拒掉。

      **首折必须读的三项遥测**（与 M2 发射包同一套纪律）：

      1. **并发降级是否与 2026-09-05 那条曲线一致**：上表的 1.27×@41 取自 16/48 路实测的线性化，
         而那次基准跑在**旧折结构的小帧**（40 个训练日）上；新折是 967 训练日 + 237 验证日，
         内存带宽压力不同，所以 41 路的降级**没有在这个帧尺寸上测过**。由检查点时间戳算单 epoch 成本，
         与 `348.9 × 1.27 = 443 s` 比。**若超过约 1.5×（即 >523 s/epoch），杀掉、用更少 worker 重启**——
         `--resume` 把代价从「全部重来」降到「只重跑未完成的部分」，且采纳检查已上提到构造折输入之前
         （`554b3d0`），重启不会为已完成的折重付预处理与批次构造。
      2. **worker 实际 RSS** 对 2.5 GiB 声明值的偏差。实测偏大就上调声明值；由于 `safe_workers = 178`，
         声明值即使翻倍到 5 GiB 也仍不会触发护栏，所以这一项的风险是「预估失真」而不是「运行被拒」。
      3. **每格的 epoch 数**：20 是从载体运行的 config0 观测（最佳 epoch 6–14 + patience 10）取的均值，
         其余 26 格的早停点未知。首折跑完即可用真实的 `search_results` 校正整轮预估——这是上表最大的
         单一不确定项，1.94 h/拟合与它成正比。

      **两处诚实标注的未知**：dropout 对单 epoch 成本的影响没有单独测（三格分别是 0.0/0.1/0.2，
      与 hidden/blocks 混在一起，无法从这三个点分离出来）；每格的 epoch 数在搜索跑完之前不可知。
      （第三项「41 路的降级因子靠外推」已在下面的首折实测中消除。）

      **首折遥测已读到，三项全部有结果（2026-09-12 02:05 发射，02:30 读数；运行未中断、未改参数）**

      发射即健康：护栏打印 `41 x 2.5 GiB + parent 0.6 GiB = 103.1 GiB of 515.5 GiB available` 通过；
      因子源 `frozen_v2 (15 factors)`；F1 三个 split 从缓存载入（train 2,296,573 / validation 696,906 /
      test 804,575 行）；02:11:33 铺开 41 个 worker，`load average` 稳定在 **41.15–41.29**，
      即 41 worker × 1 线程恰好占 41 核，钉线程如设计生效；检查点 manifest 里
      `torch_intraop_threads: 1`、`torch_interop_threads: 1` 已写入证据。日志遍历 `c0…c26`，
      **确认这是全 27 格**——与 `m1_parallel_v1` 那次 `Searched | 1`（四折都只搜 config0）截然不同。

      **遥测 1（并发降级）：实测 1.132×，比外推的 1.27× 好 12%，所以 2026-09-05 那条曲线对全尺寸折是高估的。**
      测法是干净的——不用 epoch-1 的墙钟（它含 worker 启动与 0.381 GiB 载荷反序列化，41 路同时反序列化会
      互相争用，实测那一项约 100 s，会把比值虚抬到 1.43–1.51×），而是取**同一 (config, seed) 的
      epoch-2 与 epoch-1 检查点时间戳之差**，纯稳态、无启动污染。n=23：

      | 量 | min | median | mean | max |
      |---|---|---|---|---|
      | 稳态单 epoch（s） | 361.0 | **372.6** | 372.8 | 381.8 |
      | 对 solo 的比值 | 1.100 | **1.132** | 1.130 | 1.155 |

      逐配置比值与 `hidden × blocks` 无关地聚在 1.10–1.16（`64/1` 1.134–1.155、`128/2` 1.105–1.112、
      `64/3` 1.129–1.130），说明这是**宿主级争用**而不是模型级效应。据此网格均值单 epoch 在 41 路下
      = 348.9 × 1.132 = **395.1 s（6.58 min）**。
      **含义超出 M1**：`concurrency_scaling_benchmark.txt` 的 1.14×@16 / 1.30×@48 是在 **40 个训练日的
      小帧探针**上测的，而真实折是 967 训练日；小帧下每进程启动与访存占比更高，所以那条曲线
      （线性化 `1.06 + 0.005W`，41 路给 1.27×）**系统性高估了全尺寸折的降级**。M2/M3 的降级因此
      也可能比本文件按该曲线算出的更好——但**不能直接把 1.132 搬过去**：M2 的 worker 载荷是 4.16 GiB
      （M1 的 11 倍）、面板大得多，访存争用不是同一个量级，必须各自实测。

      **遥测 2（worker RSS）：实测 0.93 GiB 起步、训练中涨到均值 1.27 GiB（42 个子进程共 53.5 GiB），
      对 2.5 GiB 声明值有约 2 倍余量。** 所以 M1 的内存约束比发射包估计的还松：按实测 1.27 GiB 算，
      81 路只要 102.9 GiB。**结论不变但更硬：M1 的约束是核数与轮数，不是内存。**

      **遥测 3（每格 epoch 数）：仍未定，但已可观测。** 02:30 时 epoch-1 检查点 41 个、epoch-2 检查点 23 个
      （即 41 个任务里 23 个在 epoch 2 仍在改进），`search/F1/` 还是 0 条记录——搜索记录按 (折, 配置) 三 seed
      全跑完才落盘，首个记录预计 04:25 前后。20 这个均值假设要等首折搜索跑完才能用真实 `best_epoch` 校正。

      **据实测重算的工期（取代上面按 1.27× 算的 27.8 h）**：单次拟合 = 20 × 395.1 s = **2.194 h**；
      41 路是种子级 2 轮 → 搜索 4.39 h/折；最终拟合 3 seed 在飞、降级取 1.05× → 2.04 h/折；
      单折 6.42 h，四折 **25.7 h**，加每折约 8 min 的父进程装配 ≈ **26.2 h ≈ 1.09 天**。
      02:05 发射 → **预计 09-13 04:20 前后出 `report.json`**。
      三个估计的相对位置值得记一笔：零降级 23.3 h、按旧曲线 1.27× 得 27.8 h、按实测 1.132× 得 25.7 h——
      **实测落在两者之间、更靠近零降级那一端**，即旧曲线的悲观修正过头了约 8%。

      **遥测 3 也有结果了，而它把工期往回推——两处误差方向相反、几乎抵消（2026-09-12 04:45，
      前 4 个配置共 12 次拟已完成）。** 搜索记录里不含 `best_epoch` 字段，所以从检查点反推：
      某 seed 的最大 epoch 检查点即其 best epoch，总 epoch = best + patience 10。

      | 配置 | 三 seed 的 best epoch | 总 epoch | 验证集集成 RankIC |
      |---|---|---|---|
      | c1 `64/1/dropout 0.1` | 11, 12, 11 | 21, 22, 21 | 0.05120 |
      | c2 `64/1/dropout 0.2` | 6, 14, 15 | 16, 24, 25 | 0.05386 |
      | c3 `64/2/dropout 0.0` | 10, 13, 14 | 20, 23, 24 | 0.05249 |
      | c6 `64/3/dropout 0.0` | 12, 11, 14 | 22, 21, 24 | 0.04801 |

      12 次拟合的总 epoch：**最小 16、均值 21.9、最大 25**。所以"均值 20"这个假设低了约 **9.5%**，
      per-fit 从 2.194 h 修正为 **21.9 × 395.1 s = 2.40 h**；最终拟合 3 seed 在飞、
      `21.9 × 348.9 × 1.05 = 2.23 h`；单折 `2 × 2.40 + 2.23 = 7.04 h`，
      四折 **28.1 h**，加装配 ≈ **28.7 h ≈ 1.20 天**，即 **预计 09-13 06:45 前后出报告**
      （取代上面的 04:20）。
      **值得记的是这两处误差几乎正好抵消**：降级被我高估（1.27 对实测 1.132，−11%）、
      epoch 数被低估（20 对实测 21.9，+9.5%），净结果 28.7 h 与最初按旧曲线算的 27.8 h 只差 3%。
      这不是"估算很准"，而是两个反号错误碰巧抵消——**分开看每一项都是必要的**。

      **第三次修正，也是第一次由实测墙钟锚定而非由假设推定：上面所有工期都用了"平均单次拟合"，
      而一轮的墙钟是由该轮最慢的那个任务决定的（2026-09-12 05:30，13 个配置已完成）。**
      每条搜索记录在该配置**三个 seed 全部跑完**时才落盘，所以记录时间戳直接给出"该配置最慢 seed 的
      单次拟合墙钟"，这正是决定轮次时长的统计量。13 个配置的实测：

      | 统计量 | min | mean | median | max |
      |---|---|---|---|---|
      | 单次拟合墙钟（最慢 seed） | 2.29 h | **2.75 h** | 2.75 h | **3.11 h** |
      | 隐含 epoch 数 | 22.9 | 26.7 | — | 30.2 |

      **尾部因子 `max/mean = 1.131`**（种子级轮次）。据此重算：
      单折 = 2 轮 × 3.11 h + 最终拟合（3 seed 并发、近乎 solo，`26.7 × 348.9 × 1.05 = 2.72 h`）
      = **8.94 h**；四折加装配 ≈ **36.2 h ≈ 1.51 天**，即 **预计 09-13 14:00 前后出报告**。
      **这取代上面的 06:45，但它自己也已被取代——当前值是 09-13 18:30–19:52，见本节末尾的第六次修正。**
      两处要更正：其一，"×4"这一步是假设、不是算术，实测四折的 train+val 行数比是
      1.000/1.110/1.220/1.321，乘数应为 **4.651**；其二，这里猜的"**F4 更小**"是**错的**——
      F4 是四折里**最大**的（3,955,048 行 train+val，比 F1 多 32.1%），
      而且当时还写了"方向未实测"，所以那不是有据的推断而是一句猜测，**现在被实测推翻了**。

      **模型被实测反验了一次，而且是对的**：按"轮次 = 最慢任务"预测首轮墙钟 = 3.11 h，
      而实际首轮 worker 02:11:33 起、最后一条首轮记录 05:17:55 落，**实测正是 3.11 h**。

      **这条修正对 M2/M3 同样适用，它们此前也是按均值算的**：种子级轮次乘 1.131；
      配置级轮次（一个 worker 串跑 3 次拟合）因方差被平均掉，乘 `1 + (1.131−1)/√3 = 1.076`。
      修正后 **M2 = 4.62 / 5.61 / 6.83 天**（按 18 / 21.9 / 26.7 epoch，原为 4.37 / 5.31 / 6.29），
      **M3 = 24.3 h ≈ 1.01 天**（原为 0.95 天）。
      **教训一般化**：makespan 是 `轮数 × 该轮最慢任务`，不是 `轮数 × 平均任务`；
      单次拟合的离散度（这里 ±15%）会**逐轮**进入总工期，而不会被多轮平均掉。

      **第四次修正，也是漏得最久的一个因子：四折并不同价，而我一直按同价算（2026-09-12 07:55）。**
      各折 `train + validation` 的真实行数（读四个已发布数据集的 parquet 元数据得出，不载入数据）：

      | 折 | train | validation | test | **train+val** | 相对 F1 |
      |---|---|---|---|---|---|
      | F1 | 2,296,573 | 696,906 | 804,575 | **2,993,479** | 1.000 |
      | F2 | 2,539,884 | 783,477 | 882,426 | **3,323,361** | **1.110** |
      | F3 | 2,790,995 | 859,748 | 932,167 | **3,650,743** | **1.220** |
      | F4 | 3,046,399 | 908,649 | 938,594 | **3,955,048** | **1.321** |

      静态家族每 epoch 逐日遍历、每日成本正比于当日行数，所以**每折墙钟大致正比于 train+val 行数**，
      四折的相对成本之和是 **4.651**（不是 4.000）。
      F1 的实测墙钟约 **9.4 h**（02:05 发射；首轮 3.11 h 已实测，第二轮最重的 c24–c26
      是 `hidden 256 / blocks 3`、约 464 s/epoch × 约 27 epoch ≈ 3.5 h、最晚 05:18 起 → 约 08:48 结束；
      再加最终拟合约 2.7 h → 约 11:30 出 F1 结果），
      据此 **四折 ≈ 9.4 × 4.651 = 43.7 h**，即 **预计 09-13 21:00–23:00 出 `report.json`**
      （取代上面的 14:00）。
      **【本段已被第六次修正取代】4.651 这个"按行数线性折算"的乘数被三折实测推翻
      （F1 7.93 / F2 11.07 / F3 11.25 h；F3 比 F2 多 9.9% 的行只慢 1.6%）；
      当前 ETA 是 **09-13 18:30–19:52、全程约 41 h**，见本节末尾。保留本段只为留下修正轨迹。**

      **这个因子为什么之前一直没进来**：前三次修正分别动的是降级、epoch 数、尾部离散度，
      每次都把"单折成本"算得更准，然后**乘以 4**——而"乘以 4"这一步从没被质疑过，
      因为 M1 的载体运行和所有基准都用 F1 做样本，F1 恰好是**最小**的那一折。
      **用最小折标定、再乘以折数，是系统性低估**，与"用最小格 config0 标定单 epoch"是同一个错误形状
      （本节早先已经为后者做过修正）。
      同样的检查应当回头套到 M2/M3 上：它们的工期表也是 `单折 × 4`，
      而两者的 per-fit 也同样是按 F1（最小折）标定的。按 4.651 而不是 4.000 重算：

      | 家族 | worker | 单折（F1 口径） | 原按 ×4 | **改按 ×4.651** |
      |---|---|---|---|---|
      | M1 | 41 | 9.4 h（实测） | 37.6 h ≈ 1.57 天 | **43.7 h ≈ 1.82 天** |

      **上表的 M1 行已被实测取代（实际约 41 h）；M2/M3 两行仍是 `×4.651` 的形式，
      而第六次修正已说明这个模型形式本身是错的**（驱动因素是早停 epoch 数、不是行数）。
      M2/M3 的每折 epoch 数在跑之前无从测起，所以那两行应当读作
      **"量级正确、可能偏差 ±30%"的规划值**，不是可核对的预测。
      | M2（per-fit 6.88 h，21.9 epoch） | 24 | 31.8 h | 127.2 h ≈ 5.30 天 | **148.0 h ≈ 6.17 天** |
      | M2（per-fit 8.38 h，26.7 epoch） | 24 | 38.7 h | 154.8 h ≈ 6.45 天 | **180.1 h ≈ 7.50 天** |
      | M3（per-fit 2.61 h） | 48 | 6.08 h | 24.3 h ≈ 1.01 天 | **28.3 h ≈ 1.18 天** |

      即 **M2 的规划区间应从 4.4–6.3 天上修到约 5.1–7.5 天、中心值约 6.2 天**，
      **M3 从约 1 天上修到约 1.2 天**。这两个家族的成本模型其余部分（单 epoch、降级、粒度）
      都是实测或经实测标定的，**只有"×4"这一步是假设**，而它现在被实测行数比推翻了。

      **最后两个此前靠假设的阶段，现在也测了（2026-09-12 08:05）。**

      **(i) 最终拟合是并行的，不是串行的——这一点先读代码确认，因为它差 3 倍。**
      `_run_full_cross_section` 在 `execution.workers > 1` 时把三个 seed 的最终拟合包成
      `final_tasks` 交给 `run_parallel_search`（`capped_worker_count` 会把 worker 数压到任务数 3），
      注释也写明"三个最终拟合彼此独立，而在网格已占满 worker 的折上它们是仅剩的串行部分……
      worker 交回 checkpoint 路径，父进程冻结的不是自己训出来的权重"。
      所以最终拟合 = **1 个 per-fit 时长（约 2.6–2.7 h，3 路并发、近乎 solo）**，
      不是 3 × per-fit（那会是 8.1 h）。`workers == 1` 时才走 `for seed in REQUIRED_SEEDS` 的串行分支。

      **(ii) 评价阶段 ≈ 16 分钟，不是一次载体的时间戳里看起来的 3 小时。**
      `m1_parallel_v1` 的最后一个预测工件写于 18:24:46、`report.json` 写于 21:37:03，
      中间 **3 h 12 min**——若照这个数规划，四折之外要白加 3 小时。**但这个区间不能用**：
      那次运行正是在评价阶段崩过、修完再 `--resume` 的（本文件与 09-11 交接都记了），
      所以那 3 小时里混着崩溃、排障与重启。
      干净的测量来自 **M0R 的两次独立全量跑**（`m0r_v2`、`m0r_v3`，报告逐字节相同、从未崩过）：

      | 运行 | 首个工件 | 最后一个预测 | `report.json` | **评价阶段** | 全程 |
      |---|---|---|---|---|---|
      | `m0r_v2` | 09-09 02:01:04 | 02:43:27 | 02:59:49 | **16.4 min** | 0.98 h |
      | `m0r_v3` | 09-09 03:38:53 | 04:21:11 | 04:37:22 | **16.2 min** | 0.97 h |

      两次相差 0.2 min，所以**评价阶段 ≈ 16 min（0.27 h）是可复现的**，且它是**一次性**的
      （四折全跑完后才算 OOS 门），不是每折一次。
      **要如实标注的口径差别**：这是 LightGBM 路径的测量；torch 路径的评价还要多加载并**重验**
      四个 model bundle（校验和 + 签名证据 + `model_bundle_digest` 比对）。
      但那些文件都很小（每折 3 个 137 KB 的 seed 权重 + 一个 6.4 MB 的验证集预测），
      哈希成本是秒级，而评价的主开销（945 个有效日上的 RankIC / HAC / 逐年分解）两条路径共用。
      所以 torch 路径预计同为 **十几分钟量级**；本次运行结束时会给出干净实测值。

      **方法论一条，值得单独记**：**不要从崩过又被 resume 的运行里做工件时间戳考古。**
      那类时间线里混着人工排障与重启的空档，读出来的"阶段时长"可以是真值的十几倍
      （这里是 3 h 12 min 对 16 min，**差 12 倍**）。要测阶段时长，
      用**从未中断过**的运行（`m0r_v2`/`m0r_v3`），或者等当前这次单次不间断的运行自己走完。

      **折 1 已经跑完了，上面所有推算第一次能被整折实测校验（2026-09-12 10:30；
      `m1_full_v1` 单次不间断运行，27/27 配置、`models/m1-f1` bundle 已写出、折 2 搜索已开始、日志 0 错误）。**

      | 阶段 | 实测 | 此前预测 | 误差 |
      |---|---|---|---|
      | 父进程装配（02:05:00 → 02:11:33） | **0.11 h** | ~0.13 h | −15% |
      | 搜索（02:11:33 → 08:22:09，27 格 × 3 seed = 81 次拟合） | **6.18 h** | **6.22 h**（2 轮 × 最慢 3.11 h） | **−0.6%** |
      | 最终拟合（08:22:09 → 10:00:47，3 seed 并行） | **1.64 h** | 2.6–2.7 h | **−38%（高估了）** |
      | **折 1 合计** | **7.93 h** | 8.9–9.4 h | −12% 至 −16% |

      **"轮数 × 该轮最慢任务"这个模型被整折验证了，误差 0.6%**——这是本轮最硬的一次反验，
      因为它预测的是 6 小时量级的量、而且是在折 1 只跑完 13 个配置时就用实测尾部因子定下的。
      最终拟合那一项我高估了 38%，原因也查清了：我用的是**搜索期"每配置最慢 seed"的平均隐含
      epoch 数 26.7**，而胜出格 c15 的三个 seed 实际收敛得很早
      （best epoch 分别是 **4 / 1 / 7**，加 patience 10 → 总 **14 / 11 / 17** epoch），
      最慢那个 17 epoch × 360.0 s = **1.70 h**，与实测 **1.64 h** 差 3.5%。
      **教训**：搜索期的"最慢 seed"统计量适合估**轮次**（它决定一轮何时结束），
      但不适合估最终拟合——最终拟合只跑**胜出那一格**的 3 个 seed，
      而胜出格的收敛速度与该格在搜索期的表现无关，用全网格的尾部统计量去估它会系统性偏高。

      **ETA 第六次修正：`×4.651` 这个"按行数线性折算"的模型被三折实测推翻了（2026-09-13 10:40）。**
      三折已全部完成，实测墙钟与分解如下（bundle 时间戳相减，单次不间断运行，无崩溃污染）：

      | 折 | train+val 行数比 | 搜索阶段 | 最终拟合 | **折墙钟** | 胜者 | 胜者三 seed 的总 epoch |
      |---|---|---|---|---|---|---|
      | F1 | 1.000 | 6.18 h | **1.64 h** | **7.93 h** | c15 `128/3/0.0` | 17 / 14 / 11 |
      | F2 | 1.110 | 8.10 h | **2.97 h** | **11.07 h** | c14 `128/2/0.2` | 30 / 25 / 12 |
      | F3 | 1.220 | 8.26 h | **2.99 h** | **11.25 h** | c12 `128/2/0.0` | 31 / 14 / 25 |

      **线性模型低估了 15%**：`7.93 × (1.000+1.110+1.220) = 26.4 h`，实测三折合计 **30.25 h**。
      而 F3 比 F2 多 9.9% 的行、只慢 **1.6%**——**墙钟根本不跟着行数走**。
      （我昨晚曾从 F1→F2 两点推出"弹性 3.21、超线性"，那是**对噪声过拟合**：
      两个点定不出基线与指数，F3 一进来就把它否掉了。）

      **真正的驱动因素是早停的 epoch 数**，证据在上面最后两列：最终拟合耗时
      1.64 → 2.97 → 2.99 h 跟着**最慢 seed 的总 epoch 数 17 → 30 → 31** 走，
      而行数比是 1.000 → 1.110 → 1.220。搜索阶段同理（6.18 → 8.10 → 8.26 h，
      F2→F3 只涨 2%）。**makespan 由"该阶段最慢任务的早停点"决定，而早停点是数据相关的、
      跑之前不可知的量**——这与本节早先那条"轮次由最慢任务决定"是同一个原理，
      只是把"最慢"从 worker 层面推到了 epoch 层面。

      **可一般化的结论：对这个工作负载，任何"按数据量折算工期"的模型都是错的形式。**
      行数只影响**每个 epoch** 的成本，而**epoch 数**由早停决定、逐折波动可达 1.8 倍
      （17 对 31），后者的方差远大于前者。
      **这也意味着本文件给 M2/M3 的工期（M2 约 6.2 天、M3 约 1.2 天）形式上同样不可靠**：
      它们也是 `单折 × 4.651`。但 M2/M3 的每折 epoch 数在跑之前无从测起，
      所以目前**没有更好的模型可用**——应当把它们当成"量级正确、可能偏差 ±30%"的规划值，
      而不是像 M1 这样当作可核对的预测。M2 首折跑完就能把它的 epoch 分布测出来。

      **F4 的 ETA（用配速而非行数外推）**：F4 于 09-13 08:20 开始，
      到 10:29（2.20 h）已有 **5/27** 条记录，而 **F3 在同一时刻只有 4 条**——F4 不慢于 F3。
      按 F3 的搜索 8.26 h + 最终拟合 1.64–3.0 h + 评价 0.27 h：
      **`report.json` 预计 09-13 18:30–19:52**，全程约 **41 h**。
      （取代上一版的 15:14。这是第六次修正；前五次的误差来源依次是
      降级外推、epoch 数假设、尾部离散度、四折同价假设、行数线性折算——
      **每一次都是把一个假设换成实测**，而最后两次的教训是：换对了量还不够，还得换对模型形式。）

      **折 1 的科学结果：全网格找到了比载体那一格更好的配置。**
      胜者是 **c15 = `hidden_dim 128 / num_blocks 3 / dropout 0.0`**，
      验证集集成 RankIC **0.060023**；载体用的 **c0（`64/1/0.0`）排第 2，0.056058**，
      即 c15 高 **7.1%**。前五名是 c15 0.060023、c0 0.056058、c21（`256/2/0.0`）0.055802、
      c18（`256/1/0.0`）0.055721、c14（`128/2/0.2`）0.054495——
      **前五里有四个是 `dropout 0.0`**。
      **必须标注的局限**：这只是**折 1 的验证阶段**结果。本文件与 09-11 交接 §3/§8 都已实测过
      **验证阶段的差异极难外推到 OOS 门槛**（谱系在验证阶段四折极差 5.684e-03，
      到 OOS 只差 1.284e-04，**小 44 倍**），所以"c15 在 F1 验证集上高 7.1%"
      **不能**读成"M1 的 G4 会提高 7.1%"。OOS 的 G4 要等四折全跑完、由 TEST 数据算。
      **也正因为 c0 没有胜出，第 6 节那条验收 C（"若 c0 胜出则 G4 必须逐位复现 +0.005845"）
      在折 1 不适用**；它已被一条更强的检查取代，见交接文档。

      **顺带一个早期信号，但要如实标注它的局限**：已完成的 4 格里，验证集集成 RankIC 最高是
      c2 的 0.05386，而载体运行里 c0（`64/1/dropout 0.0`）在 F1 是 **0.0561**，
      即**目前完成的 4 格没有一个超过 c0**。局限是实的：这 4 格全是 `hidden 64`
      （最便宜、因此最先被派发并完成），`hidden 128/256` 的 18 格一个都还没跑完，
      而 M1 的尺寸敏感度很低（38.5 倍参数只换 1.29 倍时间），所以大格既可能更好也可能差不多。
      **在 27 格全部跑完之前，这个信号不足以推断"全网格不会改善 M1"。**

      **顺带得到一个比"推荐 41 路"更硬的理由**：41 不是挑出来的整数，它正是 M1 进入"种子级 2 轮"区间的
      **最小 worker 数**——`ceil(81 / 2) = 41`（81 = 27 配置 × 3 seed）。40 路就退回 3 轮、
      工期跳到 1.52 天；而 48 路仍是 2 轮、降级更高，反而比 41 路慢（1.18 天对 1.16 天）。
      下一个台阶是 `ceil(81 / 1) = 81` 路（1 轮）。按实测 RSS 1.27 GiB，81 路内存只要 102.9 GiB、
      完全可行，若降级如趋势约 1.20× 则四折 **17.4 h ≈ 0.73 天**，比 41 路省约 8.5 h。
      **本轮没有改用它**，理由三条：81 路降级未实测（而遥测 1 的杀线是 1.5×，留不出诊断余量）；
      本机是共享宿主，41/104 核已经是明确的资源声明，81/104 属于要所有者点头的资源占用决定；
      且重启会丢掉当时 41 个 worker 各约 2 个 epoch 的在飞进度（搜索记录尚未落盘，`--resume` 无从采纳）。
      若所有者愿意占 81 核，这是一次约 20 分钟代价换约 8.5 小时的可选项。

- [x] **3.5 判定归档**：通过则渲染判定文档到
  `docs/reports/<date>-m1-real-candidate-walk-forward.md`（复用判定器模式），
  更新主计划 10.2；失败则归档失败报告并分析（不得用 2026 数据补救）。

      **【已完成 2026-09-14 11:35】** 判定文档已渲染到
      `docs/reports/2026-09-13-m1-full-grid-real-candidate-walk-forward.md`
      （`summarize_real_candidate_walk_forward.py` 直接生成，输出 `M1 candidate: WALK_FORWARD_PASSED`；
      同一输入渲染到 `m1_full_v1/verdict.md` 与 docs 两处，**md5 全同 `f31293f38389f5c52c0d4ec2efbe614e`**）。
      **归档前做的两项验收都通过了**：
      **验收 A**——同工作区 `--resume` 重跑（18:00:15 → 18:54:56，四折整折采纳、零次拟合），
      两份报告 **md5 逐字节相同 `600a1fd945324a2bfa2af8de9b5ed954`**。
      **这是该判据第一次在 27 格规模上被验证**：此前的五份逐字节相同报告全部来自
      **单配置**的 M0R，而 M1 载体也只搜了 1 格，所以"由采纳记录按原网格顺序用同一个 `max`
      重建 `best_config` 与 `search_results`"这句话此前只在 1 条记录上验过。
      **验收 C′**——四折胜出格的最终拟合检查点与其搜索阶段检查点**逐字节相同，54/54**
      （F1 c15、F2 c14、F3 c12、F4 c16）。
      另外**评价预检**在 F1/F2/F3 完成时就已逐折跑过（`_require_verified_model_bundle`、
      `_require_candidate_evidence`、`_require_best_simple_evidence`、`_validate_oos_frame` 全 PASSED），
      链条最后一环也已验证：`load_walk_forward_evidence` 接受 `promotion_evidence.json`
      （sha256 `e941274d44922d2a118a70ca…`），**registry 独立重算的 G1–G6 与报告逐位相同**。
      **注**：`5.2 更新主计划 10.2` 仍**未做**——它要求 M1–M3 全部结果，M2 正在跑、M3 未发射。

      **归档前必须做的两项验收（2026-09-12 02:55 补，都是本次全网格运行才第一次具备条件的检查）**

      **验收 A：27 格下的"续跑报告逐字节一致"判据至今只在 1 格上验过，必须在全网格上重验。**
      本文件早先记录该判据已实测——五份独立 M0R 全量跑的 `report.json` 同为
      `834fb523f0177b3d47c5e53797e07d69`——但 **M0R 的网格是被钉死的单配置**，
      `m1_parallel_v1` 也只搜了 `c0`（四折 `Searched | 1`）。也就是说
      「`best_config` 与 `search_results` 由采纳记录按原网格顺序用同一个 `max` 重建」这句话，
      端到端只在 **1 条记录**上被验证过；而历史上真正咬人的那个 bug（`sort_keys=True` 导致续跑报告
      键序不同、33 条单元测试全绿、只有端到端能发现）恰恰是这一类。
      单元层并非裸奔——`_ordered_search_results` 有用 **2 格、乱序完成、且含平局**的测试
      （`test_a_fully_adopted_grid_still_yields_ordered_search_results`、
      `test_ordered_search_results_restore_grid_order_after_parallel_completion`），
      而运行中的日志也已实证 `c0…c26` 各有互不相同的 config hash 与记录名。
      但 2 格单元 ≠ 27 格端到端。**做法**：`m1_full_v1` 出报告后，在**同一工作区**加 `--resume` 再跑一次；
      预期四折全部整折采纳、零次拟合（因此很快），然后比对两份 `report.json` 的 md5。
      不一致就是真 bug，**处置原则沿用本文件已确立的那条：改代码去满足判据，而不是放宽判据。**

      **验收 B：可比性已用内容寻址 id 核实，不是假设。** `m1_full_v1` 的 F1 数据集 id
      `226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538` 与基线证据 id
      `6fe33eda88430e5ee90e8d975c16f36c8d0de1a44a70eda594b12b82ff5678b9` **与 `m1_parallel_v1` 逐字相同**；
      基线工件是同一批 `baseline_v2/baseline_oos/F1..F4.parquet`。两个已通过候选的评价口径也已核对一致：
      `m0r_v2` 与 `m1_parallel_v1` 同为 `mode=full`、4 折、`valid_day_count = 945`、
      `baseline_mean_rank_ic = 0.0852268662737014`（16 位全同）。
      所以 Task 5 的横向比较是**同基线、同日集、同折结构**下的比较，
      `+0.009149`（M0R）与 `+0.005845`（M1 单配置载体）可直接相减，全网格结果落进来也同样可比。
      （注：`baseline_v2/report.json` 自身的 `valid_day_count` 是 969、`mean_rank_ic` 0.08706，
      与候选报告里的 945 / 0.08523 不同——前者是基线家族自己的全日集口径，
      后者是候选与所选 `best_simple` 基线的**交集日集**；G4 只用后者，两者不矛盾。）

---

### Task 4: M2 → M3 顺序执行

**Files:** 同 Task 3（`--family M2` / `--family M3` 分支，时序家族走
`TemporalTrainingBatch` + history_length=20 契约，横截面家族走
`FeatureBatch`）。

- [x] **4.1 M2 分支 + 合成 smoke**（GRU/TCN 双家族内选择，4×2×3×2=48 组合
  超出 30 预算 → 按 E7「受限搜索」先以验证集 RankIC 粗筛到 ≤30 再精评；
  粗筛规则写入报告）。
- [ ] **4.2 M2 全量运行 + 判定归档**（同 3.4/3.5）。

      **发射包（2026-09-10 23:19 备好；参数全部取自当晚实测，发射时不需要重新推导）**

      前置：M1 必须六道门全过——顺序守卫会自己读 `--prior-report` 里的报告，失败报告与 smoke 报告一律拒绝。

      **M2 没有串行选项。** 按计划的 8.60 h/拟合，一折 24 配置 × 3 seed + 3 次最终拟合 = 75 次拟合 ≈ 645 h，
      四折 ≈ 107 天（计划 §3.4 的全链估算 161 天，同量级）。所以「启动 M2」这件事本身就等于采用**钉线程的并行
      谱系**（`--torch-threads 1`），也就是裁决 2；发射前应把它记录为已裁决偏差。这**不影响 G4 的可比性**：
      折匹配基线是冻结的 LightGBM 工件、其 `num_threads` 早已钉死，所以候选减基线是「钉线程谱系的 torch 候选」
      对「未变的基线」，两侧口径各自自洽；受影响的是与其它 torch 运行的**逐位**可比性。

      **worker 数（2026-09-12 00:05 第二次依实测改写；原为「19 而不是 20」，09-11 02:26 改为 24）：**
      足迹又降了一次。`temporal_fold_batches` 原先把已经死掉的原始 train 分片与两个 prepared 帧一直握到窗口
      构造结束，逐阶段实测后改成用完即释放（`9e8628a`）。F1 实测：worker 峰值 **13.61 → 12.85 GiB**、父进程
      峰值 **19.15 → 17.07 GiB**，而批次数（953 / 237 / 242）、张量字节数（4,470,614,049 与 5,689,592,480）
      与 `(1,1610,20,15)` float32 形状全部不变。按 F4 行数等比折算：单 worker **16.56 GiB**、父进程
      **21.99 GiB**。据此 `--worker-memory-gb` 从 17.6 改为 **16.6**。（09-11 02:26 那次的数字——单 worker
      13.61 GiB、父进程 19.15 GiB、折算 F4 为 17.54 / 24.67 GiB——已被本次取代。）

      **但 worker 数不是"能开多少就开多少"，M2 的 makespan 对它是台阶函数，而台阶正好在 24。** 24 配置在 24 路下
      一轮跑完（配置级 1 轮 × 3 次拟合 = 3 个拟合时，与种子级 `ceil(72/24)=3` 打平、按规则取粗粒度），搜索
      25.8 h；而 23 路与 19 路完全相同（种子级 4 轮 = 4 个拟合时，34.4 h）。所以**要么拿到 24 路，要么退回
      19 路，中间的 20–23 路买不到任何搜索时长**（但仍白拿重建从 40 分钟到 7.6 分钟这一份）。

      **⚠ 上面这个"24 路台阶"建立在一个已被实测推翻的申报值上（2026-09-12 06:45，生产路径只读探针
      `/tmp/probe_tcn128_worker_peak.py`，日志 `/tmp/probe_tcn128_peak.log`、`/tmp/probe_tcn128_peak_F4.log`）。**

      `--worker-memory-gb 16.6` 追溯到的是 2026-09-11 交接 §11 那张阶段表，而**那张表测的是
      `temporal_fold_batches` 的峰值、止于"validation 批次构造后 12.77 GiB"——它不含训练期**。
      M2 至今一次都没真正跑过（`m2_v1` 不存在），所以**训练期的 worker 峰值从来没被测过**。
      本次用生产路径直接测（调真正的 `temporal_fold_batches`，因此 `9e8628a` 的逐阶段 `del`
      释放序列照常发生；逐阶段同时采样 VmRSS 与 VmHWM）：

      | 折 | 阶段 | VmHWM | 与申报值 16.6 的关系 |
      |---|---|---|---|
      | F1 | 批次构造后 | **12.99 GiB** | 复现了交接记录的 12.85 GiB（差 1%）→ **探针保真** |
      | F1 | TCN/128 训练 1 epoch 后 | **16.29 GiB** | 训练期增量 **+3.29 GiB**；只剩 **1.9%** 余量 |
      | **F4** | **批次构造后（尚未训练）** | **17.06 GiB** | **已超过申报值**；交接按行数外推的 16.56 低了 3% |
      | **F4** | **TCN/128 训练 1 epoch 后（2573 s）** | **17.06 GiB** | 训练期增量 **+0.00 GiB**（机制见下） |

      **为什么 F1 的训练增量是 +3.29 GiB、F4 却是 +0.00**：`VmHWM` 是生命周期高水位，
      而 PyTorch 的缓存分配器会**复用构造期已经撑大的 arena**。F1 的构造峰只有 12.99 GiB，
      不够装 TCN/128 的训练激活，于是 arena 被迫长到 16.29；F4 的构造峰已经是 17.06 GiB，
      训练激活装得下，高水位就一动不动。**所以 F4 的真实 worker 峰值 = 17.06 GiB，
      而不是把 F1 的 +3.29 按行数放大后叠加得到的 ~21 GiB**——我最初就是这么外推的，实测否掉了它。

      **关键结构事实：批次构造峰值与配置无关。** 每个 worker 都要建同一套折批次，
      所以 F4 的 17.06 GiB 是**所有 24 个 worker 的地板**，与它拿到的是 GRU 还是 TCN 无关；
      TCN/128 只是在这之上再加训练增量。

      **而申报值正好卡在台阶的悬崖边上。** 用驱动自己的 `worker_memory_budget` +
      `search_granularity` 现场算（parent 取实测 21.99 GiB、`MemAvailable` 467.7 GiB）：

      | 申报值 | 24 路所需 | 等价 MemAvailable 门槛 | `safe_workers` | 粒度 | 轮数 | M2 四折 |
      |---|---|---|---|---|---|---|
      | **16.6**（现值） | 420.4 GiB | ≥467.1 GiB | **24** | config | 3 | **5.30 天** |
      | **17.1** | 432.4 GiB | ≥480.4 GiB | **23** | seed | 4 | **6.64 天** |
      | 18.0 | 454.0 GiB | ≥504.4 GiB | 22 | seed | 4 | 6.61 天 |
      | 20.4 | 511.6 GiB | ≥568.4 GiB | 19 | seed | 4 | 6.54 天 |
      | 21.3 | 533.2 GiB | ≥592.4 GiB | 18 | seed | 4 | 6.52 天 |

      **这张表的 `safe_workers` 列是在 `MemAvailable = 467.7 GiB` 那一刻算的（两个探针还在跑，
      多占了约 33 GiB），所以它会随后两列一起移动——不要把它当成常数。**
      同一申报值 17.1 下：`MemAvailable` 467.7 → `safe_workers=23`（丢掉台阶）、
      **483.2 → 24（保住台阶）**、515 → 25。
      **判据因此是"发射那一刻的 MemAvailable 是否 ≥480.4 GiB"，不是申报值本身**；
      第三列给的门槛才是不随时间变的那个数。

      **悬崖是真的，但我上面第一版的结论过头了，在此更正（07:25，F4 训练段测完之后）。**
      我曾写下"实测的 F4 地板 17.06 GiB 已经在悬崖另一侧，所以 24 路在诚实申报值下不成立"——
      **那句话是错的**。17.06 向上取整是 **17.1**，而 24 × 17.1 + 21.99 = **432.39 GiB**
      → 需 `MemAvailable ≥ 480.4333 GiB`（前置检查因此取 **480.5**，见下面那条 awk 的注释）；
      现场实测（M1 在跑、探针已退出）`MemAvailable = 483.2–483.6 GiB` →
      `worker_memory_budget` 返回 **`fits=True`、`safe_workers=24`**。
      **即诚实申报仍然保得住 24 路那个台阶**，只是把前置门槛从 467.1 GiB 抬到 **480.43 GiB**（严 13.3 GiB）。
      M1 结束后 `MemAvailable` 约 515 GiB → `safe_workers=25`，更宽裕。

      **而且低报并不构成 OOM 风险，因为 0.9 的提交系数把它吸收掉了。** 按实测峰值算，
      24 路的**真实**聚合需求是 `24 × 17.06 + 21.99 = 431.4 GiB`；
      而护栏用 16.6 的申报值要求 `MemAvailable ≥ 467.1 GiB`。
      两者之差 = **35.7 GiB（真实需求的 8.3%）的真余量**。
      所以"申报值低 2.7%"这件事的后果是**护栏偏严而不是偏松**：
      把它改成实测的 17.1 会让门槛更高、更保守，而不是更容易 OOM。
      **应当改**（申报值就该是实测值，否则这个数字失去意义），但改完不会丢掉 24 路。

      **仍然成立的那条结构性事实**：`_check_worker_memory` 只实测**父进程** RSS，
      worker 那一份**完全采信申报值**，所以护栏**原则上**无法发现低报——
      这次是 0.9 的系数恰好兜住了，不是机制兜住的。
      若哪一天申报值低得超过 11%（0.9 系数的余量），失败模式就会变成
      几天后 kernel OOM 杀掉 worker → `run_parallel_search` 抛错 → 该折搜索中断
      （搜索记录逐配置落盘，`--resume` 可救回已完成的配置，但那一折要重跑）。
      **这正是护栏被写出来要防的那种"昂贵的失败"，而它的有效性取决于申报值的准确性。**

      **对"是否与 M1 并发"的含义（更正 2026-09-12 05:40 那个结论的余量部分）**：
      用诚实申报值 17.1，并发要求 `MemAvailable ≥ 480.43 GiB`，而 M1 在跑时实测只有
      **480.9–483.6 GiB**，**门槛余量仅 0.5–3.1 GiB（约 0.6%）**；护栏每折装配后还会再查一次，
      而 M1 的足迹在 F2/F3（行数比 F1 多 11%/21%）会变大，所以并发**很可能在某一折被拒**。
      M1 结束后 `MemAvailable` 约 515 GiB、余量 34.6 GiB。
      **结论：不是"不能并发"，而是"并发的余量薄到不值得"——按顺序跑，M2 在 M1 之后发射。**

      **但 25.8 h / 34.4 h 是零降级数字，与 M1 草稿同一个错误，已按实测降级重算（2026-09-12 02:35）。**
      `concurrency_scaling_benchmark.txt` 实测 16 路 1.14×、48 路 1.30×，线性化 `1.06 + 0.005W`；
      轮数与粒度择优由驱动自己的 `search_granularity` / `granularity_makespans` 现场调用得出
      （24 路确为配置级 3 个拟合时、19 与 23 路确为种子级 4 个，与上面那段一致），最终拟合按 3 个 seed
      在飞取 1.05×：

      | W | gran | 轮数 | 降级 | 搜索/折 | 单折（含最终拟合） | 四折 |
      |---|---|---|---|---|---|---|
      | 19 | seed | 4 | 1.16× | 39.7 h | 48.8 h | 8.13 天 |
      | 23 | seed | 4 | 1.18× | 40.4 h | 49.5 h | 8.24 天 |
      | **24** | **config** | **3** | 1.18× | **30.4 h** | **39.5 h** | **6.58 天** |
      | 36 | seed | 2 | 1.24× | 21.3 h | 30.4 h | 5.06 天（内存不可达） |

      即**发射 M2 要按约 6.6 天规划，不是按 25.8 h × 4 ≈ 5.7 天**（低估约 15%）。台阶结构不变：
      24 路仍比 19/23 路快 1.6 天，36 路更快但 686.7 GiB 超过本机 `MemTotal`，确定不可达。

      **24 路仍有前提，发射前必须先读 MemAvailable。** 命令现在传的是 **`--worker-memory-gb 17.1`**
      （2026-09-12 07:30 由生产路径直测的 F4 worker 峰值 17.06 GiB 向上取整；此前是 16.6，
      依据的是只含批次构造期的折算，见本节上面那块 ⚠ 标注），所以护栏算的是
      24 × 17.1 + 21.99 = **432.39 GiB**，等价于要求 **MemAvailable ≥ 480.4333 GiB**
      （前置检查取 480.5，见下面那条 awk 的注释——**四舍五入到精确门槛下方会让检查与护栏不自洽**）。
      这个门槛经历过三次修正：466.0（按 16.56 折算，与命令行不自洽）→ 467.1（与 16.6 自洽）→
      **480.4（与实测峰值自洽）**。最初那次不自洽的后果是实的：MemAvailable 落在 466.0–467.1 GiB
      这 1.1 GiB 窗口里时，前置检查会说「24 workers」而护栏随即拒绝——
      正好是在操作者已经读完 HMAC key、决定发射之后。
      当前实测（2026-09-12 07:35，M1 在跑、探针已退出）：`MemAvailable = 483.5 GiB`、
      按 17.1 申报 `fits=True`、`safe_workers = 24`——但**门槛余量只有 3.1 GiB（0.6%）**，
      而护栏每折装配后还会再查、M1 的足迹在 F2/F3（行数比 F1 多 11% / 21%）还会变大，
      所以**不建议与 M1 并发**；
      M1 结束后 `MemAvailable` 约 515 GiB、`safe_workers = 25`、余量 34.6 GiB，那才是发射 M2 的时机。
      这个差别是实的：2026-09-11 02:00 前后另一位租户的 4 个进程占着约 86.6 GiB、MemAvailable 只有 488.5 GiB，
      那时按 16.6 申报 F4 的安全 worker 数是 23、**24 路会被护栏拒**；按现在的 17.1 申报，
      488.5 GiB 下 `fits=True`、`safe_workers = 24`，24 路**能过，但门槛余量只有 8.1 GiB**
      （我先前写成"`safe_workers = 23`、仍会被拒"是**算错了**，已按 `worker_memory_budget`
      的实测输出更正——见上面那张门槛表）。注意护栏**每折装配后还会再查一次**，
      所以启动时过得去、到 F4 才拒仍有可能；
      `--resume` 把那种情况的代价降到"重启后采纳已完成的折"。**36 路（下一个台阶，搜索 17.2 h）需要
      686.7 GiB，超过本机 `MemTotal` 629.5 GiB（2026-09-12 02:25 实测；此处原写「660 GiB」，
      结论不变且更宽裕）→ 确定不可达**——24 路是当前足迹下的天花板，这是算术而不是保守估计。

      ```bash
      # 发射前置检查：>=480.5 GiB 才用 24 路（精确门槛 480.4333，取 480.5 留保守余量，理由见下）
      awk '/MemAvailable/{printf "MemAvailable = %.1f GiB -> %s\n", $2/1048576, ($2/1048576 >= 480.5 ? "24 workers" : "19 workers")}' /proc/meminfo
      ```

      **这个阈值必须写 480.5、不能写 480.4——我自己刚踩了这个坑，而它与本文件上面记过的
      466.0/467.1 不自洽是同一类错误。** 精确门槛是 `(24 × 17.1 + 21.99) / 0.9 = 480.4333… GiB`。
      现场调用 `worker_memory_budget` 验证：`MemAvailable = 480.4` 时 24 路 **`fits=False`**
      （所需 432.39 GiB 对可提交 432.36 GiB，**差 0.03 GiB 而失败**）、`safe_workers = 23`；
      到 `483.5` 才 `fits=True`、`safe_workers = 24`。
      所以检查若写 `>= 480.4`，在 480.40–480.43 这段里会说"24 workers"而护栏随即拒绝——
      **正是上面刚批评过的"检查与命令不自洽"，只是窗口从 1.1 GiB 缩到 0.03 GiB**。
      教训可以一般化：**把门槛四舍五入到精确值的下方是不安全的，宁可向上取。**

      **申报值 17.1 下的实测门槛表**（`worker_memory_budget` 现场调用，parent 取 21.99 GiB）：

      | MemAvailable | 16.6 申报 | **17.1 申报** | `safe_workers`@16.6 | `safe_workers`@17.1 |
      |---|---|---|---|---|
      | 467.7（M1 + 两个探针在跑） | fits | **拒** | 24 | 23 |
      | 480.4 | fits | **拒**（差 0.03 GiB） | 24 | 23 |
      | **483.5**（M1 在跑，07:35 实测） | fits | **fits** | 24 | **24** |
      | 488.5（09-11 别的租户占用时） | fits | **fits** | 25 | **24** |
      | 515.0（M1 结束后的预期） | fits | **fits** | 26 | **25** |

```bash
cd /data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2   # 必需：下面用的是相对路径 scripts/...
KEY=$(cat /data/zhangyudong/qrv2_research_artifacts/secrets/selection_evidence_hmac.key)
setsid nohup env QRV2_SELECTION_EVIDENCE_HMAC_KEY="$KEY" QRV2_PREDICTION_EVIDENCE_HMAC_KEY="$KEY" \
  QRV2_MODEL_REGISTRY_ROOT=/data/zhangyudong/qrv2_research_artifacts/model_registry \
  /data/zhangyudong/.conda/envs/quantyd/bin/python scripts/run_real_candidate_training.py \
  --family M2 \
  --baseline-artifacts /data/zhangyudong/qrv2_research_artifacts/baseline_v2/baseline_oos \
  --baseline-report /data/zhangyudong/qrv2_research_artifacts/baseline_v2/report.json \
  --workspace /data/zhangyudong/qrv2_research_artifacts/m2_v1 \
  --report /data/zhangyudong/qrv2_research_artifacts/m2_v1/report.json \
  --factor-source frozen_v2 \
  --prior-report M1=/data/zhangyudong/qrv2_research_artifacts/m1_parallel_v1/report.json \
  --parallel-workers 24 --torch-threads 1 --worker-memory-gb 17.1 --resume \
  > /data/zhangyudong/qrv2_research_artifacts/m2_v1.log 2>&1 &
```

      `--resume` 从第一次就带上：新工作区没有记录，采纳为空、行为与不带等同；一旦中断，重启即采纳已完成的配置
      与已完成的折（折级采纳要求 bundle 重哈希、签名证据的 `model_bundle_digest` 与该重哈希一致、两个签名预测
      工件在索引中、且每个配置的搜索记录都可采纳）。

      **`--prior-report` 的路径已在 2026-09-12 01:15 更正（原为 `m1_v2/report.json`）。** 那份报告不存在、也
      不可能存在：`m1_v2` 早于 `142f8ea`（搜索记录逐条落盘），`search/` 为空，`--resume` 会在任何拟合之前就
      SystemExit（实测理由：`config 0 has no adoptable search record`），所以它的四个已冻结折永远无法被采纳、
      报告永远写不出来；它的六道门数值只能由只读路径取回（见交接第 8 节），而只读路径**不产出可归档的
      report.json**。原路径会让 M2 在发射第一秒就死于 `--prior-report path does not exist`。
      现在指向 `m1_parallel_v1/report.json`——已核实 `mode=full`、`family=M1`、`oos.all_passed=true`，
      正是 `prior_families_from_reports` 要求的全部三项。
      **含义要写明**：为 M2 的顺序前置作保的是**钉 1 线程那条谱系**的 M1。若所有者希望由未钉线程的
      `m1_v2` 谱系作保，唯一途径是新工作区重跑约 44 小时以产出归档报告；这与裁决 2 是同一个选择。
      另：`FAMILY_ORDER_PREREQUISITES` 里 **M1 自身没有前置**，所以 M1 的全网格运行不需要任何 `--prior-report`。

      **首折必须读的三项遥测**——这是全链里唯一还靠外推的数字，M2 在 19 路下的并发降级从未实测：

      **M2 的单 epoch 成本已在真实新折上实测，8.60 h/拟合这个规划值可以收窄了（2026-09-12 04:45，
      只读探针 `/tmp/probe_newfold_cost.py`，日志 `/tmp/probe_temporal_newfold.log`）。**
      探针读同一个已发布 F1 数据集（`226a8dfc…`，15 因子），面板组合与 `temporal_fold_batches`
      逐项相同（train + `concat(train, validation)` 验证面板），模型走驱动自己的 `_m2_model_factory`。
      M2 网格 24 格 = 2 家族 × 2 hidden × 3 dropout × 2 lr，**成本只随家族与 hidden 变**，
      四个 (家族, hidden) 组合各占 6 格、权重各 1/4，所以四格测全就等于测出了网格均值：

      | 格 | 参数量 | 争用下单 epoch | 折算 solo | ms/训练日（争用下） |
      |---|---|---|---|---|
      | GRU / hidden 64 | 18,497 | 848.68 s | **749.7 s** | 890.5 |
      | GRU / hidden 128 | 61,569 | 1338.29 s | **1182.2 s** | 1404.3 |
      | TCN / hidden 64 | 15,297 | 1039.33 s | **918.1 s** | 1090.6 |
      | TCN / hidden 128 | 55,169 | 1897.07 s | **1675.9 s** | 1990.6 |
      | **网格均值** | — | — | **1131.5 s（18.86 min）** | — |

      折算用的 1.132 是 M1 在 41 路的实测降级（探针与生产同时段、同一背景负载）。
      **与 2026-09-05 旧基准的同格对比，方向是相反的，这正好坐实了 (a) 项那条"ms/日 不可跨折迁移"**：
      GRU/128 旧 1301.3 → 新 solo 1240.5 ms/日（**−4.7%**），
      TCN/128 旧 1518.7 → 新 solo 1758.6 ms/日（**+15.8%**）。
      一个变便宜一个变贵，所以任何"按 ms/日 折算"的做法在 M2 上都会给出方向都说不准的答案；
      参数量也印证了探针确实在 15 因子上：TCN/128 旧记 54,913、新测 **55,169**，差 **+256**，
      与 M3 那格同样的 +256（输入投影多一个因子）。

      **据实测重算的 M2 工期（24 路、配置级 3 个拟合时、降级仍按未实测的 1.18）**：
      剩下的唯一大未知是**每格 epoch 数**，M1 的实测均值是 21.9（见 3.4 首折遥测），
      按它以及上下界算：

      | 每拟合 epoch 数 | per-fit | 单折 | 四折 |
      |---|---|---|---|
      | 18（本文件早先的下界口径） | 5.66 h | 26.2 h | 104.9 h ≈ **4.37 天** |
      | **21.9（M1 实测均值）** | **6.88 h** | **31.8 h** | **127.4 h ≈ 5.31 天** |
      | 26（M1 观测到的最长） | 8.17 h | 37.8 h | 151.0 h ≈ **6.29 天** |

      即 **M2 约 4.4–6.3 天、中心值 5.3 天**，比本文件目前写的 6.61 天乐观约 20%；
      而 8.60 h/拟合这个规划值落在实测区间的最上端之外（它相当于 27 epoch）。
      **降级因子仍是外推的**，M2 的载荷与面板远大于 M1，1.132 不能搬过来——首折仍须实测。

      **顺带澄清一个会造成误报的数字：探针里 TCN/128 的峰值 RSS 是 16.12 GiB，看着像是超过了
      `--worker-memory-gb 16.6` 的余量，但它不能用来上调声明值。** 探针**没有**做生产
      `temporal_fold_batches` 的逐阶段 `del` 释放序列（`9e8628a` 释放的那 1.24 GiB 死帧，
      加上原始 train 分片与两个 prepared 帧），它把 train / validation / prepared×2 / panel
      全都一直握着。扣掉这部分（GRU 各格探针峰值 12.18 GiB 对生产 F1 实测 12.85 GiB 可标定出
      约 3.3 GiB 的量级）后，探针的 TCN/128 折算到生产口径约 **12.8 GiB**，
      与生产 F1 实测的 **12.85 GiB** 吻合；再按 F4 行数等比放大就是文档里那个 16.56 GiB。
      而 TCN/128 比 GRU 各格重约 4 GiB 这件事是真的（模型与激活）。

      **但上面那句"所以 16.6 GiB 的声明值站得住、不需要改"是错的，已在 07:25 被直测推翻，在此撤回。**
      对那个 flawed 探针的分析本身是对的（它确实没做 `del`、确实有跨格碎片累积，16.12 GiB 确实虚高），
      错的是从它推出的结论——因为我拿它去比的是**批次构造期**的 12.85 / 16.56 GiB，
      而那两个数**本来就不含训练期**。真正该问的是"生产路径上 TCN/128 训练完的峰值是多少"，
      而这个数当时**谁都没测过**（M2 从未真跑，`m2_v1` 不存在）。后来用生产路径直测得到：
      **F1 = 16.29 GiB（构造 12.99 + 训练 3.29）、F4 = 17.06 GiB（构造即峰值、训练 +0.00）**。
      所以 **16.6 低报了 2.7%，已改为 17.1**，发射命令与前置检查同步改成 **480.4 GiB**。
      为什么这个低报反而**不构成 OOM 风险**（0.9 提交系数吸收了它，真余量 35.7 GiB / 8.3%）、
      以及为什么 24 路那个台阶**保得住**（480.43 GiB 门槛对实测 483.2–483.6 GiB 仍 `fits=True`），
      见本节上面那块 ⚠ 标注。

      1. **单 epoch 成本**：由检查点时间戳算，与单 worker 期望值比（计划实测 1301.3 ms/train日 × 约 1220 训练日
         ≈ 26.5 min/epoch）。静态家族在 3 路下降级约 1.00×，但 M2 的单 worker 面板大得多，24 路未知。
      2. **worker 实际 RSS** 对 16.6 GiB 声明值的偏差（`ps -o rss` 打在 spawned worker 上）。F1 实测
         **12.85 GiB**（批次构造重写前 16.80、释放死帧前 13.61），16.6 是按 F4 行数等比外推的 16.56 向上取整；
         父进程实测 **17.07 GiB**（F1；释放死帧前 19.15），护栏减的就是它。**同时读父进程的 RSS**：F4 的安全
         worker 数是 25、24 路的要求是 MemAvailable ≥ 466.0 GiB，所以父进程若比外推的 21.99 GiB 大得多，
         护栏仍会在 F4 装配后拒掉整轮。
      3. **7.6 分钟的面板重建是否每 worker 只发生一次**（缓存命中；重写前是 40 分钟）。若每个任务都重建，说明
         缓存键写错了，那会让每折多出几十小时——重建变快使这个故障**更隐蔽**，因为绝对数字小了，所以这一项
         不能因为"看起来只花几分钟"就跳过。

      **关于这 7.6 min 的一个口径澄清（2026-09-12 03:30），免得后来人拿别的数字去"修正"它。**
      本次的只读探针在真实 F1 上构造同一套时序面板（train + `concat(train, validation)` 验证面板，
      与 `temporal_fold_batches` 的组合逐项相同）只用了 **72.5 s**，乍看与 7.6 min 差 6 倍。
      两者测的不是同一件事：探针直接 `pd.read_parquet` 读帧（0.9 s，**不做校验**），
      而 worker 走的 `temporal_fold_batches` 里是 `DatasetArtifactStore.load_split(…,
      expected_fold_id=…)`——**带分片哈希校验与折 id 绑定**，那部分才是 7.6 min 的主体。
      所以 72.5 s 是"批次构造"、7.6 min 是"经校验的完整重建"，**7.6 min 不需要下修**。
      顺带，探针这套面板的其余量与生产记录逐项吻合，可作为它忠实的旁证：
      `train_batches = 953 / val_batches = 237`（本文件早先记的是 953 / 237 / 242），
      峰值 RSS **12.18 GiB**（`temporal_fold_batches` 注释里记的是 12.77 GiB，
      差值来自探针不返回 `validation_labels`、也不持有 store 的校验副本）。

      若第 1 项的降级超过约 1.5×，直接杀掉、用更少 worker 重启：`--resume` 把这件事的代价从「全部重来」降到
      「只重跑未完成的部分」。**预计时长（24 路，2026-09-11 依实测改写；原为 19 路 44 h/折、7.4 天）**：搜索
      1 轮 × 3 次拟合 = 25.8 h + 每 worker 一次 7.6 min 重建 + 最终拟合 1 轮 ≈ 8.6 h + 新一批 worker 的
      7.6 min ≈ **34.7 h/折，四折约 5.8 天**。若 MemAvailable 不足 466.0 GiB 而退回 19 路，搜索回到 4 轮
      × 8.6 h = 34.4 h，即 **约 43.5 h/折、四折约 7.3 天**——重建变快省下的那 4.3 小时仍然拿到，台阶那一份
      拿不到。

      **上面这段与本次新增的降级表口径不同，已合并（2026-09-12 02:40）。** 这段算了每 worker 一次
      7.6 min 的面板重建、但没算并发降级；新表算了降级、但没算重建。两者都对一半，合并后才是可发射的数：
      搜索 3 个拟合时 × 8.60 h × 1.18 = **30.4 h** + 最终拟合 8.60 × 1.05 = **9.0 h** + 两批 worker 的
      面板重建 2 × 7.6 min = **0.25 h** ≈ **39.7 h/折，四折 6.61 天**（退回 19 路则是 4 × 8.60 × 1.16 +
      9.0 + 0.25 = 49.2 h/折、**8.20 天**）。所以规划 M2 要按 **约 6.6 天**，不是 5.8 天；
      台阶结构不变，24 路仍比 19 路省 1.6 天。该段的「MemAvailable 不足 466.0 GiB」应为
      **480.4 GiB**（与命令里已改的 `--worker-memory-gb 17.1` 自洽；这个值经历过三次修正：
      466.0 是按 16.56 折算的、与命令行不自洽 → 467.1 与 16.6 自洽 → **480.4 与实测峰值 17.06 自洽**，
      见本节上面那块 ⚠ 标注）。

      **另需实测的两项，都不能从 M1 搬。**

      **【两项都已在 M2 首折实测，结论是工期要大幅上修：约 9.7–11.1 天，不是 6.17 天（2026-09-14 11:30）】**

      M2 于 **2026-09-13 18:54:56** 由链式监视器自动发射（M1 全网格通过六道门之后），
      24 路、钉 1 线程、`--worker-memory-gb 17.1`。首折遥测两项都读到了：

      **(b) 降级因子实测 = 1.437（中位），不是外推的 1.18，也远不是 M1 的 1.132。**
      测法与 M1 相同：取同一 (config, seed) 相邻两个 epoch 检查点的时间戳之差，
      按配置分组后与该格的 solo 实测值相比。**24 个配置全部落在 1.332–1.583 之间**（n=39 个差值样本）：

      | 格 | solo 单 epoch（探针实测） | 24 路实测中位 | 降级 |
      |---|---|---|---|
      | GRU/64（c0–c5） | 749.7 s | 1017–1146 s | 1.357–1.528 |
      | GRU/128（c6–c11） | 1182.2 s | 1639–1869 s | 1.387–1.581 |
      | TCN/64（c12–c17） | 918.1 s | 1298–1453 s | 1.413–1.583 |
      | TCN/128（c18–c23） | 1675.9 s | 2232–2533 s | 1.332–1.512 |
      | **全网格** | 均值 1131.5 s | 中位 1391 s | **中位 1.437、均值 1.460** |

      **"M1 的 1.132 不能搬给 M2"这句警告被实测证实了，而且方向是对的**：
      M2 的 worker 载荷是 M1 的 11 倍（4.16 GiB 批次张量对 0.381 GiB），
      24 路下 24 个 worker 同时流式读写约 100 GiB 张量，**访存带宽争用主导**，
      所以降级比 M1 在 41 路下的还差——**尽管 M2 的并发度只有 M1 的 59%**。
      并发度不是降级的自变量，**每 worker 的访存强度才是**。

      **比降级更要命的是第二项：配置级粒度下，一轮的时长由最慢的那一格决定。**
      24 配置 / 24 worker → 配置级 1 轮，每个 worker **串行跑该配置的 3 个 seed**。
      所以单折搜索 = `3 × 最慢格的单次拟合`，而最慢格是 **TCN/128**：
      实测 21–22 个 epoch × 约 2400 s ≈ **14.1 h/次拟合**（c18 的 seed 20260801
      于 19:03 起、09:15 结束，实测 14.2 h，与 21 × 2408 s = 14.05 h 吻合）。
      **单折搜索 = 3 × 14.1 ≈ 42.3 h**，而计划按"网格均值 per-fit"算的是 24.4 h——**低估 1.74 倍**。
      分解：降级 1.437/1.18 = **1.22×**，最慢格对网格均值 = **1.43×**，两者相乘 = 1.74×。

      **据此的 M2 工期**：单折 = 42.3 h 搜索 + 最终拟合（3 seed 并行、近乎 solo，
      胜者未知，取网格均值 22 epoch → 7.3 h；若胜者是 TCN/128 则 10.8 h、若 GRU/64 则 4.8 h）
      + 面板重建 0.3 h ≈ **49.9 h**（F1 口径）。四折按实测行数比 1.000/1.110/1.220/1.321 折算
      = **232 h ≈ 9.7 天 → 预计 09-23 11:00 前后**；
      若计入 M1 学到的"按行数折算会低估约 15%"，则 **267 h ≈ 11.1 天 → 09-24 22:00 前后**。
      **即 M2 的规划值应为约 10–11 天，本文件此前写的 6.17 天低估了约 60%。**

      **内存不是瓶颈，已核实**：父进程实测 RSS **14.74 GiB**（低于文档假设的 21.99），
      按行数折算到 F4 约 19.5 GiB；四折的护栏复查全部 `fits=True`，
      F4 的余量 12.5 GiB（`required 429.9` 对 `0.9 × 491.5 = 442.4`）。
      worker 实测 RSS 均值 **12.46 GiB**、最大 **17.05 GiB**——
      与我探针在生产路径上测的 **17.06 GiB** 相差 0.01 GiB，**申报值 17.1 是准的**。

      **24 路仍是内存上限下的最优选择，没有免费的加速**：makespan ≈ `ceil(72/W) × 最慢单次拟合 × deg(W)`，
      W=24 给 3 × 1.437 = 4.31；W=16 给 5 × deg(16) ≈ 6.5；W=12 给 6 × deg(12) ≈ 7.5——**都更差**。
      要真正提速只能动那 4.16 GiB 的批次张量本身（惰性构造/按需生成/对连续内存做视图），
      它同时降低内存与访存争用；但那是改一个已测试契约、且要重启 M2（损失已跑的 18 h），
      **属所有者裁决，本轮未动**。

      **(a) 8.60 h/拟合是旧折口径，但"按 ms/日 折算到新折"这条路不可靠——我试过，两次测量的基准根本不同。**
      8.60 h 来自 `1301.3 ms/train日 × 约 1220 训练日 × 约 18 epoch`，新折是 **967 训练日**，
      直觉上该往下修。但拿 M1 检验这个直觉时它**只对了一半**：M1 最小格旧测 333.7 ms/日、
      新测 `318.4 s / 967 日 = 329 ms/日`（−1.3%，看着可迁移）；M1 最大格旧测 388.2 ms/日、
      新测 `409.8 / 967 = 423.8 ms/日`（**+9.2%，反方向**）。原因是两次测量的分母不同——
      旧基准是 `split=train/year=2014` 的 **245 个训练日 + 60 个验证日**探针、按训练日归一，
      新数字是整折 967 训练日 + 237 验证日，**每日行数也不同**，所以 ms/日 跨折不可比。
      只引最小格那一个点就断言"ms/日 可迁移"是以偏概全，本文件不采用。
      **处置**：M2 的 8.60 h 保留为规划值并标注 ±20% 不确定度；发射前若要收窄，
      用与 M1 相同的方法（只读已发布数据集、`max_epochs=1`、走驱动自己的 `_train_cross_section_seed`
      对应物与 `_m2_model_factory`）在真实 F1 上实测 GRU 与 TCN 各一格。

      **(b) 降级因子必须实测，而且 M1 的 1.132× 明确不能搬给 M2。** M1 在 41 路实测 1.132×、
      远好于旧曲线外推的 1.27×，但 M1 的 worker 载荷只有 **0.381 GiB**、批次张量 0.27 GiB；
      M2 是 **4.16 GiB 载荷**（11 倍）、走 `temporal_fold_batches` 与 `history_length=20` 的时序批，
      访存争用不是同一个量级。把 1.132 搬给 M2 是没有依据的乐观，本文件继续按旧曲线的 1.18×@24 规划，
      并在首折实测后改写。测法与 M1 相同：**取同一 (config, seed) 相邻两个 epoch 检查点的时间戳之差**，
      不要用 epoch-1 墙钟（含 worker 启动与载荷反序列化；M1 实测那一项约 100 s，会把比值从 1.13
      虚抬到 1.43–1.51）。
- [x] **4.3 M3 分支 + 合成 smoke**（32 组合 ≤30 不成立时同 4.1 粗筛规则）。
- [ ] **4.4 M3 全量运行 + 判定归档**。

      **M3 发射包（2026-09-12 02:50 补；此前本项只有一行勾选框，没有任何可执行的参数）**

      **先纠一个极易踩的误读：M3 不是时序家族。** 本任务标题把 M2 与 M3 并列、开头那句又同时提到
      `TemporalTrainingBatch` 与 `FeatureBatch` 却没说哪个家族走哪条，读起来像是两者共用时序路径。
      实际核对（2026-09-12 02:45，读代码而非读文档）：`run_full_m3` 调的是
      `_run_full_cross_section(family="M3", …)`，与 `run_full_m1` **同一个函数**；
      `families.py` 里 `STATIC_TORCH_FAMILIES = {ResidualMLP, SetAttention}`，
      `TEMPORAL_TORCH_FAMILIES = {TemporalGRU, TemporalTCN}` 只含 M2 的两个架构。
      所以 **M3 走 `build_static_training_batches` 与 `run_cross_section_seed_task`，足迹属 M1 量级，
      不属 M2 量级**；也没有 M2 那个每 worker 7.6 min 的时序面板重建（静态批次构造在父进程只要 9.9 s）。
      这一条直接决定了下面的 worker 数：**M3 不受 M2 那个 24 路内存天花板约束**。

      **已核对的事实**：网格 **16 配置**（`m3_config_grid` 现场执行得出：`hidden{64,128} × heads{2,4} ×
      inducing_points{32,64} × blocks{1,2}`，dropout 固定 0.1、lr 固定 0.0003——yaml 的 64 组合超 E7 预算，
      按已记录的降维固定两个维度）；每折 16 × 3 seed = **48 次拟合**；
      前置 `FAMILY_ORDER_PREREQUISITES["M3"] = ("M1", "M2")`，所以 `--prior-report` **要同时给 M1 和 M2**。

      **内存：按 M1 的实测值声明 2.5 GiB/worker。** M1 在 41 路下实测 worker RSS 0.93 GiB 起步、
      训练中均值 1.27 GiB；M3 最大格 418,177 参数对 M1 最大格 404,481 参数（1.7 MB 对 1.6 MB fp32），
      模型本身在 16 GiB 量级的批次面前可以忽略，所以同一个 2.5 GiB 声明值适用、且留约 2 倍余量。
      现场调用 `worker_memory_budget`：48 路 = `48 × 2.5 + 0.6 = 120.6 GiB`，
      在 `MemAvailable = 482.1 GiB`（M1 正在跑时的读数）下 `fits=True`、`safe_workers = 173`。
      **M3 的约束是核数与轮数，内存完全不是问题。**

      **worker 数：48 是甜点，而且不是挑出来的整数——`48 = 16 × 3` 恰好用尽种子级并行，1 轮完美装箱。**
      轮数与粒度由驱动自己的 `search_granularity` / `granularity_makespans` 现场算出（打平取粗）：

      | W | gran | 粗 | 细 | 轮数 | 单折 | 四折（per-fit 2.31 h） | 内存 |
      |---|---|---|---|---|---|---|---|
      | 16 | config | 3 | 3 | 3 | 10.3 h | 41.1 h ≈ 1.71 天 | 40.6 GiB |
      | 24 | seed | 3 | 2 | 2 | 7.7 h | 30.6 h ≈ 1.28 天 | 60.6 GiB |
      | 32 | seed | 3 | 2 | 2 | 7.7 h | 31.0 h ≈ 1.29 天 | 80.6 GiB |
      | **48** | **seed** | 3 | **1** | **1** | **5.1 h** | **20.3 h ≈ 0.85 天** | 120.6 GiB |
      | 64 | seed | 3 | 1 | 1 | 5.1 h | 20.3 h ≈ 0.85 天（无改善） | 160.6 GiB |

      **64 路比 48 路一点没快**（都已是 1 轮），所以 48 是有用性的上限，不必去占更多核。
      串行反事实：192 次拟合 × 2.31 h = 444 h ≈ **18.5 天**，即 M3 与 M2 一样没有串行选项。

      **per-fit 2.31 h 是规划值、不是实测值，不确定度 ±20%**：由 M3 最大格旧测 430.6 ms/日 ×
      新折 967 训练日 × 20 epoch 折算。上面 (a) 项已经说明 ms/日 跨折不可比（M1 最小格 −1.3%、
      最大格 +9.2%，方向相反），所以这个折算只能当量级用。工期对它的敏感度是线性的：
      per-fit 2.0 / 2.31 / 2.63 h → 四折 **17.6 / 20.3 / 23.1 h**（0.73 / 0.85 / 0.96 天），
      即最坏情形也不到一天。**降级因子取 1.15**：这是从 M1 在 41 路的实测 1.132× **代理**过来的，
      而这次代理是站得住的——M3 与 M1 走同一条静态批次代码路径、模型规模同级，
      与 M2 那种"载荷差 11 倍、批类型不同"的情况不一样。仍标注为代理值，首折按 M1 的测法实测
      （相邻两个 epoch 检查点时间戳之差，不用 epoch-1 墙钟）。

      **per-fit 已在真实新折上实测，上面的 2.31 h 折算值被取代（2026-09-12 03:15，只读探针
      `/tmp/probe_newfold_cost.py`，日志 `/tmp/probe_static_newfold.log`）。**
      探针读的是已发布的 F1 数据集 `226a8dfc…`（与在跑的 `m1_full_v1` 同一个内容寻址 id）、
      因子序 `f00…f14` 取自载体 bundle 自己的 `feature_order.json`，批次构造走驱动自己的
      `_prepare_frame` 与 `build_static_training_batches`，模型走驱动自己的 `_m3_model_factory`。

      **探针保真度先验证过，再用它的数字**：同一探针跑 M1 config0 得到
      `epoch_s = 361.88`、checkpoint `sha256 = 85e9b4a35585865a…`，
      而**生产运行 `m1_full_v1` 里 config0/seed 20260801 的 epoch-1 检查点 sha256 正是
      `85e9b4a35585865a…`——逐位相同**；时间上 361.88 s 也落在生产实测的稳态区间 361.0–367.6 s 内。
      另一格 M1 最大格同样对得上：探针 454.60 s 对此前 solo 实测 409.8 s = **1.109×**，
      config0 是 361.88 / 318.4 = **1.137×**，两者都收敛到生产实测的 **1.132×**。
      三处独立测量互相印证，所以探针的争用环境是已标定的。

      | M3 最大格（`hidden 128 / heads 4 / points 64 / blocks 2`） | 值 |
      |---|---|
      | 参数量 | **418,433**（2026-09-05 基准记的是 418,177，差 **+256** = 输入投影从 14 因子变 15 因子，反证探针确实在 frozen_v2 上） |
      | 争用下单 epoch | 486.26 s（502.9 ms/训练日） |
      | 折算 solo 单 epoch | 486.26 / 1.132 = **429.6 s** |
      | **per-fit（20 epoch）** | **2.386 h** |

      即实测 2.386 h 对折算值 2.31 h 只差 **3.4%**、对旧口径 2.63 h 差 **−9.3%**——
      这次 ms/日 的跨折折算碰巧准，但上面 (a) 项的结论不变：它准是运气，不是可依赖的规则。

      **据实测重算的工期**：48 路 1 轮 → 单折 `2.386 × deg + 2.386 × 1.05`；
      deg 取实测代理 1.132 得 **20.8 h（0.87 天）**，取旧曲线 1.30 得 **22.4 h（0.93 天）**——
      因为只有 1 轮，降级的不确定度对总工期几乎无影响（区间仅 0.06 天）。
      24 路 2 轮则是 **31.6 h（1.32 天）**。所以 **48 路那个台阶值 0.45 天，是本项最划算的一步**。

      **这仍是上界**：2.386 h 是**最大格**的成本，16 格里更小的格子更便宜。M1 的实测尺寸敏感度是
      38.5 倍参数量只换来 1.29 倍时间，据此 M3 最小格大概便宜 20–25%，网格均值 per-fit 约 1.9–2.1 h、
      四折约 **17–19 h**。要精确到网格均值需要把 16 格都测一遍（约 2 小时探针），
      而对一个不到一天的运行不值得——**按 0.87 天上界规划即可**。

      **epoch 数按 M1 的实测均值 21.9 修正后（取代上面按 20 epoch 算的数）**：
      per-fit = `21.9 × 429.6 s = 2.61 h`（最大格），
      48 路 1 轮 → 四折 **22.8 h（0.95 天，deg 1.132）到 24.6 h（1.02 天，deg 1.30）**；
      24 路 2 轮 → **34.6 h（1.44 天）**。
      即 M3 的规划值应为 **约 1 天（48 路）**，48 路那一步仍值约 0.45 天。
      上面那段"网格均值约 1.9–2.1 h、四折 17–19 h"是按 20 epoch 说的，
      同样上调约 9.5% 到 **约 19–21 h**；作为上界的结论不变。

      **M3 的 `--worker-memory-gb 2.5` 也已实测确认够用，不再是"借用 M1 的足迹"（2026-09-12 07:47，
      只读探针 `/tmp/probe_static_worker_peak.py`，日志 `/tmp/probe_static_m3.log`、
      `/tmp/probe_static_m1.log`）。** 动机正是 M2 那次教训：拿一个家族的足迹去给另一个家族定 worker 数，
      是让多天运行按"半个 worker 生命周期"定容量的方式。M3 的 SetAttention 注意力激活此前从未被测过。
      测法模拟生产静态 worker（建批次 → **释放帧** → 以 batches-only 常驻为基线 → 训练 → 再测），
      并**同时跑 M1 最大格作对照**，以便把架构差异与载荷差异分开：

      | 格 | 参数量 | batches-only 基线 VmRSS | 训练后 VmRSS | **训练增量** | VmHWM |
      |---|---|---|---|---|---|
      | M3 `128/4/64/2`（SetAttention） | 418,433 | 2.61 GiB | 2.93 GiB | **+0.32 GiB** | 3.92（未动） |
      | M1 `256/3`（ResidualMLP，对照） | 404,481 | 2.61 GiB | 2.78 GiB | **+0.17 GiB** | 3.92（未动） |

      **结论：M3 只比同路径的 M1 重 0.15 GiB**（增量 1.9 倍，但绝对量极小），
      而按生产口径 M1 worker 实测常驻是 1.26–1.33 GiB，所以 M3 约 **1.5 GiB**，
      对 2.5 GiB 的申报值有约 1.7 倍余量。
      **而且这个申报值对 48 路工作点根本不敏感**：现场调用 `worker_memory_budget`（`MemAvailable` 515 GiB、
      parent 0.6 GiB）扫申报值 2.5 / 3 / 5 / 7 / 9 GiB，48 路都 `fits=True`
      （`safe_workers` 依次为 185 / 154 / 92 / 66 / 51）；**只有真实峰值超过约 9.6 GiB 才会让 48 路被拒**，
      而实测增量是 0.32 GiB——差一个数量级以上。
      所以 M3 的内存这一项**可以从风险清单上划掉**，这与 M2 的情况正好相反
      （M2 的申报值直接决定 24 路那个台阶能否保住）。

      **一个必须说明的混淆，免得有人拿 2.61 GiB 当生产数字**：探针的 batches-only 基线是 2.61 GiB，
      高于生产 M1 worker 实测的 1.33 GiB，因为探针在同一进程里先载入了 230 万行的原始帧与 prepared 帧、
      再 `del` + `gc.collect()`，而 **CPython/PyTorch 的分配器不会把已释放的内存立刻还给 OS**，
      所以 VmRSS 停在高位；生产 worker 是在**全新进程**里反序列化 0.381 GiB 的批次载荷，没有这份残留。
      **因此这里可用的是"增量"（+0.32 对 +0.17），不是绝对值**——而增量正是问题所在
      （M3 的注意力激活是否远重于 M1）。两格的 VmHWM 都停在 3.92 GiB 未动，
      也说明训练期没有超过帧载入期的高水位。

      **顺带交叉验证了 M3 的成本模型**：同一次运行、同一争用条件下，
      M3 最大格 epoch **452.8 s** 对 M1 最大格 **428.7 s**，比值 **1.056**；
      而本节前面那两个数（M3 solo 429.6 s、M1 solo 409.8 s，来自两次独立测量）给出的比值是 **1.048**——
      **相差不到 1%**。即 M3 的 per-fit 2.386 h（20 epoch）/ 2.61 h（21.9 epoch）站得住。

      ```bash
      cd /data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2   # 必需：下面用的是相对路径 scripts/...
      KEY=$(cat /data/zhangyudong/qrv2_research_artifacts/secrets/selection_evidence_hmac.key)
      setsid nohup env QRV2_SELECTION_EVIDENCE_HMAC_KEY="$KEY" QRV2_PREDICTION_EVIDENCE_HMAC_KEY="$KEY" \
        QRV2_MODEL_REGISTRY_ROOT=/data/zhangyudong/qrv2_research_artifacts/model_registry \
        /data/zhangyudong/.conda/envs/quantyd/bin/python scripts/run_real_candidate_training.py \
        --family M3 \
        --baseline-artifacts /data/zhangyudong/qrv2_research_artifacts/baseline_v2/baseline_oos \
        --baseline-report /data/zhangyudong/qrv2_research_artifacts/baseline_v2/report.json \
        --workspace /data/zhangyudong/qrv2_research_artifacts/m3_v1 \
        --report /data/zhangyudong/qrv2_research_artifacts/m3_v1/report.json \
        --factor-source frozen_v2 \
        --prior-report M1=<已裁决的 M1 报告> --prior-report M2=<M2 报告> \
        --parallel-workers 48 --torch-threads 1 --worker-memory-gb 2.5 --resume \
        > /data/zhangyudong/qrv2_research_artifacts/m3_v1.log 2>&1 &
      ```

      两个 `--prior-report` 都必须是 `mode=full`、family 对得上、`oos.all_passed=true` 的报告，
      否则顺序守卫在发射第一秒就拒。**M1 那一份填哪份是所有者裁决项**：`m1_parallel_v1`（单配置载体、
      已归档、六道门全过）与 `m1_full_v1`（全 27 格、进行中）都可能满足条件，
      而这两者代表不同的搜索规模与不同的谱系证据。工作区同样必须**新开**（bundle 不可变）。

      **M3 不会重演 M1 那六个阻塞项——这一条是核对过的，不是推定的（2026-09-12 02:55）。**
      M1 两次真实运行都是四折训练跑完、然后在评价阶段死掉、写不出 `report.json`；六个阻塞项里
      第 4 项（card family）打的是全部 M2、第 5 项（bundle config 字段）打的是 M2 的 GRU 与 TCN
      **以及 M3**、第 6 项（loader 缺 `TEMPORAL_TCN` 条目与工厂分支）让 TCN 即使赢了也无法服务。
      所以 M3 至少在两个轴上是真的暴露过的。核对结果：
      `artifact_loader.py` 的 code-version 映射含 `SetAttention.model_family -> "set-attention-1"`、
      config 映射含 `SetAttentionConfig`、家族分派里有 SetAttention 的工厂分支，
      且这三处**都以模型类自己的 `model_family` 属性为键**，不是手抄字面量；
      `test_a_bundle_written_by_the_drivers_own_writers_loads_back` 的参数化覆盖
      **M1 / M3 / M2-GRU / M2-TCN 四个载体全部**，用的是驱动自己的 `torch_model_card` 与
      `torch_bundle_config` 的真实输出；另有
      `test_every_allowlisted_family_has_a_factory_behind_it` 专门钉住"映射里有、分派里没有"
      这一类漂移——正是当年让 `TEMPORAL_TCN` 无法加载的那个形状。
      **仍要如实说明的是**：这些是测试层的保证，M3 从未在真实数据上跑过一折；
      本项的 per-fit 与降级因子也因此都还是规划值。

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

      **进展注记（2026-09-12 05:25）：本项的触发条件已满足并已对 M0R 执行，但本项保持未勾选。**
      `m0r_v2` 的模型 `sha256:32315c89…` 在签名 registry 记录里的状态就是
      **`WALK_FORWARD_PASSED`**（事件链 `DRAFT → WALK_FORWARD_PASSED`，2026-09-08T19:51:31Z，
      理由 `all G1-G6 walk-forward gates passed`，绑定证据哈希 `64b413507bf4…`），
      六道门全过、G4 = **+0.009149095118820283**。据此已产出两份文档：
      `docs/reports/2026-09-12-m0r-real-candidate-walk-forward.md`（判定文档，
      由 `summarize_real_candidate_walk_forward.py` 直接生成；同一输入三处渲染 md5 全同
      `2bfa79833db1bcb9b5bb8036ceac48d1`，渲染器确定性已验）与
      `docs/reports/2026-09-12-m0r-shadow-approval-preconditions.md`（前置清单，
      逐项对齐 F 计划 F6.1 的五项人工核验、并摘录批准后即生效的六条约束）。
      **未启动 SHADOW、未做任何状态转换、未写 registry 与数据库**——`WALK_FORWARD_PASSED → SHADOW`
      按 F6.1 是人工动作。
      **不勾选的理由**：本项写的是"若有候选"，而 M1/M2/M3 尚无 `WALK_FORWARD_PASSED` 的
      全网格候选（M1 全网格进行中，预计 09-13 06:45 前后出报告）；等三者齐备再判定本项完成。
      **另需所有者裁断一个解释问题**（前置清单 §4）：F 计划第 15 行的正式运营前置写的是
      "至少一个**复杂候选模型**处于 `WALK_FORWARD_PASSED`"，而 M0R 是 LightGBM 的
      `LGBM_RANKREG` 载体、不是 torch 架构。它算不算"复杂候选"会移动 60 日影子计时的起点，
      而 60 有效日又卡在 2026 标签成熟与 F9 之前，**是有时间代价的决定**，本计划不代为预设。

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

## 勾选审计与实测偏差记录（2026-09-07 17:35）：折结构改造 F01–F06 → F1–F4（v2 契约，owner 批准）

**起因**：v2 冻结集契约（`factor_set_frozen_v2.json`，owner 于 2026-09-07 在因子会话批准）把
walk-forward 评价折从设计文档 §7 的 **F01–F06（5 年训练 × 6 折，测试年 2020–2025）** 改为
**F1–F4（4 年训练 × 4 折，测试年 2022–2025）**，理由是 2014–2016 漂移最剧、4 年窗足够。
**这是设计文档 §7 折网格的一次已批准偏离**；本节记录实现与后果。

**核心改动（TDD，先改直接测试到失败再实现）**：

- `datasets/splits.py`：`annual_folds()` 生成器 `range(2014,2020)` → `range(2017,2021)`、
  训练窗 5 年 → 4 年、折 ID `F0x` → `Fx`（**换名使新旧数据集缓存键天然不相交，无需清旧缓存**）；
  `inner_folds()` 由「恰好 5 年 × (2,3,4) 三内折」改为「恰好 4 年 × (2,3) 两内折」。
  **锁定 FINAL 折不变**（5 年 2020–2024、2026 锁定测试）：v2 只重定义评价折、未涉及生产门，
  保持 FINAL 不动以不触碰锁定测试治理（链 9 域）——**4y/5y 训练窗在 FINAL 上的不一致留作开放问题**。
- `baselines/cross_validation.py`：`expanding_inner_folds` 由「3 内折 / 5 年 / (2,3,4)」
  改为「2 内折 / 4 年 / (2,3)」，错误文案同步。
- `datasets/selection.py`：`inner_fold_count` 3 → 2。**设计注记**：`minimum_positive_inner_folds=2`
  在 2 内折下 = **两折一致**（旧 2-of-3 的保守类比；1-1 平票意味着方向不可靠，故取全票）。

**消费方**：`run_real_candidate_training.py`（ORDINARY_FOLDS）、`run_real_walk_forward.py`
（docstring + expected + `--folds` 默认）、三个 summarize 脚本（REQUIRED_FOLDS + **REQUIRED_YEARS
2020–2025 → 2022–2025** + G3 注记 + yearly-complete 行）、两个 verify_baseline 脚本（EXPECTED_FOLDS）。
`walk_forward_common.py` / `walk_forward_factor_daily.py` / `audit_legacy_train_coverage.py`
**本就动态调 `annual_folds()`，无需改**。

**一处语义修正**：`verify_baseline_rerun_consistency.compare_reports` 的折集合**从"当前契约"
改为"以参照报告自己的 `folds` 字段为准"**（回退 EXPECTED_FOLDS）。理由：它比对的是
「重跑 vs 归档参照」，归档的 F01–F06 报告应能复现为 F01–F06，而不该被新契约判为缺失。
（`check_artifacts` 仍用 EXPECTED_FOLDS——**新**报告必须满足**当前**契约。）

**归档证据的处理**：`2026-09-04-real-walk-forward-criterion-303.{json,md}` **原样保留**（历史记录）。
在新契约下重渲该归档报告，判其为**过时**（F01–F06 不再满足 REQUIRED_FOLDS、折表 F1–F4 全 MISSING、
verdict 由 SATISFIED 翻为 NOT SATISFIED）；新渲染落为
`2026-09-04-real-walk-forward-criterion-303.v2-renderer.md`，归档 golden 测试改为对它**逐字节**断言
（保留"渲染器对真实归档工件逐字节稳定"这一最强钉子，v1 文件不动）。

**验证（全部实测）**：`pytest tests -q` → **1710 passed / 45 skipped**；`ruff check src tests` 清；
`mypy src` 158 文件清。爆炸半径实测远小于字符串 grep 的表象——多数测试只把 fold_id 当标签，
真正结构性断裂集中在 3 个核心文件 + 折元组脚本 + 直接测试。

**后续（按依赖排序，非承诺）**：
1. **因子源**：`FROZEN_FACTOR_SET_V1`(17) → v2(15)；基线需单独的 9 稠密子集（角色感知因子源）。
2. **基线按新折 + 9 稠密因子重算**（G4 标尺，约 17h）——旧的 0.0864 legacy 基线随之作废。
3. **M0 全量运行**（15 因子 × F1–F4，CPU 数小时）。
4. **§5.1 协调**：补物化范围 2014–2019 → 2017–2019（冻结集计划 Task 3.0，**属因子会话**，本会话不动）。

## 勾选审计与实测偏差记录（2026-09-08 01:17）：v2 基线已重算 + 修掉第三处折契约漂移（G3/G5 假 FAIL）

**v2 基线重算已完成**（`run_real_walk_forward.py --folds F1,F2,F3,F4 --factor-source frozen_v2_baseline`，
9 稠密因子 × F1–F4）。结果：**mean RankIC 0.08706、ICIR 0.7507、HAC t 12.75、逐年 2022–2025 全为正、
有效日 969**；逐折赢家 F1–F3 = B2(Ridge)、F4 = B3(LightGBM)。**注意它显著强于旧标尺需要的程度**：
对比旧 legacy 基线（0.0864 / t=13.1），换 9 稠密因子后**水平略升**——这把 G4 的 0.005 增量门槛
放在了一根更硬的标尺上。

**过程中修掉第三处折契约漂移**（前两处是 walk_forward.py 与 search.py 的硬编码 F01–F06，
已随 dbf6c30 修）：`evaluate_oos_gates` 的 `required_years` 默认是**硬编码 `(2020,...,2025)`**——
旧折结构的测试年。v2 基线的逐年 IC 只有 2022–2025，于是 2020/2021 取到 None → `valid_years=False`
→ **G3/G5 算成 NaN → `observed_value=None` → FAIL**，而逐年明明全为正（G3 应为 1.0、G5 应为 0.28）。
**这是"重述一个会漂移的常量"导致的假失败**——运行时报告曾据此记 `all_passed=false`。

**修法（与 walk_forward 同一原则：派生而非重述）**：`evaluate_oos_gates` 的 `required_years`
**缺省时从 `annual_folds()` 派生**（普通折的测试年，F1–F4 → 2022–2025），调用方测门逻辑时显式传。
`test_gates.py` 的 22 处门逻辑测试改为经 `_gates(...)` 辅助显式传旧年份集（与折契约解耦、行为不变）；
新增 1 条回归测试（v2 式逐年证据下缺省门必须 PASS G3/G5 且有真观测值）。
**用保存的基线预测 + 对齐标签重评估**（脚本只读、命中缓存）：G1–G6 **全过**，主报告 `report.json`
的门字段已用修正值合并（其余汇总逐位一致，自校验通过）。

**对 M0 的直接含义**：G4 标尺 = `baseline_mean_rank_ic = 0.08706`（比旧的 0.0864 略硬）。
**验证**：`pytest tests -q` 1716 passed / 45 skipped；`ruff src/tests` 清；`mypy src` 158 文件清。

## 勾选审计与实测偏差记录（2026-09-08 03:39）：M0 可跑满 15 因子（NaN + 线程 + 三处撞出的 bug）

**M0 全量运行前置已做**：

1. **NaN 处理（方案 a：LightGBM 原生）**：`LGBMRankerBaseline` 加 `allow_nan`——`True` 时把 NaN 交给
   LightGBM 原生分裂（"下跌日无值"本身是信息、**最贴合 v2"禁止数据层插补"**，比 FoldPreprocessor 的
   0 填充更忠实）；默认 `False`（B3 的 9 稠密因子无 NaN、契约逐位不变）。M0 拟合与 bundle 重建都传 `True`。
   3 条测试钉住：默认拒 NaN、allow_nan 能在稀疏 NaN 特征上拟合+预测、allow_nan 仍拒 inf。
2. **num_threads 控制**：`LGBMHyperparameters.num_threads`（默认 1=B3 契约）。实测 `num_threads=1`
   时单个拟合 >15 分钟（全量 300 次拟合 25h+，不可行）；提到 **M0_NUM_THREADS=16**（`deterministic=true`
   保留）约 **2 分钟/拟合**。1 条测试钉住**固定线程数下确定性**（同种子两次逐字节一致）。

**过程中撞出并修掉的 bug（都是 M0 作为第一个走到评估步的候选才暴露的）**：

- **预测证据密钥缺口**：`evaluate_model_walk_forward` 用独立的 `QRV2_PREDICTION_EVIDENCE_HMAC_KEY` 验签，
  而驱动只用 `QRV2_SELECTION_EVIDENCE_HMAC_KEY` 签预测 → 跑到评估步才崩。M1 崩得早、从没走到这步。
  **运行口径**：两个 env 都设为同一信任根密钥（设计是"单一信任根"）。
- **`model_walk_forward.MODEL_FAMILIES` 漏注册 M0**（它是与驱动**重复的常量**，只有 M1/M2/M3）：
  补 `M0` 与 `_MODEL_CARD_FAMILIES["M0"]={"M0","LGBM_RANKER"}`。
- **bundle `dataset_id` 写错**：`run_full_m0` 把裸 `dataset_id` 写进 model_card，而
  `_require_verified_model_bundle` 与测试约定都要求 `card.dataset_id == selection.training_dataset_id`
  （`dataset_id:train`）。改为写 `train_dataset_id`。**注意**：`_run_full_cross_section`（M1/M3）与
  `run_full_m2` 有**同样的约定不一致**（bundle 写裸 id、`load_inference_artifact` 也期望裸 id，
  与评估检查冲突），但它们被 裁决 2/3 挡住、从没走到评估步——**记为 M1–M3 重启时的必修项，本轮未动**。
- **`--folds` 在全量运行里被接受但被忽略**（`_run_full_cross_section`/`run_full_m0` 固定跑 `ORDINARY_FOLDS`）：
  **latent bug，已记待修**；本次"单折 smoke"因此实际是全量 4 折运行。

**验证**：`pytest tests -q` **1720 passed / 45 skipped**（1716 + 本轮 4 条）；`ruff src/tests` 清；
`mypy src` 158 文件清。M0 干净重跑（修正后代码 + 双密钥 + 缓存数据集）进行中。
