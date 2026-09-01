# Quant Research V2 E — Models and Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在同一不可变数据集和基线评价框架上，依次实现 M1 Residual MLP、M2 GRU/TCN、M3 Set Attention，并仅在有证据时实现 M4 经济关系图；通过受限搜索、3 seeds、组合排序损失和全 OOS 双重门槛决定模型是否进入影子阶段。

**Architecture:** 模型共享强类型 batch、prediction 和 artifact contract；每个交易日横截面等权。训练器只看 train，early stopping/超参只看 validation，test 由冻结 evaluator 单独加载。模型预测统一转为当日横截面 score/rank，再与 fold-matched best simple baseline 比较。M4 的边只能来自点时行业/指数经济关系，绝不来自隐藏状态相似度或全样本收益。

**Tech Stack:** Python 3.11、PyTorch 2.6/CUDA 12.4、NumPy/Pandas/PyArrow、scikit-learn、4×RTX 4090、pytest/Hypothesis。

---

## 前置与非目标

- 前置：D 计划完成，F01–F06 manifests、B0–B3 与每折 best simple baseline 已冻结。
- 执行顺序不可跳跃：M1 → M2 → M3；M4 必须通过本计划 E9 的条件审批。
- 不以 2026 数据/标签选模型；FINAL 只训练到 2025 validation，本计划内模型状态最多到 `WALK_FORWARD_PASSED`，`SHADOW` 由 F 计划在独立部署审批后设置。
- 不把 turnover 作为主损失；未来若加入，必须 ID 对齐、可微、权重 ≤5%，并另行修改批准规格。
- 不因神经网络复杂而降低相对简单基线的 ΔRankIC 门槛。

## 文件清单

**Create:**

```text
research_v2/configs/models/residual_mlp.yaml
research_v2/configs/models/temporal.yaml
research_v2/configs/models/set_attention.yaml
research_v2/configs/models/economic_graph.yaml
research_v2/configs/training/default.yaml
research_v2/src/research_v2/models/base.py
research_v2/src/research_v2/models/residual_mlp.py
research_v2/src/research_v2/models/temporal.py
research_v2/src/research_v2/models/set_attention.py
research_v2/src/research_v2/models/economic_graph.py
research_v2/src/research_v2/models/components.py
research_v2/src/research_v2/training/batches.py
research_v2/src/research_v2/training/losses.py
research_v2/src/research_v2/training/sampler.py
research_v2/src/research_v2/training/trainer.py
research_v2/src/research_v2/training/search.py
research_v2/src/research_v2/training/seeds.py
research_v2/src/research_v2/training/artifacts.py
research_v2/src/research_v2/evaluation/model_evaluator.py
research_v2/src/research_v2/registry/model_registry.py
research_v2/src/research_v2/cli/models.py
research_v2/src/research_v2/cli/registry.py
research_v2/migrations/005_economic_graph_tables.dos
research_v2/tests/unit/models/test_base.py
research_v2/tests/unit/models/test_residual_mlp.py
research_v2/tests/unit/models/test_temporal.py
research_v2/tests/unit/models/test_set_attention.py
research_v2/tests/unit/models/test_economic_graph.py
research_v2/tests/unit/training/test_batches.py
research_v2/tests/unit/training/test_losses.py
research_v2/tests/unit/training/test_sampler.py
research_v2/tests/unit/training/test_trainer.py
research_v2/tests/unit/training/test_search.py
research_v2/tests/unit/training/test_artifacts.py
research_v2/tests/unit/evaluation/test_model_evaluator.py
research_v2/tests/unit/registry/test_model_registry.py
research_v2/tests/unit/cli/test_registry.py
research_v2/tests/property/models/test_permutation_equivariance.py
research_v2/tests/property/training/test_loss_invariants.py
research_v2/tests/leakage/test_training_data_access.py
research_v2/tests/leakage/test_model_selection_scope.py
research_v2/tests/golden/data/model_batch.pt
research_v2/tests/golden/data/expected_model_outputs.json
research_v2/tests/golden/test_models.py
research_v2/tests/integration/models/test_training_smoke.py
research_v2/tests/integration/models/test_model_walk_forward.py
```

## Task E1: 模型、batch 与 prediction 契约

**Files:**

- Create: `research_v2/src/research_v2/models/base.py`
- Create: `research_v2/src/research_v2/training/batches.py`
- Create: `research_v2/tests/unit/models/test_base.py`
- Create: `research_v2/tests/unit/training/test_batches.py`

- [x] **E1.1 写失败测试：所有维度都有业务键**

```python
@dataclass(frozen=True)
class FeatureBatch:
    signal_dates: tuple[date, ...]
    ts_codes: tuple[tuple[str, ...], ...]
    features: torch.Tensor        # [D, Nmax, F], F 为 15–60 个原始因子
    feature_missing: torch.Tensor # [D, Nmax, F]
    stock_mask: torch.Tensor       # [D, Nmax]


@dataclass(frozen=True)
class TrainingBatch:
    inputs: FeatureBatch
    target_raw: torch.Tensor       # [D, Nmax]
    target_rank: torch.Tensor      # [D, Nmax]
```

测试 D/N/F、mask、日期、代码长度任一不匹配即失败；padding 股票不得出现在 prediction。推理只构建不含 target 的 `FeatureBatch`，因此无需伪造 label tensor。

- [x] **E1.2 定义 TemporalBatch**

`TemporalFeatureBatch.features` 形状 `[D, Nmax, L, F]`，另含同形 feature-missing mask 和 `[D,Nmax,L]` history mask；训练 wrapper 再附 target。每个 endpoint 只包含 `<= signal_date` 的 20 个交易日，不足 20 日可 padding，但有效历史少于 15 日的样本不参与 loss。

- [x] **E1.3 定义 QuantModel protocol**

```python
class QuantModel(nn.Module, Protocol):
    model_family: str
    def forward(self, batch: FeatureBatch | TemporalFeatureBatch) -> torch.Tensor: ...  # [D, Nmax]
    def artifact_signature(self) -> ModelSignature: ...
```

输出是连续 score，不在模型内部选择 top-K；模型不得保存训练数据引用。

- [x] **E1.4 定义稳定 prediction 转换**

只保留 `stock_mask=true`，输出 `signal_date, ts_code, raw_score`；当日 percentile rank 使用 average ties，最终稳定排序 `signal_date, score DESC, ts_code ASC`。非 finite score 触发 blocker。

- [x] **E1.5 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/models/test_base.py tests/unit/training/test_batches.py -q`

Expected: shape、mask、history cutoff 和 prediction key 测试通过。

```bash
git add research_v2/src/research_v2/models/base.py research_v2/src/research_v2/training/batches.py research_v2/tests/unit/models/test_base.py research_v2/tests/unit/training/test_batches.py
git commit -m "feat(v2): define keyed model batch contracts"
```

## Task E2: 可微组合损失

**Files:**

- Create: `research_v2/src/research_v2/training/losses.py`
- Test: `research_v2/tests/unit/training/test_losses.py`
- Test: `research_v2/tests/property/training/test_loss_invariants.py`

- [x] **E2.1 写 IC loss 数值测试**

每个有效日期对预测 score 与预先计算的 target percentile rank 做 Pearson correlation：`L_ic = 1 - corr(score, target_rank)`；横截面有效数 <30 或任一边零方差的日期不进入均值。完全同向接近 0，完全反向接近 2。

- [x] **E2.2 写 pairwise ranking loss**

每日期从有效股票中按 `sha256(run_seed|date|code_i|code_j)` 稳定抽最多 4096 对；仅保留 label 差绝对值位于当日前 50% 的 informative pairs。`L_pair = softplus(-sign(y_i-y_j)*(s_i-s_j))`，日期等权。

- [x] **E2.3 写 Huber loss**

对当日去均值 raw return，使用 `SmoothL1Loss(beta=0.01)`；每日期先平均、再跨日期平均，避免大横截面日期权重更高。mask/padding/invalid label 不参与。

- [x] **E2.4 固化权重和可选小网格**

默认 `0.60*L_ic + 0.30*L_pair + 0.10*L_huber`。只允许 validation 比较三个预注册组合：`(0.7,0.2,0.1)`、`(0.6,0.3,0.1)`、`(0.5,0.4,0.1)`；三者之和必须为 1，Huber 固定 0.1。

- [x] **E2.5 性质测试**

验证股票 permutation 不改变 loss；预测加常数不改变 IC/pair；正比例缩放不改变 IC 但可改变 pair/Huber；padding 值任意改变不影响 loss；所有日期无效时明确失败而不是返回 0/NaN。

- [x] **E2.6 梯度测试并提交**

Run: `cd research_v2 && pytest tests/unit/training/test_losses.py tests/property/training/test_loss_invariants.py -q`

Expected: 数值参考、finite gradient、mask 和 permutation 测试通过。

```bash
git add research_v2/src/research_v2/training/losses.py research_v2/tests/unit/training/test_losses.py research_v2/tests/property/training/test_loss_invariants.py
git commit -m "feat(v2): add cross-sectional ranking losses"
```

## Task E3: 日期均衡 sampler 与可复现训练器

**Files:**

- Create: `research_v2/configs/training/default.yaml`
- Create: `research_v2/src/research_v2/training/sampler.py`
- Create: `research_v2/src/research_v2/training/seeds.py`
- Create: `research_v2/src/research_v2/training/trainer.py`
- Test: `research_v2/tests/unit/training/test_sampler.py`
- Test: `research_v2/tests/unit/training/test_trainer.py`
- Test: `research_v2/tests/leakage/test_training_data_access.py`

- [x] **E3.1 固化训练默认值**

```yaml
max_epochs: 100
early_stopping_patience: 10
early_stopping_metric: validation_mean_rank_ic
optimizer: AdamW
learning_rate: 0.001
weight_decay: 0.0001
gradient_clip_norm: 1.0
mixed_precision: bf16
seeds: [20260801, 20260802, 20260803]
min_valid_stocks_per_date: 30
```

- [x] **E3.2 写日期等权 sampler 测试**

batch 以完整交易日为最小单位；不同 N 不改变日期被抽概率。大型日期可在模型内部 padding/induced attention，不能把股票 chunk 当独立 loss 日期。epoch 对每个 train date 恰访问一次，顺序由 seed 决定。

- [x] **E3.3 实现 reproducibility setup**

设置 Python/NumPy/Torch/CUDA seeds、deterministic algorithms 和 cuDNN flags；记录 GPU 型号/driver/library。若某算子无法 deterministic，启动时失败并列出算子，不静默降级。

- [x] **E3.4 实现 Trainer 状态机**

`INITIALIZED → TRAINING → VALIDATING → CHECKPOINTED → COMPLETED`，异常转 `FAILED`。每 epoch 保存 train loss components、validation RankIC、coverage、lr、gradient norm、显存峰值；只按 validation mean RankIC 保存 best checkpoint。

- [x] **E3.5 强制数据访问 role**

Trainer constructor 只接受 train/validation `DatasetView`；任何 TEST/LOCKED role 立即失败。mock repository 记录所有 query，测试断言训练期间无 test/2026 请求。

- [x] **E3.6 实现 GPU lease**

每训练进程显式领取一个 GPU ID，默认单 GPU，不跨卡 DDP；lease 记录 run/trial/seed，进程退出释放。最多四 trial 并行，但同一 trial 的结果不依赖调度顺序。

- [x] **E3.7 CPU/CUDA smoke test 并提交**

Run: `cd research_v2 && pytest tests/unit/training/test_sampler.py tests/unit/training/test_trainer.py tests/leakage/test_training_data_access.py -q`

Expected: determinism、early stopping、best checkpoint、role guard 通过。

```bash
git add research_v2/configs/training/default.yaml research_v2/src/research_v2/training/sampler.py research_v2/src/research_v2/training/seeds.py research_v2/src/research_v2/training/trainer.py research_v2/tests/unit/training/test_sampler.py research_v2/tests/unit/training/test_trainer.py research_v2/tests/leakage/test_training_data_access.py
git commit -m "feat(v2): train models reproducibly by date"
```

## Task E4: M1 Residual MLP

**Files:**

- Create: `research_v2/configs/models/residual_mlp.yaml`
- Create: `research_v2/src/research_v2/models/components.py`
- Create: `research_v2/src/research_v2/models/residual_mlp.py`
- Test: `research_v2/tests/unit/models/test_residual_mlp.py`

- [x] **E4.1 写 shape/mask/gradient 失败测试**

任意原始因子数 `F=15..60`、可变 N、padding 下输出 `[D,Nmax]`；模型在 input projection 前拼接 transformed values 与 missing mask，实际输入宽度为 `2F`。padding score 被 mask；前向/反向 finite；eval mode 同输入精确一致。

- [x] **E4.2 实现 residual block**

默认结构：`concat(features, feature_missing.float())` → input projection → 2 个 `LayerNorm → Linear → GELU → Dropout → Linear` residual blocks → scalar head。默认 hidden 128、dropout 0.1；不用 BatchNorm（避免横截面/批次统计泄漏）。

- [x] **E4.3 固化受限搜索空间**

```text
hidden_dim: 64, 128, 256
num_blocks: 1, 2, 3
dropout: 0.0, 0.1, 0.2
learning_rate: 0.0003, 0.001
loss_weights: 三个预注册组合
```

Trial scheduler 只抽取预先按 hash 排序的前 30 个组合，不根据 test 扩展搜索。

- [x] **E4.4 写过拟合小样本测试**

在 3 日×40 股、单个线性信号的 fixture 上，100 小步后 combined loss 显著下降且 prediction 与 target rank 正相关；该测试只验证实现，不设置真实业绩期望。

- [x] **E4.5 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/models/test_residual_mlp.py -q`

Expected: shape、determinism、gradient 与小样本过拟合通过。

```bash
git add research_v2/configs/models/residual_mlp.yaml research_v2/src/research_v2/models/components.py research_v2/src/research_v2/models/residual_mlp.py research_v2/tests/unit/models/test_residual_mlp.py
git commit -m "feat(v2): implement residual MLP candidate"
```

## Task E5: M2 GRU/TCN 时序候选

**Files:**

- Create: `research_v2/configs/models/temporal.yaml`
- Create: `research_v2/src/research_v2/models/temporal.py`
- Test: `research_v2/tests/unit/models/test_temporal.py`

- [x] **E5.1 写 causal window 测试**

endpoint T 的序列严格为交易日 `[T-19, ..., T]`；扰动 T+1 及以后不改变输出；股票上市前/padding history mask 正确；代码缺日不把另一股票行接入序列。

- [x] **E5.2 实现 GRU encoder**

每只股票独立编码 20×(2F) 的 transformed value + missing mask，使用 masked packed sequence，取最后有效 hidden，经 residual head 输出 score。默认 hidden 128、1 layer、dropout 0.1。

- [x] **E5.3 实现 causal TCN encoder**

3 个 causal depthwise-separable Conv1d residual blocks，kernel=3、dilation=1/2/4；左 padding，绝不使用右侧未来。masked global last-state head 输出 score。

- [x] **E5.4 固化 M2 选择空间**

family `{GRU, TCN}`，hidden `{64,128}`，dropout `{0.0,0.1,0.2}`，lr `{0.0003,0.001}`，loss weights 三组；按稳定 hash 取最多 30 trials。validation 决定 family/config，test 不参与。

- [x] **E5.5 写 temporal permutation 测试**

交换股票顺序后输出按同样交换；交换时间顺序必须改变输出；改变 masked history 不影响输出；GRU/TCN 都能在小因果 fixture 过拟合。

- [x] **E5.6 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/models/test_temporal.py -q`

Expected: causal、mask、equivariance、gradient 与过拟合测试通过。

```bash
git add research_v2/configs/models/temporal.yaml research_v2/src/research_v2/models/temporal.py research_v2/tests/unit/models/test_temporal.py
git commit -m "feat(v2): implement causal GRU and TCN candidates"
```

## Task E6: M3 Set Attention 横截面候选

**Files:**

- Create: `research_v2/configs/models/set_attention.yaml`
- Create: `research_v2/src/research_v2/models/set_attention.py`
- Test: `research_v2/tests/unit/models/test_set_attention.py`
- Test: `research_v2/tests/property/models/test_permutation_equivariance.py`

- [x] **E6.1 写 permutation equivariance 失败测试**

随机 permutation 同日股票，输出必须按相同 permutation 变化；反向恢复后误差 `<1e-6`。padding 股票的值/数量变化不能改变有效股票输出。

- [x] **E6.2 实现 induced set attention**

将 transformed values 与 missing mask 拼接后做 input projection，再使用 2 个 ISAB，每层 64 个 learned inducing points、4 heads、hidden 128，最后以每个证券表示加全局 PMA summary 的 residual head 输出 score。复杂度 O(N×64)，不构造 N² 全 attention、不使用证券位置编码或 ts_code embedding。

- [x] **E6.3 保证日期隔离**

attention 仅在同一个 signal date 的有效股票间执行；不同日期打包只用于 GPU 并行，mask 防止跨日交互。日期顺序改变不改变逐日输出。

- [x] **E6.4 固化 M3 搜索空间**

hidden `{64,128}`、heads `{2,4}`、inducing points `{32,64}`、blocks `{1,2}`、dropout `{0.0,0.1}`、lr `{0.0003,0.001}`，按稳定 hash 限 30 trials；hidden 必须整除 heads。

- [x] **E6.5 显存和极端横截面测试**

在 CUDA 可用时用 `N=6000, raw F=60, effective input=120, D=1` 前向/反向，峰值显存必须 <20 GiB，给 24 GiB 卡保留余量；CPU CI 使用 N=256 等价逻辑。OOM 不能通过裁掉低分股票解决，只能调 batch date 数/inducing config。

- [x] **E6.6 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/models/test_set_attention.py tests/property/models/test_permutation_equivariance.py -q`

Expected: equivariance、日期隔离、mask、shape 和资源约束测试通过。

```bash
git add research_v2/configs/models/set_attention.yaml research_v2/src/research_v2/models/set_attention.py research_v2/tests/unit/models/test_set_attention.py research_v2/tests/property/models/test_permutation_equivariance.py
git commit -m "feat(v2): implement scalable set-attention candidate"
```

## Task E7: 受限超参搜索、3 seeds 与 artifact

**Files:**

- Create: `research_v2/src/research_v2/training/search.py`
- Create: `research_v2/src/research_v2/training/artifacts.py`
- Test: `research_v2/tests/unit/training/test_search.py`
- Test: `research_v2/tests/unit/training/test_artifacts.py`
- Test: `research_v2/tests/leakage/test_model_selection_scope.py`

- [x] **E7.1 实现可审计 TrialScheduler**

对批准搜索空间做 canonical product，以 `sha256(model_family|fold|config_json)` 排序，最多执行 30 个 config trials；不采用会根据 test 反馈新增 trial 的自适应搜索。trial 状态和 GPU lease 写 experiment audit。

- [x] **E7.2 定义超参选择**

每个 config 先用 seed 20260801 在 3 inner folds评估，按 mean inner validation RankIC 选前 3；再在 outer train 完整训练、outer validation 比较，tie-break 为 validation HAC t-stat、较少参数、config hash。最终 config 用 3 个固定 seeds 各训练一次。

- [x] **E7.3 定义 seed ensemble**

每个 seed 输出当日 percentile rank，ensemble score 为三者等权均值；不得挑最好 seed。报告 seed mean/std、最差 seed 和 ensemble；任一 seed 失败则整个 config incomplete，不以两 seed 替代。

- [x] **E7.4 实现 immutable artifact bundle**

```text
artifacts/models/{model_id}/
  model_card.json
  config.json
  signature.json
  feature_order.json
  preprocessing.json
  seed-20260801.pt
  seed-20260802.pt
  seed-20260803.pt
  validation_predictions.parquet
  checksums.sha256
```

只保存 `state_dict` 和已知类/config，不 pickle 任意 Python 对象。加载先验证每个 hash、schema/model code version、feature order 和 dataset id。

- [x] **E7.5 写选择泄漏测试**

改变 outer test/2026 labels 不改变 trial list、inner metrics、chosen config、seed artifacts；改变 outer validation 允许改变 chosen config但不能重训已完成 trial 的 train input。

- [x] **E7.6 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/training/test_search.py tests/unit/training/test_artifacts.py tests/leakage/test_model_selection_scope.py -q`

Expected: trial cap、稳定顺序、3-seed ensemble、artifact tamper 和泄漏测试通过。

```bash
git add research_v2/src/research_v2/training/search.py research_v2/src/research_v2/training/artifacts.py research_v2/tests/unit/training/test_search.py research_v2/tests/unit/training/test_artifacts.py research_v2/tests/leakage/test_model_selection_scope.py
git commit -m "feat(v2): audit model search and seed ensembles"
```

## Task E8: 候选模型 walk-forward 评价

**Files:**

- Create: `research_v2/src/research_v2/evaluation/model_evaluator.py`
- Create: `research_v2/src/research_v2/cli/models.py`
- Test: `research_v2/tests/unit/evaluation/test_model_evaluator.py`
- Test: `research_v2/tests/integration/models/test_model_walk_forward.py`

- [x] **E8.1 定义命令**

```text
qr-v2 model train --family M1 --fold F01
qr-v2 model train --family M2 --fold F01
qr-v2 model train --family M3 --fold F01
qr-v2 evaluate walk-forward --experiment-id sha256:... --compare best-simple
```

family 必须按 registry 顺序启用；命令拒绝 FINAL test 和缺失 best-simple 的 fold。

- [x] **E8.2 每折冻结候选后再评价 test**

训练/验证完成后写 signed selection record；evaluator 独立加载冻结 artifact 与 test manifest，生成 prediction_daily、metric_daily、metric_summary。训练进程本身无 test repository capability。

- [x] **E8.3 使用 fold-matched baseline 计算增量**

每个测试日同时要求 candidate 与当折 best-simple 有效，先在相同 date/code coverage 上各算 RankIC，再计算年度/总体差。主 G4 为各折 test 日拼接后的 candidate mean RankIC 减 baseline mean RankIC，不用两个不同样本集指标相减。

- [x] **E8.4 运行 D 计划全部门槛**

模型只有 G1–G6 全 PASS 才获得 `WALK_FORWARD_PASSED` 候选资格；同时报告相对 B0/B1/B2/B3 各自增量、参数量、训练时长和最差 seed。任何缺 fold、缺 seed、score collapse、年度集中均失败。

- [x] **E8.5 端到端小规模测试**

用合成 F01/F02 小数据依次训练 M1/M2/M3 smoke config，验证 artifact → prediction → matched baseline → gates；测试只验证编排，不将合成业绩作为模型通过证据。

- [x] **E8.6 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/evaluation/test_model_evaluator.py -q`

Run: `cd research_v2 && QRV2_INTEGRATION=1 pytest tests/integration/models/test_model_walk_forward.py -q -m integration`

Expected: fold/test isolation、coverage matching、G1–G6 和状态候选通过。

```bash
git add research_v2/src/research_v2/evaluation/model_evaluator.py research_v2/src/research_v2/cli/models.py research_v2/tests/unit/evaluation/test_model_evaluator.py research_v2/tests/integration/models/test_model_walk_forward.py
git commit -m "feat(v2): evaluate model candidates against simple baselines"
```

## Task E9: M4 条件门与点时经济关系图

**Files:**

- Create only after approval: `research_v2/migrations/005_economic_graph_tables.dos`
- Create only after approval: `research_v2/configs/models/economic_graph.yaml`
- Create only after approval: `research_v2/src/research_v2/models/economic_graph.py`
- Create only after approval: `research_v2/tests/unit/models/test_economic_graph.py`
- Modify only after approval: `research_v2/tests/contract/db/test_schema_contracts.py`

- [ ] **E9.1 评估准入条件，未通过即停止本 Task**

M3 必须在 F01–F06 全 OOS 同时满足 G1–G6，且相对 best-simple ΔRankIC≥0.005；另外至少 5/6 测试年相对 baseline 增量为正。若不满足，记录 `M4_NOT_AUTHORIZED`，不创建 migration/code/config。

- [ ] **E9.2 由人工批准经济边定义**

只允许 YesLab 的 PIT `sw_ind_code` 与 `index_component_table`：证券在 signal date 所属申万行业、当日已生效指数成分。禁止收益相关、未来成分、模型 hidden top-K 或 label-derived edges。

- [ ] **E9.3 写 005 migration 与 contract**

新增 `economic_membership_scd(entity_type, entity_id, ts_code, valid_from, valid_to, source_observation_date, available_date, run_id, data_version, created_at, source_watermark_id, record_hash)`；唯一键 `entity_type, entity_id, ts_code, valid_from`。迁移遵循 A 计划 checksum/幂等/测试库先行。

- [ ] **E9.4 写 PIT/未来扰动测试**

成分/行业变更只从 available date 生效；修改未来 membership 不改变过去图；同一信号日只连接当日有效实体。无行业股票保留 self representation，不强行映射 OTHER 大团。

- [ ] **E9.5 实现稀疏实体聚合模型**

先按行业/指数实体对成员表示做 masked mean/attention pooling，再把证券自己的表示与其实体 summaries 拼接进 residual head；不用 stock-stock clique 或 N² adjacency。实体 membership 完全由 batch 显式传入。

- [ ] **E9.6 使用与 M1–M3 相同搜索/评价预算**

最多 30 configs、3 seeds、F01–F06、同一 G1–G6；M4 必须额外相对 M3 ΔRankIC≥0.002 才有替代价值，否则保留 M3。

- [ ] **E9.7 运行条件分支验证并提交**

Run: `cd research_v2 && pytest tests/unit/models/test_economic_graph.py tests/contract/db/test_schema_contracts.py -q`

Expected: PIT membership、无隐藏图、稀疏聚合、permutation equivariance 与 schema contract 通过。

```bash
git add research_v2/migrations/005_economic_graph_tables.dos research_v2/configs/models/economic_graph.yaml research_v2/src/research_v2/models/economic_graph.py research_v2/tests/unit/models/test_economic_graph.py research_v2/tests/contract/db/test_schema_contracts.py
git commit -m "feat(v2): add approved PIT economic graph candidate"
```

## Task E10: 模型 registry 状态机、黄金输出与全量验证

**Files:**

- Create: `research_v2/src/research_v2/registry/model_registry.py`
- Create: `research_v2/src/research_v2/cli/registry.py`
- Test: `research_v2/tests/unit/registry/test_model_registry.py`
- Test: `research_v2/tests/unit/cli/test_registry.py`
- Create: `research_v2/tests/golden/data/model_batch.pt`
- Create: `research_v2/tests/golden/data/expected_model_outputs.json`
- Create: `research_v2/tests/golden/test_models.py`

- [x] **E10.1 写状态机失败测试**

唯一合法主路径：

```text
DRAFT → WALK_FORWARD_PASSED → SHADOW → LOCKED_TEST_APPROVED
→ LOCKED_TEST_PASSED → PRODUCTION → RETIRED
```

`LOCKED_TEST_APPROVED → LOCKED_TEST_FAILED`、任意非 RETIRED 状态 → RETIRED 合法；禁止 DRAFT 直接 SHADOW/PRODUCTION、SHADOW 直接 PRODUCTION、FAILED 回到 PASSED。

- [x] **E10.2 实现 transition guards**

`WALK_FORWARD_PASSED` 要求 E8 G1–G6 artifact；`SHADOW` 由 F 计划创建且不在本 Task 直接设置；locked/production transitions 要求 F 计划专用审批 evidence。每次 transition append-only，保存 actor、previous/new、reason、evidence hashes、timestamp。

- [x] **E10.3 实现历史门槛晋级 CLI**

```text
qr-v2 registry promote --model-id sha256:MODEL_ID --to WALK_FORWARD_PASSED --evidence artifacts/reports/walk_forward.json --dry-run
qr-v2 registry promote --model-id sha256:MODEL_ID --to WALK_FORWARD_PASSED --evidence artifacts/reports/walk_forward.json
```

命令重算 evidence hash 与 G1–G6，不相信报告中的布尔字段；本计划只允许目标 `WALK_FORWARD_PASSED`。F 计划扩展同一 CLI 以处理带 shadow/locked 专用 evidence 的后续转换。

- [x] **E10.4 建黄金 batch 输出**

固定 tiny batch/config/state_dict，保存 M1/M2/M3 CPU eval outputs 与 tolerance；验证模型重构、artifact load 和 feature order。黄金 fixture 不包含从真实标签学到的权重。

- [x] **E10.5 运行完整 E 计划验证**

Run: `cd research_v2 && ruff check src tests && mypy src`

Run: `cd research_v2 && pytest tests/unit/models tests/unit/training tests/unit/evaluation/test_model_evaluator.py tests/unit/registry tests/unit/cli/test_registry.py tests/property/models tests/property/training tests/leakage/test_training_data_access.py tests/leakage/test_model_selection_scope.py tests/golden/test_models.py -q`

Run: `cd research_v2 && QRV2_INTEGRATION=1 pytest tests/integration/models -q -m integration`

Expected: 全部退出码 0；若 E9 未获授权，则 M4 测试明确 skip reason=`M4_NOT_AUTHORIZED`，不能伪装通过。

- [x] **E10.6 提交并请求审查**

```bash
git add research_v2/src/research_v2/registry/model_registry.py research_v2/src/research_v2/cli/registry.py research_v2/tests/unit/registry/test_model_registry.py research_v2/tests/unit/cli/test_registry.py research_v2/tests/golden
git commit -m "feat(v2): govern model evidence and transitions"
```

随后使用 `superpowers:requesting-code-review`，重点审查损失日期权重、因果序列、set equivariance、超参预算、test isolation、基线覆盖匹配和 registry guards。

## E 计划完成标准

- [x] M1/M2/M3 共享同一 batch/loss/evaluator contract，均通过 shape、mask、causal/permutation 和 artifact 测试。
- [x] 每 family 每 fold 搜索不超过 30 configs；最终配置固定 3 seeds 并等权 ensemble。
- [x] 训练/选模进程无法访问 test/2026；test 只由冻结 evaluator 加载。
- [x] 所有候选与 fold-matched best simple baseline 在相同 coverage 上比较。
- [x] 只有 G1–G6 全通过模型可进入 `WALK_FORWARD_PASSED`。
- [x] M4 未满足条件时没有 migration/code 副作用；满足后也只使用 PIT 经济边。

## 状态核对（2026-09-01）

已按现有代码与测试逐项核对并勾选完成步骤（测试文件按演进后的实际命名存在，如
`test_pre_registered_search.py`、`test_cross_sectional_losses.py`、
`test_frozen_oos_evaluator.py` 等）。全量单元套件通过（969），`ruff check` 与
`mypy src` 干净。2026-09-01 补齐：E5.5 temporal 置换等变测试
（`tests/unit/models/test_temporal_gru.py`）、E4.4 小样本过拟合与 M1 置换等变
测试（`tests/unit/models/test_residual_mlp.py`）、E5.6 temporal GRU 小样本
过拟合测试、E6.5 极端横截面资源测试（CPU 等价 N=256 + CUDA N=6000 峰值
显存 <20 GiB，已在 4090 实测）与日期隔离测试；完成标准第 1 条随之闭合。
2026-09-01 再补 E8.5/E8.6：`tests/integration/models/test_model_walk_forward.py`
对 M1（Residual MLP）、M2（Temporal GRU）、M3（Set Attention）smoke 配置执行
合成数据端到端链路——3 seeds 训练→不可变 checkpoint→三种子模型 bundle
（覆盖写被拒）→`load_inference_artifact` 白名单重建→冻结 TEST 评价→
G1–G6 门控报告，断言门控键集/有效天数/基线增量有限与重复评价确定性；
`QRV2_INTEGRATION=1` 门控运行 3 通过（18s）。仍开放的条目及原因：

- E9.1–E9.7：M4 条件门未获授权，按设计保持零代码/零迁移副作用。
- 另注（2026-09-01 更新）：训练侧时序批次装配已实现——
  `src/research_v2/training/temporal_batches.py` 按因果窗口（≤signal_date 的
  最近 `history_length` 个观测、左对齐、缺日自动收缩）构造
  `TemporalTrainingBatch`，`minimum_valid_history_length`（与
  `configs/models/temporal.yaml` 的 15 一致）之下样本不参与 loss，2026 锁与
  业务键/目标校验与静态路径同构；测试 6 项（因果性、最小历史、缺日、锁年、
  无效目标、配置守卫）。推理侧时序装配已于 2026-09-01 补齐：
  `src/research_v2/inference/builder.py::build_label_free_temporal_inference_batch`
  从 universe + 因子长表为信号日重建因果左对齐历史窗口（最近
  `history_length` 个观测、可用日守卫、历史不足 `minimum_valid_history_length`
  的证券剔除且全缺时 fail-closed），预测器类型放宽为
  `PredictionInputBatch`；测试 6 项（窗口装配、未来行不变性、历史下限、
  可用日拒绝、配置守卫、GRU 贯通预测帧）。2026-09-01 再补族感知分派
  `build_inference_batch_for_family`（横截面族→同日帧、时序族→历史窗口、
  未知族 fail-closed；含三种子集成贯通测试 5 项），日频运行接线在
  runtime backend 建立时直接调用该入口。
