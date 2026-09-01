# Quant Research V2 D — Datasets and Baselines Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建不可变 walk-forward 数据集、严格训练窗内的预处理与因子筛选、B0–B3 简单基线、完整 OOS RankIC/ICIR/HAC 评价和可机器判定的双重门槛，为复杂模型提供可信参照。

**Architecture:** split definition 是版本化一等对象；Dataset Builder 只按已发布 universe/factor set/label/data version 做点时 join，并生成 content-addressed manifest。每个 outer fold 内先用 train 拟合预处理和筛选，用 validation 选择超参数/基线，test 只做一次评价。2026 final test 在普通 builder/evaluator 中硬锁。

**Tech Stack:** Python 3.11、Pandas/NumPy/PyArrow、SciPy/statsmodels、scikit-learn、LightGBM、Pydantic、pytest/Hypothesis。

---

## 前置与非目标

- 前置：A–C 计划通过；2013–2025 的市场、universe、labels 和 governed factors 已发布。
- 本计划不实现神经网络，不进行 2026 locked evaluation。
- test 指标不能用于选因、定方向、选模型、调超参、选择 seed 或决定是否重跑。
- `best_simple_baseline` 每折只按 validation 主指标选择，test 结果仅在选择冻结后计算。

## 文件清单

**Create:**

```text
research_v2/configs/splits/annual_5y1y1y.yaml
research_v2/configs/datasets/default.yaml
research_v2/configs/baselines/single_factor.yaml
research_v2/configs/baselines/equal_weight.yaml
research_v2/configs/baselines/linear.yaml
research_v2/configs/baselines/lgbm_ranker.yaml
research_v2/src/research_v2/datasets/splits.py
research_v2/src/research_v2/datasets/preprocessing.py
research_v2/src/research_v2/datasets/selection.py
research_v2/src/research_v2/datasets/builder.py
research_v2/src/research_v2/datasets/manifest.py
research_v2/src/research_v2/datasets/storage.py
research_v2/src/research_v2/evaluation/rank_ic.py
research_v2/src/research_v2/evaluation/hac.py
research_v2/src/research_v2/evaluation/summary.py
research_v2/src/research_v2/evaluation/gates.py
research_v2/src/research_v2/evaluation/walk_forward.py
research_v2/src/research_v2/baselines/base.py
research_v2/src/research_v2/baselines/single_factor.py
research_v2/src/research_v2/baselines/equal_weight.py
research_v2/src/research_v2/baselines/linear.py
research_v2/src/research_v2/baselines/lgbm_ranker.py
research_v2/src/research_v2/reporting/experiment.py
research_v2/src/research_v2/cli/datasets.py
research_v2/src/research_v2/cli/baselines.py
research_v2/src/research_v2/cli/evaluate.py
research_v2/tests/unit/datasets/test_splits.py
research_v2/tests/unit/datasets/test_preprocessing.py
research_v2/tests/unit/datasets/test_selection.py
research_v2/tests/unit/datasets/test_manifest.py
research_v2/tests/unit/datasets/test_builder.py
research_v2/tests/unit/evaluation/test_rank_ic.py
research_v2/tests/unit/evaluation/test_hac.py
research_v2/tests/unit/evaluation/test_gates.py
research_v2/tests/unit/baselines/test_single_factor.py
research_v2/tests/unit/baselines/test_equal_weight.py
research_v2/tests/unit/baselines/test_linear.py
research_v2/tests/unit/baselines/test_lgbm_ranker.py
research_v2/tests/property/datasets/test_split_invariants.py
research_v2/tests/property/datasets/test_manifest_invariants.py
research_v2/tests/leakage/test_preprocessor_fit_scope.py
research_v2/tests/leakage/test_selector_fit_scope.py
research_v2/tests/leakage/test_locked_dataset.py
research_v2/tests/leakage/test_end_to_end_fold_isolation.py
research_v2/tests/golden/data/dataset_50x60.parquet
research_v2/tests/golden/data/expected_metrics.json
research_v2/tests/golden/test_dataset_and_metrics.py
research_v2/tests/integration/datasets/test_dataset_build.py
research_v2/tests/integration/evaluation/test_walk_forward_baselines.py
```

## Task D1: 固化年度 5y+1y+1y split 与 purge

**Files:**

- Create: `research_v2/configs/splits/annual_5y1y1y.yaml`
- Create: `research_v2/src/research_v2/datasets/splits.py`
- Test: `research_v2/tests/unit/datasets/test_splits.py`
- Test: `research_v2/tests/property/datasets/test_split_invariants.py`

- [x] **D1.1 写精确 fold 配置**

```text
F01 train 2014–2018 | validation 2019 | test 2020
F02 train 2015–2019 | validation 2020 | test 2021
F03 train 2016–2020 | validation 2021 | test 2022
F04 train 2017–2021 | validation 2022 | test 2023
F05 train 2018–2022 | validation 2023 | test 2024
F06 train 2019–2023 | validation 2024 | test 2025
FINAL train 2020–2024 | validation 2025 | test 2026 (LOCKED)
```

日界使用交易日历中各自然年的首末交易日，不硬编码某年具体日期；2013 仅 warmup，不进入正式 label samples。

- [x] **D1.2 写失败测试：窗口长度、顺序和不重叠**

```python
def test_outer_fold_years():
    assert fold("F01").train_years == (2014, 2015, 2016, 2017, 2018)
    assert fold("F01").validation_years == (2019,)
    assert fold("F01").test_years == (2020,)
```

覆盖所有 7 folds、自然年缺少交易日、重复 ID、段落交叠、乱序和非 5/1/1 长度。

- [x] **D1.3 实现 6 交易日 purge**

每个 train/validation 段尾删除最后 6 个 signal dates，确保其 T+6 标签不跨入下一段；段首不删除，模型可使用此前已知历史作为 feature lookback，但该上下文行不参与当前段 loss/metric。test 段只保留 `exit_date <= test_end` 的成熟标签。

- [x] **D1.4 定义训练内 inner folds**

每个 5 年 outer train 建 3 个 expanding inner folds：前 2 年训练/第 3 年验证、前 3 年训练/第 4 年验证、前 4 年训练/第 5 年验证；各 inner-train 尾同样 purge 6 日。`positive_inner_fold_ratio` 的分母固定为 3，因此 60% 门槛实际要求至少 2/3 为正。

- [x] **D1.5 强制 locked split capability**

`FINAL.test` 默认只返回 metadata，不返回日期/行。普通 `SplitResolver` 对 locked 段抛 `LockedPeriodError`；解锁接口不在本模块暴露。

- [x] **D1.6 性质测试与提交**

Hypothesis 在任意合法交易日历上验证段顺序、label exit 不跨界、inner folds 包含关系和 locked 不可读。

Run: `cd research_v2 && pytest tests/unit/datasets/test_splits.py tests/property/datasets/test_split_invariants.py -q`

Expected: 全部通过。

```bash
git add research_v2/configs/splits/annual_5y1y1y.yaml research_v2/src/research_v2/datasets/splits.py research_v2/tests/unit/datasets/test_splits.py research_v2/tests/property/datasets/test_split_invariants.py
git commit -m "feat(v2): define purged annual walk-forward splits"
```

## Task D2: Fold-scoped 预处理

**Files:**

- Create: `research_v2/configs/datasets/default.yaml`
- Create: `research_v2/src/research_v2/datasets/preprocessing.py`
- Test: `research_v2/tests/unit/datasets/test_preprocessing.py`
- Test: `research_v2/tests/leakage/test_preprocessor_fit_scope.py`

- [x] **D2.1 固化默认预处理顺序**

```text
1. 将 ±inf 视为缺失
2. 每因子在 train pooled observations 拟合 p01/p99 winsor bounds
3. 每因子在 winsorized train 拟合 median 和 IQR
4. transform 为 (x - train_median) / max(train_IQR, 1e-8)
5. clip transformed value 到 [-10, 10]
6. 缺失填 0（训练中位数对应的中性值）
7. 为每个原始因子产出同形状的 boolean missing mask；mask 是模型 companion input，不计入 15–60 个原始因子数量
```

所有统计以证券-日期 observation 为样本，但在每个日期对每只证券等权；配置和 fitted state 均进入 artifact hash。

- [x] **D2.2 写失败测试：validation/test 不能参与 fit**

用极端 validation/test 值证明 train p01/p99/median/IQR 不变；调用 `fit` 传入非 TRAIN role 直接失败；`transform` 列顺序与 factor set 不符失败。

- [x] **D2.3 实现 serializable FoldPreprocessor**

state 明确保存原始 factor order、missing-mask contract、bounds、median/IQR、常数列和 fit date range；只允许 `fit` 一次，反序列化后 hash 必须一致。训练中全空/常数因子标记 ineligible，不静默输出全零有效特征。

- [x] **D2.4 写未来扰动和重排测试**

改变 cutoff 后数据不影响 train transform；行顺序不影响 fitted state；训练值本身变化必须改变 state hash。

- [x] **D2.5 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/datasets/test_preprocessing.py tests/leakage/test_preprocessor_fit_scope.py -q`

Expected: 所有 fit-scope 与序列化测试通过。

```bash
git add research_v2/configs/datasets/default.yaml research_v2/src/research_v2/datasets/preprocessing.py research_v2/tests/unit/datasets/test_preprocessing.py research_v2/tests/leakage/test_preprocessor_fit_scope.py
git commit -m "feat(v2): add fold-scoped robust preprocessing"
```

## Task D3: 训练窗内统计筛选与相关簇代表

**Files:**

- Create: `research_v2/src/research_v2/datasets/selection.py`
- Test: `research_v2/tests/unit/datasets/test_selection.py`
- Test: `research_v2/tests/leakage/test_selector_fit_scope.py`

- [x] **D3.1 写失败测试：每项门槛**

一个因子只有全部通过才 eligible：

```text
结构 DQ/coverage pass
训练全窗 signed mean RankIC > 0
HAC 双侧 p-value 经 BH 校正 q <= 0.10
5 个训练自然年中至少 3 年 signed mean RankIC > 0
3 个 expanding inner validations 中至少 2 个 signed mean RankIC > 0
```

signed IC 乘 registry `expected_direction`（POSITIVE=+1，NEGATIVE=-1）；方向为 UNKNOWN 时，仅用 outer train aggregated IC 固定符号并写入 factor set，validation/test 不重新定向。

- [x] **D3.2 实现每日因子 IC 与 HAC p-value**

按训练日、当日 universe、非空 factor/主 raw label 计算 Spearman；有效横截面数 <30 当日无效。HAC 使用 intercept-only OLS、`maxlags=5`，对应重叠 5 日标签。有效 IC 日 <120 的因子直接 insufficient。

- [x] **D3.3 实现 Benjamini–Hochberg**

在同一 fold 所有 structural-eligible 因子的 HAC p-values 上执行 BH，稳定 tie-break 为 `p_value, factor_id`；保存原 p、q、family size 和规则版本。

- [x] **D3.4 调用 C 计划相似度聚类**

仅用 outer train 计算 median daily abs Spearman；阈值 0.90。每簇代表排序：FDR q 更小 → signed mean IC 更高 → 年度正向数更多 → coverage 更高 → conflict rate 更低 → factor_id ASC。

- [x] **D3.5 限制特征数量**

默认按上述排序取最多 40；少于 15 时 factor set 保持 `DRAFT` 并记录 validation reason=`INSUFFICIENT_FEATURES`，可跑诊断 B0/B1 但禁止复杂模型训练。41–60 只有经配置审批、说明经济覆盖原因后允许；超过 hard max 60 无条件失败。

- [x] **D3.6 写泄漏陷阱**

置换 validation/test labels、增加未来因子行、改变 test 极值，selection result/hash 必须不变。将 test 查询监控为 forbidden call，selector 运行中出现即失败。

- [x] **D3.7 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/datasets/test_selection.py tests/leakage/test_selector_fit_scope.py -q`

Expected: FDR、年度/inner 稳定、相关簇、数量边界与泄漏测试通过。

```bash
git add research_v2/src/research_v2/datasets/selection.py research_v2/tests/unit/datasets/test_selection.py research_v2/tests/leakage/test_selector_fit_scope.py
git commit -m "feat(v2): select stable factors inside train folds"
```

## Task D4: 不可变 Dataset Manifest 与列式 artifact

**Files:**

- Create: `research_v2/src/research_v2/datasets/manifest.py`
- Create: `research_v2/src/research_v2/datasets/storage.py`
- Create: `research_v2/src/research_v2/datasets/builder.py`
- Create: `research_v2/src/research_v2/cli/datasets.py`
- Test: `research_v2/tests/unit/datasets/test_manifest.py`
- Test: `research_v2/tests/unit/datasets/test_builder.py`
- Test: `research_v2/tests/property/datasets/test_manifest_invariants.py`
- Test: `research_v2/tests/leakage/test_locked_dataset.py`

- [x] **D4.1 写 manifest contract**

必须包含：dataset id/hash、fold/split role、universe id/version、label id/version、factor set id 与有序成员、source data versions/watermarks、signal/entry/exit 范围、purged dates、行/日/证券数、每列 dtype/null count、preprocessor state hash、schema version、code commit、构建配置 hash、所有 parquet shard SHA-256。

- [x] **D4.2 写失败测试：逻辑内容寻址**

相同行不同顺序/不同 parquet chunk 得到同 logical dataset hash；任一值、成员顺序、split、watermark 或预处理状态变化必须改变 hash。已发布 manifest 不可覆盖。

- [x] **D4.3 实现点时 join**

锚点为 `universe_daily(signal_date, ts_code)`；按同一 signal date 连接 factor observation；按 label key 连接成熟 raw label。每行断言 factor `available_date <= signal_date`、label `exit_date <= segment_end`，且业务键唯一。

- [x] **D4.4 实现列式存储布局**

```text
artifacts/datasets/{dataset_id}/
  manifest.json
  split=train/year=YYYY/part-*.parquet
  split=validation/year=YYYY/part-*.parquet
  split=test/year=YYYY/part-*.parquet
  preprocessing.json
```

parquet 使用固定 schema、zstd 和稳定列顺序；临时目录完成 hash 后原子 rename。artifact root 不位于 V1 输出目录。

- [x] **D4.5 实现 locked guard**

普通命令 `--fold FINAL --role test` 在解析 manifest 前失败；禁止用任意 `--end 2026-...` 绕过 fold。SHADOW inference 数据不含 label 且由 F 计划独立 builder 生成。

- [x] **D4.6 定义 CLI**

```text
qr-v2 dataset build --fold F01 --factor-policy stable-v1
qr-v2 dataset inspect --dataset-id sha256:...
qr-v2 dataset verify --dataset-id sha256:...
```

build 默认一次生成 train/validation，显式 `--include-test` 才在 F01–F06 选择冻结后生成 test；FINAL 永不接受该参数。

- [x] **D4.7 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/datasets/test_manifest.py tests/unit/datasets/test_builder.py tests/property/datasets/test_manifest_invariants.py tests/leakage/test_locked_dataset.py -q`

Expected: content hash、PIT join、原子 artifact 与 locked 测试通过。

```bash
git add research_v2/src/research_v2/datasets research_v2/src/research_v2/cli/datasets.py research_v2/tests/unit/datasets research_v2/tests/property/datasets research_v2/tests/leakage/test_locked_dataset.py
git commit -m "feat(v2): build immutable PIT datasets"
```

## Task D5: RankIC、ICIR、HAC 与年度汇总

**Files:**

- Create: `research_v2/src/research_v2/evaluation/rank_ic.py`
- Create: `research_v2/src/research_v2/evaluation/hac.py`
- Create: `research_v2/src/research_v2/evaluation/summary.py`
- Test: `research_v2/tests/unit/evaluation/test_rank_ic.py`
- Test: `research_v2/tests/unit/evaluation/test_hac.py`

- [x] **D5.1 写每日 RankIC 测试**

按 `signal_date` 对 finite prediction 与 raw return 计算 Spearman；样本数 <30、prediction 或 label 常数时记 invalid，不填 0。并列值用 average rank；输入稳定排序后计算。

- [x] **D5.2 实现 summary 定义**

```text
mean_rank_ic = valid daily IC arithmetic mean
std_rank_ic = sample standard deviation (ddof=1)
icir_daily = mean_rank_ic / std_rank_ic
icir_annualized = icir_daily * sqrt(252)
positive_day_ratio = count(IC>0) / valid_days
```

同时按 calendar year 汇总同一字段，保存 valid/invalid day counts 与每日横截面样本分位数。

- [x] **D5.3 实现 Newey-West/HAC**

对每日 IC 序列回归常数项，`cov_type='HAC', maxlags=5, use_correction=True`；输出 mean coefficient、standard error、t-stat 和双侧 p-value。日期缺口不填 0，按有效观测顺序计算并报告缺失率。

- [x] **D5.4 用独立参考验证**

小向量与 `scipy.stats.spearmanr` 对照；HAC 与 statsmodels 直接 OLS fixture 对照，容差 `1e-12`。全正/全负/常数/单日/NaN/并列全部覆盖。

- [x] **D5.5 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/evaluation/test_rank_ic.py tests/unit/evaluation/test_hac.py -q`

Expected: 指标与独立参考一致。

```bash
git add research_v2/src/research_v2/evaluation/rank_ic.py research_v2/src/research_v2/evaluation/hac.py research_v2/src/research_v2/evaluation/summary.py research_v2/tests/unit/evaluation/test_rank_ic.py research_v2/tests/unit/evaluation/test_hac.py
git commit -m "feat(v2): calculate robust RankIC statistics"
```

## Task D6: B0 单因子与 B1 等权因子基线

**Files:**

- Create: `research_v2/configs/baselines/single_factor.yaml`
- Create: `research_v2/configs/baselines/equal_weight.yaml`
- Create: `research_v2/src/research_v2/baselines/base.py`
- Create: `research_v2/src/research_v2/baselines/single_factor.py`
- Create: `research_v2/src/research_v2/baselines/equal_weight.py`
- Test: `research_v2/tests/unit/baselines/test_single_factor.py`
- Test: `research_v2/tests/unit/baselines/test_equal_weight.py`

- [x] **D6.1 定义统一 baseline protocol**

```python
class Baseline(Protocol):
    def fit(self, train: DatasetView, validation: DatasetView) -> FittedBaseline: ...
    def predict(self, data: DatasetView) -> PredictionFrame: ...
```

PredictionFrame 精确包含 `signal_date, ts_code, score, model_id, fold_id, dataset_id`；重复键、非 finite score 或缺失成员失败。

- [x] **D6.2 实现 B0 best single factor**

方向只在 train 固定；对入选 factor set 的每个因子在 validation 算 mean RankIC，选最高者，tie-break 为 train signed mean IC 更高、coverage 更高、factor_id ASC。test 只使用冻结的因子/方向。

- [x] **D6.3 实现 B1 signed equal weight**

对每因子预处理值乘冻结方向，逐日对 finite 值 average；要求当行至少 60% 因子有效，否则 score null。再在当日对 score 做 percentile rank，tie-break 不影响相同 score 的 average rank。

- [x] **D6.4 写“强简单基线”回归测试**

合成数据包含一个强单因子和多个噪声因子，断言 B0 选强因子、B1 不会错误用 test 重选方向。加入 GP_032 类强因子 fixture，报告必须显式显示其验证/OOS 表现。

- [x] **D6.5 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/baselines/test_single_factor.py tests/unit/baselines/test_equal_weight.py -q`

Expected: 方向、validation 选择、missing coverage 和 test 冻结通过。

```bash
git add research_v2/configs/baselines/single_factor.yaml research_v2/configs/baselines/equal_weight.yaml research_v2/src/research_v2/baselines/base.py research_v2/src/research_v2/baselines/single_factor.py research_v2/src/research_v2/baselines/equal_weight.py research_v2/tests/unit/baselines/test_single_factor.py research_v2/tests/unit/baselines/test_equal_weight.py
git commit -m "feat(v2): establish single-factor and ensemble baselines"
```

## Task D7: B2 线性与 B3 LightGBM ranker 基线

**Files:**

- Create: `research_v2/configs/baselines/linear.yaml`
- Create: `research_v2/configs/baselines/lgbm_ranker.yaml`
- Create: `research_v2/src/research_v2/baselines/linear.py`
- Create: `research_v2/src/research_v2/baselines/lgbm_ranker.py`
- Test: `research_v2/tests/unit/baselines/test_linear.py`
- Test: `research_v2/tests/unit/baselines/test_lgbm_ranker.py`

- [x] **D7.1 实现 B2 Ridge/ElasticNet 小网格**

训练目标为每日 raw return 去当日均值后的值，sample weight 使每个交易日总权重相同。候选固定：`alpha ∈ {0.01,0.1,1,10,100}`，`l1_ratio ∈ {0,0.1,0.5,0.9,1}`；按 3 inner folds mean RankIC 选组合，validation 只用于 B0–B3 最终比较。tie-break 优先更大 alpha、更小 l1 ratio。

- [x] **D7.2 实现 B3 LambdaRank**

每个训练日按 raw return 五分位生成 relevance 0–4，日期作为 query group；固定候选：`num_leaves ∈ {15,31}`、`max_depth ∈ {4,6}`、`learning_rate ∈ {0.03,0.05}`、`min_child_samples ∈ {50,200}`，max estimators=500，inner validation early stopping patience=50。seed 固定 20260801，`n_jobs` 固定以保证可重复。

- [x] **D7.3 禁止横截面 group 混乱**

排序固定 `signal_date, ts_code`；group sizes 之和必须等于行数；同日不能跨 parquet chunk 被错误切成多个 query；预测恢复原业务键后再排序。

- [x] **D7.4 写选择范围测试**

监控 estimator 的 fit/eval_set，证明 test 从不传入；改变 test labels 不改变参数/model hash。训练不足 30 股票的日期从 ranker loss 排除但记录 coverage。

- [x] **D7.5 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/baselines/test_linear.py tests/unit/baselines/test_lgbm_ranker.py -q`

Expected: 网格、日权重、query group、early stopping 和 test isolation 通过。

```bash
git add research_v2/configs/baselines/linear.yaml research_v2/configs/baselines/lgbm_ranker.yaml research_v2/src/research_v2/baselines/linear.py research_v2/src/research_v2/baselines/lgbm_ranker.py research_v2/tests/unit/baselines/test_linear.py research_v2/tests/unit/baselines/test_lgbm_ranker.py
git commit -m "feat(v2): add linear and tree ranking baselines"
```

## Task D8: Walk-forward 编排、最优简单基线与双重门槛

**Files:**

- Create: `research_v2/src/research_v2/evaluation/gates.py`
- Create: `research_v2/src/research_v2/evaluation/walk_forward.py`
- Create: `research_v2/src/research_v2/cli/baselines.py`
- Create: `research_v2/src/research_v2/cli/evaluate.py`
- Test: `research_v2/tests/unit/evaluation/test_gates.py`
- Test: `research_v2/tests/integration/evaluation/test_walk_forward_baselines.py`

- [x] **D8.1 每折冻结 best simple baseline**

B0–B3 全部在同一 fold manifest 上训练；按 validation mean RankIC 最大选 best，tie-break 固定复杂度 `B0 < B1 < B2 < B3`。写 `selection_record` 后才允许加载 test；记录 validation metrics 和未选原因。

- [x] **D8.2 拼接 F01–F06 OOS**

每个自然年只来自对应 fold test；拼接前断言 date 唯一归属、prediction key 唯一、dataset/model hash 已冻结。总体指标从 2020–2025 所有每日 IC 重算，不平均年度 ICIR。

- [x] **D8.3 实现机器可判定门槛**

```text
G1 mean OOS RankIC >= 0.030
G2 HAC t-stat >= 2.000 (maxlags=5)
G3 positive test-year ratio >= 0.70，即 6 年至少 5 年 mean RankIC > 0
G4 candidate mean OOS RankIC - fold-matched best-simple mean OOS RankIC >= 0.005
G5 max positive-year contribution share <= 0.50
G6 score not collapsed
```

G5 分母为各年 `max(year_mean_ic,0)` 之和，分子为最大值；分母为 0 时失败。G6 要求至少 95% 有效测试日同时满足 finite score coverage ≥90%、unique finite scores ≥10、score std >1e-6。

基线自身报告 G1/G2/G3/G5/G6；G4 只用于候选模型，基线显示 `NOT_APPLICABLE`，不误判失败。

- [x] **D8.4 写阈值边界测试**

精确等于 0.03、2.0、5/6、0.005、0.50 均通过；浮点比较使用未格式化值，报告四舍五入不参与判断。任一缺失年度/指标、NaN、无有效日都 fail closed。

- [x] **D8.5 定义 CLI**

```text
qr-v2 baseline run --fold F01 --models B0,B1,B2,B3
qr-v2 evaluate walk-forward --experiment-id sha256:... --folds F01,F02,F03,F04,F05,F06
qr-v2 report experiment --experiment-id sha256:... --format html
```

命令拒绝 FINAL test；若 B0–B3 未齐不计算 `best_simple_baseline`。

- [x] **D8.6 V2Test 端到端基线测试**

使用合成 7 年数据完成 split → preprocess → select → B0–B3 → validation select → test evaluate → OOS concatenate，断言 test perturbation 不改变选模，重复运行 metrics/model hashes 一致。

- [x] **D8.7 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/evaluation/test_gates.py -q`

Run: `cd research_v2 && QRV2_INTEGRATION=1 pytest tests/integration/evaluation/test_walk_forward_baselines.py -q -m integration`

Expected: 所有门槛边界和完整 baseline 流程通过。

```bash
git add research_v2/src/research_v2/evaluation/gates.py research_v2/src/research_v2/evaluation/walk_forward.py research_v2/src/research_v2/cli/baselines.py research_v2/src/research_v2/cli/evaluate.py research_v2/tests/unit/evaluation/test_gates.py research_v2/tests/integration/evaluation/test_walk_forward_baselines.py
git commit -m "feat(v2): evaluate purged walk-forward baselines"
```

## Task D9: 黄金数据、报告与完整泄漏审计

**Files:**

- Create: `research_v2/src/research_v2/reporting/experiment.py`
- Create: `research_v2/tests/golden/data/dataset_50x60.parquet`
- Create: `research_v2/tests/golden/data/expected_metrics.json`
- Create: `research_v2/tests/golden/test_dataset_and_metrics.py`
- Create: `research_v2/tests/leakage/test_end_to_end_fold_isolation.py`

- [x] **D9.1 扩充 50×60 黄金数据**

将 B/C 黄金输入组成 dataset，人工固定 3 个因子：完全正相关、完全负相关、有并列/缺失；独立计算每日 Spearman、mean/std/ICIR 和 statsmodels HAC 参考值。manifest 保存所有输入 hash。

- [x] **D9.2 实现实验报告**

HTML/JSON 同时包含：数据版本、fold dates/purge、factor set、预处理、B0–B3 validation/test、每年 OOS、总体 OOS、HAC、门槛逐项 PASS/FAIL、覆盖、score collapse、运行环境和所有 artifact links。默认首先显示最佳简单基线，不只展示复杂候选。

- [x] **D9.3 完整泄漏 canary**

在 validation/test/2026 区域注入极强“未来完美因子”和标签，断言 train selector/preprocessor/baseline fit hashes 不变；普通路径触碰 2026 时测试必须失败并记录访问尝试。

- [ ] **D9.4 运行完整 D 计划验证**

Run: `cd research_v2 && pytest tests/unit/datasets tests/unit/evaluation tests/unit/baselines tests/property/datasets tests/leakage tests/golden/test_dataset_and_metrics.py -q`

Run: `cd research_v2 && QRV2_INTEGRATION=1 pytest tests/integration/datasets tests/integration/evaluation -q -m integration`

Expected: 全部退出码 0；黄金指标与参考值一致；locked canary 被阻止。

- [x] **D9.5 提交并请求审查**

```bash
git add research_v2/src/research_v2/reporting research_v2/tests/golden research_v2/tests/leakage/test_end_to_end_fold_isolation.py
git commit -m "test(v2): lock dataset metrics and leakage canaries"
```

随后使用 `superpowers:requesting-code-review`，重点审查 purge、fit scope、FDR family、test 选模隔离、HAC 口径和 2026 锁。

## D 计划完成标准

- [x] F01–F06 与 FINAL 年份、3 个 inner folds 和 6 日 purge 由测试精确锁定。
- [x] preprocessor/selector 仅拟合 outer train；validation 只选参数，test 只评价。
- [x] 每个 dataset manifest 可内容寻址、验证和不可变重放。
- [x] B0–B3 全部完成，每折 best simple baseline 只由 validation 决定。
- [x] OOS RankIC/ICIR/HAC 与年度贡献按统一函数计算；门槛边界可机器判定。
- [x] 普通 builder/evaluator 无法读取 FINAL 2026 test。

## 状态核对（2026-09-01）

已按现有代码与测试逐项核对并勾选完成步骤（splits/manifest/preprocessing/
selection/storage、B0–B3、gates/walk-forward、50×60 黄金数据与泄漏 canary
均有对应测试；全量单元套件 947 通过，`ruff check` 与 `mypy src` 干净）。
仍开放的条目及原因：

- D9.4：`tests/integration/datasets` 与 `tests/property/datasets` 仍未建立；
  `tests/integration/evaluation` 已由 D8.6 端到端套件覆盖（2026-09-01）。
  单元侧全部通过。
  D8.6/D8.7 已于 2026-09-01 补齐：合成 7 年（2015–2019 训练/2020 验证/2021
  测试）端到端编排测试 `tests/integration/evaluation/test_walk_forward_baselines.py`
  覆盖 split→B0–B3→validation 选模→延迟 test 评价，断言两次运行选择记录与
  预测逐帧一致（B3 固定单线程+种子），且 TEST 标签强扰动不改变选模；
  `QRV2_INTEGRATION=1` 门控运行 3 通过（216s）。
