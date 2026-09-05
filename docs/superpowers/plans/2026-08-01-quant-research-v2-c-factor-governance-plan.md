# Quant Research V2 C — Factor Governance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将现有 YesLab 因子以只读方式纳入版本化 registry，完成确定性去重、冲突隔离、点时可用日转换、质量快照、训练窗内相关性聚类和不可变 factor set，消除重复因子、未来填充与定义漂移。

**Architecture:** registry 先描述每个因子的来源、语义、频率、方向和可用日政策，再由按类型 dispatch 的 PIT materializer 生成统一 long-form observations。技术因子不向前填充；基本面因子在公告日后的下一个交易日可用，最多持有 180 个日历日。结构性质量检查与标签无关，可全历史运行；预测性质量、相似度和选因 scope 必须绑定 fold 的训练窗口。

**Tech Stack:** Python 3.11、DolphinDB SDK、Pandas/NumPy、SciPy、statsmodels、Pydantic、PyArrow、pytest/Hypothesis。

---

## 前置与非目标

- 前置：A 计划通过；B 计划至少提供稳定交易日历、PIT security master、universe 和成熟历史标签。
- 不修改、修复或去重 YesLab 源表；所有清洗结果写 V2。
- 不在本计划中根据全样本 IC 选因；预测性筛选在 D 计划每折训练范围内执行。
- 不把重复因子列简单平均，也不以 `first`/`last` 解决冲突。
- 不使用模型 hidden state 构造图或选因。

## 文件清单

**Create:**

```text
research_v2/configs/factors/source_catalog.yaml
research_v2/configs/factors/quality_rules.yaml
research_v2/src/research_v2/contracts/factor.py
research_v2/src/research_v2/factors/registry.py
research_v2/src/research_v2/factors/source_reader.py
research_v2/src/research_v2/factors/deduplication.py
research_v2/src/research_v2/factors/availability.py
research_v2/src/research_v2/factors/materializer.py
research_v2/src/research_v2/factors/quality.py
research_v2/src/research_v2/factors/similarity.py
research_v2/src/research_v2/factors/factor_sets.py
research_v2/src/research_v2/cli/factors.py
research_v2/tests/unit/factors/test_registry.py
research_v2/tests/unit/factors/test_source_reader.py
research_v2/tests/unit/factors/test_deduplication.py
research_v2/tests/unit/factors/test_availability.py
research_v2/tests/unit/factors/test_materializer.py
research_v2/tests/unit/factors/test_quality.py
research_v2/tests/unit/factors/test_similarity.py
research_v2/tests/unit/factors/test_factor_sets.py
research_v2/tests/property/factors/test_deduplication_invariants.py
research_v2/tests/property/factors/test_availability_invariants.py
research_v2/tests/leakage/test_factor_future_perturbation.py
research_v2/tests/golden/data/factors_50x60.parquet
research_v2/tests/golden/data/expected_factor_observations.parquet
research_v2/tests/golden/test_factor_materialization.py
research_v2/tests/integration/factors/test_factor_publish.py
research_v2/tests/integration/factors/test_factor_registry_import.py
```

## Task C1: 因子源目录与 registry 契约

**Files:**

- Create: `research_v2/configs/factors/source_catalog.yaml`
- Create: `research_v2/src/research_v2/contracts/factor.py`
- Create: `research_v2/src/research_v2/factors/registry.py`
- Test: `research_v2/tests/unit/factors/test_registry.py`
- Test: `research_v2/tests/integration/factors/test_factor_registry_import.py`

- [ ] **C1.1 根据 Phase 0 清单逐因子建立 catalog**

每个现有因子必须有唯一条目，不能用运行时 `GP_*` 通配符自动接纳新列。每条 YAML 记录必须包含 `factor_id`、`factor_version`、`display_name`、`source_database`、`source_table`、`source_value_column`、`source_date_column`、`source_code_column`、`factor_type`、`frequency`、`expected_direction`、`availability_policy`、`max_staleness_calendar_days`、`unit` 和 `owner`。

`source_database` 固定为 `dfs://YesLab`；`source_table` 必须是 Phase 0 从 `factor_metadata.store_factor` 冻结出的 `factor_data1` 至 `factor_data11` 中的准确表名；值/日期/证券字段按当前源契约固定为 `value`、`trade_date`、`ts_code`。C1 的生成命令必须把冻结的 metadata snapshot 转成显式逐因子 YAML，并在人审后保存 snapshot hash；不存在待运行时替换的表名。

- [ ] **C1.2 写失败测试：稳定 ID 与版本不可变**

```python
def test_definition_change_requires_new_factor_version(registry):
    registry.register(FACTOR_V1)
    with pytest.raises(ImmutableDefinitionError):
        registry.register(replace(FACTOR_V1, unit="percent"))
```

覆盖重复 ID/version、未知 factor type、无 owner、源 URI 非 YesLab、technical 配置 staleness>0、fundamental 缺 ann_date mapping、definition hash 变化。

- [ ] **C1.3 定义类型与 definition hash**

```python
class FactorType(StrEnum):
    TECHNICAL = "TECHNICAL"
    FUNDAMENTAL = "FUNDAMENTAL"
    ALTERNATIVE = "ALTERNATIVE"


class AvailabilityPolicy(StrEnum):
    SAME_DAY_CLOSE = "SAME_DAY_CLOSE"
    NEXT_TRADING_DAY_AFTER_ANNOUNCEMENT = "NEXT_TRADING_DAY_AFTER_ANNOUNCEMENT"
    EXPLICIT_AVAILABLE_DATE = "EXPLICIT_AVAILABLE_DATE"
```

definition hash 对规范化的语义、源映射、方向、公式/来源版本、频率、单位和 availability policy 做 SHA-256；描述文案和 owner 变更可通过 metadata revision，不改变数学定义。

- [ ] **C1.4 实现 registry 状态**

状态 `DRAFT → VALIDATED → ACTIVE → RETIRED`；只有 `ACTIVE` 因子可加入 factor set。定义字段在 `VALIDATED` 后不可修改；需要变更时创建新 `factor_version`。

- [ ] **C1.5 实现 catalog 对源 schema 的 dry-run**

逐条验证表/列存在、类型可转换、日期/代码键明确，输出 `mapped/blocked/retired` 数量和未映射源列；不读取全量值、不写数据库。

- [ ] **C1.6 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/factors/test_registry.py -q`

Run: `cd research_v2 && QRV2_INTEGRATION=1 pytest tests/integration/factors/test_factor_registry_import.py -q -m integration`

Expected: 全部因子有显式映射；定义漂移要求升版；V1 无写入。

```bash
git add research_v2/configs/factors/source_catalog.yaml research_v2/src/research_v2/contracts/factor.py research_v2/src/research_v2/factors/registry.py research_v2/tests/unit/factors/test_registry.py research_v2/tests/integration/factors/test_factor_registry_import.py
git commit -m "feat(v2): register immutable factor definitions"
```

## Task C2: 因子源 reader 与 long-form 规范化

**Files:**

- Create: `research_v2/src/research_v2/factors/source_reader.py`
- Test: `research_v2/tests/unit/factors/test_source_reader.py`

- [ ] **C2.1 写失败测试：宽表/长表均映射到统一契约**

统一输出：

```text
ts_code
factor_id
factor_version
observation_date
source_available_date
raw_value
source_row_id
source_table
source_watermark_id
```

测试宽表列顺序改变不影响输出；额外未注册列被报告但不加载；缺少注册列阻止对应因子批次。

- [ ] **C2.2 实现分列/分月读取**

按注册表只查询所需列和日期，不加载 135 列全历史到单机；默认月份 chunk + factor group，输出 canonical long form。chunk 边界不影响最终 hash。

- [ ] **C2.3 实现数值规范化**

接受 DolphinDB 数值类型与可解释的 decimal；将 ±inf 转为 invalid flag 而非有效 NaN；保留 raw value 精度；不在 source reader winsorize、scale 或 impute。

- [ ] **C2.4 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/factors/test_source_reader.py -q`

Expected: 宽/长映射、列漂移、chunk 和数值异常测试通过。

```bash
git add research_v2/src/research_v2/factors/source_reader.py research_v2/tests/unit/factors/test_source_reader.py
git commit -m "feat(v2): normalize factor source batches"
```

## Task C3: 确定性去重与冲突隔离

**Files:**

- Create: `research_v2/src/research_v2/factors/deduplication.py`
- Test: `research_v2/tests/unit/factors/test_deduplication.py`
- Test: `research_v2/tests/property/factors/test_deduplication_invariants.py`

- [ ] **C3.1 写三类决策的失败测试**

业务键为 `observation_date, ts_code, factor_id, factor_version`：

```text
IDENTICAL_COLLAPSE: 所有非审计字段完全一致，只保留一行
SINGLE_NONNULL: 恰有一个有限非空值，其余为空，保留该值并打标
CONFLICT_QUARANTINE: 存在两个不相等的有限非空值，整组隔离，clean 输出 0 行
```

浮点相等使用 canonical binary/decimal value，不用宽松 `isclose` 把真实冲突吞掉；源如果声明量化精度，则先按该精度规范化再比较。

- [ ] **C3.2 实现纯函数 resolution**

```python
@dataclass(frozen=True)
class DedupResult:
    clean: pd.DataFrame
    quarantine: pd.DataFrame
    statistics: DedupStatistics
```

quarantine 保存所有原始候选、source row ids、distinct values 和 reason；不选择 first/last。

- [x] **C3.3 增加顺序/分块性质测试**

Hypothesis 验证输入 permutation、chunk 切分、DataFrame index 不影响 clean/quarantine；重复加入完全相同行不改变业务输出；冲突加入后 clean 必然移除该 key。

- [ ] **C3.4 用已知问题构造回归 fixture**

从 Phase 0 审计中抽取匿名化的 GP_017、GP_077 重复键样例，证明重复行被计数且冲突不会进入 clean。fixture 只保存必要字段和 source hash，不复制大规模源数据。

- [ ] **C3.5 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/factors/test_deduplication.py tests/property/factors/test_deduplication_invariants.py -q`

Expected: 三类 resolution 和全部不变量通过。

```bash
git add research_v2/src/research_v2/factors/deduplication.py research_v2/tests/unit/factors/test_deduplication.py research_v2/tests/property/factors/test_deduplication_invariants.py
git commit -m "feat(v2): quarantine conflicting factor duplicates"
```

## Task C4: 因子点时可用日与 staleness

**Files:**

- Create: `research_v2/src/research_v2/factors/availability.py`
- Test: `research_v2/tests/unit/factors/test_availability.py`
- Test: `research_v2/tests/property/factors/test_availability_invariants.py`
- Test: `research_v2/tests/leakage/test_factor_future_perturbation.py`

- [ ] **C4.1 写技术因子可用日测试**

`SAME_DAY_CLOSE`：`observation_date = available_date = signal_date = trade_date`。缺值保持缺值，不从前一日 ffill；同日收盘后才可用于当日收盘信号语义。

- [ ] **C4.2 写基本面事件测试**

```python
def test_fundamental_is_available_next_trading_day(calendar):
    event = FundamentalEvent(report_period=date(2024, 12, 31), ann_date=date(2025, 3, 28))
    assert assign_available_date(event, calendar) == date(2025, 3, 31)
```

周末公告、交易日公告、修订公告、同报告期多版本、退市、180 日边界都要覆盖。无 `ann_date` 的基本面记录隔离，不能用报告期代替公告日。

- [ ] **C4.3 实现事件区间展开而非全笛卡尔积**

基本面值仅与实际 `universe_daily` 成员日期做 interval/as-of join；有效区间从 available date 到 `min(next_version_available_date - 1 trading day, available_date + 180 calendar days)`。输出 `stale_days`；第 181 个自然日值为空。

- [ ] **C4.4 实现修订版本语义**

同一 report period 的后续修订只从其新 available date 开始生效；过去 signal dates 继续看到当时已公布版本。记录 `source_revision_id` 和 `report_period`。

- [ ] **C4.5 强制 PIT 不变量**

对每行断言：`observation_date <= available_date <= signal_date`；technical 三者相等；fundamental `available_date > ann_date`（按交易日规则）；source watermark 的捕获时间不晚于 run publish。

- [x] **C4.6 写未来扰动测试**

添加/修改未来公告、未来修订或未来技术值，不得改变此前任何 signal date 的 clean factor record/hash。删除未来 universe 日期不改变过去展开。

- [ ] **C4.7 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/factors/test_availability.py tests/property/factors/test_availability_invariants.py tests/leakage/test_factor_future_perturbation.py -q`

Expected: 日历、staleness、revision 和未来扰动测试通过。

```bash
git add research_v2/src/research_v2/factors/availability.py research_v2/tests/unit/factors/test_availability.py research_v2/tests/property/factors/test_availability_invariants.py research_v2/tests/leakage/test_factor_future_perturbation.py
git commit -m "feat(v2): enforce point-in-time factor availability"
```

## Task C5: 因子物化与发布

**Files:**

- Create: `research_v2/src/research_v2/factors/materializer.py`
- Create: `research_v2/src/research_v2/cli/factors.py`
- Modify: `research_v2/src/research_v2/cli/app.py`
- Test: `research_v2/tests/unit/factors/test_materializer.py`
- Test: `research_v2/tests/integration/factors/test_factor_publish.py`

- [ ] **C5.1 写 pipeline 组合测试**

顺序固定：registry validate → source read → normalize → deduplicate → availability → universe filter → structural DQ → stage → validate → publish。任一步失败均不产生正式可见行，pipeline run 标记失败。

- [ ] **C5.2 实现统一 observation 输出**

`factor_observation_clean` 最小字段：

```text
signal_date, ts_code, factor_id, factor_version, value,
observation_date, available_date, report_period, stale_days,
dedup_resolution, source_revision_id,
run_id, data_version, created_at, source_watermark_id, record_hash
```

技术因子的 `report_period/source_revision_id` 可空；值为空的行默认不写事实表，但缺失在质量分母中保留。

- [ ] **C5.3 定义 CLI 与 dry-run**

```text
qr-v2 factor registry validate --catalog configs/factors/source_catalog.yaml
qr-v2 factor registry import --catalog ...
qr-v2 materialize factors --start 2014-01-01 --end 2025-12-31 --factor-set all-active
qr-v2 factor quarantine report --run-id RUN_UUID
```

dry-run 输出因子数、源表/列、日期 chunk、估算行数和目标分区；禁止读取值或写库。

- [ ] **C5.4 实现增量影响范围**

技术日频源变化只重算其变化 signal dates；基本面公告/修订变化重算该证券从 available date 到下次版本或 180 日上限与 universe 交集；registry 新版本不覆盖旧版本。

- [ ] **C5.5 V2Test 集成故障测试**

覆盖源 schema 漂移、源水位中途变化、重复冲突、available>signal、publish 前进程异常和相同版本重跑。

- [ ] **C5.6 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/factors/test_materializer.py -q`

Run: `cd research_v2 && QRV2_INTEGRATION=1 pytest tests/integration/factors/test_factor_publish.py -q -m integration`

Expected: 正常发布幂等；所有 blocker 均保持旧版本可见。

```bash
git add research_v2/src/research_v2/factors/materializer.py research_v2/src/research_v2/cli/factors.py research_v2/src/research_v2/cli/app.py research_v2/tests/unit/factors/test_materializer.py research_v2/tests/integration/factors/test_factor_publish.py
git commit -m "feat(v2): materialize governed factor observations"
```

## Task C6: 标签无关的结构质量快照

**Files:**

- Create: `research_v2/configs/factors/quality_rules.yaml`
- Create: `research_v2/src/research_v2/factors/quality.py`
- Test: `research_v2/tests/unit/factors/test_quality.py`

- [x] **C6.1 固化结构质量指标**

按 `as_of_date, factor_id, factor_version, scope_id` 计算：universe member denominator、coverage、finite ratio、cross-sectional unique ratio、zero ratio、median/MAD、p01/p50/p99、每日 dispersion、stale share、duplicate/conflict rate、最长缺失连续天数、日期覆盖和证券覆盖。

- [ ] **C6.2 区分 blocker 与 eligibility flag**

BLOCKER：PIT 违规、冲突键进入 clean、非有限值进入 value、未知 registry 版本、业务键重复。`ELIGIBILITY_FAIL`：训练 scope coverage <70%、median daily unique ratio <1%、超过 20% 日期横截面 dispersion=0。后者使因子不能进入该 fold 选因，但不删除历史记录。

- [x] **C6.3 写 denominator 与缺失测试**

coverage 分母必须是同日 universe 成员，不是“因子表中出现过的股票”；空日期返回 0 coverage 和明确状态，不返回 NaN 后被误当 PASS。基本面 stale >180 的值视为缺失。

- [x] **C6.4 实现无标签质量报告**

报告不读取 `label_daily`，因此可用于 2026 SHADOW 运维；predictive IC 字段在本模块不存在，防止全期偷看标签。

- [x] **C6.5 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/factors/test_quality.py -q`

Expected: 分母、阈值、空集、stale 和无标签依赖测试通过。

```bash
git add research_v2/configs/factors/quality_rules.yaml research_v2/src/research_v2/factors/quality.py research_v2/tests/unit/factors/test_quality.py
git commit -m "feat(v2): score structural factor quality"
```

## Task C7: 训练 scope 内相关性与确定性聚类

**Files:**

- Create: `research_v2/src/research_v2/factors/similarity.py`
- Test: `research_v2/tests/unit/factors/test_similarity.py`

- [ ] **C7.1 写失败测试：相似度不能越过 scope**

输入明确 `start_date/end_date/universe_id/data_version`，query builder 必须带闭区间；测试在 end_date 后添加高度相关数据不改变历史 scope 相似度。

- [ ] **C7.2 实现每日横截面 Spearman 汇总**

每个因子对仅在同日共同非空数 ≥30 时计算 Spearman；scope 相似度为有效每日 `abs(rho)` 的中位数，并记录有效天数/覆盖。有效日 <60 的 pair 标记 insufficient，不连边。

- [ ] **C7.3 实现阈值图 connected components**

`median_daily_abs_spearman >= 0.90` 建无向边，因子簇为 connected components。cluster id 对排序后的成员 ID 做 SHA-256，保证输入顺序不影响结果。

- [ ] **C7.4 实现训练内代表因子排序接口**

本模块只接受 D 计划传入的训练内 quality/predictive statistics；代表排序固定：eligible → FDR pass → direction stability → median abs IC → coverage → lower conflict rate → factor_id ASC。不得读取 validation/test 指标。

- [ ] **C7.5 回归测试已知高相关因子**

用匿名化小 fixture 覆盖完全重复、符号相反但绝对相关、缺失错位和链式 A–B/B–C 相关；验证 abs 相关使符号相反落同簇，connected component 处理传递边。

- [ ] **C7.6 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/factors/test_similarity.py -q`

Expected: scope 隔离、每日汇总、阈值边界、稳定 cluster id 和代表排序通过。

```bash
git add research_v2/src/research_v2/factors/similarity.py research_v2/tests/unit/factors/test_similarity.py
git commit -m "feat(v2): cluster correlated factors within train scopes"
```

## Task C8: 不可变 factor set 与黄金样本

**Files:**

- Create: `research_v2/src/research_v2/factors/factor_sets.py`
- Test: `research_v2/tests/unit/factors/test_factor_sets.py`
- Create: `research_v2/tests/golden/data/factors_50x60.parquet`
- Create: `research_v2/tests/golden/data/expected_factor_observations.parquet`
- Create: `research_v2/tests/golden/test_factor_materialization.py`

- [x] **C8.1 写 factor set 状态测试**

状态 `DRAFT → VALIDATED → FROZEN → RETIRED`。`FROZEN` 后成员/顺序/版本不可变；改变任何成员创建新 factor_set_id。一个 set 不得同时包含同一 factor 的两个版本。

- [x] **C8.2 实现 content-addressed factor set**

factor_set_id 对排序后的 `(factor_id, factor_version, expected_direction)`、selection scope、规则版本和 registry definition hashes 做 SHA-256。member 表保存 deterministic feature order，训练/推理严格复用。

- [x] **C8.3 扩充 B 计划黄金样本**

为同一 50×60 universe 增加至少：日频技术缺失、基本面周五公告、180 日 stale、修订公告、完全重复、single-nonnull、冲突、完全相同/符号相反因子。预期值由独立审计逻辑生成并逐案写 README。

- [x] **C8.4 运行完整 C 计划验证**

Run: `cd research_v2 && pytest tests/unit/factors tests/property/factors tests/leakage/test_factor_future_perturbation.py tests/golden/test_factor_materialization.py -q`

Expected: 全部退出码 0；黄金输出 hash 与批准 fixture 一致。

- [ ] **C8.5 提交并请求审查**

```bash
git add research_v2/src/research_v2/factors/factor_sets.py research_v2/tests/unit/factors/test_factor_sets.py research_v2/tests/golden
git commit -m "feat(v2): freeze content-addressed factor sets"
```

随后使用 `superpowers:requesting-code-review`，重点审查重复冲突处理、公告可用日、staleness、scope 边界和 factor set 不可变性。

## C 计划完成标准

- [ ] 所有启用因子都有显式、版本化、不可变 registry 定义；源新增列不会被自动纳入。
- [ ] 已知 GP_017/GP_077 类重复问题被统计，冲突键全部隔离，clean 表唯一。
- [ ] 技术因子不 ffill；基本面严格在公告后下一交易日可用且最多持有 180 日。
- [ ] 修改未来公告/值不改变任何过去 factor observation。
- [ ] 结构质量不读取标签；相似度与代表选择严格绑定训练 scope。
- [ ] factor set content-addressed、顺序固定、冻结后不可变，可被 D/E/F 精确复用。

## 勾选审计（2026-09-06 06:45）

本次只勾选**本轮真正交付并验证**的两项（C3.3、C4.6），已用 diff 核验**除复选框外零字符变化**
（2 增 2 删）。其余项**未勾选**，因为逐项验收尚未完成；下面记录已核到的现状，供后续接续。

**实现层：C1–C8 的点名模块全部存在，且相关测试通过。**
`pytest tests/unit/factors tests/unit/quality tests/unit/pit tests/unit/sources
tests/unit/registry tests/unit/materializers -q` → **232 passed**。逐项核到的证据：
C1 `configs/factors/source_catalog.yaml` + `contracts/factor.py`（`definition_hash` 在 :78）
+ `factors/registry.py`（`ImmutableDefinitionError`、`FactorRegistry`）；
C2 `factors/source_reader.py`；C3 `factors/deduplication.py`（`DedupResult` 含
clean/quarantine/statistics，与 C3.2 的契约逐字段相符）；C4 `factors/availability.py`；
C5 `factors/materializer.py` + `configs/factors/quality_rules.yaml`；C6 `factors/quality.py`；
C7 `factors/similarity.py`（docstring 明写「only within the explicit training scope」，
先按 `[start_date, end_date]` 过滤 → C7.1）与 **`datasets/selection.py:177
select_cluster_representatives`**（对 `status==PASS 且 median_daily_abs_spearman ≥ 阈值`
的因子对做**带路径减半的并查集** → 即 C7.3 的阈值图连通分量、C7.4 的代表选择；
注意它在 `datasets/` 而非 `factors/`，按模块名找不到）；
C8 `factors/factor_sets.py`（`FactorSetStatus`、content-addressed 摘要、状态迁移）。

**验证层：计划点名的 7 个产物中，3 个此前确实不存在，本轮已补齐；4 个仍缺。**

| 点名产物 | 对应项 | 本轮前 | 现状 |
|---|---|---|---|
| `tests/property/factors/test_deduplication_invariants.py` | C3.3 | **不存在**（连 `tests/property/factors/` 目录都没有） | **已建，7 条属性全绿** |
| `tests/leakage/test_factor_future_perturbation.py` | C4.6 | **不存在**（`tests/leakage/` 有 4 个文件，无因子可用日） | **已建，6 条全绿，并经两次变异验证非空** |
| `tests/property/factors/test_availability_invariants.py` | C4.x | **不存在** | **已建，6 条全绿** |
| `src/research_v2/cli/factors.py` | C5.3 | 不存在 | **仍缺**；`cli/materialize.py` 有 `dry_run` 与 `emit_plan`，但是否覆盖因子物化尚未核定 |
| `tests/golden/test_factor_materialization.py` 及 `data/expected_factor_observations.parquet`、`data/factors_50x60.parquet` | C8.3 | 不存在（golden 有 dataset/market/labels/models/compatibility，无因子物化） | **已建（见下节），C8.3/C8.4 已勾** |
| `tests/integration/factors/test_factor_publish.py` | C5.5 | 不存在（`tests/integration/factors/` 整个目录不存在） | **仍缺**，需 `QRV2_INTEGRATION=1` 与活库 |
| `tests/integration/factors/test_factor_registry_import.py` | C1.6 的集成半 | 同上 | **仍缺**，同上 |

**本轮顺带查出一个实现层真实缺陷并已修**（commit `b57373d`）：
`expand_fundamental_events` 在**没有任何事件版本符合 staleness 条件**时，
由空 `rows` 构造出的 `pd.DataFrame([])` **没有列**，尾部 `sort_values(["signal_date","ts_code"])`
随即抛 `KeyError: 'signal_date'`。最小复现：事件公告日 2025-11-03、universe 只有 2025-03-03
一个信号日。**这在生产中可达**——早于首次公告的早期折、新上市证券、或 staleness 上限紧到
全部过期时都会发生；而调用方是**跨因子跨折拼接**展开结果的，故一个空结果会中断整次物化。
报错指向一个列名，与真实原因（无符合条件记录）无关，诊断成本很高。
现返回带同样 8 列、同样 dtype 的空帧，两条到达空结果的路径（尚未可用 / 已可用但过期 182>180 天）
均有单测。**该缺陷是新的属性测试发现的**——随机日期组合会到达三个手写单测样例从未覆盖的情形。

**变异验证（证明新测试非装饰）**：把 `available_date <= signal_date` 换成全局最大值 → 6 条泄漏
测试中 **4 条失败**（另 2 条针对 universe 截断与 technical 路径，本就不应受此变异影响）；
把 technical 的 `available_date/signal_date` 都赋成最大 observation_date → technical 那条失败，
且触发的是新加的**空作用域守卫**（`_digest` 原本对空作用域返回哨兵哈希，两侧皆空即相等、
会静默通过；现改为抛错）。两次变异后 `availability.py` 均以 `git diff --quiet` 确认还原干净。

**未勾选的原因，逐项说明**：C4.5 只完成了一半——本计划要求的
「`observation_date <= available_date <= signal_date`、technical 三者相等、fundamental
`available_date > ann_date`」已由属性测试与泄漏测试覆盖，但**「source watermark 的捕获时间
不晚于 run publish」属物化/发布路径，本轮未覆盖**，故 C4.5 不勾。C3.5 / C4.7 是「运行测试并提交」，
其点名命令涉及的文件现已齐备（C4.7 的三个文件全部存在并通过），但 C4.5 未完整，故一并暂不勾。
C1.1–C1.5、C2.x、C3.1/C3.2/C3.4、C4.1–C4.4、C5.x、C6.x、C7.x、C8.x 的**逐条验收标准**
（而非仅模块存在）尚未逐项核对，故不勾。

## 勾选审计续（2026-09-06 07:10）：C8.3 与 C8.4 已交付并验证

**C8.3 已交付**：`scripts/compute_factor_golden_expected.py`（生成器 + **独立参考实现**）、
`tests/golden/test_factor_materialization.py`（6 测试）、`tests/golden/README_factor_cases.md`（逐案说明），
以及 6 个夹具：`factors_50x60.parquet`（18,000 行技术源抽取）、`fundamental_events_50x60.parquet`、
`universe_50x60.parquet`、`expected_factor_observations.parquet`（17,900 行）、
`expected_factor_quarantine.parquet`（25 行）、`expected_fundamental_expansion.parquet`（1,400 行）。
**计划只点名了两个数据文件，但一个文件装不下两种源 schema**（技术因子是
`ts_code/trade_date/factor_name/value` 的 YesLab 抽取形态，基本面事件是
`report_period/ann_date/raw_value/source_revision_id`），故拆为三个输入夹具；计划用词为「至少」，属扩充而非偏离。

**「独立审计逻辑」是硬要求，已严格满足**：生成器内的参考实现**不 import `research_v2.factors` 的任何东西**——
去重是 dict-of-groups 手写循环、可用日是按 C4 规格转述的逐行扫描。用实现自身跑出结果再快照会使黄金测试
**循环论证**（它只会钉住代码当下的行为，缺陷一并钉住），这与 B 计划 `compute_pit_golden_expected.py` 的
既有做法一致。**之所以能独立**：三种去重类别中存活值都是无歧义的（全同组只有一个值、single-nonnull 组只有一个
有限值、冲突组不发布任何值），故参考实现**无需复刻** `source_row_id` 的哈希 tie-break。

**八类案例逐一在夹具中核实成立**（由 `test_the_eight_required_cases_are_each_present_in_the_expectation` 断言，
该测试的作用是守卫夹具本身——某个案例若悄悄消失，上面几条黄金比对会一起变弱而无人察觉）：

| # | 案例 | 实测证据 |
|---|---|---|
| 1 | 日频技术缺失 | `yeslab_901` clean **2975 = 3000 − 25**，缺失格**未被 ffill** |
| 2 | 基本面周五公告 | `ann_date 2025-01-03`(周五) → `available_date 2025-01-06`(次周一) |
| 3 | 180 日 stale | 9 月块 **0 行**（`assert not (signal_date >= 2025-09-01).any()`） |
| 4 | 修订公告 | r1 覆盖 01-06..01-17（**500 行**）、r2 覆盖 01-20..02-12（**900 行**），且 `max(r1 日) < min(r2 日)` |
| 5 | 完全重复 | `IDENTICAL_COLLAPSE` **25** |
| 6 | single-nonnull | `SINGLE_NONNULL` **25** |
| 7 | 冲突 | `CONFLICT_QUARANTINE` **25**，且冲突键与 `yeslab_904` 已发布键**交集为空** |
| 8 | 完全相同/符号相反 | `yeslab_906`=副本、`yeslab_905`=镜像，对 `yeslab_901` 的 `median_daily_abs_spearman` 均 ≈1.0 且 ≥0.90 |

**60 个信号日取「1 月 30 个 + 9 月 30 个」两块而非连续 60 天**，是刻意的：连续 60 天（约 84 个自然日）
**够不到 180 日 staleness 上限**，案例 3 就只能靠调低阈值来迁就夹具——那会把被测的门槛本身改掉。

**案例 4 曾一度静默失效，已修正**：修订公告原定 2025-02-14（可用 02-17、失效 08-16），
而信号日只在 1 月与 9 月，**该窗口内一个信号日都没有**，于是 r2 永不被发布、案例 4 形同不存在。
生成器首次运行即输出 `fundamental revisions seen: ['r1']` 暴露了这一点；把修订公告移到
2025-01-17（同为周五，可用 01-20）后变为 `['r1','r2']`。**这也是「逐案断言夹具」那条守卫测试的价值所在。**

**夹具摘要已钉住**：`data/approved_factor_golden_digests.json` 记录 6 个夹具的 SHA-256，
`test_the_fixtures_are_the_approved_ones` 逐一比对。否则「用有缺陷的实现重跑生成器」会同时移动期望值，
测试两头都过。已实测**重新生成字节稳定**（`expected_factor_observations.parquet` 前后同为 `9e2529b7…`）。

**只比对业务字段**：`created_at` 是墙钟时间戳、`record_hash` 由它派生，二者均不可复现，故排除——
与 B 计划市场黄金测试排除自身非确定字段的做法一致。比对列为
clean `signal_date/ts_code/factor_id/factor_version/value/dedup_resolution`、
quarantine `observation_date/ts_code/factor_id/factor_version/reason/distinct_values_json`、
fundamental 的全部 8 列。

**C8.4 已按其点名命令验证**：
`pytest tests/unit/factors tests/property/factors tests/leakage/test_factor_future_perturbation.py
tests/golden/test_factor_materialization.py -q` → **167 passed，exit 0**，四个路径现全部存在。

**C8.1 / C8.2 已于 2026-09-06 07:05 逐条核实并勾选。**上一节曾记「仍不勾，因为逐条验收尚未核实」，
现补齐缺失的断言后核实完成：`tests/unit/factors/test_factor_sets.py` 由 3 条增至 **17 条**
（参数化展开），新增覆盖：

- **不得跳过 VALIDATED**：`DRAFT -> FROZEN` 直接迁移被 `FactorSetTransitionError` 拒绝
  （实现的允许表为 `DRAFT→{VALIDATED,RETIRED}`、`VALIDATED→{FROZEN,RETIRED}`、
  `FROZEN→{RETIRED}`、`RETIRED→{}`）。
- **RETIRED 从每个前置状态均可达且为终态**：对 DRAFT/VALIDATED/FROZEN 三种起点参数化，
  迁移到 RETIRED 后再向**任何**状态迁移均被拒。此前三条测试完全没有触及 RETIRED。
- **FROZEN 后成员/顺序/版本逐字段不变**：`frozen.members == draft.members`、
  `factor_set_id` 与 `content_hash` 均不变、`feature_order` 仍为 `[1, 2]`。
- **改变任何成员都产生新 `factor_set_id`**：对 `factor_id`、`factor_version`、
  `frozen_direction`、`definition_hash` 各改一项，以及删除一个成员、新增一个成员，
  共 6 种变更参数化，全部得到不同 id。
- **C8.2 点名的哈希输入逐项验证**：改 `selection_scope_id` 或 `rules_version` 都得到新 id；
  并且**仅**改某成员的 `definition_hash`（成员、顺序、scope、rules 全同）也得到新 id——
  否则因子可被就地重定义，而所有对旧 set id 签名的证据会静默指向新数据。
  读源码确认哈希载荷为 `{members:[{definition_hash, factor_id, factor_version,
  frozen_direction}], rules_version, selection_scope_id}` 经 `sort_keys` 后做 SHA-256，
  成员先按 `(factor_id, factor_version)` 排序、`feature_order` 依该序从 1 编号——
  **C8.2 要求的四类输入全部在内**。

**与裁决 4 的关联**：「改变任何成员即新 id」正是选项 (iii)（17 → 11）所需的性质——
重新裁决出的因子集是**新身份**而非对旧身份的修改，故对旧 `factor_set_id` 签名的证据
仍可归属。这与本文书前述「应新发 `factor_set_frozen_v2.json` 而非就地改写 V1」的结论一致，
且该性质现已有测试保障。

**C8.5 仍不勾**：提交已完成，但「请求审查」是人工步骤，尚未发生。

## 勾选审计续二（2026-09-06 07:25）：三条逐款核验的结论，其中两条不支持勾选

**(1) C3.1 不勾——一句已满足但此前无测试，一句未实现。**
该条要求「浮点相等使用 canonical binary/decimal value，**不用宽松 `isclose` 把真实冲突吞掉**；
源如果声明量化精度，则先按该精度规范化再比较」。
**前半句已满足**：`deduplication.py` 用 `nunique(dropna=True)` 判 distinct，是精确相等。
但**此前没有任何测试钉住它**——`test_deduplication.py` 只有 3 条测试，无一条涉及浮点精度，
故日后有人为「合并近似重复」而加入容差，全部现有测试仍会绿。现补 4 条：
相差一个 ULP 的两值必须判为 CONFLICT 而非 IDENTICAL_COLLAPSE；
相差 1e-6 的两值（**在 `np.isclose` 默认 `rtol=1e-5` 之内**）必须判为 CONFLICT；
发布值必须与观测值**逐位相同**（用 `float.hex()` 比较，因为 `0.1+0.2` 这类值用十进制断言会被放过）；
隔离区必须保留全部原始候选行、值不失真（C3.2 的契约；聚合出的 `distinct_values_json` 属物化器层，
其全精度保留已由因子黄金样本的冲突案例钉住）。
**后半句未实现**：全仓 `grep -rn "quantiz|precision|round("` 在
`deduplication.py`、`source_reader.py`、`contracts/factor.py` 中**零命中**，
且 `FactorDefinition` **没有任何声明量化精度的字段**——即该能力不存在，
子句在「无源声明精度」的意义下空满足。**这是设计取舍还是缺口，应由 C 计划的 owner 判定，不代为决定**，
故 C3.1 保持未勾并记录在此。

**变异验证（并证明新测试与黄金样本非冗余）**：把判重用的 `_finite_numeric` 改为 `.round(5)`
（模拟「加了个容差」），结果**恰好那 3 条容差敏感测试失败**；
而**因子黄金样本在该变异下通过**——因为它的冲突案例两值相差 0.5，五位小数容差对它不可见。
即新单测覆盖了黄金样本**确实覆盖不到**的尺度。逐位相同那条在变异下通过是正确的：
变异只作用于判重列、不改发布值，它钉的是另一件事。变异后 `deduplication.py` 以
`git diff --quiet` 确认还原干净。

**(2) C4.5 仍不勾——未覆盖的那半需要尚不存在的管线。**
该条末句要求「source watermark 的捕获时间不晚于 run publish」。实测：
`SourceBatch.extracted_at` 由四个源适配器产生（`sources/yeslab.py:104/174/251`、
`sources/tushare.py:202`，均为 `pd.Timestamp.now(tz="UTC")`），并被 `pit/security_profile.py:97`
向下传递；但**全仓没有任何一处把它与 publish 时刻比较**，也没有测试断言该顺序
（`grep -rn "extracted_at" src/ tests/` 的全部命中都只是产生或透传）。
而 publish 路径（`db/publish.py` 写 `[now()] as published_at`）**并不接收** `extracted_at`，
故强制该不变量需要新增贯穿发布路径的管线——那是对核心发布路径的侵入式改动，
且该不变量在正常操作下平凡成立（先抽取后发布），只有时钟偏移或「用旧 staged batch 在新抽取之后重发布」
才会违反。**判断：本轮不做**，如实记录以免 C4.5 被误勾。前三款（`observation_date <= available_date
<= signal_date`、technical 三者相等、fundamental `available_date > ann_date`）已由
C4.6 泄漏测试与 C4 属性测试覆盖。

**(3) C5.3 确认为真缺，但本轮不做，理由是口径未定。**
`cli/materialize.py` 的 `_PLANS` 只有 `security-master`/`market`/`universe`/`labels`，**没有 `factors`**；
`cli/app.py` 也未注册 `factor` 子命令（现有 11 个子应用中无它）。故计划点名的四条命令均不存在。
不做的理由不是工作量，而是**规格内部有一处需先定口径**：dry-run 要求「输出估算行数」同时「禁止读取值」，
而行数估算需要 universe 的逐日计数；这属于「元数据查询是否算读取值」的解释问题。
在没有该口径的情况下实现，等于把一个未定语义写进 CLI 契约。**建议先定口径再实现**，
或按本项目既有做法（`omp_num_threads` 未设置即记 `null` 而非编造）输出 `null` 并注明所需输入。

## 勾选审计续三（2026-09-06 07:40）：C6 逐款核验，勾 3 项、2 项带明确缺口不勾

**已勾：C6.3 / C6.4 / C6.5。**

- **C6.4「实现无标签质量报告」——已满足，且此前无测试钉住。** `quality.py` 全模块
  **无 `label`/`raw_return` 字样**，其 docstring 明写「using only universe and factor
  observations」，实现上在 merge 前**显式投影**到 `signal_date/ts_code/value/stale_days`，
  故任何额外列（含标签）都被丢弃；模块内也**不存在 predictive IC 字段**。
  新增两条测试把它钉住：一条断言快照对标签列的**存在与取值完全不敏感**
  （变体含完全正相关、完全反相关、`label_valid` 全假、以及**非有限标签** nan/inf），
  另一条断言**指标词汇表本身**不含 label/return/rank_ic/icir/outcome。
  **两次变异验证**：把标签代入有限性检查、把标签代入参与指标的 `value` 投影，
  两次都被捕获；`quality.py` 均以 `git diff --quiet` 确认还原干净。
  **第一次变异最初未被捕获**——因为原始变体的标签值全是有限的，只能触发指标路径、
  触发不了有限性检查路径；补入非有限标签变体后才捕获。这条教训已写进测试注释。
- **C6.3「写 denominator 与缺失测试」——三款齐备。** 分母为同日 universe 成员（既有测试）；
  基本面 stale >180 视为缺失（既有测试，`is_usable` 要求 `stale_days <= maximum_staleness_calendar_days`）；
  **空集**此前无测试，现补：无观测与空 universe 两条路径都必须给出 `coverage == 0.0`、
  `finite_value_count == 0`、状态非 PASS、含 `LOW_COVERAGE`，且**没有任何指标为 NaN**。
  NaN 才是危险结果——它既不低于也不高于阈值，基于比较的门会把它当作通过，
  从而准入一个从未被观测到的因子。
- **C6.5「运行测试并提交」**点名的五类（分母、阈值、空集、stale、无标签依赖）现已全部有测试且通过；
  `pytest tests/unit/factors/test_quality.py -q` → **6 passed**。

**未勾：C6.1 / C6.2，缺口已量化。**

- **C6.1「固化结构质量指标」——约 13 项点名指标中只发出 5 项。**
  实际 `metrics` 为 `universe_member_count`、`finite_value_count`、`coverage`、
  `median_daily_unique_ratio`、`zero_dispersion_date_share`。
  **缺失**：zero ratio、median/MAD（内部为判 dispersion 而算，但**未发出**）、p01/p50/p99、
  每日 dispersion（只发出零 dispersion 日**占比**）、stale share、duplicate/conflict **rate**
  （现为布尔 blocker，无比率）、最长缺失连续天数、日期覆盖、证券覆盖。
- **C6.2「区分 blocker 与 eligibility flag」——eligibility 半完全相符，blocker 半缺 3 条。**
  `ELIGIBILITY_FAIL` 三条与规格**逐一相符**：`LOW_COVERAGE`(<0.70)、`LOW_UNIQUE_RATIO`(<1.00)、
  `EXCESS_ZERO_DISPERSION`(>0.20)，默认值即规格值。
  `BLOCKER` 已实现三条：`DUPLICATE_UNIVERSE_KEY`、`DUPLICATE_BUSINESS_KEY`、`NONFINITE_CLEAN_VALUE`；
  **缺失三条**：PIT 违规、冲突键进入 clean、未知 registry 版本。
  （其中「冲突键进入 clean」在上游 `deduplicate_factor_observations` 已把冲突整组隔离，
  故本层或属防御性重复——但那正是需要断言的不变量，不应靠「上游大概做了」。）

**这两项是 C 计划剩余离线工作中最实质的缺口**，且都不需要活库：C6.1 是补发指标，
C6.2 是补三条 blocker 判定。二者都会改变 `FactorQualitySnapshot.metrics` 的内容，
而该快照被 C6.2 筛查与准入一致性门消费，故落地时应同步核对
`screen_factor_daily_coverage.py` 与 `verify_admission_consistency.py` 的字段依赖。

## 勾选审计续四（2026-09-06 07:55）：C6.1 已补齐并勾选，C6.2 补完判定但仍不勾

**先更正上一节的一个假设。**上节写「该快照被 C6.2 筛查与准入一致性门消费，故落地时应同步核对
字段依赖」——**这是未经核实的假设，已被 grep 推翻**：`assess_factor_quality` 与
`FactorQualitySnapshot` 在 `src/`、`scripts/`、`tests/`（除自身单测外）**没有任何消费者**。
`screen_factor_daily_coverage.py` 自带一套 C6.2 阈值与逐日计数逻辑，**不调用本模块**。
故扩展 `quality.py` 是**零下游风险**的；但这也暴露一个架构事实：
**「C6.2 结构质量」现有两套并行实现**，一套（本模块）未被接线。

**顺序约束（重要）**：这两套实现的合并**必须排在裁决 4 落地之后**。当前生效的那套
（`screen_factor_daily_coverage.py`）正是前置门 `factor_coverage_screen` 所跑、
并产出 **PASS 11 / ELIGIBILITY_FAIL 6** 分类的实现，而该分类是选项 (iii) 因子集的定义来源、
也是本轮全部覆盖率测量的基准。在裁决期间替换它，会让已记录的判定失去可复现的出处。

**C6.1 已补齐并勾选。**13 项点名指标此前只发出 5 项，现全部发出：
新增 `zero_ratio`、`value_median`、`value_mad`、`value_p01/p50/p99`、
`median_daily_dispersion`、`stale_share`、`duplicate_observation_rate`、`conflict_key_rate`、
`longest_missing_date_streak`、`date_coverage`、`security_coverage`、`finite_ratio`。
两处设计取舍已写进代码注释：**空日期不贡献 dispersion 样本**（记为 0 会把中位 dispersion 拉低、
让有缺失的因子看起来比单纯平坦的因子更稳定）；**`finite_ratio` 与 `coverage` 不可合并**
（前者描述源交付了什么、后者描述 universe 能用什么，一个因子可以完全有限却几乎不覆盖，
也可以全覆盖而源交付的一半不可用）。
`longest_missing_date_streak` 取**连续**空日而非计数——一天黑与一周黑是两种故障。
**并按规格补上快照主键**：`FactorQualitySnapshot` 新增 `scope_id` 与 `as_of_date`，
二者为**必填关键字参数**，空 `scope_id` 直接拒绝。字段名与类型对齐
`verify_admission_consistency.py::ScreeningRecord.scope_id`（既有契约，非新造）。
缺了这两项，快照只能与自身比较、无法被解释：同一因子在不同训练 scope、不同源版本下得分不同。
`conflict_key_rate` **仅在真的测了的时候才发出**——0.0 与「未测」是两个不同的陈述，
后者由 `unchecked_blockers` 指名。

**C6.2 的判定部分已补完，但仍不勾。**5 条 BLOCKER 现已齐备：
`PIT_VIOLATION`（`stale_days < 0` ⇔ `available_date > signal_date`）、
`CONFLICT_KEY_IN_CLEAN`、`NONFINITE_CLEAN_VALUE`、`UNKNOWN_REGISTRY_VERSION`、
`DUPLICATE_BUSINESS_KEY`（另含既有的 `DUPLICATE_UNIVERSE_KEY`）；
3 条 ELIGIBILITY_FAIL 阈值与规格逐一相符（`LOW_COVERAGE`<0.70、`LOW_UNIQUE_RATIO`<1.00、
`EXCESS_ZERO_DISPERSION`>0.20）。
后两条 BLOCKER 需要调用方提供输入（隔离键集、registry 已知版本），
**未提供时不静默跳过，而是记入 `FactorQualitySnapshot.unchecked_blockers`**——
「因为没看所以通过」正是本计划反复防范的失效模式，且该遗漏应留在被归档的快照里。
键比较两侧都规范化为 `(date, str)`：本仓库装配帧存 `datetime.date`、其它路径存
`pandas.Timestamp`，而类型不匹配的失效方向是**静默的**（永不相交 → 报「无重叠」），
已有专门测试覆盖跨类型命中。
**顺带修正一处相关的正确性问题**：`is_usable` 原本只要求 `stale_days <= 上限`、
**不要求 `>= 0`**，故一条 PIT 违规行会被算作可用、**虚增 coverage**，
与新加的 blocker 自相矛盾。现两者一致，并有断言钉住（违规行不计入 `finite_value_count`）。
**不勾的理由**：该条末句「ELIGIBILITY_FAIL 使因子不能进入该 fold 选因，但不删除历史记录」
描述的是**下游效果**，需要一个消费该快照的选因环节才能验证；本模块目前无消费者，
故只能验证到「分类正确、且纯函数不删除任何历史」为止。**接线之后应回来勾这一条。**

**变异验证**：把 `is_usable` 的 `>= 0` 放宽 → PIT 测试的 coverage 断言失败（捕获）；
`quality.py` 以 `git diff` 确认还原干净。
**全量：1305 passed / 45 skipped**（本轮起点 1289），`ruff check src tests` 清、
`mypy src` 156 文件清。
