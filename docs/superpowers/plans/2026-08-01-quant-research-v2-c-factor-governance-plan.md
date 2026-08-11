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

- [ ] **C3.3 增加顺序/分块性质测试**

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

- [ ] **C4.6 写未来扰动测试**

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

- [ ] **C6.1 固化结构质量指标**

按 `as_of_date, factor_id, factor_version, scope_id` 计算：universe member denominator、coverage、finite ratio、cross-sectional unique ratio、zero ratio、median/MAD、p01/p50/p99、每日 dispersion、stale share、duplicate/conflict rate、最长缺失连续天数、日期覆盖和证券覆盖。

- [ ] **C6.2 区分 blocker 与 eligibility flag**

BLOCKER：PIT 违规、冲突键进入 clean、非有限值进入 value、未知 registry 版本、业务键重复。`ELIGIBILITY_FAIL`：训练 scope coverage <70%、median daily unique ratio <1%、超过 20% 日期横截面 dispersion=0。后者使因子不能进入该 fold 选因，但不删除历史记录。

- [ ] **C6.3 写 denominator 与缺失测试**

coverage 分母必须是同日 universe 成员，不是“因子表中出现过的股票”；空日期返回 0 coverage 和明确状态，不返回 NaN 后被误当 PASS。基本面 stale >180 的值视为缺失。

- [ ] **C6.4 实现无标签质量报告**

报告不读取 `label_daily`，因此可用于 2026 SHADOW 运维；predictive IC 字段在本模块不存在，防止全期偷看标签。

- [ ] **C6.5 运行测试并提交**

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

- [ ] **C8.1 写 factor set 状态测试**

状态 `DRAFT → VALIDATED → FROZEN → RETIRED`。`FROZEN` 后成员/顺序/版本不可变；改变任何成员创建新 factor_set_id。一个 set 不得同时包含同一 factor 的两个版本。

- [ ] **C8.2 实现 content-addressed factor set**

factor_set_id 对排序后的 `(factor_id, factor_version, expected_direction)`、selection scope、规则版本和 registry definition hashes 做 SHA-256。member 表保存 deterministic feature order，训练/推理严格复用。

- [ ] **C8.3 扩充 B 计划黄金样本**

为同一 50×60 universe 增加至少：日频技术缺失、基本面周五公告、180 日 stale、修订公告、完全重复、single-nonnull、冲突、完全相同/符号相反因子。预期值由独立审计逻辑生成并逐案写 README。

- [ ] **C8.4 运行完整 C 计划验证**

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
