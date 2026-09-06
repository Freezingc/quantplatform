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

- [x] **C1.1 根据 Phase 0 清单逐因子建立 catalog**

每个现有因子必须有唯一条目，不能用运行时 `GP_*` 通配符自动接纳新列。每条 YAML 记录必须包含 `factor_id`、`factor_version`、`display_name`、`source_database`、`source_table`、`source_value_column`、`source_date_column`、`source_code_column`、`factor_type`、`frequency`、`expected_direction`、`availability_policy`、`max_staleness_calendar_days`、`unit` 和 `owner`。

`source_database` 固定为 `dfs://YesLab`；`source_table` 必须是 Phase 0 从 `factor_metadata.store_factor` 冻结出的 `factor_data1` 至 `factor_data11` 中的准确表名；值/日期/证券字段按当前源契约固定为 `value`、`trade_date`、`ts_code`。C1 的生成命令必须把冻结的 metadata snapshot 转成显式逐因子 YAML，并在人审后保存 snapshot hash；不存在待运行时替换的表名。

- [x] **C1.2 写失败测试：稳定 ID 与版本不可变**

```python
def test_definition_change_requires_new_factor_version(registry):
    registry.register(FACTOR_V1)
    with pytest.raises(ImmutableDefinitionError):
        registry.register(replace(FACTOR_V1, unit="percent"))
```

覆盖重复 ID/version、未知 factor type、无 owner、源 URI 非 YesLab、technical 配置 staleness>0、fundamental 缺 ann_date mapping、definition hash 变化。

- [x] **C1.3 定义类型与 definition hash**

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

- [x] **C1.5 实现 catalog 对源 schema 的 dry-run**

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

- [x] **C2.1 写失败测试：宽表/长表均映射到统一契约**

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

- [x] **C2.2 实现分列/分月读取**

按注册表只查询所需列和日期，不加载 135 列全历史到单机；默认月份 chunk + factor group，输出 canonical long form。chunk 边界不影响最终 hash。

- [x] **C2.3 实现数值规范化**

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

- [x] **C3.2 实现纯函数 resolution**

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

- [x] **C4.1 写技术因子可用日测试**

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

- [x] **C5.2 实现统一 observation 输出**

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

- [x] **C7.3 实现阈值图 connected components**

`median_daily_abs_spearman >= 0.90` 建无向边，因子簇为 connected components。cluster id 对排序后的成员 ID 做 SHA-256，保证输入顺序不影响结果。

- [ ] **C7.4 实现训练内代表因子排序接口**

本模块只接受 D 计划传入的训练内 quality/predictive statistics；代表排序固定：eligible → FDR pass → direction stability → median abs IC → coverage → lower conflict rate → factor_id ASC。不得读取 validation/test 指标。

- [x] **C7.5 回归测试已知高相关因子**

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

## 勾选审计续五（2026-09-06 08:05）：C1.2 / C1.3 已勾，并修掉一个 fail-open 缺陷

**查出一个真实缺陷（已修，commit 见下）：`FactorDefinition` 的类型专属校验是 fail-open 的。**
`contracts/factor.py::__post_init__` 用**身份比较**分派类型专属规则
（`self.factor_type is FactorType.TECHNICAL`、`self.availability_policy is not
AvailabilityPolicy.NEXT_TRADING_DAY_AFTER_ANNOUNCEMENT`），却**从不校验这两个字段是否为合法枚举成员**。
`FactorType` 是 `StrEnum`，值相等的普通字符串与枚举成员**并非同一对象**，故 `is` 为假、
分支整体跳过、规则静默不适用。dataclass 也不在运行时强制类型注解。实测四种组合：

| 输入 | 期望 | 修复前实测 |
|---|---|---|
| `factor_type=FactorType.TECHNICAL`, staleness=30 | 拒绝 | 拒绝 ✓ |
| `factor_type="TECHNICAL"`（字符串）, staleness=30 | 拒绝 | **接受** ✗ |
| `factor_type="BOGUS"` | 拒绝 | **接受** ✗ |
| `factor_type="FUNDAMENTAL"` + `availability_policy="SAME_DAY_CLOSE"`（均为字符串） | 拒绝 | **接受** ✗ |

**第四例后果最严重**：availability 正是把公告挡在「它存在之前」的那道机制，
一个被当作 SAME_DAY_CLOSE 的基本面因子会**在其公告日当天即可用**，即市场尚无从反应之前——
**那是前视偏差**，正是 PIT 层存在的理由。注意该绕过需要**两个字段都是字符串**：
若 `factor_type` 是真枚举，原有的 fundamental 检查仍会触发。

**当前是否可达**：`src/` 中唯一的构造点 `factors/source_backfill.py:324` **做对了**
（`FactorType(str(...))`、`AvailabilityPolicy(str(...))` 显式转换，未知值会被枚举本身拒绝）。
且 **catalog → `FactorDefinition` 的加载器目前并不存在**（那正是 C1.5 dry-run 与 C5.3
`registry import --catalog` 缺失的部分）。所以缺陷尚未被触发——但
`configs/factors/source_catalog.yaml` 里 487 个条目的 `factor_type` **本来就是 YAML 字符串**，
未来任何加载器只要忘了转换就会命中。**在一个自称「Fail-closed registry」的模块里，
契约对象自身 fail-open 是设计缺陷，不是调用方的问题。**

**修法**：在类型专属规则**之前**校验两个字段均为对应枚举成员，否则 `ValueError`，
错误信息直接给出修法（「coerce with FactorType(value) rather than passing the raw catalog
string」）。选择**拒绝而非静默转换**：这是不可变契约对象，在 frozen dataclass 内部
改写调用方输入属于隐式变更，与本项目其余部分的做法相反
（如 `describe_runtime_environment` 只观测不配置）；且唯一的现有调用方本就显式转换，故拒绝零成本。
**附带收益**：修复前对字符串类型的定义调用 `definition_hash` 会在**哈希时**抛
`AttributeError: 'str' object has no attribute 'value'`（已实测），即一个晚期且与真因无关的崩溃；
现在改为构造期的清晰拒绝。

**C1.2 已勾**：其列举的 7 类情形现已全部有测试——重复 ID/version（同定义幂等返回原条目、
改定义则 `ImmutableDefinitionError`）、未知 factor type、无 owner、源 URI 非 YesLab、
technical staleness>0、fundamental 缺公告可用策略、definition hash 变化。
其中「无 owner」「fundamental 缺公告可用策略」「重复注册幂等」三类**此前已实现但无测试**，
补测即通过；另三类中未知类型与两处字符串绕过**此前既未实现也未测试**。
**C1.3 已勾**：`definition_hash` 对 13 个字段（availability_policy、expected_direction、
factor_id、factor_type、factor_version、frequency、max_staleness_calendar_days、
四个 source 映射列、source_database、unit）做 `sort_keys` 规范 JSON 后 SHA-256，
覆盖规格点名的语义/源映射/方向/版本/频率/单位/availability policy；
且 `display_name` 与 `owner` **不在载荷内**，正是规格要求的「描述文案和 owner 变更可通过
metadata revision」，已由 `test_metadata_changes_do_not_change_definition_hash_or_active_state` 断言。

**全量：1312 passed / 45 skipped**（本节起点 1305），`ruff check src tests` 清、`mypy src` 156 文件清。

### 同一反模式的全仓排查（2026-09-06 08:12）：仅因子契约一处，已修

既然 `is <Enum>.MEMBER` 在字符串误入时会静默跳过守卫，就把全仓的正向身份比较都过了一遍。
**结论：只有 `contracts/factor.py` 一处构成 fail-open，即上面已修的那处。**

判据是**值能否以字符串形态到达比较点**，而不是比较本身有多少处：

- **正向 `is <Enum>.X`（危险方向：字符串会让守卫不触发）** 共 20+ 处，分布在
  `monitoring/notifier.py`、`monitoring/data_health.py`、`monitoring/drift.py`、`db/publish.py`、
  `inference/daily_pipeline.py`、`reporting/daily.py`。逐一核查其比较对象：
  **全部是内部状态机自行赋值的枚举成员**（`StagedBatch.state`、pipeline step status、
  drift/notifier severity），没有反序列化边界喂字符串进来，故安全。
- **`is not <Enum>.X`（安全方向：字符串会让守卫更严）** 分布在
  `registry/model_registry.py`（6 处）、`registry/active_pointer.py`、`registry/locked_test.py`、
  `monitoring/shadow.py`、`compact/publisher.py`、`training/trainer.py`、`training/search.py`。
  这些即便有字符串误入也是 fail-closed（把「是 PRODUCTION」误判为「不是」→ 拒绝迁移），
  方向安全。**并且经核查它们根本不可能收到字符串**：`ModelRegistry` 把记录保存在内存
  （`self._records: dict[str, ModelRecord]`，`model_registry.py:145/170`，用枚举字面量构造），
  **该类没有从 DB 反序列化记录的路径**；全仓 `ModelStatus(...)` 除类定义外**零次调用**，
  也不存在 `status=row[...]` / `status=str(...)` 这类从行赋值的写法。

**为何唯独因子契约中招**：它是唯一一个**输入直接来自文本配置**的契约对象——
`configs/factors/source_catalog.yaml` 的 487 个条目里 `factor_type` / `availability_policy`
本来就是 YAML 字符串，而其余状态机的枚举值都由代码自己产生。
**可推广的规则（建议写进后续 review 清单）**：任何在**反序列化边界**（YAML / JSON / DB 行）
构造的契约对象，必须在构造时校验枚举成员身份，因为其下游的类型专属规则很可能用身份比较分派；
`StrEnum` 让「值相等的字符串」看起来完全正常，只有 `is` 会暴露它。

## 勾选审计续六（2026-09-06 08:30）：C2.2 / C2.3 已勾，C2.1 差一款

**C2.3 已勾——三款全部有测试，其中两款此前无测试。**
「±inf 转为 invalid flag 而非有效 NaN」此前已测；本轮补上另外两款：
**「保留 raw value 精度」与「不在 source reader winsorize、scale 或 impute」**。
新测试 `test_the_reader_never_alters_drops_or_invents_a_value` 断言输出行数等于输入行数
（不丢不增），且每个发布值与源在同键上交付的某个值**逐位相同**（用 `float.hex()` 比较，
两个 NaN 视为同一种缺失）；`test_extreme_but_finite_values_are_passed_through_not_clipped`
断言 ±1e308 原样通过且被标记为有效。这与去重层「永不合成值」是同一性质、同一理由：
source reader 是值进入平台的入口，**它改动的任何东西都与「源本来就这么说」无法区分**，
被裁剪的离群值或被插补的均值会传进每一个因子值、每一个基线、每一个信任根，而下游无从察觉。
**变异验证**：在 reader 里插入 `.clip(-1e6, 1e6).fillna(0.0)`（即 winsorize + 插补）→
**3 条测试失败**（新增两条 + 既有的 `marks_nonfinite_values`，因 `fillna(0.0)` 把 NaN 变成了有效的 0）；
而列序、额外列、分块三条不变量测试在变异下仍通过是**正确的**——它们比较的是 reader 输出对 reader 输出，
属不变量而非值保真。变异后 `source_reader.py` 以 `git diff --quiet` 确认还原干净。

**C2.2 已勾——四款全部核实，其中取数层此前完全无测试。**
`factors/source_backfill.py::_query_source_month` 实测：
**只投影 4 列**（`select ts_code, trade_date, factor_name, value`）、
**按 factor group 过滤**（`factor_name in ['Alpha1', 'Alpha2']`）、
**按月份 chunk 过滤**（`trade_date between date('2025.03.01') and date('2025.03.31')`，
来自 `FactorMonth.from_partition`），故不会把 135 列全历史拉到单机；
输出为 canonical long form（11 列，已由 C2.1 相关测试钉住列集合）。
新增 `tests/unit/factors/test_source_backfill_query.py`（10 测试）钉住查询形状，
并把**表名白名单当作注入防护来测**：6 个不安全表名（`factor_data12`、`factor_data0`、
`factor_data`、`stock_factor_daily`、`factor_data1') ; drop table x --`、`factor_data1,factor_data2`）
必须在**发出任何查询之前**被拒——测试断言 `session.queries == []`，而不只是断言最终抛错；
另测因子名内的单引号被转义为 `''`、返回 schema 缺列被拒、以及返回帧被**重投影到恰好那 4 列**
（多出的列不得搭便车进入管线）。
**「chunk 边界不影响最终 hash」已测**：`_source_row_id` 哈希的是
`(表名, 代码, 观测日, 因子名, 值, 组内出现序号)`，**不含帧内位置**，
而 `occurrence` 是 `(代码, 日期, 因子名, 值)` 组内的 `cumcount`；
故按 C2.2 规定的两个轴（月份、因子组）分块时同一组永不被切开，row id 必然不变。
测试按月份切成两块分别归一化再拼接，与整帧一次归一化的结果**逐帧相等（含 `source_row_id`）**。
注：任意行切分不在此列，也不是规格所指的 chunk。

**C2.1 仍不勾——五款中四款已满足，差「额外未注册列被报告」。**
已满足并测试：统一输出契约（11 列，含 C2.3 要求的 `value_is_valid` / `invalid_reason`，
是规格所列 9 列的超集）、**源列顺序改变不影响输出**、额外未注册列**不被加载**、
缺少注册列时抛 `SourceFactorSchemaError` 阻止该因子批次。
**未满足的一款**：「额外未注册列**被报告**」——reader 只是丢弃它们，没有任何报告机制。
按 C1.5 的措辞（dry-run 输出「`mapped/blocked/retired` 数量和**未映射源列**」），
报告本应属于 catalog dry-run，而 **C1.5 尚未实现**，故该款无处落地。
**关于「宽表」**：实测所有 11 张 `factor_data1..11` 源表都是**长表**
（取数一律 `select ts_code, trade_date, factor_name, value ... where factor_name in [...]`，
且该路径已在生产中用于冻结集物化），故「一列一个因子」的宽表映射在本源上**不出现**；
规格标题里的「宽表列顺序改变不影响输出」按字面已由列序不变测试覆盖。
**因此 C2.1 的唯一实质缺口与 C1.5 是同一件事**，应一并处理。

**全量：1327 passed / 45 skipped**（本节起点 1312），`ruff check src tests` 清、`mypy src` 156 文件清。

## 勾选审计续七（2026-09-06 08:45）：C7.3 已补齐并勾选，C7.4 有两处不符（一处可证无实质影响）

**先更正本文书前面一处定位。**「勾选审计（06:45）」曾记 C7.3/C7.4 在
`datasets/selection.py:177 select_cluster_representatives`，并以「带路径减半的并查集」为据认为
C7.3 已实现。**只对了一半**：连通分量确实实现了（内部用于挑代表），但 C7.3 还要求
「**cluster id 对排序后的成员 ID 做 SHA-256，保证输入顺序不影响结果**」——而
`ClusterSelection` 只有 `selected_factor_keys / status / validation_reason`，
**既无 `cluster_id`，簇本身也完全不对外暴露**，`selection.py` 里当时没有任何 SHA-256。
后果是聚类结果**无法被引用、无法跨折比较**；而 `factor_set_id` 早已内容寻址（C8.2），
于是管线里唯一没有稳定身份的分组就是因子簇。

**已实现并勾选 C7.3**：新增 `FactorCluster(cluster_id, members)` 与 `cluster_identity(members)`
（对排序后的 `[[factor_id, factor_version], ...]` 做 `sort_keys` 紧凑 JSON 后 SHA-256），
`ClusterSelection` 增加 `clusters` 字段并按 `cluster_id` 排序返回。6 条新测试覆盖：
内容寻址 id 与独立算出的期望值一致、两个输入帧各自反序后 `clusters` 完全相同、
**链式 A–B / B–C 合成一个簇 {A,B,C}**（C7.5 的传递性案例；按对分组会错分成两簇、
从而留下两个实为同一信号的代表）、低于阈值或 `status != PASS` 的边不连簇、
非 ELIGIBLE 候选不参与聚类、以及版本不同必须得到不同 id。
`select_cluster_representatives` **无生产消费者**（仅自身测试），故扩展返回类型零下游风险。

**变异验证顺带查出一个可复现性事实（值得单独记）。**把 `cluster_identity` 内部的 `sorted()`
去掉后，**哪条测试捕获它在两次运行间不一致**：第一次只有「与期望 id 比对」那条失败，
第二次「顺序无关」那条也失败。原因是 `select_cluster_representatives` 用**集合推导**构造 `keys`，
而 Python 的字符串哈希**每进程随机化**，故 `grouped` 的成员顺序跨进程会变。
即：**去掉那次看似冗余的内部排序，cluster id 会变成依赖 PYTHONHASHSEED** ——
这是可复现性缺陷，不是外观问题（同一聚类两次运行得不到同一个 id）。
已实测当前实现在 `PYTHONHASHSEED=0/1/12345/99999` 下 cluster id 与成员顺序**逐一相同**。
该理由已写进 `cluster_identity` 的 docstring，以防这次「冗余」排序日后被清理掉；
并补了一条**直接针对该函数**的顺序无关测试，因为经 `select_cluster_representatives`
写的那条无法钉住它（构造点已先排过序）。

**C7.4 仍不勾：实现的排序键与规格有两处不符，其中一处会改变选出的代表因子。**
规格：`eligible → FDR pass → direction stability → median abs IC → coverage → lower conflict rate → factor_id ASC`
实现（`_representative_sort_key`）：`(fdr_q_value ASC, -signed_mean_rank_ic, -positive_year_count, -coverage, conflict_rate ASC, factor_id ASC, factor_version ASC)`

- **(a) 中间两项顺序相反（实质影响，需裁决）**：规格是 direction stability 先、IC 后；
  实现是 IC 先（`-signed_mean_rank_ic`）、`-positive_year_count` 后。
  当两个因子一个 IC 更高而方向稳定性更差时，**选出的代表不同**，即模型输入不同。
  **故不代改**：这不是补测试能解决的，而是改判据；应由 owner 裁定以哪份为准。
- **(b) IC 字段是 signed mean 而非 median abs（可证一半无影响）**：
  「signed vs abs」**在本管线上无实质差别**——`ELIGIBLE` 只由 `_selection_status` 赋予，
  而它在 `frozen_direction == 0 或 signed_mean_rank_ic <= 0` 时返回 `NON_POSITIVE_MEAN_IC`，
  故进入代表排序的候选必有 `signed_mean_rank_ic > 0`，此时 signed 与 abs 相等。
  但「**mean vs median**」是**真实差异**：候选帧里只有 `signed_mean_rank_ic`，
  没有 median IC 字段，故要按规格实现必须**在上游新增一个统计量**，属更大改动。
- 其余各项相符：eligible 由上游 `status == "ELIGIBLE"` 过滤、FDR 为第一排序键、
  coverage 降序、conflict_rate 升序、factor_id 升序（实现另加 factor_version 作最终 tie-break，
  规格未提但属合理补强）。
**「不得读取 validation/test 指标」这一款已满足**：排序键只用
fdr_q_value / signed_mean_rank_ic / positive_year_count / coverage / conflict_rate / 身份字段，
全部是训练内统计，无任何 validation/test 指标。

**全量：1333 passed / 45 skipped**（本节起点 1327），`ruff check src tests` 清、`mypy src` 156 文件清。

## 勾选审计续八（2026-09-06 08:50）：C7.5 已勾，C7.1 / C7.2 各差一款且都不宜代补

**已勾选 C7.5。**规格要求四款案例，逐款落点如下（均已通过，且都做了变异验证）：

- **完全重复**与**符号相反但绝对相关**：新增端到端测试
  `test_selection.py::test_identical_and_mirrored_factors_share_one_cluster_and_one_representative`。
  它从**原始数值**出发，经 `compute_factor_similarity` 到 `select_cluster_representatives`，
  构造 `dense / copy（=dense）/ mirror（=-dense）/ other（乱序置换）` 四个因子，
  先断言前提（三对互为 ±1、含 `other` 的各对 <0.90、全部 `status == PASS`），
  **再**断言聚类结果：`{copy, dense, mirror}` 合成一簇、`{other}` 独立一簇、
  代表为 `dense` 与 `other`。先钉前提是为了让聚类断言**不可能靠退化的 fixture 蒙过**。
  选 `dense` 而非 `copy`/`mirror` 作代表，是因为三者 FDR q 值递增（0.001/0.010/0.020），
  而 FDR 是排序首键——这一点由 C7.4 的排序键决定，本测试只是顺着它断言。
- **缺失错位**：`test_similarity.py::test_misaligned_missingness_skips_the_day_rather_than_counting_it`。
  两个因子各自都populate得很满，但第 2、3 天只在 10 个标的上重叠；
  这两天必须**被整天跳过**而不是拿 10 个样本去算相关，于是有效日只剩 1 天、落到 `INSUFFICIENT`。
  这条与「有效日下限」是同一处代码的两个后果，但只有它钉住了**为什么**要跳天：
  在 10 个标的上算出的 rho 看起来是个测量值，其实不是。
- **链式 A–B / B–C 相关与传递边**：已由续七的
  `test_selection.py::test_chained_edges_form_one_cluster_through_transitivity` 覆盖（本节不重复）。

**变异验证：4 个变异，全部被预期的那条测试捕获——但其中一个变异起初没被捕获，值得记。**

1. 去掉 `correlations.append(abs(...))` 的 `abs` → 端到端镜像测试失败
   （`rho == -1.0`，既过不了 `== 1.0` 的前提断言，也因 `-1.0 < 0.90` 不连边而使簇数从 2 变 3）。
2. 把「共同成员数 < `minimum_common_members` 则跳过该日」的下限改成 2 →
   缺失错位测试失败（`valid_days` 从 1 变 3、`INSUFFICIENT` 变 `PASS`）。
3. 让 `_is_finite` 只查 `pd.notna`（即**无穷大算作可用观测**）→ **起初 15 条测试全过**。
4. 让 `_is_finite` 恒为真（即**完全不筛非有限值**）→ 同上，起初不失败。

**3、4 暴露的是我第一版非有限值测试写得不对。**它把因子 `b` 的第 31–40 号标的设为 `inf`，
留下**恰好 30** 个有限对，然后断言 `valid_days == 1 / PASS`。问题在于：
把 `inf` 也算进去时共同成员是 40，同样 `>= 30`，**结论一模一样**。
也就是说那条测试在「正确实现」与「被破坏的实现」下都通过——它钉住的是 `>=` 这个边界，
**完全没有钉住"非有限值不算观测"这件事**。这类测试比没有测试更危险，因为它给人已覆盖的错觉。

**已按"变异必须改变结论"的原则重写为两条**：

- `test_non_finite_values_cannot_prop_up_a_day_below_the_member_floor[inf|nan]`：
  40 个共同标的中**破坏 11 个**，只剩 29 个有限对——比下限 30 **少一个**。
  正确实现必须整天跳过（`valid_days == 0`、`INSUFFICIENT`、中位数为 `NaN`）；
  若把填充值算作观测则 `>= 30` 而给出 `PASS`。**两种实现的结论相反**，故能区分。
  参数化同时覆盖 `inf` 与 `nan`，因为二者的失效方式不同：
  `inf` 会真的参与 `rank()` 从而算出一个 rho；
  `nan` 会被 pandas 的 `rank(na_option="keep")` 与 `corr()` 悄悄丢弃，
  于是**用 29 个标的算出的相关，却对外声称覆盖了 40 个**——数值更可信、错得更隐蔽。
- `test_a_day_exactly_at_the_common_member_floor_still_counts`：破坏 10 个、剩**恰好 30**，
  断言 `valid_days == 1 / PASS`，把原先那条真正想钉的 `>=`（而非 `>`）边界单独保住。

重写后重跑变异 3、4：`[inf]` 与 `[nan]` 两个参数**都失败**，即变异被捕获。
**方法上的教训**：变异验证的判据不是"改了实现有没有测试红"，而是
"**这条测试在正确实现与被破坏实现下是否给出相反结论**"。只跑到"全绿"就收工，
会把一条恒真的测试当成覆盖。

**C7.1 仍不勾——差「universe_id / data_version 绑定与 query builder」。**
规格要求「输入明确 `start_date/end_date/universe_id/data_version`，query builder 必须带闭区间」，
并要求「测试在 end_date 后添加高度相关数据不改变历史 scope 相似度」。

- **日期闭区间与 scope 隔离这一款已满足并有测试**：
  `test_similarity.py::test_similarity_is_scope_bounded_and_uses_absolute_daily_spearman`
  正是在 `end_date`（2025-01-03）之后追加了 2025-01-06 的一对完全相同数据，
  断言 `valid_days` 仍是 2 而非 3。实现里
  `(signal_date >= start_date) & (signal_date <= end_date)` 为闭区间。
- **缺的是绑定与取数**：`compute_factor_similarity` 的签名只有
  `(observations, *, start_date, end_date, minimum_common_members, minimum_valid_days)`，
  **没有 `universe_id`，没有 `data_version`，模块里也没有任何 query builder / SQL**
  （`grep universe_id|data_version|SELECT` 在 `similarity.py` 中零命中）。
  它接收的是**调用方已装配好的内存长表**，因此"scope"目前只由日期界定：
  **同一个 universe 换个版本、或换个 universe 但日期相同，算出的相似度无从区分，也不会报错。**
  这与 C7.3 已交付的 `cluster_id` 内容寻址形成对照——簇有了稳定身份，
  但**产生它的相似度矩阵没有记录自己是在哪个 universe / 哪个数据版本上算的**，
  于是跨折或跨版本比较簇时无法判断可比性。
- **不代补的理由**：与 C7.4 同理，`select_cluster_representatives` 与
  `compute_factor_similarity` 目前**都没有生产调用者**（全仓 grep 仅命中自身定义与测试）。
  把 `universe_id/data_version` 加进签名要同时决定它们的**来源与校验方式**
  （是从 `factor_daily` 的查询条件回填，还是由 D 计划的 scope 对象传入并要求与观测帧自洽），
  这属于接口契约设计，应由 owner 定，而不是在测试轮里顺手加两个未被消费的参数。

**C7.2 仍不勾——差「记录覆盖」，且覆盖的分母口径规格里没有定义。**
规格四款中三款已满足：同日共同非空数 `>= 30` 才算 Spearman（`minimum_common_members` 默认 30）、
scope 相似度取有效每日 `abs(rho)` 的**中位数**、有效日 `< 60` 标 `INSUFFICIENT`
（`minimum_valid_days` 默认 60）；「不连边」也已满足——
`select_cluster_representatives` 只用 `status == "PASS"` 的行建边
（由 `test_edges_below_the_threshold_or_not_passing_do_not_join_clusters` 钉住）。

**缺的是输出里没有覆盖字段**：返回列为
`factor_id_left/version_left/factor_id_right/version_right/median_daily_abs_spearman/valid_days/status`，
`valid_days` 是**有效天数**，不是规格另外要求的**覆盖**。二者不能互相代替：
`valid_days` 只回答"有几天够格算"，不回答"这几天里每个横截面有多满"。
两个 pair 可以同为 `valid_days == 60`，一个每天在 4000 个标的上算、另一个每天刚好卡在 30 个下限——
**当前的输出无法区分这两种证据强度**，而下游正是拿这个中位数去连边、去选代表的。

**不代补的理由是口径未定**：规格只写「记录有效天数/覆盖」，没说覆盖是什么比值。
候选至少有三种，且给出的是不同的数：
(i) 有效日的共同成员数中位数 ÷ 该日 universe 全量；
(ii) 有效日共同成员数 ÷ 两因子各自覆盖数的并集；
(iii) 有效天数 ÷ scope 内总交易日数（这一种其实已被 `valid_days` 隐含，只差归一）。
(i) 最贴近"证据强度"的直觉，但它需要**额外的 universe 口径输入**——
而这恰好是 C7.1 缺的那个绑定；也就是说 **C7.2 的覆盖字段依赖 C7.1 先定下 universe 的来源**，
两项应当一并裁决，不宜先挑一个分母写进去、事后再改（改分母会改变 `INSUFFICIENT` 之外的
所有输出数值，进而可能改变连边与代表选择）。

**全量：1340 passed / 45 skipped**（本节起点 1338，+2 来自非有限值测试改写为 1 条参数化 2 例 + 1 条边界），
`ruff check src tests` 清、`mypy src` 156 文件清。
`ruff format` 未纳入门禁（全仓 159 个文件本就不符合其风格、且无任何配置或 CI 钩子引用它），故不据此改动测试文件。

## 勾选审计续九（2026-09-06 09:15）：C1.5 已实现并勾选，C2.1 的唯一缺口随之关闭

续六的结论是「**C2.1 的唯一实质缺口与 C1.5 是同一件事**」——C2.1 差「额外未注册列**被报告**」，
而按 C1.5 的措辞（输出「`mapped/blocked/retired` 数量和**未映射源列**」），报告本就属于 dry-run，
但 C1.5 当时未实现，故该款无处落地。**本节交付 C1.5，两项一并勾选。**

### 交付内容

- **`research_v2/src/research_v2/factors/catalog_dry_run.py`**（新）：纯函数
  `run_catalog_dry_run(entries, schemas, *, retired, schema_errors)` 输出
  `CatalogDryRunReport(findings, unmapped_source_columns, schema_fingerprints)`，
  带 `mapped_count / blocked_count / retired_count` 与 `to_frame()`。
  另暴露 `registered_source_columns(entries, table)`（该表仍被消费的列，排序后）与
  `collect_source_schemas(session, tables)`（唯一会碰会话的函数）。
- **`research_v2/scripts/audit_factor_catalog_schema_dry_run.py`**（新）：跑真实 catalog，
  打印逐表 schema 与三类计数、未映射源列、blocked 明细，写 JSON 报告；
  有 blocked 或 unmapped 时**退出码 1**（可作 catalog 变更的门），`--report-only` 降级为只打印；
  `--save-schemas` / `--schemas-from` 让判定可离线复演。
- **41 条新测试**：`tests/unit/factors/test_catalog_dry_run.py`（28）与
  `tests/unit/scripts/test_audit_factor_catalog_schema_dry_run.py`（13）。

### 逐条对应 C1.5 的四款检查

- **表/列存在**：表名先过冻结集白名单 `factor_data(?:[1-9]|10|11)`；表不在 schema 集内 → BLOCKED；
  角色列缺失 → BLOCKED。**并额外要求判别列 `factor_name`**：catalog 里没有任何条目把它登记为列
  （条目存的是因子**名**作为值），但 `source_reader` 与两条取数路径都靠它
  `where factor_name in [...]` 选出单个因子，**缺它则整表不可读**。
- **类型可转换**：按各角色**实际的读法**判定，而不是笼统的"类型对不对"——
  `value` 走 `pd.to_numeric`，故必须是数值类型（**STRING 会被静默 coerce 成全 NaN**，
  看起来像"这个因子没有数据"而不是"schema 不匹配"，这正是必须前置拦下的失效模式）；
  `trade_date` 走 `pd.to_datetime(errors="raise")`，故必须是时间类型；
  `ts_code` 与 `factor_name` 必须是 SYMBOL/STRING/CHAR。DolphinDB 的参数化类型
  （如 `DECIMAL32(S)`）只取头部 token 再比对。
- **日期/代码键明确**：一个角色只能指一列、**一列只能担任一个角色**，否则读路径对
  "哪一列是日期键"没有唯一答案（如 `source_code_column="trade_date"` → BLOCKED，理由含 ambiguous）；
  源 schema 内**同名列出现两次**也判为不明确。
- **输出三类计数与未映射源列**：未映射列**按表**给出（同名列在两张表是两件事）。
  参照集是**非 retired 条目**登记的列 ∪ 判别列——retirement 是消费的主动终止，
  故只有 retired 条目用过的列在其退役后**应当**变为未映射并被报告出来（有专测）。
  schema 查询失败的表**不计入**未映射（无从知晓），但会以
  `RuntimeError: ...` 原文出现在该表因子的 BLOCKED 理由里，
  **让连接故障与 schema 漂移可区分**——否则一次断连会读作"487 个因子全部 blocked"。

### 三条红线是可验证的，不是文档声明

- **不读取全量值**：`collect_source_schemas` 能发出的**唯一**脚本由 `_schema_query` 生成，
  形如 `schema(loadTable('dfs://YesLab', '<table>')).colDefs`。
  实跑记录到 JSON 里复核过：**恰好 11 条查询、全部匹配该形状、无一含 `select`/`where`**。
  测试另从模块 AST 的字符串常量里断言不存在 `select ... from` 形态的查询。
- **不写数据库**：脚本**根本不构造 `V2WriteSession`**（也不 import）。
  该性质用 **AST 而非文本**钉住：遍历 `ast.Name`/`ast.Attribute` 收集标识符，
  断言 `V2WriteSession`、`UploadClient` 不在其中而 `YesLabReadOnlySession` 在。
  选 AST 是因为脚本的 docstring 与注释里**都提到了** `V2WriteSession`（用来说明它没被用），
  纯文本匹配会同时产生假阳与假阴；而 AST 也拦得住"在 `main()` 内部再 import"这种规避。
  另用 `db.session._MUTATING_COMMAND_PATTERN` 扫全部字符串常量，确认无任何 DDB 变更动词。
- **不探测任意表**：表名白名单在**发出任何查询之前**校验，
  测试断言的是 `session.queries == []` 而不只是"最终抛错"——与续六对取数层的注入防护同一标准。
  一个拼错的 catalog 条目因此不能变成探测源集群上别的表的手段。

### 实跑证据（2026-09-06 09:05，只读）

```
catalog=source_catalog.yaml entries=487 tables=11
metadata_snapshot_hash=afb9cc396189e47bb4963d8df65674867905606371ebec9f6998e8e2c69ce3f6
  factor_data1..11: ts_code:SYMBOL, trade_date:DATE, factor_name:STRING, value:DOUBLE
mapped=487 blocked=0 retired=0 total=487
unmapped source columns: none
DRY_RUN_PASSED
```

产物：`/data/zhangyudong/qrv2_research_artifacts/catalog_schema_dry_run{.txt,_report.json,_schemas.json}`。
**11 张源表的 schema 完全一致**（4 列、同类型），故真实数据下 dry-run 报不出任何东西。
这本身是个值得记的结论：**它当下的价值不在"发现了什么"，而在"日后源表加列/改类型/删表时会立刻发现"**，
也意味着它的正确性**只能由测试确立**（合成漂移场景），不能由这次干净的空跑证明。
离线复演（`--schemas-from` 读回保存的 schema）产出的报告与实跑**逐字节相同**。

### 顺带闭上一处真实的三角缺口（新发现，非规格所列）

写测试时发现「**哪些列会被读**」此前有**两个互不校验的真值来源**：
catalog 的逐条登记，与 `source_backfill._SOURCE_COLUMNS = ("ts_code","trade_date","factor_name","value")`
（`source_evaluation._query_source_range` 里是同一份硬编码投影）。
dry-run 校验 catalog ↔ **源**，续六的取数测试校验 **reader** ↔ 返回帧，
**唯独 catalog ↔ 读路径这一条边没有任何东西看着**。
后果是具体的：若某条目登记 `source_value_column: "val"` 而源表同时有 `val` 和 `value`，
dry-run 会判 MAPPED（列确实存在、类型也对），而取数路径**照旧 select `value`** ——
该因子被从错误的列读出，或根本读不到，且**两道现有检查都放行**。
已补 `test_the_catalog_registration_agrees_with_the_columns_the_read_path_projects`：
对**真实 487 条**逐条断言 `registered_source_columns([entry], table) == sorted(_SOURCE_COLUMNS)`。
三条边现已闭合。

### 变异验证：10 个变异全部被预期的测试捕获，其中一个变异第一次写错了位置

| # | 变异 | 捕获它的测试 |
|---|---|---|
| 1 | 去掉 `abs()`（C7.5 相关，见续八） | 端到端镜像测试 |
| 2 | 降低共同成员下限（C7.5 相关，见续八） | 缺失错位测试 |
| 3 | 无穷大算作可用观测（C7.5 相关，见续八） | 改写后的非有限值测试（两参数） |
| 4 | 完全不筛非有限值（C7.5 相关，见续八） | 同上 |
| 5 | 白名单校验挪到查询循环之后 | `..._refused_without_being_queried`（断言 `queries == []`） |
| 6 | `_schema_query` 改成 `select top 1 *` | `..._issues_schema_queries_only_and_never_reads_values` |
| 7 | retired 条目照常被重新检查 | retired 两条测试 |
| 8 | retired 条目仍算作消费其列 | `test_columns_only_a_retired_entry_used_become_unmapped` |
| 9 | 跳过类型可转换检查 | 6 个参数化类型用例全部失败 |
| 10 | 角色列硬编码而不读 catalog 登记 | 4 条测试（含 blocked 退出码那条） |

**方法论上的第二次教训，与续八那次不同但同源。**第一版「只报第一个缺陷」的变异**活了下来**：
我把 `if reasons: break` 插在了类型检查之前，而该测试的两个缺陷都是**列不存在**，
走的是更早的 `continue` 分支，**根本到不了我插入的那一行**。
变异没生效不等于性质没被测到——**变异必须真的改变被断言的那条路径**。
把它挪到角色循环开头（M6b）、以及在 `_check_entry` 末尾截断 `reasons[:1]`（M7）后，
`test_every_defect_of_one_factor_is_reported_not_just_the_first` 两次都失败。
与续八的教训合起来是同一句话的两面：**变异验证的判据是"正确实现与被破坏实现结论相反"，
而且要确认变异真的落在了被测代码上。**

### C2.1 勾选的边界（沿用续六的判定，明确留档）

五款现已全部有测试：统一输出契约（11 列）、源列顺序不影响输出、
**额外未注册列被报告**（本节 dry-run + `test_extra_unregistered_source_columns_are_reported_but_not_loaded`）
**且不被加载**（续六的 `test_extra_unregistered_columns_are_not_loaded`、取数层重投影到恰好 4 列，
加本节的 `registered_source_columns` 派生）、缺少注册列阻止该因子批次。
**沿用续六的范围判定并在此复述以免被误读**：实测 11 张源表**全部是长表**，
「一列一个因子」的宽表在本源上不出现，故「宽表映射」一款按字面由列序不变测试覆盖，
**并非在某个宽表源上验证过**。若日后接入宽表源，该款需重新核验。

**全量：1381 passed / 45 skipped**（本节起点 1340，+41 新测试），
`ruff check src tests` 清、`mypy src` 157 文件清。

## 勾选审计续十（2026-09-06 09:45）：C1.1 已勾并实测可复现；C1.4 有一款根本没实现，且实现了也拦不住任何东西

### C1.1 已勾选——并顺带证明了一件此前无人验证过的事

C1.1 的实体产物（`configs/factors/source_catalog.yaml`，487 条）早已存在，但**从未被验证过**。
本轮补两类证据：

**(1) 离线一致性**（`tests/unit/factors/test_source_catalog_conformance.py`，13 测试，全部针对**真实 YAML**而非 fixture）：
15 个必填字段在 487 条上**逐条齐备且非空**（仅查 key 存在会让 `owner: null` 蒙过，故单独一条测空值）、
factor_id 唯一且形如 `yeslab_<n>`、`source_database` 恒为 `dfs://YesLab`、
值/日期/证券列恒为 `value`/`trade_date`/`ts_code`、`source_table` 全在冻结的 11 张表内、
`_ALLOWED_TABLES` 是闭集、无任何条目在 4 个字段里含通配符 `* ? [ ]`、
TECHNICAL 因子 staleness 一律为 0 且 policy 一律 `SAME_DAY_CLOSE`（**任何非零值都等于允许 ffill**，
正是完成标准第 3 条禁止的）。
**用真实文件而非 fixture 是刻意的**：要断言的性质是"管线实际加载的那个产物合格"，
合成 catalog 可以全部满足而真实的那个不满足。

**(2) 可复现性（新，需要连库；这是离线测试结构上到不了的一条边）**：
新增 `scripts/verify_factor_catalog_reproducibility.py`，从**活的** `factor_metadata` 重跑生成器
`build_yeslab_catalog_document`，与已提交的 YAML 比对三项：snapshot hash、逐条条目、双向差集。
**2026-09-06 09:35 实跑结果：**

```
metadata rows        : 487
stored snapshot hash : afb9cc396189e47bb4963d8df65674867905606371ebec9f6998e8e2c69ce3f6
rebuilt snapshot hash: afb9cc396189e47bb4963d8df65674867905606371ebec9f6998e8e2c69ce3f6
catalog entries      : 487 stored / 487 rebuilt
only in source       : 0     only in catalog : 0     entries differing : 0
CATALOG_REPRODUCIBLE
```

即：**已提交的 catalog 是那份冻结 metadata snapshot 的逐字节展开**，没有因子被漏登记，
也没有条目被手工改离生成器的输出。离线复演（`--offline-metadata` 读回保存的 parquet）
产出与实跑**逐字节相同**的报告；快照已存
`/data/zhangyudong/qrv2_research_artifacts/factor_metadata_frozen_snapshot.parquet`。
**为什么这条边必须由脚本而不是单测覆盖**：单测只读 YAML，能证明"catalog 描述得对"，
**不能证明"catalog 描述全了"**——源里新增一个因子而从未登记，只读 YAML 的测试永远看不见。
verifier 自身有 8 条测试**证明它会失败**（源多一个因子 / 源少一个因子 / 手工改一条 owner /
只查 metadata 不碰因子表），否则 `CATALOG_REPRODUCIBLE` 毫无意义。
其中"手工改 owner"一例暴露了一个非显然的事实：**snapshot hash 覆盖的是 metadata 而非条目**，
故改条目时 hash 仍然匹配——这正是逐条比对必须与 hash 比对并存的理由。

**(3) 补上一处 C1.1 明文要求但生成器并不执行的防护。**
「不能用运行时 `GP_*` 通配符自动接纳新列」此前**只由事后的一致性测试保证**：
`build_yeslab_catalog` 遇到名为 `GP_*` 的 metadata 行会**照字面收下**，生成一条
`display_name: "GP_*"` 的条目——它不展开通配符，但会往 catalog 里放进一个**不描述任何单一因子**的名字，
而读路径是 `where factor_name in [...]`，于是它要么选不到东西、要么选中某个无人定义过的字面量。
已在生成器里加拒绝（`CatalogBuildError`），并测 `GP_*`/`Alpha?`/`Alpha[1-9]`/`*` 四种形态。
**加防护后重跑可复现性检查：hash 与 487 条条目完全不变**（活 metadata 里没有通配符名），
即这次收紧只影响**下一次**重新生成，不影响已冻结的产物。

**留档的边界**：C1.1 还要求「在**人审**后保存 snapshot hash」。人审是流程步骤，代码无从验证；
**我能验证的是它的产物**——所记录的 hash 可从源重算出来且一致，故按"人审通过并记录了 hash"接受。

### C1.4 仍不勾：状态机部分已补测，但关键一款**根本没有实现**

**已实现的部分此前几乎没有测试。**`FactorRegistry.transition` 的四状态机
（DRAFT→VALIDATED→ACTIVE→RETIRED）在 `test_registry.py` 里**只被一条 happy path 顺带走过**，
没有任何一条测试断言非法迁移被拒。本轮补 20 条：迁移**只能向前**（11 个非法组合逐一参数化，
含"迁移到当前状态"——那不是 no-op，而是调用方以为自己身在别处）、
**被拒的迁移必须让条目留在原状态**、RETIRED 是**终态**（四个状态全部拒绝）、
DRAFT/VALIDATED/ACTIVE 都可直接退役（逃生门不必走完生命周期）、
新注册一律从 DRAFT 起、未注册因子迁移报 KeyError、
**新版本与旧版本并存且各有自己的生命周期**（C1.4 规定的补救方式是新版本而非改旧的）。
辅助函数 `_registry_at` **自带一条断言**确认自己真的走到了目标状态——
因为它第一版写错了（要 DRAFT 却走到了 ACTIVE），**实现是对的、测试辅助是错的**，
这类错误会让"非法迁移被拒"的测试以错误的理由通过。

**另一处与规格的偏差已钉住并留档**：规格说「定义字段在 **VALIDATED 后**不可修改」，
实现是**从 DRAFT 起**就拒绝改 hash（更严）。这不是缺陷，但两种语义下"草稿能否就地编辑"相反，
而 definition hash 是早期引用它的东西的键，故写成测试固定实际行为。

**未实现的一款：「只有 `ACTIVE` 因子可加入 factor set」。**
全仓 grep `FactorStatus.ACTIVE`：**只出现在 `registry.py` 自己的迁移表里**（第 53、54 行），
没有任何消费方。`create_factor_set` 收的是裸元组
`(factor_id, factor_version, direction, definition_hash)`，**从头到尾不接触任何 registry**——
尽管它的 docstring 写的是 "Build a deterministic DRAFT factor set from **active-registry** member identities"。
**docstring 声称了一个代码并不执行的约束**，这类偏差比缺失更危险，因为它会让读者以为门已经在了。

**而且即使实现了，今天也拦不住任何东西。**这是本节更重要的发现：

- **`FactorRegistry` 的生产调用者为 0**（`research_v2/src` 与 `research_v2/scripts` 全仓 grep：
  `FactorRegistry()` 与 `.transition(` 零命中，只有 `registry.py` 自身与单测）。
- **`create_factor_set` 的生产调用者也为 0**（24 处命中全在 `tests/`）。
- 生产中真实的冻结集是 **`artifacts/diagnostics/factor_set_frozen_v1.json`**，
  由 **`scripts/apply_admission_decisions.py`** 直接写成一个 JSON dict
  （`scope/frozen_date/factors/count/special_decisions/conflicts/warnings/data_version`），
  被 `materialize_frozen_factor_daily.py`、`verify_admission_consistency.py`、
  `walk_forward_factor_daily.py` 消费。该脚本里 grep `ACTIVE|registry|FactorStatus`：**零命中**。
  它的成员资格来自**人工裁决 D1–D4** 加一个 `data_version`，不是来自 registry 状态，
  `factor_set_id` 也不是 `create_factor_set` 的内容寻址哈希。

**结论：C 计划的 registry + factor-set 机制（C1.2/C1.3/C1.4/C8.1/C8.2）与生产中真实使用的冻结集
是两套互不相通的实现，前者未接线。**这是本轮发现的**第三处**同一模式
（续四：C6.2 结构质量有两套并行实现、本模块未被接线；
续七：`select_cluster_representatives` 无生产消费者）。
**三处同型已构成一个模式，建议作为整体上报 owner**，因为它决定了 C 计划里
有多少治理是**真正承重**的、多少是**写了但没接上**的——这不是逐条补测试能解决的问题。

**不代实现的理由**：把 `registry` 变成 `create_factor_set` 的**必需**参数会破坏 20+ 个调用点
以及已勾选的 C8.1/C8.2 所依赖的接口；做成**可选**参数则等于没有强制。
而"以哪条路径为准"正是**裁决 4** 已经在问的同一个问题
（`factor_set_frozen_v1.json` 是已结案契约条款，应新发 V2 而非就地改）。
**故记录、不实现。**

### C1.6 无法执行：它要求的集成测试文件从未被创建

C1.6 的第二条 Run 命令是
`QRV2_INTEGRATION=1 pytest tests/integration/factors/test_factor_registry_import.py -q -m integration`，
但 **`research_v2/tests/integration/factors/` 目录不存在**
（现有 integration 子目录：`data/datasets/db/evaluation/inference/models/pit/registry`）。
计划文件清单里它标为 "Create"，即**从未创建**。
其 Expected 三项中，「全部因子有显式映射」由 C1.1 本轮覆盖、
「定义漂移要求升版」由 registry 单测覆盖、「V1 无写入」属只读会话性质；
但在 registry 未接线的前提下，一个"registry import"集成测试**没有可断言的实质内容**。
**不代为编造**：先解决上面的接线归属，再决定这个集成测试该测什么。

### 变异验证（本节 7 个，全部被预期测试捕获）

| # | 变异 | 捕获它的测试 |
|---|---|---|
| 11 | 生成器去掉通配符拒绝 | 通配符两条（`GP_*` 与四种形态） |
| 12 | `compare()` 忽略逐条差异（只比 hash） | 源多因子 / 手改 owner / 重排序 三条 |
| 13 | verifier 的查询指向 `factor_data1` | `..._selects_metadata_columns_and_no_factor_values` |
| 14 | DRAFT 允许直接跳到 ACTIVE | `..._only_moves_forward[DRAFT-ACTIVE]` |
| 15 | RETIRED 不再是终态 | `..._only_moves_forward[RETIRED-ACTIVE]` 与 `test_retired_is_terminal` |
| 16 | 改定义时静默替换旧条目 | 4 条（含既有的 `test_definition_change_requires_a_new_factor_version`） |
| 17 | `transition` 完全不校验 | 12 条 |

**全量：1422 passed / 45 skipped**（本节起点 1381；+13 一致性 +8 verifier +20 registry），
`ruff check src tests` 清、`mypy src` 157 文件清。

## 勾选审计续十一（2026-09-06 10:10）：C3.2 已勾；C3.1 有两处实现与规格不符（均不改，理由具体）；C3.4 缺输入；另查出一个计划外的证据格式缺陷

### C3.2 已勾选——但其中一半的缺口是"两个已发布的数字从未被断言过"

`deduplicate_factor_observations` 是纯函数、返回 `DedupResult(clean, quarantine, statistics)`，
quarantine 保留**全部原始候选行**（含 `source_row_id`、`reason`、全精度 `raw_value`，
±inf 也原样保留），冲突组**不选 first/last**。这些此前已有测试。

**真正的缺口在 `statistics`**：`DedupStatistics` 四个计数里，
**`invalid_groups` 与 `single_nonnull_groups` 在整个测试套件中从未被断言过任何一次**
（`identical_groups`/`conflict_groups` 也只在两个窄场景里各断言过一次）。
而这四个数**不是内部记账**：`conflict_groups` 是持久化质量快照里的**存储列**
（`repair_persistence`、`migrations` 都引用它），四个数合起来是
"源里有多少被折叠、多少被保留、多少被隔离、多少不可用"的**唯一汇总**。
**任一计数错算，读起来就是"源里的冲突比实际少"**——正是这类数字最不该出错的方向。

新增 `test_statistics_count_groups_not_rows_and_cover_all_four_classes`：
一个 fixture 同时含**四类各一组**（identical / single-nonnull / conflict / 全空重复组）
外加**一行唯一有限值**与**一行唯一非有限值**，期望值**独立手算**后写入断言：
`identical=1, single_nonnull=1, conflict=1, invalid=2`，clean 3 行、quarantine 5 行。
**`invalid_groups == 2` 而非 1 是这个 fixture 的关键**：它由**两条不同的代码路径**相加
（唯一非有限行走 `len(invalid_unique)`，全空重复组走 `reason == "NO_FINITE_VALUE"` 计数），
只有同时包含两者的 fixture 才钉得住这个和；单独任一类都会让"漏加一项"的变异存活。
另补 `test_a_conflict_never_leaves_a_partial_group_behind_in_clean`（冲突键贡献 **0** 行 clean，
三行候选组不得留下"平局幸存者"）。

**分层说明（避免误读 C3.2 的"distinct values"）**：C3.2 约束的是**这个纯函数**的 quarantine，
它保留逐行候选，故四要素齐备。发布层（materializer，属 C5）再把它聚合成每组一行
`source_row_ids_json` + `distinct_values_json` + `reason`——那是 C5 的输出形状，不是 C3.2 的。

### C3.1 仍不勾：两处实现与规格不符，**都不改**，理由不是"麻烦"而是"会改动物化数据"

**前置事实：`deduplicate_factor_observations` 在活的物化路径上**——
`materializer.py:98` 直接调用它。因此任何收紧判定的改动都会**重新分类真实行**、
改变 `factor_observation_clean` 的内容。而 Task 2.4（遗留链 criterion-303 逐位复现）**仍在跑**
（本节写作时已 12h58m），红线明确：改变因子数据会推进版本指针、使遗留折缓存失效、破坏逐位复现。
**故本节只钉行为、只记录偏差，不改实现。**

**(a) IDENTICAL_COLLAPSE 只比 `raw_value`，不比"所有非审计字段"。**
规格：「所有非审计字段完全一致，只保留一行」。实现：`distinct_finite <= 1` 即折叠。
后果是具体的，不是理论上的——两行业务键相同、`raw_value` 相同、但
**`source_available_date` 不同**时会被折叠，而**谁存活由最小的 `source_row_id` 决定**，
那是一个**溯源字段、与语义无关**。也就是说：**可用日（决定该值何时可被下游使用的 PIT 字段）
是被溯源排序"顺手挑中"的**，不是按任何关于可用性的规则挑的。
已写 `test_identical_collapse_compares_the_value_only_not_every_non_audit_field` 钉住现状，
docstring 明写这是与 C3.1 的偏差、以及为什么现在不能改。

**(b) SINGLE_NONNULL 不要求"恰有一个有限非空值"。**
规格：「恰有一个有限非空值，其余为空」。实现的判据是"非冲突且并非全部有限"，
故 **[1.0, 1.0, NaN] 也被标为 SINGLE_NONNULL**（有两个相等的非空值）。
**发布值不受影响**（两个有限值相等；若不等就是冲突），故这是**标签不精确**而非数字错误——
但该标签正是复核者用来理解"这行为什么活下来"的依据，且会被持久化。
已写 `test_two_equal_finite_values_and_a_null_are_labelled_single_nonnull` 钉住。

**(c)「源如果声明量化精度，则先按该精度规范化再比较」——对本源为空条款，已核实。**
这是一个**前件为假**的条件句：`FactorDefinition` 无任何精度/量化字段（17 个字段逐一核对）、
`SourceBatch` 无（8 个字段）、`source_catalog.yaml` 中 `precision|quantiz` **命中 0 次**、
`unit` 全部为 `UNKNOWN`。故本源**未声明量化精度**，该款的规范化步骤无从触发。
**与续六对"宽表"的处理同一形状**：不是跳过，而是核实前件为假并留档。
若日后接入声明精度的源，该款需重新核验（届时"先规范化再比较"会与
「不用宽松 isclose 把真实冲突吞掉」直接张力，需要明确规范化只用于**折叠判定**而不用于**发布值**）。

**已满足并有测试的部分**（连同上面三点即 C3.1 的全貌）：业务键四元组正确、
三类决策的主干行为正确、**浮点相等用精确二进制而非 isclose**
（既有测试钉住"相差 1 ULP 即冲突"与"落在 np.isclose 默认容差内仍是冲突"，
后者特意用 numpy 而非 math.isclose，因为要向量化实现最容易顺手拿的正是 `np.isclose`）、
**发布值逐位保真**（用 `float.hex()` 比较 `0.1+0.2`）。

### 计划外发现：quarantine 的证据列会写出**非法 JSON**（已钉住，未修）

写 (a)(b) 的测试时顺着发布层查出一处规格没提到的缺陷，**经公开 API 实测可达**：

- C2.3 要求「±inf 转为 invalid flag 而非有效 NaN」，故 `raw_value` **确实可以是无穷**；
- 某键的观测全为非有限值 → 该组以 `NO_FINITE_VALUE` 进 quarantine；
- materializer 聚合证据时用 `json.dumps`，过滤条件是 `pd.notna` ——
  **`pd.notna(inf)` 为 True**，故无穷被留下，输出 **`distinct_values_json == "[Infinity]"`**；
- `Infinity` 是 Python 的扩展，**RFC 8259 不允许**；严格解析器（数据库 JSON 函数、
  JS `JSON.parse`、serde_json）**直接拒绝**。

**代价落在最痛的地方**：quarantine 就是"不可发布观测"的审计轨迹，
而**恰好是描述源里最坏数据的那条记录无法被标准工具读取**。
同一次聚合还**丢掉了 reader 的 `invalid_reason`**，故发布行**无法区分
"源说了无穷"与"源什么都没说"**——而这个区分正是 C2.3 特意建立的。
**clean 批次全程不受影响**，这纯属证据格式问题。

已写 `test_a_quarantined_infinity_is_published_as_non_standard_json` 钉住现状
（断言输出串为 `"[Infinity]"`、断言严格解析器抛错、断言 `invalid_reason` 不在发布列中），
并变异验证：把过滤改成 `pd.notna(v) and np.isfinite(v)` → 该测试失败（M22）。
**不修的理由**：如何表示无穷是**格式决定**（字符串哨兵 / null 加标志位 / 把 `invalid_reason` 带下去），
三者都会改变**已持久化列的内容**从而改变 `record_hash`，而 migrations 已定义该列。
**记录在案，留给 owner 连同 C5 一并裁定。**

### C3.4 缺输入：所谓「Phase 0 审计中的 GP_017/GP_077 重复键样例」在仓库里不存在

已把能查的都查了：全仓 grep `GP_017|GP_077` 命中 13 个文件，逐个看过性质——
`gp_candidate_screen_*`（候选筛查的 IC/覆盖率统计）、`explicit_admission_review_checklist_v1.json`（准入清单）、
`factor_catalog_final_status_v1.json`、`band10_*`/`alpha_386_387_*`（训练质量）等，
**没有一个是"重复业务键"的审计**。唯一以 GP_077 为主角的诊断
`gp_turnover_scalar_degeneracy_diagnostic_v1.json` 讲的是**标量退化**
（`turnover × 每日横截面标量` → 与 turnover_rate 完全冗余，yeslab_436/446/468 三者 RankIC 逐日完全相同），
**那是 C7 聚类的问题域，不是 C3 去重的问题域**。
另 grep `duplicate_key|dup_key|conflict_key|重复键` 于 artifacts/docs：命中的都是
compact universe coverage 与 repair manifest，与重复键统计无关。

**顺带确立了一个有用的映射**（后续 C3.4/C7 都用得上）：
**GP_017 = `yeslab_386`，GP_077 = `yeslab_446`**
（出自 `gp_candidate_screen_2020_2024/summary.csv` 的 `factor_name,factor_id` 两列）。
注意 `yeslab_386` 与待裁决的 `yeslab_387`@2017-10-09 是相邻的一对，
artifacts 里已有 `alpha_386_387_train_2020_quality_v1.json` 与 `manual_alpha_386_387_train_2020/`。

**两条前进路径，都需要 owner 或需要等待**：
(i) 在 V1 树（`数据库/daily_task_new/`，他人维护）里找到那份 Phase 0 审计原件；
(ii) 对 `yeslab_386`/`yeslab_446` 跑一次**有界的只读重复键普查**，据实构造匿名化 fixture。
**(ii) 本节刻意不做**：它要对 `factor_data*` 发聚合查询，而 Task 2.4 正在跑第 13 个小时、
还剩约 1.5 小时，且它是**裁决 3 与裁决 4 的共同前置**。在关键路径收尾阶段给源库加负载，
换取一个可以晚两小时做的 fixture，**不划算**。**等 2.4 落地后再跑。**
**不代造 fixture**：C3.4 要的是"从真实审计中抽取的样例"，凭空编一个"匿名化 GP_017 重复键"
会让回归测试**看起来有真实出处而没有**，那比缺这条测试更糟。

**C3.5 因此也不勾**：它的 Expected 是"三类 resolution 和全部不变量通过"，
而 C3.1 的两处偏差与 C3.4 的缺失都还开着；`C3.5` 是 Task 级的收口项，不应先于其子项勾选。

### 变异验证（本节 5 个，全部被预期测试捕获）

| # | 变异 | 捕获它的测试 |
|---|---|---|
| 18 | `invalid_groups` 漏掉唯一非有限行 | 四类统计测试 |
| 19 | 计数改成按行而非按组 | 四类统计测试 + 两相等值加空值那条 |
| 20 | 代表行改按帧内顺序而非 `source_row_id` | IDENTICAL_COLLAPSE 偏差钉住测试 |
| 21 | 两个 resolution 标签互换 | 4 条 |
| 22 | 发布证据里滤掉无穷 | 非法 JSON 钉住测试 |

**全量：1427 passed / 45 skipped**（本节起点 1422；+4 dedup +1 materializer），
`ruff check src tests` 清、`mypy src` 157 文件清；
`deduplication.py` 与 `materializer.py` 变异后均以 `git status --porcelain` 确认**还原干净**。

## 勾选审计续十二（2026-09-06 10:30）：C4.1 已勾；C4.2/C4.3/C4.4 各有具体缺口，其中 C4.4 查出一处会写错报告期版本的缺陷，并新增一条裁决

### C4.1 已勾选——但我第一版"不 ffill"测试又一次是恒真的

C4.1 三款：`SAME_DAY_CLOSE` 下 `observation_date = available_date = signal_date`（既有测试已覆盖）、
**缺值保持缺值不从前一日 ffill**（本轮补）、同日收盘后才可用于当日收盘信号语义（即上款）。

新增 `test_a_missing_technical_day_is_left_missing_rather_than_filled_from_the_previous_one`。
**第一版只钉住了"不凭空造行"**：断言输出行数等于输入行数、缺的那天不在 `signal_date` 里。
变异验证时发现它**抓不到值层面的 ffill**——插入
`result.groupby("ts_code")["raw_value"].ffill()` 后**所有测试仍全绿**，
因为 ffill 不改变行数，只把一个已存在的空值换成前一日的值。
**「不 ffill」有两个读法，只测其一的测试在另一读法退化时照样通过**：
(a) 某天**根本没有观测**时不得凭空多出一行；
(b) 某天**有行但值缺失**时该值必须保持缺失，不得继承前一日。
(b) 才是 `groupby.ffill()` 会静默做错的那个，而它把行数留得完完整整。
已把 fixture 扩成两只证券同时覆盖两个读法（000001.SZ 缺 03-31 整天；000002.SZ 的 03-31 值为 NaN），
重跑变异 M24 → **该测试失败**。
**这是本轮第三次同一类教训**（续八：非有限值测试在 inf 计入与否下结论相同；
续十：`_registry_at` 要 DRAFT 却走到 ACTIVE；本节：只测行数不测值）。
统一表述：**一条测试是否有效，判据是"被破坏的实现下它是否给出相反结论"，
而这必须对每一类可能的破坏分别验证——覆盖了"造行"不等于覆盖了"改值"。**

### C4.2 仍不勾：两款未满足

已满足并有测试：下一交易日可用（`assign_fundamental_available_date`）、
周末/非交易日公告顺延到下一个开市日、交易日公告顺延一天、修订公告、同报告期多版本、
180 日边界（本轮把**比较符本身**钉住了，见 C4.3）。

**(a)「无 `ann_date` 的基本面记录隔离，不能用报告期代替公告日」——后半成立，前半不成立。**
实测：`ann_date = pd.NaT` 时抛
**`TypeError: Cannot compare NaT with datetime.date object`**。
**后半款确实成立**：代码任何地方都没有拿 `report_period` 顶替 `ann_date`，故**没有前视**。
**前半款不成立**：规格要的是"**隔离**"，而现状是**整个展开被一个不透明的类型错误打断**，
错误信息指向 pandas 的一次比较、而不是"这条记录缺公告日"。
**这与本模块自己声明的设计意图直接矛盾**：`_empty_expansion` 的 docstring 明写
"one empty result must not abort the whole materialization"，
并且列举了空展开是**正常结果**的三种情形。缺公告日不是正常结果，但**它也不该炸掉整批**。
**不代实现"隔离"**：该函数返回**单个 DataFrame，没有第二条通道也没有 reason 列**，
要隔离就必须改它的返回契约；而**它现在没有任何生产调用者**（见下），
**改契约的最佳时机正是接线之前**，不是由我替 owner 选一个返回形状。

**(b)「退市」——无任何测试覆盖，模块里也没有出现 `delist` 字样。**
机制上是**隐式**成立的：`expand_fundamental_events` 只对 `universe` 里真实存在的
(signal_date, ts_code) 产出行，故退市证券自然不再有 universe 行、也就不再有因子行。
**这个设计是对的**（成员资格归 universe 所有，可用性层不该自己再判一次），
但 C4.2 把"退市"明列为要覆盖的案例，而**它没有被测**：
没有任何测试断言"某证券在 universe 中消失后其展开也随之终止"，
也没有测试断言"退市前已公布的基本面值在退市后不会继续被展开"。
**隐式成立且未测 ≈ 无人看守**：一旦 universe 的构造改了（例如为覆盖率而延长成员区间），
这条性质会静默失效，而 180 日 staleness 会让退市证券继续"持有"旧值最多半年。

### C4.3 仍不勾：输出语义正确，但**实测成本使它不可接线**

语义三款全部成立并已测：**只对实际 universe 成员日期展开**（逐 universe 行迭代）、
**输出 `stale_days`**、**第 181 个自然日值为空**。
本轮把**比较符**单独钉住了：既有测试用的是 stale=179（保留）与 stale=182（丢弃），
**中间留了空档**——写成 `>=` 再配一个差一错误可以**同时通过这两条**。
新测试固定 stale=179，只变动上界：`max=179` 保留、`max=178` 丢弃，
从而把 `stale_days > max`（严格大于）孤立出来。变异 M23（改成 `>=`）→ 该测试失败。

**但「而非全笛卡尔积」这一款，实现只做到了一半。**
**输出**确实不是笛卡尔积：行数只为实际 universe 成员产出。
**时间**上却是：实现是**逐 universe 行的 Python 循环**，每行都对**整个 events 帧**做一次布尔掩码，
即 `O(universe_rows × events)`。实测（离线合成，`TradingCalendar` 400 个工作日）：

```
 universe  events   seconds    us/universe-row  rows out
      500     100      0.83             1664.4       190
     1000     200      1.29             1286.0       190
     2000     400      2.74             1371.1       380
     4000     400      6.71             1677.9      1350
     8000     800     13.99             1748.4      2339
```

按 ~1676 µs/universe-row 外推：
**单因子单折**（5000 证券 × 250 信号日 = 1,250,000 行）≈ **0.6 小时**；
**487 个 catalog 因子单折** ≈ **12 因子·天**；**6 折** ≈ **71 天**。
每行成本在 universe 规模 16 倍区间内**持平到上升**，故不是预热假象。
证据存于 `/data/zhangyudong/qrv2_research_artifacts/fundamental_expansion_scaling.txt`。

**这一款的存在理由就是避免爆炸，而爆炸还在，只是从内存挪到了时间。**
12 因子·天/折意味着**照现状接线是不可行的**，故不勾。

**修法与 C4.4 的未决问题耦合，这是不现在就改的原因。**
正解是真正的 as-of/区间连接（按证券把事件版本按 `available_date` 排序，
用 `merge_asof` 或 `searchsorted`，`O((N+M) log M)`），
**但它必须逐位复现现在的 tie-break 顺序**——而那个顺序正是下面 C4.4 里有争议的东西。
**先优化等于把一个有争议的规则固化进一条更快的路径**，日后改语义要改两处。
**故：先定语义，再重写。**

### C4.4 仍不勾：三条子款各自成立，但同一条选择规则里有一处会写错报告期版本的缺陷

**三条子款逐一核实为成立且有测试**：
同一 report period 的后续修订**只从其新 available date 起生效**（修订版 available 更晚，
故此前用原版、此后用修订版）；**过去 signal dates 继续看到当时已公布的版本**
（`tests/leakage/` 四条未来扰动测试覆盖）；**`source_revision_id` 与 `report_period` 都被记录**
（二者都在 `EXPANSION_COLUMNS` 里，逐行输出）。

**但实测确认了一处缺陷：一份对较旧报告期的迟到重述，会顶掉较新报告期的值。**
构造：FY2024（report_period 2024-12-31）于 2025-03-28 公告（available 03-31）；
随后 2025-04-10 发行人**重述更旧的 Q3 2024**（report_period 2024-09-30，available 04-11）。
在 signal date 2025-04-15：

```
signal_date report_period  raw_value  available_date  source_revision_id  stale_days
 2025-04-15    2024-09-30        1.5      2025-04-11    q3-2024-restated           4
```

**发布的是被重述的 Q3（1.5），FY2024（2.0）被丢弃**——因为版本排序是
`["available_date", "report_period", "source_revision_id"]` 全降序，
**`available_date` 是第一键**，而重述恰好是"最近才可用"的那个。
在 2025-04-01（重述可用之前）发布的是 FY2024，故**报告期在跨越重述时向后退了一格**
（新测试对此有显式断言）。
**按通常的 PIT 基本面语义这是错的**：某证券某日的值应来自**当时可得的最新报告期**，
修订只在**同一报告期内部**决胜。因为一份旧报告被迟到重述，就把整只证券的
**报告年份倒回上一季**，是静默改写基本面版本。

**规格在此确实两可，所以这是裁决而不是修复**：
C4.3 把有效区间定义为到 `next_version_available_date - 1 trading day`，
**"单一版本序列、每个新版本终结上一个"**——那正是代码实现的东西；
C4.4 只讲**同一 report period 内部**的修订。**两者都没说跨报告期时谁压谁。**

**顺带查出一条"测试名过度承诺"的问题，值得单独记。**
既有测试 `test_fundamental_event_prefers_latest_report_period_before_revision`
的名字断言的是"**报告期优先于修订**"这一 precedence，
但它的 fixture 给两个版本**同一个 `ann_date`**，于是 `available_date` 相同、
排序**落到第二键 `report_period`** 才得到正确结果。
**即：名字读起来像一条关于"报告期优先"的保证，实际只是一条关于"平局如何打破"的保证。**
（该测试本身是好的——它还刻意把 revision id 命名为 `z-earlier-period` / `a-later-period`，
使"按 revision id 排序"的实现会挑错，这一点很扎实。）
**教训与续八/续十/本节那条同源**：一条测试的**名字所声称的范围**可能大于**它的 fixture 所钉住的范围**，
而读者只会读名字。

**不勾 C4.4 的理由**：该款的标题是「实现修订版本语义」，
而修订语义的选择规则在一个**确实会发生**的场景下给出错误的报告版本。
三条子款成立不足以支撑标题级的勾选——这与续六对"宽表"的处理不同：
那里是**前件在本源上不出现**（真空成立），这里是**场景会出现且结果是错的**。

### 第四处"未接线"：`expand_fundamental_events` 无生产调用者

全仓 grep：`materialize_technical_availability` 被 `materializer.py:101` 调用，
而 **`expand_fundamental_events` 只出现在自身定义与测试里**。
**这是本轮发现的第四处同型**（续四：C6.2 两套并行实现；续七：`select_cluster_representatives`；
续十：`FactorRegistry` 与 `create_factor_set`）。
**好消息是它降低了本节两个缺陷的当下风险**（没有基本面因子正在被错误版本物化），
**坏消息是它意味着 C4 的基本面半边整体处于"写了但没接上"的状态**，
而 12 因子·天/折的成本问题会在接线那一刻才暴露。

### 新增裁决 5：跨报告期的基本面版本优先级

**问题**：当同一证券同时存在"较新报告期的原版"与"较旧报告期的迟到修订"时，
某 signal date 应发布哪一个？

- **选项 (i) 最新报告期优先，修订仅在同一报告期内决胜**（通常的 PIT 语义；
  排序键改为 `report_period DESC, available_date DESC, source_revision_id DESC`）。
  代价：C4.3 的区间公式 `next_version_available_date` 需要重述为**按报告期分组**的区间，
  因为不同报告期的有效区间会**重叠**而不再是一条线性序列。
- **选项 (ii) 最新可用版本优先**（即现状；与 C4.3 的字面公式一致）。
  代价：如上，报告版本会因一次旧期重述而倒退。
- **选项 (iii) 两者都不选，改为并列输出并由下游按因子语义决定**。
  代价：展开不再是每 (signal_date, ts_code) 一行，会波及 C5 的统一 observation 输出契约。

**与 C4.3 的性能重写绑定**：无论选哪个，as-of 连接的排序键都必须与之一致，
**故应先裁决再重写**，否则要改两遍。
**与裁决 4 无关**（那是冻结因子集的覆盖率问题），可独立决定。

### 变异验证（本节 3 个，全部被预期测试捕获）

| # | 变异 | 捕获它的测试 |
|---|---|---|
| 23 | staleness 上界改成 `>=` | 上界比较符测试 |
| 24 | 技术因子按证券 `ffill` 值 | 强化后的不-ffill 测试（**强化前抓不到**） |
| 25 | 版本排序把 `report_period` 提到第一键 | 迟到重述钉住测试 **+** 既有的未来修订不改写历史测试 |

M25 同时被一条**既有** leakage 测试捕获，说明那条测试对排序键是敏感的——
这也反过来印证：现状的排序键并非无人看守，只是**看守的是"历史不被改写"，
而不是"跨期谁压谁"**，后者是新增测试才钉住的。

**全量：1431 passed / 45 skipped**（本节起点 1427；+4 availability），
`ruff check src tests` 清、`mypy src` 157 文件清；
`availability.py` 三次变异后均以 `git status --porcelain` 确认**还原干净**。

## 勾选审计续十三（2026-09-06 11:10）：C5.2 已勾并闭上一条无人看守的边；C5.1/C5.3/C5.4/C5.5 的缺口逐一定位（其中一处比预想的好）

### C5.2 已勾选——四款逐一核实，其中一款的机制在另一个模块里

**(1) 16 个最小字段：与规格逐字一致，且顺序一致。**
`build_technical_factor_batches` 产出的 `factor_observation_clean` 帧列**恰好**是
C5.2 列出的 16 个字段、**且顺序相同**（11 个业务字段 + 5 个 lineage 字段）。

**(2) 顺带闭上一条此前无人看守的边：产出列 ↔ migration 声明列。**
`migrations/003_factor_label_tables.dos` 用反引号列表声明
`factor_observation_clean` 的 16 列与类型，materializer 用另一处字面量按名装配同一批列，
**两者之间没有任何共享真值来源**。而现有的两类检查都碰不到这条边：
`tests/unit/db/test_migrations.py` 只断言**表名**出现在脚本文本里
（`assert f"\`{table_name}" in factor_migration.script`），
黄金样本只比较 **6 列子集**（`CLEAN_COLUMNS`）。
**故任何一侧增列、删列、改名或换序，两道检查都会同时放行。**
新增测试从 migration 脚本里**解析**出列名与类型，与 materializer 的实际输出逐位比对，
`factor_quarantine`（12 列，在另一处字面量里装配）同样比对。
**顺序也要断言，不只是集合**：若写入是按位置的，换序会把一列的值发布到另一列名下——
那是**静默损坏**而不是报错。变异 M26（交换 `observation_date`/`available_date` 两列顺序）
与 M27（删掉 `stale_days`）**都被这两条测试捕获**。

**(3)「技术因子的 `report_period/source_revision_id` 可空」——一个真空、一个比要求更强。**
`report_period` 发布为 `NaT`：日频技术观测**不属于任何报告期**，
NaT 是诚实的值，填一个日期会**暗示一个并不存在的会计期间**。
`source_revision_id` 则被填成 source row id（规格只要求"可空"）——
**这比要求更强，且值得钉住**，因为它是**从一条已发布行回到它所来自的那一条源观测的唯一链接**。
已断言其非空且逐行唯一。

**(4)「值为空的行默认不写事实表」——成立，且不是静默丢弃。**
空值行不进 clean，而是进 quarantine 并带 `reason = NO_FINITE_VALUE`。
**这一点必须与"丢弃"区分开**：事实表里留一个 null `value` 会迫使每个下游消费者
各自重新决定它是什么意思；而带理由进隔离区意味着缺失**被记录**了。

**(5)「但缺失在质量分母中保留」——成立，机制在 C6 的模块里，已定位到行。**
`quality.py:142`：`coverage = finite_value_count / universe_member_count`——
**分母是 universe 成员数，不是有限值个数**，故 universe 里一个没有值的成员**会拉低覆盖率**。
该模块自己的注释把这对指标区分得很清楚（`quality.py:185`）：
"finite ratio describes what the source delivered, coverage what the universe could [have delivered]"。
**即 C5.2 的这一款由两处共同discharge**：materializer 把缺失送进 quarantine（信息不丢），
quality 用 universe 作分母（缺失计入代价）。**二者缺一，"缺失被保留"都只是半句话。**

### C5.1 仍不勾：十步固定顺序里，**三步不在物化链上**——但其中一步的位置比预想的好

规格顺序：registry validate → source read → normalize → deduplicate → availability →
universe filter → structural DQ → stage → validate → publish。

**已接线的部分（实测有生产调用者）**：normalize → deduplicate → availability → stage → publish。
`build_technical_factor_batches` 被 `source_backfill.py:536`、`source_evaluation.py:235`
及 4 个脚本调用，**这是本轮核验的所有模块里接线最实的一个**（对照续十/续十二的四处未接线）。

**不在链上的三步，逐一说明**：

- **registry validate**：registry 整体未接线（续十、决定 A）。**故这一步不是"漏了"，
  而是它依赖的那套东西还没被采用。**在决定 A 之前无法实现。
- **structural DQ**：C6 的结构质量模块未被接线（续四已记两套并行实现），
  生产中用的是 `scripts/screen_factor_daily_coverage.py` 自带的一套阈值。
  **同样是决定 A 类的归属问题，不是补测试能解决的。**
- **universe filter**：**这一步实际上是有的，只是在下游、不在物化层**——
  `datasets/builder.py:164` 的 docstring 明写 "Build a stable feature frame **anchored only on
  the eligible daily universe**"，即特征帧以合格 universe 为锚。
  **这个位置是合理的，甚至更好**：事实表存"源里观测到什么"，数据集选"合格 universe 里要什么"，
  于是**改 universe 定义不需要重新物化因子**。
  **故 C5.1 在这一款上不是缺失，而是步骤被放在了规格的顺序之外**——
  但这仍意味着**规格所写的那条"任一步失败均不产生正式可见行"的单一 pipeline 并不存在**：
  物化与 universe 过滤是两个独立阶段，中间隔着一次发布。
  **不勾的理由是这个，不是 universe filter 缺失。**

**另需说明「任一步失败均不产生正式可见行」**：stage → validate → publish 的指针式发布
确有实现（`db/publish.py` 与指针回滚集成测试），故"失败不产生正式可见行"这一**性质**
在已接线的阶段上是成立的；但它**不覆盖**上面两步未接线的阶段，
因为那两步根本不在同一条 pipeline 里。**待决定 A 之后才能整体验证。**

### C5.3 仍不勾：四条命令一条都不存在，但 dry-run 的形状已有先例可循

**`research_v2/src/research_v2/cli/factors.py` 不存在**（`cli/` 下为
`app/baselines/datasets/db/evaluate/export/inference/materialize/models/registry/reports`）。
规格的四条命令逐条核实：
`qr-v2 factor registry validate` / `factor registry import` / `materialize factors` /
`factor quarantine report` —— **全部不存在**（`cli/` 下 grep `registry validate|registry import|quarantine`
只命中 `registry/model_registry.py` 的模型注册表，与因子 registry 无关）。

**部分可复用的先例**：`cli/materialize.py::emit_plan` 是一个**fail-closed 的计划发射器**——
校验日期、输出全限定的源表与目标表、`dry_run=False` 时**直接拒绝执行**
（"runtime backend is not configured; refusing to materialize"，退出码 2）。
**这正是 C5.3 要的形状**，但 `_PLANS` 只覆盖 `security-master / market / universe / labels`
——**没有 factors 这一项**；且它输出的是 operation/日期/源表/目标表，
**不含**规格要求的**因子数、源列、日期 chunk、估算行数、目标分区**。
**本轮 C1.5 的 dry-run 已独立提供了其中的"源表/列"部分**
（并证明了自己只发 schema 查询、不读值、不写库），故 C5.3 落地时可直接复用其输出，
不必另写一套 schema 探测。

### C5.4 仍不勾：增量影响范围完全没有实现

`research_v2/src/research_v2/factors/` 下 grep `incremental|affected_signal_dates|impact_scope`
**零命中**。规格三款——技术日频源变化只重算变化的 signal dates、
基本面公告/修订变化重算该证券从 available date 到下次版本或 180 日上限与 universe 的交集、
registry 新版本不覆盖旧版本——**前两款无任何实现**。
第三款（新版本不覆盖旧版本）在 registry 层成立（续十：新旧版本并存、各有生命周期），
但 registry 未接线，故对物化无实际约束力。
**与决定 B/C 相关**：基本面那一半的重算范围定义**正是**"到下次版本或 180 日上限"，
而"下次版本"的语义就是决定 B 争议的跨期优先级。**故 C5.4 的基本面半部应等 B 定后再实现。**

### C5.5 仍不勾：`tests/integration/factors/` 目录不存在

与续十对 C1.6 的发现同因：`tests/integration/factors/` **从未被创建**
（现有 integration 子目录：`data/datasets/db/evaluation/inference/models/pit/registry`），
故 C5.5 要覆盖的六种故障（源 schema 漂移、源水位中途变化、重复冲突、
`available > signal`、publish 前进程异常、相同版本重跑）**没有集成测试载体**。
其中**源 schema 漂移**本轮已在单元层由 C1.5 的 dry-run 覆盖（表/列缺失、类型不可转换、
表不可读三种），**重复冲突**由 C3 的去重测试覆盖，**`available > signal`** 由
`tests/property/factors/test_availability_invariants.py` 的 PIT 排序不变量覆盖——
**但"源水位中途变化""publish 前进程异常""相同版本重跑（幂等）"三项确无任何覆盖**，
而 C5.6 的 Expected 恰恰写的是"正常发布幂等；所有 blocker 均保持旧版本可见"。
**这三项是 C5.5 的实质内容，需要真实或仿真的发布会话，属集成测试范畴，不宜在单元层伪造。**

### 变异验证（本节 2 个，均被预期测试捕获）

| # | 变异 | 捕获它的测试 |
|---|---|---|
| 26 | 交换 `observation_date`/`available_date` 的发布顺序 | C5.2 字段列表测试 **+** migration 一致性测试 |
| 27 | 从发布帧里删掉 `stale_days` | 同上两条 |

**全量：1436 passed / 45 skipped**（本节起点 1431；+5 materializer），
`ruff check src tests` 清、`mypy src` 157 文件清；
`materializer.py` 两次变异后均以 `git status --porcelain` 确认**还原干净**。
