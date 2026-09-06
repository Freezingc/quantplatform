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

- [x] **C2.4 运行测试并提交**

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

- [x] **C3.4 用已知问题构造回归 fixture**

从 Phase 0 审计中抽取匿名化的 GP_017、GP_077 重复键样例，证明重复行被计数且冲突不会进入 clean。fixture 只保存必要字段和 source hash，不复制大规模源数据。

> **已交付（2026-09-06 15:45，详见续二十二）**：本款点名的「Phase 0 审计」在本仓库**不存在**
> （续十一逐一核过 13 个命中文件），故改走同一节列出的出路 (ii)：对 `yeslab_386`/`yeslab_446`
> 跑一次**有界只读普查**并据实构造 fixture。「匿名化」「只保存必要字段和 source hash」
> 「不复制大规模源数据」三款逐条满足——fixture 里**没有一个值来自源**，业务键与数值全为合成，
> 从源带过来的只有**实测分布**（六类键的占比、重复重数、组内可用行数与相异值数）与
> 所依据普查工件的 sha256。

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

> **出处注记（2026-09-06 15:55，续二十二）**：本款标题是"回归测试**已知**高相关因子"，
> 而下面四条 fixture **全部是合成的**。**这与 C3.4 的处理相反**：C3.4 同样点名了两个真实因子
> （GP_017/GP_077），当时**因缺真实出处而宁可不勾**，理由是"凭空编一个会让回归测试
> **看起来有真实出处而没有**，那比缺这条测试更糟"。**同一个问题在两处被作了相反的决定，
> 且当时未注明**——这是本节的一处疏漏，据此更正。
> **为本款辩护的理由是有的、但不充分**：C7.5 的条目正文枚举的是**案例类型**
> （完全重复／符号相反／缺失错位／链式传递），并未像 C3.4 那样点名某份源工件，
> 故按正文逐款满足即成立。**但仓库里确实存在一个可用的真实案例**：
> `gp_turnover_scalar_degeneracy_diagnostic_v1.json` 记录了 `yeslab_436/446/468`
> 三者**日度 RankIC 完全相同**（`turnover × 每日横截面标量` → 与 `turnover_rate` 完全冗余），
> 续十一 明确称其为"C7 聚类的问题域"。
> **可选的第二条路**：从该退化派生一条真实 fixture。**本轮未做**，理由具体：
> 它需要的是**相关性**证据（日度 RankIC 序列）而非本轮普查的**重复键**证据，
> 属另一份工件；且 C7.1 的 scope 绑定与 C7.2 的覆盖率分母仍在**决定 E** 上未定，
> 在 scope 未定时给 C7 补真实 fixture，可能要按定后的 scope 重做。
> **勾选状态不变**（本款按正文已满足），但**读者应知道四条 fixture 都是合成的**。

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

## 勾选审计续十三（2026-09-06 10:45）：C5.2 已勾并闭上一条无人看守的边；C5.1/C5.3/C5.4/C5.5 的缺口逐一定位（其中一处比预想的好）

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

## 勾选审计续十四（2026-09-06 10:55）：六条完成标准的逐条判定——一条完全满足，五条各缺一块，且缺口高度集中

续七～续十三把 C1–C5、C7 逐款核过之后，六条完成标准第一次**可以据实判定**而不是估计。
结论先说：**只有第 4 条完全满足**；其余五条各缺一块，而**缺口不是均匀分布的**——
其中三条（1、3、6）缺的是**同一件事**：治理代码写了但没接到生产路径上。

| # | 完成标准 | 判定 | 已满足部分的依据 | 未满足部分 |
|---|---|---|---|---|
| 1 | 启用因子都有显式、版本化、不可变 registry 定义；源新增列不会被自动纳入 | **半满足** | 后半**完全成立**：`_ALLOWED_TABLES` 是闭集、生成器拒绝通配符名、C1.5 dry-run 报告未映射源列（实跑 11 表 0 未映射列）。前半的**定义**成立：487 条显式逐因子条目、带 `factor_version`、`FactorDefinition` 不可变（`ImmutableDefinitionError`，且比规格更严：从 DRAFT 起即不可改） | **「启用因子…registry 定义」不成立**：`FactorRegistry` 生产调用者为 **0**，没有任何被启用的因子真的被注册过；生产冻结集是 `factor_set_frozen_v1.json`，成员资格来自人工裁决 D1–D4。**依赖决定 A** |
| 2 | GP_017/GP_077 类重复问题被统计，冲突键全部隔离，clean 表唯一 | **三分之二** | 「冲突键全部隔离」✓：新增不变量测试断言 clean 与 quarantine 的业务键**互斥**（冲突键不只是被记录，而是**对 clean 零贡献**）。「clean 表唯一」✓：新增测试断言 clean 在四元组业务键上无重复 | **「重复问题被统计」✗**：C3.4 缺输入——Phase 0 审计在本仓库不存在（13 个命中文件逐一核过，无一是重复键审计；唯一的 GP_077 诊断讲的是标量退化，属 C7 问题域）。**GP_017 = `yeslab_386`、GP_077 = `yeslab_446` 已确立** |
| 3 | 技术因子不 ffill；基本面严格在公告后下一交易日可用且最多持有 180 日 | **性质成立，路径半接线** | 三款性质**逐一实测成立**：技术不 ffill（**两个读法都钉住**：不凭空造行 + 已有行的缺失值不继承前一日）；基本面下一交易日可用（含周末/非交易日公告顺延）；180 日上限（**把比较符本身孤立出来**：180 保留、181 丢弃） | **技术半部已接线**（`materialize_technical_availability` ← `materializer.py:101`）；**基本面半部未接线**（`expand_fundamental_events` 生产调用者为 0），且它带有**决定 B 的跨期版本缺陷**（旧报告期的迟到重述会顶掉新报告期） |
| 4 | 修改未来公告/值不改变任何过去 factor observation | **完全满足** | `tests/leakage/test_factor_future_perturbation.py` 六条：过去报告期的未来修订不改写历史、新增纯未来公告不改动更早记录、修改未来值不改动更早摘要、删除未来 universe 日期不改变过去展开、未来技术值不改动更早记录、每条发布行满足 PIT 排序 | 无。**这是六条里唯一有专属测试目录的一条**，也是覆盖最扎实的。（基本面那半在 `expand_fundamental_events` 上验证，该函数未接线，故对已发布数据而言是"既实测成立、又暂无对象"） |
| 5 | 结构质量不读取标签；相似度与代表选择严格绑定训练 scope | **前半满足，后半部分** | 前半**完全成立且钉得很硬**：`quality.py` 中 `label|target|forward_open|return_5d` **零命中**，并有 `test_structural_quality_is_invariant_to_a_perfectly_predictive_label_column`——**塞进一个完美预测的标签列，快照不得有任何变化**。后半的日期维度成立（闭区间、scope 隔离已测），代表排序**只用训练内统计**（"不得读取 validation/test 指标"已满足） | 相似度**未绑定 `universe_id`/`data_version`**（模块内无任何 query builder，scope 只由日期界定）→ **决定 E**；代表排序键与规格**两处不符**（中间两项顺序相反、mean vs median）→ **决定 D** |
| 6 | factor set content-addressed、顺序固定、冻结后不可变，可被 D/E/F 精确复用 | **代码层四款成立，复用款不成立** | content-addressed ✓（对排序后的 `(factor_id, factor_version, expected_direction)` + selection scope + rules version + registry definition hashes 做 SHA-256）；顺序固定 ✓（`feature_order` 确定性，已测）；冻结后不可变 ✓（FROZEN 为终态，改任何成员产生新 `factor_set_id`，已测） | **「可被 D/E/F 精确复用」不成立**：`create_factor_set` 生产调用者为 **0**；生产中 D/E/F 实际读的是 `factor_set_frozen_v1.json`，且是**按 `scope` 字符串引用**（`materialize_frozen_factor_daily.py`、`verify_admission_consistency.py`、`walk_forward_factor_daily.py`），**那个工件没有 content-addressed id**。**依赖决定 A** |

### 缺口的集中度：五条未满足里，三条是同一个原因

**标准 1、3、6 的未满足部分，根源都是续十/续十二查出的"未接线"**：
`FactorRegistry`、`create_factor_set`、`expand_fundamental_events` 三者的生产调用者都是 **0**，
加上 C6.2 结构质量模块与 `select_cluster_representatives`，**共五处**。
**这不是五个独立缺陷，而是一个归属问题（决定 A）的五个表现。**
因此**在决定 A 之前，标准 1/3/6 都无法通过补测试推进**——
它们的代码层性质已经全部成立并被变异验证，缺的是"这套代码是否就是要用的那套"。

**标准 2 与标准 5 的缺口性质不同，可独立推进**：
标准 2 缺一份**输入**（Phase 0 重复键审计；Task 2.4 落地后跑一次有界只读普查即可解锁，无需裁决）；
标准 5 缺两个**接口/判据决定**（决定 D 的排序键、决定 E 的 scope 绑定与覆盖率分母）。

### 一个值得单独记的正面结论

**标准 4 与标准 5 前半的钉法，是本计划里最接近"不可绕过"的两处。**
标准 4 有专属 leakage 目录，六条覆盖"加/改/删未来"三类扰动的两个因子类型；
标准 5 前半用的是**不变量式**测试——不是断言"没读标签"（那只能靠 grep，而 grep 会被
间接引用绕过），而是**塞进一个完美预测的标签列、断言输出逐位不变**。
**一个真在读标签的实现不可能通过它**，这与"只检查代码里有没有出现某个词"是不同强度的证据。
**其余各条宜以此为范式**：能写成不变量的，不要写成结构检查。

### 本节的六条判定为何**不勾选**完成标准复选框

六条里只有第 4 条可勾。但**完成标准是 C 计划的出口条件，不是逐款清单**：
在标准 1/3/6 依赖的决定 A 未定、标准 2 缺输入、标准 5 缺两个决定的情况下，
**单独勾掉第 4 条会让"完成标准"这一节呈现出部分完成的假象**，
而读者最需要知道的是**五条同缺、且三条同因**。
**故本节只出判定、不动复选框**；待决定 A 落定后按新的事实一次性重判六条。

### 本轮补的两条不变量（属标准 2，已变异验证）

`test_the_clean_output_is_unique_on_the_business_key` 与
`test_no_business_key_appears_in_both_clean_and_quarantine`。
**两条性质此前都为真，但都没有被断言过**：`clean 表唯一` 是事实表的键性质
（两行共享一个业务键意味着同一 (日期, 证券, 因子, 版本) 有两个已发布值，
下游要么重复计数、要么静默保留先到的那个）；
`冲突键全部隔离` 的关键在"**从 clean 中隔离**"而不只是"被记录"——
一个键同时出现在两个帧里，意味着冲突观测**既进了事实表、又在审计轨迹里声称被扣下了**。
变异 M28（代表行不再按键归约）被 **12 条**测试捕获、M29（冲突键也产出 clean 代表）被 **6 条**捕获，
**两条新测试都在捕获集合里**，故非装饰性。

**全量：1438 passed / 45 skipped**（本节起点 1436；+2 dedup 不变量），
`ruff check src tests` 清、`mypy src` 157 文件清；
`deduplication.py` 两次变异后以 `git status --porcelain` 确认**还原干净**。

## 勾选审计续十五（2026-09-06 11:05）：C2.4 已勾（Task C2 收口）；C4.5 差一款，且缺的那款暴露出一个"只写不读"的溯源字段

### C2.4 已勾选：Task C2 四款全部收口

C2.4 是 Task C2 的收口项，其两条 Run 命令实测：

```
pytest tests/unit/factors/test_source_reader.py        -q  ->  8 passed
pytest tests/unit/factors/test_source_backfill_query.py -q  -> 10 passed
```

Expected 的四类逐一对上：**宽/长映射**（长表已在生产中用于冻结集物化；宽表在本源不出现，
按字面由列序不变测试覆盖——沿用续六的范围判定）、**列漂移**（C1.5 dry-run 报未映射源列 +
reader 重投影到恰好注册的那几列 + 续九新增的 catalog↔读路径一致性）、
**chunk**（月份/因子组分块与整帧一次归一化逐帧相等，含 `source_row_id`）、
**数值异常**（±inf 转 invalid flag、raw 精度逐位保真、不 winsorize/scale/impute）。
**C2.1/C2.2/C2.3 三款实质项均已勾**，故收口项可以成立。

### C4.5 仍不勾：四款中三款由 Hypothesis 性质测试钉得很硬，第四款**没有任何执行点**

**先说好消息——C4.5 前三款是本计划里证据强度最高的部分之一。**
`tests/property/factors/test_availability_invariants.py` 用 Hypothesis 在**生成的**
事件帧/universe 帧上断言，而不是在几个手写样例上：

- **`observation_date <= available_date <= signal_date`**：
  `test_every_published_row_satisfies_the_point_in_time_ordering` 对**每一条发布行**断言
  `ann_date < available_date <= signal_date`，**并同时钉住 `stale_days` 的公式与上下界**
  （`stale_days == (signal_date - available_date).days` 且 `0 <= stale_days <= 180`）。
- **technical 三者相等**：由 C4.1 的单测钉住（本轮又补了"不 ffill"的两个读法）。
- **fundamental `available_date > ann_date`（按交易日规则）**：
  `test_available_date_is_the_earliest_trading_day_strictly_after_the_announcement` 断言三件事——
  `available > ann_date`、`available` **落在开市日**、且 **ann_date 与 available 之间不存在更早的开市日**。
  第三条是关键：只断言"严格晚于"的话，顺延到下周也算通过。
- **附带一条 C4.3 的强性质**：`test_expansion_never_invents_a_key_or_duplicates_one` 断言
  **发布键是 universe 请求键的子集**且无重复——即展开**既不发明也不复制**键。

**第四款「source watermark 的捕获时间不晚于 run publish」——完全没有实现，且缺得很隐蔽。**
`SourceBatch.extracted_at` 由**四个源适配器**认真填写
（`sources/yeslab.py` 三处、`sources/tushare.py` 一处，都是 `pd.Timestamp.now(tz="UTC")`），
`contracts/source.py:22` 也把它声明为契约字段，`pit/security_profile.py:97` 还往下传递。
**但全仓没有任何一处读取它、更没有任何一处把它与发布时间比较**：

```
grep -rn "extracted_at" research_v2/src research_v2/scripts
  → 4 处赋值（源适配器）、1 处契约声明、1 处向下传递、1 处脚本自写 now()
  → 比较运算（<= / >= / < / > / max / min）命中：0
  → 非赋值形式的读取（即真正的消费者）：0
```

**这是一个"只写不读"的溯源字段**：每一次取数都忠实地记下"我是什么时候抓的"，
然后没有任何东西去问"这个时间是不是晚于它被发布的时刻"。
而这一款要防的失效是具体的——**如果一次 run 用了在发布时刻之后才抓取的水位，
那么被发布的数据里可能含有发布当时还不存在的信息**，这正是 PIT 的最后一道缝。
前三款把"行内的时间顺序"钉得很死，**第四款管的是"批次与 run 之间的时间顺序"，而它没有执行点。**

**这与续十查出的 C1.4 是同一族问题**：那里是 `create_factor_set` 的 docstring 声称
"from active-registry member identities" 而代码从不接触 registry；
这里是契约声明了一个字段、四处代码认真填它、而没有任何一处校验它。
**共同点是：一个看起来已经在了的保证，其实没有执行点。**
这类问题 grep 字段名查不出来（字段确实在、确实被填），只有**grep 它的消费者**才查得出来。

**不代实现的理由**：要执行这一款，需要决定**比较发生在哪一层**——
是在 `StagedBatch`/发布器里把 `extracted_at` 与 `created_at` 对比并拒绝发布，
还是在证据层只记录不拒绝。前者会**新增一条能让发布失败的路径**，
在 Task 2.4 正在逐位复现、且 `factor_observation_clean` 有版本指针红线的当口，
**新增拒绝路径属于会改变运行链行为的改动**，不宜代做。

### 顺带更正本文书一处时间戳

续十三原标 11:10，实际写作时刻为 10:45，已更正。审计时间戳是追溯顺序的依据，
**标在未来会让人误判各节之间的先后**，故记一笔。

**全量：1438 passed / 45 skipped**（本节无新增测试，仅收口与判定），
`ruff check src tests` 清、`mypy src` 157 文件清。

## 勾选审计续十六（2026-09-06 12:15）：C4.2 的「退市」款已补测（该项仍不勾，因另一款需改契约）；第五次查出"测试声称的范围大于它钉住的范围"

### 补上 C4.2 唯一可在不改契约的前提下关闭的一款

续十五记 C4.2 差两款。其中「无 `ann_date` 的记录隔离」需要改函数返回契约（见续十五，不代做）；
**「退市」这一款不需要**——它只是没有测试。本轮补两条：

- **`test_a_delisted_security_stops_expanding_while_its_value_is_still_fresh`**
- **`test_a_suspended_security_gaps_and_then_resumes_without_losing_the_value`**

**机制本身是对的，且值得说清为什么对**：模块里没有任何 `delist` 逻辑
（`grep delist availability.py` 零命中），退市之所以成立，是因为
**行只为 universe 里真实存在的 `(signal_date, ts_code)` 产出**——
成员资格归 universe 所有，可用性层不该再判一次。**这个分工是对的**，
但"隐式成立且未测"就是"无人看守"：
**如果哪天有人把 staleness 变成唯一的门（一个看起来很合理的"简化"），
退市证券会继续发布它最后一次公告的基本面值，最多再发 180 天。**
那不是前视（值确实是当时已公布的），**但那是"在一个已经不能交易的证券上继续给出因子值"**，
对以"可投资 universe 上的每日横截面排序"为目标的管线是实质错误。

**测试设计上的关键是对照组。**只断言"退市后没有行"是**不够的**——
"没有行"同样可以由 staleness 窗口关闭造成，那样测试就证明不了是**成员资格**在起作用。
故 fixture 里放一只**携带完全相同事件、且在 2025-04-03 仍在 universe 里**的对照证券：
它当天拿到行、`stale_days == 3`，**从而证明那天窗口是开着的**，
退市那只的沉默只能归因于成员资格。
停牌那条则钉住"成员资格可以不连续"：中途缺席两天后恢复，值必须**重新出现**、
`stale_days` 按可用日在自然日上算（不因中断而延长、也不因中断而永久丢失）。

**变异验证**：

- **M30**（展开改为对 `calendar.dates × 事件里的证券` 做笛卡尔积、无视 universe）
  → **7 条测试失败**，含两条新增。
  顺带一个可观测的副作用：**该变异让 `tests/property/` 的 Hypothesis 用例跑到无法在数分钟内结束**，
  只能中断后改为只对单测文件跑——**这本身就是 C4.3「而非全笛卡尔积」的一次实测印证**：
  笛卡尔展开的代价不是理论上的，它把属性测试直接拖死。
- **M31**（`stale_days` 改为相对上一条已发布行计算）→ **5 条测试失败**，含停牌那条。

### 第五次同源教训：停牌那条测试**第一版没有钉住它 docstring 声称的东西**

第一版停牌 fixture 只有**一个事件**。而 M31（从上一行的 `available_date` 起算 staleness）
**跑下来它照样通过**——因为单一事件的 `available_date` 恒定，
"从获胜版本自己的可用日算"与"从上一条已发布行的可用日算"**给出同一个数**。
**它的 docstring 却写着"staleness is measured from the available date, not from the last
published row"**，即声称了一个 fixture 根本没有区分辨力的保证。

**修法**：让缺席区间**跨过一次修订**——原版可用 03-31、修订版可用 04-02（同一 report period，
故仍在 C4.4 的同期修订语义内，**不牵涉未决的跨期优先级问题**）。
于是 04-03 的获胜者是修订版（可用 04-02 → `stale_days == 1`），
而上一条已发布行携带的是原版的可用日（03-31 → 会给出 3）。**两种实现结论相反。**
重跑 M31 → 该测试失败，加入捕获集合。

**这是本轮第五次同一类问题**（续八：非有限值测试在 inf 计入与否下结论相同；
续十：`_registry_at` 要 DRAFT 却走到 ACTIVE；续十一：变异插在 `continue` 之后没落到被测路径；
续十五：只断言行数的不-ffill 测试在 `groupby.ffill()` 下全绿；本节：单事件停牌测试）。
**五次的共同形状是同一个：测试的 docstring 或名字声称的范围，大于它的 fixture 实际能区分的范围。**
而读者（包括未来的我）只读 docstring。
**可操作的判据已经稳定下来**：写完一条性质测试，**必须构造至少一个"被破坏的实现"并确认它失败**；
如果构造不出来，说明还没想清楚这条测试到底在排除什么。

### C4.2 的判定不变

**仍不勾**：两款里现在只差「无 `ann_date` 的基本面记录隔离」，
而它需要给 `expand_fundamental_events` 增加隔离通道（改返回契约）。
**该函数无生产调用者，故改契约的最佳时机仍是接线之前**——留给 owner 决定形状，
理由与续十五一致，不重复。

**全量：1440 passed / 45 skipped**（本节起点 1438；+2 availability），
`ruff check src tests` 清、`mypy src` 157 文件清；
`availability.py` 两次变异后以 `git status --porcelain` 确认**还原干净**。

## 勾选审计续十七（2026-09-06 12:25）：C5.4 不勾——已有的月度续跑器回答的是**另一个问题**；并查出第六处未接线组件

### 先把"看起来已经有了"的东西辨清楚

`research_v2/src/research_v2/backfill/` 下确实有一套续跑机制：
`resolve_monthly_backfill_windows`（按**真实交易月**而非日历月切分，带回溯窗口）与
`MonthlyBackfillRunner`（`plan()` 给出每月仍缺的对象，`run()` 只发布 pending 的那些，
**已完成的月份永不进 publish 回调**）。

**但它回答的问题不是 C5.4 的问题。**两者差在**两个正交的维度**上：

| | `MonthlyBackfillRunner`（已有） | C5.4（要求） |
|---|---|---|
| **问题** | 哪些月**从未发布过** | 哪些 signal date **自发布后变了** |
| **判据** | 指针是否存在（`pointer_reader(object, month)`） | 源数据是否变化 |
| **粒度** | 月 × 对象 | signal date |
| **触发** | 缺失（completeness-driven） | 变更（change-driven） |

**关键后果**：**一个已有指针的月份会被无条件跳过，无论它的源后来发生了什么。**
源里某天的值被更正、某行被重述、水位推进——**已发布月份里的这些变化永远不会被重算**，
而且 `plan()` 的结果**一个字都不会变**。

**这不是 runner 的缺陷。**续跑（resumability）与变更检测（change detection）是两份不同的工作，
**把它们混在一起会让一次续跑重新发布它早已完成的月份**，那才是 bug。
**记下它是因为"我们有一个可续跑的月度 runner"读起来像"我们有增量重算"，而后者不存在。**
C5.4 要检测源变更，需要**按日期的摘要或水位与已持久化状态做比对**——
那是一份**新的持久化状态**，属 schema 决定（新表或新列），且要动发布路径，
**在 Task 2.4 逐位复现期间不宜代做**。

### 第六处未接线组件

`MonthlyBackfillRunner(` 在 `research_v2/src` 与 `research_v2/scripts` 中
**生产调用者为 0**（只有自身定义与单测）。
**这是本轮查出的第六处**（续四 C6.2 结构质量、续七 `select_cluster_representatives`、
续十 `FactorRegistry` 与 `create_factor_set`、续十二 `expand_fundamental_events`）。
已同步进决策清单的未接线表格。

### 本轮补的 7 条测试：三道 fail-closed 守卫此前**完全没有测试**

runner 的 docstring 明写这些守卫是为了"prevent accidental cross-month or duplicate writes"，
但**没有一条测试触及它们**——既有 3 条测试全是 happy path（plan 只列缺失对象、
run 跳过完成月并发布 pending 月）。已补：

- **`test_the_publish_callback_cannot_report_an_object_that_was_not_pending`**：
  回调报告一个**不在该月 pending 集合里**的对象必须被拒。
  否则结果会声称某对象在某月已发布，而**那个月的指针从未被检查过**——
  这正是"续跑最后把某月标成 complete、依据却是另一个月的证据"的路径。
  **fixture 细节值得记**：回调必须报告一个**根本不在 `required_objects` 里**的对象名
  （用 `label_daily`）。第一版报告了一个必需对象名，而该月在 pending 状态下三个对象**都在**
  pending 集合里，于是**先触发的是唯一性守卫**，跨对象守卫根本没被测到——测试通过了，
  但通过的理由是错的。
- **`test_the_publish_callback_cannot_report_the_same_object_twice`**：重复报告必须被拒。
- **`test_a_publish_callback_that_reports_nothing_is_refused`**：pending 月的回调**空手而归**必须被拒。
  若静默接受，该月会被记为 `published` 而**没有任何指针被写**，
  于是下次续跑又把它规划为 pending，**结果与指针表互相矛盾**。
  **顺带钉住一处误导性错误信息**：空报告与重复报告**共用同一条** "must report unique object
  names"，对空报告这个情形是**答非所问**的。已按现状钉住并在 docstring 注明，
  使改进它成为一次**有意的**变更而不是顺手改文案。
- **`test_required_objects_must_be_non_empty_and_unique[empty|duplicated]`**：
  **空的需求集合会让每个月都平凡地"完成"、从而什么都不发布**——
  这是一个会让整条回填静默空转的配置错误。
- **`test_the_runner_is_blind_to_source_change_and_answers_only_pointer_presence`**：
  把上面的判定写成可执行的证据。fixture 里放两份不同的"源摘要"
  （`digest-a` / `digest-b-changed`），**两者都不是 runner 的输入**，
  断言正是这一点：**两次 `plan()` 结果完全相同、且都是 `complete`**。
- **`test_a_missing_pointer_for_one_object_leaves_only_that_object_pending`**：
  月内粒度是**逐对象**的，故半个月只补那一个缺口。

**变异验证**：M32（去掉跨对象守卫）→ 1 条失败；
M33（去掉唯一性/非空守卫）→ 2 条失败；M34（去掉 `required_objects` 校验）→ 2 条失败。
**三个变异各自只被对应的测试捕获**，说明这三条守卫现在各有专属看守。
`runner.py` 三次变异后以 `git status --porcelain` 确认**还原干净**。

### C5.4 三款的逐款判定

- **技术日频源变化只重算其变化 signal dates** → **未实现**（如上：现有机制是 completeness-driven，
  且粒度是月不是 signal date）。
- **基本面公告/修订变化重算该证券从 available date 到下次版本或 180 日上限与 universe 交集** →
  **未实现，且被决定 B 阻塞**：这一款的"到下次版本"里的**"下次版本"语义正是决定 B 争议的
  跨期优先级**（若按 (B-i) 最新报告期优先，则不同报告期的有效区间会重叠，
  重算范围就不再是"到下一个版本的可用日"这么简单）。**故应等 B 定后再实现，否则会改两遍。**
- **registry 新版本不覆盖旧版本** → **在 registry 层成立且有测试**（续十：
  新旧版本并存、各有自己的生命周期、definition hash 不同），
  **但 registry 未接线**，故它对物化**没有实际约束力**。

**故 C5.4 不勾。**

**全量：1447 passed / 45 skipped**（本节起点 1440；+7 backfill），
`ruff check src tests` 清、`mypy src` 157 文件清。

## 勾选审计续十八（2026-09-06 12:35）：C5.3 的 dry-run 实质已交付（规划器 + 17 测试）；该项仍不勾，因为四条命令里两条属决定 A

C5.3 有两半：**dry-run 要输出什么**，与**四条 CLI 命令**。本轮交付前者的全部实质，
因为它**不依赖任何未决裁决**——它只需要已经交付的三样东西组合起来：
**C1.1 的 catalog**（因子数、源绑定）、**C1.5 的 schema dry-run**（源表/列、未映射列、blocked）、
**`backfill/windows.py` 的月度窗口**（日期 chunk、回溯起点）。

### 交付内容

**`research_v2/src/research_v2/factors/materialize_plan.py`**（新，纯函数，无 I/O）：
`plan_factor_materialization(entries, *, calendar, start_date, end_date, schemas=None, retired=...,
cross_section_size=None, lookback_trading_days=20, strict=True, target_table=..., target_partition_column=...)`
→ `FactorMaterializePlan`，其 `chunks` 为逐月 `FactorMaterializeChunk`。
**17 条测试**（`tests/unit/factors/test_materialize_plan.py`）。

**规格要求的五项输出，逐项落点：**

- **因子数**：`factor_count`，**退役因子从中扣除**（有专测：退役一个后计数、估算、列集合三处同时变化）。
- **源表/列**：`source_tables` 与 `source_columns`，后者**来自 catalog 登记**
  （复用 C1.5 的 `registered_source_columns`），不是硬编码；
  专测用一个登记了 `val`/`dt` 而非 `value`/`trade_date` 的条目钉住这一点——
  **这正是续九查出的"两个真值来源"那条边**，规划器因此不会与读路径分家。
- **日期 chunk**：直接复用 `resolve_monthly_backfill_windows`，故 chunk 与生产回填**同源同定义**
  （按真实交易月切分、带回溯起点）；专测断言规划器的 chunk 与该函数的输出**逐项相同**。
- **估算行数**：见下。
- **目标分区**：`target_table = factor_observation_clean`、`target_partition_column = signal_date`
  ——**两者取自 migration 003 的实际声明**，不是猜的；`logical_partition` 为 `YYYY-MM`。

### 三个设计决定，每个都对应一个具体的失效模式

**(1)「禁止读取值或写库」做成了签名性质，而不是函数内部的自律。**
规划器**不接受任何 session / client / connection 参数**：输入只有 catalog 条目、
一份**已由别处只读采集好**的 schema 映射、以及交易日历。
**没有能让语句到达数据库的参数，就没有能被说服去用的通道。**
测试用 AST 钉住：模块不 import `research_v2.db.*` 也不 import `dolphindb`，
标识符里没有 `query`，`plan_factor_materialization` 的参数里没有 `session`/`client`。
**用 AST 而非文本**的理由与续九相同——docstring 里正是要提到"不查询、不写库"，
文本匹配会同时产生假阳与假阴。

**(2) 估算行数在没有横截面宽度时是 `None`，不是猜一个。**
这是元数据规划器**唯一真正无法知道**的数：它取决于每天有多少证券在交易。
**猜一个宽度会产出一个看起来像测量值的数字**，故 `cross_section_size` 必须由调用方显式给出，
否则 `estimated_rows` 与 `estimated_rows_total` 都是 `None`（专测钉住 `None` 会**传播到总数**，
使调用方无法把"估算"与"猜测"混着相加）。
**取得真实宽度需要对源做 count——那正是 dry-run 不该做的事**，故这条边界写在 docstring 里。
公式集中在一处：`factor_count × trading_days × cross_section_size`。

**(3) `trading_days` 只数目标月，绝不数回溯窗口。**
回溯是为了算特征而读的，**不是本月要发布的**，把它算进估算会按回溯比例高估写入量，
更糟的是**让估算与物化器实际发布的行数不一致**。
专测用"回溯跨到上一年 12 月"的日历把两者拉开（目标 5 天 vs 含回溯 8 天），
并同时断言 `estimated_rows == 2×5×1000` **且** `!= 2×8×1000`——
**只断言前者是不够的**，因为一个把回溯算进去的实现可能碰巧也满足某个等式。

**另有一处刻意的不对称**：`materializable` 要求 `schema_verified and not blocked_factors`，
故**"没对着源校验过"与"校验过且干净"报告不同的结果**。
若两者都为真，跳过 schema dry-run 就会**看起来像通过了它**——
而这正是 C1.5 存在的意义。专测钉住：不传 `schemas` 时 `schema_verified is False`
**且 `materializable is False`**。

**fail-closed 默认**：`strict=True` 时**任何一个 blocked 因子都会中止规划**并列出它与原因。
这是把 C5.1 的原则（任一步失败均不产生正式可见行）**前移到规划阶段**：
一个登记列在源里不存在的因子根本读不出来，就不该出现在计划里。
`strict=False` 则返回带 `blocked_factors` 的计划，**让调用方能看到全貌而不是只看到第一个失败**
（与 C1.5 dry-run"收集全部理由"同一取向）。

### 变异验证（4 个，各被其对应测试捕获）

| # | 变异 | 捕获它的测试 |
|---|---|---|
| 35 | `materializable` 不再要求 `schema_verified` | 未校验计划那条 |
| 36 | `trading_days` 从回溯起点算起 | 估算不含回溯那条 **+** 退役影响估算那条 |
| 37 | 调用方没给宽度时自己编一个 5000 | 「宁可 `None` 也不编造」那条 |
| 38 | `strict` 不再因 blocked 因子中止 | fail-closed 那条 |

`materialize_plan.py` 四次变异后以 `git status --porcelain` 确认**还原干净**。

### C5.3 仍不勾：剩下的是 CLI，而其中两条命令属决定 A

规格的四条命令：

- `qr-v2 factor registry validate --catalog ...` / `qr-v2 factor registry import --catalog ...`
  → **这两条是 `FactorRegistry` 之上的 CLI，而 registry 无生产调用者**（续十、决定 A）。
  **在"以哪条冻结集路径为准"未定之前给一套未采用的 registry 做命令行，
  是在为一个可能被降级为"设计参考"的东西建门面。故不做。**
- `qr-v2 materialize factors --start ... --end ... --factor-set all-active`
  → **dry-run 的实质已交付**，剩下的是 typer 接线；且 `--factor-set all-active` 里的
  "active" **正是 C1.4 未实现的那个状态**，故该旗标的语义也依赖决定 A。
- `qr-v2 factor quarantine report --run-id RUN_UUID`
  → 独立于决定 A，可做；但它是**读已发布的 `factor_quarantine`**，
  而该表的 `distinct_values_json` 有续十一记的**非法 JSON** 缺陷（`[Infinity]`），
  **报告命令会把那个缺陷直接暴露给用户**，故宜与决定 F 一并处理。

**因此 C5.3 的判定是：dry-run 的输出内容已全部可产出并经测试，CLI 表面尚未建，
而其中三条命令的语义分别依赖决定 A（两条）与决定 F（一条）。**
**规划器本身不依赖任何裁决**，这也是它现在就能交付的原因。

**全量：1464 passed / 45 skipped**（本节起点 1447；+17 planner），
`ruff check src tests` 清、`mypy src` **158** 文件清。
> **勘误 2026-09-06（续二十二）**：本节 `ruff` 半句**写下时即不成立**（`ed7fb5a` 引入的 import 块未排序，`aa92fcd` 才修）；`mypy` 半句经复核准确。

## 勾选审计续十九（2026-09-06 12:47）：C5.3 的 CLI 表面交付了不依赖裁决的那一条命令，并对真实 catalog 实跑通过

续十八交付了规划器；本节把它接成 **`qr-v2 materialize factors`**，
即 C5.3 四条命令里**唯一一条语义完全不依赖未决裁决**的。

### 交付内容

- **`cli/materialize.py::plan_factors`**（新，+139 行）：解析参数、读 catalog 与日历、
  可选读入已保存的 schema 集、调用规划器、输出 JSON、可选写文件。
- **`cli/app.py`**：注册 `materialize_app.command("factors")`（+37 行），
  `Path` 型选项按本仓既有约定加 `# noqa: B008`（与 `app.py:147/181` 一致）。
- **15 条 CLI 测试**（`tests/unit/cli/test_materialize.py`，用 `CliRunner`，与该文件既有两条同风格）。
- **一处去重重构**：把"已保存 schema 集"的磁盘格式从脚本里提到
  `catalog_dry_run.schema_set_to_payload` / `schema_set_from_payload`，
  **C1.5 的脚本与 CLI 共用同一份定义**。理由与续九查出的"两个真值来源"同类：
  **一个格式有两个独立编码器，就是保存的工件某天开始读不回来的成因**——
  而且它会在**日后有人要复核判定时**才失败，不在改格式的那一刻失败。
  **已验证向后兼容**：重构前保存的 `catalog_schema_dry_run_schemas.json` 仍能载入，
  且离线复演报告与实跑报告**逐字节相同**。

### 对真实 catalog 实跑（离线，未连库）

用**已提交的 487 条 catalog** + **续九实跑保存的真实 schema 集** + 一份日历文件：

```
qr-v2 materialize factors --start 2014-01-01 --end 2014-03-31 \
  --catalog configs/factors/source_catalog.yaml --calendar <dates> \
  --schemas <续九保存的 schema 集> --cross-section 2500 --dry-run

factor_count           : 487
source_tables          : 11 张（factor_data1..11）
schema_verified        : True        materializable : True
blocked_factors        : []          unmapped_source_columns : 11 张表全为空
chunks                 : 2014-01 (23 交易日, 28,002,500 行)
                         2014-02 (20 交易日, 24,350,000 行)
                         2014-03 (21 交易日, 25,567,500 行)
estimated_rows_total   : 77,920,000
target_table           : factor_observation_clean   target_partition_column : signal_date
```

**估算的算术已独立核算**：487 × 23 × 2500 = 28,002,500 ✓、487 × 20 × 2500 = 24,350,000 ✓、
487 × 21 × 2500 = 25,567,500 ✓、三月合计 77,920,000 ✓。
产物存于 `/data/zhangyudong/qrv2_research_artifacts/factor_materialize_plan_demo.json`。
**注意该演示用的是近似（工作日）日历**；生产规划必须喂真实的 `trade_calendar` 日期，
文件首行已注明 APPROXIMATE。**这也正是把日历做成显式输入的价值**：
用哪份日历是**可见的、被记录的**，而不是藏在一次取数里。

### 三个设计决定

**(1) 日历是显式文件输入，不是命令内部去取。**
这同时买到三样东西：命令**在没有集群可达时也能跑**（C5.3 的"禁止读取值"因此是结构性的，
不只是"这次没读"）；输出**可复现**（同一份日历文件 + 同一份 catalog → 同一个计划）；
以及**用哪份日历这件事本身进了证据**（`calendar` 路径写进输出文档）。
支持"每行一个 ISO 日期"（允许 `#` 注释与空行）与 JSON 列表两种格式。

**(2) 不带 `--dry-run` 时 fail-closed，退出码 2**，与同文件既有的 `emit_plan` **同一惯例**
（`materialize market/labels/...` 在不带 `--dry-run` 时也是打印计划后拒绝执行）。
**理由是这条命令背后确实没有物化执行器**：`build_technical_factor_batches` 有生产调用者，
但没有从 CLI 到它的已接线路径，**假装能执行比明确拒绝更危险**。

**(3) `--no-strict` 用于报告全貌。**默认 `strict` 下任何一个 blocked 因子都以退出码 2 中止
并把因子与原因写到 stderr；`--no-strict` 则输出带 `blocked_factors` 的完整计划、退出码 0，
**让调用方看到全部问题而不是第一个**。两种模式各有专测。

### 变异验证（2 个 CLI 层，各被其对应测试捕获）

| # | 变异 | 捕获它的测试 |
|---|---|---|
| 39 | 去掉不带 `--dry-run` 时的拒绝分支 | fail-closed 那条 |
| 40 | CLI 内部把 `strict` 硬写成 `False` | blocked 因子中止整份计划那条 |

`cli/materialize.py` 两次变异后确认无残留（`strict=strict` 与两处拒绝分支均在原位）。

### C5.3 的判定：仍不勾，剩下的三条命令各有归属

- `qr-v2 materialize factors` → **本轮交付**（以 `--catalog` 界定范围）。
  **但规格里写的是 `--factor-set all-active`，而 "active" 正是 C1.4 未实现的那个状态**，
  故该旗标仍依赖**决定 A**；本轮用 `--catalog` 而非 `--factor-set` 正是为了不预判 A。
- `qr-v2 factor registry validate` / `factor registry import` → **`FactorRegistry` 之上的 CLI，
  而 registry 无生产调用者**（决定 A）。不建。
- `qr-v2 factor quarantine report --run-id` → 独立于决定 A，但它读的是已发布的
  `factor_quarantine`，而该表的 `distinct_values_json` 有续十一记的**非法 JSON** 缺陷
  （`[Infinity]`，严格解析器直接拒绝）。**一个报告命令会把那个缺陷原样递到用户手里**，
  故宜与**决定 F** 一并处理。

**故 C5.3 现状：dry-run 的输出内容已全部可产出且经测试（续十八），
CLI 表面交付了四条里不依赖裁决的一条；余下三条分别绑定决定 A（两条）与决定 F（一条）。**

**全量：1479 passed / 45 skipped**（本节起点 1464；+15 CLI），
`ruff check src tests` 清、`mypy src` 158 文件清。
> **勘误 2026-09-06（续二十二）**：同上，`ruff` 半句写下时即不成立；`mypy` 半句准确。

## 勾选审计续二十（2026-09-06 12:55）：发布器的"同版本重跑"路径此前零单元覆盖，已补 8 条；C5.5/C5.6 仍不勾（要的是集成测试）

### 缺口：`V2BatchPublisher.publish` 的 retry 分支是一条四出口校验路径，而它一条单测都没有

C5.6 的 Expected 写的是「**正常发布幂等**；所有 blocker 均保持旧版本可见」，
C5.5 要覆盖的六种故障里有「**相同版本重跑**」与「**publish 前进程异常**」。
两者都落在同一段代码上：`publish(..., idempotent=True)`（**默认值**）在
"目标分区的当前 `data_version` 与本批相同"时走的分支。

**此前只有 `idempotent=False` 的拒绝路径被测**
（`test_database_publisher_rejects_republishing_the_current_data_version`），
另有一条集成测试在 `QRV2_INTEGRATION=1` 后面。
**也就是说：默认路径——真正会被走到的那条——没有单元测试。**
而它并不简单，它有四个不同出口：

1. 行数与目标已有行数**不符** → 拒（"row count does not match the retry batch"）
2. DQ 门**不过** → 置 REJECTED 并拒
3. 批次状态**既非 CREATED/VALIDATED 也非 PUBLISHED** → 拒（"must be newly created or validated for retry"）
4. 以上都过 → **不写任何东西**，必要时把 VALIDATED 推到 PUBLISHED，返回

### 8 条新测试，逐出口钉住

- **`test_a_same_version_retry_writes_nothing_at_all`**：`uploads == []` 且 `scripts == []`，
  批次终态 PUBLISHED。**这是"幂等"的实质**——重跑对目标是 no-op，不是第二次 append；
  若重跑会重新上传重新插入，分区里就有两份同样的行，
  而发布器入口刚校验过的业务键唯一性**在表里已经被破坏了**。
- **`test_a_retry_carrying_different_data_under_the_same_version_is_refused`**：**本轮最锋利的一条。**
  `data_version` 是内容寻址的，故"同版本但行数不同"意味着**调用方在给不同的数据复用同一个版本标识**。
  放它过去，版本指针的含义还在、而它背后的行已经变了——
  **这正是指针这个机制存在的目的所要防的事，而且事后完全看不出来**，
  因为版本字符串本身没有任何异常。
- **`test_a_retry_still_passes_through_the_quality_gate`**：幂等**省掉的是写入，不是检查**；
  DQ 不过则 REJECTED 且零写入。
- **`test_an_empty_quality_run_never_satisfies_a_retry`**：**"没跑检查"不等于"检查通过"**
  （`can_publish` 对空列表返回 False，专测钉住）。
- **`test_a_rejected_batch_cannot_be_retried_into_existence`**：retry 是给**被中断的发布**续跑的，
  不是给**被拒绝的发布**翻案的。
- **`test_a_batch_that_was_never_validated_is_validated_by_the_retry`**：
  CREATED 正是"在 staging 与 validate 之间崩掉"后的常态，故 retry 必须能接住它。
- **`test_an_already_published_batch_may_be_retried_without_writing_again`**。
- **`test_a_different_version_is_a_real_publish_not_a_retry`**：**反向对照**——
  新版本必须真的写（`uploads` 与 `scripts` 都非空）。
  没有这一条，"retry 什么都不写"的测试可以靠**让所有发布都什么都不写**来通过。

### fixture 精度：又一次"测试先错、实现对"

第一版 `RetrySession` 对**任何** `count(*) as row_count` 查询都返回同一个行数，
于是 `test_a_different_version_is_a_real_publish_not_a_retry` **失败**，
报的是 "target contains orphan rows for data_version"。
**实现是对的**：真实的 `_current_version_row_count` 查询**按 `data_version` 过滤**
（`where data_version='<version>'`），而我的假会话不区分版本，
于是一个**全新的版本**看起来像是已经有孤儿行了。
**修正后的假会话按版本应答**，并在注释里写明理由：
**若对所有计数都答同一个数，retry 分支与首次发布就无法区分**——
那正是这组测试要分辨的东西。
**这是本轮第六次同一类教训**，且是第二次由我自己的 fixture 触发（续十的 `_registry_at` 是第一次）。

### 变异验证（3 个，各被其对应测试捕获）

| # | 变异 | 捕获它的测试 |
|---|---|---|
| 41 | retry 跳过行数一致性校验 | 同版本不同数据那条 |
| 42 | retry 跳过 DQ 门 | DQ 门那条 **+** 空检查集那条 |
| 43 | REJECTED 批次也允许 retry | 被拒批次不可 retry 那条 |

`db/publish.py` 三次变异后以 `git status --porcelain` 确认**还原干净**。

### C5.5 / C5.6 仍不勾：交付的是单元层覆盖，规格要的是 V2Test 集成故障测试

C5.5 的六种故障，按**当前实际覆盖**逐一点名：

| 故障 | 覆盖情况 |
|---|---|
| 源 schema 漂移 | **单元层已覆盖**（续九 C1.5 dry-run：表/列缺失、类型不可转换、表不可读三种） |
| 源水位中途变化 | **无任何覆盖** |
| 重复冲突 | **已覆盖**（C3 去重测试 + 本轮续十一的两条不变量） |
| `available > signal` | **已覆盖**（`tests/property/` 与 `tests/leakage/` 的 PIT 排序不变量） |
| publish 前进程异常 | **本轮起实质覆盖**（retry 的四种入口状态：CREATED/VALIDATED/PUBLISHED/REJECTED） |
| 相同版本重跑 | **本轮起覆盖**（含"同版本不同数据必须被拒"） |

**即六种里四种在单元层有了覆盖，此前是零到一之间**。
**但 C5.5 的标题是「V2Test 集成故障测试」**，而 `tests/integration/factors/` **目录仍不存在**
（与续十对 C1.6、续十三对 C5.5 的发现同因）。
**单元层证明的是发布器的状态机正确，集成层要证明的是它在真实 V2Test 上、
真实会话中断与真实水位变化下也正确——后者无法由前者代替。**
**故不勾**，并把"源水位中途变化"记为**唯一六种里仍零覆盖**的一项。

C5.6 同理：其第一条 Run（`pytest tests/unit/factors/test_materializer.py -q`）通过，
Expected 的前半「正常发布幂等」**本轮已在单元层钉住**，
后半「所有 blocker 均保持旧版本可见」由既有的
`test_database_publisher_does_not_write_when_dq_rejects_batch` 与本轮的 REJECTED 断言共同覆盖；
但**第二条 Run 指向的 `tests/integration/factors/test_factor_publish.py` 不存在**，故不勾。

**全量：1487 passed / 45 skipped**（本节起点 1479；+8 publish retry），
`ruff check src tests` 清、`mypy src` 158 文件清。
> **勘误 2026-09-06（续二十二）**：同上，`ruff` 半句写下时即不成立；`mypy` 半句准确。

## 勾选审计续二十一（2026-09-06 13:05）：C5.5 最后一种零覆盖故障已查清——「源水位中途变化」在**任何一层都不被检测**；新增决定 H

续二十把 C5.5 六种故障里四种补到了单元层，剩下「源水位中途变化」标注为**唯一零覆盖**。
本轮把它查清了，结论不是"缺一条测试"，而是**缺一条不变量**。

### 实测证据链

1. **发布器根本不看 lineage。**`grep source_watermark_id research_v2/src/research_v2/db/publish.py`
   → **零命中**。`V2BatchPublisher` 校验业务键唯一、月份分区归属、DQ 结果、版本与指针状态，
   **唯独不碰 `source_watermark_id`**。故一个**行来自两个不同源快照**的批次，
   与一个干净批次**发布路径完全相同**。
2. **批内混合当前不可能发生**——每个 materializer 都对整帧盖**同一个**水位
   （`materializers/market.py` 四处 `materialized["source_watermark_id"] = source.watermark_id`、
   `materializers/labels.py`、`materializers/factor_evaluation.py::_append_lineage`）。
   **所以暴露面不在批内，而在批间**：同一个对象的两个月、在同一个 `data_version` 下发布、
   一个在源变动前读、一个在变动后读。**每一批内部都是一致的，故任何按批的检查都看不见它**，
   而这个版本最终装着**混源的行**。
3. **`RunContext.build` 把水位折进 `data_version`，但折的是一个无键的集合。**
   `payload["watermarks"] = sorted(set(watermarks))`，故**源变动确实会改变版本号**
   ——这是要紧的一半：读过移动源的 run **不会与干净 run 的身份相撞**。
   但集合**不记录每个水位来自哪个源**，且**重复被折叠**。
   于是版本号能回答"涉及了这组水位"，**不能回答"是同一张表中途动了，还是我本来就读了几张表"**。
4. **回填核验器查的是版本字符串，不是水位列。**
   `scripts/verify_factor_daily_backfill.py` 重算出**预测的** `data_version`
   并断言观测到的版本集合等于它（第 126 行），
   但**没有**断言已发布行的 `source_watermark_id` 列是单值的。

**合起来**：一次源中途移动会让 run **正常完成、正常发布**，产出一个**与干净 run 不同**的
`data_version`，其中装着**混源的行**，而**没有任何一层报错、也没有任何一层记录"哪张表在中途动了"**。
指针照常推进，于是混源的行成为**正式可见版本**。

### 两条钉住现状的测试

- **`test_a_batch_carrying_two_source_watermarks_publishes_without_comment`**：
  构造一个 `source_watermark_id` 列**含两个不同值**的批次，断言它**发布成功**、
  上传帧里水位**确实有两个**（fixture 前提）、且**指针照样推进**。
- **`test_run_identity_folds_watermarks_into_an_unkeyed_sorted_set`**：
  钉住 `data_version` **对水位顺序与重数不敏感**（`["w1","w2"] == ["w2","w1"] == ["w1","w1","w2"]`），
  而**换一个或加一个水位确实改变版本**（`["w1","w3"] != baseline`、`["w1","w2","w3"] != baseline`）。
  **两个方向都要断言**：只断言"不敏感"会让"版本号根本不包含水位"的实现通过。

**变异验证**：
**M44 = 直接把缺失的不变量加上去**（在 `StagedBatch.validate` 里拒绝水位不唯一的批次）
→ 第一条测试**失败**。**这正是 characterization test 应有的行为**：
它钉住现状，使得日后补这条不变量时**必须显式改动一条测试**，从而是一次有意识的决定。
**M45**（`RunContext` 改为按有序 list 而非 sorted set 哈希）→ 第二条测试失败，
**并同时使既有的 `test_data_version_is_stable_for_equivalent_orderings` 失败**，
说明那条既有测试也在看守同一性质。
`db/publish.py` 与 `runs/context.py` 变异后均确认**还原干净**。

### 不实施的理由：不变量该放在哪一层是个设计决定（新增决定 H）

三个候选位置，各自的能力不同：

- **(H-i) `StagedBatch.validate` 里加批内检查**：**改动最小，但抓不到真正的暴露面**。
  批内混合当前根本不可能发生（见证据 2），故这条检查会对所有现存批次静默通过，
  而**批间混合照旧**。**它给人一种已经防住了的错觉**——这正是本轮反复遇到的失效模式。
- **(H-ii) 发布器记住该 `data_version` 已写过的水位，跨批比对**：能抓住批间混合，
  但需要**发布器持有跨调用状态**（或查已发布行），
  而 `V2BatchPublisher` 现在是无状态的、只靠注入的 pointer reader 感知外部。
- **(H-iii) 发布后核验器断言"一个 `data_version` × 一个对象 × 一张源表 ⇒ 一个水位"**：
  **能抓住，且不改动发布路径**，位置与 `verify_factor_daily_backfill.py` 已有的
  "单一 `data_version`"核验同类，是**最贴合现有结构的一处**。
  代价是**事后发现而非事前阻止**——混源的行会先落库，再由核验器拦下不推进指针。

**倾向 (H-iii)**，但**不代做**：它要么新增一条能让发布失败的拒绝路径（(H-ii)），
要么新增一个必须被编排器调用的核验步骤（(H-iii)）——
**两者都改变运行链行为，而 Task 2.4 正在逐位复现**（本节写作时已 15h57m）。
**已作为决定 H 写入决策清单。**

**全量：1489 passed / 45 skipped**（本节起点 1487；+2 watermark），
`ruff check src tests` 清、`mypy src` 158 文件清。
> **勘误 2026-09-06（续二十二）**：同上，`ruff` 半句写下时即不成立；`mypy` 半句准确。

## 勾选审计续二十二（2026-09-06 15:55）：C3.4 已勾（出处是实测）；代码评审 18 条逐条处置；冻结 17 的六格普查推翻了三处既有陈述

本节三条线并行：**(1)** Task 2.4 落地后立刻跑通 C3.4 缺的那份输入并交付 fixture；
**(2)** 处理上一轮请求的代码评审（`130934c..33636ab`，18 条发现，逐条核实后处置）；
**(3)** 评审要求的两项"补证据"（可复现性核验器重跑、相似度成本测量）。
**未改动任何已物化数据，未推进任何版本指针**，`/tmp/frozen_chain_hold.txt` 仍在（裁决 4 未定）。

### 一、C3.4 已勾：出处换成了第一手测量，fixture 里没有一个值来自源

续十一 留下两条出路，并刻意不做 (ii)，理由是不与逐位复现争源库。**2.4 已于 13:16 落地 CONSISTENT**，
该理由消失，故本轮走 (ii)：新增 `scripts/census_duplicate_key_shapes.py`，
对源做**有界只读普查**，再据实生成 fixture。

**三条不可让的性质**（都有测试钉住，见 `tests/unit/scripts/test_census_duplicate_key_shapes.py`，18 条）：

1. **只回聚合、绝不回行**：每个查询以"对按键聚合的表做一次单行 select"收尾，
   故 2700 万行的因子只花一行网络流量。测试断言**每个上报指标的表达式都以聚合函数开头**——
   把其中一个改成裸列（M52）会让"不可能返回行"这条测试失败，这正是它存在的意思。
2. **每因子一次查询**：上一轮把 `factor_data8` 的 5 个因子批在一条查询里**打爆了服务端内存**；
   改成每因子一条后同一张表跑通。仍保留**按年回退**：年份划分 (ts_code, trade_date) 键空间，
   故所有计数可加、**唯独最大重数必须取 max 而非求和**（M51 只让回退那条测试失败）。
3. **不信任 `nunique`**：见第四节。

**fixture**（`tests/unit/factors/data/dedup_regression_gp017_gp077_v1.json`，48,657 字节，
每因子 200 个采样键）：六类键的**占比按最大余数法**从实测比例取整、重数用实测值、
组内"可用行数/相异值数"用实测均值；**业务键与数值全为合成**。
它记录所依据普查工件的 sha256，而第一条测试就拿这个哈希去比**仓库里那份工件**——
改任一边而不同时改另一边即失败（M60 只让这一条失败）。
生成器 `scripts/build_dedup_regression_fixture.py` **拒绝发明它没测到的形状**：
重数若不均匀（如每键 2.5 行）直接报错，而不是四舍五入成一个数——
那正是 C3.4 之所以宁可不勾也要避免的"看起来测过其实没测"。

**C3.5 仍不勾**：它的 Expected 是"三类 resolution 和全部不变量通过"，
命令实跑**20 passed**（`tests/unit/factors/test_deduplication.py` +
`tests/property/factors/test_deduplication_invariants.py`），**但 C3.1 的两处规格偏差仍开着**
（决定 G），故 Task 级收口项不应先于其子项勾选。**与续十一的判断一致，未变。**

### 二、冻结 17 的六格普查：四种形状，且**不是按表分的**

"六格"= 去重器对一个业务键**只可能做的六件事**，与普查的六个计数一一对应。
`check_partition` 断言六格之和等于该次普查自己的 `key_count`，
**合成数据与真实数据都查**（18 个因子零违例）——一份六格不闭合的普查在测别的东西，
而据它造的 fixture 会把错误当成证据带进测试套件。

| factor_id | 名称 | 表 | 键数 | 重数 | 唯一可用 | 唯一不可用 | 重复·全同 | 重复·单非空 | 重复·冲突 | 重复·全不可用 | naive 冲突 | 无可用值占比 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| yeslab_206 | 60日成交量去均值相对偏离 | data5 | 9,552,130 | — | 7,433,313 | 2,118,817 | 0 | 0 | 0 | 0 | 0 | 22.2% |
| yeslab_357 | 行业内收益标准化绝对值 | data8 | 8,825,600 | 2 | 360,566 | 42,634 | 7,312,137 | 0 | 0 | 1,110,263 | 0 | 13.1% |
| yeslab_372 | GP_003 | data8 | 9,589,250 | 2 | 299,375 | 338,067 | 3,528,553 | 0 | 0 | 5,423,255 | 0 | 60.1% |
| **yeslab_375** | **GP_006** | data8 | 9,457,875 | **3** | 106,688 | 399,379 | **0** | 3,262,169 | **1,721,876** | 3,967,763 | 4,984,045 | 46.2% |
| yeslab_381 | GP_012 | data8 | 9,588,877 | 2 | 612,369 | 24,700 | 7,322,304 | 0 | 0 | 1,629,504 | 0 | 17.3% |
| yeslab_384 | GP_015 | data8 | 9,589,250 | 2 | 299,426 | 338,016 | 3,591,736 | 0 | 0 | 5,360,072 | 0 | 59.4% |
| yeslab_389 | GP_020 | data9 | 9,588,937 | 2 | 294,978 | 342,151 | 3,655,091 | 0 | 0 | 5,296,717 | 0 | 58.8% |
| yeslab_390 | GP_021 | data9 | 9,592,597 | 2 | 304,785 | 332,996 | 3,637,978 | 0 | 0 | 5,316,838 | 0 | 58.9% |
| yeslab_398 | GP_029 | data9 | 9,592,597 | — | 7,187,066 | 2,405,531 | 0 | 0 | 0 | 0 | 0 | 25.1% |
| yeslab_405 | GP_036 | data9 | 9,590,797 | — | 7,934,673 | 1,656,124 | 0 | 0 | 0 | 0 | 0 | 17.3% |
| yeslab_406 | GP_037 | data9 | 9,591,033 | — | 3,697,315 | 5,893,718 | 0 | 0 | 0 | 0 | 0 | **61.5%** |
| yeslab_425 | GP_056 | data9 | 9,592,756 | — | 7,883,970 | 1,708,786 | 0 | 0 | 0 | 0 | 0 | 17.8% |
| yeslab_464 | GP_095 | data10 | 9,594,514 | — | 7,371,584 | 2,222,930 | 0 | 0 | 0 | 0 | 0 | 23.2% |
| yeslab_471 | AttentionSpillover | data10 | 9,635,152 | — | 7,392,559 | 2,242,593 | 0 | 0 | 0 | 0 | 0 | 23.3% |
| yeslab_484 | 羊群效应_CSAD波动率21 | data10 | 9,482,202 | — | 7,743,517 | 1,738,685 | 0 | 0 | 0 | 0 | 0 | 18.3% |
| yeslab_96 | 修正日间反转 | data3 | 9,552,140 | 2 | 733,192 | 91,558 | 7,055,910 | 0 | 0 | 1,671,480 | 0 | 18.5% |
| yeslab_386 | GP_017（**不在冻结集**） | data8 | 9,457,875 | **3** | 102,870 | 403,197 | **0** | 4,097,870 | **3,184,961** | 1,668,977 | 7,282,831 | 21.9% |
| yeslab_446 | GP_077（**不在冻结集**） | data10 | 9,665,247 | 2 | 199,149 | 503,823 | **0** | 6,764,199 | **0** | 2,198,076 | 6,764,199 | 28.0% |

`yeslab_443_reg30_max20_v1` **未普查**：`factor_metadata` 里没有它的行（它是修复变体，不是原始源因子），
已在工件里以 `NOT_CENSUSED` + 理由记录，**不静默跳过**。

**四种形状，按因子而非按表分布**：

- **8 个因子零重复**（206、398、405、406、425、464、471、484）。
- **7 个因子是纯精确重复**（357、372、381、384、389、390、96）：重数恰为 2，
  重复键**要么两行完全相同、要么两行都是 NULL**，`conflict = single_nonnull = naive = 0`。
  **`factor_data9` 内部同时存在"重复"与"零重复"两类**（389/390 重复，398/405/406/425 不重复），
  故重复是**装载批次级**现象，不是表级、更不是因子语义级。
  另注：`dup = 8,951,808` 这个数在 372/375/381/384/389 五个因子上**完全相同**（跨 data8 与 data9），
  即被重复写入的是**同一批 (日期, 代码) 键**。
- **只有 1 个冻结因子有真冲突**：`yeslab_375`（GP_006），**1,721,876 键（占 18.2%）**的三行**互不相等**。
  这些键被去重器整组隔离，故 GP_006 有 18.2% 的 (日期, 代码) **发布不出任何值**——
  **不是源没说话，而是源自相矛盾**。这与"覆盖稀疏"是两种不同的缺陷，现在按因子分开测到了。
- **GP_006 与 GP_017 是同一形状**：重数都是 3、`重复·全同` 都是 0、
  且 `key_count` 都是 **9,457,875**、`unique_key_count` 都是 **506,067**（逐位相同）。
  这与准入结论 `REJECT_REDUNDANT_WITH_CANDIDATE_375`（386 因与 375 冗余被拒）互相印证：
  两者出自同一次生产运行、共享同一键宇宙，只是值的自洽程度不同。

**一个此前未测、且与去重无关的覆盖率成因**：**13.1%–61.5% 的键"源里有行但值不可用"**。
`yeslab_406`（GP_037）达 **61.5% 且零重复**——这部分**任何去重工作都救不回来**。
Task 3.1b 把冻结 17 的覆盖率缺口归因于 universe 从 ~2,700 增至 ~4,300；
本次测量给出**第二个、可加的成因**：源键本身的 NULL 率。两者应分开记账，
因为对策不同（前者是口径问题，后者是源数据问题）。

### 三、这次测量**推翻或收窄**了三处既有陈述

**(a) `deduplicate_factor_observations` 的 docstring 把"每个键都有一行完全相同的重复"当作 YesLab 常态——对 C3.4 点名的两个因子，实测为 0。**
GP_017 的 `重复·全同` = **0 / 8,951,808**，GP_077 同样为 0，GP_006 也为 0。
**该分支并非死代码**（上表 7 个冻结因子除了它没有别的形状），
但"常态"这个词对这三个因子不成立：它们的重行是**"一个值 + 一到两个 NULL"**。
已由 `test_gp017_has_no_identical_duplicate_at_all_which_contradicts_the_docstring` 连数字一起钉住。
**未改 docstring**：改它等于改一处对源的总体断言，而本轮只有 18 个因子的证据；
**留给 owner 连同决定 G 一并定**（见第七节）。

**(b) 续十一 记录的 `[Infinity]` 非法 JSON 缺陷：可达性比当时写的窄得多——它到不了源读取路径。**
续十一 写的是"经公开 API 实测可达"，这话没错，但容易被读成"源数据会触发它"。实测否证了后者：

- **DolphinDB 的 DOUBLE 列根本存不下 ±inf**：上传 `inf`/`-inf`/`1e309` 读回来**都是 NULL**；
  所有算术通路（`1/0`、`log(0)`、`exp(1000)`、`pow(10.0,400)`、`1e308*10`）**也都返回 NULL**。
- 活的 `value` 列在 `factor_data1..11` 上**全是 DOUBLE**（C1.5 实测的 typeString）。
- reader 对 `value` **不做任何算术**（只有 `pd.to_numeric`）；
  对 `source_reader.py`/`materializer.py`/`deduplication.py` grep 算术运算符与会上溢的 numpy 函数**零命中**。

**故 `raw_value` 在任何真实读取路径上都不可能是 ±inf**；`"[Infinity]"` 只能由**进程内手工构造的帧**产生。
**这不降低缺陷的真实性**（列格式确实违反 RFC 8259，`invalid_reason` 确实被丢掉），
但**改变它的定级**：它是"潜在的 API 契约缺陷"，不是"正在发生的源数据缺陷"。
**已写入决定 F**（见第七节），并把"是否仍值得改 `record_hash`"这个权衡交回 owner。

**(c) 上一轮交接摘要里"GP_077 = 纯精确重复（0% 冲突）"——冲突数对，形状错。**
GP_077 的 `重复·全同` 是 **0**；它的 8,962,275 个重复键是
**6,764,199 个"一个值 + 一个 NULL"** 加 **2,198,076 个"两个 NULL"**。
"纯精确重复"会让人以为 `IDENTICAL_COLLAPSE` 是它的主路径，而实际主路径是 `SINGLE_NONNULL`。
**这个区别有后果**：续十一 的 C3.1(b) 偏差（`[1.0, 1.0, NaN]` 也被标 SINGLE_NONNULL）
对 GP_077 **不适用**——它的真实形状是 `[v, NULL]`，**恰好符合规格**的"恰有一个有限非空值"。
即 C3.1(b) 是**标签不精确**，而**在实测到的真实数据上它一次也没有不精确过**。

### 四、两条平台事实（都是实测，都改变了查询本身）

**事实 1：`nunique` 把 NULL 计为一个取值。** `nunique([1.0, NULL, 2.0, 1.0]) == 3`，
`nunique([NULL, NULL]) == 1`（实测）。故朴素判据 `nunique(value) > 1` 会把
"重复行全是 NULL"和"一个值加一个 NULL"都报成值冲突。修正法是**当组内存在不可用行时减掉 NULL 桶**。
**代价是具体的**：GP_017 朴素 **7,282,831（77.0%）** vs 修正 **3,184,961（33.7%）**；
GP_077 朴素 **6,764,199（70.0%）** vs 修正 **0**。
上一轮把这两个朴素数字写成"会报 77%/70%"是**推算**，本轮是**实测**，且与推算一致。

**由此得到一条可校验的恒等式**：`naive_conflict == 真冲突 + 单非空`，**精确成立**。
理由：`nunique` 只多加一个 NULL 桶，故只有"≥1 个可用值且≥1 个 NULL"的组会被多算；
全 NULL 组、全同组、单行键的 `nunique` 都是 1。
**在全部 18 个因子上零违例**（GP_006：4,984,045 = 1,721,876 + 3,262,169；
GP_017：7,282,831 = 3,184,961 + 4,097,870；GP_077：6,764,199 = 0 + 6,764,199）。
**这把"修正在 8 行合成数据上验过"升级成"修正所依赖的 NULL 模型就是 DolphinDB 实际实现的那个"**，
已作为 `check_partition` 的第五条不变量写入（并对合成自测同样成立）。

**事实 2：DOUBLE 列存不下 ±inf，且 `abs(NULL) <= 1e308` 求值为 True。**
第二条是**关键**：它意味着一个 `abs(value) <= 1e308` 的"无穷防护"**永远不可能排除任何行**——
它只会**看起来像**在过滤无穷。本轮先写了这个条款，自测报出 naive 计数与手算不符，
追查后确认该条款是**死条款**，遂**删除**并把"可用"简化为 `isValid(value)`（等价，已实测）。
**这与本项目反复抓到的失效模式同源**：一条不会失败的检查比没有检查更糟，
因为它让读的人以为防住了。

**自测与交叉验证**：`--scope self-test` 用 8 个手算形状（含两个陷阱：值+NULL、意图为 ±inf 的两行）
在**内存表**上验证全部 18 个指标，**逐项与手算一致**；
`--scope crossval` 重测 GP_017/GP_077 并**要求复现上一轮记录的数字**——
`key_count`/`row_count`/`duplicated_key_count`/`max_multiplicity`/`conflict_key_count`/`no_usable_value_key_count`
**六项全部精确复现**，且判定出上一轮那个"无可用值"计数用的是**全键分母**（不是重复键分母）。
**这是让冻结集普查与它可比的前提。**

### 五、代码评审 18 条的逐条处置（`130934c..33636ab`，24 文件 6,491 行）

评审先做了独立复核（211 条测试正反序各跑一遍、62 条在 4 个 `PYTHONHASHSEED` 下重跑、
`mypy` 逐条对、三个已记录缺陷**读代码**确认而非采信文书），**Critical 0 条**。
下表是逐条处置，**核实过才写"属实"**：

| # | 级别 | 发现 | 处置 | 证据 |
|---|---|---|---|---|
| 1 | Important | `_entry_key` 对非整数 `factor_version` 抛裸 `ValueError`，绕过 CLI 的 `except FactorPlanError`，命令 exit 1 且**输出为空** | **已修，且比建议更进一层** | `aa92fcd`；M46–M49 |
| 2 | Important | `ruff check src tests` **不清**，而四节审计与交接都声称清 | **已修 + 四处勘误** | `aa92fcd`；本节第六部分 |
| 3 | Important | 交接决策清单 §0/§5 **内部陈旧**（"四处"vs 自家表格的"六处"等 5 项） | **已改** | 本次提交 |
| 4 | Important | C1.1 引用的可复现性工件（09:24–09:28）**早于**它要验证的通配符防护（09:43） | **已重跑，离线 + 活库各一次** | 见下 |
| 5 | Important | `test_a_reordered_catalog_is_still_reproducible` 的名字与 `main()` 的判决相反 | **已修，但用评审的 (b) 而非 (a)，理由具体** | `e738941`；M54–M56 |
| 6 | Minor | `schema_errors` 被 CLI 加载后**在传给规划器前丢掉**，strict 拒绝信息只剩 "unknown schema query failure" | **已修，并补接线测试** | `806b4a5`；M57/M58 |
| 7 | Minor | `estimated_rows` 是稠密上界但未标注 | **已补 docstring，并给出实测误差量级** | `1d0af11` |
| 8 | Minor | 类型白名单漏了若干真实 DolphinDB 类型 | **不改，已把"故意"写进注释** | `1d0af11` |
| 9 | Minor | `_LIVE_TYPES` 在 4 个测试文件里逐字重复 | **不改，理由：它是自我看管的** | 见下 |
| 10 | Minor | 两处 `# noqa: E402` 的文件中部 import | **不改**（纯外观，改动要碰既有测试文件） | — |
| 11 | Minor | 两条 CLI 测试只断言 `exit_code != 0` | **已修，并多修一条同病的** | `43cef77` |
| 12 | Minor | CLI 测试依赖 click ≥ 8.2 的流分离，而项目未声明 | **已声明 `click>=8.2`（dev extra）** | `1d0af11` |
| 13 | Minor | 一条测试扫全部字符串常量，与散文耦合 | **不改，理由具体** | 见下 |
| 14 | Minor | 审计脚本**手工重建**它本该复用的查询文本 | **已改为调用 `_schema_query`** | `1d0af11` |
| 15 | Minor | 一条既有测试的名字**声称的范围大于它钉住的范围** | **已改名** | `1d0af11` |
| 16 | Minor | `compare()` 的 diff 只按 `factor_id` 建索引，多版本会误归因 | **驳回，附证据** | 见下 |
| 17 | Minor | 两个新脚本沿用了仓库既有的硬编码口令/主机默认值 | **不改，已记账**（本轮又加 3 个脚本沿用同一约定） | — |
| 18 | Minor | `compute_factor_similarity` 的成本**从未被测过** | **已测，并给出拟合与外推** | 见第七节 |

**#1 为什么要比建议更进一层。** 评审建议"给 `_entry_key` 加同样的 try/except，
于是 `_check_entry` 里既有的处理就可达了，会产出一条具名 BLOCKED finding"。
**核实后发现后半句不成立**：`_check_entry` 既有的 try/except **只是把版本静默强制成 0，并不追加任何 reason**。
故只修崩溃的话，`factor_version: v1` 会被报成 **MAPPED / version=0**——
**比崩溃更安静**：`factor_version` 既是 finding 要报告的身份的一部分，
也是它与 `retired` 集合匹配的键。现在不可解析的版本是一条**具名 BLOCKED reason**。
**M46 就是"评审的字面修法"**（保留崩溃修复、去掉新增 reason）→ **5 条测试在三个层次上失败**，
这正说明新增 reason 是承重的。另：非整数浮点**拒绝而非截断**（`int(1.5)==1` 会静默指向另一个版本），
而**整数浮点仍通过**（`1.0` 是手改 YAML 常见的"1"写法）——两侧各有测试（M48）。

**#4 的重跑结果（两条路径都跑了）。** 离线（读冻结 parquet）与**活库**（查 `factor_metadata`）
**逐项相同**：487 行、snapshot hash `afb9cc39…`、487 stored / 487 rebuilt、
单向差集 0、逐条差异 0、`CATALOG_REPRODUCIBLE`、exit 0。
两份保存的 metadata 快照**逐帧相等**（487×5），故**冻结快照未过期**，
且"加防护后 hash 与条目不变"这句话现在**由一次晚于防护的运行支撑**，不再靠推断。

**#5 为什么选 (b) 不选 (a)。** 评审建议 (a)：让 `compare()` 的 `entries_identical` 对顺序不敏感，
"这才符合 docstring 的本意"。**核实后：(a) 修不好它要修的那个矛盾。**
`reproducible` 还要求 **snapshot hash 匹配**，而 hash 覆盖字节；
故即使 (a) 让条目比对通过，`main()` 对同一个重排文件**仍返回 1**，测试名字**仍然是错的**。
而 (a) 会让诊断**失去**说"这些是同一批条目、只是被挪动了"的能力。
所以按 (b) 做：测试改名为 `..._is_drift_because_the_gate_is_byte_exact`，
新增 `entries_identical_ignoring_order` 字段（**诊断，不是判决**），
`main()` 对"条目全同、仅字节不同"给出**专门的漂移信息**（不再让操作者去找一个根本没丢的因子），
并补 **3 条 `main()` 级测试**——**此前 `main()` 零覆盖**，
一个永远打印 `CATALOG_REPRODUCIBLE` 并返回 0 的 gate 能通过原来那 8 条测试的全部（M54 证实）。

**#16 驳回，附证据。** `build_yeslab_catalog` 在 `catalog.py:31-32` 对**重复或空的 `factor_id` 直接抛
`CatalogBuildError`**，且 `factor_version` 是**硬编码常量 1**（`catalog.py:55`）。
故 `rebuilt_entries` **可证明**每个 `factor_id` 至多一条、版本全为 1，
`by_id` 的折叠**不可达**；而在唯一可达的重复形态（同 id 同版本）下，
按 `(factor_id, factor_version)` 建索引**也不会更好**。
**这是不可达分支上的防御性改动**，与本轮刚删掉的那个 `abs(NULL)<=1e308` 死条款同类，故不做。

**#9 不改的理由。** `_LIVE_TYPES` 是对源的一处**观测事实**的转录，重复在 4 个测试文件里。
评审担心"源 schema 变了要同时改 4 处、且没有东西强制"。**但这四处都在断言活类型**，
故源一变**四条测试一起失败**——它是**自我看管**的重复，不会静默错。
提取共享 helper 是纯外观收益，且要动 4 个既有测试文件的 import 块。

**#13 不改的理由。** 那条测试扫**全部**字符串常量（含 docstring 与 argparse help），
评审指出将来一句散文里出现 "inserted" 就会误报。
**但它的失效方向是"测试假失败"，不是"漏放一个真的变更语句"**，且会指名 offending 字符串；
而它上面那条基于 AST 的检查才是强的那条、且不脆。
**把扫描范围收窄到"看起来像 DDB 脚本的字符串"会引入一个判断规则，而规则本身需要被测试**——
收益不抵新增的判断面。

**#17 记账。** 评审自己没把它算在这批变更头上（仓库既有 56 个文件同此约定）。
**本轮新增的 3 个脚本沿用了同一约定**，故该数字在增长。
**这是一次独立清理任务，不该夹在 C 计划里做**：改默认值会影响所有既有脚本的调用方式。

**评审的两条计划级意见，都不代改，转为决定/记账：**

- **C5.3 的条目文本把四条 CLI 命令和一个 dry-run 规划器捆在一款里**，导致审计要跨续十八/续十九拆分，
  且其中两条命令绑定决定 A。这是**计划的撰写问题**，不是实现问题——
  **条目文本的拆分权在 owner**，本轮不改计划正文（除勾选与勘误）。
- **计划里没有任何一款问"治理代码是否接上了生产路径"**。
  本轮所有审计中后果最重的一条发现——**六个组件零生产调用者**、
  生产用的冻结集是一份人工裁决的 JSON、其身份不是内容寻址哈希——
  **是 grep 出来的，不是任何验收标准要求的**。
  这六个组件**每一个都能完全满足自己的条目文本，而生产里没有任何东西用它**。
  评审建议的判据（"每个治理模块至少有一个生产调用者，或显式标注为设计参考"）
  **若在 C1 就有，决定 A 会在 C1 浮现而不是在 C8**。
  **已作为新增决定 I 写入交接清单，不代加进验收标准。**
- **C7.5 的"已知高相关因子"全部是合成 fixture，而 C3.4 曾因缺真实出处不勾**——
  同一个问题在两处被作了相反的决定且未注明。**已在续八就地注明**（见第六节），
  并给出可选的第二条路（从 `yeslab_436/446/468` 的日度同 RankIC 退化派生一条真实 fixture）。
  **本轮未造该 fixture**：它需要的是**相关性**证据而非重复证据，属 C7 问题域，
  且 C7.1/C7.2 的 scope 绑定仍在决定 E 上。

### 六、文书更正（评审 #2/#3/#4 的文书半）

1. **四处 `ruff` 声明**（续十八/十九/二十/二十一）**各加一行勘误**。
   **这不是"过时"而是"写下时即不成立"**：`ed7fb5a`（续十八自己那轮）引入的
   `tests/unit/factors/test_materialize_plan.py` import 块未排序，
   `ruff check --no-cache src tests` 报 1 处 `I001`，直到 `aa92fcd` 才修。
   **`mypy` 半句经复核准确**（158 文件，逐条对上）。
   **为什么这条要紧**：这些审计以"已实测"为信用基础，
   而"ruff 清 / mypy 清"是重复了十几次的常备断言；
   **其中一半是假的，会让另一半真话获得它不该有的信用**。
2. **续八就地注明 C7.5 的出处不一致**（见上）。
3. **交接清单 §0/§5 逐项对齐到续二十二**：
   "四处未接线"→**六处**（与其自家 §1 标题、表格一致）；
   来源范围"续七～续十二"→**续七～续二十二**；
   累计变异 **25 → 61**；测试自身缺陷 **四次 → 十次**；
   全量 **1431 → 1537**、mypy **157 → 158**；
   `ruff` 那句改为**如实**（曾不成立、现已清）。
   §5 另**显式标注它是截至续十三的快照**并给出当前值，避免"半新半旧"。

### 七、本轮新增/更新的裁决输入

- **决定 F（quarantine JSON 格式）**：**收窄**。`[Infinity]` **到不了源读取路径**（第三节 (b)）。
  缺陷仍真、仍违反 RFC 8259、`invalid_reason` 仍被丢，
  但**"正在污染审计轨迹"应改为"潜在契约缺陷"**。
  **权衡因此改变**：三个候选都会改 `record_hash`，
  而收益侧不再是"源里最坏那条记录读不出来"，而是"API 使用者构造出 inf 时读不出来"。
  **仍可独立随时定，但优先级应据此重估。**
- **决定 G（C3.1 两处去重偏差）**：**Task 2.4 已落地 CONSISTENT，"不得动"的前置解除**。
  但**顺序不变**：决定 B → 决定 C 仍应先于任何 as-of join 重写（评审亦持此见）。
  **新增证据改变了 (a)(b) 两款的实际权重**：
  (b)（`[v,v,NaN]` 也被标 SINGLE_NONNULL）**在实测的 18 个因子上一次也没发生**——
  真实形状是 `[v, NULL]`，**恰好符合规格**；故 (b) 是**纯标签问题、且当前无实例**。
  (a)（折叠只比 `raw_value`、存活者由最小 `source_row_id` 决定）
  **不需要普查来回答——读代码即可，答案是"当前结构性无值可挑"**：
  `source_available_date` **全仓只有一处写入**（`source_reader.py:102`）且**直接等于
  `observation_date`**，而 `observation_date` 是去重的四个业务键之一；
  技术路径的 PIT 步骤又赋一次（`availability.py:77`
  `result["available_date"] = result["observation_date"]`），
  经 `materializer.py:110` 写进发布列 `available_date`（migration 003 声明）。
  **故同一重复组内所有行的可用日必然相同**：续十一 那句"可用日是被溯源排序顺手挑中的"
  **对代码的描述成立，但候选之间在这一列上没有任何差别，挑谁结果都一样。**
  **它何时会变成真的**：`deduplicate_factor_observations` 的**唯一调用者**是
  `materializer.py:98`（技术路径）；基本面路径的可用日**从 `ann_date` 派生**
  （`availability.py:102-103` → `assign_fundamental_available_date`），与 `observation_date`
  **不同值**，而它**当前既未接线、也不经过这个去重器**。
  **故 (a) 款的生效条件是"决定 B/C 把基本面路径接进这个去重器"**——
  这是 **B/C 与 G 之间一条此前没被写下的耦合**：同一报告期的两个修订若带着不同 `ann_date`
  落到同一业务键，(a) 款当场从"无值可挑"变成"可用日由溯源排序决定"。
  **结论：决定 G 的两款在当前生产路径上都是零实例**，
  其优先级应据此下调；**而它重新变要紧的时刻由 B/C 决定，不由 G 自己决定。**
- **决定 E（C7.1 scope 绑定 / C7.2 覆盖率分母）**：**新增成本上限**（评审 #18）。
  `compute_factor_similarity` **零生产调用者**，实测标度律为
  **`t = pairs × R × (1.7457e-05 + 1.2547e-07 × F)` 秒**（R = 每因子 天数×股票数），
  在 F=8/16/24 三个测点上**四位有效数字吻合**；
  两项分别是"每对因子自己的 merge + 逐日 rank/corr"（b 项）与
  "两次全帧 `astype(str)` 掩码"（a·F 项，F≈140 以上才反超）。
  实测每行 **187 字节**（`ts_code`/`factor_id`/`signal_date` 都是 object dtype），
  而函数会 `copy(deep=True)`。外推：

  | 场景 | pairs | 行数 | 时间 | 输入内存（未计深拷贝） |
  |---|---|---|---|---|
  | C6.2 通过的 11 个，2020–2024 | 55 | 57,516,800 | ~1.5 h | ~10.0 GiB |
  | 冻结 17，2020–2024 | 136 | 88,889,600 | ~3.9 h | ~15.5 GiB |
  | 冻结 17，2014–2025 | 136 | 213,305,800 | ~9.3 h | ~37.2 GiB |
  | 全 catalog 487，一年 | 118,341 | 508,866,300 | ~2,698 h（112 天） | ~88.6 GiB |

  本机 629 GiB 内存 / 521 GiB 可用 / 104 核，**故约束是时间而非内存**：
  冻结 17 尺度**可跑**（数小时），**全 catalog 尺度不可跑**（112 天，且函数是串行循环、无并行）。
  **外推的已知偏乐观处**：两项拟合是在 **N ≤ 48,000 行**上测的，
  而 88.9M 行时掩码项受内存带宽支配、**大概率差于线性外推**，故 3.9 h 是**下限**。
  **两处结构性改法已看出但未做**（无生产调用者，且接线属决定 E/A）：
  把 `astype(str)` 提到循环外可去掉 a·F 项；
  按因子分组一次而非每对掩码全帧可去掉 b 项的大部分常数。
- **新增决定 I**：是否给 C 计划补一条验收标准——
  **"每个治理模块至少有一个生产调用者，或显式标注为设计参考"**。
  理由见第五节末。**这是计划撰写权的决定，不代加。**
- **Task 3.1b（覆盖率）**：**新增第二个成因**（源键 NULL 率 13.1%–61.5%），
  与既有归因（universe 增长）**可加、对策不同**。
  **并新增一个具体缺口**：GP_006 有 **18.2%** 的键因**源自相矛盾**而被整组隔离，
  这部分**既不是稀疏也不是口径**，只能靠"冲突如何解"的规则解决——
  而该规则**目前不存在**（C3.1 明确不选 first/last）。

### 八、变异验证（本节 16 个，M46–M61，全部被预期测试捕获）

| # | 变异 | 被谁抓住 |
|---|---|---|
| M46 | **评审 #1 的字面修法**：保留崩溃修复、去掉新增 reason | 5 条（catalog / planner / CLI 三层） |
| M47 | `entry_key` 恢复抛异常 | 7 条 |
| M48 | 非整数浮点改为静默截断 | 仅"非整数被拒"那条 |
| M49 | 去掉"缺失/空"豁免（空值也报不可解析） | 仅"缺失 vs 不可解析"那条，**且只经其新增的否定断言** |
| M50 | `check_partition` 永远返回 `()` | 5 条 |
| M51 | 跨年合并时把最大重数**求和**而非取 max | 仅回退那条 |
| M52 | 一个指标从聚合函数改成裸列（会开始返回行） | 仅"不可能返回行"那条 |
| M53 | 去掉表名/因子名的注入防护 | 4 条不安全输入全部 |
| M54 | gate 永远打印 `CATALOG_REPRODUCIBLE` 并返回 0 | 2 条 `main()` 级测试 |
| M55 | 去掉"仅顺序不同"的专门诊断 | 仅顺序那条 |
| M56 | 顺序无关比对恒为 True | 仅"手改一个字段"那条，**且只经其新增断言** |
| M57 | 规划器有参数、**CLI 不传** | **仅 CLI 接线那条** |
| M58 | 把一句像样的原因**硬编码**而不读 `schema_errors` | 具名那条 + 其控制 |
| M59 | 改一个组里的**一个值**、不改它的类别标签 | 3 条（统计 / 行会计 / GP_017 形状） |
| M60 | 改动仓库里那份普查工件的**一个计数** | **仅溯源那条**（sha256 链） |
| M61 | 让所有组共用一个业务键 | 6 条 |

**M49 与 M56 值得单独记**：两者都**只被一条"新增的否定断言"抓住**。
写下那两条断言之前，两个变异都能存活——
即**只断言正向（"报了缺失"／"忽略顺序后相同"）不足以钉住语义边界**。

**M57 是本轮最该记的一条**：它对应本项目**后果最重的发现**（六个组件有测试、零生产调用者）。
给规划器加一个参数、而唯一的生产调用者不传它，
**会让所有规划器级测试全绿**，而操作者照旧读到 "unknown"。
**接线本身必须有测试**，否则"已修"只是"已写"。

### 九、本节自查：四处我自己的断言先错了，都是被运行抓住的

延续 §5 的判据（**一条测试是否有效，看被破坏的实现下它是否给出相反结论**），
本轮同类问题**四次**，均已改断言/期望而非放过：

1. **`conflict_key_count_naive` 手算为 3，实测为 2**：我把 ±inf 当成两个相异值，
   而**平台根本存不下 inf**。**自测抓住了它**——这正是自测存在的理由；
   追查后不仅改了期望，还**删掉了那条死防护条款**（第四节事实 2）。
2. **分区参数化测试断言 `len(violations) == 1`，而扰动一格会同时破坏总数**，实际报 2–3 条。
   改成**每例只隔离一条违规**（同时抬高总数），四条各自只报一条。
3. **`read_metrics` 的长度断言写成 "want 16"，实际 15 个指标**（后增至 18，
   那条钉住数量的断言**如设计般失败**，提醒新指标必须被手算进自测与分区）。
4. **空 calendar 的 CLI 测试断言整条路径出现在输出里**，
   而 click 用 rich 边框渲染、**长路径被硬折行**，故整串匹配不上。改为只断言层次判别。

**另有一次不在测试里、但同源**：相似度成本我**先按立方模型外推**（用了中位常数），
而两列标度里**平的那列是二次模型**（波动 1.36 倍 vs 14.72 倍）。
**在看清哪列平之前就报数字，等于报了个模型错的数**；已按两项拟合重做并标注外推偏乐观处。

### 十、测试与门槛

**全量：1537 passed / 45 skipped**（本节起点 1489；+48：9 版本解析 / 3 schema_errors /
3 verifier 判决级 / 18 普查 / 15 fixture），于 `7f1f7e6` 时实测。
其后 `1d0af11` **只改 docstring、注释、一个测试名与一处依赖声明**（无行为改动），
已单独实测 `tests/unit/{factors,cli,scripts}` **942 passed**。
`ruff check src tests` 清（**本轮 3 个新脚本单独查亦清**）、`mypy src` **158 文件清**。
**`scripts/` 不在项目 lint 门内**（门是 `ruff check src tests`），
故其历史 555 处告警与本节无关；**本轮新增脚本仍逐个查过**。
