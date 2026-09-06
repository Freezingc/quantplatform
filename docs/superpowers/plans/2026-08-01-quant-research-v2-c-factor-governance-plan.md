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

- [x] **C8.5 提交并请求审查**

```bash
git add research_v2/src/research_v2/factors/factor_sets.py research_v2/tests/unit/factors/test_factor_sets.py research_v2/tests/golden
git commit -m "feat(v2): freeze content-addressed factor sets"
```

随后使用 `superpowers:requesting-code-review`，重点审查重复冲突处理、公告可用日、staleness、scope 边界和 factor set 不可变性。

> **已交付（2026-09-06 16:40，详见续二十三）**：审查已按该款指定的技能发出并**已回**，
> 覆盖 `130934c..33636ab`（24 文件 / 6,491 行），**Critical 0 条、Important 5 条、Minor 13 条**，
> 18 条已在续二十二逐条处置。五个重点区逐项对应：重复冲突处理（C3.2 判定 + 六格普查 + fixture）、
> 公告可用日（`availability.py:118-122` 的排序缺陷经读码确认 → 决定 B/C）、
> staleness（隔离 `>` 与 `>=` 的 179/178 测试）、scope 边界（C7.1 绑定 → 决定 E；
> 只读与白名单保证经结构性核验）、factor set 不可变性（C8.1/C8.2 判定，
> 并查出 `create_factor_set` docstring 的虚假保证——续二十三已修）。
> **一处如实说明**：该款示例的 `git commit -m "feat(v2): freeze content-addressed factor sets"`
> **没有逐字执行过**，因为 `factor_sets.py` 在 C 计划核验开始之前就已在树里
> （`44b7a46`），其测试补在 `c2950fd`；评审范围里它**未被修改**，
> 评审对此明确说明"五个重点区是被审计而非被改动"。**该款要的是"已提交 + 已请求审查"，
> 两者都成立。**

## C 计划完成标准

- [ ] 所有启用因子都有显式、版本化、不可变 registry 定义；源新增列不会被自动纳入。
- [x] 已知 GP_017/GP_077 类重复问题被统计，冲突键全部隔离，clean 表唯一。

> **已满足（2026-09-06 18:00，详见续二十六）**：续十四判本条为"三分之二"，
> 缺的正是**「重复问题被统计」**（当时 C3.4 缺输入）。该款已由本轮普查交付：
> GP_017/GP_077 **加全部冻结 17** 的六格分布（18 个指标、分区自检、
> `naive == 真冲突 + 单非空` 恒等式在 18 个因子上零违例、crossval 逐项复现既有记录）。
> 另两款此前已满足且**测试仍在**：「冲突键全部隔离」
> （`test_a_conflict_never_leaves_a_partial_group_behind_in_clean`、
> `test_no_business_key_appears_in_both_clean_and_quarantine`，
> 以及**按 GP_017 实测形状**构造的
> `test_a_conflicting_key_contributes_no_row_to_clean_and_keeps_all_of_its_rows_in_quarantine`）、
> 「clean 表唯一」（`test_the_clean_output_is_unique_on_the_business_key`，
> 其 docstring 就写着"Completion criterion: clean 表唯一"）。
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

上表的纯文本版另存 `/data/zhangyudong/qrv2_research_artifacts/frozen17_six_cell_table.txt`；
**权威数据是入库的 `duplicate_key_shape_census_v3.json`**（fixture 的 sha256 指向它），
表格只是它的渲染。

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
  **原始测量存于** `/data/zhangyudong/qrv2_research_artifacts/factor_similarity_scaling.txt`
  （与 C4.3 的 `fundamental_expansion_scaling.txt` 同一处置：测量脚本不入库，数字入文书）。
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

## 勾选审计续二十三（2026-09-06 16:45）：C8.5 已勾；三处"读起来像保证"的地方各补一条绊线；勘察全部 19 项未勾项后结论是——C 计划已由**决定**约束，不再由**实现**约束

### 一、C8.5 已勾

该款要两件事：把 C8 的产物提交、并用 `superpowers:requesting-code-review` 请求审查（重点五区）。
**两件都成立**：审查已发出并**已回**（`130934c..33636ab`，24 文件 / 6,491 行，
Critical 0 / Important 5 / Minor 13），18 条已在续二十二逐条处置。
**一处如实说明**：该款示例的提交命令**没有逐字执行过**——`factor_sets.py` 在 C 计划核验
开始之前就已在树里（`44b7a46`），其条款测试补在 `c2950fd`；评审范围里它**未被修改**，
评审对此明确写了"五个重点区是被审计而非被改动"。**这不改变该款的实质要求已满足，
但示例命令与真实历史不一致这一点应当写下来，而不是让勾选去暗示它发生过。**

### 二、三处"读起来像保证"的地方，现在各有一条绊线

本轮不再新增功能，而是把**三处已经成立、但没有任何东西在看着它成立**的性质变成被测性质。
三处的共同形状是：**一句散文或一个巧合在承担一条保证**。

**(a) 决定 G(a) 的"无值可挑"只有一个写入者在撑。**
续二十二结案了 G(a)：`source_available_date` 全仓只有一处写入（`source_reader.py:102`）
且等于业务键 `observation_date`，故重复组内所有行的可用日必然相同。
**但"只有一处写入"是一个随时会变的偶然事实，不是一条被看守的不变量。**
新增 `test_every_row_of_a_duplicate_group_carries_the_same_available_date`：
同时断言**机制**（`source_available_date == observation_date` 逐行成立）与
**后果**（每个重复组内该列的相异值数恒为 1）。
**两个方向都要断言**：只断言后果，则"reader 给所有行盖同一个常量日期"也会通过；
只断言等值，则"`observation_date` 不再是业务键"会静默溜过。
**变异 M63**（让可用日按行位置变化，即基本面路径接进来后的形状）
→ **只有这一条新测试失败，套件里没有任何别的东西在看着这条性质**。
**M62**（盖一个常量日期）→ 等值那半失败。**这正是"绊线"的定义**：
日后谁把基本面路径接进这个去重器，会先撞上一条测试，从而是在**做决定**而不是**继承决定**。

**(b) `create_factor_set` 的 docstring 声称了一条代码从不提供的保证。**
原文："Build a deterministic DRAFT factor set from **active-registry** member identities"。
实际：收 `list[tuple[str, int, int, str]]` 裸元组，**从不接触 registry**，
只校验形状（每因子一个版本、版本 ≥1、方向 ∈ {-1,0,1}、hash 非空）。
**故"该因子是 ACTIVE"与"该 hash 是 registry 登记的那个"都不是成功调用所确立的事实**，
而 docstring 让读者以为它们是。评审把这类缺口称为"更危险的形状，因为它读起来像一个存在的保证"。
**已改 docstring 说明真实范围**（内容寻址使集合**不可变且与输入顺序无关**，
但**不使其权威**），并补两条测试：一条钉"registry 从未听过的身份照收"，
一条**基于 AST** 钉"模块没有任何通往 registry 的 import 通道"。
**两条互补而非冗余**：**M64**（加一条 registry import）→ **只有 AST 那条失败**；
**M65**（不加 import、只把 hash 校验收紧成 64 位）→ **AST 那条存活、行为那条失败**。
**即"把集合变成 registry 核验的"这一改动，无论走不走 import，都会撞上一条测试**——
而那正是决定 A 要裁的事，不该由一次顺手收紧完成。
**未预判决定 A**：docstring 只描述现状，并明写"生产在用的冻结集是一份从未经过此函数的人工裁决 JSON"。

**(c) C7.5 的"已知高相关因子"全是合成的——续二十二注明了这处不一致，本节把它关掉。**
仓内已有真实案例：`artifacts/diagnostics/gp_turnover_scalar_degeneracy_diagnostic_v1.json`
记录 `yeslab_436/446/468` 的末步都是 `turnover.mul(<逐日 Series>, axis=0)`
（一个乘逐日均值、一个乘逐日最大值），
且三者**日度 RankIC 序列在 36 个 signal day 上完全相同**；
同一份工件还给出**真实对照例**：`yeslab_464`（GP_095）末步是逐股票回归残差、序列不同、被放行进入完整评估。
据此新增 `test_a_per_date_scalar_multiple_is_one_factor_however_much_its_values_move`。

**它与既有那条端到端测试的区别正是要点**：既有的 `copy` 是**恒等**、`mirror` 是**常数倍**取负，
两者都是 `dense` 的**固定倍数**，任何基于数值的比较都能抓住；
而**逐日标量两者都不是**——数值每天都在变，三个因子在整段窗口上**互不为标量倍**。
只有"逐日、基于秩"的相似度能看穿它，而这恰是 C7.2 规格里"每日横截面 Spearman"那一半。
**变异 M67 把这点钉死了**：把排名**预先在整段 scope 上算一次**再逐日取相关
（**一个词级别的改写，而且正是续二十二实测出的成本上限最容易诱发的"优化"**）
→ **只有这条新测试失败**；既有 C7.5 测试**照常通过**，因为常数倍在池化排名下不变。
**在此之前，套件里没有任何东西看守"秩是日内秩"这一半。**
**数值仍是合成的**：诊断工件记录的是**结构**而不是源样本，把源行搬进树里正是 C3.4 明令禁止的；
真实的是**机制与对照例**，两者都引自已提交的工件。
前提断言两条都在结论之前：三个退化因子**逐日共享同一秩向量**、
而其**逐日均值在五天里各不相同**（M66 把标量压平为常量 → 这条前提失败，
故 fixture 不可能悄悄退化成既有测试已覆盖的情形）。

### 三、勘察全部 19 项未勾项：结论是 C 计划已由**决定**约束

逐项查过每个未勾项的门，**没有一项是"现在就能写代码推进"的**：

| 未勾项 | 门 | 门的性质 |
|---|---|---|
| C1.4 实现 registry 状态 | 决定 A（registry 路径是否为准）+ 决定 I | 裁决 |
| C1.6 运行测试并提交 | 需 C1.4；第二条命令要 `tests/integration/factors/`（**不存在**）→ 决定 K | 裁决 + 协调 |
| C3.1 写三类决策的失败测试 | 测试已写；两处规格偏差**不改**（决定 G），且两款现均**零实例** | 裁决 |
| C3.5 运行测试并提交 | 需 C3.1（其命令实跑 **20 passed**） | 依赖 |
| C4.2 写基本面事件测试 | 退市款已补；另一款需改契约 → 决定 B | 裁决 |
| C4.3 实现事件区间展开 | 决定 B；且成本已测为**不可接线** | 裁决 |
| C4.4 实现修订版本语义 | 决定 B | 裁决 |
| C4.5 强制 PIT 不变量 | `extracted_at` 是**只写**溯源字段，接线即改活路径 | 裁决 |
| C4.7 运行测试并提交 | 需 C4.2–C4.5 | 依赖 |
| C5.1 写 pipeline 组合测试 | 十步固定顺序里**三步不在物化链上**；`registry validate` 属决定 A | 裁决 |
| C5.3 定义 CLI 与 dry-run | dry-run 半已交付；四条命令**两条属决定 A、一条属决定 F** | 裁决 |
| C5.4 实现增量影响范围 | 第一款 → **决定 J（新）**；第二款 → 决定 B；第三款 → 决定 A | 裁决 |
| C5.5 V2Test 集成故障测试 | `tests/integration/factors/` **不存在**；写 V2Test → **决定 K（新）** | 协调 |
| C5.6 运行测试并提交 | 需 C5.1/C5.3/C5.4/C5.5 | 依赖 |
| C6.2 区分 blocker 与 eligibility flag | 模块**未接线**（决定 I）；生产另有 `screen_factor_daily_coverage.py` 一套阈值 | 裁决 |
| C7.1 写失败测试：相似度不能越过 scope | 决定 E（universe 来源未定） | 裁决 |
| C7.2 实现每日横截面 Spearman 汇总 | **已实现**；覆盖率分母属决定 E | 裁决 |
| C7.4 实现训练内代表因子排序接口 | 决定 D（排序键以规格还是实现为准） | 裁决 |
| C7.6 运行测试并提交 | 需 C7.1/C7.2/C7.4 | 依赖 |

**含义**：19 项全落在 **8 个决定（A/B/D/E/F/G/I，加本轮新增的 J/K）**
或 **4 个"Task 级收口依赖子项"**上。**下一步的顺序完全由 owner 的决定顺序给出**，
而该顺序已写在交接清单 §4。**本轮因此没有新增任何治理实现**——
在决定 I 悬而未决时再写一个零调用者的模块，只会加深 I 所描述的问题。

### 四、两个新增决定（都由本轮勘察逼出来，此前没被命名）

**决定 J：C5.4 第一款的"比对状态"存在哪里。**
第一款（技术日频源变化只重算其变化 signal dates）**未被 A/B/C 任何一条阻塞**，
看起来是最可做的一项。但实现它需要**把源按 signal date 的摘要与"上次已发布的摘要"比对**，
而**这份已发布摘要当前无处可存**：

- **(J-i) 新建一张 V2 表**（如 `factor_source_digest`）：**最干净，但要动 migrations**，
  即改一个**另一条对话正在使用的库的 schema**。
- **(J-ii) 复用 `artifacts/diagnostics/factor_daily_materialization/materialization_status.json`**：
  已有逐月状态，但**该文件属另一条对话的在制品**，本轮红线明令不碰。
- **(J-iii) 新建一个独立的诊断工件命名空间**：不碰任何在制品，
  但**它不是权威状态**（重启/重跑无人保证它与库一致），
  于是"只重算变化的 signal date"会建立在一份可能过期的比对基准上——
  **这比全量重算更危险，因为它看起来是增量的**。

**倾向 (J-i)**，但**不代做**：新增表是 schema 决定，且它与决定 I（是否要求治理模块有生产调用者）
直接相关——**一张只为未接线模块服务的表，正是 I 要防的东西**。

**决定 K：C5.5 的故障注入集成测试能不能写 `dfs://QuantResearchV2Test`。**
C5.5 要覆盖六种故障（源 schema 漂移、源水位中途变化、重复冲突、`available>signal`、
publish 前进程异常、相同版本重跑），**其中"publish 前进程异常"与"相同版本重跑"本质上要真的写库再回退**。
**实测的协调事实**：`materialization_status.json`（只读查看，未修改）记录
`target_database = dfs://QuantResearchV2Test`、`factors_materialized_total = 17`、
`total_rows = 68,363,458`、`complete = true`、`materializer_running = false`、`materializer_pids = []`
——**即那个库里已有一份完成的、6836 万行的冻结 17 `factor_daily` 物化**，
是另一条对话的产物，也是冻结集计划 Task 3.1 的输入。
**故障注入测试会发布并回退行、移动版本指针**，故：

- **(K-i) 就写 V2Test，测完重新物化**：最贴近 C5.5 的字面要求，**代价是可能毁掉那份 6836 万行的产物**。
- **(K-ii) 用独立的表名前缀或独立库**：安全，但**与 C5.5 要的"真实发布路径"有距离**，
  且需要 `V2WriteSession` 的白名单扩容（`db/session.py` 的 `_V2_DATABASES` 只有两个库）。
- **(K-iii) 只写不需要落库的那几种故障**（schema 漂移、重复冲突、`available>signal` 可在纯函数层覆盖），
  把"进程异常"与"同版本重跑"两款**显式记为未覆盖**：
  **续二十/续二十一 实际走的就是这条路**（同版本重跑 8 条单元测试、源水位 2 条 characterization test），
  故 (K-iii) 等于**承认现状并把它写清楚**。

**倾向 (K-iii) + 就 (K-i)/(K-ii) 请示**，理由：**在一份已完成的 6836 万行产物上做故障注入，
属于会造成外部状态改变的動作，不该由执行方默认。**

### 五、变异验证（本节 6 个，M62–M67，全部被预期测试捕获；其中一个**第一版是无效的**）

| # | 变异 | 被谁抓住 |
|---|---|---|
| M62 | reader 给所有行盖**同一个常量**可用日 | 2 条（新测试的等值那半 + 一条既有测试） |
| M63 | 可用日**按行位置变化**（= 基本面路径接进来后的形状） | **只有新测试**（后果那半） |
| M64 | `factor_sets.py` 出现一条 registry import | **只有 AST 那条** |
| M65 | 不加 import、只把 hash 校验收紧成 64 位 | 17 条，**AST 那条存活**、行为那条失败 |
| M66 | 把 fixture 的逐日标量压平为常量 | 只有前提断言那条 |
| M67 | 排名**预先在整段 scope 上算一次**（成本上限最容易诱发的"优化"） | **只有新的 C7.5 测试** |

**M63、M64、M67 三个都"只被一条测试抓住"**，这正是本节想要的结果：
它们各自看守的性质**此前无人看守**，而三处都在散文里被当成已经成立。

**本节自查——本轮第二次"变异没落在被测路径上"（第一次是续九）。**
M67 的**第一版是无效的**：我把逐日过滤的布尔条件从 `daily` 换成 `merged`，
以为这样就把相关算到了整段窗口上。**实测 18 条全绿**，即变异存活。
追查原因：fixture 里没有非有限值，故那个掩码**全为 True**；
而 `.loc` 用的是 `daily`，布尔序列的索引是 `merged` 的**超集**，
pandas 按标签对齐后**又把它裁回 `daily`**——**变异等于没改**。
**"变异存活"有两种完全不同的原因：性质没被测到，或变异根本没生效**；
两者在输出上都是"全绿"，**只有追查到底才能分辨**。
改成"预先在整段 scope 上排名"后才真正落在被测路径上，并如预期只被新测试抓住。
**判据因此补一句：变异存活时，先证明变异确实改变了行为，再谈测试有没有看守。**

### 六、测试与门槛

**全量：1541 passed / 45 skipped**（本节起点 1537；+4：1 reader 绊线 / 2 factor_sets /
1 C7.5 真实出处），`ruff check src tests` 清、`mypy src` **158 文件清**。
**未新增任何生产行为**：本节改的只有 docstring（`create_factor_set`）与测试；
`source_reader.py`、`similarity.py`、`factor_sets.py` 的变异**均已确认还原干净**（`git diff` 为空）。
**未碰** `materialization_status.{json,md}`（只读查看过一次以取得决定 K 的证据），
**未释放** `/tmp/frozen_chain_hold.txt`（裁决 4 未定），**未写任何数据库**。

## 勾选审计续二十四（2026-09-06 17:15）：裁决 4 有了第三套独立证据；一个存活变异暴露出"同一记录写两遍、只测了一遍"

### 一、为什么是这件事

续二十三的勘察结论是"C 计划余下 19 项全部由决定约束"。**但决定要能被做出来，先得有证据。**
`docs/handoffs/2026-09-05-compute-and-coverage-decision-brief.md` 的裁决 3/4 段落止于今日 02:35，
**早于 Task 2.4 落地（13:16）与本轮全部普查**；而**裁决 4 门着冻结集计划的 Task 3.1，
Task 3.1 又门着 3.2 / 3.3 / 4.1 / 5.x 整条链**——那是当前唯一的关键路径。
故本轮最有价值的非裁决工作不是再写代码，而是**把这两个裁决的证据基础补齐**，
让 owner 能真的做决定。两项都已写入该简报（`裁决 3 更新（17:00）`、`裁决 4 新增证据（17:00）`）。

为做到"同窗可比"，普查新增了 `--scope selected --factors … --start … --end …`：
**窗口约束同时进入查询与按年回退**——回退若仍走 2010..2026，会在一次服务端拒绝之后
**悄悄把有界普查重新放宽成全区**，而窗口外的年份返回零行、**看起来一切正常**。
**M68 去掉该约束，只有窗口那条测试失败。**

### 二、裁决 4：第三套独立方法，同样六个因子，同样双峰

**口径**：不碰覆盖率门、不碰评分、不碰 `factor_daily`，只普查原始 `factor_data*`，
**与筛查同窗 2020-01-01..2024-12-31**。指标是"**能发布出值的源键占比**"
= (唯一可用 + 重复全同 + 重复单非空) / 源键总数，即去重器真会写进事实表的那三类。
工件已入库：`artifacts/diagnostics/duplicate_key_shape_census_screen_window.json`
（11 因子、**分区自检零违例**、naive 恒等式**全部成立**）。

| 因子 | 名称 | 窗口键数 | 可发布 | 源键不可用 | 冲突隔离 |
|---|---|---|---|---|---|
| `yeslab_375` | GP_006 | 7,278,060 | **35.1%** | 46.7% | **18.18%** |
| `yeslab_406` | GP_037 | 7,215,036 | **37.2%** | 62.8% | 0 |
| `yeslab_372` | GP_003 | 7,213,824 | **38.0%** | 62.0% | 0 |
| `yeslab_384` | GP_015 | 7,213,824 | **39.0%** | 61.0% | 0 |
| `yeslab_390` | GP_021 | 7,216,248 | **39.3%** | 60.7% | 0 |
| `yeslab_389` | GP_020 | 7,213,824 | **39.4%** | 60.6% | 0 |
| `yeslab_206` | 60日成交量去均值相对偏离 | 7,195,644 | 76.8% | 23.2% | 0 |
| `yeslab_96` | 修正日间反转 | 7,195,644 | 78.9% | 21.1% | 0 |
| **`yeslab_484`** | **羊群效应_CSAD波动率21（dense 对照）** | 7,282,908 | **79.0%** | 21.0% | 0 |
| `yeslab_381` | GP_012 | 7,213,824 | 79.6% | 20.4% | 0 |
| `yeslab_357` | 行业内收益标准化绝对值 | 6,787,200 | 85.0% | 15.0% | 0 |

**(1) 双峰，且六个最低者与覆盖率门判 FAIL 的六个完全一致**（372/384/389/390/406 + 375）：
最低 **35.1%–39.4%**，第七名直接跳到 **76.8%**，**中间没有过渡带**。
**这是第三套互不引用的方法**（他们的筛查 / 我从 `factor_daily` 的重放 / 现在的原始源普查）
给出同一分组。

**(2) 分母不同 ⇒ 数值不同、档位相同。** 门是"当日 universe 成员中有值的比例，**逐日取中位**"，
普查是"源键中能发布的比例，**全窗池化**"；源键含 universe 之外的代码，故普查值系统性偏低：
GP_003 门 **0.4636** vs 普查 **0.380**、GP_015 门 **0.4922** vs 普查 **0.390**。
**没有任何因子换档。**这一点必须写清，否则"第三套方法印证"会变成假印证。

**(3) 一处不一致，如实报告。** dense 对照 `yeslab_484` 同窗普查 **79.0% 可发布**，
而门给它 `median_cov` **1.0018**——差 **21 个百分点**，远大于 GP_003 的 8 个百分点，
**"分母不同"盖不住**。两个候选解释，**本轮未区分**：
**(a) 中位数 vs 池化**——若其 NULL **集中在少数整天**，逐日比值的中位仍可为 1.0 而池化掉到 0.79，
这与 Task 3.1b 已记录的"崩盘日塌缩"**同型**（11 个丢失日在五年训练段里是 99.1% 有效日占比，
**按日占比的门槛抓不到**）；**(b) 源键含 universe 之外的代码**且这些代码系统性为 NULL。
**区分它们需要一次按日测量**（每因子：零可用值天数、逐日可用比的中位与 p05），本轮**未做**，
约 7 条只读查询即可。**在区分之前，"`yeslab_484` 是 dense 对照"不应继续被当作已确立的事实使用**——
若真相是 (a)，则它"看起来 dense"是度量假象，而基线的一律 `dropna` 会把那些天整天删掉。

**(4) 机制明确 ⇒ 选项 (i) 插补的可行性判断变了。**
六个里**五个的损失全部来自"源里有行、值是 NULL"**（60.6%–62.8% 不可用、冲突 0；
`yeslab_406` 更是**零重复**，其问题与去重无关）。
**这不是"源没说话"而是"源明确说了没有值"**，故换源/补齐救不回来，
而 (i) 是**为源刻意留空的格子造数**。

**(5) GP_006 有第三种成因，使选项 (ii) 的代价被低估。**
**64.9% 不可发布 = 46.7% 源键不可用 + 18.18% 冲突隔离**。
冲突隔离源于**源自相矛盾**，而 C3.1 明确**不选 first/last、整组隔离**，
**故这 18.18% 在"冲突如何解"的规则出现之前无法恢复，而该规则不存在（决定 G）**。
**全源区间同样是 18.2%**（1,721,876 / 9,457,875），**即冲突在时间上均匀，不是早年现象**。
**含义**：即使裁决 4 选 (ii) 对齐 universe 口径，GP_006 仍有 18.18% 的键因冲突缺席；
而 (iii)（复议这六个）现在多了一条独立理由：**GP_006 的问题不全是覆盖率，还有一部分是源自洽性。**

**(6) 全区间的对照**（说明双峰不是窗口选择的产物）：可发布占比
35.6% / 38.5% / 39.9% / 40.6% / 41.1% / 41.2% 对其余十个 74.9%–86.9%。**同样六个、同样双峰。**

### 三、裁决 3：复现性不再是声称

Task 2.4 的重跑（F01–F06、16h09m、不带任何新旗标）于 13:16:50 落地 **CONSISTENT / exit=0**，
独立逐字段比对确认 `folds`/`data_versions`/`oos` **逐字节相同**。
**故"遗留链可复现"已被一次不带货运的完整重跑证实**，
这**消除了简报 02:35 更新里第一个选项（修复后重启遗留链）在复现性维度上的顾虑**；
余下的障碍只有算力（串行 42.7 天）与裁决 2 的并发改造。
**保留的限定同样写进了简报**：同机同环境；过程中 **345 次 ElasticNet 非收敛警告**与
**90 次数值奇异求解**（rcond 低至 9.7e-114）仍逐位复现，故**跨环境稳定性未被证明**，
且**没有任何工件记录"这个被复现出来的数来自一次未收敛、含奇异求解的计算"**——**裁决 2 的必要性不减。**

### 四、一个存活的变异暴露出的真缺口：同一记录写两遍、只测了一遍

`run_frozen` 与 `run_selected` 各有一份**逐字相同**的 `NOT_CENSUSED` 记录
（源目录解析不出的因子——即冻结集里的修复变体 `yeslab_443_reg30_max20_v1`）。
我 aimed 于 `run_selected` 的**变异 M69 被抓住**，而**同一段文字在 `run_frozen` 里的 M70 存活了**。

**这不是小事**：`run_frozen` 正是**产出已提交工件**的那个 scope，
而工件里 `yeslab_443_reg30_max20_v1` 的 `NOT_CENSUSED` 记录
**是"17 个成员里有意只普查了 16 个"与"漏了一个"之间唯一的区别**。
静默丢掉它，冻结集普查会报告 16 个因子、**看起来完全正常**。

**已修**：两份合并为一个 `_not_censused()` helper，**两个 scope 各有一条测试**，
M70 重跑**被抓**。

**方法论上值得单记**：这处缺口**不是评审发现的、也不是读代码发现的，是变异存活发现的**。
而且**第一次变异打偏了**——我以为在改 `run_selected`，实际 apply_patch 匹配到的是 `run_frozen`
（两段文字逐字相同），于是"20 passed"被我一度读成"变异存活"，
而真相是**变异根本没落在我以为的地方**。**这与续二十二 M67 第一版是同一个陷阱的两种形态**：
**变异的结果只有在"变异确实改了我以为的那处"之后才有意义。**
**故判据再补一句：变异之后先确认它改的是哪一份拷贝，再谈存活与否。**

### 五、测试桩自己错了（同源教训第十一次）

新写的窗口测试**第一次运行就失败**，原因**不在被测代码**：
`_YearAwareClient` 用"脚本里有没有 `date('YYYY.01.01')`"来区分整段查询与按年查询，
而**窗口下界恰好是 1 月 1 日**，于是带窗口的整段查询被认成 2020 年的分年查询、
返回了 2020 的数——**测试因一个与被测代码无关的理由失败**。
已改为**按调用次序**区分（第一条必是整段尝试）。
**教训与前十次同源**：断言/期望/桩里都藏着未经验证的假设，
而**桩错得最隐蔽**，因为它让失败看起来像被测代码的问题。

### 六、变异验证（本节 3 个，M68–M70）

| # | 变异 | 结果 |
|---|---|---|
| M68 | 接受窗口参数但**忽略**它（查询与回退都不收窄） | 仅窗口那条测试失败 |
| M69 | `run_selected` **静默跳过**解析不出的因子 | 仅 NOT_CENSUSED 那条失败 |
| M70 | `run_frozen` 里**同样的静默跳过** | **第一次存活** → 补测试后被抓 |

**累计 70 个（M46–M70 本轮 25 个）**，其中**一个存活过一次**（M70），
**存活的原因是被测代码有两份相同文字而只有一份被测**——已修。

### 七、测试与门槛

**全量：1544 passed / 45 skipped**（续二十三 起点 1541；+3：窗口约束 / selected 的 NOT_CENSUSED /
frozen 的 NOT_CENSUSED），`ruff check src tests` 清（本轮新脚本单独查亦清）、
`mypy src` **158 文件清**（本轮未改 `src/`，仅 `scripts/` 与 `tests/`）。
**未改动任何生产行为**：`src/` 本轮零改动；`source_reader.py` / `similarity.py` /
`factor_sets.py` / `catalog_dry_run.py` 的变异**均确认还原干净**。
**未写任何数据库**（普查全部只读）；**未碰** `materialization_status.{json,md}`
（本轮只读查看一次以取得决定 K 的证据）；**未释放** `/tmp/frozen_chain_hold.txt`。

## 勾选审计续二十五（2026-09-06 17:50）：续二十四留下的悬置问题已关闭——`yeslab_484` 的 dense 对照成立；并更正了我自己写错的一条判读规则

### 一、为什么接着做这件事

续二十四如实记了一处不一致并**明确留置**：dense 对照 `yeslab_484` 在覆盖率门上是
`median_cov` **1.0018**，而同窗原始源普查是 **79.0% 可发布**，差 21 个百分点；
两个候选解释——**(a)** NULL 集中在整天、被"逐日比值取中位"的门看不见；
**(b)** 分母不同（门按当日 universe 成员、普查按源里存在的键）——**后果相反**：
若 (a) 成立，则"484 是 dense 对照"是度量假象，而基线的一律 `dropna` 会把那些天整天删掉。
当时写下"**在区分之前不应把它当作已确立的事实继续使用**"。

**这是一个便宜、只读、且门着裁决 4 的问题，故本轮把它关掉**，而不是留给下一轮。
新增 `scripts/measure_daily_usable_share.py`：测每因子"逐日可用比"的分布
（天数、空日数、低于半数的天数、逐日中位、最差单日、全窗池化），
同样只回聚合、每因子一次查询、窗口强制、`--scope self-test` 的 **7 项指标与手算逐项一致**后才指向真实源。
工件已入库：`artifacts/diagnostics/daily_usable_share_screen_window_v2.json`。

### 二、结论：(a) 被否证，(b) 成立，dense 对照成立

**十一个因子（同窗 2020-01-01..2024-12-31，各 1212 个交易日）：空日总数 = 0；
逐日中位与全窗池化之差最大 0.019。**
`yeslab_484`：中位 **0.807**、池化 **0.790**、空日 **0**、低于半数的天 **0**、最差单日 **0.627**。

**结论不依赖阈值，只依赖量级对比**：日级分布的最大效应是 **0.019**，
而要解释的分歧是 **0.20**——**小一个数量级**，故 (a) 不成立。
484 的不可用值**均匀分布在每一天**，不存在"中位数看不见、池化看得见"的结构。
**于是 (b) 成立：21 个百分点是分母差。**

**并且这个分母差是近似恒定的乘子，这使它可核对而不只是可解释**：
GP_003 **0.361/0.4636 = 0.779**、GP_015 **0.380/0.4922 = 0.771**、
`yeslab_484` **0.807/1.0018 = 0.805**——
**三个覆盖率水平完全不同的因子，比值落在 0.771–0.805**，
即源里每天出现的 (代码,日期) 键比当日 universe 成员多约 **23%–29%**。
**故档位对分母不敏感、dense 对照成立**，续二十四那句保留性告诫**据此撤回**（简报 (3b) 已同步）。

**一个对裁决 4 与 Task 3.1b 都有用的区分**：六个稀疏因子的"低于半数的天"占 **70%–86%**
（846–1043 / 1212）而**空日为 0**，最差单日低到 **0.019–0.111**。
即它们不是"有几天坏了"，而是**几乎每天都薄**。
**这与 3.1b 记录的"崩盘日塌缩"是两种不同的病**：那种是**少数整天丢失**、
按"有效日占比"的门看不见（11 个丢失日在五年训练段里是 99.1%）；
这种是**天天薄**，按"当日横截面宽度"的门看得见。
**两个门的灵敏度差得很远**：宽度门对 GP_003 只报了 F01 的 15 天，
而"低于半数可用"的天有 **879** 天——**说明现有宽度门的阈值远在 50% 覆盖之下**，
这是评估选项 (ii) 代价时要紧的背景。

### 三、我自己写错了一条判读规则，是真实数据把它抓出来的

`classify()` 的**第一版**把判决挂在**日计数**上：只要"有空日或有低于半数的天"就判
`WHOLE_DAY_HOLES`，并附一句"中位数比窗口读起来更好"。
**第一批真实数据就把它证伪了**：GP_003 有 **879 / 1216** 天低于半数、**0** 个空日，
而它的中位（0.361）与池化（0.380）**几乎相等**——
**它的稀疏是慢性的、均匀的，那句"中位数读起来更好"对它是假的。**

**已改**：判别式改为**中位与池化之差**（日计数降级为限定判决的细节，永远写在判决里、
但不再决定判决），并新增一条**直接取自真实输出**的测试
`test_chronic_sparsity_that_is_even_across_days_is_not_called_a_hole`
（用 GP_003 的实测数：0.361 / 0.380 / 879 天 / 0 空日）。
**变异 M75 就是把规则改回日计数版** → **只有这一条测试失败**，即这条测试正是那次更正的看守。
**M77**（把限定计数从判决文本里删掉）→ 4 条失败，
故"细节必须可见"也是被钉住的，不是装饰。

**同源教训，第二次**（第一次是续二十二把相似度成本按立方模型外推）：
**在看清数据之前就写下的判读规则，等于把假设写成了结论。**
两次的共同形状是：先有一个"看起来合理"的分类逻辑，再用真实数据检验它——
**而两次都是真实数据说不**。区别只在第一次错于模型选择、这一次错于判别变量选择。

**一处如实的保留**：`classify` 的容差是 **0.02**，而十一个因子的实测最大差是 **0.019**——
**贴着边界**，故**判决标签对容差敏感**，这一点写在这里而不是藏起来。
**但第二节的结论不受影响**，因为它用的是量级对比（0.019 vs 0.20，差一个数量级），不是那个标签。
**若日后要在容差附近做判断，应改成"与待解释的分歧是否同量级"，而不是一个固定的绝对阈值。**

### 四、变异验证（本节 4 个：M71、M72、M75、M77；M73/M74 未使用）

| # | 变异 | 结果 |
|---|---|---|
| M71 | 分支顺序对调（先判"均匀"再判"日级"） | 2 条失败（空日 / 低于半数各一条） |
| M72 | 把容差从 0.02 放宽到 0.50（足以吞掉真实分歧） | 2 条失败（阈值两点各一条） |
| M75 | **判别式改回日计数版**（即 mislabel GP_003 的那一版） | **只有取自真实输出的那条失败** |
| M77 | 把限定计数从判决文本里删掉 | 4 条失败 |

**M73/M74 原计划用于"窗口约束"与"只回聚合"**，但这两条已由续二十四的 M68 与
本节 `test_every_reported_metric_is_an_aggregate…` 覆盖，故未另设变异。
**M76 是本节编号时的跳号**（实际只做了 M75 与 M77），
**它不对应任何未做的验证**。**三个编号一律留空而不复用，以免账目对不上。**
**累计 74 个（M46–M72、M75、M77）。**

### 五、测试与门槛

**全量：1560 passed / 45 skipped**（续二十四 起点 1544；+16 按日测量），
`ruff check src tests` 清（新脚本单独查亦清）、`mypy src` **158 文件清**（`src/` 本节零改动）。
**未写任何数据库**（按日测量全部只读、11 条查询）；
**未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`；
**未改动任何已物化数据、版本指针或 `dropna` 口径**——
**裁决 4 的四个选项一个都没被预判，改变的是证据强度而不是选项集合。**

## 勾选审计续二十六（2026-09-06 18:00）：六条完成标准重判——第 2 条转为完全满足，于是余下四条的缺口全部落在四个决定上

续十四 是六条完成标准第一次据实判定，结论是"只有第 4 条完全满足"。
本节不重写它，只按本轮新增的证据**重判一次**，并逐条写明变或不变的理由。

| # | 完成标准 | 续十四 | 现在 | 变/不变的理由 |
|---|---|---|---|---|
| 1 | 启用因子都有显式、版本化、不可变 registry 定义；源新增列不会被自动纳入 | 半满足 | **半满足（不变）** | 后半仍完全成立、前半仍卡在 `FactorRegistry` **零生产调用者**。本轮只改了 `create_factor_set` 的 docstring——那是**诚实性**修复（它声称了一条代码不执行的约束），**不是满足度**变化。**依赖决定 A** |
| 2 | GP_017/GP_077 类重复问题被统计，冲突键全部隔离，clean 表唯一 | **三分之二** | **完全满足** | 缺的那一块（「重复问题被统计」）已由普查交付，且**比该款要求的更多**：不止 GP_017/GP_077，**全部冻结 17 都被统计**，并有分区自检与一条在 18 个因子上零违例的恒等式看守测量本身。另两款的测试逐一点名核实仍在（见上文勾选注记） |
| 3 | 技术因子不 ffill；基本面严格在公告后下一交易日可用且最多持有 180 日 | 性质成立、路径半接线 | **不变** | 三款性质仍逐一实测成立；基本面半部仍**未接线**且仍带**决定 B** 的跨期版本缺陷。本轮新增的证据反而**收紧了它**：G(a) 的绊线测试意味着"把基本面路径接进这个去重器"会**先撞上测试**（见续二十三） |
| 4 | 修改未来公告/值不改变任何过去 factor observation | **完全满足** | **完全满足（不变）** | 六条泄漏测试未受影响 |
| 5 | 结构质量不读取标签；相似度与代表选择严格绑定训练 scope | 前半满足、后半部分 | **不变（但新增成本上限）** | 前半仍钉得很硬；后半仍卡 **决定 E**（scope 未绑 `universe_id`/`data_version`）与 **决定 D**（排序键两处不符）。本轮**新增**：`compute_factor_similarity` 的实测标度律与四个尺度的时间/内存外推（续二十二），**故"绑定 scope"这件事现在同时是一个成本决定** |
| 6 | factor set content-addressed、顺序固定、冻结后不可变，可被 D/E/F 精确复用 | 代码层四款成立、复用款不成立 | **不变** | 「可被精确复用」仍不成立：`create_factor_set` 零生产调用者，生产读的是**没有 content-addressed id** 的 JSON 工件。本轮补的两条绊线（行为 + AST）钉住的是**当前范围**，不是把它接上线。**依赖决定 A** |

**净结果**：**六条里两条完全满足（第 2、第 4），四条各缺一块，
而这四块的缺口全部落在 A / B / D / E 四个决定上**——
第 1、6 条依赖 **A**，第 3 条依赖 **B**，第 5 条依赖 **D 与 E**。
**即：完成标准的剩余部分已完全由四个决定门着，没有一块是"再写点代码就能补上"的。**
这与续二十三对 19 个未勾项的勘察结论一致，且是它在**验收标准层面**的同一幅图。

**一条方法论记号**：第 2 条之所以能从"三分之二"变成"完全满足"，
不是因为有人改了实现，而是因为**缺的那一块从来不是实现问题、而是输入问题**——
续十一 拒绝凭空造 fixture、续二十二 去把输入测出来，本条才 closes。
**"缺输入"与"缺实现"是两种不同的未完成**，混在一起记会让计划看起来比实际更难推进。

**门槛**：本节**零代码改动**（只勾选与重判），故沿用续二十五的实测值：
**全量 1560 passed / 45 skipped**、`ruff check src tests` 清、`mypy src` 158 文件清。

## 勾选审计续二十七（2026-09-06 18:20）：给冻结集计划的 3.2/3.3/4.1 做预检，查出一个此前无人见过的顺序依赖缺陷

### 一、为什么是"预检"

C 计划的 19 个未勾项已全部由决定门着（续二十三），于是本轮转向**另一份计划里同样被门住、
但可以提前验证的环节**：冻结集计划的 **3.2（判定渲染）/ 3.3（扩展选拔字段随报告签发）/
4.1（核验器参考指向新报告）**都要等 Task 3.1 的产物，而 Task 3.1 门在裁决 4 上。
**但"那三步会不会跑通"是可以现在就验证的**——一次 15 小时的冻结集运行之后才发现渲染器少显示一个字段、
或核验器不比较版本映射，代价远高于现在拿既有报告预检一遍。**这就是当前唯一还能并行推进的方向。**

### 二、3.3 的字段到了 JSON、没到判定（已修）

3.3 写着"折匹配基线预测工件 + 扩展选拔字段（`invalid_day_count`/`unselected_reasons`）随报告签发
（**现有代码已含**）"。**这句话对生产者成立、对渲染器不成立**：
`run_real_walk_forward.py` 确实发出这三样（每个 validation score 带 `invalid_day_count`、
每个 selection 带 `unselected_reasons`、顶层带 `baseline_oos_artifacts`），
而 `summarize_real_walk_forward.py` **一个都不渲染**——已归档的 criterion-303 markdown 里
三者**零命中**。于是判定的读者看到 F01/B0 是 0.0724，**无从知道它来自 238 天还是 118 天**，
而 Task 5.1 要当 `--baseline-artifacts` 消费的那些路径**只能开 JSON 才找得到**。

**已补三个条件段落**，并加了一条比"预期字符串"强得多的钉：
**用当前渲染器重渲染已归档的 criterion-303 JSON，输出与已归档的 .md 逐字节相同**
（对真实的两个提交文件断言，且 CLI 端到端也验过）。
**这就是为什么新段落必须是条件式的**：归档报告早于这三个字段，
无条件加一列 "Invalid days" 会显示短横或 0，而**0 读起来是"没有丢天"、真相是"当时没记"**。
故 `selection_detail` / `baseline_oos_artifacts` 缺失时返回 **None 而不是空**，
只带日计数、不带工件路径的报告（**中断运行正是这个形状**）渲染前者、不声称有后者。
拿**真实的 2.4 重跑报告**跑通：24 行逐折日计数 + 每个落选者的原因 + 6 条工件路径，判定仍在最后。

### 三、4.1 的核验器有两个洞（已修，且修法受一条硬约束）

4.1 要把 `--reference` 指向冻结集报告、使它成为**新信任根**。预检发现它**不比较两样东西**：

1. **`data_versions` 从未被比较过**。2.4 的 CONSISTENT 结论靠的是**手工逐字段比对**
   （那次确认版本映射逐字节相同），而核验器自己只看 folds、winner、每基线的 validation RankIC、
   OOS 统计、gates 与年度序列。**故一次跑在不同已发布表版本上的重跑会被报 CONSISTENT**——
   而这正是这个核验器存在的理由：版本映射是证据里"到底算了哪份数据"的那一半。
2. **每折每基线的 `valid_day_count` / `invalid_day_count` 从未被比较过**。
   只比 `mean_rank_ic` 意味着：一次只算了 **228 天而非 240 天**、却仍落在 1e-9 内的重跑会通过。
   而 `invalid_day_count` **正是 3.3 签发、3.1b 覆盖率问题所在的那个字段**。
   `unselected_reasons` 同病：winner 变了会被抓，**落选者换了个落选原因不会**。

**硬约束（也是修法的形状）**：已归档的 criterion-303 参考**不带这三个字段**，
而 2.4 那次 CONSISTENT 正是"带新字段的重跑 vs 不带新字段的参考"。
**若强制要求，就会把一个已签发、且已被依赖的判决推翻。**
故比较**由参考驱动**：参考没有的字段不要求重跑有，参考有的字段必须逐项相符。
**这正是让核验器在 4.1 之后变成"更强的信任根"而不是"另一个信任根"的关键**——
冻结集报告三者皆有，于是三者都会被验。

**端到端证明而非口头论证**：用**真实的归档参考 + 真实的 2.4 重跑报告**跑 CLI，
仍打印 `CONSISTENT` / `exit=0`；同一比对也进了测试套件（两个文件任一缺失即 skip）。
版本不一致的信息**指名变动的表**，而不是把 151 分区的映射整个倒出来。

### 四、本轮最重要的发现：`expand_fundamental_events` **不是顺序不变的**，而套件一直说它是

`tests/property/factors/test_availability_invariants.py::test_event_row_order_does_not_change_the_expansion`
**今天失败了**，反例极简：

```
       ts_code report_period    ann_date  raw_value source_revision_id
0  000001.SZ    2024-09-30  2025-01-01        0.0                 r1
1  000001.SZ    2024-09-30  2025-01-01        1.0                 r1
universe: signal_date 2025-01-02, ts_code 000001.SZ
→ 正序发布 0.0，逆序发布 1.0
```

**根因（已定位到行）**：`availability.py` 的候选排序是
`sort_values(["available_date","report_period","source_revision_id"], ascending=[False]*3, kind="mergesort").iloc[0]`。
**mergesort 是稳定排序**，故**三键全同时保留输入行序**；两行的 `raw_value` 不同 ⇒
**发布哪个值由抽取的行序决定**。

**这不是本轮引入的**：`git diff 37dd575 HEAD -- .../availability.py .../test_availability_invariants.py`
**为空**（全会话零改动）。**是 hypothesis 在今天多次运行中找到了反例并存进本机示例库**，
此后确定性重放。而**`.hypothesis/` 自带 `.gitignore`（示例库不入库）**，
故**换一台机器它是偶发红的**——下一个人会撞上间歇性失败、且**没有任何记录解释为什么**。
这正是必须把它钉成确定性测试的理由。

**为什么这条要紧**：property 测试自己的 docstring 早就写明了理由——
YesLab 抽取按列、按月、按因子组到达，**顺序依赖不会可复现地失败，
它会悄悄改变某个过去 signal date 看到的是哪一个修订版本**。
而该 docstring 声称展开"**不依赖公告行到达的顺序**"。**在一个输入形状上它是依赖的**，
故 docstring 已改为如实陈述（**与本轮修的 `create_factor_set` docstring 同一类缺陷**）。

**未修，两个理由都是决定而非谨慎**：

- **语义选择属决定 B**：两个修订版无法区分时**应该**发布哪个，是语义问题；
  而该函数的排序语义**正是决定 B**，B 会重述排序键、并连带重述 C4.3 的区间公式。
- **更深的缺口是结构性的**：技术路径遇到自相矛盾的重复键，有一整套**去重 + 隔离**层
  （`deduplicate_factor_observations`，明确规定**不在冲突值之间做选择**）；
  **基本面路径完全没有对应物**。故诚实的修法大概是"给它补一层同样的东西"，
  **那是架构，不是一个 tie-break**。

**已做的三件事**：**(1)** 确定性 characterization test 用上面那个最小反例钉住现状
（正序 0.0 / 逆序 1.0）；**(2)** property 用显式 `assume` 排除该形状并指向那条测试；
**(3)** 另加一条测试**把过滤器钉窄**——两行全同键但**值相同**不叫矛盾、必须留在 property 内，
**否则 `assume` 可以被一路放宽到性质什么都不测而仍然全绿**（与"不会失败的检查"同源）。
**变异 M82 = 给排序加 `raw_value` 作 tie-break（最可能的修法）→ 只有那条 characterization test 失败**，
故它是绊线而不是备注。

**影响面**：`expand_fundamental_events` **零生产调用者**，故**当前没有任何已发布数据受影响**；
但它在**决定 B/C 要接线的那条路径上**。**已作为决定 B 的附加问题写入交接清单。**

### 五、变异验证（本节 5 个，M78–M82，全部被预期测试捕获）

| # | 变异 | 被谁抓住 |
|---|---|---|
| M78 | 参考缺字段时仍渲染空段落 | 2 条：**逐字节黄金测试** + "不得长出空段落" |
| M79 | 去掉 Invalid days 那一列 | 3 条 |
| M80 | 日计数比较改成**无条件**（"直接要求"这个看似合理的修法） | **恰好 2 条向后兼容测试**，含真实 criterion-303 那对 |
| M81 | 版本映射**与自己**比较 | 2 条 |
| M82 | 给基本面排序加确定性 tie-break | **只有那条 characterization test** |

**M80 值得单记**：它是"最直觉的修法"，而它会**推翻一个已签发、已被依赖的判决**。
**在信任根上加严之前，先问"加严之后旧证据还算不算数"**——本节的答案是"由参考驱动"，
而那条真实数据的回归测试就是这个回答的证明。

### 六、测试与门槛

**全量：1576 passed / 45 skipped**（续二十五 起点 1560；**+16**，逐文件核过：
`test_summarize_real_walk_forward.py` 14→22、`test_verify_baseline_rerun_consistency.py` 7→13、
`test_availability_invariants.py` 6→7、`test_availability.py` 11→12；8+6+1+1=16，与总数吻合）。
**属性测试的条数没变**，只是其中一条改为条件式（`assume` 排除矛盾并列），
新增的两条是"过滤器必须窄"与那条 characterization。
`ruff check src tests` 清、`mypy src` **158 文件清**。
**本节 `src/` 零改动**：改的两个文件 `verify_baseline_rerun_consistency.py` 与
`summarize_real_walk_forward.py` **都属 `scripts/`**，其余全是 `tests/`；
`availability.py` 变异后以 `git diff` 确认**还原干净**。
**未写任何数据库**；**未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`。

## 勾选审计续二十八（2026-09-06 18:35）：Task 4.2 的 runbook 命令逐条核验（**结果：零错误**），以及一条控制测试的 docstring 过度承诺

### 一、4.2 已勾，但它的命令此前没人验过

冻结集计划 **4.2「复现 runbook 增补『冻结集基线』章节与命令」已勾**，
依据是 `runbooks/real-walk-forward-reproduction.md` 里那些命令。
**但"文书里有命令"与"命令能跑"是两件事**，而 runbook 的使用场景恰恰最不容出错：
它是在一次 15 小时运行的末尾、在压力下被照着敲的。

**逐条核验结果：零错误。** 抽取代码块里的命令行（含续行拼接）：
**11 个被引用的脚本全部存在**，**17 个 (脚本, flag) 对全部对得上**，
且 11 个脚本的 `--help` 全部可运行；正文里解释的两个 flag
（`verify_frozen_fold_coverage.py` 的 `--factor-set` / `--minimum-factors`）也确认已注册。
**这是一次已核实的阴性结果**——没有东西需要修，故本轮**未改任何生产脚本**。

**但"我查过一次"与"它会一直被查"是两回事**，故已做成常驻测试
`tests/unit/scripts/test_runbook_commands.py`（6 条）：
脚本存在性、flag 注册性、**按脚本归属**、正文 flag、以及两条控制。
flag 从各脚本的 **AST**（`add_argument` 的字符串实参）读取而非跑 `--help`：
字符串常量可能出现在 docstring 或报错信息里而并不是选项，
而 11 个子进程各自 import dolphindb 比解析 11 个文件贵。
**M84 = 把 `preflight_frozen_baseline.py` 的 `--waiver-reason` 改名而不动 runbook
→ 只有注册性那条失败**，即这正是该文件存在的理由。

### 二、一条控制测试的 docstring 过度承诺，是被"照它说的去变异"抓出来的

我写的控制测试自称：若 `_declared_flags` "返回每一个 flag"，整个文件就会无论 runbook 写什么都通过，
而这条控制能抓住那种 bug。**照这句话去构造变异（M85：把两个脚本的 flag 汇成一池）后发现——
旧控制测试照常通过。**

**原因是具体的**：一个把所有脚本的 flag 汇成一池的收集器，
**仍然不会包含一个凭空编造的 flag**，故我那条"塞一个 `--definitely-not-a-flag` 进去"的控制
**照样通过，而真正的检查已经空转**。
**即：旧控制的 docstring 声称的能力，它并不具备。**

**已补第二条控制**，钉住**按脚本归属**：`--months` 属版本发布脚本、`--folds` 属走查脚本，
**互不得记到对方名下**。M85 正是**只被这条新控制抓住**（旧控制同时通过），
**这就实证了旧控制的承诺是假的**，而不是我推测的。
旧控制的 docstring 已改为**如实说明它钉的是更窄的一件事**（"未知 flag 的计算确实会报告"），
并明写它抓不住池化那种 bug、以及哪一条才抓得住。

**这是本轮第三次"变异无效或打偏"，也是第二次"控制测试只看起来像控制"**：

- 续二十二 M67 第一版：掩码全 True、被 pandas 对齐回原切片，**变异等于没改**；
- 续二十四 M70 第一版：两段逐字相同的文字，**变异落在了另一份拷贝上**；
- 本节 M83：`add_argument` 的位置实参在这些脚本里本来就全是 flag，**变异几乎不改变行为**。

**判据因此再补一句**：**控制测试的价值必须用"它自称要抓的那个变异"来检验**，
而不是用"它能抓住某个变异"来推断。**写下控制时顺手把它声称的变异真做一遍**，
是本轮唯一能发现这处过度承诺的手段。

### 三、变异验证（本节 M83–M85：M83 无效不计，M84/M85 均被捕获）

| # | 变异 | 结果 |
|---|---|---|
| M83 | 去掉 `startswith("--")` 过滤 | **存活——因为变异无效**（位置实参本来就全是 flag），不计入 |
| M84 | 脚本 flag 改名、runbook 不动 | 仅注册性那条失败 |
| M85 | flag 收集器把两个脚本汇成一池 | **仅新的归属性控制失败**（旧控制通过，故旧控制的承诺被证伪） |

**累计 81 个有效变异**（M46–M72、M75、M77–M82、M84、M85；M73/M74 留空未复用，M83 无效不计）。

### 四、测试与门槛

**全量：1582 passed / 45 skipped**（续二十七 起点 1576；+6 runbook），
`ruff check src tests` 清、`mypy src` **158 文件清**。
**本节 `src/` 与 `scripts/` 均零改动**（runbook 核验未发现任何错误，故无可修）；
`preflight_frozen_baseline.py` 的变异已用 `git diff` 确认**还原干净**。
**未写任何数据库**；**未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`。

## 勾选审计续二十九（2026-09-06 18:50）：Task 3.1 的三道失败门逐一定性——其中一道读的是**已知有误的快照**，而修正版从未入库（新增决定 L）

### 一、Task 3.1 的实际门不止裁决 4：预检现在是 REFUSED，五道门里三道 FAIL

`preflight_frozen_baseline_five_door_run.txt`（今日 06:04）的末行是
**`FROZEN BASELINE PRE-FLIGHT: REFUSED`**，阻塞门为
`admission_consistency` / `factor_coverage_screen` / `fold_coverage`，**无一被 waive**。
**故 Task 3.1 的门不只是"裁决 4 未定"，而是"预检明确拒绝启动"**，
而预检自己写了理由：*其输出会成为 Task 4.1 的信任根、并被 Task 5.1 当 `--baseline-report` 消费*。

**逐道定性后，三道门不是同一个问题**——这是本节的主要结果：

| 门 | 读什么 | 失败原因 | 定性 |
|---|---|---|---|
| `backfill_completeness` | `factor_daily` 逐年 | **PASS**（2014–2025 每年 17 因子、duplicates=0） | — |
| `admission_consistency` | **持久化快照** `factor_quality_snapshot` | 7 个已准入因子带阻塞性筛查失败、而冻结契约 `conflicts` 为空 | **读的是已知有误的快照**（见二） |
| `factor_coverage_screen` | **活体** `factor_daily`（3 个窗口） | 6 个因子 `ELIGIBILITY_FAIL`（**三个窗口都是 PASS 11/17**） | **真问题**，即裁决 4 的标的 |
| `fold_coverage` | 逐折横截面宽度 | F01/F02 各 15 天、F03 4 天低于门槛 | **真问题**，同上（裁决 4 / 3.1b） |
| `report_path` | 报告可序列化 + 工件目录 | **PASS** | — |

### 二、三套覆盖率度量的精确对账（本轮只读查询，工件已存）

证据：`/data/zhangyudong/qrv2_research_artifacts/structural_coverage_scope_reconciliation.txt`
（只读查 `dfs://QuantResearchV2Test.factor_quality_snapshot`，未写任何行）。

**持久化快照里只存在两个 `STRUCTURAL_COVERAGE` scope**：
`SCREEN_TRAIN2020_2024_STRUCTURAL_COVERAGE_V1` 与 `_V2`，**都是 `as_of_date=2026-08-04`、各 70 行**。
**修正后的 `factor_daily` 筛查（`screen_factor_daily_coverage.py`，今日 00:55 跑过）从未被持久化**——
它只产出工件、不写快照。

| 因子 | 持久化快照（源表版，08-04） | 活体筛查（`factor_daily`，3 窗口） | 原始源普查（本轮，同窗） |
|---|---|---|---|
| 372 / 384 / 389 / 390 / 406 | **FAIL** | **FAIL** | **37.2%–39.4% 可发布** |
| **375（GP_006）** | **无行** | **FAIL**（median 0.3705/0.4627/0.4641） | **35.1% 可发布 + 18.18% 冲突隔离** |
| **398 / 464** | **FAIL**（`MISSING_MONTH_OR_LOW_MONTHLY_COVERAGE`，而 median **0.9978**） | **PASS**（median 0.9967、min_month 1.0000） | 74.9% / 76.8%（dense 档） |
| **206 / 443** | **无行** | **PASS** | 76.8% / 未普查（无 metadata 行） |
| 357 / 381 / 405 / 425 / 471 / 484 / 96 | PASS | PASS | 78.9%–86.9% |

**权威结论：失败的是六个 —— 372 / 375 / 384 / 389 / 390 / 406。**
**两套互不引用的方法在三个窗口上一致**：活体筛查（**按 universe 成员分母、逐日取中位**）
与本轮原始源普查（**按源键分母、全窗池化**）。

**持久化快照的那份七因子清单在两个方向上都不对**，而这不是新发现——
`screen_factor_daily_coverage.py` 的 docstring **早已写明**：
准入那次筛查跑在 **YesLab 源表**上、而冻结链消费的是 **`factor_daily`**，
该错配产生了**一个假阴性**（375 **从未被结构筛查**，实测 43.9–62.8% 覆盖，低于 C6.2 的 0.70 门）
**和两个假阳性**（398/464 在**源表**上 1159 / 1185 天（共 1212），
而 `factor_daily` 对两者都是 **1212/1212 天、60/60 月**）。
预检自己也打了 NOTE：*"blocking status despite median coverage 0.9978 >= 0.7; inspect the scope's other criteria"*
——**本节就是那次 inspect。**

**另有三个冻结因子从未被结构筛查**（快照里无行）：**206、375、443_reg30_max20_v1**。
其中 206 与 443 经活体筛查为 PASS（0.9842–0.9862 / 0.8864–0.9883），
**而 375 是三套度量里最差的一个**。即：**唯一从未被筛查、又确实有问题的因子，
正是那个问题最重的**——这不是巧合，而是"筛查跑在错误的源上"的直接后果。

### 三、为什么"把修正版发布进去"不能顺手做（新增决定 L）

直觉的修法是：把 `factor_daily` 版筛查发布成一个新 scope，门 2 就读到对的了。
**核实后不成立**：`verify_admission_consistency.py` 的
`BLOCKING_SCOPE_SUBSTRINGS = ("STRUCTURAL_COVERAGE",)` 是**子串匹配**，
故**新旧两份 scope 会同时被当作阻塞**，门 2 的清单会从 7 个变成 **8 个并集**
（含两个已知假阳性）——**发布修正版反而让门 2 更错**。

**决定 L（新增）：结构覆盖率的权威 scope 是哪一个，门 2 该读谁。**

- **(L-i) 发布修正 scope + 让门 2 只读指定 scope**：需要把子串匹配改成"权威 scope 名单"，
  **并向 `dfs://QuantResearchV2Test` 写入一个新 scope**（additive、不动既有行）。
  **最正确**，但**是一次 DB 写入**，故与**决定 K** 相关（若 K 选独立库/前缀，L 的写入目标随之变）。
- **(L-ii) 不发布，改为在冻结契约 `conflicts` 数组里记录这六个**：这正是门 2 的字面要求
  （*"the contract's conflicts array must record this or the factors must be removed"*），
  但**它会改动冻结契约**（含其 `data_version`），属 owner 行为，
  **且它就是裁决 4 (iii) 的机械后果**——故 **L-ii 与裁决 4 应一并定，不该分两次**。
- **(L-iii) waive 门 2**，理由写清（"快照基于错误的源、已由活体筛查取代，权威清单是这六个"）：
  **改动最小**，但**把一份已知有误的快照留在库里继续被别的消费者读到**
  （准入决策表、未来任何审计），**且 waive 是一次性动作、不会修好数据**。

**倾向 (L-i) 先、(L-ii) 随后，(L-iii) 只作临时措施且必须写明过期条件。**
**不代做**：(L-i) 要写库并改一道治理门的匹配语义，(L-ii) 要改冻结契约，
(L-iii) 是放弃一道门——**三者都是 owner 的动作**。

### 四、对裁决 4 的直接影响

裁决 4 的 (iii)「复议那六个因子」**现在有了准确的名单与准确的依据**：

1. **是六个，不是七个、也不是八个**：398/464 应从复议名单里**去掉**（假阳性，
   `factor_daily` 上 1212/1212 天、60/60 月、median 0.9967），
   375 必须**加进来**（假阴性，且是最差的一个）。
   > **续三十二 勘误（保留原文、不重写历史）**：本条把 398/464 定性为"假阳性"**不成立**。
   > 取出快照里两者的完整行后，九个字段里**八个达标**（median 0.9978、p05 0.9944、min 0.9906、
   > `duplicate_row_count` 0、day_ratio 0.9563/0.9777），**唯一失败的是
   > `minimum_monthly_day_ratio = 0.0000`（有一整月观测天数为零）**。
   > **故"去掉"这个结论暂时不变，但它现在是决定 L 的函数，不是既定事实**：
   > 若 L 选"门 2 管能不能被打分"，398/464 必须**留在**名单里并解释那一个缺月。
   > 详见续三十二 §五。
2. **六个的失败在三个窗口上稳定**（2014–2019 / 2020–2024 / 2025 都是同样六个、都 PASS 11/17），
   故**不是某段行情的偶然**。
3. **其中五个的成因是"源里有行、值是 NULL"**（60.6%–62.8% 不可用），
   **一个（375）另有 18.18% 的冲突隔离**，而冲突隔离**在无解决规则前不可恢复**（决定 G）。
4. **门 2 与门 3/4 不是同一个问题**：门 3/4 是**真实的覆盖率不足**（裁决 4 的标的），
   门 2 是**读错了工件**（决定 L 的标的）。
   **把三者混为"覆盖率问题"会导致 waive 掉一道本该修的门、或修一道本该 waive 的门。**

### 五、门槛

**本节零代码改动**（只读查询 + 文书），故沿用续二十八的实测值：
**全量 1582 passed / 45 skipped**、`ruff check src tests` 清、`mypy src` **158 文件清**。
**未写任何数据库**（本轮对 `factor_quality_snapshot` 只发 `select`）；
**未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`；
**未改动冻结契约、未 waive 任何门、未发布任何 scope**——三者都属决定 L / 裁决 4。

## 勾选审计续三十（2026-09-06 19:10）：存在**两条互不相干的因子谱系**，冻结基线走的那条完全不经过本计划治理的去重器——据此更正我在裁决 4 里的机制归因与决定 K 里的一个数字

### 一、两条谱系（此前没有任何文书把它们并列过）

本轮为闭合"普查测的是源、而冻结链消费的是 `factor_daily`"这个环而做的只读对账，
查出了一件结构性的事：**平台里有两条互不相干的因子谱系**，
而 **C 计划治理的是 (A)、冻结集基线消费的是 (B)**。

| | **(A) 治理 / 去重谱系** | **(B) 冻结集基线谱系** |
|---|---|---|
| 数据源 | **YesLab `factor_data*` 的 `value` 列** | **Tushare 全市场行情 + 公式重放** |
| 路径 | `normalize_technical_factor_source` → `deduplicate_factor_observations` → PIT → 落表 | `replay_factor_values` → 按日 universe 过滤 → 落表 |
| 落表 | `factor_observation_clean`：**226,568,607 行、81 个 factor_id、2014-01-02..2026-07-31** | `factor_daily`：**106,598,549 行、17 个因子、2014-01-02..2025-12-31、每因子 2919 天** |
| 隔离表 | `factor_quarantine`：**536,627 行** | **无**（重放路径不经去重器） |
| 消费者 | **遗留链**（criterion-303 基线，14 个 id） | **冻结集链**（Task 3.1 及以后、4.1 的新信任根） |
| 与本轮普查的关系 | **普查测的就是它的源** | **普查测不到它**（它不读 YesLab） |

**证据**：`materialize_frozen_factor_daily.py` 的 docstring 明写
*"Replays each frozen factor over the requested window **from Tushare market data**"*，
且该脚本对 `deduplicate` / `factor_observation_clean` / `normalize_technical` **零命中**；
`walk_forward_factor_daily.py` 的 docstring 亦明写冻结路径用 `factor_daily`
**"instead of `factor_observation_clean`"**、且两者缓存键空间**天然不相交**。
**故 (B) 完全不经过去重器**——本轮所有关于去重、冲突隔离、`[Infinity]` 的工作**都不在冻结基线的路径上**。

### 二、对账结果：两条谱系在每个因子上的可用格数相差仅约 2%

证据：`/data/zhangyudong/qrv2_research_artifacts/factor_daily_vs_source_census.txt`（只读，6 条查询）。

| factor_id | `factor_daily` 行数 | 天数 | 源可发布键（全区） | 比值 | 源可发布键（2020–24） | 2020+ 行数 |
|---|---|---|---|---|---|---|
| yeslab_206 | 7,335,697 | 2919 | 7,433,313 | 0.987 | 5,526,788 | 4,864,245 |
| yeslab_357 | 7,633,151 | 2919 | 7,672,703 | 0.995 | 5,768,598 | 4,813,956 |
| **yeslab_372** | **3,798,999** | 2919 | 3,827,928 | 0.992 | 2,743,527 | 2,383,174 |
| **yeslab_375** | **3,463,207** | 2919 | 3,368,857 | 1.028 | 2,554,780 | 2,356,031 |
| yeslab_381 | 7,759,064 | 2919 | 7,934,673 | 0.978 | 5,744,684 | 4,934,859 |
| **yeslab_384** | **3,882,191** | 2919 | 3,891,162 | 0.998 | 2,814,488 | 2,476,706 |
| **yeslab_389** | **3,921,867** | 2919 | 3,950,069 | 0.993 | 2,844,348 | 2,459,984 |
| **yeslab_390** | **3,884,216** | 2919 | 3,942,763 | 0.985 | 2,834,466 | 2,463,428 |
| yeslab_398 | 7,699,059 | 2919 | 7,187,066 | 1.071 | — | 4,923,049 |
| yeslab_405 | 7,759,064 | 2919 | 7,934,673 | 0.978 | — | 4,934,859 |
| **yeslab_406** | **3,700,662** | 2919 | 3,697,315 | 1.001 | 2,687,577 | 2,380,873 |
| yeslab_425 | 7,725,690 | 2919 | 7,883,970 | 0.980 | — | 4,928,295 |
| **yeslab_443_reg30_max20_v1** | **7,415,381** | 2919 | **无法普查**（无 metadata 行） | — | — | 4,877,692 |
| yeslab_464 | 7,699,059 | 2919 | 7,371,584 | 1.044 | — | 4,923,049 |
| yeslab_471 | 7,493,733 | 2919 | 7,392,559 | 1.014 | — | 4,788,768 |
| yeslab_484 | 7,780,414 | 2919 | 7,743,517 | 1.005 | 5,754,732 | 4,939,817 |
| yeslab_96 | 7,647,095 | 2919 | 7,789,102 | 0.982 | 5,678,139 | 4,914,673 |

**这比"三套方法一致"更强**：不是同一份数据换分母，而是**两条独立谱系**
（一条读 YesLab 的 `value`、一条从 Tushare 行情重放公式）在**每个因子**上给出几乎相同的可用格数
（比值 **0.978–1.071**）。**六个稀疏因子在重放谱系里同样只有 dense 因子约一半的行数**
（**3.46–3.92M** vs **7.34–7.78M**），且 **17 个因子的天数都是 2919**
（**无缺日**，与按日测量"空日总数为 0"一致）。
**故六个因子的稀疏是因子自身的性质，不是任一数据源的缺陷。**

**并补上了第 17 个因子**：`yeslab_443_reg30_max20_v1` 在源侧无法普查（`factor_metadata` 无其行），
但在 `factor_daily` 里有 **7,415,381 行 / 2919 天**，属 **dense 档**——
**故"17 个因子里有 1 个完全没被测过"这个缺口已部分闭合**（源侧仍无法测，消费侧已测）。

### 三、必须更正：我在裁决 4 里写的机制归因指错了谱系

续二十四/二十五 与简报 (3b)/(4) 写的是"六个因子的损失**全部来自源里有行、值是 NULL**"。
**这句对 (A) 谱系成立，但裁决 4 说的是 (B)**：冻结链的 `dropna` 作用在 `factor_daily` 上，
而 `factor_daily` 的稀疏来自**重放产出非有限值**（脚本的过滤条件是
`ts_code in universe[date] and np.isfinite(replay_value)`）**或 universe 过滤**，
**与 YesLab 的 NULL 没有因果关系**。

**正确的说法**：**两条独立谱系一致地给出同样六个稀疏因子**；
(A) 侧的机制已测（源明确说了没有值，60.6%–62.8% 不可用），
**(B) 侧的机制本轮未测**（要测需按因子统计重放产出非有限值的比例，以及 universe 过滤丢了多少）。
**结论不变、依据变了**：选项 (i) 插补仍然是为空格子造数（**两条谱系都没有值**），
(iii) 仍然成立且依据更强；但**"因为源是 NULL"这个理由对冻结链不适用**，
已据此更正简报与决策清单（**不删原文，加 dated 更正**）。

### 四、一个后果更重的推论：去重器**从未处理过任何一个有冲突的冻结因子**

`factor_observation_clean` 的 **81 个** factor_id 里，冻结 17 只占 **9 个**
（357 / 372 / 381 / 405 / 406 / 425 / 464 / 471 / 484），
**缺 8 个**（**206 / 375 / 384 / 389 / 390 / 398 / 443_reg30_max20_v1 / 96**）。
**而普查中唯一有真冲突的冻结因子 `yeslab_375`（1,721,876 个冲突键）恰好在缺失的 8 个里。**
在那 9 个中，普查显示**冲突数全为 0**（372 与 406 属稀疏档，但冲突为 0）。

**故：去重器的冲突隔离路径从未处理过任何一个冻结因子的冲突**，
`factor_quarantine` 的 **536,627 行不含任何冻结因子的冲突组**。

**对三个决定的影响（都已写入决策清单）**：

- **决定 F**（`[Infinity]` 格式）：不仅"到不了源读取路径"，
  而且**它所在的谱系根本不是冻结链的谱系**——**优先级应进一步下调**。
- **决定 G**（去重偏差）：同理**对冻结基线零影响**；它影响的是**遗留链**
  （criterion-303 基线读 `factor_observation_clean`）。
  **故 G 的正确定位是"遗留链的去重语义"，而不是"冻结链的"**——
  续二十三说它"生效条件是 B/C 把基本面路径接进去"仍然对，
  **但即便接进去，动的也是 (A) 谱系，不是冻结基线**。
- **决定 A**（以哪条路径为准）：**这条事实把 A 的代价具体化了**。
  若选 **(A2)**（以 registry + `create_factor_set` 为准，即走治理谱系），
  则冻结集的数据要从 `factor_daily` 换成 `factor_observation_clean`，
  而后者**缺 8 个冻结成员**（含 375），
  且**首次把 375 的 172 万冲突键引进隔离**——
  **那不是"接线"，那是一次数据重建，并且会改变基线的数值。**
  **A 的代价此前从未被这样量化过。**

### 五、必须更正的一个数字（决定 K）

决定 K 里我写"该库里已有一份完成的 **6836 万行**冻结 17 物化"。
**实测 `factor_daily` 是 106,598,549 行**（2014-01-02..2025-12-31、17 因子、每因子 2919 天）。
**68,363,458 是其中 2020-01-01 之后的部分**，也正是 `materialization_status.json` 记录的 `total_rows`
——即**那份状态文件记的是一个批次的行数，不是整表行数**（该文件属另一条对话，本轮只读、未改）。
**决定 K 的论证不变（量级其实更大），数字已更正。**

### 六、门槛

**本节零代码改动**（只读查询 6 条 + 文书），故沿用续二十八的实测值：
**全量 1582 passed / 45 skipped**、`ruff check src tests` 清、`mypy src` **158 文件清**。
**未写任何数据库**（对 `factor_daily` / `factor_observation_clean` / `factor_quarantine` 只发 `select`）；
**未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`。
**方言记录**（供后续只读查询避坑）：`count(distinct col)` 在分组 select 里报
*"Cannot nest aggregate function"*，要用 `nunique(col)`；
而 `nunique` 作用在 **SYMBOL** 列上、且**无 group by** 的全表聚合会报
*"Array vector does not support SYMBOL or STRING type"*，要改成 `group by` 后在客户端数。

## 勾选审计续三十一（2026-09-06 19:40）：续三十标为"未测"的那一项已测——冻结链上六个因子的稀疏，成因是**它们自己的公式对约一半的格子无定义**

### 一、测的是什么，为什么必须测

续三十更正了裁决 4 的机制归因：那句"损失来自源里有行、值是 NULL"只对 **(A) YesLab 谱系**成立，
而裁决 4 的 `dropna` 作用在 **(B) 重放谱系**的 `factor_daily` 上，**(B) 侧的机制当时标为"本轮未测"**。
不测的代价是具体的：**裁决 4 的选项 (i) 插补与"换个更好的输入源"这两条路，
只有知道 (B) 侧的稀疏成才能判断有没有意义**。本节把它测了。

### 二、方法：同一装置下分解，唯一变量是因子公式

有界窗口 **2020-01-01..2020-03-31**，直接调用生产脚本自己的
`replay_factor_values`（**它返回的是 universe 过滤之前的帧**，故能在过滤前计数），
六个稀疏因子 **+ 两个 dense 对照**（484、381）。
**行情帧只建一次、八个因子共用**：316 个交易日（自 2018-12-11，即物化脚本用的 90 日回看）、
**1,153,644 行**，**全部来自 `artifacts/tushare_cache`**——
`QRV2_TUSHARE_TOKEN` **未设置**，而 `fetch_full_tushare_shsz_market` 在缓存不全时
**抛错而不是静默返回部分数据**（*"QRV2_TUSHARE_TOKEN is required when canonical market
cache is incomplete"*），**故缓存缺口不可能悄悄造成这些数字**。
universe 成员格数：**158,712**（58 个日期）。**全程只读，未写任何数据库。**
证据：`/data/zhangyudong/qrv2_research_artifacts/replay_sparsity_decomposition_2020q1.txt`。

| factor | 重放格数 | 有限 | **非有限占比** | `factor_daily` 行数 | **占 universe** |
|---|---|---|---|---|---|
| **yeslab_372** | 221,676 | 109,822 | **50.5%** | 79,816 | **0.503** |
| **yeslab_375** | 221,676 | 117,253 | **47.1%** | 87,803 | **0.553** |
| **yeslab_384** | 221,676 | 126,532 | **42.9%** | 93,340 | **0.588** |
| **yeslab_389** | 221,676 | 112,731 | **49.1%** | 81,420 | **0.513** |
| **yeslab_390** | 221,676 | 111,090 | **49.9%** | 81,481 | **0.513** |
| **yeslab_406** | 221,676 | 122,276 | **44.8%** | 90,440 | **0.570** |
| yeslab_484（对照） | 221,676 | 217,932 | **1.7%** | **158,712** | **1.000** |
| yeslab_381（对照） | 221,676 | 218,149 | **1.6%** | 158,511 | **0.999** |

### 三、结论：稀疏是因子公式的性质，不是数据可得性的性质

**同窗口、同行情帧、同 universe、同重放装置，唯一变量是因子自己的公式**：
六个被点名的因子对 **42.9%–50.5% 的格子产出非有限值**，覆盖率 **0.503–0.588**；
两个对照只有 **1.6%–1.7% 非有限**，覆盖率 **0.999–1.000**——
**`yeslab_484` 的 `factor_daily` 行数（158,712）恰等于 universe 成员格数**，
即它在重放谱系里是**逐格全满**的（**这也从第三个方向再次确认了 (3b) 的"dense 对照成立"**）。

**对裁决 4 的两个直接后果**：

1. **选项 (i) 插补在正确的谱系上同样不成立**，而且理由比 (A) 谱系更硬：
   缺的不是"数据没到"，而是**该因子的公式对这些格子无定义**（产出 NaN）。
   **为公式无定义的格子插值，是在给一个没有定义的量编造数值。**
2. **"换个更好的输入源"救不了**：输入是齐的（316 个交易日全部命中缓存、1,153,644 行行情），
   **而两个对照用完全相同的输入达到了 ~100% 覆盖**。

**故裁决 4 的现实选项收窄为 (ii) 与 (iii)**：
(ii) 同口径子群（其代价计划已定量：只保留 **19.5%–24.6%** 的训练行），
或 (iii) 复议那六个因子。**这与本节之前所有独立测量指向同一处。**

### 四、如实记录：本次测量脚本有一列是坏的

`in_universe` 列**对所有因子都输出 0**：`load_official_universe` 返回的字典键是
**pandas Timestamp**，而查找时用了 `datetime.date`，**故没有一次查得中**。
**该量无需重跑即可恢复**为 `fd_rows / finite`，六个稀疏因子为 **0.722–0.749**，
与另行实测的"源键/universe 成员"比值 **0.771–0.805** 一致（同为"源侧格子比 universe 多约 25%"）。
**其余各列不受影响**：`replay_cells` 与 `finite` 直接来自重放输出，`fd_rows` 是对
`factor_daily` 的直接计数。**缺陷与恢复方式都写进了工件本身**，
而不是只在审计里提一句——工件会被单独引用，它自己得说清哪一列不能用。

### 五、门槛

**本节零代码改动**（一个 `/tmp` 下的只读测量脚本 + 文书；未新增仓内脚本，
因为它调用的是生产脚本自己的函数，复制一份反而会产生第二个可能与生产漂移的实现）。
故沿用续二十八的实测值：**全量 1582 passed / 45 skipped**、`ruff check src tests` 清、
`mypy src` **158 文件清**。
**未写任何数据库**；**未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`；
**未改动任何已物化数据、版本指针或 `dropna` 口径**——
**裁决 4 的选项一个都没被预判，但 (i) 与"换源"两条现在有了反对证据。**

## 勾选审计续三十二（2026-09-06 20:10）：稀疏机制已用**生产计算器本身**复核并量化，但机制**不是本轮发现的**；决定 L 得到三处新事实，两道 FAIL 门确认不是同一个问题

### 一、先更正归属：这个机制昨天就写在代码里

`research_v2/scripts/verify_frozen_fold_coverage.py` 的 docstring（**2026-09-05**）已写明：

> *"the frozen 17 contain six factors that raise a **sign-capable base to a real power*
> *(`pct_chg ** x` and `tanh(delta20(vol|turnover)) ** x`), which numpy leaves undefined for*
> *falling stocks and declining volume. Their coverage therefore equals the market's*
> ***advancing breadth***."*

即"带符号的底取实数次幂 → 下跌股与缩量日无定义 → 覆盖率等于市场上涨广度"，
**形式、成因、后果三样都已写下**。故**续三十一把"稀疏是因子自己的公式"作为新测得的结论，
在归属上不成立**：机制此前已被记录。续三十一真正新增的是**"在冻结链实际消费的重放谱系上、
同一装置下分解，唯一变量是公式"**这一步，即把一个断言变成了对照测量。
本节继续这一步，**但不再称其为发现**。

### 二、本轮做的：跑生产计算器本身，并配一个同源同库的对照

**不重实现任何公式**（本计划反复踩过的坑：第二实现会与生产漂移）。
做法是 **import 六个生产计算器**（`数据库/factor_calculators/*.py`，各 24–38 行，**未修改**），
只替换它们自己的取数缝 `fetch_data`，喂进**只由 `artifacts/tushare_cache` 构造**的行情帧：
**203 个交易日**（2019-06-03..2020-03-31）、**3,989 个 ts_code**、5 个字段、
每字段 809,767 格，**7.8 秒建成，无数据库、无网络**。
计数窗口 **2020-01-01..2020-03-31**（58 日），之前 145 日作为 warm-up 吸收；每因子计 **231,362 格**。

| factor | 生产计算器 | 被取幂的底 | nan% | **P(nan \| 底<0)** | 解释了全部 nan 的 | 底≥0 仍 nan |
|---|---|---|---|---|---|---|
| **yeslab_372** | `GP_Factor_003` | `pct_chg` | 52.5% | **0.9999** | 0.876 | 5,834 |
| **yeslab_375** | `GP_Factor_006` | `Δ20 vol` | 49.3% | **1.0000** | 0.824 | 8,000 |
| **yeslab_384** | `GP_Factor_015` | `Δ20 vol` | 45.3% | **0.9743** | 0.874 | 1,139 |
| **yeslab_389** | `GP_Factor_020` | `pct_chg` | 51.3% | **0.9999** | 0.898 | 2,921 |
| **yeslab_390** | `GP_Factor_021` | `pct_chg` | 51.5% | **0.9998** | 0.893 | 3,496 |
| **yeslab_406** | `GP_Factor_037` | `Δ20 turnover` | 47.1% | **0.9999** | 0.867 | 371 |
| yeslab_381（对照） | `GP_Factor_012` | **无 pow** | **4.9%** | **0.0066** | 0.062 | 1,380 |

六个因子的最佳解释底**与其公式真正取幂的底逐一 MATCH**（脚本对三个候选底都算了，
未被取幂的两个候选精度只有 0.44–0.57，即随机水平）。
**对照 `yeslab_381` = `GP_Factor_012`，与 372/375/384 同出 GP 源 CSV `20260215_163844.csv`、
同存 `factor_data8`**，表达式 `ts_max_5(mul(cs_zscore(pct_chg), amount))`——**不含任何 pow**；
其 nan 仅 4.9%，且 **P(nan|pct_chg<0)=0.0066**：下跌日对它几乎从不产生 nan。
**这是双向证伪**：同一底、同一窗口、同一数据，六个稀疏因子的精度 0.9743–1.0000，
同源同库对照 0.0066，**相差约 150 倍**。
证据：`/data/zhangyudong/qrv2_research_artifacts/sparse_factor_pow_mechanism_production_calculators.txt`（含脚本副本）。

### 三、一处只有本轮才看到的具体细节：384 的 0.9743 不是噪声，是"最低价恰好是整数"

五个因子的 P(nan|底<0) 是 0.9998–1.0000，**唯独 `yeslab_384`（GP_015）是 0.9743**。
它的指数是 **`low`（当日最低价）**，而**负底的整数次幂在实数上有定义**
（工件 PART 0 已列：`(-0.02)**2.0 = 0.0004`，`(-0.02)**0.5 = nan`）。实测：

- 底<0 却仍有限的残差 = **2,418 格**，其中 **`low` 恰为整数的占 100.00%**；
- 而在**全部**底<0 格里 `low` 为整数的基率只有 **2.65%**（**约 38 倍富集**）；
- 反向：底<0 **且** `low` 为整数的 2,493 格里，**96.99% 确实有限**。

**含义**：384 的缺失**不是**干净的"缩量 ⇒ 缺失"，而是**额外取决于当日最低价是不是整数**——
一个与任何经济含义无关的数值巧合。**六个里它是最不可解释的一个**：
若将来有人试图"修"这六个因子，384 的行为会随价格水平漂移。
证据：`/data/zhangyudong/qrv2_research_artifacts/gp015_integer_low_residual.txt`（含脚本副本）。

### 四、对裁决 4 的净增量：不改变选项集，只把 (iii) 的"复议"从动词变成两个可答的问题

裁决 4 的三个选项在覆盖率维度上**早已全部实测**（简报 2026-09-06 04:52 表：
(i) `≥11/17` 与 (iii) `C6.2 PASS 11` 都 **18/18 通过、exit 0**，(ii) **定义性 FAIL**；
`c62pass11` 逐折 train 中位 1446/1674/2029/2494/2651/2737，对遗留 14 的 1402/1615/1949/2464/2656/2794
相差 **−2.0% 到 +4.1%**）。**故本节不动选项集**，只回答 (iii) 的"复议"到底复议什么：

1. **能不能在不改变因子定义的前提下把它们变稠密？不能。**
   缺失格 = **公式无实数值**的格。任何"修复"（`sign(x)·|x|^y`、把底截断为正、取绝对值）
   **都会改变因子的数值，即产生另一个因子**——那是重新定义而非修复，
   并且会改变基线数字，属于**应新发 V2 而非就地改 `factor_set_frozen_v1.json`** 的同类问题。
2. **它们的历史 Rank-IC 是在什么样本上算出来的？**（**本轮未测**）
   六个 docstring 各标 Rank-IC **0.0522–0.0589**，来源同为 `20260215_163844.csv`。
   而它们的可算格**正是上涨日 / 放量日**（384 还额外偏向整数价格）。
   **故这六个的 Rank-IC 是一个符号选择子样本上的 IC，不是全横截面的 IC。**
   重算它需要触碰评价链，超出本轮只读范围，故**只记录为 (iii) 该问的下一个问题**：
   复议的对象**不只是"覆盖率够不够"，而是"这六个的入选证据本身是否成立"**。

### 五、决定 L 的三处新事实：gate-2 读的清单被误读过，而且它与活体筛查不是同一台仪器

**事实 1（**不是新发现，是摘要表落后于 §3f**）：无结构覆盖证据的冻结成员是 3 个，摘要表只写了 1 个。**
决定清单 **§3f 已明确记录**"另有三个冻结因子在快照里根本没有行：206、375、443_reg30_max20_v1"，
但 **§0b 的一屏决策表（第 46 行）仍只写"375 必须加进来（快照里根本没有它的行）"**——
**读者只看那张表就会以为只有 1 个。** 本轮独立复核确认 §3f 的数字：
`factor_quality_snapshot` 对冻结 17 共返回 **178 行**、其中 STRUCTURAL_COVERAGE **28 行**
（= 14 个 factor_id × 2 个 scope；§3f 记的"各 70 行"是**两个 scope 各自的全量**，
不限于冻结 17，两者不矛盾），**`yeslab_206`、`yeslab_375`、`yeslab_443_reg30_max20_v1`
在两个 scope 里都没有任何行**。即 gate-2 对 17 个成员中的 **3 个零证据**。
（活体筛查在 2014-2019 窗对 206 报 median 0.8614 PASS、对 443 报 0.8864 PASS，
故它们大概率没问题——但**门里没有记录**，与 375 的处境同类。）

**事实 2：V1 与 V2 是同一次测量的两份标注，`admission_consistency` 把它数成了两倍。**
14 个因子的 `structural` 指标字典**逐一完全相同**（`expected/valid_day_count`、`day_ratio`、
median、p05、minimum、`minimum_monthly_day_ratio`、`duplicate_row_count`、`coverage_gate_pass`
九个字段全等，**14/14 相同**），`eligibility_status` 也 14/14 相同；
**唯一差别是 reason_codes 的粒度**：V1 只有裸 `COVERAGE_GATE_FAIL`，V2 追加具体子因。
故前置门实跑打出的 **"blocking failures: 14 across 7 admitted factor(s)"是 7 个发现被记了两遍**
（V1 失败 7 + V2 失败 7 = 14 行，**distinct 因子 7 个**）。
**对豁免簿记的直接后果**：override / waiver 必须**按 factor 记，不能按行记**，
否则要么需要 14 条 override，要么会漏掉一半。

**事实 3（最重要）：398 与 464 不是"快照假阳性"，它们败在一个完全不同的判据上。**
续二十九据此建议"把 398 与 464 从复议名单里去掉"，理由写的是"持久化快照的**假阳性**"。
**这个理由不成立**。两者在快照里的完整行是：
`valid_day_count` **1159 / 1185**（`day_ratio` 0.9563 / 0.9777，≥0.80 ✓）、
median **0.9978**、p05 0.9944、min 0.9906（三项覆盖判据全过 ✓）、`duplicate_row_count` **0** ✓，
而 **`minimum_monthly_day_ratio = 0.0000`**——**有一个整月的观测天数为零**（缺 53 / 27 个交易日）。
`record_structural_quality_snapshot.py:107-131` 的 gate 是**六个条件的合取**，
而 `COVERAGE_GATE_FAIL` 是**任一条件失败时都贴的总标签**、具体子因另行追加；
故"median=0.9978 却 COVERAGE_GATE_FAIL"**不是自相矛盾，而是标签过粗**。
V2 行确实追加了 `MISSING_MONTH_OR_LOW_MONTHLY_COVERAGE`；**V1 行一个子因都没追加**，于是被读成"覆盖率失败"。
（`verify_admission_consistency` 已有对应的 NOTE——"blocking status despite median coverage 0.9978 >= 0.7;
inspect the scope's other criteria"——**它指对了方向，但没有把那个 other criterion 取出来**；本轮取出来了。）

**为什么这让决定 L 变成可决定的**：两份清单**不是同一台仪器的两次读数**，而是两台不同的仪器——

| | 持久化快照（gate-2 现在读的） | 活体筛查 `screen_factor_daily_coverage.py` |
|---|---|---|
| 源 | 每日 **RankIC `sample_count`**（对合成参照因子 `yeslab_0`） | **`factor_daily` 行数** |
| 分母 | 参照因子同日 `sample_count` | 同日 `universe_daily` 且 `is_member=1`（C6.3） |
| median 阈 | **≥0.80** | **≥0.70** |
| p05 阈 | **≥0.70** | **≥0.30** |
| 月度阈 | **≥0.70** | **≥0.90** |
| 量的是 | 该因子当日**能否被打分** | 该因子当日**有没有行** |

**故"快照说 398/464 失败、活体说 PASS"不是矛盾，是两件事**：
它们在 `factor_daily` 里逐日齐全（1212/1212、每月 1.0000），
但在 RankIC 侧**有一整月没有任何可打分横截面**。
**决定 L 因此不是"选一份对的、扔一份错的"，而是明确 gate-2 该管哪一件事**：
管"有没有数据"（则应读 `factor_daily`，398/464 PASS，但 **206/375/443 仍需补测**），
还是管"能不能打分"（则应保留快照口径，**398/464 的缺月必须先解释**）。
**两个方向给出不同的复议名单，且方向相反**——这正是把它单列为一个决定、
不与裁决 4 合并的理由；**本轮之前没有证据支撑这个区分。**
证据：`structural_coverage_snapshot_full_rows.txt`、`structural_coverage_snapshot_gaps_and_double_count.txt`（均含脚本副本）。

### 六、豁免桥已端到端验证，且"只读"已核到代码级并用 git 证明

前置门带三个豁免实跑到底：**`FROZEN BASELINE PRE-FLIGHT: CLEARED`**，
`admission_consistency` / `factor_coverage_screen` / `fold_coverage` 三道各记 **exit=1 + WAIVED + 完整理由串**，
`blocking: none`；另两道 `backfill_completeness` 与 `report_path` 实跑 **PASS**。
**这是机制测试，不是放行**：豁免理由串自己写着 *"MECHANISM TEST ONLY … not a clearance, no decision taken, script writes nothing"*。
**"script writes nothing" 本轮核到了代码级**：`preflight_frozen_baseline.py` 用 `runpy` 在进程内跑五个门脚本；
五个门里**唯一有写能力的是 `screen_factor_daily_coverage.py --json`**（`args.json.write_text`），
而 `--json` **默认 `None`** 且前置门不传它；`verify_frozen_report_path.py` 只 `json.dumps` 后**打印字节数、从不落盘**。
**事后 `git status` 只有另一个对话的 `materialization_status.{json,md}` 两个 ` M`，本轮未新增任何改动。**
证据：`preflight_waive_bridge_cleared_zero_writes.txt`。

**这条为什么值得记**：豁免机制有 15 条单元测试，但**单元测试证明的是豁免逻辑，
不是"真实五道门 + 真实豁免"能走通**。现在走通了，
故**裁决 4 一旦落地，Task 3.1 的解冻路径在机制上已无障碍**——余下的纯粹是决定。

### 七、门槛

**本节零代码改动**（四个 `/tmp` 下的只读测量脚本 + 文书；未新增仓内脚本，
理由同续三十一：它们 import 的是生产计算器与生产脚本本身，复制一份反而会产生第二个可能漂移的实现）。
**五个工件 + 四个脚本副本**已存 `/data/zhangyudong/qrv2_research_artifacts/`。
沿用续二十八实测值：**全量 1582 passed / 45 skipped**、`ruff check src tests` 清、`mypy src` **158 文件清**。
**未写任何数据库**（四个脚本全是 SELECT 或纯缓存文件读）；
**未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`；
**未改动契约、版本指针、门阈值或 `dropna` 口径**；**裁决 4 与决定 L 一个都没被代做**——
本轮只让两者**各自少一个未知**：裁决 4 少了"(iii) 复议什么"，决定 L 少了"两份清单为何不同"。

## 勾选审计续三十三（2026-09-06 20:50）：398/464 的缺月查明是**左端截断**，故续三十二那句"取决于决定 L"应撤回；顺这条线查出**门 2 有一处结构性盲区**——冻结 17 里有 **3 个成员在全部 20 个 scope 里没有任何一条通过判定**

### 一、先把上一轮自己的话收回来

续三十二 §五 事实 3 写：398/464 败在 `minimum_monthly_day_ratio = 0.0000`，
**"若决定 L 选'管能不能打分'，398/464 就必须留在名单里并解释那一个缺月"**。
**本轮把那个缺月解释了，结论是它不需要 L 来裁决**：
续二十九"去掉这两个"的**结论成立**，而**续三十二那句"取决于决定 L"过于保守，在此撤回**。
（续二十九的**理由**仍然是错的——它说的是"快照假阳性"，而真实原因是左端截断；
**结论对、理由错**这个组合比结论错更危险，因为它下一次不会再对。）

### 二、缺月是什么：左端连续截断，且是**逐因子**的，不是批次漏跑

`factor_rank_ic_daily`（快照的源，窗口 2020-01-01..2024-12-31）实测：

| factor | 首日 | 行数 | 缺的期望日 | 缺日按月分布 | 那些天**有行吗** |
|---|---|---|---|---|---|
| yeslab_0（参照） | 2020-01-02 | **1212** | 0 | — | — |
| 其余 14 个冻结成员 | **2020-01-02** | **1212** | 0 | — | — |
| **yeslab_464** | **2020-02-18** | 1185 | **27** | 2020-01 ×16、2020-02 ×11 | **27 天全部"根本没有行"** |
| **yeslab_398** | **2020-03-25** | 1159 | **53** | 2020-01 ×16、2020-02 ×20、2020-03 ×17 | **53 天全部"根本没有行"** |

**三个特征一起把它定性为左端截断**：
(1) 缺失日**从窗口第一个交易日 2020-01-02 起连续**（398 到 03-24、464 到 02-17）；
(2) 参照因子与其余 14 个成员**同日齐全**，故评价作业那些天**确实跑过**；
(3) 缺失是**没有行**而不是"行里 `sample_count` 为 0"，即那些天**根本没产出 IC**。
`factor_daily` 侧两者都是 **1212/1212 天、60/60 月**——因为冻结物化用
`replay_lookback_days=180` **主动回看窗外 180 天**，warm-up 在窗口内已满足；
RankIC 侧没有这层回看，故开头 ~60 个交易日无值（两者 `lookback_days` 都是 **60**）。

**是不是批次效应？不是。** 全表 **476** 个因子里 **45 个（9.5%）首日晚于 2020-01-02**，
分散在 **25 个不同首日**上；除 2020-01-03 有 15 个同日起，其余每个首日只有 1–3 个因子，
而 **398（2020-03-25）与 464（2020-02-18）各自都是"该首日唯一的一个"**。
**逐因子、不共享日期 ⇒ 是各自的 warm-up 长度，不是某一批漏跑。**

**故 L 在这两个因子上没有可选的东西**：无论门 2 读哪台仪器，398/464 都**不是**结构性稀疏因子
（`factor_daily` 上齐全、活体筛查 median **0.9978**、min_month **1.0000**）。
**续二十九的六个（372/375/384/389/390/406）在本轮第四次成立，而且这次理由是对的。**

### 三、顺着这条线查出的盲区：门 2 读的那个 scope，其 population 由它本该检查的那道筛查决定

`record_structural_quality_snapshot.py:236-241` 的候选集**不是**冻结 17，而是

```python
prior = target.query("... factor_quality_snapshot where scope_id='SCREEN_TRAIN2020_2024_DAILY_HAC_FDR_V2'")
candidates = tuple(... if json.loads(metrics_json).get("decision_status")
                        == "TRAIN_PREDICTIVE_PASS_PENDING_DQ")
```

实测该 predictive scope 有 **487 行**：`TRAIN_PREDICTIVE_FAIL` **404**、
`TRAIN_PREDICTIVE_PASS_PENDING_DQ` **70**、`UNSCREENED` **11**、`EXCLUDED_SYNTHETIC` **2**。
**结构快照的 population 就是那 70 个**（故它"各 70 行"，§3f 早先记的数字对上了）。

而 `verify_admission_consistency.py` 的 `BLOCKING_SCOPE_SUBSTRINGS = ("STRUCTURAL_COVERAGE",)`
——**它只读结构 scope**。于是：

> **一个因子若 predictive 筛查 FAIL，它就不会出现在结构快照里；
> 而门 2 只看结构快照，"没有行"在门 2 里不等于"有问题"。**
> **故门 2 对它本该拦的那一类失败完全看不见——恰恰因为那道失败把因子从它的视野里移除了。**

门 2 的 docstring 自述目的是 *"so a blocking screening failure cannot be admitted silently again"*。
**这个目的在它当前的 scope 选择下达不到。** 这不是阈值或源选错（决定 L 的议题），
**是 scope 选择本身的逻辑漏洞**，故单列为决定 M。

### 四、后果不是假设的：冻结 17 里有 3 个成员在**全部 20 个 scope** 里没有任何一条通过判定

`factor_quality_snapshot` 对冻结 17 共 **178 行、跨 20 个 scope**。逐个查那三个：

| factor | `factor_rank_ic_daily` | `factor_quality_snapshot`（20 scope） | `factor_quality_grade`（决策表的来源） | 准入决策表（2026-09-03）的记载 |
|---|---|---|---|---|
| **yeslab_206** | 1212 行齐全 | 出现的 **4 个 scope 全部 ELIGIBILITY_FAIL**：`DAILY_HAC_FDR_V1` 与 `_V2` 均 **`TRAIN_PREDICTIVE_FAIL`**（reason `TRAIN_DAILY_HAC_FDR_GATE`）、`ANNUAL_SUMMARY_PREFILTER_V1` **`ANNUAL_PREDICTIVE_FAIL`**、`VALIDATION2025_STRICT26_V1` `UNSCREENED` | **没有行** | **A / 10 分 / 训练ICIR 0.561 / 验证 0.733 / 决定=准入** |
| **yeslab_375** | **零行** | 出现的 **4 个 scope 全部 ELIGIBILITY_FAIL**，`decision_status=UNSCREENED`、reason **`NO_TRAIN_DAILY_RANK_IC`** | **有行：A / 9 分 / train_icir 0.632 / validation_icir 0.791 / regime_min 0.6548** | **A / 9 / 准入**（备注"隔离范围批准"） |
| **yeslab_443_reg30_max20_v1** | **零行** | **20 个 scope 里零行** | **没有行**（只有 canonical `yeslab_443`：**D / 6 分 / regime_min 0.2618**） | 经 **D1=A** 准入，理由"**修复变体全评估已过**" |

**三条各自不同，且都不在门 2 的视野里**：

1. **206**：`apply_admission_decisions.py:164` 有 `grade_alias = {"yeslab_206": "yeslab_206_median_abs_v2"}`，
   即它的 A 级来自**另一个 factor_id** 的分级；冻结集里的 `yeslab_206` 在 `factor_quality_grade` 里**没有自己的行**。
   **这不是身份错配**——已核实冻结物化对该 id 走 `materialize_alpha206_median_repair`
   （`materialize_frozen_factor_daily.py:62`，且第 118 行把它与 443 变体一起排除在
   `REPLAY_DEFINITIONS` 之外），**故 `factor_daily` 里存的确实是 median 修复版，别名与物化一致**。
   **问题在别处**：snapshot 侧那 4 条 FAIL 是对 id **`yeslab_206`** 记的，
   分级侧的 A 是对 id **`yeslab_206_median_abs_v2`** 记的——
   **两台仪器在两个不同 id 上各说各话，没有任何文书说明那 4 条 FAIL 针对的是修复前还是修复后。**
2. **375**：`factor_rank_ic_daily` **零行**、snapshot 的 reason 明写 **`NO_TRAIN_DAILY_RANK_IC`**，
   而 `factor_quality_grade` 里它有 **train_icir 0.632 / validation_icir 0.791**。
   **两台仪器对"375 有没有日频 RankIC 序列"给出了相反的答案。**
   而 375 同时是：六个稀疏因子里最差的一个（median **0.3705**）、
   冻结 17 里唯一有真冲突的（**1,721,876** 个冲突键，见决定 A 的代价量化）、
   且从未进入结构筛查。**三个"最差"集中在同一个因子上，而它对两台仪器分别是零行与 UNSCREENED。**
3. **443 变体**：`apply_admission_decisions.py:177-178` 对它是
   `if fid == REPAIR_VARIANT_443: grade = "REPAIR"` ——**grade 是硬编码字符串，不是评出来的**。
   **故决策表里"修复变体全评估已过"这句话，在 `factor_quality_grade`、
   `factor_quality_snapshot`（20 scope）、`factor_rank_ic_daily` 三处都找不到对应的行。**
   canonical 443 的评级是 **D / regime_min 0.2618**（2017 崩塌），**而变体没有任何评级记录**。

### 五、~~还有一处发散~~ → **本节初稿的断言是错的，已核实并撤回；真实情况窄得多**

**初稿写了什么**：`REPLAY_DEFINITIONS["yeslab_206"]` 记的是 `Bias_Vol_Demean_60_Calculator`
（"60日成交量**去均值相对偏离**"、`factor_data5`），而生产物化对同一 id 用 median 修复版，
"**修正只以 `if fid == MEDIAN_206` 的形式存在于一个消费者里**"，
"故**任何**经 `REPLAY_DEFINITIONS` 取 206 的审计，算的是去均值版、比的是 median 版"。

**为什么错**：逐个查了 `REPLAY_DEFINITIONS` 的消费者，两个断言都不成立。

1. **"只有一个消费者做特例判断"是错的——有 8 个，而且它们共用同一个 helper。**
   `rg -l materialize_alpha206_median_repair scripts/` 命中 **8 个脚本**：
   `audit_canonical_candidate_pairs`（**helper 的定义处**，并以
   `CandidateSpec("yeslab_206_median_abs_v2", "yeslab_206", is_alpha206_median_repair=True)`
   把该候选显式标出、第 109-111 行**只在 `not is_alpha206_median_repair` 时**才用
   `REPLAY_DEFINITIONS`）、`audit_pretraining_regime_redundancy`、`research_historical_window_eval`、
   `analyze_pool18_rank_turnover`、`analyze_pool23_rank_turnover`、`audit_extension_candidate_pairs`、
   `audit_candidate464_pairs`、`materialize_frozen_factor_daily`。
   `research_historical_window_eval.py:116` 甚至留有注释说明**为什么**仍要读那条定义：
   *"its market requirements come from the source yeslab_206"*——
   即取的是**行情字段需求**，不是公式。**这是有意的、正确的用法。**
2. **"任何经它取 206 的审计都比错了"也是错的——那两个审计比的是另一条谱系，本来就对。**
   `audit_formula_replay_consistency.py` 的 docstring 第一行是
   *"Replay source-file calculators against **stored YesLab factor observations**"*，
   第 1004 行读 `loadTable('dfs://YesLab','{store_factor}')`；
   `audit_formula_replay_tushare.py` 第 481 行同样读 `dfs://YesLab`。
   **在 YesLab 源谱系里，`yeslab_206` 就是去均值版（`factor_data5`），
   而 median 修复版另有其 id `yeslab_206_median_abs_v2`。**
   故这两个审计用 `REPLAY_DEFINITIONS["yeslab_206"]` = 去均值版**是正确的**，
   它们比的也是源表的去均值值。**没有比错。**

**真实情况（窄得多，但仍值得记）**：

> **同一个 id `yeslab_206` 在两条谱系里指两个不同的因子**——
> 源谱系（`dfs://YesLab`）里它是**去均值版**、median 修复版叫 `yeslab_206_median_abs_v2`；
> 冻结谱系（`factor_daily`）里 `yeslab_206` 存的是**median 修复版**
> （`materialize_frozen_factor_daily.py:62` → `materialize_alpha206_median_repair`，
> 且 `apply_admission_decisions.py:164` 的 `grade_alias` 与之一致）。
> **当前 8 个可能混淆的消费者全部经同一个 helper 正确绕行，两个源谱系审计本来就对，
> 故这是一处"潜在 id 冲突隐患 + 一组目前正确的绕行"，不是一个在产的缺陷。**
> **没有任何已发布工件因它而错。**

**为什么仍然要记**：这是**决定 A"两条互不相通的谱系"在单个 id 上的实例**。
决定 A 的现有文字讲的是谱系整体（`factor_observation_clean` 对 `factor_daily`、
缺 8 个成员、1,721,876 个冲突键），**从未提到有一个 id 在两条谱系里指两个不同因子**。
**这一条对决定 A 的 (A2) 选项有直接影响**：(A2) 要把冻结集换成经 registry 的
内容寻址身份，而**一个 id 指两个因子正是内容寻址要消除的东西**——
故 (A2) 的代价里应加上"必须给 206 的两个版本分配两个身份"，
而 (A1) 的代价里应加上"必须在文书里写明 `yeslab_206` 在哪条谱系里指哪一个"。
**风险是前瞻性的**：任何**新**消费者若经 `REPLAY_DEFINITIONS["yeslab_206"]` 取值
又与 `factor_daily` 比，会**静默**算错因子——8 个绕行都靠约定，没有一处是类型或断言强制的。

**方法论教训（本轮第二次自我更正，与续三十、续三十二同类）**：
初稿从"**一个**消费者里有特例"直接推广到"**所有**消费者都缺特例"，
而只查了 `audit_formula_replay_consistency.py` 一个文件的 grep。
**"某处有一个 workaround"不能推出"别处没有"**；这类断言必须逐个消费者查，
或者降级为"存在 id 冲突隐患"这种不需要穷举的表述。

### 六、这产生一个新决定（M），并**收窄** L

**决定 L 收窄**：398/464 出局已由左端截断解释清楚，**L 不再需要处理它们**。
L 剩下的真问题是 §3f-续三十二 (3) 那张仪器对照表（源/分母/三个阈值都不同），
以及**要不要给结构快照补 206/375/443 的行**（补 = 一次 DB 写入，故与**决定 K** 相关）。

**决定 M（新）**：门 2 该不该同时读 predictive scope，以及那 3 个"无任何通过判定"的成员怎么处置。

- **(M-i) 扩 `BLOCKING_SCOPE_SUBSTRINGS`**，把 `DAILY_HAC_FDR` 也纳入阻塞 scope。
  **改动最小、直接堵住盲区**；但会立刻让门 2 对 **206 与 375** 报 FAIL
  （**443 仍报不到，因为它零行**），**于是必须同时给出 override 或移出冻结集**——
  **即 M-i 会立刻把 206/375 推到裁决台前，而它对 443 无效。**
- **(M-ii) 让门 2 把"缺席"也当作发现**：对冻结集的每个成员，
  若它在任一阻塞 scope 里**没有行**，就报 `NO_EVIDENCE` 而不是静默通过。
  **这条能同时抓到 206/375/443 三个**（前两个 FAIL/UNSCREENED、第三个零行），
  **是三个选项里唯一能抓到 443 的。**
- **(M-iii) 维持现状，在冻结契约 `conflicts` 数组里人工记录这三条**。
  这正是门 2 报错信息的字面要求（*"the contract's conflicts array must record this or the
  factors must be removed"*），但**它改冻结契约**（含 `data_version`），属 owner 行为，
  且与 **L-ii 是同一处改动**——**故 M-iii、L-ii 与裁决 4 (iii) 应一并定，不该分三次。**

**倾向（不代做）**：**(M-ii)**。M-i 抓不到 443（零行不等于 FAIL），
M-iii 只修这一次、不修机制；**"缺席即发现"是唯一补上机制的一条，
且它对本轮查出的三个实例同时生效。**

### 七、如实记录：本轮一个测量脚本有一列是坏的，且是同一类错误的第二次

`rankic_month_gap.py` 的 Q1b 表里 **`ref_days` 列有 36 个 NaN 且数值不可信**（16/10/22/10…）：
我用 `pd.Series(list(...)).map(...)` 造的月份键与 `ref_per_month["m"]` 的索引不对齐，
`groupby` 因此错位——**与续三十一的 `in_universe` 列同一类错误（索引/键不对齐），这是第二次犯。**
**同表另两列不受影响**：`distinct_factors_with_positive` 与 `median_days_per_factor`
各自独立 groupby、数值自洽（451→476 单调上升、中位天数 15.9–23.0）。
**结论未依赖坏列**：左端截断的定性来自 `rankic_left_edge.py` 的逐因子首日普查
（45/476 晚起、25 个不同首日、398 与 464 各自唯一），那份没有对齐问题。
**缺陷与恢复方式都写进了工件本身。**

### 八、门槛

**本节零代码改动**（四个 `/tmp` 下的只读测量脚本 + 文书）。**四个工件 + 四个脚本副本**已存
`/data/zhangyudong/qrv2_research_artifacts/`：`rankic_month_gap_and_predictive_pool.txt`、
`rankic_left_edge_census.txt`、`predictive_verdict_vs_grade_instrument.txt`、
`factor_quality_grade_frozen17.txt`。
沿用续二十八实测值：**全量 1582 passed / 45 skipped**、`ruff check src tests` 清、`mypy src` **158 文件清**。
**未写任何数据库**（四个脚本全是 SELECT）；**未碰** `materialization_status.{json,md}`；
**未释放** `/tmp/frozen_chain_hold.txt`；**未改动契约、scope、门阈值或 `dropna` 口径**；
**裁决 4 / 决定 L / 新增决定 M 一个都没被代做**——
本轮把 **L 收窄**了一个问题（398/464 出局有了正确理由），并**新开 M**（门 2 的 scope 盲区）。

## 勾选审计续三十四（2026-09-06 21:25）：三条门槛在当前 HEAD 上重测通过；而续三十二我自己提的那条"IC 是符号选择子样本上算的"**未被数据支持，在此撤回**

### 一、门槛不再"沿用"，改为重测——因为它只花 91 秒

续二十八以来的四轮（续二十九～续三十三）在"门槛"一节都写**"沿用续二十八实测值"**。
**这个做法本轮终止**，理由不是怀疑，是成本：

```
cd research_v2 && pytest tests -q          → 1582 passed, 45 skipped, 3 warnings in 91.54s
cd research_v2 && ruff check src tests     → All checks passed!
cd research_v2 && mypy src                 → Success: no issues found in 158 source files
```

**三条全部在当前 HEAD（`665c231`）上重测通过，数字与续二十八逐一相同。**
证据：`/data/zhangyudong/qrv2_research_artifacts/fullsuite_reverified_at_head_2026_09_06_2115.txt`。

**为什么"沿用"是不必要的风险**：续二十八的 1582 是 **18:35** 测的，
而在它之前的 **09-05 21:34 / 21:40 与 09-06 02:34** 分别落了 `d708864`、`7f2e007`、`0cbbc88`
三个**改代码**的提交（后两个改了 `training/runner.py` 与候选驱动的评分路径）。
本轮核对 `git log --since="2026-09-06 18:35" -- research_v2/{src,tests,scripts}` **为空**
（此后只有本轮五个 docs 提交），**故 1582 这次恰好仍然有效**——
但"恰好有效"是靠查 git log 得出的，不是靠沿用得出的。
**若另一个对话在此期间落了代码，四轮文书里的门槛数字就会全是假的。**
**91 秒换掉这个风险，没有理由不换。**

**顺带对上一个数**：`45 skipped` **恰好等于** `tests/integration` 的收集数
（逐目录 `--collect-only`：golden 21 + integration 45 + leakage 14 + property 19 + unit 1528 = 1627；
1627 − 45 = **1582**）。集成测试全部因未设 `QRV2_INTEGRATION` 而 skip，
**故"全量"这条命令不写任何数据库**——这一点本轮据 skip 原因逐条确认
（如 *"set QRV2_INTEGRATION=1 to run the shadow switch/rollback drill"*）。

### 二、撤回：续三十二 §四.2 提的"这六个的 Rank-IC 是符号选择子样本上算的"**未被支持**

续三十二写：六个因子的可算格**正是上涨日/放量日**，
"**故这六个因子的历史 Rank-IC 是一个符号选择子样本上的 IC，不是全横截面的 IC**"，
并把"它们的入选证据本身是否成立"列为 (iii) 该问的下一个问题，同时标注**"本轮未测"**。
**本轮测了，结论不支持那句推断。**

方法：把两台仪器 join 起来——`factor_quality_grade`（分级，含 `train_icir`/`validation_icir`）
与结构快照 V2 的 `median_coverage_ratio`（= 该因子日 `sample_count` ÷ 参照因子 `yeslab_0` 同日值，
**正是"IC 是在多大横截面上算的"**）。**不只比冻结 17，而是比整个 join 得到的池子。**
实测 `factor_quality_grade` 共 **32 行**、结构快照 V2 共 **70 行**、**join 得 30 个因子**。

| 分组（以 C6.2 的 median 0.70 为界） | n | `median_coverage` 均值 | **`train_icir` 均值** | `train_icir` 中位 | `validation_icir` 均值 | `total_score` 均值 | 等级分布 |
|---|---|---|---|---|---|---|---|
| **稀疏**（<0.70） | 8 | 0.4816 | **0.5946** | 0.5991 | 0.7219 | 9.125 | A×4、S×2、B×1、C×1 |
| **稠密**（≥0.70） | 22 | 0.9961 | **0.6055** | 0.5880 | 0.7579 | 7.864 | B×7、A×6、C×4、D×3、S×2 |

**全池关联（n=30）**：

| 对比 | pearson | spearman |
|---|---|---|
| `median_coverage` vs **`train_icir`** | **+0.0544** | **+0.1128** |
| `median_coverage` vs `validation_icir` | +0.0980 | +0.0557 |
| `median_coverage` vs `total_score` | −0.2618 | −0.0695 |
| `median_coverage` vs `regime_min_official_icir` | −0.3092 | −0.2102 |

**结论：稀疏组与稠密组的 `train_icir` 实际上相同（0.5946 对 0.6055，稠密组还略高），
关联系数 +0.05/+0.11 在 n=30 下与零不可区分。**
**故"覆盖率低 ⇒ IC 被选择效应抬高"这个方向没有证据**；
续三十二那句"它们的 Rank-IC 是符号选择子样本上的 IC"**在量级上站不住**，**在此撤回**。

**四条限定，必须一起读**：

1. **n=30，很小**；且稀疏组 8 个里 **5 个正是被点名的冻结因子**（372/384/389/390/406），
   375 因**没有覆盖行**而根本不在 join 里。**故"稀疏组"与"被质疑的那六个"高度重叠，
   这不是一个独立的对照。**
2. **ICIR 量级相同 ≠ 估计对象相同。** 选择可以改变"哪些股票排在前面"而不改变相关系数的量级。
   **本轮否证的是"IC 被系统性抬高"，没有否证"IC 的语义与稠密因子不同"**——
   后者要重算 IC 才能判，仍**未测**。
3. **`total_score` 的差异是真的但不来自 ICIR**：稀疏组 9.125 对稠密组 7.864，
   而两者 ICIR 相同；差在 `regime_min_official_icir`（0.5936 对 0.4935，
   与覆盖率的相关 −0.31/−0.21）与等级分布（稀疏组 75% 是 A/S，稠密组 36%）。
   **n=30 下不足以定性，只记录为"分级结果与覆盖率有弱负相关，而 ICIR 没有"。**
4. **顺带量到一件与本轮无关但值得记的事**：`factor_quality_grade` **只有 32 行**，
   而 predictive 筛查有 **487** 个因子、通过 **70** 个。
   **即分级只曾施加在 32 个因子上**，而冻结 17 里有 **3 个**（206/375/443 变体）
   连分级都没有——**这与续三十三 §四的三条完全一致，是同一件事的另一个侧面。**

**对裁决 4 的净效果：(iii) 的论据少了一条，但没有变弱。**
续三十二曾建议 (iii) 的"复议"应指向"这六个的入选证据本身是否成立"；
**本轮把那条撤回了**，故 (iii) 的理由**回到简报已实测的三条**——
覆盖率（(ii) 定义性 FAIL）、**B2/B3 的 NaN 契约**（(i) 必须给 B2/B3 加插补或让二者退出基线集，
(iii) 不需要任何基线代码改动）、**横截面内的同质性**（(i) 下 B1 的等权秩集成会在同一横截面内
对不同因子子集取平均，(iii) 下每行都是同 11 个）——
外加简报 05:00 那条 (iii) 独有的下游成本（`raw_factor_count_min: 15` 的 yaml 声明与守卫测试）。
**换言之：裁决 4 应当在"工程口径"上定，不该在"这六个因子的 alpha 可能是假的"上定——
后者本轮没有找到证据。**

证据：`/data/zhangyudong/qrv2_research_artifacts/coverage_vs_icir_association.txt`（含脚本副本）。

### 三、门槛

**本节零代码改动**（一个只读测量脚本 + 三次只读验证命令 + 文书）。
**两个工件 + 一个脚本副本**已存 `/data/zhangyudong/qrv2_research_artifacts/`。
**门槛本轮重测而非沿用**：**全量 1582 passed / 45 skipped（91.54s）**、
`ruff check src tests` **All checks passed!**、`mypy src` **158 文件清**，
三者均在 HEAD `665c231` 上实测。
**未写任何数据库**（`pytest tests -q` 的 45 个 skip 全部因未设 `QRV2_INTEGRATION`，
测量脚本全是 SELECT）；**未碰** `materialization_status.{json,md}`；
**未释放** `/tmp/frozen_chain_hold.txt`；**未改动契约、scope、门阈值或 `dropna` 口径**；
**裁决 4 / 决定 L / 决定 M 一个都没被代做**——
本轮**撤回了自己上一轮加给 (iii) 的一条论据**，并把四轮沿用的门槛数字换成了实测值。

## 勾选审计续三十五（2026-09-06 21:55）：全量套件在 HEAD 上变红——hypothesis 钉住了 `expand_fundamental_events` 的**第二个**顺序依赖；本轮已修，并**证明**该修复不侵入决定 B；另补上写信任根那个脚本的零测试覆盖

**本轮是续二十八以来第一次改动 `src/`**，故门槛全部重测（见 §五），不再沿用。

### 一、发现经过：红的不是本轮新加的测试

本轮先给冻结物化脚本补路由测试（§四），随后跑全量门槛：
**`1 failed, 1586 passed, 45 skipped`**，失败项是
`tests/property/factors/test_availability_invariants.py::test_event_row_order_does_not_change_the_expansion`。
**20 分钟前同一命令是 `1582 passed, 45 skipped`、零失败**（续三十四 §一），
而本轮唯一新增是一个测试文件，不可能影响基本面可用性路径。
**单独跑也失败**，且加不加 `-p no:randomly` 一样（仓内并未安装 pytest-randomly，
只有 hypothesis 6.164.0），**故不是测试顺序依赖**。

**真实原因**：`.hypothesis/examples/` 里有 **21:00 之后新写入的条目**——
hypothesis 在某次运行中**找到反例并持久化**，此后每次都先重放它，故现在**确定性失败**。
**即这不是本轮引入的：是一个此前潜伏、现被 hypothesis 钉住的真实缺陷，而套件当时已经红着。**
（**红着的全量套件会影响每一个在本仓工作的人**，包括"因子优化"那条对话，故优先处理它。）

### 二、根因：日历映射是多对一，而排序键只有三个

反例（hypothesis 输出的字面参数）：

```
events:   000001.SZ  report_period 2024-09-30  ann_date 2025-08-29  raw_value 0.0  r1
          000001.SZ  report_period 2024-09-30  ann_date 2025-08-30  raw_value 0.0  r1
universe: signal_date 2025-09-01  000001.SZ      seed=0
断言失败：ann_date 列  left=[2025-08-29]  right=[2025-08-30]
```

实测 `assign_fundamental_available_date`（2025 年 bdate 日历）：

| ann_date | 星期 | → available_date |
|---|---|---|
| 2025-08-28 | 四 | 2025-08-29 |
| **2025-08-29** | **五** | **2025-09-01** |
| **2025-08-30** | **六** | **2025-09-01** |
| **2025-08-31** | **日** | **2025-09-01** |

**三个连续日历日塌到同一个 available_date**——该映射**多对一**（非交易日顺延到下一交易日）。
而 `expand_fundamental_events`（`src/research_v2/factors/availability.py`）原来只按
`["available_date", "report_period", "source_revision_id"]` 三键、全降序、
`kind="mergesort"`（**稳定**排序）取第一行。上面两行在**三键上全等**
（available_date 都是 09-01、report_period 都是 2024-09-30、revision 都是 r1），
**而 `ann_date` 不同**，于是稳定排序**按输入行序**决定取谁——
`ann_date` 是**发布出去的溯源列**，故**"记录的公告日"取决于抽取文件里行的先后**，
**即使两行的值完全相同（都是 0.0）**。

### 三、为什么这个可以修，而昨天那个不可以——三条可验证的理由

该文件的 docstring 原本写着 **"One input shape violates that last property"**，
指 `test_availability.py::test_two_events_tied_on_every_sort_key_are_resolved_by_input_order`
钉住的形状（**同 `ann_date`、不同 `raw_value`**），并明说
**"because the fix belongs to decision B"**、
"decision B … would **restate the sort key** and with it the C4.3 interval formula"。
**本轮是第二个形状，不在那句话覆盖范围内**：

| | **形状 1（本轮修的）** | **形状 2（仍延后给决定 B）** |
|---|---|---|
| `ann_date` | **不同**（08-29 / 08-30），塌到同一 `available_date` | **相同** |
| `raw_value` | **相同**（0.0 / 0.0） | **不同**（0.0 / 1.0） |
| 争的是 | **发布哪个公告日作为溯源** | **发布哪个数值** |
| 是语义选择吗 | **不是**：无值冲突、且不重排前三个键的优先级 | **是**：两个不可区分的修订版谁压谁 |
| 处置 | **修**：`ann_date` 作**第四个、最后一个**排序键 | **不修**：仍按输入行序，仍由那条 DEFECT 测试钉住 |

**"不侵入决定 B"是被验证的，不是被声称的**：

1. **只加第四个键，前三个键的相对次序一字未动。** 决定 B 要定的是"跨报告期谁压谁"，
   即前三个键的次序；**一个只在三者已全等时才生效的第四键，与 B 的每一种可能结论都相容。**
2. **形状 2 的 tripwire 测试在修复前后都原样通过**（它两行同 `ann_date`，加第四键后仍全等、仍按输入行序）。
   那条测试自称 *"any tie-break, refusal, or dedup layer added here has to change it deliberately"*——
   **本轮改动没有触发它，这正是两个形状不相交的证据。**
3. **修复后 `_has_contradictory_tie` 的分组键恰好成为完整 tie 集**：
   它按 `(ts_code, report_period, ann_date, source_revision_id)` 分组，
   而 `available_date` 是 `ann_date` 的函数，故"同 `ann_date` ⇒ 同 `available_date`"，
   **四键全等的行恰好就是该分组内 `raw_value` 不唯一的行**。
   **过滤器一行未改就正好描述了修复后唯一剩下的顺序依赖**——
   这是"该修复是设计本意的补全、不是外加补丁"的最强内部一致性证据。

**变异验证**：把第四个键撤掉 → **恰好这两条失败**
（新的确定性单测 + hypothesis 属性），**其余 18 条全过（含形状 2 的 tripwire）**；
还原后 `git diff --stat` 对 `availability.py` 只有 **20 insertions / 2 deletions**（即修复本身加注释）。

**同时更正两处写着错误前提的 docstring**（只改文字、不改断言）：

- 属性测试的模块 docstring："One input shape" → **两个形状**，一个已修、一个仍延后，并写明为何只修一个。
- `_has_contradictory_tie` 原有那句 *"``available_date`` is derived from ``ann_date`` through the
  calendar, **so tying on ``ann_date`` is tying on ``available_date``**"*——
  **该推理只在一个方向成立**，却被当等价关系用（"不同 `ann_date` ⇒ 不同 `available_date`"），
  而日历顺延使反向为假。**这正是第二个形状漏过过滤器的原因。** 已改写为明说"蕴含只朝一个方向"，
  并写明该过滤器**不得被放宽**（放宽就会把已修的形状重新藏起来）。

### 四、另补一处零覆盖：写信任根的那个脚本此前**没有任何测试**

`rg -ln "materialize_frozen_factor_daily|replay_factor_values" tests/` **零命中**——
**产出 `factor_daily`（新信任根）的脚本此前没有任何测试。**
新增 `tests/unit/scripts/test_materialize_frozen_factor_daily_routing.py`（**5 项**），
把续三十三 §五记录的那处 id 冲突变成断言：

1. 冻结 `yeslab_206` **必须**走 `materialize_alpha206_median_repair`，
   **不得**落到 `REPLAY_DEFINITIONS`（那里是 `Bias_Vol_Demean_60_Calculator` / `factor_data5`，**另一个因子**）；
2. `yeslab_443_reg30_max20_v1` **必须**走 `compute_gp074_variant(wide, 30, 20)`
   （**窗口写错就是另一个因子**），且它**不在** `REPLAY_DEFINITIONS` 里（走错会 KeyError，fail-closed）；
3. 其余每个冻结成员**必须**用**自己那条**定义走 `_replay_factor`，且不得借用两条修复路径；
4. **只有** `PIT_FACTORS` 拿到 industry provider（它会改变行业相关因子的值，故属路由契约而非顺带参数）；
5. 把 id 冲突本身钉住：`REPLAY_DEFINITIONS["yeslab_206"]` 的 calculator/store 仍是源谱系那一套，
   且 `apply_admission_decisions.py` 的 `grade_alias` 仍指向 `yeslab_206_median_abs_v2`。

**5 项全部经变异验证，5/5 被对应测试杀掉**：
M1 删 206 分支 / M2 删 443 分支 / M3 把 443 窗口 `30,20` 换成 `20,30` /
M4 让所有因子都拿 provider / M5 把 206 的 calculator 名改成 median 版；
每次变异后**只有一条失败**，还原后 `git diff` 对两个被变异脚本**为空**。
**这把续三十三 §五那句"8 个绕行全靠约定、没有一处由断言强制"，
在"写信任根"这一处（最重要的一处）变成了强制。**
路由用哨兵替换三个被调方来断言，**故不触数据库、不读行情缓存、不走网络。**

### 五、门槛（本轮全部重测，不沿用）

| | 修复前 | 修复后 |
|---|---|---|
| `pytest tests -q` | **1 failed, 1586 passed, 45 skipped** | **1588 passed / 45 skipped**（95.25s） |
| `ruff check src tests` | Found 1 error（新测试名超长，已改） | **All checks passed!** |
| `mypy src` | — | **Success: no issues found in 158 source files** |

**1588 = 1582（续三十四实测）+ 5（新路由测试）+ 1（新确定性测试）**，逐条对得上。
证据：`fullsuite_reverified_at_head_2026_09_06_2115.txt`（修复前的 1582 基线）。

**改动的落点与影响面**：`src/research_v2/factors/availability.py` 一处（20+/2−，含注释）。
`expand_fundamental_events` 按 §1 的普查**生产调用者为 0**
（"只有 `materialize_technical_availability` 被接线；**基本面可用性整条路径未接线**"），
**故本修复不影响任何已物化数据、任何已发布版本指针、任何基线数值。**
**未写任何数据库**（`.hypothesis/` 自带 `.gitignore`，未入库）；
**未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`；
**未改动契约、scope、门阈值或 `dropna` 口径**。
**决定 B 未被代做**——形状 2 原样保留、原测试原样通过；
**裁决 2 / 裁决 3 / 裁决 4 / 决定 A–M 一个都没被代做。**

## 勾选审计续三十六（2026-09-06 22:25）：把续三十五那个缺陷**按类扫了一遍**——同类 8 处：2 处已修（1 处潜伏）、2 处本来就是本仓的正确范式、2 处实测零实例、**2 处未核实故未动**

### 一、缺陷类抽象出来是一句话

> **稳定排序（或 `keep="first"/"last"`）的键集不是全序，于是并列行由输入行序决定，而结果被发布出去。**

它危险在**不报错**：值可能完全正确，只有溯源列（或版本指针）随抽取顺序漂移。
而属性测试的 docstring 自己写了为什么行序不固定：
*"YesLab extracts arrive per column, per month and per factor group"*。

### 二、扫描结果（8 处；逐处读代码，两处另做实测）

| 位置 | 写法 | 冲突隔离 | 全序 tiebreaker | 判定 |
|---|---|---|---|---|
| `factors/deduplication.py:71-74` | `sort_values(_BUSINESS_KEYS + ["source_row_id"], mergesort)` → `drop_duplicates(keep="first")` | **有**：`distinct_finite>1` → `CONFLICT_QUARANTINE`，且冲突组**不进** representatives | **有**：`source_row_id` | **正确——本仓的参考范式** |
| `pit/market.py:48-50` | `drop_duplicates(key_columns, keep="first")` | **有**：`if not group.eq(group.iloc[0]).all(axis=None)` 先行 | 不需要 | **正确（fail-closed）** |
| `factors/availability.py:115` | `sort_values(3 keys, desc, mergesort).iloc[0]` | 无 | **无** → 续三十五补第 4 键 `ann_date` | **曾是活体缺陷**（hypothesis 已持久化反例、套件曾红） |
| `factors/fundamental_snapshot.py:110` | `sort_values(4 keys, mergesort).drop_duplicates(["ts_code","available_date"], keep="last")` | 无 | **无** → **本轮补第 4 键 `ann_date`** | **潜伏缺陷，实测零实例**（§三 a） |
| `compact/publisher.py:144-145`、`compact/verification.py:153-154` | `sort_values("published_at", desc, mergesort)` → `drop_duplicates("logical_partition", keep="first")`；SQL 侧同样 `order by published_at desc` | 无 | **无**；且 `published_at` 由 `[now()]` 写入（毫秒精度，同一插入语句内多行会同值） | **潜伏，实测零实例**（§三 b） |
| `factors/repair.py:185-194` | 先 `_has_distinct_finite_values` → `CONFLICT_QUARANTINE`；否则 `finite_rows.iloc[0]` | **有** | 无 | **发布值是确定的**；同值重复行之间取哪一行仍由输入序定 → **其溯源列是否被发布：本轮未核实** |
| `factors/compact_coverage.py:58-65` | 先 `SOURCE_VALUE_CONFLICT` → `COMPACT_DQ_BLOCKED` 早退；否则 `drop_duplicates(_KEYS, keep="last")`（**前面没有排序**） | **有** | 无 | 同上 → **未核实** |
| `factors/source_backfill.py:271` | `sort_values(keys, mergesort).drop_duplicates(keys, keep="first")`，而 **`keys` 与排序键完全相同** | **本轮未核实** | **结构性地不可能有**：排序键＝去重键 ⇒ 同组行在排序键上必然全等 ⇒ **这个 `sort_values` 对打破并列毫无作用**，`keep="first"` 实际就是输入序 | **候选实例**（§四） |

### 三、两处"潜伏"是实测的零实例，不是推测

**(a) `fundamental_snapshot`。** 它写出 `fundamental_pit_snapshot`，实测 **32,689,537 行**。
**这里要更正决定清单 §1 的一处易误读**：§1 说"基本面可用性整条路径未接线"，
那说的是 **`expand_fundamental_events`**（C4.2–C4.4，生产调用者 0）；
而 **`build_fundamental_snapshot` 没有任何 orchestrator / runbook / docs / src 引用**
（`rg -ln materialize_fundamental_pit_snapshot scripts/ runbooks/ ../docs/ src/` **零命中**），
**却已经被跑过并落了 3269 万行**。
**故"未接线"≠"没跑过"**：无编排调用者的脚本仍可能已产出在库数据，
而 §1 那张表的读者很容易把前者读成后者——**这与 §1 自己的论点（未接线故改契约不影响已物化数据）直接相关：对本模块该论点不成立。**

并列形状是否在源里出现过？**逐组精确判定**——不按天数间隔近似，
因为**间隔 5 天以上仍可能整段落在春节/国庆休市里**，故用交易日历精确算
（`available_date(x)` = 严格晚于 x 的第一个交易日；a0<a1 塌陷 ⟺ 区间 (a0,a1] 内无交易日）：

| 源表 | `(ts_code, end_date, update_flag)` 组数 | 含 >1 个 `ann_date` | **其中塌到同一 `available_date`** |
|---|---|---|---|
| `financial_metrics_table` | 570,149 | 0 | **0** |
| `balance_sheet` | 421,742 | 286（间隔中位 80 天、**最短 2 天**） | **0** |
| `company_income_table` | 386,068 | 219（间隔中位 65 天、最短 5 天） | **0** |
| **合计** | **1,377,959** | **505** | **0** |

**故那 3269 万行从未遇到该形状，本修复对现有已发布数据是可证明的 no-op**——
**这也正是本轮敢改它的前提**。（注意 `update_flag` 只有 `{0,1}` 两个取值，
即 `source_revision_id` 基数极低，"三键全等"并非天方夜谭；
只是那 505 个多公告日的组**恰好都跨了至少一个交易日**。
`balance_sheet` 最短间隔 2 天，**是最接近触发的一处**。）
证据：`fundamental_tie_shape_census_zero_instances.txt`。

**(b) `compact/publisher`。** `published_version` 实测 **1,210 行**、
**1,210 个不同 `(logical_partition, published_at)` 对**、**1,210 个不同三元组**——
三者相等即**每个 (分区, 时间戳) 恰好只有一个 `data_version`**，故并列歧义**零实例**。
**但 SQL 侧 `order by published_at desc` 仍不是全序**，且 `published_at` 由 `[now()]` 写入，
**故这是潜伏、不是不存在**。（`latest_manifest` 无 `data_version` 列，该判定不适用。）
证据：`published_version_tie_census_zero_instances.txt`。

### 四、本轮改了什么、以及为什么另外 4 处不动

**改了（1 处 src + 1 条测试）**：`fundamental_snapshot._expand_metric_events` 的排序补第四键
`ann_date`（升序 + `keep="last"` ⇒ 取较晚的公告日），
**与 `availability.expand_fundamental_events` 的降序 + 取首行得到同一结果**——
两者不只都变确定，而且**结果一致**。

**这一处不是"顺手改"，是补自己上一轮留下的口子**：`_expand_metric_events` 的 docstring
自称是 `expand_fundamental_events` 的 *"Vectorized per-security **equivalent**"*，
而**续三十五只修了一边**，等于把一个 docstring 声称的等价关系改成了"置换下不等价"。
仓内**没有任何测试钉这个等价**（`rg expand_fundamental_events tests/unit/factors/test_fundamental_snapshot.py` 零命中），
**故全量 1588 全绿也测不出这个口子**——这正是"声称的等价没有测试"的代价。

新增 `test_snapshot_is_order_invariant_when_announcement_dates_collapse`，
用**构造**的塌陷形状钉住（真实源里没有该形状，故只能构造），
并在 docstring 里**写明实测零实例、故对现有数据是 no-op**，
免得下一个读者以为在库数据曾受影响。
**变异验证**：撤掉第四键 → **恰好这一条失败、其余 4 条通过**；还原后 diff 只有该修复（**20+/1−**）。

**没改的 4 处，理由各自不同**：

- **`repair.py` / `compact_coverage.py`**：两者都**先隔离冲突再取行**，故**发布出去的值是确定的**；
  余下的只是"同值重复行里取哪一行的溯源列"。**要不要修取决于那些溯源列是否被发布——
  本轮未核实，故不猜、不改**，只登记为待核实。
- **`source_backfill.py:271`**：**本轮扫描里最像问题的一处**（排序键＝去重键，
  故那个 `sort_values` 对打破并列**结构性地无效**），
  但它**上游是否有冲突隔离本轮未核实**；
  **在未核实之前改一条回填路径，等于在不知道会不会撞上真实数据的地方动手**，故只登记。
- **`publisher.py` / `verification.py`**：零实例（§三 b）。
  而且**改法本身不是纯技术问题**：同一 (分区, 时间戳) 出现两个 `data_version` 时
  "取哪一个"是语义选择；本仓对同类歧义的既有答案是 **`pit/market.py` 那样先拒绝**。
  照那个范式加一道 fail-closed 拒绝大概是对的，
  **但它会让一条当前正常运行的指针路径在歧义时报错而不是选一个**——
  这是会改变运行期行为的改动，**且与决定 B 的"歧义该隔离还是该选"是同一个问题**，故只登记不改。

### 五、对决定 B 的一条增量：本仓**已有**"拒绝在冲突值之间挑选"的参考实现

决定 B 与 `availability.py` 那条 DEFECT 测试的 docstring 都写着：
技术路径遇到矛盾重复值有**一整层去重与隔离**（拒绝在冲突值之间挑选），
而基本面路径**没有任何对应物**，故"诚实的修法大概是给这条路径加同一层，那是架构问题"。
**本轮把那层"参考实现"定位到了具体代码**，使 B 的这个选项从抽象变成二选一：

- **`factors/deduplication.py:57-90`**：`distinct_finite > 1` → `reason = "CONFLICT_QUARANTINE"`，
  且 `representatives = representatives.loc[representatives["reason"].isna()]`
  ——**冲突组一律不进代表行、只进隔离**；非冲突时用
  `sort_values(_BUSINESS_KEYS + ["source_row_id"])` 取 `keep="first"`，
  **即用一个显式全序键（`source_row_id`）而非输入序来定代表**。
- **`pit/market.py:48-50`**：另一种范式——**组内不完全相等就不折叠**（fail-closed）。

**故 B 若要给基本面路径补一层，本仓已有两种已接线在产的范式可选。**
**这没有替 B 做决定**（B 要定的是跨报告期谁压谁，那仍是键的**次序**问题，
而本轮两处修复都只**追加**第四键、不重排前三键，**与 B 的每一种结论都相容**），
**但它把"要不要加一层、加哪种"从开放问题变成了二选一。**

### 六、门槛（全部重测）

- `pytest tests -q` → **1589 passed / 45 skipped**（92.47s）＝续三十五的 1588 **+1**（本轮新测试）
- `ruff check src tests` → **All checks passed!**
- `mypy src` → **Success: no issues found in 158 source files**
- 变异还原核验：`git diff --stat` 对 `fundamental_snapshot.py` 只有 **20 insertions / 1 deletion**
- **未写任何数据库**（普查全是 SELECT）；**未碰** `materialization_status.{json,md}`；
  **未释放** `/tmp/frozen_chain_hold.txt`；**未改动契约、scope、门阈值或 `dropna` 口径**。
- **决定 B 未被代做**：`availability.py` 的形状 2（同 `ann_date`、不同 `raw_value`）原样保留、
  原 DEFECT 测试原样通过；本轮改动同样只追加第四键。
  **裁决 2/3/4 与决定 A–M 全部仍开放。**
- **两处明确未核实**（`repair.py`/`compact_coverage.py` 的溯源列是否被发布；
  `source_backfill.py:271` 上游有无冲突隔离），**已登记在 §二/§四，未据猜测改动任何代码**。

## 勾选审计续三十七（2026-09-06 23:00）：续三十六登记的两处"未核实"已核实——**两处都不是缺陷，且是结构上不可能**；第三处是真缺陷，已加 fail-closed 守卫并双向变异验证；20 个 hypothesis 种子复跑**零新反例**

### 一、先关掉自己上一轮留的两个未核实项

续三十六 §四写了两处"未核实故未动"。本轮核实了，**结论都是"不是缺陷"**，
而且理由比"实测零实例"更硬——**是结构上没有通道**：

**(1) `repair.py:193`（`finite_rows.iloc[0]`）——并列行在**所有列**上相同，故取哪一行无关。**
`_normalize_overlay_source` → `_normalize_factor_source`（`repair.py:695-702`）做的是
`source.loc[:, list(_FACTOR_COLUMNS)]`，而 `_FACTOR_COLUMNS` **只有四列**
（`trade_date`、`ts_code`、`factor_name`、`value`），
故 `source_representatives[key]` 存的行**不含任何溯源列**。
又因取值前已过 `_has_distinct_finite_values` → `CONFLICT_QUARANTINE`（第 186-190 行），
**留下的同键行 `value` 必然全等**；`trade_date`/`ts_code` 由分组键相同；
`factor_name` 由第 163 行 `(source_normalized["factor_name"] == factor_name).all()` 强制相同。
**四列全部相同 ⇒ `.iloc[0]` 的"输入序依赖"是空的。**
**核实的关键点在这里**：`clean_rows.append(source_row)`（第 296 行）追加的是**整行**，
而第 314-316 行 `repaired = source_row.copy()` 同样保留整行，
**故这条结论依赖"整行只有四列"，而不依赖"下游只用了 value"**——
续三十六把它登记为"其溯源列是否被发布未核实"，答案是**它根本没有溯源列可发布**。

**(2) `compact_coverage.py:65`（`drop_duplicates(_KEYS, keep="last")`）——并列之后只投影出键与 value。**
第 65 行之后紧跟 `merged = mature.merge(source_frame.loc[:, [*list(_KEYS), "value"]], ...)`：
**显式只取 `_KEYS` 与 `value`**，故即使源帧带别的列也带不出去；
而其上方（第 58-64 行）已因 `SOURCE_VALUE_CONFLICT` 早退为 `COMPACT_DQ_BLOCKED`，
**value 冲突到不了这一行**。**故 `keep="last"` 选谁对输出没有任何影响。**

**这两条把续三十六 §二那张 8 行表里的两行从"未核实"改成"不适用"**，
且靠的是**结构无通道**而非零实例：零实例只说明今天没发生，
结构无通道说明**输入怎么变都不会发生**。**该表现已无"未核实"项。**

### 二、第三处是真缺陷：`source_backfill.py:271`，已加守卫

**为什么它与上面两处不同**：那条路径读 `factor_observation_clean` 的 **16 列**，
去重后只覆盖 `run_id`/`data_version`/`created_at`/`record_hash` 四列，
**其余 8 列原样重新发布**：`value`、`observation_date`、`available_date`、`report_period`、
`stale_days`、`dedup_resolution`、`source_revision_id`、`source_watermark_id`。
而它的 `sort_values(keys)` 与 `drop_duplicates(keys)` **用同一组键**，
同组行在排序键上必然全等 ⇒ **这个排序对打破并列结构性地无效**，
`keep="first"` 的实际语义是"数据库先返回谁就用谁"。
**故重复行若在那 8 列上有分歧，发布出去的版本由查询行序决定，
而 `record_hash` 还会为胜者重新计算、替它背书。**

**它是未接线的**：`publish_client_deduplicated_repair` 的调用者**为零**
（`rg` 只命中定义处与 `factors/__init__.py` 的导出），
**故是潜伏缺陷而非在产缺陷**。但**潜伏不等于可以留着**：
这条路径的**存在目的**就是"在源不可信时把它救回来"，
而一个在最需要确定性的恢复路径上依赖行序的选择，**会在被接线那天直接变成在产缺陷**。

**修法是拒绝，不是挑选**，依据是本仓已有的两种范式（续三十六 §五）：
`deduplication.py` 隔离冲突 + 显式全序键；`pit/market.py` 组内不等即不折叠。
**"挑哪一个"是语义选择、无人授权；"拒绝歧义"不是。**
新增 `_REPUBLISHED_UNCHANGED`（那 8 列）与 `_duplicate_payload_conflicts(...)`，
在去重**之前**检查：**同一业务键的重复行若在这 8 列中任一列上不止一个取值，
即抛 `FactorBackfillError`**；文案写明是哪个键、哪几列，并说明为什么
`run_id`/`data_version`/`created_at`/`record_hash` 的分歧是**允许**的
（这四列会被覆盖，正是中断的 chunk 流应产生的形状）。

**守卫的**宽度**才是这条修复的实质，故两个方向都做了变异验证**：

| 变异 | 结果 |
|---|---|
| **M1** 整段删除守卫 | **恰好 2 条失败**（逐列 parametrize 的"分歧必须被拒" + 两种行序的那条），其余 11 条通过 |
| **M2** 把那 4 个会被覆盖的簿记列也加进 `_REPUBLISHED_UNCHANGED` | **恰好 1 条失败**（"只在簿记列上分歧的重复行必须**不**被拒"），其余 12 条通过 |

**M2 是更重要的一半**：只测 M1 只能证明守卫"存在"，
**M2 才证明守卫的宽度是对的**——太宽会让这条恢复路径
**对它被写来处理的那种数据一律报错**，即永远跑不起来。
还原后 `git diff --stat` 只有 **65 insertions / 2 deletions**。

新增 4 条测试，其中两条是**为守卫本身设的对照**：
`test_the_guard_is_indifferent_to_row_order_which_is_the_point_of_it`
（**两种行序都必须被拒**——一个只在某一行序下触发的守卫，与它所替换的代码一样依赖行序）、
`test_distinct_business_keys_are_never_treated_as_duplicates`
（不同键的多行日**不得**被当成重复）。
测试桩 `_RepairTargetSession` **按查询里的日期过滤**再返回：
`_iter_client_repair_batches` 逐日走完整月，**一个对每天都返回整帧的桩会把每行乘以月长**
（本轮第一版就是这么写的，得到 93 行而不是 3 行），
**从而把正在被测的"逐日去重"掩盖掉**——桩的错误形状会让测试通过而什么都没测。

### 三、顺带核实：那条路径承诺的 "duplicate-key validation" 是由构造保证的，不是被测量的

`publish_client_deduplicated_repair` 的 docstring 说
*"the new version becomes visible only after all daily chunks pass **duplicate-key validation**"*，
而它传下去的是 `CheckResult("repair_keys_unique", Severity.BLOCKER, CheckStatus.PASS)`——
**一个写死为 PASS 的常量**（第 230 行）；且 `drop_duplicates` 发生在 `StagedBatch.create` **之前**，
故暂存帧按构造已无重复键。**即那道"验证"验证的是自己刚做完的事。**
这与本计划反复出现的"一条不可能失败的检查"同模式（参见 C7.5 那两条被重写的控制测试）。

**本轮未改它**：把它改成真检查需要先决定"重复键在哪一层被拒绝"，
而**刚加的守卫已经在更靠前的一层做了这件事**。故只登记两者关系——
**守卫在语义层拒绝（重复行能否安全折叠），`repair_keys_unique` 在形式层恒真（暂存帧是否已无重复键）**，
两者不重复也不冲突；**但 docstring 那句话现在应读成"由构造保证"而非"经校验通过"**。

### 四、20 个 hypothesis 种子复跑：零新反例

续三十五那个缺陷是 hypothesis **在通过若干轮之后**才找到并持久化的，
故"当前全绿"不等于"没有潜伏反例"。本轮用 **20 个显式种子**复跑整个 `tests/property`
（`--hypothesis-seed=1..20`，`-p no:randomly`）：**20/20 全部 `19 passed`、零失败**，
即 **380 次属性测试执行、零新反例**。
证据：`property_suite_20_hypothesis_seeds_zero_new_counterexamples.txt`（含 `seed_hunt.sh`）。
**这不能证明没有潜伏反例**（hypothesis 的搜索是概率性的），
但它把"续三十五那次是孤例还是普遍"从猜测变成了有界证据：
**在 20 个不同种子下，属性套件里没有第二处会塌的地方。**

### 五、门槛（全部重测）

- `pytest tests -q` → **1593 passed / 45 skipped**（95.21s）＝续三十六的 1589 **+4**（本轮新测试）
- `ruff check src tests` → **All checks passed!**（中途 1 处 import 排序已 `--fix`）
- `mypy src` → **Success: no issues found in 158 source files**
- 变异还原核验：`git diff --stat` 对 `source_backfill.py` 只有 **65+/2−**
- **未写任何数据库**；**未碰** `materialization_status.{json,md}`；
  **未释放** `/tmp/frozen_chain_hold.txt`；**未改动契约、scope、门阈值或 `dropna` 口径**。
- **决定 B 未被代做**：守卫是"拒绝歧义"而非"挑选胜者"；
  `availability.py` 的形状 2 与其 DEFECT 测试原样不动。
  **裁决 2/3/4 与决定 A–M 全部仍开放。**
- **续三十六 §二那张 8 行表的收尾**：2 处缺陷已修（`availability`、`fundamental_snapshot`）、
  2 处是本仓正确范式（`deduplication`、`pit/market`）、
  2 处潜伏且实测零实例（`publisher`、`verification`）、
  **2 处经核实为不适用**（`repair`、`compact_coverage`）——
  而本轮新修的 `source_backfill` 是该表第 8 行之外**同一类的第 9 处**，
  故该类的最终清点为 **9 处：3 修、2 范式、2 潜伏零实例、2 不适用**。

## 勾选审计续三十八（2026-09-06 23:30）：把"硬编码为 PASS 的 BLOCKER 检查"按类扫了一遍——**13 处全部有支撑，无一处在产漏洞**；但这类记录**不可能失败**，故 `can_publish` 的 blocker 分支在这四个调用点上永不触发

### 一、为什么扫这一类

续三十七 §三发现 `publish_client_deduplicated_repair` 传下去的
`CheckResult("repair_keys_unique", Severity.BLOCKER, CheckStatus.PASS)` **是一个写死的常量**，
而 docstring 却承诺"only after all daily chunks pass duplicate-key validation"。
本计划此前已两次遇到同一形状：**C7.5 那两条控制测试**（"在正确与被破坏实现下结论相同"，已被重写）。
**三处同类 ⇒ 值得按类扫一遍，而不是逐个偶发地发现。**

全仓 `rg "CheckResult\("` 命中 **4 个文件、14 处**，其中 **13 处**的状态是字面量
`CheckStatus.PASS`，**1 处**是计算出来的（`repair_persistence.py:946`：
`CheckStatus.PASS if passed else CheckStatus.FAIL`）——**后者是本仓的正确范式**。

### 二、门本身怎么消费这些状态（决定了严重性）

- `quality/engine.py::can_publish`：结果为空 → False；任一 `ERROR` → False；
  **`check_name` 重复 → False**；**任一 `BLOCKER` 状态不是 `PASS` → False**；
  `required_blocking_checks` 里点名而缺失或非 PASS → False。
- `db/publish.py::publish_chunk_stream` 第 302 行调用它，False 即
  `raise PublishValidationError("data-quality gate rejected staged batch")`。
- **同函数另有独立强制**（与 checks 无关）：第 307-313 行"所有 chunk 必须同一发布身份"、
  第 314 行 `batch.validate()`、第 322-324 行**跨 chunk 业务键重复即拒**。
- `StagedBatch.validate()`（第 109-139 行）另强制：分区日期列存在且非空且**全部落在本逻辑分区内**、
  业务键列存在、帧非空、业务键非空、**帧内业务键重复即 REJECTED**（第 138-139 行）。

**故严重性取决于一件事**：每个被写死为 PASS 的名字，**是否在别处真的被强制**。
逐个查了，13 处全部有支撑，**支撑方式分四类**：

| 支撑方式 | 名字 | 证据 |
|---|---|---|
| **A. 同函数内显式 `raise`** | `candidate_scope_approved_factors`、`candidate_values_finite`、`candidate_no_overwrite`、`candidate_not_locked_test`、`candidate_keys_unique`（`publish_turnover_gap_candidates.py:200-204`，共 5 个） | `load_candidate_scope` 第 79-80 行 `set(factor_ids).issubset(APPROVED_FACTOR_IDS)` 否则 raise；第 119-120 行 `candidate_value.notna().all()` 否则 raise *"must all be finite"*；第 121-122 行源值必须全 NaN 否则 raise *"must not overwrite finite source values"*；**第 123-124 行 `observation_date.max() >= LOCKED_TEST_START` 否则 raise "candidate scope includes locked-test dates"**；第 126 行重复键 raise |
| **B. 发布器自己强制** | `repair_keys_unique`、`factor_keys_unique`、`tradability_keys_unique`、`universe_keys_unique`、`market_keys_unique` | `StagedBatch.validate()` 第 138-139 行拒帧内重复；`publish_chunk_stream` 第 322-324 行拒跨 chunk 重复。**`materialize_train_years_pit_chain.py:205-206` 还另有一道本地 `raise ValueError("market batch contains duplicate keys")`，故 market 那一个是双保险** |
| **C. 上游先隔离再暂存** | `repair_finite_values`、`factor_finite_values`、`market_finite_values` | 前两者读的是 `factor_observation_clean`／经 clean-quarantine 分流的帧，**有限性由 `deduplication.py` 的 `clean_unique = unique.loc[finite_unique]` 按构造成立**；market 那个由 `partition_invalid_ohlc(frame)`（`materialize_train_years_pit_chain.py:173`）**把非法 OHLC 切成 `ohlc_invalid` 并单独发隔离批**（第 190-194 行） |
| **D. 该值是被计算出来的，不是被喂进来的** | `tradability_bitmap_valid`、`universe_membership_valid`、`market_source_schema`、`factor_source_schema`、`repair_source_rows` | `universe/tradability.py:118-141`：`reason_bitmap = 0` 后按各标志位 `\|=`，**且 `is_eligible_pre_liquidity = (reason_bitmap == 0)` 由它派生** ⇒ 位图与标志位**按构造互洽**，没有任何外部输入能令其不成立；`universe/builder.py:47` 的 `is_member`/`exclusion_reason`/`reason_bitmap` 出自 `build_universe(candidates, definition.rules)` 的**输出投影**（第 51-63 行），**不是调用方喂入的列** ⇒ 成员资格由规则算出；两个 schema 名由 `materialize_train_years_pit_chain.py:114/124/171`、`source_backfill.py:710`（*"YesLab factor query returned an invalid schema"*）的显式 raise 支撑；`repair_source_rows` 由 `_iter_client_repair_batches` 末尾"整月无行即 raise"支撑 |

**结论：13 处里没有一处在产漏洞。** 特别是名字最 alarming 的那个——
**`candidate_not_locked_test` 是有真实强制的**（`LOCKED_TEST_START = 2026-01-01`，
第 123-124 行对 `observation_date.max()` 硬拒），
**故 2026 锁定段的红线并不依赖那个常量 PASS**。

### 三、但这不是一个可以留着不管的设计：这类记录**不可能失败**

**A/C/D 三类的支撑都在 `CheckResult` 之外**，而 `CheckResult` 的状态是常量，故：

> **`can_publish` 的 `has_failed_blocker` 分支，在这四个调用点上永远不会触发。**
> **DQ 门在这些路径上提供的保护是零**；全部保护来自上游的 `raise`、隔离分流、
> 发布器自己的 `validate()`／跨 chunk 查重、以及"该值是算出来的"。

这**不是漏洞**（保护确实在，只是不在门里），但它有两个具体后果：

1. **门的存在会误导读者。** `publish_client_deduplicated_repair` 的 docstring 就是实例：
   它写 *"only after all daily chunks pass duplicate-key validation"*，
   读起来像"有一道校验会跑"，实际是"去重已在暂存前做完，故那条校验恒真"。
   **续三十七已把这一句登记为应读成"由构造保证"。**
2. **`_dq_checks(names)` 是泛型的，故名单与强制之间没有任何绑定。**
   `materialize_train_years_pit_chain.py:84-85` 是
   `return [CheckResult(name, Severity.BLOCKER, CheckStatus.PASS) for name in names]`——
   **传任何名字都返回 PASS**。故**往元组里加一个没有任何对应强制的名字，全套件依然全绿**，
   而门会记录它"已通过"。**这与 C7.5 那两条被重写的控制测试是同一失效模式：
   一条不可能失败的检查，等于没有检查，但它会让读者以为有。**

**对照本仓已有的正确范式**：`repair_persistence.py:946`
`return CheckResult(name, Severity.BLOCKER, CheckStatus.PASS if passed else CheckStatus.FAIL)`
——**状态由 `passed` 计算**，故它**能**失败，`can_publish` 对它是活的。

### 四、本轮为什么不改代码

**因为 13 处都不是缺陷**，而"把常量 PASS 改成计算式"需要一个**能算出 `passed` 的判据**；
对 B/D 两类，那个判据要么与发布器的 `validate()` 完全重复（多跑一遍同样的查重），
要么根本不存在可失败的输入（该值是算出来的，没有"不成立"的可能）。
**为恒真的性质写一个"能失败的检查"，只能靠人为构造一个会失败的输入，那是假检查。**

**真正值得改的是第 2 条后果（名单与强制无绑定），而它不是本轮能单方面定的**：
要么给每个名字配一个真实判据（B/D 两类无从配起），
要么**把这些常量记录从 `CheckResult` 里拿掉、改为在文书里写明"该性质由构造保证"**。
**后者会改变"DQ 门在这些路径上长什么样"，属于治理口径的选择**，故本轮**只登记，不改**。

**但它不是决定 I 的第二个实例——决定 I 的判据对这一类完全无效，需要一条姊妹判据。**
决定 I 的判据是"每个治理模块至少有一个生产调用者，或显式标注为设计参考"；
**而这些 `CheckResult` 确实有调用者**（`publish_chunk_stream` 第 302 行把它们交给 `can_publish`）。
**问题不在"没人用"，而在"它们的状态不可能不是 PASS"。** 故需要的是：

> **每一条被记录为已通过的检查，要么能失败，要么被显式标注为"由构造保证"。**

两者同族（**一条看起来承重、实际不承重的声明**）但作用在不同层：
**I 问"有没有东西用它"，这条问"它能不能不成立"。**
**C7.5 那两条被重写的控制测试**（"在正确与被破坏实现下结论相同"）是这条判据的**已有先例**——
当时逐条发现并修，**本轮把它抽象成判据**。已挂到决定清单 §3c（决定 I）名下，
**建议与 I 一并考虑**：两者要防的是同一种"看起来已经保证了"的误读，
分开定会让同一份完成判据被改两次。

### 五、门槛（全部重测）

- `pytest tests -q` → **1593 passed / 45 skipped**（与续三十七同，**本节零代码改动**）
- `ruff check src tests` → **All checks passed!**；`mypy src` → **158 文件清**
- **未写任何数据库**（本节全是读代码 + `rg`）；**未碰** `materialization_status.{json,md}`；
  **未释放** `/tmp/frozen_chain_hold.txt`；**未改动契约、scope、门阈值或 `dropna` 口径**。
- **决定 I 未被代做**（本节只给它加了第二个实例）；**裁决 2/3/4 与决定 A–M 全部仍开放。**

## 勾选审计续三十九（2026-09-07 00:05）：把"非全序 `ORDER BY` 却被按位置消费"按类扫了一遍——24 处里 **7 处非全序、实测零实例**；而本轮**自己想出的那个修法被数据否证了**

### 一、这一类是什么，为什么接着扫

续三十六在 `compact/publisher.py` 记下过一句：SQL 侧 `order by published_at desc`
**也不是全序**（`published_at` 由 `[now()]` 写入，毫秒精度）。
它与刚收尾的那一类是**同族的另一半**：

> 上一类是**进程内**的稳定排序键不全序 ⇒ 并列由 pandas 的输入行序决定；
> 这一类是**数据库侧**的 `ORDER BY` 不全序 ⇒ 并列由存储引擎 unspecified 的返回序决定，
> 而 `select top 1` 会把并列**藏起来**（只回一行，看不出还有并列者）。

**后者更隐蔽**：进程内的那一类至少还能靠"换个行序重跑"复现，
而 `top 1` 的并列**在结果里没有任何痕迹**，只能靠查表本身才能发现。

### 二、扫描与分类：24 处 `order by`，17 处全序、7 处非全序

`rg -i "order by" src/ scripts/` 命中 **24 处**，分四组：

| 组 | 处数 | 排序键 | 是否全序 | 依据 |
|---|---|---|---|---|
| `order by cal_date` | 10 | `cal_date` | **全序** | `trade_calendar` 每个 `cal_date` 一行，键唯一 |
| `order by trade_date, ts_code` / `ts_code, trade_date` / `ts_code` / `ts_code, start_date` | 6 | 业务键 | **全序** | 分别是日行情主键、`stock_info_table` 的 `ts_code`、SCD 的 `(ts_code, start_date)` |
| `order by factor_id, yr` | 1 | 分组键 | **全序** | 紧跟 `group by factor_id, year(signal_date) as yr`，结果里该组合唯一 |
| **`order by published_at desc`** | **7** | **`published_at`** | **非全序** | `published_at` 是 TIMESTAMP（毫秒），**不保证唯一**；且 7 处全是 `select top 1` |

**7 处全是同一个查询形状**——"这个 (对象, 分区) 的当前 `data_version` 是什么"，即**版本指针读取**：

| 位置 | 用途 |
|---|---|
| **`src/research_v2/db/publish.py:446`（`_previous_version`）** | **发布器自己的指针读取**：决定要不要推进指针、以及判 `"data_version is already current"`。**后果最重的一处** |
| `src/research_v2/compact/publisher.py:131`、`:323` | 紧凑层读源对象指针 / 读单分区指针 |
| `src/research_v2/compact/verification.py:142` | 最新 manifest 核验 |
| `scripts/publish_factor_daily_versions.py:198` | `factor_daily` 版本状态汇报 |
| `scripts/run_factor_repair_overlay.py:726` | 读 `market_daily_clean` 指针 |
| `scripts/record_train_similarity_full_day.py:132` | 读 `factor_observation_clean` 指针 |

### 三、实测：零实例——但**这是"今天没有"，不是"不会有"**

`published_version` 全表实测（只读 SELECT）：

| 量 | 值 |
|---|---|
| 行数 | **1,210** |
| 不同 `(object_name, logical_partition, published_at)` | **1,210** |
| 不同 `(object_name, logical_partition, published_at, data_version)` | **1,210** |
| 不同 `(object_name, logical_partition, data_version)` | **1,206** |
| 覆盖对象数 | 19（`factor_daily`、`factor_observation_clean`、`market_daily_clean`、`universe_daily` …） |

**前两个数都等于行数 ⇒ 没有任何两行共享 `(对象, 分区, published_at)`，故当前零歧义。**
（**本轮的分组带上了 `object_name`**：续三十六那次只按 `(logical_partition, published_at)` 分组，
虽然结论方向相同，但**跨对象可能相互抵消**，故这次是更强的口径。）

**第四个数的差值也有解释**：1,210 − 1,206 = **4 行是"同一分区、同一 `data_version` 被发布两次"**，
即 `publish_chunk_stream` docstring 里那条重试路径（*"a retry can safely use the same
`data_version`"*）。**它对 `top 1` 无害**，因为两次的 `published_at` 不同、较晚者确定胜出。

**但零实例只说明今天没发生**：`published_at` 来自 `[now()]`（毫秒），
**要触发歧义只需要同一 (对象, 分区) 在同一毫秒内写入两个不同的 `data_version`**。
**而裁决 2 的 (a) 选项要把执行改成 48 路并发**——
虽然并发 worker 按设计各自负责不同分区（故不共享 `logical_partition`），
**但"毫秒级同分区双写"从此不再是一个可以按'串行所以不可能'忽略的形状。**
**这是本轮把它扫出来的实际理由：它的零实例是**当前执行口径**的函数，而那个口径正在被裁决 2 重议。**

### 四、如实记录：本轮自己想出的那个修法，被数据否证了

看到 schema 里有 `previous_data_version` 时，本轮的判断是：
**"`published_version` 本身就是一条版本链，故'谁是当前版本'可以靠走链回答，
根本不需要依赖毫秒时间戳"**——若成立，这会是四个选项里最好的一个
（用已有的列，不加 schema、不改语义、不引入武断 tiebreaker）。

**实测否证了它**：`previous_data_version` 在 **1,060 / 1,210 行（87.6%）为空**。
原因就是 `_previous_version` 的语义——**该 (对象, 分区) 首次发布时它返回 `""`**
（`db/publish.py:456-458`：`if result.empty: return ""`），
而 1,210 行分布在 19 个对象 × 各自的分区上，**绝大多数 (对象, 分区) 只发布过一次**，
故链上根本没有前驱可记。**一条 87.6% 断头的链不能用来定位链尾。**

**这条必须写下来**：它是本轮唯一一次"看起来显然更好的修法"，
而它错在一个**只有查数据才能知道的事实**上（那一列大部分是空的）。
**若按它去改，会得到一个在 150 行上看起来对、在 1,060 行上静默返回空指针的实现。**

### 五、余下三个选项，每一个都是决定而不是机械修复

| | 做法 | 为什么是决定 |
|---|---|---|
| **(a) 武断 tiebreaker** | `order by published_at desc, data_version desc` | **确定性有了，但对一个"版本指针"来说 `data_version` 的字典序没有任何语义**——它会在歧义时**挑一个胜者**，而挑胜者正是续三十七在 `source_backfill` 里**拒绝**去做的那件事 |
| **(b) 拒绝歧义** | 改成 `select top 2`，若两行 `published_at` 相同而 `data_version` 不同则 `raise` | **fail-closed，与本仓 `pit/market.py:48-50` 的既有范式一致**（组内不等即不折叠），且**只在当前零实例的那个分支上改变行为**。但它动的是**发布器自己的 `_previous_version`**——即门着指针推进的那一处，**7 处一起改才算改完**，属"会改变运行期行为"的改动 |
| **(c) 加单调列** | 给 `published_version` 加一个序列/单调递增列并改 7 处排序 | **真正的修法**，但**是 schema 变更**——而**决定 J（C5.4 的比对状态存哪里）与决定 K（C5.5 能否写 V2Test）正是"谁可以改 schema、改到哪个库"这两件事**，故 (c) 的前置是 J/K，**不能先于它们** |

**本轮不改，理由是三个选项里没有一个是机械的**：
(a) 需要授权一个武断语义，(b) 会改发布器的运行期行为，(c) 的前置是 J/K。
**这与续三十七对 `source_backfill` 的处置相反**——那里"拒绝歧义"是明确正确的，
因为**歧义形状已经被实测到、且那条路径零调用者**；
这里**歧义零实例、而受影响的是发布器主干**，故同一套范式要不要用、用在哪一层，需要 owner 定。

**倾向（不代做）**：**(b)**。理由与续三十七一致——
**"挑哪一个"是语义选择、无人授权；"拒绝歧义"不是**；
且 (b) 在**当前数据上是 no-op**（零实例已实测），
不像 (a) 会立刻改变"歧义时谁胜"的语义、也不像 (c) 需要先动 schema。
**(b) 还应与裁决 2 一并考虑**：并发化会让"同毫秒同分区双写"从"串行故不可能"变成"需要被拒绝的形状"，
**故 (b) 是 (a) 并发改造的前置守卫，而不是它的可选装饰。**

证据：`published_version_pointer_order_census.txt`（含全部计数、schema 与 7 处清单）。

### 六、门槛（全部重测）

- `pytest tests -q` → **1593 passed / 45 skipped**（**本节零代码改动**，与续三十八同）
- `ruff check src tests` → **All checks passed!**；`mypy src` → **158 文件清**
- **未写任何数据库**（本节全是读代码 + 只读 SELECT）；**未碰** `materialization_status.{json,md}`；
  **未释放** `/tmp/frozen_chain_hold.txt`；**未改动契约、schema、scope、门阈值或 `dropna` 口径**。
- **决定 J / K 未被代做**（(c) 的前置正是它们）；**裁决 2 未被代做**（本轮只指出 (b) 是它的前置守卫）；
  **裁决 3/4 与决定 A–M 全部仍开放。**

## 勾选审计续四十（2026-09-07 00:40）：`record_hash` 这一个列名在仓内有**三套互不相容的定义**，而**信任根那一套最弱**（MD5、且**不含 `value`**）；另有一处**引用了不存在的先例**，已更正（新增决定 N）

### 一、问题从哪来

续三十九查到 `source_backfill.py` 先写 `created_at = Timestamp.now()`、`run_id = uuid4()`，
**再**算 `record_hash`。于是问：**内容哈希依赖挂钟时间吗？** 若依赖，逐位复现契约就有一处例外。
答案是**依赖，但只在其中一条路径上**——而顺着这条线查下去，发现问题比"依赖挂钟"大。

### 二、三套定义（逐一读代码核实）

| 路径 | 算法 | 哈希输入 | 依赖挂钟/随机？ | **能否发现 `value` 变了** |
|---|---|---|---|---|
| **`materializers/market.py`**（`market_daily_clean` 等） | **SHA-256** | 整行**减去** `_RUN_LINEAGE_COLUMNS = {"run_id","data_version","created_at"}`（第 23、75-83 行） | **否** | **能** |
| **`factors/materializer.py`**（`factor_observation_clean` 226,568,607 行、`factor_quarantine`）+ `source_backfill.py` | SHA-256（`canonical`：整行 JSON 排序）或 `hash_pandas_object`→SHA-256（`fast`） | **整行，含那三个易变列** | **是** | 能，但**每次运行都变**，故不可复现 |
| **`scripts/materialize_frozen_factor_daily.py`**（**`factor_daily` 106,598,549 行 = 冻结信任根**） | **MD5** | **只有 `f"{factor_id}\|{signal_date}\|{ts_code}\|{data_version}"`**（第 180-184 行） | 否 | **不能——`value` 根本不在输入里** |

**三点各自都要紧**：

1. **同层同名列，性质相反。** `build_market_batch` 与 `build_technical_factor_batches` 是**同一层**
   （都在 `context` 里盖 `run_id`/`created_at` 后算哈希），**前者把易变列排除、后者不排除**。
   故"market 的 `record_hash` 可复现、factor 的不可复现"**不是层次差异，是同一层的两个选择**。
2. **信任根那一套是**身份**哈希，不是内容哈希。** `factor_daily.record_hash` 的输入里**没有 `value`**，
   故**同一 `(factor_id, signal_date, ts_code, data_version)` 下无论 `value` 是什么，哈希都相同**。
   **一个叫 `record_hash` 的列发现不了该行内容的任何变化。**
   它反而**包含 `data_version`**，故**值完全相同、只是版本号变了，哈希就会变**——
   **敏感的方向和该敏感的方向正好相反。**
3. **MD5 在全仓只出现在这一处**（`rg "hashlib.md5|md5\(" src/ scripts/` 只命中
   `materialize_frozen_factor_daily.py:180` 与 `:182`）。**其余所有路径都是 SHA-256。**
   故**跨表比较 `record_hash` 毫无意义**（三种算法、三种输入集）。

### 三、后果：仓内唯一那道"重建确定性"闸门只覆盖 market，而信任根**没有、且现在也不可能有**

`scripts/rehearse_full_range_pipeline.py:137` 做的是真闸门：

```python
rebuilt = build_market_batch(...)          # 同一输入再建一次
stable  = market_batch.frame["record_hash"].tolist() == rebuilt.frame["record_hash"].tolist()
...
if not report["market_stable_rebuild"].all():
    print("STABILITY FAILURE: market rebuild is not deterministic")
```

**它能成立，完全依赖 market 的哈希排除了易变列。** 而：

- 该脚本的 import 里**没有任何 factor 物化器**
  （`rg "build_technical_factor_batches|factor_observation_clean" rehearse_full_range_pipeline.py` 零命中），
  其末行自称覆盖 *"stable calendar/master/market/tradability/universe/labels"*——**factor 不在其中**；
- **factor 路径也不可能有这道闸门**：`factors/materializer.py` 的哈希含 `created_at`，
  两次重建**必然**不同，`stable` 恒为 False；
- **`factor_daily` 更不能**：它的哈希对 `value` 盲，**即使两次重建的数值不同，`stable` 也会是 True**。

**即：本仓对"重建是否确定"的唯一机械保证，覆盖的是 market／tradability／universe／labels，
而不覆盖 226.6M 行的 `factor_observation_clean` 与 106.6M 行的 `factor_daily`——
后两者恰恰是因子链与冻结基线消费的对象。**

**这三处都是潜伏而非在产**：`rg` 显示**没有任何 verifier／audit 读取 `factor_daily.record_hash`**
（只有写入方自己），也没有任何地方跨路径比较 `record_hash`。
**故今天没有一条检查在撒谎**；风险是**任何将来按 market 的语义去用这一列的 verifier
都会得到假的保证**——而在 `factor_daily` 上，那个假保证恰好是"值没变"。

### 四、一处引用了不存在的先例，已更正（只改文书与注释，未改任何哈希）

因子侧两份文书都为"排除 `record_hash`"给了同一个理由：

> *"`created_at` is a wall-clock timestamp and `record_hash` derives from it, so neither is
> reproducible and both are excluded, **exactly as the B plan's market golden test excludes its own
> non-deterministic fields**."*
> ——`tests/golden/README_factor_cases.md:35` 与 `tests/golden/test_factor_materialization.py` docstring

**这个先例不存在。** `tests/golden/test_pit_market.py` 测的是 **`clean_market_daily`**，
它在**盖 lineage 列的那一层之下**，其输出**根本没有** `created_at`/`run_id`/`data_version`/`record_hash`
——`rg` 这四个名字在该文件里**零命中**。**它什么都没排除，因为没有东西可排除。**
而**真正同层的对照（market 物化器）指向相反结论**：它排除易变列、故哈希可复现、并被上述闸门依赖。

**故那句类比把一个"本路径与 market 路径的差异"写成了"全仓共同的约定"，
并且把方向写反了**——factor 侧才是不可复现的那一个。
**已把两处都改成如实陈述**（写明 `test_pit_market.py` 在更低一层、同层对照是 market 物化器、
以及这属于决定 N），**未改动任何哈希计算、任何测试断言、任何数据**。
`tests/golden` **21 passed**、`ruff check src tests` 清。

**为什么这条值得单独更正**：它不是措辞不精，而是**一个可核查的事实断言错了**，
而它的作用是**为一个设计选择提供正当性**。下一个读者若照它去查"market 也这么干"，
会找到 `test_pit_market.py`、看到它确实没比 `record_hash`、于是**得出"约定如此"的错误结论**——
**而真相是同层的 market 物化器做了相反的选择并且测了它。**

### 五、新增决定 N：`record_hash` 到底该是内容哈希还是身份戳

| | 做法 | 代价 | 何时可做 |
|---|---|---|---|
| **(N-i)** | 把 `factor_daily.record_hash` 改成**内容哈希**（SHA-256、整行减去易变列、**含 `value`**），即采用 `market.py:23` 的既有范式 | 改变 106.6M 行的存量列值 ⇒ 需重新物化 | **应与裁决 4 同批**：裁决 4 的 (i)/(iii) 本来就要重物化 `factor_daily`，**故此时改哈希的边际成本是零**；单独做则要额外一次重物化 |
| **(N-ii)** | 把 `factors/materializer.py` 的哈希改为排除 `_RUN_LINEAGE_COLUMNS`，使 factor 路径也可复现、可上闸门 | 改变 `factor_observation_clean`／`factor_quarantine` 的存量列值 ⇒ 需重新物化**谱系 A** | **前置是决定 A 与 G**（谱系 A 的去留与去重语义），故不能先于它们 |
| **(N-iii)** | 三处统一为一个定义；真正是身份哈希的另起列名（如 `record_identity`） | 最大：动两条谱系 + **一次 schema 变更** | **前置是决定 J / K**（谁可改 schema、改到哪个库） |
| **(N-iv)** | **只文书化**：在三处各写明本路径的 `record_hash` 是什么、并写明"列名被重载、跨路径比较无意义、`factor_daily` 那一套对 `value` 盲" | **零代码、零数据** | **随时可做** |

**倾向（不代做）**：**先 (N-iv)，再把 (N-i) 绑到裁决 4 那一批、(N-ii) 绑到 A/G 之后。**
理由：**(N-i) 单独做要付一次 106.6M 行的重物化，而与裁决 4 同批做则免费**——
**这是一个纯粹的执行顺序收益，且它把 N 从"要不要改"变成"什么时候顺手改"**；
而 (N-iv) 立刻消除"将来某个 verifier 按 market 语义误用这一列"的风险，代价为零。
**(N-iii) 最正确但前置最多，故排在 J/K 之后。**

**本轮不代做任何一项**：改哈希会改变已物化数据的列值，
而 `factor_daily` 是冻结信任根、`factor_observation_clean` 属谱系 A（决定 A/G 的标的），
**两者都不是本轮可以单方面触碰的**。

### 六、门槛（全部重测）

- `pytest tests/golden -q` → **21 passed**；`ruff check src tests` → **All checks passed!**
- **本轮零代码改动**（两处文书/注释更正 + 本节）；全量套件沿用续三十八/三十九的实测值
  **1593 passed / 45 skipped**（`src/` 与 `tests/` 的断言与实现均未变，只改了一个 README 段落与一个 docstring）。
- **未写任何数据库**；**未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`；
  **未改动任何哈希计算、契约、schema、scope、门阈值或 `dropna` 口径**。
- **决定 N 未被代做**；**决定 J/K 未被代做**（N-iii 的前置）；**决定 A/G 未被代做**（N-ii 的前置）；
  **裁决 2/3/4 与决定 A–M 全部仍开放。**

## 勾选审计续四十一（2026-09-07 01:15）：决定 B 不是"会算错数"，是**已经算错了 18,884 行并发布在库**——但影响面已界定：不在信任根、不在基线、不在任何模型

### 一、为什么去测这一项

决定清单 §4 把决定 B 记为 *"**本清单里唯一会算错数的一项**，故必须排在 C 之前"*——
**用的是条件式（"会"）**。但续三十六已实测：`build_fundamental_snapshot` **虽无任何
orchestrator/runbook 引用，却已被跑过并落了 32,689,537 行**到 `fundamental_pit_snapshot`。
**若 B 描述的排序语义真的会让报告期倒退，那它就不是"会"，而是"已经"。**
**这一字之差决定 B 是预防性决定还是补救性决定，也就决定它在 §4 里的排位。**

### 二、判据：报告期在 signal_date 前进时**必须单调不减**

`_expand_metric_events` 按 `(ts_code, available_date, report_period, source_revision_id)`
**升序** + `keep="last"`，再由 `merge_asof(direction="backward")` 取
`available_date <= signal_date` 的最后一行。**故 `available_date` 排在 `report_period` 之前**：
一个**较晚公告的较旧报告期**会压过**较早公告的较新报告期**。
而在 B-i 的语义下（最新报告期优先、修订只在同期内决胜），
**一旦某证券的 Q3 已发布，更晚的 signal date 就不可能再看到 Q2**。

**故可测的特征是**：对同一 `(ts_code, metric_id)`，随 `signal_date` 前进，
**`report_period` 是否曾跌破自己的运行最大值**。每一次跌破 = 一个 signal date
发布了比更早的 signal date 已经见过的**更旧**的报告期。
实现为服务端 `context by ts_code order by signal_date` + `cummax(report_period)`，
只读 SELECT，**未写任何数据库**。

### 三、实测结果：**18,884 行**，8/10 个指标受影响

`fundamental_pit_snapshot` 共 **32,689,537 行**：

| metric | 行数 | **倒退行数** | 占比 | 首次 | 末次 |
|---|---|---|---|---|---|
| `cfoa` | 3,829,025 | **2,367** | 0.062% | 2020-05-14 | 2020-08-31 |
| `np_qoq` | 3,934,228 | **0** | 0.000% | — | — |
| `roa` | 3,806,031 | **2,234** | 0.059% | 2020-05-14 | 2020-08-31 |
| `road` | 2,772,797 | **2,387** | 0.086% | 2020-05-14 | **2024-02-27** |
| `roe` | 3,909,658 | **2,302** | 0.059% | **2020-01-08** | 2020-08-31 |
| `roe_yoy` | 2,846,359 | **2,420** | 0.085% | 2020-05-14 | **2024-02-27** |
| `roed` | 2,846,359 | **2,420** | 0.085% | 2020-05-14 | **2024-02-27** |
| `roicd` | 2,772,203 | **2,387** | 0.086% | 2020-05-14 | **2024-02-27** |
| `ta_yoy` | 2,846,359 → 2,785,840 | **2,367** | 0.085% | 2020-05-14 | 2020-08-31 |
| `roic` | 3,187,037 | **0** | 0.000% | — | — |
| **合计** | **32,689,537** | **18,884** | **0.058%** | 2020-01-08 | 2024-02-27 |

**两个最严重的实例（原文照录）**：

- `roe` / `002939.SZ` / signal_date **2020-01-08**：发布的是 **report_period 2016-12-31**，
  而**运行最大值已是 2019-09-30**——即**更早的 signal date 已经见过 2019 三季报，
  这一天却退回 2016 年报**，`ann_date 2020-01-07`、`available_date 2020-01-08`、value 6.0978。
  **倒退约三年。**
- `roa` / `300220.SZ` / signal_date **2020-05-14** 起：发布 **2019-12-31**，运行最大值 **2020-03-31**，
  由 `ann_date 2020-05-13` 的**旧期重述**触发，倒退一个季度。
  **`cfoa`/`road` 同一证券同一天同样倒退**；且 **`cfoa` 的 `source_revision_id` 是 `nan`**——
  **即该指标连修订号都没有，故排序的第三个键在它身上也不起作用。**

**分布形状本身说明了成因**：各受影响指标的行数紧密聚在 **2,234–2,420**，
且 6 个指标的首日**同为 2020-05-14**——**这是一波集中的旧期重述**（2019 年报在 2020-05-13 前后被重述），
而不是零散噪声；`roe` 另有 **2020-01-08** 一波（更早、跨度更大）。
**`road`/`roe_yoy`/`roed`/`roicd` 的末次延伸到 2024-02-27**，
即**这四个派生指标在四年后仍在倒退**，而直接指标止于 2020-08-31。
**`np_qoq` 与 `roic` 为 0**，故这不是"所有指标都错"，而是**特定重述事件 × 特定指标**的交集。

### 四、影响面已界定（这一步不能省，否则会高估或低估）

`rg -n "fundamental_pit_snapshot" src/ scripts/ tests/ ../docs/` 的**读取方只有两个**，且都是评价类：

- `scripts/evaluate_fundamental_snapshot_quality.py`（`SNAPSHOT_OBJECT = "fundamental_pit_snapshot"`，
  其 `input_source` 自述为 *"QuantResearchV2Test.fundamental_pit_snapshot and latest V2 universe/labels"*，
  即**它对 V2 标签算 RankIC**），产出
  `fundamental_pit_snapshot_quality_2020_v1/{summary.csv, daily_rank_ic.csv}`；
- `scripts/summarize_fundamental_annual_quality.py`（读上面那两个产物）。

**没有任何生产因子/标签/基线/walk-forward 路径读它**；
**冻结信任根 `factor_daily` 也不依赖它**——`materialize_frozen_factor_daily.py` 的值来自
Tushare 行情缓存 + `REPLAY_DEFINITIONS` 公式重放 + `universe_daily`，与基本面快照无关。

**故准确的影响面陈述是三句，缺一不可**：

1. **它已经在库里**：`fundamental_pit_snapshot` 的 18,884 行（0.058%）发布了倒退的报告期。
2. **它已经喂过一次评价**：`fundamental_pit_snapshot_quality_2020_v1` 的 summary 与 daily RankIC
   是在含这些行的表上算的。**该评价的窗口是 2020 年，而倒退行首次出现在 2020-01-08、
   集中出现在 2020-05-14 之后 ⇒ 与评价窗口重叠，不是擦肩而过。**
3. **它没有进入信任根、基线或任何模型**：`factor_daily`、criterion-303 六折、
   M1–M3 与冻结集候选链都不读这张表。**故基线数字与 G1–G6 不受影响。**

### 五、这对决定 B 与 §4 排序的后果

**B 的性质变了**：从"唯一**会**算错数的一项"变成"**唯一已经算错数、且错值已发布并已喂过一次评价**的一项"。
**故 §4 里 B 的排位应从第 3（在 A 之后）提到开放决定里的第 1**，理由是
**它是本清单里唯一一个标的已经错在库里的决定**，因而是**补救性**而非预防性的。

> **同一轮的自我更正（01:40）：本节初稿给 B 排位的理由写的是"它是唯一一个'每多等一天、
> 下游多读一天错值'的决定"，这句话比证据强，在此收紧。**
> **§四自己已经界定：读该表的只有 `evaluate_fundamental_snapshot_quality.py` 与其汇总脚本，
> 没有任何持续消费方。**故错值的代价是**按次发生**（每次有人跑那个评价脚本，18,884 行就再进入一次评价），
> **不是按天发生**。"每多等一天"暗示了一个并不存在的持续泄漏。
> **收紧后的理由仍然足以把 B 提到第 1**——"唯一标的已在库里错了的决定"是事实，
> 且**任何在 B 决定之前对该表的评价都会带上这 18,884 行**；
> **但它不比其他决定更"每天在流血"，故 B 的第 1 位是"补救优先于预防"，不是"紧急止血"。**
> **这个区别有实际后果**：它意味着**不必为了 B 打断正在进行的其他工作**，
> 只意味着**在 B 决定之前不要再跑那个评价脚本、也不要把它的既有产物当作干净证据引用**。

**A 与 B 的相对顺序也需要重述**：§4 原本把 A 排在 B 前，理由是"A 最上游、多项依赖它"。
**但 A 依赖的是谱系 A（`factor_observation_clean`）与冻结集表示，B 依赖的是基本面路径
（`fundamental_pit_snapshot`）——两者不共享数据，故不存在 A 必须先于 B 的技术依赖。**
A 的先决性来自"多项**决定**依赖它"，B 的紧迫性来自"一项**在库数据**正在错"。
**这两件事不该互相排队，应并行决定。**

**修 B 的代价（新增一条重物化成本，供批次规划用）**：
B-i 要改 `_expand_metric_events` 的排序（把 `report_period` 提到 `available_date` 之前，
或改为"先按报告期、再按可用日"），**这会让 32,689,537 行里的一部分选出不同的行**，
故 **B 落地必然伴随一次 `fundamental_pit_snapshot` 全量重物化**，
且**已归档的 `fundamental_pit_snapshot_quality_2020_v1` 需要在重物化后重算**（否则文书与数据不一致）。
**这与裁决 4 的 `factor_daily` 重物化是两张不同的表、两条不同的链，故不能互相搭车**——
但**两者都需要"重物化 + 重算下游评价"这一套动作，可以共用同一份 runbook 与同一批核验脚本。**

**本轮不代做 B**：改排序就是选语义（B-i / B-ii / 维持现状），
而 §2 决定 B 名下已写明这是 owner 的选择；**本轮只把"会"改成"已经"，并给出计数、实例与影响面。**

### 六、门槛（全部重测）

- **本节零代码改动**（一个只读测量脚本 + 文书）；`pytest tests -q` 沿用续四十的实测值
  **1593 passed / 45 skipped**，`ruff check src tests` 清、`mypy src` **158 文件清**
  （本轮未动 `src/` 与 `tests/`，故未重跑全量；**若下一轮改代码必须重跑**）。
- **未写任何数据库**（普查是 SELECT + 服务端 `context by`/`cummax`）；
  **未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`；
  **未改动任何排序、契约、schema、scope、门阈值或 `dropna` 口径**。
- **决定 B 未被代做**；**裁决 2/3/4 与决定 A、C–N 全部仍开放。**
- 证据：`decision_b_materialized_regression_census.txt`（含全部 10 个指标的计数、首末日与实例，
  脚本副本同存）。

## 勾选审计续四十二（2026-09-07 02:15）：批 0 已执行——**N-iv 落地**（三处各写明自己的 `record_hash` 是什么）、**已归档的 2020 基本面评价加上告警**，且该告警的行数是**精确数而非上界**

§0d 把批 0 定为"唯一不需要任何人决定就能动手的一批"。本轮执行了其中两项。**零行为变更。**

### 一、N-iv 落地：三处各写明自己的 `record_hash`，并互相指认

续四十查出一个列名三套定义，但**这个事实此前只存在于审计文书里**——
读代码的人在三处任何一处都看不到它，**而最危险的那一处（信任根）恰恰是最沉默的**。
本轮把三处都写了（**只加注释与 docstring，未改任何计算**）：

| 位置 | 加了什么 |
|---|---|
| `src/research_v2/materializers/market.py:23`（`_RUN_LINEAGE_COLUMNS`） | 说明**为什么**排除那三列（`run_id` 是新 uuid4、`created_at` 是新挂钟读数，含任一个都会让相同数据的两次构建哈希不同）、**这排除是哪道闸门成立的前提**（`rehearse_full_range_pipeline.py` 的 `market_stable_rebuild`）、**由哪条测试钉住**；并明写"本仓 `record_hash` 不是处处同义，这里是三套里最强的一套"，点名另两套 |
| `src/research_v2/factors/materializer.py::_record_hashes` | 新增 docstring：明写**本函数不排除**那三列，调用方在调用**之前**盖 `created_at`/`run_id`，故**同一份源数据两次运行哈希必不同**；因此 golden 测试排除该列、且 **`market_stable_rebuild` 那道闸门在 factor 侧"不是缺失，是从这里到不了"**；并写明两个 strategy 互不可比（canonical 是行 JSON 的 SHA-256、fast 是 `hash_pandas_object` 摘要的 SHA-256），以及第三套（MD5）的存在 |
| `scripts/materialize_frozen_factor_daily.py`（MD5 处） | 明写**这是身份戳不是内容哈希**：输入**不含 `value`**，故同业务键同 `data_version` 下**值怎么变哈希都不变**，而**值不变只换版本号哈希就变**——**敏感方向与列名暗示的相反**；且是 **MD5**（其余全是 SHA-256），故**跨表不可比**；并写明 N-i 会换成 market 的范式、**与裁决 4 同批最省**、以及**为什么本轮不改**（那会改动冻结信任根的一个存量列） |

**为什么这算批 0 而不是"顺手写注释"**：N-iv 是决定 N 的四个选项里**唯一零代码零数据**的一个，
而它消除的风险是具体的——**将来任何按 market 的语义去用 `factor_daily.record_hash` 的 verifier，
都会得到"值没变"的假保证**（续四十 §三）。
**在写入点把这个性质写下来，是唯一能在"还没有 verifier"的时候就堵住它的动作。**

**门槛自查**：`ruff check src tests` **All checks passed!**、`mypy src` **158 文件清**。
**`scripts/` 不在本项目的 ruff 门内**，但本轮仍核对了**没有让它变差**：
`ruff check scripts/materialize_frozen_factor_daily.py` 改前 **19 处**、改后 **19 处**
（中途我的注释曾多出 1 处 E501，已重排到 100 字内；`market.py` 同类 1 处也已修）。

### 二、已归档的 2020 基本面评价：加上告警，且重叠行数是**精确的**

续四十一把 B 的影响面界定为"已喂过一次评价"，但**只说了窗口重叠，没有量化重叠多少**。
本轮量化了，而且**比预期更硬**：

`artifacts/diagnostics/fundamental_pit_snapshot_quality_2020_v1/metadata.json` 自述
`period = 2020-01-02/2020-12-31`、`metrics = [roa, roe, cfoa]`、`snapshot_row_count = 1,879,673`。
**按同一窗口与同一三个指标重查 `fundamental_pit_snapshot`，得到 1,879,673 行——差值 +0，逐行相同。**
**故这不是"估计有重叠"，而是评价所读的行集被精确复现了**，于是重叠计数是精确的：

| metric | 窗口内行数 | **倒退行** | 占比 | 首次 | 末次 |
|---|---|---|---|---|---|
| `roa` | 618,994 | **2,234** | 0.361% | 2020-05-14 | 2020-08-31 |
| `roe` | 637,031 | **2,302** | 0.361% | 2020-01-08 | 2020-08-31 |
| `cfoa` | 623,648 | **2,367** | 0.380% | 2020-05-14 | 2020-08-31 |
| **合计** | **1,879,673** | **6,903** | **0.367%** | 2020-01-08 | 2020-08-31 |

**三点**：
1. **该评价报告的三个指标全部受影响**（不是"其中某个"）。
2. **三个指标各自的倒退跨度都完整落在评价窗口内**（首 2020-01-08、末 2020-08-31，窗口到 12-31），
   **不是擦边重叠**。
3. **6,903 是"进入 RankIC 计算"的上界而非精确值**——评价另外还施加 SH/SZ universe join 与
   `minimum_cross_section=30`；但**对 `metadata.json` 自述所读的那个行集，6,903 是精确的**。
   **上界与精确值分别属于两个不同的行集，不能混着说。**

**已做的事**：在该目录**新增** `CAVEAT_decision_b_regressed_report_periods.md`，
写明测到什么、为什么发生、影响与不影响什么、以及**在 B 决定前不要再跑
`evaluate_fundamental_snapshot_quality.py`、也不要把 `summary.csv` 的 RankIC/ICIR 当干净证据引用**。
**四个证据文件（`coverage.csv`、`daily_rank_ic.csv`、`metadata.json`、`summary.csv`）一个字节都没改**
（mtime 仍是 8 月 13 日 10:29，md5 已记录）——**告警是附加的，不是修订的**：
证据本身要能被后来者看到原样，否则"B 决定后重算"就没有对照。
**注意该目录未被 git 跟踪**，故告警同时写进了本节与决定清单（受版本控制的那一份是权威）。

**为什么这算批 0**：它不需要 B 被决定，只需要 B **尚未**被决定这一事实被写在用得到它的地方。
**而且它是续四十一那条自我更正的直接落地**——上一轮把"每多等一天"收紧成"按次发生"，
其实际后果正是"在 B 决定前不要运行、不要引用"；**本轮把那句后果写到了产物旁边。**

### 三、门槛

- **零行为变更**：三处只加注释/docstring，一处只新增一个 md 文件；
  **未改任何哈希计算、任何断言、任何数据、任何 schema。**
- `pytest tests -q` → **1593 passed / 45 skipped**（95.64s），**与改动前逐位相同**，
  即注释与 docstring 的加入对行为零影响（已实跑确认，非推断）。
- `ruff check src tests` → **All checks passed!**；`mypy src` → **158 文件清**；
  `scripts/materialize_frozen_factor_daily.py` 的 ruff 计数**改前改后同为 19**（未变差）。
- **未写任何数据库**（重叠普查是只读 SELECT）；**未碰** `materialization_status.{json,md}`；
  **未释放** `/tmp/frozen_chain_hold.txt`。
- **批 0 剩下的第三项 R8（清理 13 个陈旧检查点）本轮仍未执行**：
  它动的是另一条对话产出的工件目录，**属 owner 或受托动作**，理由同续四十一。
- **决定 N 未被代做**（N-iv 是它四个选项里唯一零成本的一个，且**它只加文字、不改任何一套哈希**；
  N-i/N-ii/N-iii 全部原样开放）；**决定 B 未被代做**（未改任何排序、未重物化、未重算评价）；
  **裁决 2/3/4 与决定 A、C–M 全部仍开放。**

## 勾选审计续四十三（2026-09-07 02:50）：把续三十八提出的姊妹判据**第一次用到测试套件本身**——四种可机械判定的"不可能失败"形状，1,545 个测试函数**零实例**；但**第一次跑出的那个零不可信，因为筛查工具自己的第四项是坏的**

### 一、为什么做这件事

续三十八提出了决定 I 的姊妹判据：

> **每一条被记录为已通过的检查，要么能失败，要么被显式标注为"由构造保证"。**

**提出之后没有用它。** 它对 `CheckResult` 用过了（13 处，全部有支撑），
但**它首先该用的地方是测试套件本身**——1,593 个通过的测试里，有没有"不可能失败"的？
本计划已有先例：**C7.5 那两条控制测试**"在正确与被破坏实现下结论相同"，被重写成两条结论相反的。
**那是逐条偶然发现的；本节把可机械判定的部分扫了一遍。**

### 二、四种可机械判定的形状

| | 形状 | 为什么它"不可能失败" |
|---|---|---|
| **S1** | 函数体里**没有任何 `assert`、没有 `pytest.raises`、也没有调用任何含断言的 helper** | 无论实现做什么都报 PASS |
| **S2** | `assert` 的**两侧都是常量**（`assert 1 == 1`） | 与被测代码无关，恒真 |
| **S3** | 宽泛的 `except Exception: pass` 或 `contextlib.suppress(...)` 包住主体 | **把该测试本应抛出的失败吞掉** |
| **S4** | 第一条可执行语句就是无条件 `pytest.skip(...)` | 永远 SKIPPED，从不执行 |

**扫描范围**：`tests/` 下 **237 个文件、1,545 个测试函数**（收集到 1,593 项，差额来自 parametrize）。
**结果：四种形状合计 0 实例。**

### 三、但第一个"零"不可信：筛查工具自己的 S4 是坏的

**一个报零的筛查，与一个不工作的筛查，在输出上无法区分。** 故先给筛查工具做了变异验证：
写了一个**探针文件**，含四种形状各一例 + **两个可失败的对照**
（一个委托给含断言的 helper、一个真断言 + `pytest.raises`），要求**四个都被抓到、两个对照都不被抓到**。

**第一次跑探针：S1 抓到 4 例、S2 抓到 2 例、S3 抓到 2 例、两个对照都干净——而 S4 报 0**，
尽管探针里就有一条教科书式的无条件 `pytest.skip`。**S4 是坏的**：
它的循环遇到函数体第一个非 `Call` 的 `Expr` 就 `break`，
**而函数体开头几乎总有 docstring（也是 `ast.Expr`），于是永远走不到那句 `skip`。**

**修好后重跑探针：S4 抓到 1 例，两个对照仍干净，四种形状全部可检出。**
**只有在这之后，对真实套件跑出的"四种形状 0 实例"才是可信的负结果。**

**这条方法论与本节结论同等重要**：本轮此前对四处代码改动都做了变异验证
（续三十五 5 处、续三十七 2 个方向、续三十六 1 处），
**但这是第一次对"验证工具本身"做变异验证——而它当场就失败了。**
**"用脚本证明没有问题"和"脚本能发现问题"是两件事，后者必须先证。**

### 四、这个负结果的边界（必须一起读，否则会被过度引用）

**它证明的是**：套件里没有**因语法结构而不可能失败**的测试。
**它没有证明**：每条断言都**强到足以区分正确实现与被破坏的实现**。
**后者只有变异测试能证**，而本计划的变异测试覆盖是 **85 个变异、集中在特定脚本**
（C7.5 的 M46–M85 等），**不是全套件**。
**一条断言很弱的测试能通过本筛查，却在实践中同样不可能失败。**

**另一处需如实说明**：筛查里"含断言的模块级 helper"计数为 **1,508**，
**这个数把测试函数自身也算进去了**（第一遍收集的是所有模块级函数，未排除 `test_*`），
故它**高估了 helper 数**。这不影响 S1 的判定方向
（一个只调用"无断言 helper"的测试仍会被抓），但**引用那个 1,508 时要知道它是什么**。

### 五、顺带否证的一个假设：`created_at` 与 `published_at` 之间没有时区偏差

查 `source_backfill.py` 时看到 `created_at = pd.Timestamp.now(tz="UTC").tz_localize(None)`
与 `materialize_frozen_factor_daily.py` 的 `pd.Timestamp.utcnow().tz_localize(None)`，
而 `published_at` 由 DolphinDB 的 `[now()]` 写入（`db/publish.py:468`、`compact/publisher.py:401`）。
**若服务端 `now()` 走本地钟而客户端写 UTC，同一次发布里就会有两个相差 8 小时的时间基。**

**实测否证**：DolphinDB 服务端 `now()` 返回 **`2026-09-06T15:32:27.177`**，
客户端 UTC-naive 为 **`2026-09-06 15:32:27.177445`**、本地（CST）为 `23:32:27`——
**服务端与 UTC 逐毫秒相同，即服务端走 UTC 钟。两个时间基一致，无偏差。**
且 `rg -n "created_at" src/ scripts/ | rg -i "published_at"` **零命中**：
**这两列从来没有被互相比较或一起排序过**，故即使有偏差也不会被消费。
**记下来是因为它看起来像一个真问题，而它不是**——
下一个看到这两种写法的人不必再查一遍。

### 六、门槛

- **本节零代码改动**（两个 `/tmp` 下的只读脚本 + 一个探针文件 + 文书；
  **探针不在 `tests/` 下，故不进套件、不影响收集数**）。
- 沿用续四十二的实测值：**1593 passed / 45 skipped**、`ruff check src tests` 清、`mypy src` **158 文件清**
  （`git diff --stat` 自 `9bdb500` 起对 `research_v2/src` 与 `research_v2/tests` **为空**，故为凭证据沿用）。
- **未写任何数据库**（本节只解析文件与一次只读 SELECT）；**未碰** `materialization_status.{json,md}`；
  **未释放** `/tmp/frozen_chain_hold.txt`；**未改动任何测试、断言、契约、schema 或门阈值**。
- **决定 I 及其姊妹判据未被代做**：本节是**应用**该判据做了一次只读筛查，
  **不是把它写进完成判据**（那是 I 的撰写权）。**裁决 2/3/4 与决定 A–N 全部仍开放。**
- 证据：`test_suite_falsifiability_screen_zero_instances.txt`（真实套件）、
  `test_suite_falsifiability_screen_PROBE_validates_all_four.txt`（探针，含修复前后对照所需的四种形状）、
  两个脚本副本（`screen_unfalsifiable_v2.py`、`test_deliberately_unfalsifiable.py`）。

## 勾选审计续四十四（2026-09-07 03:25）：姊妹判据剩下那一半（断言强度）开始补——对**冻结基线宽度门**做 12 个变异，**3 个是真洞已补测、2 个经查证不是洞**；套件 27 → 30 条

### 一、为什么是这个门

续四十三把"不可能失败"的**语法**形状扫干净了（1,545 个测试函数、四种形状、0 实例），
并写明它**不证明断言强度**——那只有变异测试能证，而本计划的变异覆盖是
**M23–M29（可用性/去重路径）与 M46–M85（catalog / planner / CLI / runbook）**，
**冻结基线的五道前置门一道都不在其中**。

本轮选了其中后果最重的一道：**`verify_frozen_fold_coverage.py`**。理由三条：
它是 Task 3.1 前置门里**当前 FAIL 的三道之一**；它判定的是
**"冻结基线每一折的训练横截面够不够宽"**，即**信任根能不能被训练**；
而它的两个常量是**标定出来的、且正在被裁决 4 重新讨论**——
计划里已写明"若裁决 4 选 (i) 或 (iii)，本门的『容忍 1 天』可以（并应当）**收回为 0**"。
**一个正被讨论要不要改的常量，若它的边界没有测试钉住，改的时候就没有任何东西会提醒改错了。**

### 二、12 个变异：初次 7 杀 5 活

针对 `assess_coverage` 的判定逻辑（它是纯函数，故变异可以精确落在比较式上）：

| | 变异 | 初次结果 |
|---|---|---|
| M-A | 低于门槛天数 `>` → `>=`（严一格） | killed |
| **M-B** | 低于门槛天数的容忍**放宽一天** | **SURVIVED** |
| **M-C** | 横截面下限 `width < minimum_cross_section` → `<=` | **SURVIVED** |
| **M-D** | 有效日占比 `share < min_valid_day_share` → `<=` | **SURVIVED** |
| M-E | 整段删掉"空横截面日"检查 | killed |
| M-F | 中位宽度下限**也施加到 test 段**（去掉角色限定） | killed |
| **M-G** | 中位宽度 `median_width < minimum_median_width` → `<=` | **SURVIVED** |
| **M-H** | 去掉 `minimum_median_width > 0`（即"0 关闭该判据"那个守卫） | **SURVIVED** |
| M-I | 判定强制为 `passed=True` | killed |
| M-J | 中位数换成均值（一个宽日就能掩盖窄中位） | killed |
| M-K | 强制降级为 note（永不失败） | killed |
| M-L | `min_width` 报 max 而非 min（藏掉最窄日） | killed |

### 三、5 个存活里**有 2 个不是洞**——查证过，不是推断

**这一步不能省**：把"存活"直接当"洞"报出去，与续三十六那次把"一个消费者有特例"
推广成"所有消费者都缺特例"是同一类错误。

- **M-B 不是洞，是我的变异做错了。** `max_days_below_floor` 在函数里被读**两次**
  （"低于门槛天数"分支与"空横截面日"分支），而我的 harness 用 `replace(..., 1)`
  **只改了第一处**。**那不是在放宽容忍度，而是把读同一个常量的两个分支去同步**——
  不是任何人能提交的改动。**忠实变异（两处一起 +1）重跑：killed。**
  **故那个标定容忍度 1 其实是被钉住的**（由 `test_zero_width_days_fail_once_they_exceed_the_calibrated_allowance`，
  它用 2 个空日对容忍度 1）。
- **M-H 不是洞，是等价变异。** 去掉 `> 0` 守卫后，判据变成 `median_width < 0`；
  **而宽度恒非负，故该式永假**——与原实现（守卫挡住）行为**完全相同**。
  直接探针对比证实：`minimum_median_width=0` 时**两种实现都 `passed=True`**，
  `=1000` 时**都 `passed=False`**，逐字相同。
  **等价变异无法被任何测试杀死，故它不构成覆盖缺口**；
  那个 `> 0` 守卫是**冗余但无害的防御**，"0 关闭该判据"这条 CLI 文档在两种实现下都成立。

### 四、3 个是真洞：三个数值比较的**边界**全都没有被钉住

`<` 改成 `<=` 后 **27 条测试全绿**——即**没有任何一条测试让被测值恰好等于阈值**：

| | 比较式 | 未被钉住的边界 | 为什么这个边界要紧 |
|---|---|---|---|
| **M-C** | `width < minimum_cross_section`（=30） | 某日宽度**恰好 30** | 30 是"当天能不能算出横截面统计"的下限；**恰好 30 的那天是有效日**，若被算作低于门槛就会**白白吃掉那个只容忍 1 天的额度** |
| **M-D** | `share < min_valid_day_share`（=0.95） | 占比**恰好 0.95**（19/20） | **落在公布阈值上就是达到阈值**；要求严格更多会让那个文档数字含义与字面不符 |
| **M-G** | `median_width < minimum_median_width`（=1000） | 中位**恰好 1000** | **1000 不是随手取的**：它被放进实测的 **4.2–4.8 倍空隙**里（冻结 17 的 train 中位 310–673 对遗留 14 的 1402–2794）。**"恰好等于下限算过"正是让六个遗留段全过、六个冻结段全不过的那条线** |

**已补 3 条测试，每条都断言边界的两侧**：
`test_a_day_exactly_at_the_cross_section_floor_is_not_below_it`（30 不算低于 / 29 算）、
`test_a_valid_day_share_exactly_at_the_threshold_passes`（0.95 过 / 0.96 不过）、
`test_a_median_exactly_at_the_calibrated_floor_passes`（1000 过 / 999 不过）。

**只断一侧钉不住比较式**：只断言"29 会失败"的测试对 `<` 与 `<=` 给出同样的结论——
**这正是 C7.5 那两条被重写的控制测试的形状**，故三条都写了两侧。

**重跑同一套 12 个变异：M-C / M-D / M-G 全部 killed，且各自只被对应那一条新测试杀掉**
（`1 failed, 29 passed`），**其余 9 条判定不变**。**最终：12 个变异，10 杀，2 个存活均已查证为非洞。**
测试数 **27 → 30**；被变异的脚本**还原后逐字节相同**，还原后 30 条全过。

### 五、这条结论的边界

- **只做了一道门、12 个变异。** 前置门的另外四道
  （`verify_factor_daily_backfill.py`、`verify_admission_consistency.py`、
  `screen_factor_daily_coverage.py`、`verify_frozen_report_path.py`）与
  `preflight_frozen_baseline.py` 自身**仍未做变异测试**，**未立项也未估工**。
  **故"套件断言强度足够"这个命题仍然没有被证明**——续四十三那句限定一字未变。
- **`verify_admission_consistency.py` 应是下一个**：它是决定 M 的标的（门 2 的 scope 盲区），
  而**一道已知有设计盲区的门，其测试强度更该先被验证**；它有 13 条测试。
- 变异测试是**手工驱动**的（改文件 → 跑测试 → 还原），故每次都要核对"还原后逐字节相同"；
  本轮两次运行都核对了（`restored byte-identical: True`）。

### 六、门槛（全部重测）

- `pytest tests -q` → **1596 passed / 45 skipped**（100.51s）＝续四十二的 1593 **+3**（本轮新测试）
- `ruff check src tests` → **All checks passed!**（中途 1 处注释超长已重排）
- `mypy src` → **Success: no issues found in 158 source files**
- **本轮只加测试，未改任何 `src/` 或 `scripts/` 的实现**（变异全部还原，已两次核对逐字节相同）
- **未写任何数据库**；**未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`；
  **未改动任何门阈值、契约、schema 或 `dropna` 口径**——
  **特别是：本轮没有动 30 / 0.95 / 1000 / 容忍 1 天这四个常量中的任何一个，只是把它们的边界钉住。**
- **裁决 4 未被代做**（"容忍 1 天该不该收回为 0"仍是它的选项，本轮只是让那个改动一旦发生就会被测到）；
  **决定 I 及其姊妹判据未被代做**；**裁决 2/3 与决定 A–N 全部仍开放。**
- 证据：`mutation_fold_coverage_before_and_after.txt`（含 12 个变异的前后对照、
  M-B 忠实变异的重跑、M-H 等价性的探针对比），两个脚本副本同存。

## 勾选审计续四十五（2026-09-07 04:00）：第二道门（门 2，决定 M 的标的）变异测试完成——**14 个变异全杀**；其中一个变异**就是决定 M 的 (M-i) 选项**，它被杀这件事本身是给 M 的一条证据

### 一、为什么接着做门 2

续四十四末尾点名了下一个：`verify_admission_consistency.py`。理由是
**一道已知有设计盲区的门，其测试强度更该先被验证**——
续三十三已证它的阻塞 scope 之 population 恰是 predictive 筛查的**通过者**，
故一道筛查失败会把因子移出它的视野（决定 M）。
**盲区是设计层的；但如果连它已有那部分职责的测试都不牢，M 落地时就没有基线可比。**

### 二、14 个变异：初次 12 杀 2 活

`audit_admission` 是纯函数（收 `admitted` + `ScreeningRecord` 元组，返回 `AdmissionAudit`），
故变异可精确落在判据上：

| | 变异 | 初次 |
|---|---|---|
| N-A | 资格判定反转（`!= "PASS"` → `== "PASS"`） | killed |
| N-B | 去掉 scope 过滤（任何 scope 都阻塞） | killed |
| N-C | `_is_blocking` 恒 True | killed |
| N-D | `_is_blocking` 恒 False（门永不阻塞） | killed |
| N-E | 去掉"仅限已准入因子"过滤 | killed |
| N-F | 逐条失败降级为 note（没有任何单条会失败） | killed（4 条测试） |
| N-G | 整条删掉"契约必须记录冲突"的要求 | killed |
| N-H | 声明冲突不再抑制那条汇总失败（更严） | killed |
| N-I | 去掉"未被使用的 override"上报（陈旧豁免静默累积） | killed |
| **N-J** | 覆盖 note 的 `>=` → `>` | **SURVIVED** |
| **N-K** | `COVERAGE_GATE_THRESHOLD` 0.70 → 0.99 | **SURVIVED** |
| N-L | 判定强制 `passed=True` | killed |
| **N-M** | **实现决定 M 的 (M-i)：把 `DAILY_HAC_FDR` 加进阻塞 scope** | **killed** |
| N-N | `parse_metrics` 不再取 median（覆盖 note 永不触发） | killed |

### 三、两个存活都在**同一条 note** 上，而那条 note 正是花了两轮才读懂的那一条

N-J 与 N-K 都落在
*"blocking status despite median coverage X >= 0.7; inspect the scope's other criteria"* 这条 note 上：
一个是它的边界（`>=` 对 `>`），一个是它的阈值常量（0.70）。
**它们能存活的原因只有一个**：既有唯一触发该 note 的测试用的 median 是 **0.9978**——
**它同时高于 0.70 与 0.99，故无法区分两者。**

**为什么这条 note 值得单独钉住**：它是**门 2 里唯一暗示 398/464 真实形状的东西**——
九个存量字段里八个达标、只有 `minimum_monthly_day_ratio = 0.0`。
**而把它读成"覆盖率失败"曾让审计走错两轮**（续二十九判为"快照假阳性"、
续三十二判为"取决于决定 L"），直到续三十三把失败的那个 conjunct 取出来才结案。
**一条花了这么长时间才被正确解读的诊断，至少应该把它的触发条件钉住。**

**已补 2 条测试**（各自钉住多个面，因为单面钉不住比较式）：
- `test_a_blocking_record_at_exactly_the_coverage_threshold_is_still_noted`：
  median **恰好 0.70 → note 必须触发**（杀 N-J）；**0.6999 → 必须不触发**
  （**这一侧同样必要**：否则把 note 放宽到"每条阻塞记录都触发"也能过，而那样它就变成噪声、
  失去它存在的目的）。
- `test_the_coverage_note_threshold_is_the_c62_gate_and_names_itself`：**三重钉**——
  常量值 `== 0.70`、median **0.80**（高于 0.70 而低于 0.99，故能区分真阈值与一个看起来合理的更严阈值）
  时 note 必须触发、且**消息里必须出现它实际用的那个阈值**
  （这样"改常量但不改读者依赖的文案"会在这里失败，而不是静默通过）。

**重跑同一套 14 个变异：全杀，0 存活**；N-J 与 N-K 各被对应那条新测试单独杀掉。
测试数 **13 → 15**；被变异脚本**还原后逐字节相同**。

### 四、N-M 被杀，这条结果本身是给决定 M 的证据

**N-M 不是缺陷搜寻，它是把决定 M 的 (M-i) 选项实现了一遍**
（`BLOCKING_SCOPE_SUBSTRINGS = ("STRUCTURAL_COVERAGE",)` →
`("STRUCTURAL_COVERAGE", "DAILY_HAC_FDR")`）。
**结果：killed**，由 `test_non_blocking_scope_failure_does_not_block_admission` 抓住。

**含义（对 M 的成本估计有直接影响）**：

1. **套件确实钉住了"当前哪些 scope 是阻塞的"**，故 **M-i 不可能被静默实现**——
   实施者必须**有意识地改那条测试**，正如 `test_two_events_tied_on_every_sort_key...` 的
   docstring 自称"tripwire"那样。**这是好事**：M-i 会立刻让门 2 对 206 与 375 报 FAIL
   （续三十三），那本来就是需要 owner 看见的后果，不该悄悄发生。
2. **故 M-i 的实施成本里应加一条**：**改一条既有测试**（而不是只加常量）。
   这与续三十六给 N-i 加的"与裁决 4 同批则边际成本为零"是同类信息——
   **都是把某个选项的成本从"看起来只是改一行"校正为它真实的样子。**
3. **M-ii（"缺席即发现"）没有被本轮的任何变异覆盖**：本轮变异的是既有逻辑，
   而 M-ii 要新增一条"没有行就是 `NO_EVIDENCE`"的判据，**那是新行为、不是变异**。
   **故 M-ii 的测试成本无法由本轮推断，仍未估。**

### 五、顺带钉住的一处跨文书事实

新测试的 docstring 里写明了：**0.70 是活体筛查（`factor_daily`）的门限，
而门 2 所读的持久化快照对同一个量用的是 0.80**——
**这正是决定 L 存在的原因之一（两台仪器不可比）**。
**把"门 2 采用的是哪一个"写进测试**，等于在代码里留了一处
**不会被文书过期影响的记录**：将来谁改这个常量，会同时读到"另一个数是 0.80、那是快照的、L 在议"。

### 六、这条结论的边界

- **已做两道门**（`verify_frozen_fold_coverage.py` 12 个变异、`verify_admission_consistency.py` 14 个）。
  **仍未做**：`screen_factor_daily_coverage.py`、`verify_frozen_report_path.py`、
  `verify_factor_daily_backfill.py`、以及 `preflight_frozen_baseline.py` 自身
  （它的豁免路径有 15 条测试，续三十二只端到端跑过一次真实门，**未变异**）。
  **未立项、未估工。**
- **故"套件断言强度足够"仍未被证明**，续四十三那句限定继续有效。
  两道门的样本（26 个变异、5 个初次存活、其中 2 个经查证非洞、3+2 个真洞已补）
  **提示真洞率不低**（5/26 ≈ 19% 的初次存活里，约 5/5 是可判定的边界类），
  **但这只是提示，两道门不足以外推到其余四道。**
- 变异仍是手工驱动（改文件 → 跑测试 → 还原），本轮两次运行都核对了
  `restored byte-identical: True`。

### 七、门槛（全部重测）

- `pytest tests -q` → **1598 passed / 45 skipped**（92.92s）＝续四十四的 1596 **+2**
- `ruff check src tests` → **All checks passed!**（中途 5 处注释/docstring 超长已重排）
- `mypy src` → **Success: no issues found in 158 source files**
- **本轮只加测试，未改任何 `src/` 或 `scripts/` 的实现**（14 个变异全部还原、已核对逐字节相同）
- **未写任何数据库**；**未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`；
  **未改动任何门阈值、scope、契约、schema 或 `dropna` 口径**——
  **特别是 `BLOCKING_SCOPE_SUBSTRINGS` 与 `COVERAGE_GATE_THRESHOLD` 都保持原值，只是被钉住了。**
- **决定 M 未被代做**（N-M 只是把 (M-i) 当作变异试了一次并**立即还原**，
  故仓内 scope 集合一字未改）；**决定 L 未被代做**；**裁决 2/3/4 与决定 A–N 全部仍开放。**
- 证据：`mutation_admission_consistency_14_of_14.txt`（含 14 个变异的前后对照与还原核对），脚本副本同存。

## 勾选审计续四十六（2026-09-07 04:40）：第三道门 15 个变异全杀；但**本轮先查出我自己的变异 harness 有字节码缓存缺陷，它会同时制造假存活与假击杀**，故续四十四/四十五的结论已用修好的 harness 逐条复核

### 一、先说工具的问题，因为它污染了前两轮的结果

第三道门（`screen_factor_daily_coverage.py`）第一次跑出 **7 个存活**。其中一个（P-B：
`median < MEDIAN_GATE` → `<=`）**与既有的 `test_threshold_boundary_is_inclusive_at_the_gate` 直接矛盾**——
那条测试用的 median 正是 **0.70**，改成 `<=` 后应当 `ELIGIBILITY_FAIL`。单独跑该测试**确实失败**。
**故 harness 报了假存活。**

**根因**：harness 每轮把变异写回脚本文件后直接跑 pytest，**没有清 `__pycache__`**；
测试用 `importlib.util.spec_from_file_location` 加载脚本，**`SourceFileLoader` 会读写字节码缓存，
并以 `(mtime, size)` 判定缓存是否有效**。而**比较符类变异恰好只增加 1 个字节**
（`<` → `<=`），于是 **P-A / P-B / P-C 三个变异产生同样大小的文件**；
它们又在同一秒内写入（15 个变异共 9 秒），**故 mtime 也相同** ⇒
**后一个变异复用了前一个变异的 `.pyc`，那个变异根本没被执行。**

**这个缺陷是双向的**：修好缓存后重跑第三道门，**存活集合本身变了**——
**P-B 由"存活"变成"击杀"，P-I 由"击杀"变成"存活"**。
**即前两轮报出的存活与击杀都不可全信。**

**修法**：每次跑 pytest 前 `shutil.rmtree` 掉 `scripts/`、`src/`、`tests/` 下所有 `__pycache__`，
并设 `PYTHONDONTWRITEBYTECODE=1`（前者防读旧缓存，后者防写新缓存）。

**这是本 session 第二次"验证工具本身有缺陷"**（续四十三那次是筛查工具的 S4 因 docstring 而失效）。
**两次的共同点是：工具报出的是"没有问题"，而工具坏掉时最容易报的正是"没有问题"。**

### 二、用修好的 harness 复核前两轮：结论成立

把续四十四与续四十五的变异**重跑一遍，并且把这两轮新加的测试 deselect 掉**——
因为要复核的命题正是"**既有**套件对这些变异是否是盲的"：

| 轮次 | 变异 | 只跑既有测试的结果 | 结论 |
|---|---|---|---|
| 续四十四（宽度门） | M-C 横截面下限 `<`→`<=` | **SURVIVED**（27 passed, 3 deselected） | **真洞，补测成立** |
| | M-D 有效日占比 `<`→`<=` | **SURVIVED** | **真洞，补测成立** |
| | M-G 中位宽度 `<`→`<=` | **SURVIVED** | **真洞，补测成立** |
| | M-B2 容忍度**两处同时** +1（M-B 的忠实形式） | **killed** | 印证续四十四的更正：M-B 的存活是 harness 只改一处所致 |
| 续四十五（门 2） | N-J 覆盖 note `>=`→`>` | **SURVIVED**（13 passed, 2 deselected） | **真洞，补测成立** |
| | N-K `COVERAGE_GATE_THRESHOLD` 0.70→0.99 | **SURVIVED** | **真洞，补测成立** |

**故两轮已提交的补测都站得住**，被污染的只是当时 harness 打印的击杀/存活**明细**，
不是最终写进文书的那几条结论（那几条本轮已逐条重验）。
**但续四十四表格里"12 杀 2 活"、续四十五表格里"12 杀 2 活"这类逐条明细，
应以本轮复核后的结果为准**；两轮文书已在此节标注，未回去改写原文（保留历史）。

### 三、第三道门：15 个变异，初次 7 存活，补 5 条测试后 **15/15 全杀**

`screen_factor_daily_coverage.py` 是**定义冻结集 11/17 划分的那台仪器**——
它的判定就是"三个窗口都 PASS 11/17、同样六个 ELIGIBILITY_FAIL"的来源，
也是 `verify_frozen_fold_coverage.py` 里 `C62_PASS_11` 常量的出处，
**即裁决 4 选项 (iii) 的实测依据**。故它的阈值若能被静默改动，**冻结名单就能被静默改动**。

**初次 7 个存活，逐个定性后是 6 个真洞 + 1 个等价变异**：

| | 变异 | 定性 |
|---|---|---|
| P-A | 月度门 `<` → `<=` | **真洞**：无测试让 `min_monthly` 恰好 0.90（既有用例只有 1.0 与 0.0） |
| P-C | p05 门 `<` → `<=` | **真洞**：无测试让 p05 恰好 0.30 |
| P-G | `MONTHLY_DAY_RATIO_GATE` 0.90 → 0.50 | **真洞：这个阈值的数值此前完全没有被钉住** |
| P-I | `MEDIAN_GATE` 0.70 → 0.50 | **真洞**：边界测试钉住的是 `COVERAGE_GATE`，不是 `MEDIAN_GATE` |
| P-J | `COVERAGE_GATE` 0.70 → 0.50 | **等价变异**（见 §四），仍一并钉住 |
| P-L | p05 变成 p10 | **真洞**：无测试能区分第 5 与第 10 百分位 |
| P-O | 分位数取整由四舍五入改成向下取整 | **真洞**：既有 `_quantile` 边界测试用 5 个值，`q*(n-1)` 恰为整数，两者不可区分 |

**补 5 条测试（10 → 15），每条都钉两面**（单面钉不住比较式，这是 C7.5 与续四十四已付过两次的学费）：
月度门"恰好 0.90 不算低 / 0.70 算低"、p05"恰好 0.30 过 / p05=0.25 而 p10=0.40 必须不过"、
两个中位常量各自 `== 0.70` 且 0.60 必须产出 `LOW_MEDIAN_SAMPLE_COVERAGE`、
`_quantile([0.1,0.2,0.3,0.4], 0.5) == 0.3`（4 个值使索引落在 1.5，四舍五入与向下取整才分得开）、
以及全比例扫描断言 `COVERAGE_GATE_FAIL` 永不出现。
**重跑 15 个变异：全杀，0 存活**，且 P-A/P-C/P-G/P-I/P-J/P-L/P-O 各被对应的新测试单独杀掉；
被变异脚本**还原后逐字节相同**。

**其中 P-G 最要紧**：`MONTHLY_DAY_RATIO_GATE = 0.90` **是三个阈值里唯一与持久化快照不共用的那个**
（快照对同一个量用 **0.70**），**也正是决定 L 那台仪器对照表里的差异项之一**。
**此前它的数值没有任何测试钉住**——即改这一个数字，10 条测试全绿。

### 四、P-J 是等价变异，但它暴露出第四处"一个名字、两台仪器"

`COVERAGE_GATE` 与 `MEDIAN_GATE` **同为 0.70**，而
`gate_pass = median >= COVERAGE_GATE and not reasons`，
且 `median < MEDIAN_GATE` 已先把 `LOW_MEDIAN_SAMPLE_COVERAGE` 放进 `reasons`。
**故只要两个常量相等，单独降低 `COVERAGE_GATE` 不可能改变任何判定**——
中位数低于它时 `reasons` 必已非空。**实测确认**：把比例从 0 扫到 1.0，
产出的 reason code 只有 `COVERAGE_GATE_PASS`、`LOW_MEDIAN_SAMPLE_COVERAGE`、
`LOW_P05_SAMPLE_COVERAGE`、`MISSING_MONTH_OR_LOW_MONTHLY_COVERAGE` 四种，
**`COVERAGE_GATE_FAIL` 一次都没出现**——
因为发出它的那个分支要求 `median < COVERAGE_GATE` **且** `LOW_MEDIAN` 不在 reasons 里，
**在两常量相等时该分支不可达**。

**而这个同名标签在另一台仪器里是活的**：`record_structural_quality_snapshot.py` 的
`else` 分支对**六个 conjunct 中任一个失败**都贴 `COVERAGE_GATE_FAIL`，
**正是续三十三里把两轮审计带错方向的那个catch-all**。
**故同一个 reason code：在活体筛查里不可达、在持久化快照里是总标签。**
**这是本 session 查出的第四处"一个名字、两台仪器"**（前三处：两套覆盖仪器＝决定 L、
两套冻结集表示＝决定 A、`yeslab_206` 跨谱系指两个因子）。

**P-J 仍被钉住**（新测试断言两个常量都 `== 0.70`，且断言"若两者发散则该分支变活、reason code 语义改变"）：
**等价变异虽不构成今天的覆盖缺口，但让两个常量的相等关系变成显式的、被测试守着的事实**，
将来谁要把 `MEDIAN_GATE` 降到 `COVERAGE_GATE` 之下，会**先撞上一条测试**而不是一次静默的语义变更。

### 五、边界与门槛

- **已做三道门**：宽度门（12 变异）、门 2（14 变异）、C6.2 筛查（15 变异），**合计 41 个变异、全部击杀**。
  **仍未做**：`verify_frozen_report_path.py`、`verify_factor_daily_backfill.py`、
  `preflight_frozen_baseline.py` 自身（豁免聚合逻辑，15 条测试，续三十二只端到端跑过一次真实门）。
  **未立项、未估工**，故续四十三那句限定继续有效：**全套件断言强度仍未被证明。**
- 三道门的初次存活率分别是 5/12、2/14、7/15，**其中经复核为等价变异或 harness 假象的共 3 个**
  （M-B、N/A、P-J），**真洞 9 个、已补 10 条测试**（3 + 2 + 5）。
- `pytest tests -q` → **1603 passed / 45 skipped**（103.48s）＝续四十五的 1598 **+5**
- `ruff check src tests` → **All checks passed!**；`mypy src` → **158 文件清**
- **本轮只加测试与修 harness，未改任何 `src/` 或 `scripts/` 实现**（41 个变异全部还原并核对逐字节相同）
- **未写任何数据库**；**未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`；
  **未改动任何门阈值、reason code、契约、schema 或 `dropna` 口径**——
  **0.70 / 0.70 / 0.30 / 0.90 四个常量一个都没动，只是被钉住了。**
- **裁决 4 未被代做**（`C62_PASS_11` 与 11/17 划分一字未改）；**决定 L 未被代做**；
  **裁决 2/3 与决定 A–N 全部仍开放。**
- 证据：`mutation_screen_factor_daily_coverage_15_of_15.txt`、
  `mutation_prior_rounds_reverified_without_bytecode_cache.txt`（含 deselect 复核），四个脚本副本同存。

## 勾选审计续四十七（2026-09-07 05:15）：第四道门——**前置门的豁免聚合逻辑**，16 个变异 15 杀，唯一存活者经查证为等价变异；被杀掉的那个真洞是"**报告说豁免已批准，而其实被拒了**"

### 一、为什么这道门比前三道更要紧

前三道是**判定**门；这一道是**决定判定要不要被采纳**的门。
Task 3.1 当前被三道门 REFUSED，而**唯一被认可的通过方式就是这个脚本的豁免机制**
（续三十二已端到端验证豁免桥可通、且零写入）。
**故"豁免是否被批准"这段代码，就是"冻结基线要不要开跑"那段代码。**
它这里若有存活的变异，意味着**存在一种放行前置门的方式而它的 15 条测试察觉不到**。

### 二、16 个变异：14 杀，2 存活

覆盖了豁免校验的全部面：未知门名（Q-A）、空理由（Q-B）、**纯空白理由**（Q-C）、
门根本没跑（Q-D）、blocking 集合反转（Q-E）、陈旧豁免与新豁免不再区分（Q-F）、
陈旧豁免不再上报（Q-G）、`cleared` 忽略 blocking（Q-H）、`cleared` 强制 True（Q-I）、
`blocking` 属性忽略豁免使豁免失效（Q-J）、`passed` 反转（Q-K）、
数据不全时的提前终止被去掉（Q-L）、提前终止后被跳过的门报成 PASS（Q-M）、
三个筛查窗口只取最后一个而非最差（Q-N）、**2025 窗口被删掉**（Q-O）、
逐门失败消息被丢弃（Q-P）。
**其中 14 个被既有的 15 条测试杀掉**——**这道门的测试质量明显高于前三道**
（前三道初次存活率 5/12、2/14、7/15，这道是 2/16）。

### 三、Q-H 是等价变异（已用探针证实，不是推断）

变异：`cleared = not failures and not blocking` → `cleared = not failures`。
**探针遍历三类输入**（一门失败未豁免 / 失败但已豁免 / 有一道门根本没跑）：

| 输入 | `cleared` | `blocking` | `failures` 条数 |
|---|---|---|---|
| 一门失败未豁免 | False | 2 项 | **2** |
| 失败门已豁免 | True | 空 | 0 |
| 一道门没跑 | False | 2 项 | **3** |

**`blocking` 非空的每一种情形都同时产生了 failure**——因为聚合函数对每个 blocking 结果
都 `failures.append(f"{name} FAILED (exit {code}) and is not waived")`，
而缺跑的门另有 `gate(s) did not run` 一条。
**故 `not failures` 与 `not failures and not blocking` 在所有输入上同值：等价变异，不构成缺口。**
`and not blocking` 是**冗余的双保险**——与续三十七 M-H 那个 `minimum_median_width > 0`
守卫同类（也是冗余但无害）。**仍如实登记，因为"冗余"与"无用"的区别要有人写下来。**

### 四、Q-C 是真洞，而且它的性质是"**报告说谎**"而不是"放行错误"

变异：`waived=bool(reason.strip()) and ...` → `bool(reason) and ...`，
即**纯空白的豁免理由（`"   "`）也算理由**。探针逐字对比：

| | `cleared` | `blocking` | `waived` | failures |
|---|---|---|---|---|
| **原实现** | **False** | `['admission_consistency', 'factor_coverage_screen']` | `[]` | "waiver without a reason" **+ "admission_consistency FAILED … and is not waived"** + 另一门 |
| **变异后** | **False** | `['factor_coverage_screen']` | **`['admission_consistency']`** | "waiver without a reason" + 另一门（**admission 那条 FAILED 行消失了**） |

**判定没变**（两版都 REFUSED，因为 `empty` 那条 failure 仍在），**变的是报告**：
变异后**把 `admission_consistency` 报成"已豁免"，并删掉了它自己那条"FAILED 且未豁免"**。

**为什么这不是化妆问题**：这个脚本的**全部产出就是那份报告**，
而操作者是**照着它决定要不要启动一次约 15 小时的冻结基线全量运行**的；
续三十二又已确立**豁免是越过三道失败门的唯一被认可路径**。
**一份把"被拒的豁免"报成"已批准的豁免"的报告，会让操作者以为自己的理由生效了**，
于是下一步"把另外两道门修好再跑"看起来该成功、实际不会。
**在一个以 fail-closed 为纲的项目里，报告与判定不一致本身就是缺陷**，
即使判定这一侧仍然安全。

**已补 1 条测试，三个面同时钉**（变异同时移动三者，只钉一个会留下部分缺口）：
门**留在 `blocking`**、**不出现在 `waived`**、**且它自己那条 FAILED 行必须存活**；
另加断言 `"waiver without a reason"` 仍在（那半边原本就有覆盖，一并钉住防止回退）。
**重跑 16 个变异：15 杀，唯一存活是已证等价的 Q-H。** 测试 **15 → 16**；
被变异脚本**还原后逐字节相同**（`git diff --stat -- scripts/ src/` 为空）。

### 五、累计与边界

**四道门已做：57 个变异（12 + 14 + 15 + 16），真洞 10 个，补测 11 条（3 + 2 + 5 + 1），
经复核为等价变异或 harness 假象的 4 个（M-B、M-H、P-J、Q-H）。**
**仍未做**：`verify_frozen_report_path.py`（报告可序列化性，前置门的第五道）、
`verify_factor_daily_backfill.py`（数据完整性，**第一道**——它失败则其余四道全部跳过）。
**未立项、未估工**，故续四十三那句限定继续有效：**全套件断言强度仍未被证明。**
**但就前置门这条链而言，五道里已有四道经变异验证**，
**剩下两道中 `verify_factor_daily_backfill` 是"失败即全停"的那一道，应排在下一位。**

### 六、门槛（全部重测）

- `pytest tests -q` → **1604 passed / 45 skipped**（100.02s）＝续四十六的 1603 **+1**
- `ruff check src tests` → **All checks passed!**；`mypy src` → **158 文件清**
- **本轮只加 1 条测试，未改任何 `src/` 或 `scripts/` 实现**（16 个变异全部还原，
  `git diff --stat -- scripts/ src/` **为空**）
- **未写任何数据库**；**未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`；
  **未改动任何门阈值、豁免语义、契约、schema 或 `dropna` 口径**——
  **豁免机制的行为一字未改，只是"空白理由必须被报成被拒"这一点被钉住了。**
- **裁决 2/3/4 与决定 A–N 全部仍开放**；**前置门当前仍是 `REFUSED`，本轮没有放行任何东西。**
- 证据：`mutation_preflight_waiver_aggregation_15_of_16.txt`（含 16 个变异对照 + Q-H/Q-C 探针输出），
  两个脚本副本同存。

## 勾选审计续四十八（2026-09-07 05:55）：第一道门（数据完整性）——14 个变异 **5 个存活、且 5 个全是真洞**（迄今最差比例），其中一条是 **`main()` 从未被任何测试调用过**

### 一、为什么这道门的存活率最要紧

`preflight_frozen_baseline.run_gates` 对这道门有特殊处理：**它失败就提前返回，并把其余四道门
全部标为失败而不运行**。故这道门有两个方向都不能错：
**报"完整"而数据不完整** ⇒ 后四道门在一个坏基础上给出结论；
**报"不完整"而数据完整** ⇒ 后四道门根本不被运行，前置门永远 REFUSED。
**它是五道门里唯一一个"错一次就影响全部五道"的。**

### 二、14 个变异：9 杀 5 存活，**5 个经探针逐一查证全是真洞**

**这是四道门里最差的比例**（宽度门 5/12、门 2 2/14、C6.2 筛查 7/15、前置门 2/16、**本门 5/14**），
且**没有一个能用"等价变异"或"被别的检查掩盖"解释掉**——每个都构造出了能区分二者的输入：

| | 变异 | 为什么别的检查救不了它（探针实测） |
|---|---|---|
| **R-G** | `observed is None or ...` → `observed is not None and ...`：**表里根本没有的那个因子被静默放过** | 构造"**一个期望因子被替换成冒名者、总数仍是 17**"：因子数门看到 17 ⇒ 不报；冒名者不在 expectations ⇒ 从不被检查；**唯一能发现的是 `observed is None` 那半边**。原实现正确 FAIL 并点名 `yeslab_0: None rows`，变异后 PASS。**这正是重物化时因子被改名/替换的真实形状。** |
| **R-H** | 去掉 `abs()`：**只抓多、不抓少** | 每因子容差 `max(4, 0.002%×6,000,000)=120`，总容差 `2,040`。**单个因子少 500 行**：超出自己的 120，却远在总的 2,040 之内 ⇒ **总行数门抓不到**，只有逐因子的 `abs()` 能抓。**而"少"正是中断的物化会产生的方向。** |
| **R-F** | 总行数容差 `>` → `>=` | 构造"**17 个因子各超 120**"：总数恰超 **2,040 = 总容差**，且**没有任何单个因子超出自己的容差**。原实现在容差上 **PASS**、再多一行 FAIL。**边界两侧都没有测试。** |
| **R-K** | `elif total <= 0` → `< 0`：**整年零行也算完整** | 构造"**17 个因子 id 全在、每个都是 0 行**"：因子数门满足（17）、版本门可满足 ⇒ **`total <= 0` 是唯一还站着的检查**。变异后**一个什么都没物化的年份报 PASS**，而因为它短路其余四道门，**这个空年份会被整个前置门放行**。 |
| **R-L** | `return 1 if failed else 0` → `return 0`：**进程退出码永远是 0** | **`grep -c "\.main(" tests/unit/scripts/test_verify_factor_daily_backfill.py` = 0**——**此前没有任何测试调用过 `main()`**。而 `preflight_frozen_baseline._run` 是用 `runpy` 跑这个脚本、**把 `SystemExit` 的码当作这道门的判定**。故那一行是**唯一决定第一道门过不过的代码**，而它**零覆盖**。 |

**R-L 是本轮最重的一条**：它不是"某个边界没钉住"，而是**整个 `main()` 层（参数解析、逐年循环、
退出码）没有测试**。变异它之后 13 条测试全绿——
**即"不完整的 `factor_daily` 会让前置门放行、并且因为提前终止而把另外四道门标为失败却不运行它们"
这个双向都错的后果，此前没有任何东西会察觉。**

### 三、补 5 条测试（13 → 18），重跑 14 个变异 **全杀**

每条都对着上面表格里那个"探针构造出的区分输入"写，而不是对着变异本身写：
冒名者替换（并断言 `factor_count == 17`，**以证明因子数门确实看不见它**）、
单因子少 500（并断言 `failures` 里**没有** `total rows` 那条，**以证明总行数门确实兜不住**）、
总超恰等于容差 → PASS 且再多一行 → FAIL、17 个因子各 0 行 → FAIL、
以及 `main()` 的退出码四种情形（单年好 → 0、单年空 → 1、两年都好 → 0、**范围里一年坏 → 1**）。
`main()` 用 monkeypatch 替换 `collect_year_state` 与 `walk_forward_common.new_holder`，
**故不连数据库**。

**重跑：14/14 全杀，0 存活**，各存活项被对应新测试单独杀掉；
被变异脚本**还原后逐字节相同**（`git diff --stat -- scripts/ src/` 为空）。

### 四、五道门的变异测试至此只剩一道未做

| 门 / 脚本 | 变异 | 初次存活 | 真洞 | 补测 | 现状 |
|---|---|---|---|---|---|
| `verify_frozen_fold_coverage.py`（宽度） | 12 | 5 | 3（M-B 是 harness 假象） | 3 | **全杀** |
| `verify_admission_consistency.py`（门 2） | 14 | 2 | 2 | 2 | **全杀** |
| `screen_factor_daily_coverage.py`（C6.2 筛查） | 15 | 7 | 6（P-J 等价） | 5 | **全杀** |
| `preflight_frozen_baseline.py`（豁免聚合） | 16 | 2 | 1（Q-H 等价） | 1 | **15 杀 + 1 已证等价** |
| **`verify_factor_daily_backfill.py`（第一道）** | **14** | **5** | **5** | **5** | **全杀** |
| `verify_frozen_report_path.py`（报告可序列化） | — | — | — | — | **未做** |

**累计：71 个变异、真洞 17 个、补测 16 条、经复核为等价变异或 harness 假象的 4 个。**
**只剩最后一道门未做**，故续四十三那句限定即将可以收窄为
"前置门这条链已全量变异验证"，但**"全套件断言强度"仍未被证明**（其余 ~1,530 个测试函数未做）。

### 五、门槛（全部重测）

- `pytest tests -q` → **1609 passed / 45 skipped**（98.18s）＝续四十七的 1604 **+5**
- `ruff check src tests` → **All checks passed!**；`mypy src` → **158 文件清**
- **本轮只加测试，未改任何 `src/` 或 `scripts/` 实现**（14 个变异全部还原并核对逐字节相同）
- **未写任何数据库**（`main()` 测试用 monkeypatch，不连库）；**未碰** `materialization_status.{json,md}`；
  **未释放** `/tmp/frozen_chain_hold.txt`；**未改动任何门阈值、容差、契约、schema 或 `dropna` 口径**——
  **容差 4 行 / 0.002%、因子数 17、`total <= 0` 一个都没动，只是被钉住了。**
- **裁决 2/3/4 与决定 A–N 全部仍开放**；**前置门当前仍是 `REFUSED`，本轮没有放行任何东西。**
- 证据：`mutation_backfill_first_door_14_of_14.txt`（含 14 个变异对照与还原核对）、
  探针脚本 `probe_backfill.py`（构造出区分五个存活者的五个输入）。

## 勾选审计续四十九（2026-09-07 06:35）：最后一道门（报告可序列化）——15 个变异 **9 个存活（六者最差）**，已杀 5 个；**剩 4 个杀不掉，原因写在下面而不是被掩盖**

### 一、根因是结构性的：8 条既有测试把两个纯函数测得很好，而 `main()` 一条都没碰到

`report_block_problems` 与 `coverage_completeness_problems` 是纯函数，
**针对它们的 6 个变异（S-A 空因子集、S-C 只查一个块、S-D 缺 `survival_ratio`、
S-E 非有限 `survival_ratio`、S-F 整折无覆盖、S-G 缺某个 role）全部被杀**。
**而 9 个存活者里 7 个都在 `main()` 内部**——与上一轮第一道门的 R-L 同一形状，但范围大得多。

### 二、已杀的 5 个（4 条新测试，8 → 12）

| | 变异 | 为什么它能被杀 |
|---|---|---|
| **S-B** | 逐块检查里去掉 `allow_nan=False` | **`survival_ratio` 那一支另有 `math.isfinite` 检查，故 NaN 放在那里会被抓两次、去掉 `allow_nan` 也看不出来**。要区分必须把 NaN 放在**别的字段**：`environment = {"factor_ids": [...], "some_metric": float("nan")}` ⇒ 原实现报 `environment is not JSON serialisable`，变异后静默通过。**这正是这道门存在的理由**——`allow_nan=False` 掉了就会写出裸 `NaN`，**那不是合法 JSON，事后无法与参考件逐字节比对**，而这个代价要在约 15 小时运行**结束时**才付 |
| **S-H / S-I** | `REQUIRED_ROLES` 去掉 `train` / `DEFAULT_ROLES` 去掉 `validation` | 两个常量此前**无任何断言**。前者是未知 `--roles` 的校验基准、也是完整性检查的默认；后者是**不带 `--roles` 时前置门实际测量的范围**，去掉 `validation` 就等于**不再检查那个在 B0–B3 之间做选择的段** |
| **S-J / S-K** | 未知 role 被接受 / 空 role 列表被接受 | 两道 `SystemExit` 守卫**在导入任何协作者之前**触发，故只需 argv 即可测；断言守卫抛出**同时钉住了"它必须先于数据库连接发生"**——一个 role 名字打错不该以一次连库为代价才发现 |

### 三、**杀不掉的 4 个**：`main()` 没有可注入的缝

| | 变异 | 后果（若真发生） |
|---|---|---|
| **S-L** | `return 1 if problems else 0` → `return 0` | **第五道门永远报 PASS**，前置门据此认为"报告能落盘" |
| **S-M** | `main()` 不再调用 `coverage_completeness_problems` | **少测一个 (fold, role) 也不会被发现**——即"静默跳过"正是该函数要防的事 |
| **S-N** | **被归档的那份 payload** 的 `json.dumps` 去掉 `allow_nan=False` | **这道门存在的核心性质失效**：零行段的 survival ratio 会写成裸 `NaN`，产出非法 JSON、事后无法逐字节比对 |
| **S-O** | 打印的判定恒为 `"PASS"` | 报告与退出码可以互相矛盾 |

**为什么杀不掉**：这四处都在 `main()` 里、且在它**于函数体内部导入八个协作者之后**——
`new_holder`、`load_calendar`、`SplitResolver`、`factor_source_plan`、`factor_members_for`、
`current_versions`、`extract_fn_for`、`summarize_frame_coverage`。
**没有任何一个是从模块级名字取的，故 monkeypatch 无从下手。**

**对照上一轮**：`verify_factor_daily_backfill.main()` 之所以可测，
是因为它把取数委托给**模块级**的 `collect_year_state`，patch 那一个名字就够了；
**本脚本把工作全部内联在 `main()` 里，故连一个可替换的接缝都没有。**
**这不是测试写得不够，是被测代码没有留出缝。**

**要杀掉这四个，需要先把接缝抽出来**（例如把"构造 payload → 序列化 → 汇总 problems → 返回"
提成一个纯函数，`main()` 只负责取数与打印）。**那是一次对门的结构性重构，不是加测试**，
**故本轮不做、只登记**：它是**前置门这条链上目前唯一已知的覆盖缺口**，
而且**缺口的位置和补法都已具体到函数级**。

**S-N 是四个里最要紧的一个**，故需说明它**并非全无防护**：
`report_block_problems` 里那层 `allow_nan=False` 已被本轮 S-B 的测试钉住，
即**同样的 NaN 会在更早一处被抓住**；未被覆盖的是"**最终被归档的那份文本**"这一层。
**两层防护里已钉住一层**，这与"完全没有防护"不同，也与"已经安全"不同。

### 四、前置门这条链的变异测试至此全部做完

| 门 / 脚本 | 变异 | 初次存活 | 真洞 | 补测 | 现状 |
|---|---|---|---|---|---|
| `verify_frozen_fold_coverage.py`（宽度） | 12 | 5 | 3 | 3 | 全杀 |
| `verify_admission_consistency.py`（门 2） | 14 | 2 | 2 | 2 | 全杀 |
| `screen_factor_daily_coverage.py`（C6.2） | 15 | 7 | 6 | 5 | 全杀 |
| `preflight_frozen_baseline.py`（豁免聚合） | 16 | 2 | 1 | 1 | 15 杀 + 1 已证等价 |
| `verify_factor_daily_backfill.py`（第一道） | 14 | 5 | 5 | 5 | 全杀 |
| **`verify_frozen_report_path.py`（第五道）** | **15** | **9** | **5** | **4** | **11 杀 + 4 因无接缝而未杀** |
| **合计** | **86** | **30** | **22** | **20** | **82 杀 / 1 等价 / 4 未杀（已定位）** |

**故续四十三那句限定现在可以收窄，但只能收窄到这里**：
**前置门这条链（6 个脚本）已全部做过变异测试，86 个变异里 82 个被杀、1 个已证等价、
4 个因被测代码无接缝而未杀且已定位到函数级。**
**而"全套件断言强度"仍未被证明**——其余约 1,530 个测试函数未审、未立项、未估工。

### 五、门槛（全部重测）

- `pytest tests -q` → **1613 passed / 45 skipped**（101.05s）＝续四十八的 1609 **+4**
- `ruff check src tests` → **All checks passed!**；`mypy src` → **158 文件清**
- **本轮只加测试，未改任何 `src/` 或 `scripts/` 实现**
  （15 个变异全部还原并核对；`git diff --stat -- scripts/ src/` **为空**）
- **如实记录一次自己的工具失误**：中途用一个自动重排脚本处理超长注释行，
  **它把跨行注释的续行留下了不带 `#` 的孤行，导致整个测试文件语法错误、套件收集中断**
  （`SyntaxError: invalid syntax`）。**已重建该注释块修复**；
  **教训与续四十六同类：修改文件的辅助脚本自己也需要被验证**，
  尤其是"批量重排"这种看似纯格式的操作。
- **未写任何数据库**；**未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`；
  **未改动任何门阈值、role 常量、契约、schema 或 `dropna` 口径**——
  **`REQUIRED_ROLES` 与 `DEFAULT_ROLES` 一个都没动，只是被钉住了。**
- **裁决 2/3/4 与决定 A–N 全部仍开放**；**前置门当前仍是 `REFUSED`，本轮没有放行任何东西。**
- 证据：`mutation_report_path_11_of_15_four_unreachable.txt`（含 15 个变异对照、4 个存活者名单与还原核对）。

## 勾选审计续五十（2026-09-07 07:20）：抽出接缝后**前置门这条链 86+15 个变异全部有归属**——最后一道门 15/15 全杀；而重构过程中我自己弄丢过一次入口，**是"实跑真实门"这一步抓到的**

### 一、做了什么

续四十九把最后 4 个存活者定位为"`main()` 没有可注入的缝"，并写明补法是抽接缝、
但那是"对门的结构性重构，不是加测试"，故只登记未做。**本轮做了。**

抽出两个函数，**行为不变**：

- **`collect_fold_coverage(holder, *, fold_id, roles, factor_source)`**——
  把"取数"整体下沉：resolver、`factor_source_plan`、`factor_members_for`、`current_versions`、
  `extract_fn_for`、逐 role 的 `extract` + `summarize_frame_coverage`、以及
  `driver.describe_run_environment`。**这与第一道门 `verify_factor_daily_backfill.collect_year_state`
  是同一种接缝、同一个理由**，故不是新发明，是把这条链上已有的范式补到唯一缺它的那个脚本上。
- **`build_report(environment, coverage, *, fold_id, roles) -> (problems, text)`**——
  **纯函数**：两个报告块的检查、完整性规则、以及**按 runner 的方式序列化最终被归档的那份 payload**。
  无数据库、无文件写、无打印。

`main()` 由此变薄：解析参数 → 两道 role 守卫 → `new_holder()` → `collect_fold_coverage` →
`build_report` → 打印 → `return 1 if problems else 0`。

### 二、"行为不变"是实测的，不是声称的——而且这一步抓到了我自己的一个回归

**重构后实跑真实门**，与续三十二那次前置门实跑留下的记录逐行对比：

```
factor source: frozen (17 factors); fold F01; roles ['validation', 'test']
  F01/validation  rows=   609899 complete=   159049 survival=0.2608 days=  238 <30=  0 empty=  0 median=  515 min=   57
  F01/test        rows=   666421 complete=   169107 survival=0.2538 days=  243 <30=  0 empty=  0 median=  599 min=  121
  report path OK: 1929 bytes of serialisable environment + assembly_coverage for F01 ['validation', 'test']
FROZEN REPORT PATH: PASS
```

**与重构前的记录逐字节相同，包括 `1929 bytes` 这个长度。**
证据：`report_path_door_after_seam_refactor_identical_1929_bytes.txt`。

**而这一步抓到了一个我自己引入的回归**：抽函数时我用
`s[s.index("def main(..."):]` 取到**文件末尾**作为被替换的旧文本，
**把结尾的 `if __name__ == "__main__": raise SystemExit(main())` 一起替换掉了**。
后果是**脚本被直接执行时什么都不做、退出码 0**——
即**第五道门会变成永远 PASS 且什么都不检查**，而这恰恰是 S-L/S-O 两个变异模拟的后果。
**16 条单元测试全绿也发现不了它**（它们 import 模块、不走 `__main__`），
**是"实跑真实门"这一步发现的：日志 0 字节、瞬间退出 0。**

**教训与本 session 前两次同类**（续四十三的筛查工具 S4 失效、续四十六的变异 harness 字节码缓存、
续四十九的注释重排脚本制造语法错误）：
**辅助性的批量改写（这次是"抽函数"）自己也需要被验证，而验证它的方式是运行被改的东西本身，
不是运行测试。** 若本轮只跑单元测试就提交，前置门的第五道会变成一道恒 PASS 的空门，
而 1617 条测试全绿。

### 三、四个存活者现已全部被杀，各由对应的新测试单独杀掉

| | 变异 | 现在由谁杀 |
|---|---|---|
| **S-L** | 退出码强制为 0 | `test_main_exits_nonzero_when_the_report_would_not_archive`（patch `collect_fold_coverage` + `new_holder`，三种情形：正常 → 0、NaN → 1、缺一个 role → 1） |
| **S-M** | `main()` 不再调用完整性检查 | `test_build_report_still_demands_every_requested_role`（缺 test 段 → 必须报 "missing coverage for the test segment"；整折无覆盖 → "no coverage recorded"） |
| **S-N** | **被归档的那份 payload** 去掉 `allow_nan=False` | `test_a_nan_makes_the_archived_payload_unserialisable_and_yields_no_text` |
| **S-O** | 打印判定恒为 PASS | `test_the_printed_verdict_agrees_with_the_exit_code`（capsys：PASS 时不得出现 FAIL，FAIL 时不得出现 PASS） |

**S-N 的定性本轮更正过一次，需如实记下**：起初怀疑它与 `report_block_problems` 那层重复
（因为 NaN 在两处都会被发现），并进一步猜测 `ensure_ascii=False` 会让合并层**更严**
（例如孤立代理字符）。**实测两条都不成立**：
孤立代理字符在 `json.dumps(..., ensure_ascii=False)` 下**不报错**
（它返回 `str`，只在**编码为字节**时才失败），故两层都接受它；
**但 S-N 仍然可杀**，判据不是"多一种失败输入"，而是**返回的 `text`**——
原实现在合并层失败时 `text = ""`（**不提供任何可归档文本**），
变异后 `json.dumps` 成功、**`text` 是一段含裸 `NaN` 的非法 JSON**。
**故测试同时断言"报了 combined payload 那条问题"与"`text == ""`"，
并断言 problems 恰好 2 条**（两层各一条；只剩一条说明有一层哑了）。

**重跑 15 个变异：全杀，0 存活**，四个原存活者各被对应新测试单独杀掉；
被变异脚本**还原后逐字节相同**。测试 **12 → 16**。

### 四、前置门这条链至此收口

| 门 / 脚本 | 变异 | 真洞 | 补测 | 现状 |
|---|---|---|---|---|
| `verify_frozen_fold_coverage.py` | 12 | 3 | 3 | 全杀 |
| `verify_admission_consistency.py` | 14 | 2 | 2 | 全杀 |
| `screen_factor_daily_coverage.py` | 15 | 6 | 5 | 全杀 |
| `preflight_frozen_baseline.py` | 16 | 1 | 1 | 15 杀 + 1 已证等价 |
| `verify_factor_daily_backfill.py` | 14 | 5 | 5 | 全杀 |
| `verify_frozen_report_path.py` | 15 | **9**（含本轮 4 个原不可达） | **4** | **全杀（抽接缝后）** |
| **合计** | **86** | **26** | **20** | **85 杀 + 1 已证等价，0 未杀** |

**故续四十三那句限定现在可以收窄到它的准确形式**：
> **前置门这条链（6 个脚本）已全量变异验证：86 个变异，85 杀、1 个经探针证明为等价变异、0 个未杀。**
> **"全套件断言强度"仍未被证明**——其余约 1,530 个测试函数未审、未立项、未估工。

**这条链之所以值得单独做完**：它是 Task 3.1 解冻时**唯一会跑的那组检查**，
而 Task 3.1 是裁决 4 落地后的第一步、其产物又是 3.2/3.3/4.1/5.x 整条链的输入。
**换言之，这 6 个脚本是"决定落地后第一批承重的代码"。**

### 五、门槛（全部重测）

- `pytest tests -q` → **1617 passed / 45 skipped**（106.99s）＝续四十九的 1613 **+4**
- `ruff check src tests` → **All checks passed!**；`mypy src` → **158 文件清**
- **本轮改了 1 个 `scripts/` 文件**（抽接缝 + 恢复入口守卫），**未改任何 `src/`**；
  **行为不变的证据是真实门实跑输出与重构前逐字节相同（含 `1929 bytes`）**
- **未写任何数据库**（真实门只读 `factor_daily` 与缓存数据集）；
  **未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`；
  **未改动任何门阈值、role 常量、契约、schema 或 `dropna` 口径**
- **裁决 2/3/4 与决定 A–N 全部仍开放**；**前置门当前仍是 `REFUSED`（三道门 FAIL），本轮没有放行任何东西**——
  本轮只让第五道门**可被测试**，没有改变它的判定
- 证据：`mutation_report_path_15_of_15_after_seam.txt`、
  `report_path_door_after_seam_refactor_identical_1929_bytes.txt`

## 勾选审计续五十一（2026-09-07 08:05）：转到**链 2**——`0cbbc88` 那个补 padding 的修复**已被测试钉住**（把历史缺陷原样装回去，测试抓住了）；12 个变异里 8 个存活，逐个定性后**没有一个是活的缺陷**，但其中 3 个是"真而未钉"

### 一、为什么转到链 2

前置门那条链（链 1 的解冻闸门）已在续五十收口。**链 2（模型候选）是与之并行的另一条**，
而它的驱动 `run_real_candidate_training.py` 在 2026-09-06 02:34 由 commit `0cbbc88` 修过一个真缺陷：
6 处直接遍历 `batch.inputs.ts_codes` 而无视 `stock_mask`，
`build_static_training_batches` 又会把窄于"该段最宽横截面"的每个日期用**空证券码**补齐，
于是每个较窄日吐出多行 `ts_code=""` → 那次 M1 全量跑了 **7h58m** 后死在
`seed predictions contain duplicate business keys`；
**更严重的另一半**是 `_percentile_rank_series` 在 `width`（含 padding）而非有效数上排名，
而那个百分位正是**挑选获胜 config 的 validation RankIC** 的输入。

**修复已落地、且带 3 条参差夹具回归测试。但"带了测试"不等于"钉住了"**——
本 session 已四次发现验证物自身不可靠。故本轮问的是一个具体问题：**把那个缺陷原样装回去，测试抓不抓得到？**

### 二、答案：抓得到

修复把 6 处收敛成**一个** helper（`_score_batch` → `predictions_to_frame`），
故变异点集中。**T-A 就是把历史缺陷在它的根部原样装回去**：
把 `predictions_to_frame` 里的
`torch.nonzero(batch.stock_mask[date_index], ...)` 换成 `list(range(batch.stock_mask.shape[1]))`
——**即保留 padding 位、按含 padding 的宽度排名**，正是 `0cbbc88` 之前的行为。

**结果：T-A killed。故那个修复是被钉住的，不是"看起来修了"。**

12 个变异（跨 `src/research_v2/models/base.py` 的共享契约与驱动的 seed 平均）：
**4 杀 8 存活**。被杀的 4 个正是最要紧的 4 个：
**T-A**（保留 padding＝历史缺陷本体）、**T-C**（不再拒绝有效位的非有限分数）、
**T-F**（seed 秩按位置而非业务键平均——即 `0cbbc88` 修的另一半）、
**T-H**（seed 均值塌成只取第一个 seed）。

### 三、8 个存活逐个定性：**没有一个是在产缺陷**，但 3 个是"真而未钉"

| | 变异 | 定性 | 依据 |
|---|---|---|---|
| **T-J** | `_percentile_frame` 里的 `model.eval()` 去掉 | **不是洞——而且这个"不对称"是正确的** | 全驱动**只有一处** `.eval()`（第 304 行，在 `_percentile_frame` 内）。**这是对的**：`_percentile_frame` 的两个调用点（357、672）拿到的是**训练环出来的模型**，而 `training/torch_loop.py:55` 是 `self._model.train()`，**故它必须自己切 eval**；而 `_mean_seed_percentile_scores` 的模型来自 `load_inference_artifact`，**`inference/artifact_loader.py:111` 已经 `model.eval()`**（四处 `_concat_date_scores(loaded.models, ...)` 全走这条路）。**故后者不需要 eval。M2/M3 确实用 `nn.Dropout(0.1)`**（`models/temporal.py:38/106`、`models/set_attention.py`），**所以这个不对称一旦弄反就是真缺陷**——而它现在是对的，**只是没有任何测试钉住它** |
| **T-G** | seed 合并去掉 `validate="one_to_one"` | **防御性、按上游契约不可达**（**推断，未实测**） | `predictions_to_frame` 每个日期只按 `nonzero(stock_mask)` 的**位置**出行，故要产生重复 `(signal_date, ts_code)` 需要**同一日期里同一个 ts_code 出现两次**，而 `FeatureBatch.__post_init__` 禁止有效位重复。其 docstring 自述这层是"把残余的键重复变成此处的错误，而不是留到 OOS 评价里" |
| **T-K / T-L** | 空 seed 列表 / 空段不再 raise | **只是换了异常类型与文案**（**推断，未实测**） | 去掉后分别是 `frames[0]` 的 `IndexError` 与 `pd.concat([])` 的 `ValueError`——**都会抛**，故不是 fail-open |
| **T-D** | 去掉 `scores.shape != stock_mask.shape` 检查 | **未核实** | 形状不符时下游索引大概也会抛，但抛在哪、文案是什么，本轮没测 |
| **T-E** | 去掉按分数排序 | **未核实** | 下游 seed 合并是按业务键的，故行序可能无关；但**输出行序**是否被别的消费方依赖，本轮没测 |
| **T-I** | `_percentile_frame` 不再按日期排序遍历 | **真而未钉** | `batches` 是 `Mapping[date, batch]`；**若调用方传入未按日期插入的映射，输出行序就跟着变**。当前调用方恰好有序，故看不出——**这是防御性排序，而防御性代码没被测就等于没有** |
| **T-B** | 百分位排名 `method="average"` → `"first"` | **真而未钉** | 两者**只在 raw_score 有并列时**不同。夹具分数互异故看不出，而**真实模型会产出并列分数**（尤其 padding 修复后同日有效股票数不同）。**并列时"按位置定序"意味着同分股票的百分位取决于列顺序**——那正是 `0cbbc88` 要消除的那类依赖 |

**故本轮的净结论有两条，方向相反，都要说清**：

1. **好消息，而且是本轮要问的那个问题**：`0cbbc88` 的修复**被钉住了**——
   把缺陷原样装回去（T-A）与把它的另一半装回去（T-F）都会被抓到。
   **那次 8 小时的损失不会因为测试薄弱而重演。**
2. **需要记的**：**8 个存活里没有一个是活的缺陷**（T-J 经核实是正确的不对称、
   T-G/T-K/T-L 是不可达或只是换文案），**但 T-B 与 T-I 是"真而未钉"**，
   T-D/T-E **本轮未核实**。**"没有活的缺陷"与"都被钉住了"是两件事**，
   本节只能声称前者，且其中 T-J/T-G/T-K/T-L 四条的定性是**读代码推断、未做探针实测**——
   **这与续三十七、续四十六、续四十九那几轮"每个存活都用探针构造出区分输入"的标准不同，
   故那四条的结论强度低于本 session 先前同类结论，引用时应知道这一点。**

### 四、为什么本轮没有补测试

**不是因为不值得，而是因为 T-B 与 T-I 各需要一个本轮没有的夹具**：
T-B 需要一个**存在并列 raw_score** 的参差夹具（现有夹具分数互异，这正是它测不出 T-B 的原因），
T-I 需要一个**键序被打乱的 `Mapping`**（现有夹具按日期插入）。
**造这两个夹具要先读懂 `test_run_real_candidate_training.py` 的参差夹具构造方式
（40/31 只、9 个 padding 槽）并确认新夹具同样"真的是参差的"**——
即还需要一条像 `test_ragged_fixture_really_is_padded` 那样的**夹具自证**控制测试，
否则新夹具可能和旧夹具犯同一个错（看起来在测 padding，实际没有 padding）。
**这是一件要做对的事，不是一件能顺手做的事**，故本轮**登记为下一步**，
而不是在预算不足时交一个可能自欺的夹具。

### 五、门槛（全部重测）

- 变异运行的测试范围是 `tests/unit/models tests/unit/training tests/property
  tests/unit/scripts/test_run_real_candidate_training.py`：**基线 116 passed**，
  12 个变异逐个跑、**两个文件还原后逐字节相同**、还原后仍 **116 passed**。
- **本轮零代码改动、零测试新增**（纯变异测量 + 文书）。
  **故全量门槛沿用续五十的实测值 1617 passed / 45 skipped**，
  且这次沿用是**有据的**：`git diff --stat` 对 `research_v2/src`、`research_v2/scripts`、
  `research_v2/tests` 自 `53fd964` 起**为空**。
- `ruff check src tests` 清、`mypy src` **158 文件清**（本轮未改代码，二者亦未变）。
- **未写任何数据库**；**未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`；
  **未改动任何契约、schema、门阈值或 `dropna` 口径。**
- **裁决 2/3/4 与决定 A–N 全部仍开放**；**链 2 仍未重启**（其前置见 §0d 批 2：
  清理 13 个陈旧检查点、指针并列守卫、续跑键含契约版本）。
- 证据：变异 harness 已存 `mutation_candidate_scoring.py`（含 12 个变异的定义、
  逐变异跑 `tests/unit/models tests/unit/training tests/property
  tests/unit/scripts/test_run_real_candidate_training.py`、以及两文件还原后的逐字节核对）。
  **如实说明：本轮那一次运行的 stdout 未归档**（约 13 分钟，未重跑以捕获），
  故"4 杀 8 存活"与"T-A killed"这两条**可由该脚本原样重跑复现，但没有留存的日志文件**。
  **这与本 session 其他各轮都附了 `mutation_*.txt` 的做法不一致，是本轮的一处欠缺**；
  下一轮若继续做链 2 的补测，应连带把这次运行重跑并归档。

## 勾选审计续五十二（2026-09-07 01:58）：链 2 补测——上一轮 4 条"真而未钉/未核实"**全部钉住**，12 个变异从 4 杀 8 存活变成 **8 杀 4 存活**；过程中查出上一轮那个"已杀"的 T-F 是**假杀**（死于 `KeyError` 而非被测语义），换忠实变异后**结论仍成立——但只在"按分数排序"那一行还在的前提下成立，而那一行当时没被钉**

> **一处时钟更正**：续四十九/五十/五十一三个标题里的时间（06:35、07:20、08:05）与它们各自的提交时间
> （`98c3bb6` 00:39、`53fd964` 00:48、`38559c1` 01:11，均为 +0800）对不上，偏移还不固定。
> **本节起标题时间取机器实际本地时间**，以便与 `git log` 交叉核对。

### 一、这轮做了什么

续五十一把 T-B 与 T-I 登记为"真而未钉"、把 T-D 与 T-E 登记为"未核实"，并说明了为什么当轮不补测试
（两个夹具都需要先读懂参差夹具的构造方式，且必须自带"夹具自证"，否则可能重复旧夹具的错）。
本轮就是把这件事做完，外加把两条"未核实"变成实测：

1. **4 条新测试**，每条都带一条**夹具自证**断言（不是单独的控制测试，而是同一测试内的前置断言，
   这样夹具一旦退化，测试会**因为自证失败而失败**，不会静默变成空测）：
   - `tests/unit/models/test_model_batch_contracts.py`
     - `test_predictions_to_frame_keeps_tied_scores_at_their_average_rank`（钉 **T-B**）
     - `test_predictions_to_frame_refuses_scores_whose_shape_disagrees_with_the_mask`（钉 **T-D**）
     - `test_predictions_to_frame_orders_each_date_by_descending_score`（钉 **T-E**）
   - `tests/unit/scripts/test_run_real_candidate_training.py`
     - `test_percentile_frame_is_date_ordered_whatever_order_the_mapping_iterates`（钉 **T-I**）
2. **两条探针实测**（T-D、T-E），把上一轮"读代码推断"的证据等级提到与续三十七/四十六/四十九同级。
3. **查出 harness 里 T-F 变异不忠实**，改成忠实变异（见第四节）。
4. **全量 harness 重跑并归档日志**——补上续五十一自己登记的那处欠缺（上一轮 stdout 未留存）。

### 二、T-B 与 T-I：钉住了，但两条的"为什么值得钉"强弱不同，要分清

| | 变异 | 新钉子 | 夹具自证 | 定性 |
|---|---|---|---|---|
| **T-B** | 百分位 `rank(method="average")` → `"first"` | 4 只股票、分数 `[1,2,2,3]`，断言并列的 b/c **同为 0.625**（2/4 与 3/4 的平均） | `frame["raw_score"].duplicated().any()` 为真 | **真缺陷**：并列时按位置定序，等于把"同一日期内的行序"（即 `ts_code` 序）注入到**用来选出优胜配置的那个分数**里；而 `static_batches.py:123-126` 的训练目标秩明确用 `average`，理由就写在它的注释里（"与评价的 Spearman 秩一致"）。**真实模型确实会产出精确并列**：塌缩的输出头或死掉的 ReLU 单元会让大量股票在 float32 下同分 |
| **T-I** | `_percentile_frame` 遍历 `sorted(batches)` → `batches` | 用参差夹具构造一个**键序被打乱**的 dict（先 01-03 后 01-02），断言输出行序仍按日期 | `list(unordered) != sorted(unordered)` | **防御性、当前不可达**（这一条是对续五十一"真而未钉"的**下调**，见下） |

**T-I 需要如实下调定性**。续五十一说"若调用方传入未按日期插入的映射，输出行序就跟着变"——这句是对的，
但本轮把所有调用方与所有消费方都核实了一遍，**当前没有任何一处会观察到这个行序**：

- 两个批构造器都按日期序建 dict：`static_batches.py:79` 是 `groupby("signal_date", sort=True)`，
  `temporal_batches.py:92` 是 `all_dates = sorted(normalized["signal_date"].unique())`；
  `_temporal_panel_batches`（驱动 1320-1331 行）只是**过滤**那个 dict，不改变序。
- 下游全部按业务键合并或重排：`_validation_rank_ic` → `compute_daily_rank_ic`（merge on keys）、
  `combine_seed_percentile_ranks`（`ensemble.py:70-78`，merge on keys 且末尾 `sort_values(keys)`）、
  `FrozenPredictionStore._normalize_scores`（`selection_evidence.py:456`，**在写入并哈希之前**就按
  `["signal_date","ts_code"]` 重排）。

**故 T-I 的钉子钉的是"文档化契约本身"**（`_percentile_frame` 的 docstring 承诺 in date order），
价值在于**将来某个调用方传入非日期序映射时不会静默改变输出行序**，而**不是**它今天挡下了一个活缺陷。
这条与续五十一的表述有强弱差别，按本轮的核实结果记。

### 三、T-D 与 T-E：探针实测的结论**与上一轮的推断不同**

**T-D（去掉 `scores.shape != stock_mask.shape` 检查）——上一轮写"未核实：下游大概也会抛"。实测：不抛，而且是静默给出错数。**

探针用 `stock_mask` 为 `(1,3)`（`[True, True, False]`，即两只真股票 + 一个 padding 槽）的批，
喂两种形状不符的 scores：

| scores 形状 | 有守卫 | 去掉守卫 |
|---|---|---|
| `(2,3)`——**多一个日期行** | `ValueError: scores must have shape [date, stock]` | **NO ERROR**，返回 `['a','b'] [9.0, 8.0] [1.0, 0.5]` |
| `(1,4)`——**多一个股票列** | 同上 `ValueError` | **NO ERROR**，静默取前两列 |

多一个日期行那一格是关键：`date_index` 0 会读到**模型放进第 0 行的任何数字**（探针里是 `9.0, 8.0`），
把它们当作 2025-01-02 这一天的分数排出百分位，**返回的帧长得完全合法**——列齐、键齐、秩在 (0,1] 内。
**这不是"换个异常类型"，是 fail-open**：错误被排成了看起来正确的百分位秩，
再往下就是用它去算选优的 validation RankIC。**故 T-D 是真守卫，本轮已钉。**

**T-E（去掉按分数排序）——上一轮写"未核实：行序可能无关"。实测：行序确实无关，但它是 T-F 那条钉子成立的前提。**

先说"无关"这一半：如第二节所列，所有消费方都按键合并或重排，**单删排序，全量套件存活**（28 passed）。

再说要紧的那一半。`_mean_seed_percentile_scores` 的 docstring 自己写着：
"`predictions_to_frame` sorts each date by score, so two seeds generally emit their rows in different
orders and averaging the columns positionally would blend different stocks"——
**那条"按业务键而非按位置平均"的回归测试之所以能区分对错，全靠这个排序**。
探针把 T-E 与**忠实的**按位置平均变异（见第四节）**组合**起来跑：

| 组合 | 结果 |
|---|---|
| 忠实 T-F 单独（排序还在） | **killed**——`comparison failed. Mismatched elements: 30 / 31`，正是"斜坡 vs 常数 32/62" |
| **T-E + 忠实 T-F**（排序删掉） | **SURVIVED**——28 passed |

**即：删掉排序会把那条回归测试静默变成同义反复**——两个 seed 的行序变得相同，
按位置平均与按键平均给出同一个答案，测试再也分不出对错。
**"未被观察的行序"与"不重要的行序"是两回事**：它是另一条钉子的**前提**，
而**前提没被钉，等于那条钉子可以被无声撤走**。故 T-E 本轮已钉；
钉上之后重测该组合，**从 SURVIVED 变 killed**（1 failed, 32 passed）。

（同一处依赖也写进了那条回归测试的 docstring，注明它的鉴别力是"借来的"，并指向排序钉子。）

### 四、上一轮的一处**证据缺陷**：T-F 那个变异不忠实——"被杀"与"因为该杀的原因被杀"是两件事

harness 里 T-F 原来是把 `on=keys, validate="one_to_one"` 换成 `left_index=True, right_index=True`。
这确实**意图**表达"按位置平均"，但它同时**保留了两套键列**：pandas 会把它们后缀成
`signal_date_x / ts_code_x / signal_date_y / ts_code_y`，于是函数自己那句
`merged.loc[:, [*keys, "score"]]` 抛 `KeyError: "['signal_date', 'ts_code'] not in index"`。
**变异体死了，但死因与被测语义无关**——它还没来得及"把不同股票混在一起平均"，就先因为列名崩了。

本轮换成忠实的按位置变异：保留 0 号帧的键，其它 seed 只按**行位置**取秩列
（`pd.concat([merged, column], axis=1)`），不崩、且真的会把不同股票混在一起。实测：

| 变异 | 结果 | 死因 |
|---|---|---|
| harness 原 T-F（index merge） | killed | `KeyError: "['signal_date', 'ts_code'] not in index"`——**附带原因** |
| 忠实 T-F（按行位置取秩列） | killed | `comparison failed. Mismatched elements: 30 / 31`——**正是被测语义**（斜坡 vs 常数 32/62） |

**故续五十一那句"`0cbbc88` 的另一半已被钉住"结论不变**，而且现在有了**正确死因**的证据。
**但本轮同时查出它的成立条件**：忠实 T-F 之所以被杀，靠的是两个 seed 的行序不同；
行序不同靠的是排序那一行——**而那一行当时没被钉**（第三节）。
harness 里的 T-F 定义已改为忠实版本，本轮归档的日志用的是改后的定义。

**这条教训与 `cf04c46` 那轮"harness 双向污染结果"是同一类**：变异测试的**输出本身**也需要被审。
**"杀了"必须追问"因为什么杀的"**，否则一个因附带原因崩掉的变异体会让审计**高估**测试的鉴别力。
本轮的做法是：对每一个声称"已杀"的关键变异，**核对失败信息里的死因**，而不是只看 returncode。

### 五、门槛与证据（全部实测，含本轮归档的日志）

- **全量 harness 重跑**（范围 `tests/unit/models tests/unit/training tests/property
  tests/unit/scripts/test_run_real_candidate_training.py`）：**基线 120 passed**
  （上一轮 116 + 本轮 4 条新测试），**12 个变异：8 杀 4 存活**——
  **杀**：T-A、**T-B**、T-C、**T-D**、**T-E**、T-F（**忠实版**）、T-H、**T-I**（加粗为本轮新增的 4 条）；
  **存活**：**T-G、T-J、T-K、T-L**——**正是上一轮已定性的那四条**（T-J 经核实是**正确的不对称**、
  T-G 按上游契约不可达、T-K/T-L 只是换异常类型与文案），**本轮没有新增存活**。
  两个被改文件**还原后逐字节相同**，还原后复跑仍 **120 passed**。
  **stdout 已归档**：`mutation_candidate_scoring_after_fixtures.txt`——**补上续五十一自己登记的那处欠缺**。
- **定向杀验证**（每条钉子单独跑，避免 13 分钟的全量把细节压成一行）：四条各
  **control PASS → mutated killed → restored PASS**；组合验证
  **T-E + 忠实 T-F = killed**（钉排序之前是 SURVIVED）、**忠实 T-F 单独 = killed（死因正确：
  `Mismatched elements: 30 / 31`）**、**harness 原 T-F = killed（死因是 `KeyError`）**、
  还原后 **33 passed**、两文件逐字节还原。
  证据：`mutation_candidate_scoring_targeted_and_probes.txt`（含两次探针的原始输出）。
- **全量门槛**：`pytest tests -q` → **1621 passed / 45 skipped**（续五十实测的 1617 + 本轮 4 条，
  **skipped 数未变**）；`ruff check src tests` 清；`mypy src` **158 文件清**。
- **本轮只改测试**：`git diff --stat` 对 `research_v2/src`、`research_v2/scripts` **为空**；
  改动是 `tests/unit/models/test_model_batch_contracts.py`（+3 测试）与
  `tests/unit/scripts/test_run_real_candidate_training.py`（+1 测试、+1 处 docstring 说明依赖来源）。
  **未改任何契约、schema、门阈值或 `dropna` 口径。**
- **未写任何数据库**；**未碰** `materialization_status.{json,md}`（工作树里那两个文件仍是另一个对话的改动，
  本轮全程未触碰）；**未释放** `/tmp/frozen_chain_hold.txt`；**链 2 仍未重启**（前置见 §0d 批 2：
  清理 13 个陈旧检查点、指针并列守卫、续跑键含契约版本）。
- **裁决 2/3/4 与决定 A–N 全部仍开放。**
- 归档到 `/data/zhangyudong/qrv2_research_artifacts/`：
  `mutation_candidate_scoring_after_fixtures.txt`（全量 12 变异日志）、
  `mutation_candidate_scoring_targeted_and_probes.txt`（定向 + 两次探针原始输出）、
  `mutate_candidate_scoring.py`（**T-F 已改为忠实版**）、`mutate_candidate_scoring_targeted.py`、
  `probe_candidate_scoring_shape_and_sort.py`、`probe_candidate_scoring_faithful_positional.py`。

### 六、本轮暴露出的**方法论修正**，以及下一轮的候选

**方法论修正（比任何单条钉子都重要）**：变异测试的 `killed` 是一个**需要被审的输出**，不是一个结论。
本轮之前，harness 只读 `returncode`；从本轮起，**对关键变异必须核对失败信息里的死因**
（`FAILED` 的测试节点 + 首行错误类型），因为一个因附带原因崩掉的变异体会让审计**高估**自己的测试鉴别力——
T-F 就是实例。同理，**"存活"也需要区分三种**：不可达（T-G）、只换文案（T-K/T-L）、
以及**正确但没人钉**（T-J）——第三种的处置方式是**钉它**，而不是把它记成"没问题"。

**下一轮候选（按本轮结果排序，非承诺）**：

1. **钉 T-J**（`_percentile_frame` 里那处唯一的 `model.eval()`）。**这是链 2 剩下四条存活里唯一
   "一旦弄反就是真缺陷"的一条**：续五十一已核实这个不对称是**正确的**（`_percentile_frame` 的调用方
   拿到的是训练环出来的 train-mode 模型，而 `_mean_seed_percentile_scores` 拿到的是
   `artifact_loader.py:111` 已 `eval()` 过的模型），而 **M2/M3 确实用 `nn.Dropout(0.1)`**，
   故若把它弄反，喂给 G1–G6 的冻结 TEST 预测就变成随机的。**正确的行为没被钉，和错误的行为没被钉，
   风险是一样的**——它可以在任何一次"看起来无害的重构"里被移走。
2. **T-G/T-K/T-L 的处置**：按上一轮的定性它们是"不可达 / 只换文案"，
   **本轮没有推翻也没有实测**（只做了代码推断）；若要把它们提到探针级证据，
   需要构造"同一日期内同一 ts_code 出现两次"的输入（`FeatureBatch.__post_init__` 禁止），
   即需要先确认这个禁止是否真的在所有构造路径上成立。
3. **链 2 之外**：本 session 仍有约 1,530 个测试函数在已审计的两条链之外未做变异测量，
   按续五十的收口方式，下一条链应挑**门阈值最集中**的那一段（而非测试最多的那一段）。

## 勾选审计续五十三（2026-09-07 02:25）：钉上链 2 剩下四条存活里**唯一"弄反就是真缺陷"的那条**（T-J）——12 个变异现在 **9 杀 3 存活**；顺带查出这个"正确不对称"的**另一半早就有钉子**，也就是说它一直是**半钉状态**，而这比全没钉更容易骗过读测试的人

### 一、为什么下一条是 T-J，而不是另外三条存活

续五十二第六节把候选排了序，T-J 排第一。理由不是它"存活"，而是它的**方向**：
剩下三条（T-G 按契约不可达、T-K/T-L 只换异常类型与文案）**反转后都不构成缺陷**，
而 T-J **反转后是缺陷**。`_percentile_frame` 里那处全驱动唯一的 `model.eval()`：

- 它的两个调用点（驱动 357、672 行）拿到的是**训练环刚出来的模型**，
  而 `training/torch_loop.py:55` 是 `self._model.train()`；
- **M2/M3 确实用 `nn.Dropout(0.1)`**（`models/temporal.py`、`models/set_attention.py`）；
- 故删掉这一行 = **在 dropout 生效的状态下给冻结 TEST 打分** =
  喂给 G1–G6 的那批预测**逐次运行都不一样**。

**"正确但没被钉"与"错误但没被钉"承担同样的风险**：它会被任何一次看起来无害的重构移走，
而全量套件一声不响。

### 二、核实了不对称的另一半：**它早已被钉**，所以缺口是"半钉"而不是"全裸"

`_mean_seed_percentile_scores` 自己不 `eval()` 是**对的**——它的模型来自 `load_inference_artifact`，
而 `inference/artifact_loader.py:111` 已经 `model.eval()`。本轮去查那半边有没有测试：**有**，
`tests/unit/inference/test_artifact_loader.py:68`：
`assert all(not candidate.training for candidate in loaded.models)`。

**这反而让缺口更值得记一笔**：一个不对称的两半，一半有钉子、一半没有，
**读测试的人会以为"模式处理这块已经测过了"**。
**半钉状态比全没钉更容易骗过人**，因为它发出了"已覆盖"的信号，
而这个信号只覆盖了不会被驱动直接调用的那一半。

### 三、钉子与它的自证

`tests/unit/scripts/test_run_real_candidate_training.py`：

- 新夹具 `_ModeRecordingScorer`（继承已有的 `_StubScorer`）：按 `nn.Module` 的语义维护 `training`，
  并在**被调用的那一刻**把模式记下来。
- `test_percentile_frame_scores_in_eval_mode_when_handed_a_train_mode_model`：
  先自证 `scorer.training is True`（**模型真的是以 train 模式到达的**，正如训练环留下的那样），
  再断言 `modes_at_call == [False, False]`（两个日期各一次调用，**两次都在 eval 模式下发生**）。
- **钉的是"调用时刻的模式"，不是"有没有调用过 `eval()`"**——
  后者可以被一个不生效的 `eval()`（比如返回新对象而不改自身）骗过。
- docstring 里写清了两半的分工**与另一半钉子的确切位置**，免得下次又只钉一半。

### 四、门槛与证据（全部实测）

- **定向杀验证**：T-J **control PASS → mutated killed → restored PASS**；
  五条钉子（T-B / T-D / T-E / T-I / T-J）**全部同样通过**；
  续五十二那三条组合复测不变——**T-E + 忠实 T-F = killed**、
  **忠实 T-F 单独 = killed（死因正确：`Mismatched elements: 30 / 31`）**、
  **harness 原 T-F = killed（死因是 `KeyError`）**；
  两个测试文件合计 **34 passed**，两文件**还原后逐字节相同**。
- **全量 harness 重跑**：基线 **121 passed**，12 个变异 → **9 杀 3 存活**
  （杀：T-A、T-B、T-C、T-D、T-E、T-F 忠实版、T-H、T-I、**T-J**；存活：**T-G、T-K、T-L**），
  **无新增存活**，还原后逐字节相同并复跑通过。
  日志 `mutation_candidate_scoring_after_eval_pin.txt`。
- **全量门槛**：`pytest tests -q` → **1622 passed / 45 skipped**；`ruff check src tests` 清；
  `mypy src` **158 文件清**。
- **本轮只改测试**：对 `research_v2/src`、`research_v2/scripts` 的 `git diff --stat` **为空**；
  **未改任何契约、schema、门阈值或 `dropna` 口径**；**未写任何数据库**；
  **未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`；
  **链 2 仍未重启**；**裁决 2/3/4 与决定 A–N 全部仍开放**。

### 五、链 2 这条线到此的收口判断

12 个变异：**9 杀 3 存活**，三条存活都有定性、**都不是活缺陷**：

- **T-G**（seed 合并去掉 `validate="one_to_one"`）：按上游契约不可达——
  要产生重复 `(signal_date, ts_code)` 需要同一日期内同一 `ts_code` 出现两次，
  而 `FeatureBatch.__post_init__` 禁止；
- **T-K / T-L**（去掉两个空输入守卫）：去掉后分别是 `frames[0]` 的 `IndexError` 与
  `pd.concat([])` 的 `ValueError`，**都会抛**，故不是 fail-open，只是换了异常类型与文案。

**如实标注证据等级**：这三条的定性仍是**读代码推断**，没有像 T-D/T-E 那样构造出区分输入。
要提到探针级需要——T-G：确认"同一日期同一 `ts_code` 两次"在**所有**构造路径上都被禁止
（不只 `FeatureBatch.__post_init__`，还有 `TemporalFeatureBatch` 与驱动自己的构造点）；
T-K/T-L：实测去掉守卫后**抛的是什么、在哪一行抛**。
**但这两件事的收益都低于去开下一条链**：它们反转后都不构成缺陷，
而本 session 还有约 1,530 个测试函数在已审计的两条链之外从未做过变异测量。

**故本轮把链 2 的变异审计收口**：除非有人要改 `_percentile_frame` /
`_mean_seed_percentile_scores` / `predictions_to_frame` 这三处，
那时应连带重跑 `mutate_candidate_scoring.py`（已归档，T-F 为忠实版）。
下一轮按续五十的收口方式**挑下一条链**——标准是**"门阈值最集中"**，不是"测试最多"。

## 勾选审计续五十四（2026-09-07 02:45）：**开链 3 = OOS 晋级门 G1–G6**（`evaluation/gates.py`，全系统阈值最密集处，此前**从未做过变异测量**）——16 个变异 **6 杀 10 存活**；其中 **U-K 把 G5 从"最大年贡献占比"改成"最小年贡献占比"，全套件一声不响**，也就是说**那条防止 alpha 由单一年份撑起来的门，它的失败路径一个测试都没有**

### 一、为什么选这条链（按续五十/续五十三定的标准：门阈值最集中，不是测试最多）

- `src/research_v2/evaluation/gates.py` **147 行**里放了**六个晋级阈值**
  （G1 mean RankIC ≥ 0.030、G2 HAC t ≥ 2.000、G3 正年份占比 ≥ 0.70、G4 delta ≥ 0.005、
  G5 最大年贡献占比 ≤ 0.50、G6 分数健康日占比 ≥ 0.95）、
  **三个逐日分数健康边界**（coverage ∈ [0.90, 1.0]、unique ≥ 10、std > 1e-6），
  以及一组 fail-closed 的 nan 路径。**这六道门决定一个模型到底能不能被晋级。**
- **它此前从未被变异测量过**：计划文档全文**零次**提到 `evaluate_oos_gates`（`rg -c` 结果为 0），
  已归档的 8 个 `mutation_*.txt` 里也没有它。
- 它的测试只有 `tests/unit/evaluation/test_gates.py`：**143 行 / 5 个测试**，
  对应六道门 + 三条逐日边界 + 若干 fail-closed 路径。

### 二、结果：16 个变异，6 杀 10 存活（基线 162 passed；`gates.py` 还原后逐字节相同）

**杀掉的 6 条**（每条都按续五十二的方法论修正**核对了死因**，不只看 returncode）：

| 变异 | 死因 |
|---|---|
| **U-C** 门比较改严格（`>=` → `>`） | `AssertionError: assert <GateStatus.FAIL> is ...`，**2 个测试**——恰好卡在阈值上的证据不再通过 |
| **U-H** 基线不再豁免 G4 | `assert 'FAIL' == 'NOT_APPLICABLE'`，**3 个测试** |
| **U-N** 每天都算分数健康（G6 永不失败） | `assert 'PASS' == 'FAIL'`，**3 个测试** |
| **U-P** 逐日 coverage 下限 0.90 → 0.89 | `assert <GateStatus.PASS> ...`，1 个测试 |
| **U-R** 逐日离散度 `> 1e-6` → `>= 1e-6` | 同上，1 个测试 |
| **U-T** `all_passed` 不再接受 NOT_APPLICABLE | `assert False`，1 个测试 |

**存活的 10 条**，按"是不是活缺陷"分三类：

**（1）语义被整个反转、而全套件不响——本轮最重要的发现**

| 变异 | 后果 |
|---|---|
| **U-K** | G5 从"**最大**年贡献占比"变成"**最小**年贡献占比"。**G5 存在的唯一理由是防止 alpha 由单一年份撑起来**；换成 min 之后它量的是"最弱的那一年有多弱"，**与集中度无关，而且永远更容易通过**。存活原因很具体：夹具 `_evidence()` 的六个年份是 `0.01×5` 与 `-0.01`，**正贡献全部相等**，max 与 min 给出同一个数（0.2）。**即 G5 的失败路径没有任何测试** |
| **U-L** | G5 的分母不再把负年份截到 0，于是**一个亏钱的年份会缩小分母、抬高占比**——让另一个年份看起来更集中。同样因为夹具年份太对称而存活 |
| **U-I** | G5 的 `<= 0.50` 变成 `< 0.50`：**恰好 0.50 的集中度会从"通过"变"不通过"**。夹具占比 0.2，够不着边界 |
| **U-D** | G3 的 `value > 0` 变成 `>= 0`：**一个均值恰好为 0 的年份会被算成"正年份"**。夹具里没有 0 年份 |

**（2）阈值可以悄悄漂移——六个阈值里，本轮试的三个都只钉了"恰好通过"，没钉"差一点就失败"**

| 变异 | 漂移 | 为什么存活 |
|---|---|---|
| **U-A** | G1 `0.030` → `0.029` | `test_double_gate_boundaries_..._pass_at_exact_thresholds` 只证明"恰好等于阈值时通过"，**它能抓到向上漂移（变严），抓不到向下漂移（变松）**——而变松才是危险方向 |
| **U-G** | G4 `0.005` → `0.004` | 同上（夹具 delta 恰好 0.005） |
| **U-M** | G6 `0.95` → `0.94` | 同上（夹具恰好 95/100 健康日） |

**G2 与 G3 本轮没有试漂移**（**故下面这两句是读夹具推断，未实测**）：
G2 的夹具恰好卡在 2.000，**推断与 G1/G4/G6 同型**（只钉上侧）；
**G3 更松**——它的夹具是 5/6 = 0.8333，**离 0.70 的阈值还有余量**，
故 G3 连"恰好通过"都没钉住，**要向上漂到 0.84 以上才会被抓到**。

**（3）fail-closed 守卫被拿掉——但晋级路径上有第二套更严的实现兜住了，报告路径没有**

| 变异 | 晋级路径（registry） | 报告路径（`all_passed`） |
|---|---|---|
| **U-F** 缺失的 candidate delta 由 FAIL 变 NOT_APPLICABLE | **不可达**：`model_registry._parse_gate_evidence` 用 `_finite_number` 解析 `candidate_delta_rank_ic`，**null 当场被拒**；且 `validate_walk_forward_evidence` 要求六道门**逐一为 PASS**，NOT_APPLICABLE 同样算失败 | **可达**：`evaluate_frozen_oos` 的签名允许 `baseline_predictions=None` 且 `is_baseline=False`，此时 delta 为 None，今天 G4=FAIL → `all_passed=False`；变异后 G4=NOT_APPLICABLE → **`all_passed` 可能变 True**，而驱动自己的准入检查读的正是报告里的 `oos.all_passed`（`run_real_candidate_training.py:187`） |
| **U-E** G3/G5 不再要求六个年份齐全且有限 | **不可达**：`_parse_gate_evidence` 要求 `set(yearly) == set(range(2020, 2026))` 且每个值有限 | 直接调用 `evaluate_oos_gates` 时可达（例如只覆盖单一年份的折叠评价） |
| **U-O** `valid_test_day_count <= 0` 不再被拒 | **不可达**：`_parse_gate_evidence` 要求它是 `Integral`、非 bool、且 `> 0` | 直接调用时，去掉守卫会把 fail-closed 的 nan 换成 **`ZeroDivisionError`** |

**这一栏是本轮读代码读出来的一个结构性事实，值得单独记**：
`model_registry.load_walk_forward_evidence` → `validate_walk_forward_evidence`
实际上是**第二套、更严的门实现**（自己解析原始 JSON、要求六个年份齐全、要求 delta 有限、
要求 day count 为正、要求六道门逐一为 PASS），
**它把 `gates.py` 里这三处 fail-open 变异在晋级路径上全部堵住了**。
**但它只覆盖晋级路径**：驱动写进 run 报告、再被 `summarize_real_*` 与驱动准入检查读取的
`oos.all_passed` 走的是 `evaluate_frozen_oos` → `GateReport.all_passed`，
**而在那条路上 NOT_APPLICABLE 等于通过**（U-T 反向证明了这一点：把 NOT_APPLICABLE 从
`all_passed` 的接受集里拿掉，测试就响了）。
**故"某个变异可不可达"必须按路径分别回答，不能一句话概括**——
这也是本轮把三条守卫类存活判为"晋级路径不可达 / 报告路径可达"而不是简单判为"不可达"的原因。

### 三、门槛与证据

- 变异范围 `tests/unit/evaluation tests/unit/monitoring tests/unit/registry tests/property`：
  **基线 162 passed**，16 个变异逐个跑，**`gates.py` 还原后逐字节相同**，还原后复跑 **162 passed**。
- **本轮零代码改动、零测试新增**（纯测量 + 文书）。故全量门槛沿用续五十三的实测值
  **1622 passed / 45 skipped**，且沿用**有据**：`git diff --stat` 对 `research_v2/src`、
  `research_v2/scripts`、`research_v2/tests` 自 `788595b` 起**为空**。
- `ruff check src tests` 清、`mypy src` **158 文件清**（本轮未改代码，二者亦未变）。
- **未写任何数据库**；**未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`；
  **链 2 仍未重启**；**裁决 2/3/4 与决定 A–N 全部仍开放**。
- 证据：harness `mutate_oos_gates.py`、日志 **`mutation_oos_gates_first_round.txt`**
  （**每条 killed 都带失败测试节点与前两行 `E` 死因**——这是续五十二那条方法论修正第一次在新 harness 里内建）。

### 四、下一轮（按本轮结果排序，非承诺）

1. **钉 G5 的失败路径（U-K / U-L / U-I）**：需要三个年份字典——
   "单一年份占大头"（杀 U-K）、"恰好 0.50"（杀 U-I）、"负年份改变判定"（杀 U-L）。
   **U-K 是本轮唯一一条把门的语义整个反转却无人察觉的变异，优先级最高。**
2. **钉三个阈值的下侧（U-A / U-G / U-M）**：各加一条"差一点就失败"的证据
   （0.0299 / 0.0049 / 94 of 100）。与现有的"恰好通过"配对后，阈值被夹在两侧之间，
   **任何方向的漂移都会被抓到**；顺带把 G2、G3 的漂移也实测一遍（本轮只做了推断）。
3. **钉 U-D**（均值为 0 的年份不算正年份）与 **U-F**（缺 delta 必须 FAIL、且 `all_passed` 必须为 False）。
4. **U-E / U-O 的处置**：晋级路径不可达这一点已由 `_parse_gate_evidence` 的实读支持；
   若仍要钉，应钉在**直接调用 `evaluate_oos_gates`** 的层面（即"守卫自身 fail-closed"），
   并在 docstring 里写清它防的是直接调用方、不是 registry——否则会重复本轮"半钉"的误读问题。

## 勾选审计续五十五（2026-09-07 03:00）：链 3 补测——G5 的失败路径、五个阈值的下侧、三处 fail-closed 读法全部钉上，**16 个变异从 6 杀 10 存活变成 16 杀 0 存活**；并把定向 harness 的输出升级成**节点级归属**，因为"死了"还得知道"是被谁弄死的"

### 一、这轮做了什么

续五十四测出 10 条存活并排了序；本轮按那个顺序把可钉的全部钉上：**9 条新测试**
（`tests/unit/evaluation/test_gates.py` 从 **5 个测试变 14 个**），覆盖
G5 的失败路径（U-K / U-L / U-I）、五个阈值的下侧（U-A / U-G / U-M，**外加续五十四只做推断、本轮实测的 G2 与 G3**）、
G3 的零年份语义（U-D）、缺失 delta 的 fail-closed 读法（U-F）、
"不可计算必须报成不可计算"（U-E）、以及日数守卫（U-O）。
**门本身一行没动**：`gates.py` 全程只在变异期间被改、之后逐字节还原。

### 二、结果：16/16 全杀，且**每条都记录了是谁杀的**

本轮把定向 harness 的输出从"returncode + 摘要"升级成**节点级归属**（`killers=<测试名>`）。
理由是续五十二那条方法论修正的**对称面**：上一轮的问题是"变异死了，但死因与被测语义无关"；
**这一轮要防的是"变异死了，但功劳被记到错误的测试头上"**——
一条变异被多个测试同时抓到时，删掉其中任何一个它都不会复活，
审计若把功劳记在某个单一测试上，就会**同时高估那个测试的必要性、低估这套测试的冗余度**。

| 变异 | 杀手数 | 杀手（测试名缩写） |
|---|---|---|
| **U-K** G5 用 min 代替 max | **3** | `g5_fails_when_one_year_carries` / `g5_passes_at_exactly_half` / `g5_clips_negative_years` |
| U-C 比较改严格 | 2 | `baseline_marks_relative_gate...` / `double_gate_boundaries...` |
| U-H 基线不再豁免 G4 | 2 | `baseline_marks_relative_gate...` / **`missing_candidate_delta_fails_g4`（本轮新增）** |
| U-N 每天都算健康 | 3 | `g1_g4_and_g6_fail_just_below`（新）/ 两条原有 score-health 测试 |
| U-P coverage 下限 0.90→0.89 | 2 | `score_health_requires_coverage...` / **`g1_g4_and_g6_fail_just_below`（新，副作用）** |
| U-A / U-G / U-M 阈值下漂 | 各 1 | `g1_g4_and_g6_fail_just_below_their_thresholds`（新） |
| U-D 零年份算正 | 1 | `g3_does_not_count_a_zero_year_as_positive`（新） |
| U-E 不再要求年份齐全 | 1 | `incomplete_evidence_reports_not_computable`（新） |
| U-F 缺 delta 变 NOT_APPLICABLE | 1 | `missing_candidate_delta_fails_g4_and_the_whole_report`（新） |
| U-I G5 `<=`→`<` | 1 | `g5_passes_at_exactly_half_and_fails_just_above`（新） |
| U-L G5 不截断负年份 | 1 | `g5_clips_negative_years_out_of_the_denominator`（新） |
| U-O 日数 ≤ 0 不再被拒 | 1 | `non_positive_valid_day_count_fails_g6_instead_of_dividing_by_zero`（新） |
| U-R 离散度 `>`→`>=` | 1 | `score_health_requires_coverage_unique_scores_and_dispersion`（原有） |
| U-T `all_passed` 不接受 N/A | 1 | `baseline_marks_relative_gate_not_applicable`（原有） |

**一处如实说明**：U-P 被新测试 `g1_g4_and_g6_fail_just_below_their_thresholds` 抓到是**副作用而非设计**——
那条测试的"不健康日"用 coverage=0.89，**恰好落在 U-P 变异后的边界上**，于是它顺手也成了 coverage 下限的钉子。
**碰上来的冗余和设计出来的冗余，可信度不一样**，故在此记明它是哪一种。

### 三、G5 那条洞为什么值得单写一段

`_evidence()` 的六个年份是 `0.01×5` 与 `-0.01`，**正贡献全部相等**。在这样的夹具上：
max 与 min 同值、截不截负数同值、`<=` 与 `<` 同值——
**G5 的四种实现错误全部不可见**。这不是"测试写得少"，
而是**夹具的对称性恰好抹平了被测门的全部判别力**，
而"5 个测试全绿"从外面看完全像是覆盖了。

本轮的三个 G5 夹具各打破一种对称：

| 夹具 | 年份 | G5 应有读数 | 杀掉的变异 |
|---|---|---|---|
| 单年独大 | `0.10` 对 `0.001×5` | 0.10/0.105 ≈ **0.952 → FAIL** | U-K（min 会给 0.0095 → PASS） |
| 恰好一半 | `0.10, 0.10` 对四个 `0.0` | **0.50 → PASS**；`0.1001` 则 FAIL | U-I（`<` 会把 0.50 判 FAIL） |
| 含负年份 | `0.20×3` 与 `-0.30` | 截断后 0.20/0.60 ≈ **0.333 → PASS** | U-L（不截断则 0.20/0.30 ≈ 0.667 → FAIL） |

**三个夹具各自带一条数值自证**：断言 `observed_value` 的**具体数值**
（`> 0.95`、`|x − 0.50| < 1e-12`、`== pytest.approx(0.20/0.60)`），而不是只断言 PASS/FAIL。
**只断言状态的话，一个把 G5 恒判 FAIL 的变异同样会被"抓到"**，
但那不是因为它测出了集中度——那正是续五十二在 T-F 上踩过的同一类坑，只是方向相反。

### 四、阈值现在被夹在两侧；顺带查出一个**测试名与它实际证明的东西不符**

| 门 | 阈值 | "恰好通过"侧 | "差一点失败"侧 |
|---|---|---|---|
| G1 | 0.030 | 原有（0.030） | **0.0299（新）** |
| G2 | 2.000 | 原有（2.000） | **1.999（新）** |
| G3 | 0.70 | **原夹具是 5/6 = 0.8333，并不在阈值上** | **4/6 = 0.6667（新）** |
| G4 | 0.005 | 原有（0.005） | **0.0049（新）** |
| G5 | 0.50 | **0.50（新）** | **0.1001/0.2001 略高于 0.5（新）** |
| G6 | 0.95 | 原有（95/100） | **94/100（新）** |

**G3 那一行是本轮读夹具读出来的一处名实不符**：原有测试叫
`test_double_gate_boundaries_and_all_other_oos_gates_pass_at_exact_thresholds`，
但 G3 的夹具值是 **0.8333，不在 0.70 上**，故"pass at exact thresholds"这个说法**对 G3 不成立**——
它对 G1/G2/G4/G6 成立（0.030 / 2.000 / 0.005 / 95 of 100），对 G3 只是"离阈值还有余量地通过"。
**续五十四据此推断"G3 要向上漂到 0.84 以上才会被抓到"，本轮实测确认了这个推断**
（新增的 4/6 失败侧现在把它夹住了）。
**本轮没有改那个测试名**——改名会让续五十四的记录与代码对不上，代价大于收益；差异记在这里。

### 五、门槛与证据（全部实测）

- **定向**（只跑 `tests/unit/evaluation/test_gates.py`，**每次 0.2s**，故 16 条变异各跑三遍也只花约 1 分钟）：
  **control 14 passed → 16 条变异逐条 killed 且记录杀手 → restored 14 passed**；
  `gates.py` **还原后逐字节相同**。日志 `mutation_oos_gates_targeted_with_killers.txt`。
- **全量 harness**（四路径）：**基线 171 passed**，**16 杀 0 存活**，
  还原后逐字节相同并复跑通过。日志 `mutation_oos_gates_16_of_16_after_pins.txt`。
- **全量门槛**：`pytest tests -q` → **1631 passed / 45 skipped**（续五十三的 1622 + 本轮 9 条，
  **skipped 数未变**）；`ruff check src tests` 清；`mypy src` **158 文件清**。
- **只改测试**：对 `research_v2/src`、`research_v2/scripts` 的 `git diff --stat` **为空**；
  **未改任何门阈值、契约、schema 或 `dropna` 口径**。
- **未写任何数据库**；**未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`；
  **链 2 仍未重启**；**裁决 2/3/4 与决定 A–N 全部仍开放**。

### 六、下一轮候选（按本轮结果排序，非承诺）

1. **`model_evaluator.py`——喂给门的证据是怎么算出来的**。本轮只测了 `gates.py` 的**判定**，
   **没测 `GateEvidence` 的组装**：`_score_health_evidence`（逐日 coverage / unique / std 从哪来）、
   `_valid_label_day_count`（G6 的分母）、`_match_prediction_coverage`（候选与基线的可比口径，
   即 G4 的 delta 到底在哪个截面上算）、`_reject_locked_or_future_dates`（2026 隔离）。
   **证据算错，门判得再准也没用**——而 G4 的"fold-matched"这个性质正是靠 `_match_prediction_coverage` 实现的。
2. **`registry/model_registry.py` 那套"第二道更严的门"**（`_parse_gate_evidence` /
   `validate_walk_forward_evidence`）同样未做变异测量。续五十四正是**靠读它**才把 U-E/U-F/U-O
   判成"晋级路径不可达"，**但"读它"与"测它"是两件事**——
   那三条不可达的结论，强度取决于这套解析器有没有被钉住。
3. 其余同前：约 1,530 个测试函数在已审计的三条链之外从未做过变异测量。

## 勾选审计续五十六（2026-09-07 03:45）：**开链 4 = 证据组装层** `model_evaluator.py`（G4 的 fold-matched 截面、G6 的分子分母、锁定期隔离都在这里）——首轮 16 个变异 **7 杀 9 存活**；**九条存活里六条做了探针实测，其中两条推翻了我自己的预判**（V-K 根本不是 fail-open，V-P 不是数值泄漏而是静默容忍）；补 6 条测试后 **13 杀 3 存活**，剩下三条有探针级证据证明它们**是三重冗余里的两重、或只换文案**

### 一、为什么这条链排在 registry 之前

续五十五第六节的排序理由是：`gates.py` **判**，`model_evaluator.py` **算证据**；
**判得对而证据算错，等于没有门**。本轮实测确认这个排序是对的——三个最要紧的性质都在这里，不在门里：

- **G4 的 "fold-matched"**：`_match_prediction_coverage` 把候选与基线限制在**完全相同的
  `(signal_date, ts_code)` 截面**上，delta 才不是拿两个不同宇宙在比；
- **G6 的分母**：`_valid_label_day_count`，以及**必须与它对齐**的分子 `_score_health_evidence`；
- **锁定期隔离**：`_reject_locked_or_future_dates` 与 `role != "TEST"` 的拒绝。

现有覆盖：`tests/unit/evaluation/test_frozen_oos_evaluator.py`，**171 行 / 6 个测试**。

### 二、首轮：16 个变异，7 杀 9 存活（基线 182 passed）

杀掉的 7 条，死因均已核对：

| 变异 | 死因 |
|---|---|
| **V-A** role 守卫去掉 | `Failed: DID NOT RAISE <class 'PermissionError'>` |
| **V-B / V-C** 2026 墙（`>=`→`>`、永不触发） | 同上（`test_frozen_oos_rejects_locked_year...`） |
| **V-D** 覆盖匹配由内连接改左连接 | `assert [30,30,30,30,30,30] == [29,29,...]`——**G4 的 fold-matched 性质是被钉住的** |
| **V-G** G6 分母 `>=`→`>` | `assert 'FAIL' == 'PASS'`，2 个测试 |
| **V-L** **delta 的两个操作数互换** | `assert -2.0 == 2.0`，3 个测试 |
| **V-N** 候选 RankIC 不再携带无效标签日 | `KeyError: 'rank_ic_valid'`，10 个测试 |

**V-N 的死因要如实记**：`include_invalid=False` 会让 `compute_daily_rank_ic` 不再输出 `rank_ic_valid` 列，
于是函数自己在 `daily.loc[daily["rank_ic_valid"]]` 崩掉。**即这个变异在任何真实运行里都会立刻炸，
不可能成为静默缺陷**——它的 "killed" 含金量低于其它几条：**不是测试认出了语义错误，是代码自己崩了**。

### 三、六条存活的探针实测：**两条推翻了我自己的预判**

| 变异 | 探针前的预判 | 探针实测（原始 vs 变异） | 定性 |
|---|---|---|---|
| **V-K** 去掉"至少五个有效日"守卫 | **fail-open：三天的证据也能出 OOS 结论** | 五日：原 `ValueError: OOS evaluation requires more than five valid daily RankIC observations`；变异 `ValueError: HAC series must contain more observations than max_lags`。**三日重测同样**：只是两句不同文案的 `ValueError` | **预判错误，只换文案**——`hac_mean_test(max_lags=5)` 要求观测数 > 5，**与五日守卫的阈值完全重合**，故去掉守卫**不可能**放行小样本结论 |
| **V-P** 基线帧不再查锁定期 | fail-open，2026 数据可能泄漏进数字 | 基线含一行 2026：原 `PermissionError: ... : baseline`；变异 **NO ERROR**，返回九日、六年、`delta 2.0` 的**完整正常结论**。而候选与标签也含 2026 时，**两种情况都被候选那侧拦下** | **不是数值泄漏，是静默容忍**：那一行被覆盖匹配的内连接丢掉了。**但"连接事后把它清掉了"不等于"输入在范围内"**——治理信号没了，而下一个不那么巧的产物不会被清掉 |
| **V-H** `label_valid` 的 NULL 当作有效 | 会放宽 G6 | 第九日 20 个有效 + 25 个 NULL：原 `valid_label_day_count = 8`、G6 **PASS 1.0**；变异 `= 9`、**G6 FAIL（observed None）** | **fail-closed**：分子分母错位 → nan → 拒绝。**但送进证据文件的日数是错的** |
| **V-I** 非有限 forward return 仍算有效日 | 会放宽 G6 | 一整日 30 个 NaN 收益：原 **5**（六日夹具）；变异 **6** | 同上：**fail-closed，但证据里的日数是错的** |
| **V-J** 薄日进入 G6 分子 | 方向不确定 | 一日只有 25 只：原 G6 **PASS 1.0**；变异 **G6 FAIL（observed None）** | **fail-closed（误拒）**：不会放过坏模型，只会把好模型判死——而这种失败在报告里长得像"模型变差了" |
| **V-M** 空基线不再被拒 | nan 会进报告 | 基线全常数：原 `ValueError: baseline has no valid OOS RankIC observations`；变异 **NO ERROR**，`baseline_mean=nan`、`delta=nan`、G4 FAIL；`json.dumps` 输出 `{"baseline_mean_rank_ic": NaN}`，**并用 `parse_constant` 实测确认它不是严格 JSON** | **门口 fail-closed（G4 拒绝），但 NaN 会走进 run 报告**，真正的拒绝被推到 registry 的 `_finite_number`——**离病因很远** |
| **V-O** 非有限年份均值不再被过滤 | 方向不确定 | 塌掉 2023 一整年：原 `yearly` **五个键**；变异**六个键且 `2023: nan`**。**G3/G5 两种情况都 FAIL** | **fail-closed**，但同样把 NaN 送进报告，且 registry 的拒绝理由从"缺年份"变成"非有限"——**线索更差** |
| **V-E / V-F** 覆盖合并的 `one_to_one` / 重复键守卫 | 防御性、只换文案 | V-F 变异后喂重复业务键：原 `ValueError: candidate predictions contain duplicate business keys`；变异 `MergeError: Merge keys are not unique in left dataset` | **两者互相冗余，且都不是第一重**：见下 |

**关于重复键，本轮顺手量到一个此前没人写下来的事实**：第一版探针里 V-F 没传基线，
于是量到的是**第三重守卫**——`compute_daily_rank_ic` 自己抛
`ValueError: predictions and labels must have unique signal-date keys`。
所以重复业务键一路上有**三道独立的拦截**：`compute_daily_rank_ic` 的唯一性检查、
`_match_prediction_coverage` 的重复键守卫（V-F）、以及两次合并上的 `validate="one_to_one"`（V-E）。
**V-E 存活是因为 V-F 先拦下，V-F 存活是因为 V-E 接着拦下，而两者都被拿掉时还有第三道。**
这是本轮"不钉 V-E / V-F"的依据，**不是推断，是三次实测**。

**方法论收获，与续五十二那条正好互补**：续五十二说的是"**killed 要追问因为什么杀的**"；
**本轮说的是"survived 也要追问'是不是我夹具没造对'"**——
第一版六个探针里**四个是废的**：V-F 没传基线（那个守卫只在有基线时才跑）、
V-J 与 V-O 把某一日弄塌后**正好撞上"至少五个有效日"守卫**（于是量到的是那个守卫，不是被测行为）、
V-H 传的是 `label_valid=False` 而不是 `None`（`fillna` 无事可做）。
**四个废探针里有三个报告的是"无差异"或"与预期一致"**——若不重做，就会被当成结论写进文档。
重做后的夹具统一改成**九日跨六年**，理由就是"弄坏任何一天也不会撞上五日守卫"。

### 四、补的 6 条测试，与补完之后的结果

`tests/unit/evaluation/test_frozen_oos_evaluator.py` 从 **6 个测试变 12 个**：

1. `test_nine_day_fixture_is_a_clean_complete_evaluation`——**夹具自证**（九日、六年齐全、G6 PASS、delta 2.0）
2. `test_locked_year_in_the_baseline_frame_is_refused_by_name`（钉 **V-P**）
3. `test_a_baseline_with_no_valid_observation_is_refused_where_it_happens`（钉 **V-M**）
4. `test_a_year_without_a_finite_mean_never_reaches_the_report`（钉 **V-O**）
5. `test_g6_denominator_ignores_null_label_flags_and_non_finite_returns`（钉 **V-H 与 V-I**）
6. `test_a_thin_day_stays_out_of_the_g6_numerator_so_the_two_sides_agree`（钉 **V-J**）

**没钉 V-E / V-F / V-K，理由写在这里而不是省略**：V-K 已被探针证明只换文案（与 HAC 的 `max_lags`
阈值重合，五日与三日两处都实测过）；V-E 与 V-F 是三重冗余里的第二、第三重，
**钉任一条都等于钉一个异常类型**，而"重复键必被拒"这个性质本身仍被剩下的两道覆盖。
**真正会变成洞的情形是"有人同时动这两处"**——这一点写进文档，比写成一条脆弱测试有用。

结果：

- **定向**（只跑 `test_frozen_oos_evaluator.py`，**1.8s/次**）：control **12 passed** →
  **13 killed（每条带杀手节点与死因）→ 3 survived（V-E / V-F / V-K）** → restored **12 passed**；
  `model_evaluator.py` **还原后逐字节相同**。日志 `mutation_oos_evidence_targeted_with_killers.txt`。
- **全量 harness**（五路径）：**基线 188 passed**，**13 杀 3 存活**，还原后逐字节相同并复跑通过。
  日志 `mutation_oos_evidence_after_pins.txt`。
- 值得记的归属变化：**V-G 从 2 个杀手变 5 个**、**V-B/V-C 从 1 个变 2 个**（新的基线墙测试也在盯 2026）、
  **V-N 从 8 个失败变 10 个**——但 **V-N 的死因仍是 `KeyError`，不是语义识别**，故其含金量不变。

### 五、门槛与证据

  **skipped 数未变**）；`ruff check src tests` 清；`mypy src` **158 文件清**。
  **warning 数由 3 变 5**，两条都来自第六节那条 pandas `FutureWarning`（新夹具里含 NULL 的
  `label_valid` 列会触发它）——**这是新测试把那条即将变更的行为纳入了监视，不是新问题**。
- **本轮只改测试**：对 `research_v2/src`、`research_v2/scripts` 的 `git diff --stat` **为空**；
  **未改任何门阈值、契约、schema 或 `dropna` 口径**。
- **未写任何数据库**；**未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`；
  **链 2 仍未重启**；**裁决 2/3/4 与决定 A–N 全部仍开放**。
- 归档：`mutation_oos_evidence_first_round.txt`（首轮 16 变异）、
  `mutation_oos_evidence_after_pins.txt`（补测后）、
  `mutation_oos_evidence_targeted_with_killers.txt`（定向 + 节点级归属）、
  `probe_oos_evidence{,2,3}.py` 与同名 `.log`（**三版探针，含四个废探针的原始输出**——
  留着是为了让"夹具造错会长什么样"有据可查）、`mutate_oos_evidence.py`、`mutate_oos_evidence_targeted.py`。

### 六、附带发现（不是本轮任务，但如实记）

`model_evaluator.py:175` 的 `normalized_labels["label_valid"].fillna(False)` 在 object dtype 列上触发
pandas `FutureWarning: Downcasting object dtype arrays on .fillna ... is deprecated and will change in
a future version`。**这一行正是 V-H 与 V-I 变异的那一行**，也就是说：
**将来 pandas 改变这里的语义时，本轮新增的
`test_g6_denominator_ignores_null_label_flags_and_non_finite_returns` 会立刻响**
（它把"NULL 不算有效、非有限收益不算有效"钉成了 G6 的可观测结果，而不是钉成实现细节）。
**本轮没有改这一行**——改 dtype 行为属于契约变更，超出"只改测试"的范围。

### 七、下一轮候选（按本轮结果排序，非承诺）

1. **`registry/model_registry.py` 的第二套门**（`load_walk_forward_evidence` /
   `_parse_gate_evidence` / `validate_walk_forward_evidence`）。
   **本 session 已经有三次结论是靠"读它"得出的**：续五十四把 U-E/U-F/U-O 判成"晋级路径不可达"、
   续五十五用它解释三处 fail-closed 的兜底位置、本轮用它解释 V-M/V-O 的 NaN 最终在哪被拦下。
   **这三个结论的强度，全都取决于它自己被钉住没有——而它至今没被变异测量过。**
2. **`evaluation/rank_ic.py`**（`compute_daily_rank_ic` / `summarize_rank_ic` /
   `summarize_rank_ic_by_year`）：**G1/G2/G3/G5 的原始数字都出自这里**，
   而且它正是本轮量到的"重复键三重守卫"里的**第一重**。
3. 其余同前：约 1,530 个测试函数在已审计的四条链之外从未做过变异测量。

## 勾选审计续五十七（2026-09-07 04:25）：**开链 5 = registry 那套"第二道更严的门"**——首轮 16 个变异 **4 杀 12 存活**（本 session 五条链里最差）；**12 条存活全部探针实测**，其中 **W-M 推翻了我的预判**（我以为是冗余守卫，实测二次晋级直接成功）；补 6 条测试后 **10 杀 6 存活**，剩下 6 条**逐条有探针证据**说明为什么不该钉。探针还牵出一个**比任何单个变异都更结构性的问题**：生命周期表在六个入口里只有一个被强制执行，而**其余五个自带的守卫中有四个至今没有任何测试**

### 一、为什么这条链值得单独一条链

本 session 已有三次结论是**靠读这个文件**得出的：续五十四判 U-E/U-F/U-O "晋级路径不可达"、
续五十五用 `validate_walk_forward_evidence` 的"逐一 PASS"解释 NOT_APPLICABLE 混不过去、
续五十六用它解释 V-M/V-O 的 NaN 最终在哪被拦下。
**读它 ≠ 测它**：这三个结论的强度取决于它自己有没有被钉住，而它此前从未被变异测量过。
本轮实测：**16 个变异只杀掉 4 个**——**比链 3 首轮的 6/16 还差**，
而那 147 行门代码我本来已经认为"很薄"了。

### 二、首轮：4 杀 12 存活

杀掉的 4 条：**W-C**（重算未通过仍拦住晋级）、
**W-D**（`evaluate_oos_gates(evidence, is_baseline=False)` 改成 `True`，**34 个测试失败，本轮爆炸半径最大的一条**）、
**W-K**（schema 版本钉在 1）、
**W-L**（证据哈希必须绑定**文件原始字节**——注意**钉它的不是 registry 单元测试，而是
`tests/unit/cli/test_registry_cli.py` 里那句 `payload["evidence_hash"] == hashlib.sha256(raw).hexdigest()`**）。

### 三、12 条存活的探针实测

**（1）真缺陷，已钉（6 条）**

| 变异 | 探针实测（原始 vs 变异） | 后果 |
|---|---|---|
| **W-M** DRAFT-only 守卫去掉 | 原：第二次晋级 `ValueError: only DRAFT models can enter WALK_FORWARD_PASSED`；变异：**NO ERROR，并追加一条 WALK_FORWARD_PASSED 事件** | **我预判错了**。我以为 `_transition` 会用 `_ALLOWED_TRANSITIONS` 兜住，**实测不会**——那个检查在**公开的** `transition()`（第 211 行），而 `promote_walk_forward` 调的是**私有 `_transition`**（第 192 行）。**即这道守卫是唯一的拦截**：SHADOW / PRODUCTION / **RETIRED** 的模型都能被拉回 WALK_FORWARD_PASSED |
| **W-J** 证据文件的 role 不再检查 | 原：`role=VALIDATION`、`role=LOCKED_TEST`、role 缺失**全部被拒**；变异：**三者都通过，六道门全 PASS** | **验证集证据可以把模型送进 WALK_FORWARD_PASSED**。评价层同一道墙（`role != "TEST"` → PermissionError）**是被钉住的**（链 4 的 V-A），**registry 这一份副本没人钉** |
| **W-G** `gate_evidence` 允许未知字段 | 原：`gate_evidence fields are invalid`；变异：**接受，且证据哈希由 `6b539dab…` 变成 `16a2a3d2…`** | 多一个字段的文件照样六道门全过，而**它的哈希与规范文件不同**——绑到其中一个的审批不会匹配另一个；同时意味着**将来新增的 schema 字段会被这一版静默忽略** |
| **W-N** 哈希不再绑定重算出的判定 | 同一份证据配两种判定（G6 PASS vs G6 FAIL）：原**哈希不同**；变异**哈希相同** | **审批不会随门代码变更而失效**：同一份证据在判定改变之后仍匹配旧审批 |
| **W-O** 哈希依赖 dict 插入顺序 | 同样内容、年份正序 vs 倒序：原**哈希相同**；变异**哈希不同** | 证据由一个进程写、另一个进程验，而 `json.dumps` 保留插入序：**一次无害的重排就会让所有已记录的审批哈希失效** |
| **W-P** `fullmatch` → `match` | 原：`sha256:<64 hex>` + 尾随 `\n`（长 72）**被拒**；变异：**被接受** | `$` 也匹配尾随换行前的位置。**model_id 来自文件与命令行，尾随换行是常态**，而多一个字符的内容寻址 id 不是同一个对象 |

**（2）不该钉，且每一条都有探针证据（6 条）**

| 变异 | 探针实测 | 定性 |
|---|---|---|
| **W-B** 只拦 FAIL、不再拦 NOT_APPLICABLE | `is_baseline=False` 下可达状态集 = **{FAIL, PASS}**；`is_baseline=True` 下才是 {NOT_APPLICABLE, PASS}；`delta=None` → **G4 FAIL** | **可证等价**：NOT_APPLICABLE 只在 `evaluate_oos_gates` 的 `is_baseline` 分支产生，而本调用点把 `is_baseline=False` **写死**。**故续五十五那句"registry 要求逐一 PASS，所以 N/A 混不过去"是对的，但今天它在空转**——N/A 根本不会出现。**它的等价性依赖 W-D 被钉住**，而 W-D 确实被 34 个测试盯着：**这次前提是有钉子的**（对照链 2 的 T-E，那次前提没有） |
| **W-A** 报告不是恰好 G1–G6 也接受 | 四种差异极大的证据下 `sorted(results) == _REQUIRED_GATES` **恒为 True** | **按构造不可达**：`evaluate_oos_gates` 只有一个返回点，恒定填充 G1–G6。它防的是 gates.py 将来的改动 |
| **W-E** 非有限数值被接受 | 原 `walk-forward evidence mean_rank_ic is invalid`；变异 `walk-forward evidence did not pass: G1`（`inf` 的 delta → `did not pass: G4`）。**bool 两侧都被拒** | **fail-closed，只换了一级**：从"哪个字段坏了"变成"哪道门没过"——结论一样，**线索更差** |
| **W-H** 年份不齐被接受 | 原 `must contain yearly values for 2020-2025`；变异 `did not pass: G3, G5` | 同上 |
| **W-I** `valid_test_day_count = 0` 被接受 | 原 `valid_test_day_count is invalid`；变异 `did not pass: G6`。**bool 两侧都被拒** | 同上 |
| **W-F** 年份键非十进制 | 原 `walk-forward evidence year is invalid`；变异 `invalid literal for int() with base 10: '20a'` | **只换文案**，同为 `ValueError` |

**W-E / W-H / W-I 这三条"fail-closed 换级"的结论，强度直接来自链 3**：
它们最终被拦下的位置是 `gates.py` 的 G1 / G3+G5 / G6，而**那三处判定在续五十五已被 16/16 钉住**。
**这是本 session 第一次出现"一条链的结论显式依赖另一条链的钉子"**——反过来说：
**若哪天有人放松 gates.py 的 isfinite 或 valid_years，这三条会同时从 fail-closed 变成 fail-open**，
而它们在 registry 这一层没有任何测试。**这不是本轮的洞，是本轮记下的一处耦合。**

### 四、补的 6 条测试与结果

`tests/unit/registry/test_model_registry.py` 从 **4 个测试变 10 个**：

1. `test_evidence_role_must_be_ordinary_test`（**W-J**；含"未改动的文件必须能加载且六道门全过"的夹具自证，三种越界 role 各测一次）
2. `test_evidence_file_carrying_unknown_fields_is_refused`（**W-G**）
3. `test_only_a_draft_can_enter_walk_forward_passed`（**W-M**；含"失败的尝试不得追加历史"与"RETIRED 不能被复活"）
4. `test_evidence_hash_binds_the_recomputed_verdict`（**W-N**；含"两个判定真的不同"的自证）
5. `test_evidence_hash_is_independent_of_yearly_insertion_order`（**W-O**；含"内容相等而插入序不同"的自证）
6. `test_model_id_must_match_exactly_not_only_at_the_front`（**W-P**；含 well-formed id 必须先通过的控制，四种越界 id 各测一次）

结果：

- **定向**（只跑 `test_model_registry.py`，**0.2s/次**）：control **10 passed** →
  **W-G / W-J / W-M / W-N / W-O / W-P 全部 killed，且各由对应的新测试杀掉** → restored **10 passed**；
  `model_registry.py` **还原后逐字节相同**。日志 `mutation_registry_gate_targeted_with_killers.txt`。
  **W-L 在这个文件里仍然存活**——它由 CLI 测试钉住。**这正是定向 harness 必须写清范围的理由**：
  范围外的钉子不能算作这个文件的钉子，反之亦然。
- **全量 harness**：**基线 183 passed / 3 skipped**，**10 杀 6 存活**（存活 W-A / W-B / W-E / W-F / W-H / W-I，
  即上表第（2）组），**无新增存活**，还原后逐字节相同并复跑通过。日志 `mutation_registry_gate_after_pins.txt`。
- **全量门槛**：`pytest tests -q` → **1643 passed / 45 skipped**（续五十六的 1637 + 本轮 6 条，skipped 未变）；
  `ruff check src tests` 清；`mypy src` **158 文件清**。
- **只改测试**：对 `research_v2/src`、`research_v2/scripts` 的 `git diff --stat` **为空**；
  未改任何门阈值、契约、schema 或 `dropna` 口径。
- **未写任何数据库**；**未碰** `materialization_status.{json,md}`；**未释放** `/tmp/frozen_chain_hold.txt`；
  **链 2 仍未重启**；**裁决 2/3/4 与决定 A–N 全部仍开放**。
- **未打开 `QRV2_INTEGRATION`**：harness 范围里那两条 integration 路径本轮**全程 skipped**，
  **因为它们可能触库，而本 session 的红线是零数据库写入**。故"六路径"里有两路径实际是空跑，**如实记**。

### 五、本轮最结构性的发现：**生命周期策略被写了两遍，而只有其中一遍被强制执行**

W-M 的探针暴露的不是一个守卫没测，而是一处设计事实：

- `_ALLOWED_TRANSITIONS`（第 47–58 行）是生命周期表的**声明式**版本；
- **但只在公开的 `transition()`（第 211 行）里被强制执行**；
- `_transition`（第 320–334 行）**不查这张表**，只校验 actor / reason / evidence_hashes；
- 而**六个入口里有五个绕过 `transition()` 直接调 `_transition`**：
  `promote_walk_forward`(192)、`enter_shadow`(250)、`_approve_locked_test`(263)、
  `_complete_locked_test`(278)、`_approve_production`(291)。

这五个入口**各自带一句手写的前置状态守卫**（"only X models can enter Y"），
所以策略实际上被写了两遍：**一张表 + 五句手写守卫**。**两者之间没有任何测试保证它们一致。**

**而这五句手写守卫里，本轮之前一句都没有被测过**，这一条是查过的，不是推断：
`git grep -c -E "only WALK_FORWARD_PASSED|only SHADOW|only LOCKED_TEST|only DRAFT" HEAD -- research_v2/tests`
**在本轮提交之前的整个测试树上零命中**；补测之后同一查询命中 2 处，**两处都是本轮新加的 "only DRAFT"**。
**即：W-M 不是五个里的例外，而是五个里的一个样本；本轮只钉住了其中一个。**

### 六、下一轮候选（按本轮结果排序，非承诺）

1. **钉住另外四句生命周期守卫**（`enter_shadow` / `_approve_locked_test` /
   `_complete_locked_test` / `_approve_production`），并**加一条一致性测试**：
   断言"五句手写守卫允许的状态迁移"与 `_ALLOWED_TRANSITIONS` 这张表**完全一致**——
   否则策略的两份写法会继续各自漂移。
   **其中 `enter_shadow` 最便宜**：它的状态守卫在审批校验**之前**执行，故传一个占位 approval 即可触发；
   另外三个要先造出有效的一次性 workflow proof，需要复用 `tests/unit/registry/test_locked_test.py` 的夹具。
2. **`evaluation/rank_ic.py`**：G1/G2/G3/G5 的原始数字都出自 `compute_daily_rank_ic` /
   `summarize_rank_ic` / `summarize_rank_ic_by_year`，而它也是续五十六量到的"重复键三重守卫"里的**第一重**。
   **链 3 钉了判定、链 4 钉了证据组装、链 5 钉了晋级把关，唯独最底层的算术还没人钉。**
3. 其余同前：约 1,530 个测试函数在已审计的五条链之外从未做过变异测量。
