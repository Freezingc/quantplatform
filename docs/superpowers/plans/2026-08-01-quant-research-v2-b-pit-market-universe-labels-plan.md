# Quant Research V2 B — PIT Market, Universe, and Labels Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 从只读 `dfs://YesLab` 构建点时正确、可审计的交易日历、证券主数据、清洗行情、可交易性、沪深流动 A 股股票池和“原始 5 日复权开盘收益”标签，并以 Tushare 仅做二级对账。

**Architecture:** Source adapter 只返回带 schema fingerprint 和 watermark 的批次；PIT 层将观察日、可用日、信号日显式分离。股票池只使用 T 收盘时已知信息，标签独立使用未来 T+1/T+6 复权开盘价并在 T+6 成熟，防止未来可交易状态反向污染 T 日股票池。所有产物先进入 staging，通过 DQ 后发布到 V2 表。

**Tech Stack:** Python 3.11、DolphinDB SDK、Pandas/NumPy、Pydantic、PyArrow、Tushare Pro（仅 reconciliation）、pytest/Hypothesis。

---

## 前置与非目标

- 前置：A 计划全部通过，V2Test 已有 `001`–`004` schema。
- 本计划的主行情、证券状态和因子基础源均来自只读 YesLab；Tushare 差异不能自动覆盖主源。
- 股票池不使用 T+1 是否停牌、是否涨跌停等未来信息；这些信息只进入 realized execution diagnostics。
- 普通回填命令只允许发布标签到 2025-12-31 及其成熟范围；2026 标签受 locked-test guard 保护。

## 文件清单

**Create:**

```text
research_v2/configs/sources/yeslab.yaml
research_v2/configs/sources/tushare_reconciliation.yaml
research_v2/configs/universe/cn_liquid_a.yaml
research_v2/configs/labels/forward_open_5d.yaml
research_v2/src/research_v2/contracts/source.py
research_v2/src/research_v2/contracts/market.py
research_v2/src/research_v2/sources/base.py
research_v2/src/research_v2/sources/yeslab.py
research_v2/src/research_v2/sources/tushare.py
research_v2/src/research_v2/pit/calendar.py
research_v2/src/research_v2/pit/security_master.py
research_v2/src/research_v2/pit/market.py
research_v2/src/research_v2/pit/reconciliation.py
research_v2/src/research_v2/universe/tradability.py
research_v2/src/research_v2/universe/rules.py
research_v2/src/research_v2/universe/builder.py
research_v2/src/research_v2/labels/forward_return.py
research_v2/src/research_v2/quality/engine.py
research_v2/src/research_v2/quality/market_checks.py
research_v2/src/research_v2/cli/materialize.py
research_v2/tests/unit/sources/test_yeslab_adapter.py
research_v2/tests/unit/pit/test_calendar.py
research_v2/tests/unit/pit/test_security_master.py
research_v2/tests/unit/pit/test_market.py
research_v2/tests/unit/pit/test_reconciliation.py
research_v2/tests/unit/universe/test_tradability.py
research_v2/tests/unit/universe/test_rules.py
research_v2/tests/unit/labels/test_forward_return.py
research_v2/tests/unit/quality/test_market_checks.py
research_v2/tests/unit/quality/test_engine.py
research_v2/tests/property/pit/test_market_invariants.py
research_v2/tests/property/universe/test_universe_invariants.py
research_v2/tests/leakage/test_future_market_perturbation.py
research_v2/tests/golden/data/market_50x60.parquet
research_v2/tests/golden/data/security_50x60.parquet
research_v2/tests/golden/data/expected_universe.parquet
research_v2/tests/golden/data/expected_labels.parquet
research_v2/tests/golden/test_market_universe_labels.py
research_v2/tests/integration/pit/test_market_publish.py
research_v2/tests/integration/pit/test_label_lock.py
```

## Task B1: 冻结源表契约与水位读取

**Files:**

- Create: `research_v2/configs/sources/yeslab.yaml`
- Create: `research_v2/src/research_v2/contracts/source.py`
- Create: `research_v2/src/research_v2/sources/base.py`
- Create: `research_v2/src/research_v2/sources/yeslab.py`
- Test: `research_v2/tests/unit/sources/test_yeslab_adapter.py`

- [ ] **B1.1 从 Phase 0 清单写源映射配置**

配置必须逐字段声明 logical field、YesLab database/table/column、DolphinDB type、单位、时区、业务键和允许的 null；禁止使用模糊表扫描或按列位置读取。至少覆盖交易日历、证券代码/上市退市、名称/ST、日 OHLCV/amount、复权因子和停牌状态。

- [ ] **B1.2 写失败的 schema fingerprint 测试**

```python
def test_adapter_fails_when_source_schema_drifts(fake_readonly_session):
    adapter = YesLabAdapter(fake_readonly_session, expected=EXPECTED_CONTRACT)
    fake_readonly_session.schema = SOURCE_SCHEMA_WITH_CHANGED_OPEN_TYPE
    with pytest.raises(SourceSchemaDrift, match="open"):
        adapter.fetch_market(DATE_RANGE)
```

- [ ] **B1.3 定义 SourceBatch 契约**

```python
@dataclass(frozen=True)
class SourceBatch:
    frame: pd.DataFrame
    source_name: str
    table_name: str
    watermark_id: str
    max_observation_date: date
    source_row_count: int
    schema_fingerprint: str
    extracted_at: datetime
```

`frame` 之外的字段必须随批次传递并最终进入审计列。

- [ ] **B1.4 实现只读、按日期/代码裁剪的 adapter**

查询必须：显式列名；`start_date <= trade_date <= end_date`；可选代码批；稳定排序 `trade_date, ts_code`；不生成 all-stock × all-date 笛卡尔积。一次取数上限由配置控制，超限按月份和 hash bucket 分批。

- [ ] **B1.5 写水位捕获与漂移停止逻辑**

每次开始物化前先记录 max date、row count 和 schema fingerprint；物化结束前重读水位，如果源在运行期间变化则将 run 标记 `SOURCE_CHANGED` 并禁止 publish。

- [ ] **B1.6 运行测试**

Run: `cd research_v2 && pytest tests/unit/sources/test_yeslab_adapter.py -q`

Expected: 正常映射、缺列、类型漂移、重复列、源运行中变化和大批次分页全部通过。

- [ ] **B1.7 提交**

```bash
git add research_v2/configs/sources/yeslab.yaml research_v2/src/research_v2/contracts/source.py research_v2/src/research_v2/sources research_v2/tests/unit/sources/test_yeslab_adapter.py
git commit -m "feat(v2): add audited readonly YesLab adapter"
```

## Task B2: 交易日历与点时证券主数据 SCD

**Files:**

- Create: `research_v2/src/research_v2/pit/calendar.py`
- Create: `research_v2/src/research_v2/pit/security_master.py`
- Test: `research_v2/tests/unit/pit/test_calendar.py`
- Test: `research_v2/tests/unit/pit/test_security_master.py`

- [ ] **B2.1 写交易日偏移失败测试**

```python
def test_signal_entry_exit_dates(calendar):
    assert calendar.shift(date(2025, 1, 3), 1) == date(2025, 1, 6)
    assert calendar.shift(date(2025, 1, 3), 6) == date(2025, 1, 13)
```

fixture 必须包含周末、法定休市和临时休市；越界返回明确异常，不按自然日猜测。

- [ ] **B2.2 实现 immutable TradingCalendar**

接口包含 `is_trading_day`、`shift`、`window`、`count`、`next_trading_day`；日历数据版本参与下游 `data_version`。同一 exchange/date 冲突阻止发布。

- [ ] **B2.3 写 SCD2 失败测试**

覆盖证券名称/ST/上市状态变化：连续相同属性合并区间；属性改变时关闭前一版本；`valid_from <= as_of < valid_to`；同日冲突进入隔离；不能用当前名称回填历史。

- [ ] **B2.4 实现 `security_master_scd` 构建器**

标准字段：`ts_code, exchange, board, list_date, delist_date, security_name, is_st, listing_status, valid_from, valid_to` 加审计字段。`board` 明确映射 `SH_MAIN, STAR, SZ_MAIN, CHINEXT, BEIJING, OTHER`；主股票池只接收前四类且 exchange 为 SSE/SZSE。

- [ ] **B2.5 运行测试**

Run: `cd research_v2 && pytest tests/unit/pit/test_calendar.py tests/unit/pit/test_security_master.py -q`

Expected: 所有闭开区间、退市、名称变更、北交所映射和日历偏移断言通过。

- [ ] **B2.6 提交**

```bash
git add research_v2/src/research_v2/pit/calendar.py research_v2/src/research_v2/pit/security_master.py research_v2/tests/unit/pit
git commit -m "feat(v2): build PIT calendar and security master"
```

## Task B3: 清洗行情与复权开盘价

**Files:**

- Create: `research_v2/src/research_v2/contracts/market.py`
- Create: `research_v2/src/research_v2/pit/market.py`
- Create: `research_v2/src/research_v2/quality/market_checks.py`
- Test: `research_v2/tests/unit/pit/test_market.py`
- Test: `research_v2/tests/unit/quality/test_market_checks.py`
- Test: `research_v2/tests/property/pit/test_market_invariants.py`

- [ ] **B3.1 写失败测试：价格清洗和复权公式**

```python
def test_adjusted_open_uses_raw_open_times_adjustment_factor():
    row = clean_market(raw_open=10.0, adj_factor=1.25)
    assert row.adjusted_open == pytest.approx(12.5)
```

覆盖 `open/high/low/close > 0`、`high >= max(open, close, low)`、`low <= min(open, close, high)`、volume/amount 非负、停牌零成交、重复键和无穷值。

- [ ] **B3.2 实现确定性行级清洗**

业务键 `trade_date, ts_code`：完全一致行折叠；一条完整且其余仅缺失时保留完整行并设置 `dedup_resolution='SINGLE_NONNULL'`；多个冲突非空值全部进入 DQ/隔离，正式 clean 表不保留任一条。

- [ ] **B3.3 计算复权字段**

`adjusted_open = raw_open * adj_factor`。标签只使用同一标的两日 adjusted open 的比值，因此 adj factor 的绝对归一尺度不重要，但必须为正且在除权日前后产生连续价格。另存 `adj_factor_ratio_1d` 用于异常检测，不修改源值。

- [ ] **B3.4 计算历史窗口字段**

按 `ts_code, trade_date` 排序，只使用截至 T：

```text
valid_trade_day = raw_open > 0 and volume > 0 and amount > 0 and not suspended
valid_days_20 = rolling_sum(valid_trade_day, 20, min_periods=20)
adv20 = rolling_mean(amount where valid_trade_day, 20, min_periods=15)
listing_age_trading_days = calendar count from list_date through T
```

禁止跨证券 rolling、centered window 或未来 backfill。

- [ ] **B3.5 建立 blocking DQ**

BLOCKER：业务键重复、负/零有效价格、OHLC 关系错误、非有限数、`adj_factor <= 0`、source watermark 不一致、schema 漂移。WARN：单日复权价格跳变绝对收益 >50%、当日主板覆盖相对前 20 日中位数下降 >5%、amount/volume 单位疑似变化。

- [ ] **B3.6 性质测试**

Hypothesis 生成多证券乱序行情，验证：排序前后结果相同；改变 T+1 以后数据不改变 T 行历史字段；乘所有 adj factors 同一正数不改变 forward return；chunked 与 full-batch 输出 hash 相同。

- [ ] **B3.7 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/pit/test_market.py tests/unit/quality/test_market_checks.py tests/property/pit/test_market_invariants.py -q`

Expected: 全部通过。

```bash
git add research_v2/src/research_v2/contracts/market.py research_v2/src/research_v2/pit/market.py research_v2/src/research_v2/quality/market_checks.py research_v2/tests/unit/pit/test_market.py research_v2/tests/unit/quality/test_market_checks.py research_v2/tests/property/pit/test_market_invariants.py
git commit -m "feat(v2): materialize clean point-in-time market data"
```

## Task B4: 可交易性与沪深流动 A 股股票池

**Files:**

- Create: `research_v2/configs/universe/cn_liquid_a.yaml`
- Create: `research_v2/src/research_v2/universe/tradability.py`
- Create: `research_v2/src/research_v2/universe/rules.py`
- Create: `research_v2/src/research_v2/universe/builder.py`
- Test: `research_v2/tests/unit/universe/test_tradability.py`
- Test: `research_v2/tests/unit/universe/test_rules.py`
- Test: `research_v2/tests/property/universe/test_universe_invariants.py`

- [ ] **B4.1 固化版本化股票池配置**

```yaml
universe_id: cn_liquid_a
version: 1
allowed_boards: [SH_MAIN, STAR, SZ_MAIN, CHINEXT]
exclude_st: true
exclude_suspended_on_signal_date: true
min_listing_trading_days: 252
lookback_trading_days: 20
min_valid_days_in_lookback: 15
adv20_bottom_quantile_excluded: 0.20
quantile_tie_break: [adv20_desc, ts_code_asc]
```

- [ ] **B4.2 写逐条规则失败测试**

构造边界：上市第 251/252 日、20 日中 14/15 日有效、ST 生效日前后、北交所、退市日、T 日停牌、ADV20 分位并列。每条排除必须产生稳定 `exclusion_reason`，优先级固定：`OUTSIDE_BOARD → NOT_LISTED → ST → SUSPENDED → YOUNG_LISTING → INSUFFICIENT_HISTORY → LOW_LIQUIDITY`。

- [ ] **B4.3 实现 signal-date tradability**

`tradability_daily` 只使用 T 日收盘可知状态，输出每项 boolean 与 reason bitmap。T+1 实际无法成交另存 execution diagnostic，不回写 T 日 `is_eligible`。

- [ ] **B4.4 实现 ADV20 横截面裁剪**

先应用所有非流动性规则，再在每日剩余集合内按 `adv20 DESC, ts_code ASC` 稳定排序；保留 `ceil(0.8 * N)` 只股票。不要用 pandas quantile 值直接处理并列，以免不同版本产生不同成员。

- [ ] **B4.5 实现版本化 Universe Builder**

输出 `signal_date, universe_id, universe_version, ts_code, is_member, exclusion_reason, adv20_rank, membership_rank` 与审计字段；这里 `signal_date` 就是用于规则计算的 T 日，并等值连接 `market_daily_clean.trade_date`。默认只发布 `is_member=true` 事实行；DQ 明细保留全体候选的 reason。

- [ ] **B4.6 性质测试**

验证代码顺序不影响成员；加入未来行情不改变过去成员；横截面保持数恰为 `ceil(0.8*N)`；任一成员均满足前置规则；同输入重跑 record hash 相同。

- [ ] **B4.7 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/universe tests/property/universe -q`

Expected: 全部通过。

```bash
git add research_v2/configs/universe/cn_liquid_a.yaml research_v2/src/research_v2/universe research_v2/tests/unit/universe research_v2/tests/property/universe
git commit -m "feat(v2): build versioned liquid A-share universe"
```

## Task B5: 原始 5 日复权开盘收益标签

**Files:**

- Create: `research_v2/configs/labels/forward_open_5d.yaml`
- Create: `research_v2/src/research_v2/labels/forward_return.py`
- Test: `research_v2/tests/unit/labels/test_forward_return.py`
- Test: `research_v2/tests/leakage/test_future_market_perturbation.py`

- [ ] **B5.1 固化标签定义**

```yaml
label_id: raw_forward_open_return_5d
version: 1
signal_time: close
entry_offset_trading_days: 1
exit_offset_trading_days: 6
price_field: adjusted_open
primary_value: raw_return
locked_year: 2026
```

- [ ] **B5.2 写人工算例失败测试**

```python
def test_five_day_open_to_open_return(calendar, prices):
    # T=2025-01-03, entry=T+1, exit=T+6
    label = build_label(signal_date=date(2025, 1, 3), ts_code="000001.SZ")
    assert label.entry_date == date(2025, 1, 6)
    assert label.exit_date == date(2025, 1, 13)
    assert label.raw_return == pytest.approx(prices["2025-01-13"] / prices["2025-01-06"] - 1.0)
```

- [ ] **B5.3 实现主标签和辅助变换**

主值：

```python
raw_return = adjusted_open_exit / adjusted_open_entry - 1.0
```

每日横截面、仅在同日有效标签内派生 `rank_pct`、MAD winsorized return、z-score；行业/市值中性 residual 只有相应 PIT 暴露可用时才计算，并明确为辅助标签。任何辅助变换都不能替代主 RankIC 的 raw return。

- [ ] **B5.4 定义缺失与可执行状态**

entry/exit adjusted open 任一缺失、非正或被隔离时，`label_valid=false`、主值 null，并记录 `invalid_reason`。这不会反向删除 T 日 universe 成员。另记 `entry_realized_tradable`/`exit_price_observed` 供覆盖分析。

- [ ] **B5.5 强制 label maturity 与 locked guard**

`label_available_date = exit_date`；数据集只能使用 `label_available_date <= fold_cutoff`。普通 materialize 的 `signal_date.year >= 2026` 一律拒绝发布，即使源里已存在；只有 F 计划一次性 locked workflow 的专用 capability 可解锁。

- [ ] **B5.6 写泄漏陷阱**

固定 T 及之前输入，扰动 T+1 之后价格：允许标签变化，不允许 T 日 market features/universe 变化。固定 fold cutoff，扰动 cutoff 之后价格：训练/验证 manifest hash 不变。

- [ ] **B5.7 运行测试并提交**

Run: `cd research_v2 && pytest tests/unit/labels/test_forward_return.py tests/leakage/test_future_market_perturbation.py -q`

Expected: T+1/T+6、节假日、缺价、退市、锁箱和未来扰动断言通过。

```bash
git add research_v2/configs/labels/forward_open_5d.yaml research_v2/src/research_v2/labels research_v2/tests/unit/labels research_v2/tests/leakage/test_future_market_perturbation.py
git commit -m "feat(v2): add locked five-day adjusted-open labels"
```

## Task B6: Tushare 二级对账

**Files:**

- Create: `research_v2/configs/sources/tushare_reconciliation.yaml`
- Create: `research_v2/src/research_v2/sources/tushare.py`
- Create: `research_v2/src/research_v2/pit/reconciliation.py`
- Create: `research_v2/tests/unit/pit/test_reconciliation.py`

- [ ] **B6.1 写 adapter contract 和 token 安全测试**

Tushare token 只从 `QRV2_TUSHARE_TOKEN` 获取，任何日志和 exception 均脱敏。adapter 实现速率限制、有限重试、本地只读缓存 key=`endpoint+params+response_date`；缓存保存原始响应 hash 与拉取时间。

- [ ] **B6.2 固化抽样方案**

每天对账当日所有 universe 成员的证券状态和 OHLCV；历史回填按每月首末交易日 + 固定 hash 抽样 5% 股票。固定 hash 以 `sha256(trade_date|ts_code) % 100 < 5`，保证重跑样本一致。

- [ ] **B6.3 固化字段容差**

```text
open/high/low/close: abs_diff <= 0.01 或 rel_diff <= 5e-4
volume/amount: rel_diff <= 1e-3（先统一单位）
adj_factor: 比较 1 日 ratio，abs_diff <= 1e-6，不比较绝对尺度
list/delist/ST status: 必须完全一致
```

差异写 `source_reconciliation`；价格/状态差异只产生 DQ 和人工调查，不覆盖 `market_daily_clean`。

- [ ] **B6.4 运行测试**

Run: `cd research_v2 && pytest tests/unit/pit/test_reconciliation.py -q`

Expected: 单位换算、容差边界、固定抽样、限速重试和不覆盖主源测试通过。

- [ ] **B6.5 提交**

```bash
git add research_v2/configs/sources/tushare_reconciliation.yaml research_v2/src/research_v2/sources/tushare.py research_v2/src/research_v2/pit/reconciliation.py research_v2/tests/unit/pit/test_reconciliation.py
git commit -m "feat(v2): reconcile market source with Tushare"
```

## Task B7: DQ 引擎、物化 CLI 与增量/回填

**Files:**

- Create: `research_v2/src/research_v2/quality/engine.py`
- Create: `research_v2/src/research_v2/cli/materialize.py`
- Modify: `research_v2/src/research_v2/cli/app.py`
- Test: `research_v2/tests/unit/quality/test_engine.py`
- Test: `research_v2/tests/integration/pit/test_market_publish.py`
- Test: `research_v2/tests/integration/pit/test_label_lock.py`

- [ ] **B7.1 定义 CLI**

```text
qr-v2 materialize security-master --start YYYY-MM-DD --end YYYY-MM-DD
qr-v2 materialize market --start YYYY-MM-DD --end YYYY-MM-DD
qr-v2 materialize universe --start YYYY-MM-DD --end YYYY-MM-DD --universe cn_liquid_a:v1
qr-v2 materialize labels --start YYYY-MM-DD --end YYYY-MM-DD --label raw_forward_open_return_5d:v1
qr-v2 reconcile market --start YYYY-MM-DD --end YYYY-MM-DD
qr-v2 dq run --scope market_daily_clean --data-version sha256:DATA_VERSION
```

所有命令支持 `--dry-run`，输出读取表、日期、估算行数、目标表和 watermark；生产写入需要 A 计划统一确认机制。

- [ ] **B7.2 写 DQ 决策测试**

严重度：`INFO/WARN/ERROR/BLOCKER`；publish 仅当 BLOCKER=0 且指定 blocking checks 全部 PASS。空检查结果不是 PASS；检查执行异常视为 BLOCKER。

- [ ] **B7.3 实现依赖顺序与受影响分区重算**

源水位新增日期时只处理新增/变化分区；market 改变日期 D 时，滚动字段重算 D 至后 19 个交易日，标签重算前 6 至当日相关 signal dates，universe 重算 D 至后 19 日。每个重算集合写审计，不全库盲扫。

- [ ] **B7.4 V2Test 集成测试**

写入 50×60 source fixture，执行 market → universe → labels → 重跑：首次发布有行、第二次相同 data version 幂等；改变一条源行只影响计算图定义的日期；在 stage 注入重复键时正式表仍保持旧版本。

- [ ] **B7.5 locked year 集成测试**

普通身份请求 2026 labels 必须在查询/计算前失败；2025-12 信号若 exit 落在 2026，按照 `signal_date` 仍属于 2025 历史标签但必须等 exit 成熟，并清晰记录跨年。专用 unlock capability 在本计划中不存在。

- [ ] **B7.6 运行验证并提交**

Run: `cd research_v2 && pytest tests/unit/quality -q`

Run: `cd research_v2 && QRV2_INTEGRATION=1 pytest tests/integration/pit -q -m integration`

Expected: DQ fail-closed、增量范围、发布原子性和 locked guard 全部通过。

```bash
git add research_v2/src/research_v2/quality research_v2/src/research_v2/cli research_v2/tests/unit/quality research_v2/tests/integration/pit
git commit -m "feat(v2): orchestrate audited PIT materialization"
```

## Task B8: 50×60 黄金样本与人工核验

**Files:**

- Create: `research_v2/tests/golden/data/market_50x60.parquet`
- Create: `research_v2/tests/golden/data/security_50x60.parquet`
- Create: `research_v2/tests/golden/data/expected_universe.parquet`
- Create: `research_v2/tests/golden/data/expected_labels.parquet`
- Create: `research_v2/tests/golden/data/README.md`
- Create: `research_v2/tests/golden/test_market_universe_labels.py`

- [ ] **B8.1 选取覆盖边界的固定样本**

从历史只读源抽取 50 只股票、连续 60 个交易日，必须包含主板/科创板/创业板/北交所、ST 变更、停牌、上市临界、流动性并列、除权和缺失开盘。parquet 中删除无关列并记录 source watermark/hash，不包含凭据。

- [ ] **B8.2 独立生成预期值**

预期 universe 和 labels 用简短审计脚本/人工 spreadsheet 独立计算，不调用生产 builder；在 README 写出 10 个具体行级算例及公式，至少人工复核 T+1/T+6 的 5 个信号日。

- [ ] **B8.3 写 byte/logical equality 测试**

数值使用明确 tolerance；成员、reason、entry/exit date 必须精确相等；排序后对 canonical rows 比较 SHA-256。

- [ ] **B8.4 运行完整 B 计划验证**

Run: `cd research_v2 && pytest tests/unit/sources tests/unit/pit tests/unit/universe tests/unit/labels tests/unit/quality tests/property/pit tests/property/universe tests/leakage/test_future_market_perturbation.py tests/golden/test_market_universe_labels.py -q`

Expected: 全部退出码 0。

- [ ] **B8.5 提交并请求审查**

```bash
git add research_v2/tests/golden
git commit -m "test(v2): add golden PIT market and label dataset"
```

随后使用 `superpowers:requesting-code-review`，重点审查信号/入场/退出日期、股票池是否偷看未来、复权单位、跨年成熟标签和 2026 锁。

## B 计划完成标准

- [ ] 2013 warmup 至 2025 成熟范围可生成稳定交易日历、主数据、clean market、tradability、universe 和 labels。
- [ ] 任意未来数据扰动不改变对应过去的行情历史字段或股票池。
- [ ] 标签严格为 T+1 到 T+6 复权开盘收益，主指标保留 raw return。
- [ ] 2026 普通标签读取/发布 fail closed；2026 SHADOW 所需无标签输入不受影响。
- [ ] Tushare 只产生 reconciliation 记录，从不静默覆盖主源。
- [ ] 全部事实行带 run/data version/watermark/hash，重复冲突不会进入正式表。
