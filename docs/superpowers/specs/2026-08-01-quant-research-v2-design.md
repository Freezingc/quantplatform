# Quant Research V2 完整重构设计

- 状态：已完成分章节评审，等待书面文档复核
- 日期：2026-08-01
- 目标：优化沪深全 A 流动性池上的样本外 RankIC/ICIR
- 架构路线：并行 V2 混合架构
- 新数据库：dfs://QuantResearchV2
- 现有数据库：dfs://YesLab，只读且不修改任何表结构

## 1. 背景与目标

现有研究链路存在固定全历史股票全集、幽灵股票、因子重复键、因子高度冗余、训练验证口径不统一、回测样本过短、价格语义不一致和实验不可复现等问题。V2 不在旧链路中继续堆叠修补，而是建立独立的、可并行验证的新研究系统。

V2 的第一目标是可信的样本外预测能力，不强制使用深度学习。如果简单基线优于复杂模型，系统应选择简单基线。

主研究口径：

- 股票池：沪深全 A 流动性池。
- 排除：北交所、ST、停牌、上市不足 252 个交易日。
- 流动性：按当日 ADV20 剔除最低 20%。
- 信号时点：T 日收盘后。
- 进入时点：T+1 开盘。
- 退出时点：T+6 开盘。
- 主标签：原始 5 日复权收益。
- 主指标：样本外 RankIC、ICIR 和 HAC 显著性。
- Walk-forward：5 年 Train、1 年 Validation、1 年 Test，按年滚动。
- 切分边界：purge 6 个交易日。
- 最终锁定测试：2026 全年。

双重成功门槛：

- 汇总样本外平均 RankIC 不低于 0.03。
- Newey-West/HAC 修正 RankIC t-stat 不低于 2.0。
- 至少 70% 的测试年份 RankIC 为正。
- 候选模型相对当期最强简单基线的跨 Fold 平均 RankIC 增量不低于 0.005。

## 2. 架构边界

### 2.1 数据库边界

dfs://YesLab 永久作为只读源。V2 读取但不修改：

- stock_factor_daily
- stock_daily_data
- stock_info_table
- stock_used_name
- stock_trading_suspension_info
- trade_calendar
- index_component_table
- code_index
- sw_ind_code
- financial_metrics_table 及其他财务表
- factor_metadata
- factor_data1 至 factor_data11

全部新增数据对象写入 dfs://QuantResearchV2。测试使用独立的 dfs://QuantResearchV2Test。

Tushare 作为数据对账和状态补充源，不直接静默覆盖 DolphinDB 原始记录。差异写入对账表。

### 2.2 代码边界

新增独立 Python 包 research_v2：

    research_v2/
      adapters/
      contracts/
      materializers/
      universe/
      labels/
      factors/
      datasets/
      baselines/
      models/
      validation/
      evaluation/
      registry/
      inference/
      cli/

不继续向旧的 模型/exp_stock.py、data_stock.py 和旧回测类增加 V2 逻辑。

数据库保存结构化数据、版本、状态和指标。训练张量、模型 checkpoint、图表和 HTML 报告保存在不可变制品目录，DolphinDB 记录路径、哈希和版本。

## 3. 新增 DolphinDB 数据模型

### 3.1 治理和任务记录

| 表 | 作用 | 业务主键 |
| --- | --- | --- |
| schema_migration | V2 schema 版本和脚本哈希 | schema_version |
| pipeline_run | 物化、训练、评估和推理任务状态 | run_id |
| source_watermark | 源表最大日期、行数和摘要哈希 | source_name, table_name, observed_at |
| data_quality_result | 数据质量检查结果 | run_id, object_name, check_name |
| source_reconciliation | DolphinDB 与 Tushare 的字段差异 | trade_date, ts_code, field_name, check_version |

任务先进入 RUNNING。只有全部硬性质量门槛通过后才可进入 SUCCESS 并发布版本。

### 3.2 PIT 市场和股票池

| 表 | 作用 | 业务主键 |
| --- | --- | --- |
| security_master_scd | 上市、退市、名称和交易所慢变维度 | ts_code, effective_from, version |
| market_daily_clean | 统一行情、复权因子、成交额和市值 | trade_date, ts_code, data_version |
| tradability_daily | ST、停牌、上市天数、ADV20 和可交易性 | trade_date, ts_code, rule_version |
| universe_definition | 股票池规则和版本 | universe_id, universe_version |
| universe_daily | 每日股票池成员和排除原因 | trade_date, ts_code, universe_version |

universe_daily 保存 eligible_signal 和排除原因位图：

    EXCHANGE
    ST
    SUSPENDED
    NEW_LISTING
    DELISTED
    NO_MARKET_DATA
    LOW_LIQUIDITY
    DATA_QUALITY

### 3.3 因子和标签

| 表 | 作用 | 业务主键 |
| --- | --- | --- |
| factor_registry | 因子公式、代码、来源、版本和 PIT 状态 | factor_id, factor_version |
| factor_observation_clean | 确定性去重后的因子观测 | trade_date, ts_code, factor_id, factor_version |
| factor_quality_snapshot | 覆盖率、异常率、稳定性、重复和陈旧度 | as_of_date, factor_id, factor_version |
| factor_similarity | Fold 内因子截面相关 | fold_id, factor_id_a, factor_id_b |
| factor_set | 可复现因子集 | factor_set_id |
| factor_set_member | 因子成员、方向和入选原因 | factor_set_id, factor_id, factor_version |
| label_daily | 1/3/5/10/20 日标准标签 | signal_date, ts_code, horizon, label_version |

数据库只保存原始清洗值、缺失状态和陈旧度，不提前进行全样本标准化。Fold-aware 变换由 Dataset Builder 完成。

### 3.4 数据集、实验和模型

| 表 | 作用 | 业务主键 |
| --- | --- | --- |
| split_definition | Walk-forward 和 purge 区间 | split_id, fold_id |
| dataset_manifest | 数据集版本、因子集、股票池、标签和源水位 | dataset_id |
| experiment_run | 模型、参数、种子、代码版本和制品 | experiment_id |
| metric_daily | 每日 IC、RankIC、覆盖率和样本数 | experiment_id, fold_id, trade_date |
| metric_summary | Fold 与汇总指标 | experiment_id, fold_id, metric_scope |
| prediction_daily | 冻结模型逐日预测 | model_version, signal_date, ts_code |
| model_registry | 模型状态、版本和晋级记录 | model_id, model_version |

模型状态：

    DRAFT
    WALK_FORWARD_PASSED
    SHADOW
    LOCKED_TEST_APPROVED
    LOCKED_TEST_PASSED
    LOCKED_TEST_FAILED
    PRODUCTION
    RETIRED

正常晋级路径为 DRAFT -> WALK_FORWARD_PASSED -> SHADOW -> LOCKED_TEST_APPROVED -> LOCKED_TEST_PASSED -> PRODUCTION。锁定测试失败的版本进入 LOCKED_TEST_FAILED，不能转入生产；它只能保留审计记录或进入 RETIRED。不得绕过状态进入模拟盘或生产。

### 3.5 分区、版本和发布

大表使用 TSDB 组合分区：

- 第一层按 trade_date 或 signal_date 月度分区。
- 第二层按 ts_code 或 factor_id HASH 分区。
- 行情、股票池和标签按 trade_date, ts_code 排序。
- 因子观测按 factor_id, trade_date, ts_code 排序。
- 指标和预测按 experiment/model_version, trade_date, ts_code 排序。

所有新表包含：

    run_id
    data_version
    created_at
    source_watermark_id
    record_hash

禁止无版本覆盖。重复业务主键使 pipeline 失败，不允许无确定规则的 first/last 静默去重。

发布采用两阶段方式：

1. 写入新 data_version 并运行 DQ。
2. DQ 通过后将版本标记为 PUBLISHED。

消费者只读取 PUBLISHED 版本。失败版本保持 UNPUBLISHED，旧版本继续可用。

## 4. PIT 数据流

### 4.1 时间语义

每条特征记录：

    observation_date
    available_date
    signal_date

硬性不变量：

    available_date <= signal_date

违反时标记 LOOKAHEAD_VIOLATION 并拒绝进入数据集。

### 4.2 市场清洗

market_daily_clean 规则：

- trade_date, ts_code 唯一。
- OHLC 满足基本价格关系。
- 价格和复权因子为正。
- 成交量、成交额单位固定并记录。
- DolphinDB 为主源，Tushare 默认只对账。
- 冲突进入 source_reconciliation。
- 不生成股票与日期笛卡尔积。
- 某天无真实记录即不存在该日市场观测。

### 4.3 主股票池

每日基础候选：

    exchange in {SSE, SZSE}
    AND 已上市
    AND 未退市
    AND 当日有有效市场数据

依次应用：

1. 上市满 252 个交易日。
2. T 日不是 ST、*ST 或退市整理。
3. T 日不是全天停牌。
4. 最近 20 个交易日至少 15 个有效成交日。
5. 计算截至 T 日 ADV20。
6. 剔除当日 ADV20 最低 20%。
7. 数据质量通过。

股票池只能使用 T 日及之前信息。T+1 实际是否可买和 T+6 是否可卖只记录为 realized execution 字段，不反向改变 T 日股票池。

### 4.4 上市、退市、ST 和停牌

- 上市退市来自 stock_info_table。
- ST 状态按 stock_used_name 历史有效期构建。
- ann_date 与 start_date 冲突时采用较晚公开日期并记录冲突。
- 停牌来自 stock_trading_suspension_info。
- 不把股票放入上市前截面。
- 不因后来退市而删除退市前历史。
- 每日股票池数量应自然变化，固定常数触发异常。

### 4.5 因子确定性去重

相同主键的记录：

- 值完全相同：折叠一条并记录重复数。
- 只有一个非空值：保留非空值并标记 SOURCE_DUPLICATE_WITH_NULL。
- 多个非空值不相等：标记 CONFLICTING_DUPLICATE 并隔离业务主键。

因子冲突比例超阈值时隔离整个因子版本。行情和股票池核心主键冲突时终止整个版本发布。

### 4.6 因子 PIT 分类

日频技术因子：

- 只读取不晚于 T 的数据。
- 窗口严格向后。
- 当日缺失不跨日填充。
- 缺失填充值必须配套 missing_mask。

基本面因子：

- 按公告日期生效，不按 end_date 提前生效。
- 因无公告时刻，采用公告日后第一个交易日才可用的保守规则。
- 按当时已公告的最新版本构建修订历史。
- 可向前保持但最长 180 个自然日。
- 输出 stale_days。

行业和慢变分类：

- 使用 SCD 有效期。
- 不用当前行业覆盖整个历史。
- 历史有效期无法确定时只作诊断，不进入主模型。

### 4.7 Fold 内预处理

数据库保存原始清洗值。进入每个 Fold 后：

1. 只使用 Train 决定因子准入和长期统计量。
2. 每日截面 Median/MAD 去极值。
3. MAD 为 0 时用 1%/99% 分位截断。
4. 变换为当日截面 Rank 并映射到 [-1, 1]。
5. 缺失填 0，同时输出 missing_mask。
6. 基本面因子输出 stale_days。
7. 禁止未来填充和双向填充。
8. Validation/Test 复用 Train 因子清单和模型参数。

### 4.8 因子质量准入

因子参加某 Fold 前必须满足：

- Train 覆盖率至少 80%。
- 任一年度覆盖率至少 60%。
- 冲突重复已解决且不静默选择。
- Inf 比例为 0。
- 每日有效股票中位数达到股票池 70%。
- 非常数或近常数。
- PIT 检查通过。
- 公式和代码哈希存在。
- 至少覆盖完整训练窗口的 80%。

不合格因子保留历史但拒绝加入当前 Fold。

## 5. 标签

### 5.1 日期映射

通过 trade_calendar 映射：

    signal_date = T
    entry_date = 市场交易日 T+1
    exit_date = 市场交易日 T+6

不使用股票下一次出现行情的日期，避免停牌导致持有周期变化。

### 5.2 复权收益

    entry_adjusted_open = entry_open_bfq * entry_adj_factor
    exit_adjusted_open = exit_open_bfq * exit_adj_factor
    return_5d = exit_adjusted_open / entry_adjusted_open - 1

保存：

- return_raw：主经济标签。
- return_winsorized：当日 1%/99% 截断。
- return_rank：当日合格股票池百分位。
- return_zscore：回归损失标签。
- 行业、市值中性残差：辅助诊断。

主验收使用预测分数与 return_raw 的 Spearman RankIC。

### 5.3 标签有效性

以下情况标签无效：

- T 日不在主股票池。
- T+1/T+6 不是可映射市场日。
- 进入或退出价格缺失或非正。
- 复权因子缺失。
- 源数据 DQ 失败。

无效标签不填 0。每日报告 eligible_count、valid_label_count 和 label_coverage。覆盖率低于 90% 告警，低于 80% 时该日不进入主验收。

## 6. 防泄漏版本链与锁定测试

每个数据集必须追溯：

    source_watermark
    -> market_data_version
    -> universe_version
    -> factor_versions
    -> label_version
    -> factor_set_id
    -> split_id
    -> dataset_id

任一上游变化生成新 dataset_id。

普通 Dataset Builder 拒绝读取 2026-01-01 之后的数据。2026 正式解锁要求：

- 模型已通过历史 Walk-forward。
- factor_set、结构、参数、集成权重和 checkpoint 已冻结。
- code commit 和制品哈希已记录。
- 2026 最后一个 T+6 标签成熟。
- 创建专用 LOCKED_TEST_RUN 和审计记录。

测试只执行一次。失败后新模型不能继续声称 2026 是纯样本外。

锁定规则只禁止带标签的数据集构建、因子筛选、调参和回测访问 2026。每日 inference 可以读取截至当日已经公开的 2026 特征，生成无标签 SHADOW 预测；这些预测不可回流到训练、因子选择或模型选择。标签成熟后的监控由独立评估任务追加，且在正式全年解锁前不向研究调参流程暴露 2026 汇总结果。

## 7. Walk-forward 和因子筛选

### 7.1 历史区间

- 源数据预热：2013-01-01 起。
- 正式样本：2014-01-01 起。
- 历史开发：截至 2025-12-31。
- 最终锁定测试：2026 全年。

| Fold | Train | Validation | Test |
| --- | --- | --- | --- |
| F01 | 2014-2018 | 2019 | 2020 |
| F02 | 2015-2019 | 2020 | 2021 |
| F03 | 2016-2020 | 2021 | 2022 |
| F04 | 2017-2021 | 2022 | 2023 |
| F05 | 2018-2022 | 2023 | 2024 |
| F06 | 2019-2023 | 2024 | 2025 |
| Final | 2020-2024 | 2025 | 2026 locked |

每个边界 purge 6 个交易日。

### 7.2 Train 内因子筛选

阶段 A：应用 DQ、覆盖率和 PIT 过滤。

阶段 B：在 5 年 Train 内计算每日和年度 RankIC、HAC t-stat、正值比例、方向一致率、覆盖率和换手代理。方向仅由 Train 决定。

准入：

- 至少 3/5 年方向一致。
- 不由单一年份贡献。
- Benjamini-Hochberg FDR q 不高于 0.10。
- 内层时间子 Fold 至少 60% 正向贡献。

阶段 C：只在 Train 计算每日截面 Spearman，日期中位数绝对值不低于 0.90 的因子归入同一冗余簇。每簇按 PIT、覆盖率、稳定性、HAC、缺失率、换手和成本选择一个代表。

阶段 D：用 Train 内部时间子 Fold 的 Ridge、ElasticNet、LightGBM、置换重要性和稳定性选择评估增量。

目标因子数 15 至 40，软上限 40，硬上限 60。不为凑数加入冗余因子。

## 8. Dataset 与基线

截面数据：

    X_t [N_t, F]
    M_t [N_t, F]
    C_t [N_t, C]
    Y_t [N_t]
    IDs [N_t]

时序数据：

    X_t [N_t, L, F]
    M_t [N_t, L, F]

N_t 每日自然变化。L 首版只比较 20 和 60。

控制变量包括 log_total_mv、adv20_percentile、20 日波动率、上市交易日数、行业编码、缺失率和陈旧度。

每个 Fold 必跑：

- B0：Train 选择方向的最强单因子。
- B1：Train 方向校准等权组合。
- B2：Ridge/ElasticNet。
- B3：按交易日分组的 LightGBM Ranker。

Validation 最优者成为该 Fold 的 best_simple_baseline。旧 STGATim 只作为参考复现。

## 9. 候选模型与训练

模型按增量顺序：

1. M1 Residual MLP。
2. M2 GRU/TCN Temporal Encoder。
3. M3 Cross-sectional Set Attention。
4. M4 Economic Graph，仅在 M3 已显著超过基线时启用。

经济图只允许使用截至 T 的行业、过去收益相关、历史指数共同成分和明确经济关系。图中只含 eligible_signal 股票。禁止用隐藏状态临时 Top-K 作为唯一图边。

神经模型默认损失：

    0.60 * IC Loss
    + 0.30 * Pairwise Ranking Loss
    + 0.10 * Huber Loss

其中 IC Loss 为 1 减预测分数与 return_rank 的 Pearson 相关。损失权重只在少量离散候选中由 Validation 选择。

首版不以换手损失为主。如果以后启用，必须按 ts_code 对齐、使用可微 soft-rank 或组合权重，且权重不超过总损失 5%。

训练规范：

- AdamW、梯度裁剪、混合精度。
- warmup 加 cosine decay。
- 最多 100 epoch。
- Validation RankIC 早停，patience 10。
- 初筛至少 3 个种子，最终候选 5 个种子。
- 每模型家族最多 30 个 Trial。
- 一个 batch 的基本单位是完整交易日截面。
- 日期按月度块打乱，不随机裁剪股票改变 RankIC 口径。

集成最多 3 个模型，只用 Validation 决定非负且和为 1 的权重，单模型上限 70%。无稳定增量则保留单模型。

## 10. 评价与双重门槛

每日保存：

- Pearson IC
- RankIC
- eligible_count
- valid_label_count
- coverage
- prediction_std
- prediction_unique_count

Fold 汇总：

- RankIC mean、median、std。
- 原始 ICIR。
- RankIC 正值比例。
- 月度和年度 RankIC。
- HAC t-stat，lag 5。
- 五组错位非重叠 5 日 RankIC。
- 行业、市值和市场阶段分组。
- 相对基线每日 RankIC 差。

不把重叠 5 日标签的普通 ICIR 简单乘 sqrt(252) 作为主要统计结论。

绝对门槛：

- 汇总样本外 RankIC 不低于 0.03。
- HAC t-stat 不低于 2.0。
- 至少 70% 测试年份为正。
- 单一年份贡献不超过总正向 IC 的 50%。
- 预测分数无塌缩。

相对门槛：

    delta_rank_ic =
        candidate_rank_ic - best_simple_baseline_rank_ic

跨 Fold mean(delta_rank_ic) 不低于 0.005，并报告配对日度差的 HAC 置信区间。

复杂模型失败时可以注册简单基线；不为了使用深度学习强行晋级。

## 11. 任务编排

历史回算：

    migration
    -> source watermark
    -> security master
    -> clean market
    -> tradability/universe
    -> factors
    -> labels
    -> DQ
    -> publish

Walk-forward：

    split
    -> Train factor selection
    -> factor_set
    -> dataset
    -> baselines
    -> candidate models
    -> Validation selection
    -> frozen Test
    -> cross-fold gate
    -> registry

统一 CLI：

    qr-v2 db migrate
    qr-v2 source inspect
    qr-v2 materialize market
    qr-v2 materialize universe
    qr-v2 materialize factors
    qr-v2 materialize labels
    qr-v2 dq run
    qr-v2 dataset build
    qr-v2 baseline run
    qr-v2 model train
    qr-v2 evaluate walk-forward
    qr-v2 registry promote
    qr-v2 inference daily
    qr-v2 report experiment

核心规则来自有版本配置，不能被临时命令行参数任意覆盖。

## 12. 每日推理和监控

每日任务由源表 watermark 触发，而不是仅依赖固定时钟。满足交易日 T、必需行情已到 T、必需因子已到 T 后执行：

1. 冻结源水位。
2. 清洗市场。
3. 构建当日股票池。
4. 获取冻结因子版本。
5. 应用 checkpoint 绑定预处理。
6. 构建可变长度截面。
7. 推理。
8. 预测 DQ。
9. 原子写 prediction_daily。
10. 发布报告。

预测只有在股票与 universe 完全一致、无幽灵股票、数值有限、分数不塌缩、覆盖率达标且版本兼容时发布。失败时不发布部分预测，也不复制前一天分数。

默认节奏：

- 每日：数据、推理和延迟监控。
- 每月：因子质量和漂移报告。
- 每季度：候选重训研究，不自动晋级。
- 每年：正式滚动更新和注册评审。

5 日标签成熟后追加 20/60/120/252 日滚动 RankIC、HAC、相对基线、分数漂移、行业市值暴露和因子漂移。短窗口恶化只告警，不自动换模型。

## 13. 错误恢复与安全

| 异常 | 行为 |
| --- | --- |
| YesLab 连接失败 | 核心任务失败，不发布 |
| Tushare 失败 | WARN，可继续但标记未对账 |
| 行情主键冲突 | 整个数据版本失败 |
| 单因子冲突 | 隔离该因子版本 |
| 股票池异常突变 | BLOCK |
| 标签覆盖不足 | 对应日期不进主验收 |
| GPU 中断 | 同一实验 checkpoint 恢复 |
| checkpoint 配置不匹配 | 拒绝加载 |
| 预测覆盖不完整 | 不发布 |
| 部分写成功 | 版本不发布，重跑缺失分块 |
| 重复任务 | 数据库锁阻止并发写同一版本 |

瞬时网络错误可重试，数据冲突和逻辑错误不得通过重试掩盖。

权限分离：

- yeslab_v2_reader 只读 dfs://YesLab。
- quant_v2_writer 只写 dfs://QuantResearchV2。
- 训练默认只读 V2 PUBLISHED 版本。
- migration 使用独立账号。
- 凭据通过环境或密钥系统注入，现有明文凭据实施时移除并轮换。

## 14. 测试体系

单元测试覆盖交易日映射、上市 252 日、ST/停牌有效期、退市边界、复权收益、重复键处理、Rank/MAD、HAC、双重门槛、状态机和 2026 锁定。

数据契约测试覆盖字段、类型、日期范围、主键、单位、代码格式和水位。源 schema 变化时显式失败。

属性测试验证：

- 上市前不入池。
- 未来公告不提前生效。
- 未来标签变化不影响训练特征。
- available_date 不晚于 signal_date。
- dataset 重建哈希一致。
- 股票顺序不改变 ID 对应。

泄漏诱捕测试构造未来收益假因子，Pipeline 必须拒绝。再构造只在全样本标准化后有效的因子，Fold-aware 预处理必须使其失效。

Golden Dataset 包含约 50 只股票和 60 个交易日，覆盖上市、退市、停牌、ST、复权、缺失和重复。

集成测试使用 dfs://QuantResearchV2Test，完成小规模清洗、股票池、标签、数据集、基线和评估，不使用生产写权限。

CI 必须通过格式、静态检查、单元、契约、Golden Dataset、小型端到端 Fold、migration dry-run、无明文密码和 YesLab 只读权限测试。

## 15. 实施阶段与验收

### Phase 0：现状冻结和容量审计

产物：

- source_inventory.json
- capacity_report.md
- v1_baseline_report.md
- credential_rotation_record

### Phase 1：V2 基础设施

- 创建测试库和生产 V2 库。
- 建立 migrations、research_v2、配置 schema、CLI、日志、测试和 CI。
- 验收 migration 幂等、YesLab 写入失败、配置哈希稳定、无明文密码。

### Phase 2：PIT 市场、股票池和标签

- 回算 2013-2025。
- 验收无主键冲突、无幽灵股票、股票池自然变化、标签日期和复权收益正确、2026 默认不可读。

### Phase 3：因子治理

- 导入全部现有因子元数据。
- 确定性去重、PIT 检查、历史物化、质量和相似度。
- 因子状态：IMPORTED、DQ_PASSED、PIT_PASSED、QUARANTINED、APPROVED、DEPRECATED。

### Phase 4：Dataset 和基线

- 完成 F01-F06、purge、factor set、dataset manifest、B0-B3 和 HAC 报告。
- 每个 Fold 产生唯一 best_simple_baseline。

### Phase 5：候选模型

- 依次 M1、M2、M3；M4 条件启用。
- 完成种子复核、消融、相对基线和双重门槛。

### Phase 6：Shadow 和兼容迁移

- prediction_daily 导出到独立 V2 CSV/PKL，不覆盖旧文件。
- Shadow 至少连续 60 个有效交易日。
- 验收每日任务、覆盖、分数、延迟 RankIC 和回滚。

### Phase 7：2026 锁定测试

- 2026 年结束且最后标签成熟后执行一次。
- 产出全年 RankIC、ICIR、HAC、月度稳定性、基线差和双重门槛结论。

## 16. 迁移与回滚

V1/V2 并行。V2 只有历史双重门槛、Shadow、运行稳定、兼容对接和回滚演练均通过，并经用户批准后才成为默认信号源。

切换通过 active_model_version 指针完成，不删除 V1。回滚恢复上一指针或停止 V2 发布，不需要恢复数据库。

## 17. 完成定义

平台重构完成：

- V2 表、代码、配置、测试和文档完成。
- 2013-2025 数据物化完成。
- 2020-2025 Walk-forward 可复现。
- 基线和候选模型可运行。
- 每日推理和 Shadow 稳定。
- V1 和 YesLab 未被修改。
- 回滚演练通过。

模型正式通过还要求：

- 历史双重门槛通过。
- Shadow 通过。
- 2026 锁定测试通过。

在 2026 锁定标签未成熟前，平台可以完成，但模型最高状态只能是 SHADOW。

## 18. 风险处置

| 风险 | 处置 |
| --- | --- |
| 因子不能回算到 2013 | 只参加覆盖满足要求的后续 Fold |
| 历史行业有效期不完整 | 只作诊断或外部重建 SCD |
| 公告时间粒度不足 | 公告日后下一交易日可用 |
| 因子表存储过大 | Phase 0 容量 Gate，按因子和月份分批物化 |
| 深度模型不超基线 | 注册基线并停止增加复杂度 |
| 2026 未结束 | 先 Shadow，不提前读取锁定结果 |
| Tushare 不可用 | 对账降级，不覆盖 DolphinDB |
| V2 日任务失败 | 不发布部分预测，V1 保持不变 |
| GPU/环境漂移 | 依赖快照、checkpoint 和配置哈希 |

## 19. 最终交付物

- V2 数据库 migrations 和字段文档。
- 数据血缘和数据契约。
- research_v2 包。
- 配置 schema 和默认配置。
- 历史回算、训练、评估和每日推理 CLI。
- DQ 和对账报告。
- Golden Dataset。
- 单元、契约、属性、泄漏和端到端测试。
- 2020-2025 Walk-forward 基线和模型报告。
- Shadow、故障恢复、锁定测试和回滚手册。
- 安全与权限说明。
