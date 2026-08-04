# 因子质量补救优先级清单

日期：2026-08-05  
数据范围：`SCREEN_TRAIN2020_2024_SOURCE_DQ_COMBINED_V1`（2020-01-02–2024-12-31）  
锁定测试：未读取  
数据库写入：仅向新建的 `factor_repair_candidate` 表追加候选隔离批次；未修改任何已有业务表、`factor_observation_clean` 或原始 YesLab 因子表

## 结论

当前 source-qualified 集合有 47 个因子。47 个因子的值冲突键均为 0、完整日期范围均通过，但它们仍全部处于 `SOURCE_DQ_PASS_PENDING_SIMILARITY`，不能直接作为模型输入。

主要问题不是“47 个因子都失效”，而是两个可分离的数据质量问题：

- 38 个因子的有限值比例低于 0.80，需要先确认缺失是否来自可重算的行情/财务依赖；
- 27 个因子的原始行重复比例为 0.50，需要做确定性去重并重新计算覆盖率；重复行不能计入有效覆盖；
- 所有因子的 PIT 数字目前仍是去重前代理统计，尚未完成 compact universe 交集检查。

## 分层结果

| 优先级 | 因子数 | 判定条件 | 处理动作 |
| --- | ---: | --- | --- |
| P0-公式门禁 | 1 | 已知公式重叠值不一致 | 保持阻断，核对源值版本/输入，不放宽 tolerance |
| P1-去重+低覆盖 | 20 | duplicate_ratio=0.50 且 finite_ratio<0.80 | 先 exact dedup，再做依赖补值和 compact-universe 覆盖 |
| P2-低覆盖 | 18 | finite_ratio<0.80 且无重复 | 查缺失依赖、按月重算；不能用横截面填充 |
| P3-仅去重 | 7 | finite_ratio≥0.80 且 duplicate_ratio=0.50 | exact dedup 后重做 PIT/相似度 |
| P4-当前结构最干净 | 2 | finite_ratio≥0.80 且无重复 | 直接进入 train-only similarity/公式哈希复核 |

P0 仅指已在代表性公式 smoke 中发现的 `yeslab_433/GP_064`。这不是预测失败结论，而是源表存值与原始计算器重算值不一致的公式一致性阻断。

## 因子分组

### P0：公式重叠阻断

`yeslab_433 (GP_064)`

2020-12 精确使用 `stock_factor_daily` 和原始预热边界后仍有 6 个 mismatch；最大被拒绝行绝对误差 `66.086938`，最大相对误差 `0.012042`。必须先审计 `factor_data9` 的源值版本或生成批次。

`yeslab_391 (GP_022)` 已经没有公式重叠 mismatch，但该月仍有非有限源值隔离，因此不等于可以发布。

### P1：重复与有限覆盖同时存在（20）

```text
yeslab_395, yeslab_391, yeslab_388, yeslab_117, yeslab_118,
yeslab_96, yeslab_114, yeslab_115, yeslab_95, yeslab_98,
yeslab_385, yeslab_376, yeslab_393, yeslab_382, yeslab_387,
yeslab_377, yeslab_380, yeslab_381, yeslab_383, yeslab_396
```

处理顺序固定为：

1. 按 `(trade_date, ts_code, factor_name)` 只删除完全相同的 revision 行；
2. 对去重后的 finite coverage 重新计算月度/横截面覆盖；
3. 只有存在可审计公式和同口径行情时，才允许补非有限值；
4. 将未能补齐的行保留在 quarantine，不把它们转成 0 或横截面均值。

### P2：低覆盖但未发现重复（18）

```text
yeslab_471, yeslab_433, yeslab_458, yeslab_473, yeslab_456,
yeslab_400, yeslab_437, yeslab_401, yeslab_442, yeslab_443,
yeslab_439, yeslab_431, yeslab_94, yeslab_484, yeslab_425,
yeslab_41, yeslab_405, yeslab_432
```

这组优先检查依赖行情的缺口、上市日/停牌边界和滚动窗口预热，不应把低 finite ratio 直接解释成预测能力差。

### P3：先去重（7）

```text
yeslab_163, yeslab_362, yeslab_363, yeslab_365,
yeslab_366, yeslab_357, yeslab_359
```

这些因子的 finite ratio 已达到 0.80 以上，但重复行会抬高 PIT 代理覆盖，必须去重后再进入相似度和 compact-universe 门禁。

### P4：结构上最干净（2）

```text
yeslab_155, yeslab_79
```

这只表示 source DQ 结构较干净，不代表已经通过预测稳定性、train-only similarity 或最终准入。

## 下一轮执行顺序

1. 完成 P1/P3 的确定性去重物化，并保存去重前后行数、finite ratio、月度 coverage 对照；
2. 对 P2 及 P1 的低覆盖因子逐个核对依赖字段和滚动预热，沿用 `NO_FINITE_CANDIDATE` 隔离；
3. 对 `yeslab_433` 单独完成源值版本核对，未解释前禁止任何覆盖；
4. 对去重后仍通过的因子运行全 TRAIN 日 cross-sectional similarity、FDR/年度稳定性和 compact-universe PIT；
5. 所有层级通过后才生成新的候选 factor set；本清单不改变当前六因子 frozen contract，也不启动模型训练。

## 代表性去重 smoke

在不写库的情况下对一个 P1 和一个 P3 因子执行 2020-12 dry-run：

| 因子 | 分组 | clean 行 | 隔离行 | 状态 | 隔离原因 |
| --- | --- | ---: | ---: | --- | --- |
| `yeslab_395/GP_026` | P1 | 89,495 | 47,447 | `PASS_WITH_UNRESOLVED_QUARANTINE` | `NO_EVIDENCE_BACKFILL` |
| `yeslab_163/Alpha95` | P3 | 97,085 | 39,420 | `PASS_WITH_UNRESOLVED_QUARANTINE` | `NO_EVIDENCE_BACKFILL` |

P1 smoke 的审计统计为 `136,942` 个 exact duplicate group/row、`94,894` 个源非有限行；P3 smoke 为 `136,505` 个 exact duplicate group/row、`78,840` 个源非有限行。经过确定性去重后，12 月有限覆盖分别只有 `89,495/136,942=0.653525` 和 `97,085/136,505=0.711219`，隔离行分别为 `47,447` 和 `39,420`，没有伪造填充值。这也说明全范围 finite ratio 不能替代逐月门禁。

这两个 smoke 说明“无冲突”并不等于“可直接发布”：原始重复行和非有限行仍被隔离，未得到候选证据的行没有被填成常数。对应 manifest：

- [`P1 yeslab_395`](../../research_v2/artifacts/diagnostics/train_factor_repair_smoke_dec2020_p1_yeslab395_v2_manifest.json)
- [`P3 yeslab_163`](../../research_v2/artifacts/diagnostics/train_factor_repair_smoke_dec2020_p3_yeslab163_v2_manifest.json)

另外重跑了此前受 V2 行情起始日影响的 `2020-01 / yeslab_391 (GP_022)`：切换到
YesLab 精确 `stock_factor_daily` 后，结果仍为 `clean=56,920`、
`quarantine=38,344`、`NO_FINITE_CANDIDATE`，且 mismatch 为 0。这说明这 38,344
行不是简单的 V2 行情表缺失，而是对应证券在原始滚动窗口内确实没有足够的有限候选；
不能为了提高覆盖率而用常数或横截面填充。详见
[`2020-01 GP_022 exact-input smoke`](../../research_v2/artifacts/diagnostics/train_factor_repair_smoke_jan2020_source_factor_market_v1_manifest.json)。

### P1/P3 全部 27 个因子的 compact-universe 复核

对 P1/P3 全部 27 个因子执行了 2020-12 的只读审计，分母为
`cn_liquid_a/v1` 与 `forward_open_5d/v1` 的有效交集，共 64,306 个成熟观测。
27 个因子全部达到当前 compact coverage 门槛：

| 指标 | 27 个因子最小值 |
| --- | ---: |
| 整体 finite coverage | 0.976892 |
| 最低日覆盖 | 0.974212 |
| 日覆盖 P05 | 0.974498 |
| 月覆盖 | 0.976887 |

因此，P1/P3 的全市场低 finite ratio 不能直接作为模型输入拒绝理由；它们需要继续通过全 TRAIN 月度稳定性、PIT 版本、相似度和 FDR 门禁。该结论也不能外推到 P2 或 P0，后两类仍需单独审计。

完整只读结果见 [`P1/P3 compact coverage JSON`](../../research_v2/artifacts/diagnostics/compact_universe_coverage_dec2020_p1_p3_all27_v1.json)。

### P1/P3 全 TRAIN（2020–2024）compact-universe 复核

随后将同一口径扩展到完整 TRAIN 区间（2020-01-01–2024-12-31）。本次查询按
`store_factor` 分组并在 DolphinDB 服务端执行有限值 `(trade_date, ts_code)` 去重；成熟分母仍是
`cn_liquid_a/v1 ∩ forward_open_5d/v1`，只使用会话级临时键，不写入任何数据库表。

27 个因子的结果为：20 个通过结构覆盖门槛、7 个因最低日覆盖失败。通过这里只表示
`COMPACT_DQ_PASS_PENDING_SIMILARITY`，尚未通过相似度、FDR、年度稳定性或准入发布门禁。

| 结果 | 数量 | 因子 |
| --- | ---: | --- |
| `COMPACT_DQ_PASS_PENDING_SIMILARITY` | 20 | `yeslab_114, 115, 117, 118, 163, 357, 359, 362, 363, 365, 366, 376, 381, 383, 391, 393, 396, 95, 96, 98` |
| `COMPACT_DQ_FAIL` | 7 | `yeslab_377, 380, 382, 385, 387, 388, 395` |

全 TRAIN 的总体范围如下：

| 指标 | 27 个因子最小值 | 说明 |
| --- | ---: | --- |
| 整体 finite coverage | 0.983577 | GP_022；仍明显高于全市场分母下的代理值 |
| 最低日覆盖 | 0.481387 | GP_026；7 个失败因子的共同门禁触发项约为 0.48–0.49 |
| 日覆盖 P05 | 0.968101 | GP_022；失败因子并非全期间普遍缺失 |
| 最低月覆盖 | 0.788443 | GP_019；其余失败因子月度最低值仍在 0.88 以上 |

这 7 个因子的失败主要表现为少数交易日的断崖式缺失，而不是整体覆盖率不足；不能用
横截面均值或常数填充掩盖。下一步应定位这些最低覆盖交易日，核对对应公式的滚动预热、依赖字段
及源表生成批次；在没有可复算证据前，将这些日期的记录保留在 quarantine。其余 20 个因子可以
进入下一道 train-only similarity/FDR/年度稳定性检查，但不能据此直接扩大 frozen factor set。

完整结果见 [`P1/P3 全 TRAIN compact coverage JSON`](../../research_v2/artifacts/diagnostics/compact_universe_coverage_train2020_2024_p1_p3_all27_v1.json)。

### 2020-09-18 换手率断档的补救审计

对上述 7 个失败因子继续做了日期定位和外部证据核对。异常并非因子公式本身首先产生：
`stock_factor_daily` 在 `2020-09-18` 的成熟宇宙中只有 `1,354/2,740` 条
`turnover_rate` 有限值，缺失行的 `amount`、`vol` 仍存在，但股本字段也同时缺失，
因此不能从 YesLab 现有字段安全反推换手率。5/10 日滚动公式将同一缺口传导到
`2020-09-22`，解释了最低日覆盖约 0.49 的现象。

Tushare `daily_basic(20200918)` 返回 4,070 条有限 `turnover_rate`，对成熟宇宙的
1,386 条缺失记录全部有可用替代值。基于这些值只在内存中重算，并对原始有限重叠值
执行严格核对：

| 因子 | 重叠 mismatch | compact 可补行 | 结论 |
| --- | ---: | ---: | --- |
| `yeslab_377 / GP_008` | 0 | 1,386 | `TURNOVER_REPAIR_CANDIDATE_READY` |
| `yeslab_380 / GP_011` | 0 | 1,386 | `TURNOVER_REPAIR_CANDIDATE_READY` |
| `yeslab_382 / GP_013` | 0 | 1,385 | `TURNOVER_REPAIR_CANDIDATE_READY` |
| `yeslab_385 / GP_016` | 0 | 1,377（含滚动传导共 6,853） | `TURNOVER_REPAIR_CANDIDATE_READY` |
| `yeslab_387 / GP_018` | 0 | 1,385 | `TURNOVER_REPAIR_CANDIDATE_READY` |
| `yeslab_388 / GP_019` | 0 | 1,377（含滚动传导共 8,204） | `TURNOVER_REPAIR_CANDIDATE_READY` |
| `yeslab_395 / GP_026` | 2,016 | 1,348 | `TURNOVER_REPAIR_BLOCKED_OVERLAP` |

六个候选的有限重叠值完全一致，并已通过纯内存 overlay dry-run：状态均为
`REPAIR_AUDIT_PASS_WITH_UNRESOLVED_QUARANTINE`。它们只填充有有限候选的缺失行，
其余 `NO_FINITE_CANDIDATE` 行继续隔离；这不代表可以覆盖 YesLab 原表，也不代表已经发布。
`GP_026` 的重叠值仍有较大不一致（最大绝对差约 `42,249.21`），overlay 为
`REPAIR_BLOCKED`，不能因为外部换手率完整就强行补值。

补救审计脚本和结果：

- [`turnover gap audit script`](../../research_v2/scripts/audit_turnover_gap_repair.py)
- [`turnover gap audit JSON`](../../research_v2/artifacts/diagnostics/turnover_gap_repair_audit_2020-09-18_v1.json)
- [`six-factor candidate batch manifest`](../../research_v2/artifacts/diagnostics/turnover_gap_repair_candidates_2020-09-18_v1.json)

首批候选批次先以本地、可复核的 CSV 形式生成，包含 6 个因子的有限候选补行（分别为
1,386、1,386、1,385、6,853、1,385、8,204 行），不包含 `GP_026`。每个批次都有
SHA-256；写入新 V2 隔离表时必须用该哈希和同一来源水印做幂等校验。

随后已将 migration 009/010 应用到 `dfs://QuantResearchV2Test`，并把通过 preflight 的
20,599 行写入新表 `factor_repair_candidate`：

- `data_version=2e74fb7328665c5ef3344f43ce13a22d891b1b890458cd88310d754f395bc731`；
- 6 个因子均有有限候选值，`source_value` 有限值为 0，业务键无重复；
- 日期范围为 2020-09-18–2020-09-25，`GP_026` 未写入；
- `factor_observation_clean` 行数保持 `170,333,348`，没有推进 clean 对象指针；
- `published_version` 只新增了 `factor_repair_candidate/2020-09` 的隔离指针。

发布前检查报告和发布结果分别见 [`candidate preflight`](../../research_v2/artifacts/diagnostics/turnover_gap_candidate_preflight_2020-09-18_v1.json)
和 [`candidate publish report`](../../research_v2/artifacts/diagnostics/turnover_gap_candidate_publish_2020-09-18_v1.json)。

首批候选随后已物化到新的 V2 隔离对象；后续延长窗口和全 TRAIN 复核见下一节。
`GP_026` 始终留在阻断清单，暂不进入候选表。任何正式发布仍需单独批准。

### GP_019 延长窗口补救与全 TRAIN 复核

对 `yeslab_388 / GP_019` 继续追踪到 2020-10-15 后，发现源表在
2020-09-18–2020-10-09 对大量证券各写入两条空值 revision（每个键的两条记录均为
非有限值），因此只修复 9 月批次仍会在 10 月 9 日的 10 日滚动窗口留下缺口。这是
源表 revision/生成批次问题，不是候选公式与源值不一致。

本次处理采取按月隔离的方式：

- 重新用同一 Tushare `daily_basic(20200918)` 水印和原始预热区间重算 GP_019；
- 对 2020-10-01 起可补的 1,343 行生成单独的 `2020-10` 候选批次；有限源值重叠
  `37,633` 行，mismatch `0`，候选值全为有限数，未覆盖任何有限源值；
- preflight 的五个阻断检查全部 PASS，批次已发布到新表，
  `data_version=ad03741eca14ea75458938ce121df3b1643afdeb87c4b4360f2ca559797b86a7`，
  `factor_repair_candidate/2020-10` 指针已推进；原始因子表与 clean 表仍未改变。

数据库核验显示候选表物理行数为 `46,601`（包括首批 9 月版本、9 月延长版本和 10 月批次，
表本身保持 append-only）；当前版本指针暴露 9 月最新 `24,659` 行和 10 月 `1,343` 行，
候选值有限行 `46,601`，`source_value` 有限行为 `0`。`factor_observation_clean` 仍为
`170,333,348` 行，schema migration 记录为 10 条。

局部 2020-09-01–10-31 复核显示 GP_019 的最低日覆盖率为 `0.981063`、最低月覆盖率
为 `0.991532`。随后对 2020-01-01–2024-12-31 的六个换手率候选因子执行完整
compact-universe 复核（成熟分母为 `cn_liquid_a/v1 ∩ forward_open_5d/v1`，共
`3,996,740` 个观测）：

| 因子 | 整体覆盖 | 最低日覆盖 | 最低月覆盖 | 结果 |
| --- | ---: | ---: | ---: | --- |
| GP_008 / `yeslab_377` | 0.999998 | 0.998548 | 0.999887 | PASS_PENDING_SIMILARITY |
| GP_011 / `yeslab_380` | 0.999998 | 0.998548 | 0.999887 | PASS_PENDING_SIMILARITY |
| GP_013 / `yeslab_382` | 0.999758 | 0.997811 | 0.999263 | PASS_PENDING_SIMILARITY |
| GP_016 / `yeslab_385` | 0.998109 | 0.987259 | 0.995044 | PASS_PENDING_SIMILARITY |
| GP_018 / `yeslab_387` | 0.999758 | 0.997811 | 0.999263 | PASS_PENDING_SIMILARITY |
| GP_019 / `yeslab_388` | 0.997580 | 0.981063 | 0.991532 | PASS_PENDING_SIMILARITY |

将这六行结果与未受 overlay 影响的其余 21 个因子合并后，27 个 P1/P3 因子中有
26 个达到 compact 双重门槛，唯一仍失败的是 `yeslab_395 / GP_026`。GP_026 继续保持
`TURNOVER_REPAIR_BLOCKED_OVERLAP`，不进入候选表，也不进入模型输入。完整合并结果见
[`turnover repaired full TRAIN compact JSON`](../../research_v2/artifacts/diagnostics/compact_universe_coverage_train2020_2024_turnover_repaired6_v3.json)，
六因子独立复核见
[`six-factor focus JSON`](../../research_v2/artifacts/diagnostics/compact_universe_coverage_train2020_2024_turnover_repaired6_focus_v3.json)。

在全日相似度和年度稳定性复核完成前，状态仍是“结构质量通过、相似度/FDR/年度稳定性
待审”，不是最终准入；GP_026 保持隔离，期间不启动任何模型训练。复核结果见下一节。

### 26 因子 TRAIN 全日相似度与年度稳定性结果

已完成 26 个结构通过因子的 2020–2024 全日横截面相似度计算。相似度计算使用每月
`published_version` 指向的最新 `factor_observation_clean` 版本；clean 表缺少的因子从
YesLab 原始表按月读取并做确定性有限值去重；6 个换手率候选继续按“只填充源值缺失键”
叠加。每个因子对均有 `1,212` 个有效交易日，未读取 2026 locked-test。

结果为 19 个冗余、7 个唯一：

- 冗余簇代表候选包括 `yeslab_114/115`（0.9474）、`yeslab_117/118`（0.9448）、
  `yeslab_357/359`（0.9675）、`yeslab_365/366`（0.9793）、
  `yeslab_377/382`（0.9823）、`yeslab_387/388`（0.9824）等；
- 唯一候选为 `yeslab_163, yeslab_363, yeslab_385, yeslab_396, yeslab_95,
  yeslab_96, yeslab_98`，它们与最近邻的绝对 Spearman 均低于 0.90。

现有五年年度预筛要求每个年度 ICIR 也达到 0.50，因此只有
`yeslab_357, yeslab_359, yeslab_376, yeslab_393, yeslab_98` 通过该保守门槛；与全日
唯一集合交集后，当前最保守的全门禁交集只有 `yeslab_98`。另一方面，主计划规定的
“五年中至少三年同向、年度均值/正向日占比达标”稳定性规则下，26 个因子均未在年度
均值或方向占比上失败；这 7 个唯一因子可作为下一轮候选，但不应在未确认年度 ICIR
规则前直接发布为 frozen factor set。

当前结论：

1. `GP_026 / yeslab_395` 仍因公式 overlap mismatch 阻断；
2. 其余 26 个通过结构覆盖、日级 HAC/FDR 和全日相似度证据；
3. 经过严格“年度 ICIR + 全日唯一”双门槛后暂只保留 `yeslab_98`；
4. 尚未训练任何模型，也未推进 frozen factor set。下一步应先审阅年度 ICIR 的规则
   是否作为每年硬门槛，再决定对 7 个唯一候选做第二类补救或保守发布。

完整门禁汇总见
[`TRAIN26 gate report`](../../research_v2/artifacts/diagnostics/factor_quality_train26_gate_report_v4.json)，
全日原始结果见
[`TRAIN26 full-day similarity`](../../research_v2/artifacts/diagnostics/train_similarity_full_day_2020_2024_repaired26_v3.csv)，
clean 版本指针审计见
[`V2 clean pointer audit`](../../research_v2/artifacts/diagnostics/v2_clean_pointer_version_audit_train26_v1.json)。

### 修复后 RankIC 再检验

按照数据修复设计，未直接沿用修复前的 RankIC。针对 2020-09-18–2020-10-09 的换手率
候选影响窗口，重新计算 6 个修复因子的每日横截面 RankIC，并把这 60 条受影响记录
替换到本地 2020–2024 RankIC 面板，再重新计算 26 因子的 HAC/FDR/ICIR。结果显示：

- 6 个因子各自仍有 `1,212` 个有效 TRAIN 日；
- 60 条 RankIC 记录发生替换，26 因子的每日 predictive gate 全部保持通过；
- 指标变化很小，没有因修复而制造新的显著性或方向性：例如 GP_019 的 mean RankIC
  从 `-0.082160` 变为 `-0.081981`，ICIR 从 `-0.573622` 变为 `-0.570411`，仍满足
  当前绝对值阈值；
- 该再检验只写本地 CSV/JSON，不写入已有 `factor_rank_ic_daily` 表，也不改变模型或
  frozen factor 指针。

对应证据见
[`repaired RankIC report`](../../research_v2/artifacts/diagnostics/repaired_factor_rankic_train26_v1.json)、
[`repaired RankIC impact rows`](../../research_v2/artifacts/diagnostics/repaired_factor_rankic_train26_v1_impact.csv)、
[`repaired TRAIN metrics`](../../research_v2/artifacts/diagnostics/repaired_factor_rankic_train26_v1_metrics.csv)
和更新后的
[`TRAIN26 gate report V3`](../../research_v2/artifacts/diagnostics/factor_quality_train26_gate_report_v5.json)。

### 冗余簇代表候选（尚未冻结）

已保存完整的 325 个因子对矩阵。按全日相似度阈值 `0.90` 的全部边组成连通分量，再按
`FDR q → signed mean RankIC → |ICIR| → factor_id` 排序选代表，可以得到 14 个代表候选：

```text
yeslab_115, yeslab_118, yeslab_163, yeslab_357, yeslab_363,
yeslab_365, yeslab_376, yeslab_380, yeslab_383, yeslab_385,
yeslab_396, yeslab_95, yeslab_96, yeslab_98
```

该代表集略低于主计划的 15–40 个特征目标，但目标是 soft target，不应为了凑数放回冗余
因子。完整 pair matrix 的每对有效日数均为 1,212，连通分量和阈值边均已落盘；因此这仍
是“候选代表集”，不能直接冻结，冻结前还需确认年度 ICIR 硬门槛采用“每年硬失败”还是
主计划中的“至少三年同向”规则。候选代表提案见
[`TRAIN26 full-pair cluster proposal`](../../research_v2/artifacts/diagnostics/train26_cluster_representative_proposal_v2.json)，
完整 pair matrix 见
[`TRAIN26 pair matrix`](../../research_v2/artifacts/diagnostics/train_similarity_pairs_2020_2024_repaired26_v1.csv)。
