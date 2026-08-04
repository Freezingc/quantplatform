# 因子质量补救优先级清单

日期：2026-08-05  
数据范围：`SCREEN_TRAIN2020_2024_SOURCE_DQ_COMBINED_V1`（2020-01-02–2024-12-31）  
锁定测试：未读取  
数据库写入：无；本报告只读取已有质量快照

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

这两个 smoke 说明“无冲突”并不等于“可直接发布”：原始重复行和非有限行仍被隔离，未得到候选证据的行没有被填成常数。对应 manifest：

- [`P1 yeslab_395`](../../research_v2/artifacts/diagnostics/train_factor_repair_smoke_dec2020_p1_yeslab395_manifest.json)
- [`P3 yeslab_163`](../../research_v2/artifacts/diagnostics/train_factor_repair_smoke_dec2020_p3_yeslab163_manifest.json)
