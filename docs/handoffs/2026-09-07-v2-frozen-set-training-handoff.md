# v2 冻结因子集 × 新折结构：训练链接口交接（2026-09-07 用户批准）

> 生成：2026-09-07 · 分支 `codex/publish-quant-platform`
> 状态：v2 已落库（追加批次，v1 未动）；factor_daily 2014–2025 **已全部就绪**（09-07 16:35 复核）
> 批准包（用户 2026-09-07「好的」确认）：W1 移除 405/398 + 裁决 4 混合方案 + 4y/1v/1t 四折结构 + 2017–2019 补物化

## 0. 一句话

冻结集 v2 = **15 因子**（9 稠密基线可用 + 6 稀疏 candidate-only），fold 结构改为 **4 年训练 + 1 年验证 + 1 年测试 × 4 折（测试年 2022–2025）**；`factor_daily` 已有 2020–2025 全量，2017–2019 正在补；2026 锁定段全程未触碰。

## 1. v2 冻结集（已写入 `factor_set_frozen`，frozen_date=2026-09-07 批次）

| 角色 | 因子 | 说明 |
|---|---|---|
| 基线+候选（9，稠密） | 357(S)、471(S)、206(A)、464(A)、96(A)、425(B)、381(B)、484(B)、443_reg30_max20_v1(REPAIR) | 线性基线 B0–B3 与候选 M1–M3 都可用 |
| 仅候选（6，稀疏） | 372(A)、375(A)、384(A)、389(A)、390(S)、406(S) | 公式对下跌日无定义（NaN≈上涨广度补集）；候选侧 FoldPreprocessor 原生处理 NaN；**不进基线**（避免 dropna 稀薄化信任根） |
| 移除（2） | 405(A)、398(B) | W1：关注对评审 FAIL（给定 381 的增量保留率 7.6%/20.3%）；物化值保留归档、不删除 |

- 契约文件：`research_v2/artifacts/diagnostics/factor_set_frozen_v2.json`（含 roles、folds、data_version）。
- 数据库：`factor_set_frozen` 追加 2026-09-07 批次 17 行（15 ADMITTED/ADMITTED_REPAIR_VARIANT + 2 REMOVED_WATCH_PAIR_REVIEW）；**v1（2026-09-03 批次）原样保留**，回滚=忽略 v2 批次。
- 逐对证据已补录 `factor_pair_redundancy`：3 对 × 2020–2025，window_id `<year>_fd`（与重放谱系 `<year>` 区分），pair_class 沿用既有阈值（381↔405、381↔398 全 ELEVATED；381↔443v1 全 OK）。

## 2. 新折结构（替代 F01–F06）

| 折 | 训练（4年） | 验证 | 测试（样本外） |
|---|---|---|---|
| F1 | 2017–2020 | 2021 | **2022** |
| F2 | 2018–2021 | 2022 | **2023** |
| F3 | 2019–2022 | 2023 | **2024** |
| F4 | 2020–2023 | 2024 | **2025** |

- 变更理由：2014–2016 漂移最剧；4 年 ≈ 340 万截面样本够用；测试年仍覆盖 熊/弱/极端/牛 四体制；训练计算省 1/3；体制指标（2016-04/2017-01 起）全程有值。
- 与 criterion-303 的逐折可比性本已随因子集更换失效，内部一致性（基线与候选同折）不受影响。

## 3. factor_daily 数据状态

| 区间 | 状态 |
|---|---|
| 2020–2025 × 17 因子 | ✅ 已物化并验收（68,363,458 行、全量查重 0 重复，`materialization_status.md`） |
| **全表 2014–2025** | ✅ **106,598,549 行、12 年 × 17 因子、0 重复键**（快照 `materialization_status.md` 2026-09-07 16:35 刷新，覆盖 F1–F4 与旧 F01–F06 全部折） |
| 2014–2019 × 17 因子 | ✅ **已完成**（并行会话 2026-09-05 20:23→23:38 按年补物化 + 逐年验证；本侧 09-07 复核：2014–2019 共 38.2M 行、17 因子/年、0 重复键）。本侧 09-07 启动的 v2 清单补跑因幂等机制全部跳过（无重复写入，零冲突） |
| 2026 | 🔒 锁定，未物化、禁止读取 |
| 405/398（2020–2025） | 已物化行保留（归档），v2 不使用 |

宇宙 `cn_liquid_a` 与标签 `forward_open_5d:v1` 2014 年起齐全（已核验），缺口只有因子值。

## 4. 训练链消费口径（给并行会话）

1. **基线 B0–B3**：只用 9 个稠密因子。dropna 覆盖率**已实测**（2026-09-07，`dense9_baseline_coverage_v1.json`）：全 9 稠密齐全占宇宙成员日 **86.6%–96.8%/年**，四个训练窗 **91.8%–93.3%**（对比全 17 齐全的 34.6% → 信任根稀薄问题消解）。覆盖率下限由 206（修复版，~90%）决定。注意：原覆盖率门槛按 5 年训练窗 ~1214 天标定，新 4 年窗 ~970 天需按本实测重标定。
2. **候选 M1–M3**：用全部 15 因子，NaN 交给 FoldPreprocessor（"下跌日无值"本身是信息，禁止在数据层插补）。
3. **G1–G6、E7 ≤30 configs、3 种子、判定器模式全部沿用**，但折匹配以 F1–F4 为准。
4. 因子数 = 15，恰好满足 `raw_factor_count_min: 15`。
5. 体制降权（D3=a）不变：大盘股主导 ON 时对 443 变体降权，先在验证段 OOS 证实再固化；新增事实：443v1 在 largecap-ON 桶保留率 0.48 未崩塌、liquidity-shock-ON 桶残差翻转 -0.26（见 watch_pair_review_v2），与既有监控设计一致。

## 5. 需要并行会话同步修改的点（协调清单）

1. 冻结集计划 Task 3.0：补数据范围 2014–2019 → **2017–2019**（折结构变了）；文档同步。
2. `walk_forward_factor_daily.py` 的 `FROZEN_FACTOR_SET_V1` 硬编码 tuple（f00..f16）→ v2 十五因子（f00..f14）。
3. `verify_admission_consistency.py` / `census_duplicate_key_shapes.py` 的默认契约路径仍指 v1 JSON——评估是否切到 `factor_set_frozen_v2.json`。
4. 覆盖率门槛按 4 年训练窗重标定（`verify_frozen_fold_coverage.py` 的口径）。
5. 新折的 fold_cache_key 与旧 F01–F06 键空间不相交（表名版本映射已含），无需清理旧缓存。
6. 你们侧的 M1 失败（09-06）与决定 B（18,884 行）处理不受本交接阻塞，照常推进。

## 6. 复核命令

```bash
# v2 批次
python -c "
import dolphindb as ddb
s=ddb.session(); s.connect('10.176.55.211',8848,'xiahongjie','123456')
print(s.run(\"select decision, count(*) as n from loadTable('dfs://QuantResearchV2Test','factor_set_frozen') where frozen_date=date('2026.09.07') group by decision\"))"
# 补物化进度
tail -5 research_v2/artifacts/diagnostics/factor_daily_materialization/backfill_2017_2019.log
# 完成后验收（行数按年 x 15 因子）
python -c "
import dolphindb as ddb
s=ddb.session(); s.connect('10.176.55.211',8848,'xiahongjie','123456')
print(s.run(\"select count(*) as n from loadTable('dfs://QuantResearchV2Test','factor_daily') where year(signal_date) between 2017 and 2019 group by factor_id, year(signal_date) as yr order by factor_id, yr\"))"
```

## 7. 回滚

v1 批次（2026-09-03）未动。回滚 = 决策文书声明忽略 2026-09-07 批次即可；`factor_pair_redundancy` 的 `*_fd` 行窗口后缀独立、不参与旧查询；`factor_daily` 的 2017–2019 追加行键空间与既有行不相交，幂等可重跑。
