# 2014–2019 准入因子重算回填完成报告（2026-09-01）

## 结论

14 个重放验证通过的准入因子已重算并发布 2014-01 … 2019-12 全部 72 个月，
`factor_observation_clean` 现覆盖 **2014-01-02 … 2025-12-31**（部分因子至
2026-07-31）。主计划标准 303 的 TRAIN 年数据缺口已消除；下一步为按 D 计划
构建 F01–F06 数据集并运行 B0–B3 walk-forward。

## 执行证据

- 因子集合（14，重放机器精度验证）：yeslab_376, yeslab_377, yeslab_380, yeslab_381, yeslab_387, yeslab_388, yeslab_391, yeslab_396, yeslab_400, yeslab_405, yeslab_425, yeslab_432, yeslab_433, yeslab_442
  （GP_026/GP_074 因重放阻塞未纳入）
- 预检：72/72 月通过，无阻塞隔离（隔离均为 NO_FINITE_VALUE 暖机缺失）
- 源发布：72/72 月，合计 **56,235,259** 行
- 压缩至 `factor_observation_clean_latest`：72/72 分区，合计 **56,235,259** 行
- 发布后核验：2020 前行数 **56,235,259** = 预检净行数（零丢失）；
  每因子首日均为 2014-01-02
- 独立抽样核对：yeslab_376 于 2016-06-13…17 重算对比 12,473 行，
  max_abs_diff = 7.49e-16，>1e-9 行数 = 0

## 谱系

- 来源模式：`recomputed-from-raw-market`（attempt_nonce=recomputed-train-years-14-v1）
- 计算器：`数据库/factor_calculators/GP_Factor_{007..074}_Calculator.py`
- 市场输入：`dfs://YesLab/stock_factor_daily`（覆盖 2013+，暖机自动回溯）
- 工具：`research_v2/scripts/publish_recomputed_factor_backfill.py`
  （preflight→publish→compact，复用 source_backfill 审计发布路径）
- 安全基础：`docs/reports/2026-09-01-admitted16-formula-replay-verification.md`
