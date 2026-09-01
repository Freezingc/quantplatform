# PIT 黄金样本（B8）

真实历史边界样本：从 `dfs://YesLab/stock_factor_daily` 只读抽取 **50 只证券 ×
60 个交易日（2025-09-01 … 2025-12-01，SSE 日历）**，覆盖主板（沪/深）、创业板、
科创板、北交所、ST 变更（000711.SZ 窗口内摘帽、002586.SZ 窗口内持续 ST）、
停牌缺口、新股上市（603370.SH 2025-09-05、920100.BJ 2025-09-08）、窗口内除权
（000062.SZ、300272.SZ、601133.SH、920007.BJ、920100.BJ）与 BJ/SSE 日历差异缺口。

- 抽取脚本：`scripts/extract_pit_golden_sample.py`（确定性：板块配额 + 流动性排序）
- 独立期望：`scripts/compute_pit_golden_expected.py`（不共享生产代码；滚动特征
  以 pandas rolling 均值为文档化数值原语）
- 等价性测试：`tests/golden/test_yeslab_market_50x60.py`（canonical rows SHA-256）
- 合成边界夹具：`pit_label_*` / `pit_market_*`（独立手工期望 + SHA-256 锚）

## 行级算例（10 个）

| 域 | 证券 | 日期 | 说明 | 公式与结果 | 派生量 |
|---|---|---|---|---|---|
| clean | 000062.SZ | 2025-09-25 | ex-rights day | adjusted_open = 29.82 x 8.1227 = 242.218914 | valid_days_20=19, adv20=1063831.732 |
| clean | 300272.SZ | 2025-09-12 | ex-rights day | adjusted_open = 6.73 x 6.3537 = 42.760401 | valid_days_20=10, adv20=nan |
| clean | 601133.SH | 2025-09-10 | ex-rights day | adjusted_open = 12.97 x 1.0492 = 13.608124 | valid_days_20=8, adv20=nan |
| clean | 920100.BJ | 2025-10-31 | ex-rights day | adjusted_open = 72.01 x 0.9648 = 69.475248 | valid_days_20=20, adv20=166614.337 |
| clean | 000711.SZ | 2025-09-01 | first window day | adjusted_open = 1.84 x 9.759 = 17.956560 | valid_days_20=1, adv20=nan |
| clean | 920007.BJ | 2025-09-30 | ex-rights day | adjusted_open = 71.06 x 1.0141 = 72.061946 | valid_days_20=20, adv20=46267.212 |
| label | 000062.SZ | 2025-09-01 | signal -> entry 2025-09-02 -> exit 2025-09-09 | raw_return_5d = exit_adj_open / entry_adj_open - 1 = -0.026229508 | valid=1 |
| label | 000711.SZ | 2025-09-09 | signal -> entry 2025-09-10 -> exit 2025-09-17 | raw_return_5d = exit_adj_open / entry_adj_open - 1 = -0.027322404 | valid=1 |
| label | 920100.BJ | 2025-09-10 | signal -> entry 2025-09-11 -> exit 2025-09-18 | raw_return_5d = exit_adj_open / entry_adj_open - 1 = -0.005882353 | valid=1 |
| label | 603370.SH | 2025-11-21 | signal -> entry 2025-11-24 -> exit 2025-12-01 | raw_return_5d = exit_adj_open / entry_adj_open - 1 = 0.043793964 | valid=1 |

## 公式

- `adjusted_open = open × adj_factor`（精确浮点乘法）
- `valid_trade_day = (vol > 0) ∧ (amount > 0)`
- `valid_days_20`：按证券排序后最近 20 个观测行内 `valid_trade_day` 计数（min_periods=1）
- `adv20`：最近 20 个观测行内有效 `amount` 均值；窗口内有效数 < 15 时为 NaN
- `raw_return_5d = exit_adj_open / entry_adj_open − 1`，entry = 信号日 +1 交易日，
  exit = 信号日 +6 交易日（交易日历平移）；任一价格缺失/非正则标签无效
- canonical 浮点：`adjusted_open` 保留 9 位小数、`adv20` 保留 3 位小数（货币量级
  ULP 吸收），其余字段精确比较；行按 (trade_date|signal_date, ts_code) 排序后
  以 `json.dumps(..., sort_keys=True, separators=(",",":"))` 计算 SHA-256
