# 模型侧走查证据包（2026-09-14）

配套叙述文档：`../../2026-09-14-model-side-walk-forward-results-v0.1.md`。
这个目录放的是**机读证据 + 被引用的原件副本**，全部是文本；
**没有权重、没有检查点、没有 parquet 面板、没有数据库导出**。

## 文件

| 文件 | 生成方式 | 内容 |
|---|---|---|
| `candidate_standings.json` | `extract_from_reports.py` | 9 个运行工作区（含基线与复现副本）的完整门控 G1–G6、OOS 指标、逐年 RankIC、逐折明细、`report.json` 与 `promotion_evidence.json` 的 sha256；基线的逐折 best_simple 选择与落选原因 |
| `per_fold_winners.json` | 同上 | 逐折胜者配置、集成与逐 seed 验证 RankIC、每折 top3、冻结选择点 id |
| `yearly_rank_ic.json` | 同上 | 逐年 RankIC + 门控状态，便于直接出表 |
| `factor_sets.json` | 同上（用 `ast` 从驱动源码解析，不 import） | 三个冻结因子集的完整 id 列表、规范特征名 `f00…f14`、基线可见性规则的原文理由 |
| `datasets.json` | 同上 | 每折已发布数据集的行数、列序、分片年份、manifest 与分片哈希、数据版本摘要 |
| `in_progress_search_records.json` | 同上 | **M2 中途快照**：逐格搜索记录（配置、集成与逐 seed 验证 RankIC、记录 sha256 与写入时刻）。**验证集口径，不是 OOS** |
| `provenance.json` | 同上 | 所有被读文件的 sha256、主树 git HEAD 与分支、提取时刻、刻意排除的内容清单 |
| `problems_found.md` | 手写，逐条给出出处 | 问题记录：评估阶段六个阻塞点、成本模型三次低估（含本轮我自己的一次 ETA 误算）、stale 门禁默认值、四处永不触发的守卫、两个驱动缺陷、GPU/内存/主机与治理未决项 |
| `comparison_interim.md` | `research_v2/scripts/summarize_candidate_comparison.py`（主分支 `39ee96a`） | **工具渲染**的跨候选对比：可比性分组与排除理由、G1–G6 矩阵、两两增量矩阵（带"共享基线恒等式"校验）、领先候选的逐年分布与逐折最差 seed。同输入两次渲染 md5 全同 `c79e5545ebad6d36c3e935935176f3b7`。**M2/M3 出报告后重跑即得计划 Task 5.1 的那份文档** |
| `yearly_baseline_deltas/` | `research_v2/scripts/analyze_yearly_baseline_deltas.py`（主分支 `78216af`，sha256 前缀 `c36aadec52042ac4`） | 三个已完成候选各一份 json + md：**基线在候选 945 日集上的逐年值与逐年增量**（此前任何工件里都没有）、四道逐位验收的实测值、**24 个不可评日的逐日清单**、`coverage_attribution`（**逐侧**覆盖率：候选四年恒为 **1.0000**、基线 0.89–0.99、**24 天全部由基线一侧决定**）、以及关掉 0.90 地板后的敏感性重算（结论更强，且与基线自己的签名报告**逐位相同**） |
| `extract_from_reports.py` | — | 只读提取器，stdlib only |
| `vendor_sources.py` | — | 原件复制器，同时重算 `vendored/MANIFEST.json` |
| `vendored/` | `vendor_sources.py` | 10 份仓库文档 + 7 份签名 `report.json` 的**逐字节副本**，`MANIFEST.json` 记源路径、sha256、取值时的 git HEAD。新增的一份是 `docs/reports/2026-09-14-yearly-deltas-and-the-24-unevaluable-days.md`——逐年增量与那 24 天归属分析的完整方法、§4.1 逐侧覆盖率表、§4.2 的设计观察与三种处置、§6 敏感性、§8 边界 |

## 为什么要复制原件

本分支 `codex/research-conditional-factor-review` 比 `codex/publish-quant-platform`
落后约 95 个提交，叙述文档引用的那些报告与交接文档**在本分支上并不存在**。
复制过来是为了让这一包在另一台机器上自包含；引用前请用 `vendored/MANIFEST.json`
里的 sha256 与主分支上的原件比对，确认拿到的是同一份。

## 重新生成

两个脚本都是**只读**的：不写工件树、不碰数据库、不动主工作树
（主树上有一个多日的 M2 运行在跑，`research_v2/scripts/run_real_candidate_training.py`
在它跑完前不能被编辑，因为下一折的 worker 会从磁盘重新导入它）。

```bash
cd docs/research/evidence/2026-09-14-model-side-walk-forward
python3 extract_from_reports.py   # 默认 --artifacts /data/zhangyudong/qrv2_research_artifacts
python3 vendor_sources.py         # 默认 --repo /data/zhangyudong/FDU/Quant-Platform-Prod_CODEX
```

重跑逐年增量分析（每个工作区约 20 分钟、单进程约 8 GiB 峰值、**只读**；
`m0r_v2` 与 `m1_full_v1` 带 `--sensitivity-no-score-health`，已被取代的载体不带）：

```bash
cd /data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2
A=/data/zhangyudong/qrv2_research_artifacts
for W in m0r_v2 m1_full_v1; do
  python scripts/analyze_yearly_baseline_deltas.py --workspace $A/$W --report $A/$W/report.json \
    --key $A/secrets/selection_evidence_hmac.key --sensitivity-no-score-health \
    --out-json $A/yearly_baseline_deltas/$W.json --out-md $A/yearly_baseline_deltas/$W.md
done
```

重渲染 `comparison_interim.md`（在主工作树里跑，M2/M3 报告落地后把 `--report` 补上即可）：

```bash
cd /data/zhangyudong/FDU/Quant-Platform-Prod_CODEX/research_v2
A=/data/zhangyudong/qrv2_research_artifacts
python scripts/summarize_candidate_comparison.py \
  --report M0R=$A/m0r_v2/report.json --report M1-full=$A/m1_full_v1/report.json \
  --report M1-carrier=$A/m1_parallel_v1/report.json --report M0-refined=$A/m0_v3/report.json \
  --report M0-base=$A/m0_v2/report.json --report M0R-replicate=$A/m0r_v3/report.json \
  --baseline-report $A/baseline_v2/report.json --out comparison_interim.md
```

`in_progress_search_records.json` 会随 M2 的进度变化；其余文件只在有新运行归档后才变。
M2 出报告后请重跑一次提取器，并把叙述文档升到 v0.2、用最终 OOS 结果整节替换第 3 节。

## 提取时刻

2026-09-14（UTC 时间戳见 `provenance.json` 的 `extracted_at_utc`），
主树 HEAD `c490353`、分支 `codex/publish-quant-platform`、工作树干净。
