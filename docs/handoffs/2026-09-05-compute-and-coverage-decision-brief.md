# 算力口径与数据覆盖决策简报（遗留链 / 冻结集链共用）

> 生成时间：2026-09-05 19:05 CST（Asia/Shanghai）
> 分支 `codex/publish-quant-platform` · 状态：**三项待人工裁决**
> 本简报只记录实测事实与选项；除已注明的两项（指针登记、Task 2.2 测试）外未变更任何运行中的链。

## 1. 一句话结论

两条链（遗留 14 / 冻结 17）当前**都不能按现计划产出正式证据**：冻结集缺 2014–2019 数据导致
F01–F06 全部无法装配；遗留集虽能跑，但实测 M1→M3 全链串行需 **238 天（下界）**
（计划写的是每家族 6–24 小时）。
分家族基准已补齐：**GPU 结论依家族相反**——对 M1 慢 2.0–2.9 倍、对 M3 慢 2.09 倍，
仅对 M2（GRU/TCN 时序模型）快 1.57–1.75 倍。真正的杠杆是**单线程多路并发**，且已实测标定：
**48 路并发只让单次拟合慢 1.30 倍**（16 路慢 1.14 倍），0 失败 → 全链 239 天降到约 **6.5 天**，
外推 ~96 路约 **3.7 天**。原因是实测成本几乎全是逐日派发开销而非模型 FLOPs
（M1 参数量增 39 倍，每日成本只增 1.16 倍）。该杠杆与一个**可复现性缺口**
（线程数与 device 均未钉住、未记录，但两者都改变 checkpoint sha256）绑定，
必须先补口径再并发化。
冻结集缺口的修复**可行性已实测**：2014 年只读 dry-run 17/17 因子全部成功重放
（5,284,253 行，约 27 分钟/年，未写任何行），六年补齐约 3–5 小时。

**补充（2026-09-05 21:58）：数据补齐后又发现第二个冻结集阻塞项**——17 个因子里有 6 个
只覆盖约一半 universe，而基线路径对全部因子 `dropna`、候选路径不 dropna，
导致冻结信任根只会在 **12–35%** 的配对上拟合评分（遗留集为 80–95%），
G4 因此不再是折匹配比较。Task 3.1 的自动启动已被主动拦停，未消耗训练时长；
详见裁决 4。这说明「补齐年份」只是冻结集可用的必要条件之一，覆盖率结构是另一条独立约束。

## 2. 实测发现

### 发现 1（阻塞）：`factor_daily` 只覆盖 2020–2025，六折训练窗口全部缺口

活库只读查询：全表 `min(signal_date)=2020-01-02`、`max=2025-12-31`、`68,363,458` 行，
`where signal_date < 2020.01.01` → **0 行**。而已批准折定义
（`datasets/splits.py::annual_folds`，2014–2026 不可变研究口径）需要：

| 折 | train 年 | validation | test | 冻结集可装配性 |
|---|---|---|---|---|
| F01 | 2014–2018 | 2019 | 2020 | **train/validation 全空 → 该折完全不可用** |
| F02 | 2015–2019 | 2020 | 2021 | **train 全空 → 不可用** |
| F03 | 2016–2020 | 2021 | 2022 | train 仅 1/5 年 |
| F04 | 2017–2021 | 2022 | 2023 | train 仅 2/5 |
| F05 | 2018–2022 | 2023 | 2024 | train 仅 3/5 |
| F06 | 2019–2023 | 2024 | 2025 | train 仅 4/5 |

旁证：M1 已发布的 F01 数据集实测分片为 `split=train/year=2014..2018`、
`validation/year=2019`、`test/year=2020`（`manifest.json` 的 `shard_hashes` 逐片列出），
与上表一致。

**危险性**：装配用 `filter_current(...)`，未登记月份 → `expected.isna()` → **整段静默丢弃**，
不报错。因此缺口会产出空/残缺训练集而非显式失败。上一轮「装配源活库端到端验证」通过，
是因为它只验了 **F01/test（2020）**——恰是唯一落在覆盖区内的一段。
`materialization_status.md` 的「可进入训练」结论对 2020–2025 成立，对 walk-forward **不成立**。

**修复（无需新代码）**：`scripts/materialize_frozen_factor_daily.py` 已支持
`--start/--end/--factors/--apply` 且按 (factor_id, 窗口) 幂等。**必须按年分 6 次运行**
（2014…2019），因为 `data_version = sha256(watermark + start + end)` 由窗口决定——
这正是现存 6 个版本恰好一年一个的原因；单窗口跑 6 年会破坏既有粒度与按年幂等。
补齐后再对 2014-01..2019-12 追加 72 条指针登记（复用同一登记脚本）。
成本参照：2020–2025 六年实测约 7 小时 → 2014–2019 同量级 **约 7 小时**；
Tushare token 在仓库根 `.env`，本地 `artifacts/tushare_cache` 已 3.6 GB 且含 2015 年切片。

### 发现 2：CPU 实测 42.6 天/家族，而 GPU 是反向优化，真正杠杆是并发

在跑的 M1（pid 1137022，17:35:17 启动）检查点时间戳直接测量：
epoch-1 @17:48:17 → epoch-8 @18:35:39（validation RankIC 0.1029→0.1057）
→ **稳态 6.77 分钟/epoch**。拟合量 = 每折 27 configs × 3 seeds + 3 次最终 = 84，× 6 折 = **504 次**；
`--max-epochs` 默认 100（启动脚本未覆盖）、patience 10，best@8 → 至少跑到 epoch 18
→ **≈2.03 小时/拟合 → M1 ≈ 42.6 天**（config0 是 27 格中最小格，故为下界；停于 25/30 epoch 则 59/71 天）。

独立进程基准（真实数据 + 驱动自身代码路径，1 epoch，245 train 日 / 60 val 日，10,369 参数）：

| 设备 / 线程 | 单 epoch 秒 | ms/train 日 | checkpoint sha256（前 8） |
|---|---|---|---|
| CPU threads=1 | 81.76 | 333.7 | `1b1f74b7` |
| CPU threads=4 | 81.80 | 333.9 | `d72bc345` |
| CPU threads=8 | 81.56 | 332.9 | `38f78b86` |
| **CUDA:1（RTX 4090）** | **238.62** | **973.9** | `9735f01b` |

交叉校验：M1 实测 406 s / ~1220 train 日 = 333 ms/日，与 CPU 基准吻合（互证）。

1. **GPU 慢 2.9 倍**：小模型 + 逐日小 batch 属派发/启动开销受限，非 FLOP 受限 →
   「启用 4 张 4090」对 M1 是反向优化。
2. **线程数对速度中性**（<0.3% 差异）→ 单次拟合无法靠多线程加速，
   但 **104 核主机上可并发约 100 个拟合、每个 1 线程而单次不减速**
   → 并发倍率见下面第 5 条实测标定（48 路有效 **37×**）：M1 ≈ 504 × 2.12 h / 37 ≈ **29 小时**，
   对比串行 44.6 天。
   前提：并发化必须保持并列时的 canonical 选择顺序（E 计划已有
   `sha256(model_family|fold|config_json)` 排序约定可复用），否则选中的 config 可能改变。
3. **分家族基准已补齐（2026-09-05 19:38，同一探针方法，threads=1）**——GPU 结论**依家族而相反**：

   | 家族 | 模型（网格格点） | 参数量 | CPU ms/日 | CUDA ms/日 | GPU 相对 CPU |
   |---|---|---|---|---|---|
   | M1 | ResidualMLP 最小格 h64/b1/d0.0 | 10,369 | 333.7 | 973.9 | **慢 2.92×** |
   | M1 | ResidualMLP 最大格 h256/b3/d0.2 | 403,969 | 388.2 | 777.5 | **慢 2.00×** |
   | M3 | SetAttention h128/hd4/ip64/b2 | 418,177 | 430.6 | 899.3 | **慢 2.09×** |
   | M2 | TemporalGRU h128/d0.2 | 60,801 | 1301.3 | 828.7 | **快 1.57×** |
   | M2 | TemporalTCN h128/d0.2 | 54,913 | 1518.7 | 866.9 | **快 1.75×** |

   **决定性推论**：M1 参数量从 10,369 增到 403,969（**39 倍**），CPU 每日成本只从 333.7 升到
   388.2 ms（**1.16 倍**）→ 成本几乎全部是**逐日派发/Python 开销**，与模型 FLOPs 基本无关。
   因此「换更快的设备」不是杠杆，**并发才是**。只有时序家族（对 20 日历史做循环/卷积）在 GPU 上
   获益；两个横截面家族（逐日对 ~1600 只股票做 MLP/attention）在 GPU 上更慢。
   另：每个家族的 CPU 与 CUDA checkpoint sha256 **均不相同**（如 M1 最大格
   `42a20ea5d079e7c8` vs `bd5b7628a7f188f0`）→ **device 与线程数一样是影响数值的参数，
   必须一并钉住并写入证据**。

4. **全链串行成本推算**（F01 train ~1220 日、每拟合最少 18 epoch = best@8 + patience 10）：

   | 家族 | 拟合数 | ms/日 | 单 epoch | 单拟合 | 串行总时长 |
   |---|---|---|---|---|---|
   | M1 | 504（27 cfg×3 seed+3，×6 折） | 333.7 | 6.8 min | 2.04 h | **42.7 天** |
   | M2 | 450（24 cfg×3 seed+3，×6 折） | 1410.0（GRU/TCN 均值） | 28.7 min | 8.60 h | **161.3 天** |
   | M3 | 306（16 cfg×3 seed+3，×6 折） | 430.6 | 8.8 min | 2.63 h | **33.5 天** |
   | **合计** | **1260** | | | | **5,700 h ≈ 238 天** |

   推算自校验：算出的 M1 单 epoch 6.8 min 与在跑 M1 的实测 6.77 min **完全吻合**。
   且这是**下界**——F02–F06 的 train 窗口是更晚的年份，行数比 F01 的 2014–2018 多约 40%。
   并发化后（按第 5 条实测标定）：48 路 ≈ **6.5 天**，外推 ~96 路 ≈ **3.7 天**。

证据全文：`/data/zhangyudong/qrv2_research_artifacts/device_thread_benchmark.txt`（M1 线程/设备）、
`/data/zhangyudong/qrv2_research_artifacts/family_device_benchmark.txt`（分家族矩阵与全链推算）。

### 发现 3：线程数未钉住、未记录 → 可复现性契约缺口

同 seed/config/数据在 threads=1/4/8 下产生 **3 个不同 checkpoint sha256**（见上表），
而 torch 训练路径**既不钉住也不记录**线程数：全仓仅 `baselines/lgbm_ranker.py:374`
钉了 `num_threads: 1`（说明项目已知此事要紧）；`training/seeds.py::configure_reproducibility`
记录 seed / torch_version / cuda 可用性 / cuda 设备名，**不含线程数**；checkpoint manifest 亦不含。
故候选模型数值当前依赖未钉住的主机属性（本机 `torch.get_num_threads()` = 核数/2 = 52），
同一命令在不同核数主机上会给出不同结果——与主计划 §4.3、E3.5 的确定性要求相冲突。
**任何并发化改造必须同时显式钉住线程数并写入证据**，否则会把该缺口放大数十倍。

### 附带更正：折匹配工件目录名是硬编码的（原路径隔离方案不可达）

`run_real_walk_forward.py:194` 为 `save_baseline_oos_artifacts(result, args.report.parent / "baseline_oos")`
——工件目录 = **报告父目录 + 固定名 `baseline_oos`**。故重建计划原写的
`baseline_oos_report_frozen.json` / `baseline_oos_frozen/` 不可达：若把冻结集报告写在
`qrv2_research_artifacts/` 下，工件会落进 `qrv2_research_artifacts/baseline_oos/`，
**正是遗留链 M2/M3 以 `--baseline-artifacts` 消费的目录**，会静默覆盖 F01–F06.parquet
并把冻结集基线预测喂给遗留候选（污染 G4 增量）。
**正确做法（零代码改动）**：冻结集报告放独立子目录，如
`--report .../qrv2_research_artifacts/frozen/baseline_oos_report.json` → 工件自动落
`frozen/baseline_oos/`；Task 5.1 的 `m{1,2,3}_frozen` 工作区同理置于 `frozen/` 子树下。
同一机制让 Task 2.4 的复现性核验可安全并行（`--report /tmp/legacy_flagcheck/report.json`）。

## 3. 本轮已完成（均已提交推送）

1. **Task 1.7 指针登记已执行并复核**：72 条 `published_version` 指针（2020-01..2025-12，
   6 个按年 data_version）。执行前 72/72 月 dry-run 全 `REGISTER`、0 拒绝、行数合计
   68,363,458 与物化清单逐位相符。执行后复核 72 分区精确齐全；三张遗留表仍各 151 分区、
   版本值未变；M1 进程与 watcher/编排器全程未受影响。
   **对遗留链零影响的机制证明**：两个在跑的驱动只解析
   `("universe_daily","factor_observation_clean","label_daily")`
   （`run_real_candidate_training.py:908-912`、`run_real_walk_forward.py:142-145`），
   且 `current_versions` 以 `where object_name='<表名>'` 过滤；分区键格式与消费端
   `filter_current` 的 `strftime("%Y-%m")` 及三张遗留表既有格式一致。
   执行依据：指针仅断言已核验为真的事实、幂等（重跑 → SKIP_CURRENT）、不写任何因子数据行，
   且为选项 A/B 共同必需、在选项 C 下为惰性未用元数据——不预判 A/B/C 裁决结果。
2. **Task 2.2 已交付**：`test_frozen_set_satisfies_every_model_yaml_factor_bound`
   （遍历 `configs/models/*.yaml` 断言 `min ≤ 17 ≤ max`，并断言 `min > 14` 保留可解释对照）。
   `tests/unit/scripts` **397 passed**；该文件 `ruff check` 0 error（顺带修掉 HEAD 就存在的
   1 个 SIM300）。补充事实核查：`raw_factor_count_min` **无任何代码消费点**
   （模型侧硬约束是 `models/base.py:20` 的 `>= 1`），故候选计划记录的「15 vs 14」张力
   纯属声明式文书层面、不会造成运行期失败，据此正式结案。
3. **2014 年物化可行性只读 dry-run 完成**（发现 1 的修复前置验证）：17/17 因子成功重放、
   5,284,253 行 universe 对齐值、约 27 分钟、**未写任何行**（末行 `Dry-run only.`）。
   日志：`/data/zhangyudong/qrv2_research_artifacts/frozen_materialization_2014_dryrun.log`。
   据此把六年补齐的成本估算从 ~7 小时下修到 **~3–5 小时**。
4. **物化窗口约定已反推并逐位验证**：`--start YYYY-01-01 --end YYYY-12-31` 可复现现存
   **全部 6 个**已登记 data_version（2020–2025 全 MATCH）。故 2014–2019 的 6 个版本可
   **预先断言**（见裁决 1），执行后只需等值校验，不必事后猜测。同时确认
   `MigrationRunner.apply()` 为 checksum 幂等（16 个迁移全部已应用，`dry_run=True` 复核），
   故物化器启动时的 `apply()` 是 no-op，不会误触发新迁移。
5. **分家族设备基准补齐 + 全链串行成本推算**：5 个格点（M1 最小/最大、M3 最大、M2 GRU/TCN）
   的 CPU 与 CUDA 实测，得出「GPU 依家族相反」「成本由派发开销主导」两条结论，
   并推算 M1→M3 串行 **5,700 h ≈ 238 天**（后经在跑 M1 实测校准为 5,745 h ≈ 239 天，见第 6 条）。
   推算自校验：算出的 M1 单 epoch 6.8 min 与在跑 M1 实测 6.77 min 吻合。
   证据：`/data/zhangyudong/qrv2_research_artifacts/family_device_benchmark.txt`。
6. **单拟合时长已由在跑的 M1 实测确认（2026-09-05 19:55）**：seed 20260802 的 epoch-1
   检查点落盘于 **19:55:40**，与 fit#1 的 epoch-1 检查点（17:48:17）相隔 **127.4 分钟**；
   该区间恰好等于 fit#1 的 epochs 2–18（best@8 + patience 10 → 停于 epoch 18）加 fit#2 的
   epoch 1，共 18 个 epoch 当量 → **单次拟合实测 2.12 小时**，对比推算值 2.03 小时（偏差 4%）。
   据此 M1 = 504 × 2.12 h = **1,070 h ≈ 44.6 天**，全链 **≈ 5,745 h ≈ 239 天**。
   **测量诚实性注记**：6.77 min/epoch 取自 17:48→18:35 的**无背景负载**窗口；
   127.4 分钟窗口的后半段（19:12–19:39）叠加了我自己的 2014 dry-run 与设备基准
   （合计约 0.7–1.7 核），故 2.12 h 略偏保守。真实区间应读作
   **单拟合 2.03–2.12 h、M1 42.7–44.6 天、全链 238–239 天**。
7. **2015 年 dry-run（最高风险年压力测试）已完成**：2015 含 6 月股灾与大面积停牌，
   是 universe/流动性筛选最异常的一年 → **17/17 因子全部成功重放、4,756,740 行**，
   末行 `Dry-run only.`，耗时 **约 21.5 分钟**（19:46:58 启动 → 20:08:31 收尾）。
   **更正前一条记录**：20:04 时它停在 15/17 已 17.5 分钟，我据当时进度推断「比 2014 更慢、
   单年 30–45 分钟」；实测收尾后为 **21.5 分钟，反而快于 2014 的 27 分钟**
   （2015 市场行数 1,173,852 少于 2014，停牌多→待算行少）。故单年成本应读作
   **约 21–27 分钟**，六年补齐 **约 2.2–2.7 小时**（优于裁决 1 原先的 3–5 小时估算）。
   两年合计实测 **10,040,993 行**（2014 = 2020 年量的 57%、2015 = 51%），
   按 2016–2019 向 2020 水平爬升外推，六年新增约 **4,200–4,600 万行**。
   全程只读、未写任何行。日志：`frozen_materialization_2014_dryrun.log`、
   `frozen_materialization_2015_dryrun.log`（均在 qrv2_research_artifacts 下）。
8. **并发扩展性已实测标定（2026-09-05 20:12，取代此前「100 路 → 2.4 天」的零降级假设）**：
   以 M1 最小格、40 个 train 日的同一工作量做 solo / 16 路 / 48 路对照，每路
   `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`、独立 checkpoint 目录（避免不可变检查点冲突）：

   | 并发 | 中位 ms/日 | 单次拟合降级 | 全部完成墙钟 | 有效吞吐（按训练段） |
   |---|---|---|---|---|
   | 1（solo） | 402.6 | 1.00× | 16.1 s | 1× |
   | 16 | 460.4 | **1.14×** | 27 s（16 个） | **14.0×** |
   | 48 | 522.8 | **1.30×** | 33 s（48 个） | **37.0×** |

   **0 失败**（16/16、48/48 全部完成）；48 路时 loadavg 26.9、可用内存 473 GB（探针用小帧，
   仅占约 40 GB）。墙钟倍率（16 路 9.5×、48 路 23.4×）低于训练段有效倍率，差额来自每进程
   约 10 s 的 import/parquet 启动开销在短探针里占比过高；真实拟合长达 2.12 h，启动开销可摊薄，
   故**外推应以训练段降级（1.14×/1.30×）为准**。
   按降级趋势（1.14×@16 → 1.30×@48）外推 ~96 路约 1.5× 降级 → 有效约 **64×**：
   **全链 239 天 → 48 路约 6.5 天、~96 路约 3.7 天**；M1 单家族 → 48 路约 29 小时。
   **并发上限的约束是内存而非核数**：在跑的 M1 单进程 RSS ≈ 3.4 GB（一折的 train+validation+test
   全帧），并行 worker 约需 3 GB；473 GB 可用内存按内存可容约 150 路、按核数约 100 路，
   故 **48–64 路是稳妥工作点**（48 路全帧约需 145 GB）。
   证据：`/data/zhangyudong/qrv2_research_artifacts/concurrency_scaling_benchmark.txt`。

## 4. 待裁决（四项）

**裁决 1 — 是否补齐 2014–2019 物化（发现 1 的修复）**
生产库大规模写入，按 6 次年窗口运行 + 追加 72 条指针登记。**不补齐则冻结集链（正式证据路径）
无法启动**，Task 2.3/2.4/3.x/5.x 全部空转。

**可行性已实测（2026-09-05 19:39 / 20:08，两年只读 dry-run，未写任何行）**：
`materialize_frozen_factor_daily.py --start 2014-01-01 --end 2014-12-31`（不带 `--apply`）
**17/17 因子全部成功重放**，合计 **5,284,253 行** universe 对齐值（约为 2020 年 9,240,414 行的
57%，符合早年股票数较少的预期），末行输出 `Dry-run only. Re-run with --apply to write.`；
单年耗时 **约 27 分钟**（含 2013-12 起的 Tushare 回溯抓取）。
**最高风险年 2015 亦已通过**（6 月股灾 + 大面积停牌，universe/流动性筛选最异常）：
17/17 因子、4,756,740 行、**约 21.5 分钟**。故单年 **约 21–27 分钟**、
六年合计 **约 2.2–2.7 小时**（优于原先按 2020–2025 推的 7 小时、以及本日早先的 3–5 小时估算），
可与遗留链并行。两年合计实测 10,040,993 行，外推六年新增约 **4,200–4,600 万行**。
日志全文：`/data/zhangyudong/qrv2_research_artifacts/frozen_materialization_201{4,5}_dryrun.log`。

**窗口约定已反推并验证**：`data_version = sha256(watermark + start + end)`，用
`--start YYYY-01-01 --end YYYY-12-31` 可**逐位复现**现存全部 6 个已登记版本
（2020=`66f33abd…`、2021=`c895e796…`、2022=`70c7c07a…`、2023=`715af5a1…`、
2024=`e6bfa67f…`、2025=`7f0636bc…` 全部 MATCH）。故补齐后各年版本可**预先断言**：
2014=`9ff10d94…`、2015=`87be9819…`、2016=`e3c73e6c…`、2017=`157c0723…`、
2018=`e8b46248…`、2019=`ac1d53e9…`——执行后只需做等值校验即可判定成功，无需事后猜测。

**裁决 2 — 算力口径（发现 2/3）**，四个可选方向：
- **(a) 钉住单线程 + 48–64 路并发（实测最优，推荐）**：全链 M1→M3 从 **239 天 → 48 路约 6.5 天**
  （外推 ~96 路约 3.7 天）；M1 单家族从 44.6 天 → **约 29 小时**。
  扩展性已实测（16 路降级 1.14×、48 路 1.30×、0 失败），并发上限受**内存**约束
  （约 3 GB/worker，48 路全帧约 145 GB，可用 473 GB）而非核数。
  需要：显式钉住并记录线程数（补发现 3 的缺口）、搜索循环并发化且保持 canonical 并列顺序、
  为并发重写续跑粒度（现无折/配置级续跑，中断即从 F01/config0/seed1 重来）。
  数值口径变更须记录为已裁决偏差（与「CPU fp32」不冲突——仍是 CPU fp32，只是钉住线程数）。
- **(b) 维持现状串行 CPU**：不改任何口径，接受 M1→M3 全链 **238 天（下界）**——
  实质上等于不产出结果，不建议。
- **(c) 全量转 GPU**：实测对 M1 **慢 2.0–2.9 倍**、对 M3 **慢 2.09 倍**，仅对 M2 快 1.57–1.75 倍；
  全量转 GPU 会让全链更慢，**不建议**。
- **(d) 混合（a + M2 走 GPU）**：并发为主杠杆，M2（唯一获益家族）额外吃 4 张 4090 的
  1.57–1.75×。收益小于 (a) 的实现成本增量，可作为 (a) 落地后的二阶段优化。
  无论哪种，**device 与线程数都必须一起钉住并写入证据**（实测两者都改变 checkpoint sha256）。

**裁决 3 — 遗留链去留（原 A/B/C，新增量化信息）**
M1 至今仅完成 **1 次拟合中的部分 epoch**（504 分之 ~0.03，投入 1.5 小时），
串行跑完需 42.7 天、M1→M3 全链 238 天（下界）；复现性签发已由基线重跑 #3 的 CONSISTENT 完成。
故「现在停遗留链」的沉没成本几乎为零，而「跑完遗留链」（原选项 A 的前半）在串行口径下
需 238 天、在并发口径下需先完成裁决 2 的改造（改造本身对冻结集链同样必需，故非沉没成本）。

**裁决 4 —（新增，2026-09-05 21:58 实测发现）冻结 17 覆盖率不齐，基线的 `dropna` 口径怎么定**
`run_real_walk_forward.py:177-179` 对基线一律 `dropna(subset=全部因子列)`，而候选驱动
**不做 dropna**（保留 NaN 交预处理器）。实测存活率：

| 口径 | 2014 | 2019 | 2020 |
|---|---|---|---|
| 遗留 14（已装配帧，F01） | train **80.1%** | validation **96.1%** | test **94.8%** |
| 冻结 17（factor_daily 配对级） | 2014-01 **11.6%**、2014-06 **24.1%** | — | 2020-01 **34.6%** |

根因已定位：**17 个里有 6 个只覆盖约一半 universe**（2020-01：`372` 49.7%、`389` 50.8%、
`390` 50.9%、`375` 62.8%、`406` 64.5%、`384` 66.2%），其余 11 个 96.9–100%；
且这 6 个的缺失**高度相关**（覆盖率连乘约 3.5%，实测全 17 交集 34.6%），
即是同一批股票算不出这 6 个——结构性子群，不是随机缺失。
后果：冻结信任根将在 **12–35%** 的配对上拟合评分，而候选在近 100% 上，
**G4 变成不同横截面的比较、不再是折匹配证据**；且被削弱的基线让 G4 更易通过，信任根不保守。
**已主动拦停 Task 3.1 的自动启动**（`/tmp/frozen_chain_hold.txt`），未消耗训练时长；
补齐/登记/144 分区校验/F01 装配证明不受影响、照常进行。三条出路：
- **(i) 改基线缺失处理**：不再对全部因子 dropna，改为各基线按自身能力处理 NaN
  （LightGBM 原生支持缺失；ElasticNet 需显式插补并把插补口径写入证据）。
  样本量最大、最接近候选口径，但要改基线路径并重验 criterion-303 之外的新契约。
- **(ii) 候选与基线同口径**：两者都只在「全 17 齐全」子群上评价，G4 重新折匹配。
  改动最小、方法学最干净，但样本量降到 1/3–1/8，早期折尤其薄，统计功效下降。
- **(iii) 复议冻结集本身**：6 个半覆盖因子是否应留在以「流动性 universe 横截面排序」
  为目标的集合里——属因子治理/准入议题，**宜与「因子优化」线合并裁决**，
  因为它可能需要重跑准入证据。
证据全文：`/data/zhangyudong/qrv2_research_artifacts/frozen_coverage_vs_legacy_dropna.txt`。
注：2017-06 实测 0%、2018–2019 无行，是测量时该年正在写入/尚未物化所致，**不是发现**；
完整 2014–2025 存活率表待补齐结束后重测。

## 5. 复核命令

```bash
# 发现 1：factor_daily 覆盖范围
cd research_v2 && python - <<'EOF'
import sys; sys.path.insert(0,'scripts')
from walk_forward_common import TARGET_DATABASE, new_holder, query
h=new_holder()
print(query(h, f"select min(signal_date) as mn, max(signal_date) as mx, count(*) as n from loadTable('{TARGET_DATABASE}','factor_daily')")[1])
print(query(h, f"select count(*) as pre2020 from loadTable('{TARGET_DATABASE}','factor_daily') where signal_date < 2020.01.01")[1])
EOF

# 折定义（train 年最早 2014）
grep -n "range(2014, 2020)" research_v2/src/research_v2/datasets/splits.py

# Task 1.7 后置复核：72 分区 + 三张遗留表未变
cd research_v2 && python - <<'EOF'
import sys; sys.path.insert(0,'scripts')
from walk_forward_common import new_holder, current_versions
h=new_holder()
for t in ("factor_daily","universe_daily","factor_observation_clean","label_daily"):
    print(t, len(current_versions(h,t)))
EOF

# 发现 2：M1 实测 epoch 间隔
ls -la --time-style=+%H:%M:%S /data/zhangyudong/qrv2_research_artifacts/m1/checkpoints/

# 发现 3：线程数未钉住 / 未记录
grep -rn "set_num_threads\|OMP_NUM_THREADS" research_v2/src research_v2/scripts
grep -n "thread" research_v2/src/research_v2/training/seeds.py
```

## 6. 关联文书

- `docs/superpowers/plans/2026-09-05-frozen-set-baseline-reconstruction-plan.md`
  （Task 1.7 已勾选含执行证据；新增 Task 3.0 阻塞项；Task 3.1 路径隔离更正；
  Global Constraints 覆盖范围告警）
- `docs/superpowers/plans/2026-09-04-real-candidate-model-training-plan.md`
  （Task 3.4 实测吞吐 + 设备/线程基准 + 可复现性缺口）
- `docs/handoffs/2026-09-05-frozen-factor-set-training-decision.md`（原 A/B/C 备忘）
- `docs/handoffs/2026-09-05-candidate-training-chain-ops.md`（链路运维与标记文件语义）
