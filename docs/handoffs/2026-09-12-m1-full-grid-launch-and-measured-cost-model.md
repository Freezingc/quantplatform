# 2026-09-12：那个"六道门全过"的 M1 只搜了 27 格里的 1 格；全网格已发射，三个家族的成本模型首次全部实测

本篇接 `2026-09-11-m1-evaluation-crash-lineage-and-footprint.md`（13 节）。
上一篇结束时，M1 有了可归档的判定、六个阻塞项都修好了、M2 的发射包备好了。
本篇记录之后 3 小时里发生的三件事：一个会改变结论的事实、一次真实发射、
以及把全链剩余的外推数字换成实测。

---

## 1. 最要紧的一条：`m1_parallel_v1` 的"全过"来自**单配置**搜索，不是全网格

`report.json` 里 `mode=full`、`oos.all_passed=true`、六道门逐项 PASS、
`delta_rank_ic = 0.005844602701716051`（G4 门槛 0.005）。这些全是真的。
但 **`mode` 只区分"真实数据路径"与"smoke 路径"，与网格搜了几格无关**——
驱动里 `mode` 只有 `"smoke"` 与 `"full"` 两个取值，`--prior-report` 的校验
（`prior_families_from_reports`）也只查 `family` / `mode=full` / `oos.all_passed` 三项，
**没有任何一处要求网格搜满**。

实际搜了几格，证据在三处、互相印证：

* `m1_parallel_v1/verdict.md` 的"Per-fold constrained search"表，四折的 `Searched` 列**都是 1**；
* `m1_parallel_v1/search/F{1..4}/` 每折**只有一条**记录，都是 `c0-9ef4cccccd3fdf78.json`；
* 那条记录的 config 是 `{"hidden_dim": 64, "num_blocks": 1, "dropout": 0.0}`，
  即 `m1_config_grid` 27 格里的**第 0 格、也是最小的一格**。

**这不是新发现，昨天的交接已经写了，我在此更正自己的措辞。**
`2026-09-11-…` 第 8 节第 2 条原话是：「M1 过门了，但这个判定是"载体验证"而不是"M1 的上限"。
两次运行都带 `--max-configs 1`……实际只搜了第一格……所以「一个未调参的配置就过了六道门」是正面信号，
而**「M1 不如 M0R」这个结论不成立**——还有 26 格从未试过。」
本轮我最初是从 `verdict.md` 与 `report.json` 读到这件事的、没先读那段散文，
所以早先的汇报把它讲成了"差点被误报的新事实"，**那个框定是错的，记录在此以免沿用**。

**本轮真正的增量是两点**：

1. **从原始工件独立复核，而不是采信散文**。上面三处证据（`Searched | 1`、每折单条 `c0` 记录、
   该记录的 config 内容）是直接读出来的，与昨天的结论一致——这是应有的验证方式，
   也让"全网格从未跑过"这句话有了可复查的依据。
2. **一个结构性事实，昨天那份没有点出，而它对治理有直接影响**：
   **M2 的顺序前置守卫并不要求 M1 搜满网格。** `prior_families_from_reports` 只校验
   `family` / `mode=full` / `oos.all_passed` 三项，`mode` 本身只有 `"smoke"` 与 `"full"` 两个取值、
   与格数无关，驱动里也没有任何一处要求 `len(search_results) == len(grid)`。
   所以 `m1_parallel_v1` 这份**单配置**报告完全能让 M2 通过前置检查而发射。
   **含义**：「M2 的前置是 M1 已过门」这句话，并不隐含「M1 已经调过参」。
   要不要在 M1 全网格出结果之前就发 M2，是一个**实质的顺序决定**、而不是被守卫自动挡住的错误——
   因此它归所有者裁决（第 10 节第 1、3 项），不能靠"反正守卫会拦"来推定。

**数字本身**：M1 目前这个 +0.005845 是"用网格里最小的一格、在四折真实数据上"拿到的，
对 G4 门槛的余量只有 **+17%**。而 M0R（LightGBM，网格被钉死为单配置、
因此其 +0.009149 **不含验证集选择偏差**）高出它 **57%**。
两个数字可比性已核实（第 6 节），所以"目前最好的候选仍是 M0R"这个结论不变。

## 2. 全网格已发射，并且是健康的

2026-09-12 02:05 发射，命令即计划 §3.4 的 M1 发射包（新工作区 `m1_full_v1`、
`--parallel-workers 41 --torch-threads 1 --worker-memory-gb 2.5 --resume`、
不带 `--max-configs` 即全 27 格、不带 `--prior-report` 因为 `FAMILY_ORDER_PREREQUISITES["M1"] = ()`）。

发射即通过的检查（都是实测输出，不是推定）：

| 项 | 实测 |
|---|---|
| 内存护栏 | `41 x 2.5 GiB + parent 0.6 GiB = 103.1 GiB of 515.5 GiB available` → 通过 |
| 因子源 | `frozen_v2 (15 factors, tables=universe_daily,factor_daily,label_daily)` |
| F1 数据 | train 2,296,573 / validation 696,906 / test 804,575 行，全部缓存命中 |
| 线程钉住 | 检查点 manifest 里 `torch_intraop_threads: 1`、`torch_interop_threads: 1` |
| 并发占用 | `load average` 稳定 41.2–42.5，即 41 worker × 1 线程恰好占 41 核 |
| 全网格 | 日志遍历 `c0…c26` 共 27 条采纳检查；04:50 时检查点已覆盖到 `c22` |
| 错误 | 日志 0 条 traceback / FAILED / SystemExit |

04:50 快照：运行 2:44:49，`search/F1/` 已有 **4/27** 条记录，312 个检查点，
42 个子进程 RSS 均值 1.31 GiB（共 55.1 GiB），`MemAvailable 478.5 GiB`。

**预计 09-13 21:00–23:00 出 `report.json`**（推导见第 4 节，那里有四次修正的记录：
①②降级与 epoch 数、③**一轮的墙钟由该轮最慢的任务决定**（尾部因子实测 1.131，28.7 → 36.2 h）、
④**四折并不同价**——实测 train+val 行数为 F1/F2/F3/F4 = 1.000/1.110/1.220/1.321，
所以乘数是 **4.651 而不是 4**，36.2 h → **43.7 h ≈ 1.82 天**）。

## 3. 发射包的算术复核：它低估了 19%，因为它忽略了这个仓库自己早已实测的降级

`/tmp` 里那份 M1 发射包草稿把 41 路定价为 23.3 h，是按**零并发降级**算的。
但 `concurrency_scaling_benchmark.txt`（早已提交、就在计划 §3.4 里引用）实测
**1 路 1.00×、16 路 1.14×、48 路 1.30×**。乘上去是 27.8 h，即草稿**低估 19%**。

同一类错误在 M2 发射包里也有：`25.8 h/折` = 3 × 8.60 h，同样零降级，
修正后是 **6.58 天**而不是文档暗示的 5.8 天。

另外修掉三处会在发射第一秒就要命的小错：

* **两条发射命令都缺 `cd research_v2`**。驱动在 `research_v2/scripts/` 下，
  命令用的是相对路径 `scripts/run_real_candidate_training.py`；
  从仓库根粘贴会立刻死于 `can't open file`——**而且是在操作者已经读完 HMAC key、
  已经决定要跑一个多天任务之后**。
* **M2 的前置内存检查与它自己的命令行不自洽**：检查写 `MemAvailable >= 466.0 GiB`
  （= 24 × 16.56 + 21.99 的要求），但命令传的是 `--worker-memory-gb 16.6`
  （需要 ≥ 467.1 GiB）。落在 466.0–467.1 这 1.1 GiB 窗口里时，
  检查打印"24 workers"、护栏随即拒绝。
* 本机 `MemTotal` 是 **629.5 GiB**，文档写的 660 GiB（36 路不可达的结论不变、且更宽裕）。

轮数与粒度择优从此不再手算：文档表格里的 `gran` / 粗 / 细三列是**现场调用驱动自己的
`search_granularity` 与 `granularity_makespans`** 的返回值。这也顺手证实了一个比
"推荐 41 路"更硬的理由——**41 = `ceil(81/2)`，正是 M1 进入"种子级 2 轮"区间的最小 worker 数**；
40 路退回 3 轮（1.52 天），48 路仍是 2 轮但降级更高（1.18 天，比 41 路还慢）。

## 4. 三个家族的成本模型现在全部是实测的

此前只有 M1 的单 epoch 是实测的，M2 的 8.60 h/拟合与 M3 的 2.63 h/拟合都是从
2026-09-05 那次**旧 14 因子折结构**的基准折算来的。本次用只读探针
`/tmp/probe_newfold_cost.py`（日志 `/tmp/probe_static_newfold.log`、
`/tmp/probe_temporal_newfold.log`）在**当前 frozen_v2 15 因子的真实 F1 折**上把三个家族都测了。

**探针的保真度先验证过，再用它的数字**——这是本次最关键的方法论点：
探针跑 M1 config0 得到 checkpoint `sha256 = 85e9b4a35585865a…`，
而**生产运行 `m1_full_v1` 里 config0/seed 20260801 的 epoch-1 检查点 sha256 正是同一个值，逐位相同**；
时间上探针 361.88 s 也落在生产实测稳态区间 361.0–367.6 s 内。
另一格独立印证：M1 最大格探针 454.60 s 对此前 solo 实测 409.8 s = 1.109×，
config0 是 361.88 / 318.4 = 1.137×，两者都收敛到生产实测的 **1.132×**。

| 家族 / 格 | 参数量 | 争用下单 epoch | 折算 solo |
|---|---|---|---|
| M1 `64/1`（config0） | 10,497 | 361.88 s | 318.4 s（此前已测） |
| M1 `256/3` | 404,481 | 454.60 s | 409.8 s（此前已测） |
| M3 最大格 `128/4/64/2` | **418,433** | 486.26 s | **429.6 s** |
| M2 GRU/64 | 18,497 | 848.68 s | **749.7 s** |
| M2 GRU/128 | 61,569 | 1338.29 s | **1182.2 s** |
| M2 TCN/64 | 15,297 | 1039.33 s | **918.1 s** |
| M2 TCN/128 | 55,169 | 1897.07 s | **1675.9 s** |

M2 网格 24 格 = 2 家族 × 2 hidden × 3 dropout × 2 lr，成本只随家族与 hidden 变、
四个组合各占 6 格，所以**四格测全就等于测出网格均值 = 1131.5 s（18.86 min）**。

**参数量差 +256 是探针确实在 15 因子上的旁证**：M3 旧记 418,177 → 新测 418,433；
M2 TCN/128 旧记 54,913 → 新测 55,169。两处都是输入投影多一个因子。

**并发降级实测 1.132×（41 路，n=23，区间 1.100–1.155）**，
测法是取同一 (config, seed) 的 **epoch-2 与 epoch-1 检查点时间戳之差**，纯稳态。
**不能用 epoch-1 墙钟**：它含 worker 启动与 0.381 GiB 载荷反序列化（实测约 100 s），
会把比值虚抬到 1.43–1.51×，**足以误触发发射包里那条 1.5× 的杀线、把一个完全健康的运行杀掉**。
这一条对 M2 同样适用，已写进 M2 的遥测项。

**据此三个家族的工期**（M1 用实测 epoch 均值 21.9；M2/M3 的 epoch 数仍未实测）：

| 家族 | worker | 轮数 | 四折 | 备注 |
|---|---|---|---|---|
| M1（27 格） | 41 | 2（种子级） | **43.7 h ≈ 1.82 天** | 单次拟合墙钟已实测（均值 2.75 h、最慢 3.11 h）+ 尾部因子 1.131 + **四折行数比 4.651**（不是 4.000） |
| M2（24 格） | 24 | 3（配置级） | **104.9 / 127.4 / 151.0 h ≈ 4.4 / 5.3 / 6.3 天** | 按 18 / 21.9 / 26 epoch；降级仍未实测 |
| M3（16 格） | 48 | 1（种子级） | **22.8–24.6 h ≈ 0.95–1.02 天** | per-fit 按最大格，是上界 |

## 5. 三个被实测推翻或收窄的旧结论

**(a) 旧降级曲线对全尺寸折是高估的。** `1.06 + 0.005W` 在 41 路给 1.27×，实测 **1.132×**，
高估 12%。原因可理解：那条曲线是在 **40 个训练日的小帧探针**上测的，
小帧下每进程启动与访存占比更高。

**(b) "ms/日 可跨折迁移"不成立——我一度写进文档，随后自己撤回。**
最初只引 M1 最小格（旧 333.7 → 新 329 ms/日，−1.3%）就断言可迁移；
但 M1 最大格是**反方向**的（旧 388.2 → 新 423.8，**+9.2%**）。
M2 的四格测完后更清楚：GRU/128 旧 1301.3 → 新 solo 1240.5 ms/日（**−4.7%**），
TCN/128 旧 1518.7 → 新 solo 1758.6 ms/日（**+15.8%**）。
**同一次折结构变更，两个家族往相反方向走**，所以不存在可用的迁移规则。
两次的分母也确实不同（旧基准是 245 训练日 + 60 验证日的探针切片、按训练日归一）。
**教训写在这里：一个数据点不构成迁移规则，尤其是当它恰好支持你已经写下的结论时。**

**(c) M3 不是时序家族。** 计划 Task 4 的标题把 M2 与 M3 并列、开头又同时提到
`TemporalTrainingBatch` 与 `FeatureBatch` 却没说哪个家族走哪条，读起来像共用时序路径。
读代码的结果：`run_full_m3` 调的是 `_run_full_cross_section(family="M3", …)`，
与 `run_full_m1` **同一个函数**；`families.py` 里
`STATIC_TORCH_FAMILIES = {ResidualMLP, SetAttention}`、
`TEMPORAL_TORCH_FAMILIES = {TemporalGRU, TemporalTCN}` 只含 M2。
**所以 M3 的足迹属 M1 量级（实测 0.93→1.31 GiB/worker），不属 M2 量级（16.6 GiB）**，
也不受 M2 那个 24 路内存天花板约束：`safe_workers` 实测 **173–178**，远超 104 核。
M3 的正确工作点是 **48 路**（`48 = 16 × 3` 恰好一轮完美装箱），64 路一点不快。

## 6. 两项验收，都是这次全网格运行才第一次具备条件的

**验收 A：27 格下的"续跑报告逐字节一致"判据至今只在 1 格上验过。**
计划里记着该判据已实测——五份独立 M0R 全量跑的 `report.json` 同为
`834fb523f0177b3d47c5e53797e07d69`——但 **M0R 的网格是被钉死的单配置**，
`m1_parallel_v1` 也只搜了 `c0`。所以"由采纳记录按原网格顺序用同一个 `max` 重建
`best_config` 与 `search_results`"这句话，端到端只在 **1 条记录**上验证过。
而历史上真正咬人的那个 bug（`sort_keys=True` 使续跑报告键序不同、33 条单元测试全绿、
只有端到端能发现）正是这一类。

单元层不是裸奔：`_ordered_search_results` 有用 **2 格、乱序完成、含平局**的测试；
运行日志也已实证 `c0…c26` 各有互不相同的 config hash。但 2 格单元 ≠ 27 格端到端。
**做法**：`m1_full_v1` 出报告后，在**同一工作区**加 `--resume` 再跑一次，
预期四折整折采纳、零次拟合（因此很快），比对两份 `report.json` 的 md5。
不一致就是真 bug。**处置原则沿用已确立的那条：改代码去满足判据，而不是放宽判据。**

**另外做了一次静态审计，因为验收 A 要等报告落地、而多格规模的风险会在首折评价阶段（约 7 小时后）
就要兑现（2026-09-12 05:10）。结论：没有发现对配置数的硬编码假设。**逐处核对：
`_ordered_search_results` 用 `set(completed) != set(range(len(configs)))` 要求**每一格都有结果**、
缺格直接 `ValueError`（即宁可大声失败，也不会静默地从子集里选一个"最优"），
再按 `enumerate(configs)` 网格序遍历，并逐格校验 `result["config"] == dict(config_payload)`；
驱动侧 `best = max(search_results, key=…)`、`search_results[config_index]` 都是对列表的通用操作；
判定渲染器（`summarize_real_candidate_walk_forward.py`）用的是 `entry.searched_config_count`
这个计数字段、外层按 `REQUIRED_FOLDS` 迭代，与配置数无关；
而**评价阶段只操作被选中的那一个 bundle**，本身就与搜了几格无关。
所以验收 A 仍然要做（它是唯一能抓键序这类端到端问题的检查），
但"27 格会在评价阶段崩"这个具体担忧，静态层面没有支撑。
**顺带一条运维事实**：即便真在评价阶段崩，搜索记录是**逐配置**落盘的
（`persist` 在每个配置的三 seed 齐了就写，不是整轮结束才写——驱动里那段注释明确说
整轮落盘会让一个 worker 失败丢掉所有已完成配置），所以加 `--resume` 重启会采纳全部 27 格、
直接跳到最终拟合与评价，代价约 2.2 h 而不是整折的 7 h。

**验收 B：可比性是用内容寻址 id 核实的，不是假设的。**
`m1_full_v1` 的 F1 数据集 id
`226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538`
与基线证据 id `6fe33eda88430e5ee90e8d975c16f36c8d0de1a44a70eda594b12b82ff5678b9`
**与 `m1_parallel_v1` 逐字相同**；`m0r_v2` 与 `m1_parallel_v1` 同为
`mode=full`、4 折、`valid_day_count = 945`、`baseline_mean_rank_ic = 0.0852268662737014`（16 位全同）。
所以 Task 5 的横向比较是同基线、同日集、同折结构下的比较，可直接相减。
（`baseline_v2/report.json` 自身的 969 日 / 0.08706 是基线家族的全日集口径，
      候选报告里的 945 / 0.08523 是与所选 `best_simple` 的**交集日集**；G4 只用后者，两者不矛盾。）

**验收 C（本轮新增，而且是可证伪的）：若 c0 赢下全网格，G4 必须逐位复现 `+0.005844602701716051`。**
依据是实测的：`m1_full_v1/search/F1/c0-9ef4cccccd3fdf78.json` 与
`m1_parallel_v1/search/F1/c0-9ef4cccccd3fdf78.json` **文件名相同**（同一 config hash），
而**整个 `result` 字典完全相等**——集成值 `0.056057666322798204`、三个 seed 的验证集 RankIC
（`0.046977822604612057` / `0.052457092506752845` / `0.061353544679677595`）、
`worst_seed` 同为 20260801、config 逐项一致。
即 **41 路并行 + 钉 1 线程的全网格运行，与单配置载体运行在同一 (配置, 折, seed) 上给出
bit 级相同的结果**；这把 2026-09-11 交接 §4 那个在 3,553,163 行预测上实测的 `max|diff| 0.0`
从"预测层"延伸到了"搜索记录层"。
最终拟合同样是 (配置, 折, seed, 线程数) 的确定函数，OOS 又用同一批冻结基线工件，
所以 **c0 胜出 ⇒ G4 应当逐位回到 `+0.005844602701716051`**。
**若不是这个数，那不是抖动，是全网格路径在某处扰动了数值，要当 bug 查、不要四舍五入掉。**
这条比验收 A/B 更有用：A 与 B 只能回答"相同或不相同"，而 C 在报告出来**之前**就说出了应当是哪个数。
（截至 05:45，F1 已完成的 13 格里 c0 以 0.05606 领先次名 c2 的 0.05386 约 4.1%；
但 `hidden 256` 的 9 格最贵、尚未完成，领先者仍可能变。完整排名见
`docs/reports/2026-09-12-candidate-standings-comparison.md` §3(c)。）

## 7. M3 不会重演那六个阻塞项——核对过，不是推定

M1 两次真实运行都是四折训练跑完、然后死在评估阶段。六个阻塞项里
第 4 项（card family）打的是全部 M2、第 5 项（bundle config 字段）打的是 M2 的 GRU 与 TCN
**以及 M3**、第 6 项（loader 缺 `TEMPORAL_TCN` 条目与工厂分支）让 TCN 即使赢了也无法服务。
所以 M3 至少在两个轴上真的暴露过。核对结果：
`artifact_loader.py` 的 code-version 映射、config 映射、家族分派**三处都有 SetAttention**，
且都以模型类自己的 `model_family` 属性为键、不是手抄字面量；
`test_a_bundle_written_by_the_drivers_own_writers_loads_back` 的参数化覆盖
**M1 / M3 / M2-GRU / M2-TCN 四个载体全部**，用的是驱动自己 `torch_model_card` 与
`torch_bundle_config` 的真实输出；另有
`test_every_allowlisted_family_has_a_factory_behind_it` 专门钉住"映射里有、分派里没有"这类漂移。
**仍要如实说明**：这些是测试层保证，M3 从未在真实数据上跑过一折。

## 8. 一个看着像矛盾、其实不是的数字

探针构造时序面板只用 **72.5 s**，而 M2 发射包里每 worker 的面板重建是 **7.6 min**，差 6 倍。
两者测的不是同一件事：探针直接 `pd.read_parquet`（0.9 s，**不校验**），
worker 走的是 `DatasetArtifactStore.load_split(…, expected_fold_id=…)`——
**带分片哈希校验与折 id 绑定**，那才是 7.6 min 的主体。**7.6 min 不需要下修。**
探针其余量与生产逐项吻合，可作它忠实的旁证：`train_batches = 953 / val_batches = 237`
（文档早先记的是 953 / 237 / 242）。

同理，**探针里 TCN/128 峰值 RSS 16.12 GiB 不能用来上调 16.6 GiB 的声明值**：
探针没有做生产 `temporal_fold_batches` 的逐阶段 `del`（`9e8628a` 释放的那 1.24 GiB 死帧，
外加原始 train 分片与两个 prepared 帧），一直握着它们。
用 GRU 各格标定这个开销（探针 12.18 GiB 对生产 F1 实测 12.85 GiB）后，
TCN/128 折算到生产口径约 **12.8 GiB**，与生产实测吻合，再按 F4 行数放大就是文档里的 16.56 GiB。
**声明值站得住。** TCN/128 比 GRU 各格重约 4 GiB 是真的（模型与激活），已含在那条折算里。

## 9. 本轮提交

| commit | 内容 |
|---|---|
| `79e6845` | M1 全网格发射包入计划；makespan 低估 19% 的修正；两条命令补 `cd`；MemTotal 更正 |
| `7a4d7c8` | M2 同类降级错误；前置内存检查 466.0 → 467.1（与命令行的 16.6 自洽） |
| `13452fe` | 首折遥测：降级实测 1.132、RSS 实测、41 = `ceil(81/2)`；M3 发射包（此前只有勾选框）；撤回 ms/日 迁移主张 |
| `43e3e8a` | M3 的写入→加载覆盖是核对过的，不是推定的 |
| `75d2278` | 72.5 s 与 7.6 min 的口径澄清 |
| `e689eb3` | M2 四格实测成本模型；探针 RSS 的误报澄清；M1 实测 epoch 均值 21.9 |
| `e7b7537` | M3 工期随实测 epoch 数上调到约 1 天 |

**门禁：本轮实跑复核过，不是靠"只改了 markdown 所以仍然成立"推定的**（2026-09-12 05:03）：
`mypy src` → `Success: no issues found in 170 source files`；
`pytest tests -q` → **`2107 passed, 45 skipped, 5 warnings in 167.06s`**（与 `889aaad` 那次逐项相同）；
`ruff check src tests` 与 `ruff check scripts/run_real_candidate_training.py` → 均 `All checks passed!`。
一个坑记下来：`pytest --timeout=900` 会直接 `error: unrecognized arguments` 退出——
**本环境没装 `pytest-timeout`**，第一次跑因此 0 个测试就退出了，别把那次当成通过。
（另：`ruff check scripts` 会报 589 个错，全在因子侧一次性分析脚本里、以 E501/I001 为主，
**不在门禁范围**，本轮未触碰。）

## 10. 待所有者裁决的事项（本轮一个都没擅自决定）

1. **M1 全网格跑完后，哪一份 M1 报告作 M2/M3 的 `--prior-report`**：
   `m1_parallel_v1`（单配置载体、已归档、六道门全过）还是 `m1_full_v1`（全 27 格）。
   两者都满足守卫的三项要求，但代表不同搜索规模与不同证据强度。
2. **裁决 2（钉线程）**：M1 全网格已按 `--torch-threads 1` 跑，
   M2 与 M3 的发射包也都按钉 1 线程写。**并行运行时驱动强制要求显式 `--torch-threads`**
   （`workers > 1 and threads is None` 直接 SystemExit），而 M2 串行要 107 天、
   M3 串行约 19–21 天（192 次拟合 × 2.31–2.61 h），
   所以"钉线程"事实上没有替代方案——但它仍应被**记录为已裁决偏差**，而不是默认发生。
3. **是否发 M2**：约 4.4–6.3 天、24 路、需 `MemAvailable ≥ 467.1 GiB`（04:50 实测 478.5 GiB，够）。
   **更正（2026-09-12 05:40）：本项早先写的"与 M1 全网格互斥、只能串行"是错的，已用驱动自己的
   `worker_memory_budget` 配实时 MemAvailable 重算。** 那个结论把 M1 的**申报**需求 103.1 GiB
   与 M2 的 420.4 GiB 相加（523.5 GiB）去比**发射前**的 `0.9 × 515.9 = 464.3 GiB`，
   而 M1 的实际驻留（42 个子进程 54.4 GiB + 父进程 4.6 GiB = **59.0 GiB**）
   **已经反映在降低后的 MemAvailable 里**——等于把 M1 算了两遍。
   两个偏差还同向叠加：M1 每 worker 申报 2.5 GiB 而实测只有 1.33 GiB（1.9 倍保守）。
   实测（05:35，M1 正在跑）：`MemAvailable = 480.3 GiB`、可提交上限 `0.9 × = 432.3 GiB`，
   **M2 @24 路需 420.39 GiB → `fits=True`、`safe_workers = 24`、余量 +11.9 GiB**。
   即 **M2 现在就能与 M1 并发发射**，不必等约 38 小时。

   **但余量只有 2.8%，且有三条要如实说明的代价**，所以这仍然是所有者的决定、不是"既然能跑就跑"：

   * **护栏每折装配后会再查一次**，而 M1 的足迹在后三折会变大（F2/F3 比 F1 大）。
     若届时 MemAvailable 掉到 467.1 GiB 以下，M2 会在**中途某折**被拒——
     可用 `--resume` 恢复（搜索记录逐配置落盘），但代价是重启那一折。
   * **65 路下的降级未实测**：41（M1）+ 24（M2）= 65 核，未超 104 核、每 worker 仍各占一核，
     但 M1 那个 1.132× 是在**其余空闲**的宿主上测的；两个运行并发会推高访存争用，
     双方都会比各自的规划值慢，幅度未知。
   * **净收益要打折**：并发大约省 1.3–1.6 天（M2 提前约 38 小时开始），
     但 M1 会因争用推迟若干小时，且两个多天运行同时在线会让故障归因变难。

   **顺带查清了一个此前没人写下来的共享状态，以及一个真实但本次未触发的竞态。**
   数据集装配有一层**工作区之外**的共享缓存 `FOLD_CACHE_ROOT = /tmp/walk_forward_cache_v1`
   （`walk_forward_common.py:49`），缓存键是 `sha256(fold_id, role, versions)`
   ——**不含工作区、不含家族**，所以 M1 与 M2 用同一 `frozen_v2` 源时会命中**同一批条目**。
   写入是 `tmp_path = cache_path.with_suffix(".parquet.tmp")` → `to_parquet(tmp_path)` →
   `tmp_path.replace(cache_path)`：**rename 本身原子，但 tmp 名是确定的**，
   两个进程同时构建同一条目会写同一个 tmp 文件、内容交错，再由其中一方把损坏结果发布出去；
   读侧只在缺列时报 `cached dataset ... is corrupt`，**数据错但列全的情况是静默的**。
   **本次不会触发**，证据是硬的：缓存 46 个条目、最新的是 **9 月 8 日**，
   `find /tmp/walk_forward_cache_v1 -newermt "2026-09-12 00:00"` 返回 **0 个文件**，
   而 M1 日志里是 3 条 `loaded cached dataset for …`（读）对 **0 条** `cached dataset for … at …`（写）。
   即缓存对这四折**已经是热的**，M2 也只会读。**但这个竞态在代码里是真的**：
   冷缓存、或两条链用不同因子版本时就会活过来。
   修法很小（tmp 名加上 pid/uuid 即可），但**改的是 `walk_forward_common.py`、且要在无运行在飞时改**
   （第 11 节最后一条：spawn 会从磁盘重新导入，而 M1 还要跑约 38 小时），所以本轮**没有动**。

   **另一条与红线相关的核实**：`materialization_status.{json,md}` **只**由独立脚本
   `scripts/check_materialization_status.py` 写入（第 127 行），训练驱动与数据集装配路径都不碰它，
   所以发射 M2 不会触碰红线文件。
4. **是否现在晋级 M0R**（+0.009149，目前最好的通过候选），还是等 M1 全网格与 M2。
5. **81 路这个可选项**：按实测 RSS 1.27 GiB，81 路只要 102.9 GiB 内存、
   四折约 17.4 h（比 41 路省约 8.5 h）。本轮没有改用，理由：81 路降级未实测
   （而杀线是 1.5×，留不出诊断余量）、共享宿主上 81/104 核属于要点头的资源占用、
   且重启会丢掉在飞进度（搜索记录按配置落盘，未完成的配置救不回来）。

## 11. 红线（继承，仍然有效）

* **零 DB 写入**；不碰 `materialization_status.{json,md}`、`/tmp/frozen_chain_hold.txt`、2026 locked 段。
* 提交只加相关文件；`docs/research/` 保持未跟踪（属因子侧那个对话）。
  **这条我今天自己违反了一次，记下来防止重犯**：为了提交四份改动过的文档，
  我用了 `git add -A docs/`，它把 `docs/research/` 下三个属因子侧对话的文件
  （`2026-09-10-conditional-factor-idea-review-v0.1.{md,html,evidence.json}`）一并暂存，
  随 `05a485c` 提交并推送。已用 `git rm --cached -r docs/research/` 撤回
  （`12f7537`）：**文件在磁盘上完好、`git ls-files docs/research/` 为 0、
  `git status` 回到只有 `?? docs/research/`**，即恢复了应有状态。
  **没有改写已推送的历史**——分支已推送，改写是更具破坏性的动作；
  所以那三个文件的 blob 仍存在于 `05a485c` 那一个提交里。
  内容是因子想法评审及其证据 JSON 与渲染 HTML，**不含密钥**，
  所以这是范围越界而不是泄密，但它确实违反了一条明确约束。
  **根因是暂存路径太宽，不是对那些文件的判断**；本系列其它每个提交都是逐个文件加的。
  **规则：只 `git add` 明确要提交的路径，不要用 `git add -A <目录>`。**
  也**故意没有改 `.gitignore`**：`docs/research/` 是否应被永久忽略属因子侧对话的决定，
  不该由这边代为定。
* 推送：`https_proxy=http://127.0.0.1:7890 git push origin codex/publish-quant-platform`。
* HMAC key：`cat /data/zhangyudong/qrv2_research_artifacts/secrets/selection_evidence_hmac.key`
  （`QRV2_SELECTION_EVIDENCE_HMAC_KEY` 与 `QRV2_PREDICTION_EVIDENCE_HMAC_KEY` 同值）；
  registry root `QRV2_MODEL_REGISTRY_ROOT=/data/zhangyudong/qrv2_research_artifacts/model_registry`。
* Python：`/data/zhangyudong/.conda/envs/quantyd/bin/python`；测试从 `research_v2/` 跑。
* 负结果如实报告，不美化；治理决定归所有者。
* **运行在飞期间不要编辑驱动脚本 `research_v2/scripts/run_real_candidate_training.py`。**
  这条不显然、代价高、且失败方式很迷惑人，所以单独记：
  `run_parallel_search` 用 `ProcessPoolExecutor(mp_context=get_context("spawn"))`，
  而且这个 pool 是**每折新建**的（`with` 块在每次调用里）；spawn 的子进程会通过
  `multiprocessing.spawn._fixup_main_from_path` 用 `runpy` 以 `__mp_main__` 之名
  **按路径重新执行主脚本**。所以第 2/3/4 折（本轮约在 7 / 14 / 21 小时后）新建的 worker
  会读到**当时磁盘上的驱动**，不是发射时那份。
  入口有 `if __name__ == '__main__': main()` 守卫，所以重执行只跑模块级代码、不会重复训练；
  但**一个语法错误就足以让那一折的所有 worker 起不来**，表现为"跑了 7 小时、第 1 折完好、
  第 2 折突然全挂"，而 `git log` 上那次编辑看起来完全无害。
  本轮因此**推迟**了一个本来想做的驱动改动（见第 13 节第 5 项）。
  要改驱动，等报告落地；只改 markdown 与新增文档不受此限（本轮全部提交都属此类）。

## 12. 家族分派漂移这一类在 HEAD 上是封死的（地毯式审计，含一个被撤回的改动）

M2 的两个架构与 M3 的 `SetAttention` **都从未在真实数据上跑过一折**，而那六个阻塞项全部属于
同一类：某个按家族名分派的地方漏了一个家族。所以在 M2/M3 发射前把这一类整体审了一遍，
而不是等它们各自跑数天再撞。

**(a) 字面量地毯式扫描**：全仓 `src/` 与 `scripts/` 里出现的家族名字面量共 **13 处**，逐处归类后
**没有一处是 torch 家族名的手抄**：4 处是模型类自己的定义（`residual_mlp.py:34`、
`set_attention.py:84`、`temporal.py:32/119`，它们本身就是唯一真源），
9 处属 LightGBM 专属（`lgbm_artifact_loader.py` 的 `LGBMModelFamily` 与其两张映射、
`model_walk_forward.py` 的 `M0`/`M0R` 两行、驱动的 `M0_FAMILY_NAME`/`M0R_FAMILY_NAME`）。
torch 侧全部经由 `families.py` 从类属性派生：`STATIC_TORCH_FAMILIES`、
`TEMPORAL_TORCH_FAMILIES`、`TORCH_FAMILIES`、`TEMPORAL_CODE_VERSIONS`、`CARRIER_CARD_FAMILIES`。

**(b) 三个路由器 + loader 都有遍历派生集合的测试**（`tests/unit/models/test_families.py`）：
`test_every_family_router_knows_every_torch_family` 对每个静态家族断言它在
`builder._CROSS_SECTIONAL_FAMILIES`、`reproduction._STATIC_TORCH_FAMILIES`、
以及 `model_walk_forward._MODEL_CARD_FAMILIES[对应载体]` 里；对每个时序家族断言同样三处；
对**每个** torch 家族断言它在 `artifact_loader._MODEL_CODE_VERSIONS` 里。
`test_every_allowlisted_family_has_a_factory_behind_it` 则**遍历类声明的家族而不是遍历映射本身**，
并真的构造出来比对 `model_family`——它的 docstring 记着为什么必须这样：
把映射里的 `TEMPORAL_TCN` 删掉后，遍历映射的写法只是不再覆盖它、测试照样绿。
实跑：`test_families.py` + `test_family_batch_dispatch.py` + `test_reproduction.py` +
`test_artifact_loader.py` → **123 passed**（2026-09-12 05:15）。

**(c) 一个如实记录的负结果：我改了一处测试，变异检验后发现是多余的，已撤回。**
`test_family_batch_dispatch.py` 的时序侧用 `sorted(TEMPORAL_TORCH_FAMILIES)` 参数化、
静态侧却手抄 `("RESIDUAL_MLP", "SET_ATTENTION")`，看着正是交接第 10 节点名的那个反模式
（"夹具在复述写入方的约定"），于是把静态侧也改成参数化。**变异检验推翻了这个判断**：
往 `STATIC_TORCH_FAMILIES` 里加第三个家族后，参数化的三个用例**全过**——
因为 `builder._CROSS_SECTIONAL_FAMILIES = STATIC_TORCH_FAMILIES`，分派本来就派生自同一个集合，
新家族自动被路由；而真正会抓住"第三个架构没被路由"的是上面 (b) 那条测试，它早就存在、位置也更对。
所以那个改动不抓任何当前的东西，而它的 docstring 还会声称一份其实住在别处的保护——
**留下它是负价值，已用 `apply_patch` 逐字还原，`git diff` 为空**。
记在这里是为了不让下一个人重做一遍。
（顺带：`test_every_family_router_knows_every_torch_family` 里
`carrier = "M1" if family == ResidualMLP.model_family else "M3"` 是个 `else` 兜底，
真加第三个静态家族时它会被拿去和 M3 的白名单比从而失败——是"大声失败"，可以接受，不必改。）

## 13. 接手就能做的下一步


1. **等 `m1_full_v1/report.json`**（预计 09-13 21:00–23:00）。监控三项：
   `ls m1_full_v1/search/F1 | wc -l` 应到 27；`find m1_full_v1/checkpoints -maxdepth 1 -type d -name 'm1-f1-c*'`
   的最大编号应到 26；日志 0 错误。
2. **做验收 A**：同工作区 `--resume` 再跑一次，比对两份报告 md5（第 6 节）。
   同时做**验收 C**：若 `best_config` 是 c0，检查 G4 是否逐位等于 `+0.005844602701716051`（第 6 节）。
3. **做 §3.5 判定归档**，并把全网格的 G4 与 `m1_parallel_v1` 的 +0.005845、
   `m0r_v2` 的 +0.009149 并列——三者可比性已核实（第 6 节验收 B）。
4. **若发 M2**：先读 `MemAvailable`（需 ≥467.1 GiB；05:35 实测 480.3 GiB，够，余量 2.8%）。
   **不必等 M1 结束**（第 10 节第 3 项的更正：两者不互斥），但要接受那里的三条代价，
   首折用 **相邻 epoch 检查点时间戳之差**实测降级（不要用 epoch-1 墙钟），
   以及实测每格 epoch 数——这是 M2 工期 4.4–6.3 天这个区间的唯一来源。

   **发射就绪快照（2026-09-12 05:50 逐项只读核验，六项全过；但 `MemAvailable` 会变，
   所以真发射时必须重跑第 5 项，不能采信这个快照）**：

   | # | 前置 | 结果 |
   |---|---|---|
   | 1 | 工作区 `m2_v1` **必须不存在**（bundle 不可变） | `No such file or directory` ✓ |
   | 2 | `--prior-report M1=m1_parallel_v1/report.json` 满足守卫三项 | `family='M1'` ✓ `mode='full'` ✓ `oos.all_passed=True` ✓ |
   | 3 | 基线工件与报告 | `baseline_v2/baseline_oos/` 4 个 parquet ✓；`baseline_v2/report.json` 48,655 字节 ✓ |
   | 4 | HMAC key 与 registry root | key 64 字节 ✓；`model_registry/` 存在 ✓ |
   | 5 | 内存前置检查（已更正的 467.1 阈值） | `MemAvailable = 480.9 GiB -> 24 workers` ✓ |
   | 6 | 核数 | 104 核，M1 占 41、M2 需 24 → 65，不超订 ✓ |

   即 **M2 此刻就可发射**，不需要等 M1。仍属所有者裁决的理由见第 10 节第 3 项：
   余量只有 2.8%、护栏每折会再查一次、65 路下的降级未实测，
   且发射 M2 等于采用钉线程谱系（裁决 2）并选定用哪份 M1 报告作前置。
5. **一个被推迟的驱动改动（等 M1 报告落地后再做，理由见第 11 节最后一条）**：
   让 `prior_families_from_reports` 在**接受**一份前置报告时，把它实际搜了几格打印出来，
   例如 `M1 prerequisite satisfied by a report that searched 1 of 27 configs`。
   动机就是第 1 节那个结构性事实：守卫不要求网格搜满，所以一份载体验证级的 M1 报告
   能让 M2 静默通过前置、进而发射一个数天的运行。
   **这是纯可观测性、不改任何准入语义**（不新增拒绝条件），与本文件已有的先例同一风格——
   未声明 `--worker-memory-gb` 时也是"把数字打印进 transcript"而不是直接拒。
   **格数的取法（已核对字段，别写错）**：`report.json` 的 fold 条目里**没有** `searched_config_count`
   这个字段——它只有 `fold_id / dataset_id / model_id / best_config / search_results /
   baseline_selection_id / model_selection_id` 七个键。`searched_config_count` 是渲染器
   `summarize_real_candidate_walk_forward.py` 自己 dataclass 上的派生字段，
   第 139 行 `searched_config_count=len(search_results)`。
   所以守卫要读的是 **`len(report["folds"][i]["search_results"])`**，
   与 `m1_config_grid(None)` 的长度（27）对比。
   （顺带，这也是对第 1 节那个结论的第四处独立印证：`m1_parallel_v1/report.json` 里
   每折的 `search_results` 数组**长度就是 1**，唯一那项的 config 是
   `{hidden_dim:64, num_blocks:1, dropout:0.0}`——不必读 `verdict.md` 也能看出来。）
   做的时候要按本项目的纪律配测试，且**测试要向驱动索取真实输出、不要复述它的约定**
   （那正是六个阻塞项的结构性根因）。
6. **第二个被推迟的驱动改动：`--folds` 是完全失效的参数，应当改成"非默认值即拒绝"。**
   这条 2026-09-08 的勾选审计里就记过（"latent bug，已记待修"，并说"本次'单折 smoke'
   因此实际是全量 4 折运行"），**四天过去仍未修**，本轮复核确认它在 HEAD 上照旧：
   `grep -n "args\.folds\|\.folds\b" scripts/run_real_candidate_training.py` → **0 处匹配**。
   即 argparse 里有定义、`--help` 里承诺"comma-separated subset of ordinary folds"，
   而**驱动没有任何一处读它**；四处全量运行循环（第 339 / 1806 / 2292 / 2944 行）
   一律遍历硬编码的 `ORDINARY_FOLDS = ("F1","F2","F3","F4")`。
   **测试也一条都没有**（`grep '"--folds"' tests/unit/scripts/test_run_real_candidate_training.py` → 0 处）。

   **后果是花钱的那种**：传 `--folds F1` 想只跑一折省时间，实际会跑满四折——
   对 M2 是**约 6.2 天而不是预期的约 1.5 天**；传 `--folds F99` 这种非法值也被静默接受、
   同样跑满四折，所以它连"拼错会被挡住"这点保护都没有。

   **为什么正确的修法是拒绝、而不是实现它**：`summarize_real_candidate_walk_forward.py` 第 158 行
   `folds_complete = sorted(folds) == list(REQUIRED_FOLDS)`，而它是 `CandidateVerdict.satisfied`
   的合取项之一、直接决定 `WALK_FORWARD_PASSED` 标签。
   **所以任何折子集的运行都不可能产出通过的判定**——真去实现 `--folds`，
   得到的只是"必定 FAILED 的报告"，对候选评价毫无用处，反而多一条要维护的路径。
   **最小且方向正确的修法**：当 `--folds` 不等于默认的全集时 **SystemExit 拒绝**，
   理由写清"本驱动总是跑全部普通折；折子集永远无法满足 `folds_complete`，
   所以这个参数没有可用取值"。这把"静默的 4 倍成本意外"变成"第 0 秒的明确拒绝"，
   与内存护栏、`--prior-report` 守卫是同一套设计哲学；
   而且**默认路径逐字不变**（现有调用都不传或传默认值），所以对已归档的运行零影响。
   （备选是直接删掉这个参数，但那会破坏可能存在的调用脚本，不如拒绝温和。）
   配一条测试：传 `--folds F1` 必须 SystemExit 且理由里含 `folds_complete`；
   传默认值必须行为不变。**同样要等 M1 跑完再改**（第 11 节最后一条的 spawn 重导入约束）。

   **两条被推迟的改动放在一起看，有个过程问题值得记**：`--folds` 从 09-08 记到 09-12
   一直"已记待修"而没人动，而它今天仍是会让操作者白等四天的形状。
   **"记下来待修"在没有触发条件的情况下等于不修**。所以这里给它们绑一个明确触发点：
   **M1 报告落地之后、发射 M2 之前**，那个窗口本来就要停下来做验收 A/C 与 §3.5 归档，
   顺手把这两条驱动改动一起做掉（并各自配测试与变异检验）。

## 14. M2 的 `--worker-memory-gb 16.6` 从未覆盖训练期；实测后改为 17.1，而我第一版的结论是错的

**这一节更正第 10 节第 3 项里那个余量分析所依赖的申报值，并撤回我自己在过程中写下的两个错误结论。**

**问题的来源**：16.6 GiB 追溯到 2026-09-11 交接 §11 那张阶段表，而那张表的标题就是
「worker 峰值 RSS（**`temporal_fold_batches`**）」、最后一行是「validation 批次构造后 12.77 GiB」——
**它测的是批次构造期，不含训练期**。M2 至今一次都没真跑过（`m2_v1` 不存在），
所以**训练期的 worker 峰值从来没有被任何人测过**，而一个五天的运行要按它来定 worker 数。

**测法**：只读探针 `/tmp/probe_tcn128_worker_peak.py`（日志 `/tmp/probe_tcn128_peak.log`、
`/tmp/probe_tcn128_peak_F4.log`），调**真正的** `temporal_fold_batches`
（因此 `9e8628a` 的逐阶段 `del` 释放照常发生），逐阶段同时采样 `VmRSS` 与 `VmHWM`，
测最重的一格 TCN/`hidden 128`。保真度有对照：F1 批次构造峰 **12.99 GiB**
对交接记录的 **12.85 GiB**（差 1%），构造耗时 409 s 对记录的 7.6 min，批次数 953/237 逐项吻合。

| 折 | 批次构造后 VmHWM | 训练 1 epoch 后 VmHWM | 训练期增量 |
|---|---|---|---|
| F1 | 12.99 GiB | **16.29 GiB** | **+3.29 GiB** |
| **F4**（最大折，行数 1.288×F1） | **17.06 GiB** | **17.06 GiB** | **+0.00 GiB** |

**为什么 F1 涨 3.29 而 F4 涨 0.00**——这是本次最反直觉、也最有用的一条：
`VmHWM` 是生命周期高水位，而 **PyTorch 缓存分配器会复用构造期已经撑大的 arena**。
F1 的构造峰 12.99 GiB 装不下 TCN/128 的训练激活，arena 被迫长到 16.29；
F4 的构造峰已经 17.06 GiB，训练激活直接装进去，高水位一动不动。
**所以我最初的外推——把 F1 的 +3.29 按行数 1.288 放大后叠到 F4 的构造峰上、得到约 21 GiB——是错的，
错了约 4 GiB。** 这类"增量可加、可等比放大"的直觉在带缓存分配器的进程里不成立。

**两个被我写下又撤回的结论**（都记在这里，免得有人重新推一遍）：

1. **"F4 地板 17.06 已在悬崖另一侧，所以 24 路在诚实申报值下不成立"——错。**
   17.06 向上取整是 17.1，`24 × 17.1 + 21.99 = 432.39 GiB`，
   精确门槛 `432.39 / 0.9 = 480.4333 GiB`；实测 `MemAvailable = 483.2–483.6 GiB` →
   `worker_memory_budget` 返回 **`fits=True`、`safe_workers = 24`**。**台阶保得住**，
   只是门槛从 467.1 抬到 480.43（严 13.3 GiB）。
2. **"MemAvailable 488.5 时 `safe_workers = 23`、24 路仍会被拒"——算错了。**
   函数返回 `fits=True`、`safe_workers = 24`，门槛余量 8.1 GiB。

**低报并不构成 OOM 风险，这一点也要说清楚，别把这次写成一次险情**：
按实测峰值算 24 路的真实聚合需求是 `24 × 17.06 + 21.99 = 431.4 GiB`，
而护栏用 16.6 要求 `MemAvailable ≥ 467.1 GiB`——两者之差是 **35.7 GiB（真实需求的 8.3%）的真余量**。
**0.9 的提交系数把 2.7% 的低报吸收掉了**，所以改成 17.1 让护栏**更严**、不是更险。
**仍然成立的结构性事实**：`_check_worker_memory` 只实测父进程 RSS、worker 那份完全采信申报值，
所以护栏**原则上**发现不了低报；这次是安全系数兜住的，不是机制兜住的。
若哪天低报超过 11%（0.9 系数的余量），失败模式就会变成几天后 OOM 杀 worker。
**申报值就该是实测值，否则这个数字失去意义。**

**一个新踩的坑，与本文件上面记过的 466.0/467.1 不自洽同类**：
我把门槛四舍五入成 **480.4** 写进了前置检查，而精确值是 **480.4333**——
实测 `MemAvailable = 480.4` 时 24 路 **`fits=False`**（432.39 对 432.36，**差 0.03 GiB 失败**）。
于是检查会在 480.40–480.43 这段说"24 workers"而护栏随即拒绝，
**窗口从 1.1 GiB 缩到 0.03 GiB，但故障形状一模一样**。已改为 **480.5**。
**可一般化的教训：把门槛四舍五入到精确值的下方是不安全的，宁可向上取。**

**已落地的改动**（都在计划文档，命令与检查同步）：`--worker-memory-gb` **16.6 → 17.1**；
前置检查 awk **467.1 → 480.5**；并补了一张"申报值 17.1 下的实测门槛表"，
因为**真正的判据是发射那一刻的 `MemAvailable`，不是申报值本身**
（同一个 17.1：480.4 → 23 路、483.5 → 24 路、515 → 25 路）。

**对第 10 节第 3 项（是否与 M1 并发）的结论收紧**：不是"不能"，而是**余量薄到不值得**——
用 17.1 申报，并发要求 `MemAvailable ≥ 480.43`，而 M1 在跑时实测只有 480.9–483.6，
**余量 0.5–3.1 GiB（约 0.6%）**；护栏每折还会再查，而 M1 的足迹在 F2/F3
（行数比 F1 多 11% / 21%）会变大，很可能在某一折被拒。
M1 结束后约 515 GiB、余量 34.6 GiB。**所以按顺序跑：M2 在 M1 之后发射。**
