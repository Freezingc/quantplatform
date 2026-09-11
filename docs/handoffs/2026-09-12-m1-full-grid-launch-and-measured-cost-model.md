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

**预计 09-13 06:45 前后出 `report.json`**（推导见第 4 节）。

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
| M1（27 格） | 41 | 2（种子级） | **28.7 h ≈ 1.20 天** | epoch 数已实测（12 次拟合，均值 21.9） |
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
   **与 M1 全网格互斥**：M2 要 420.4 GiB、M1 要 103.1 GiB，合计 523.5 GiB 超过
   `0.9 × MemAvailable` 的可提交上限，后启动的那个会被自己的护栏拒。只能串行。
4. **是否现在晋级 M0R**（+0.009149，目前最好的通过候选），还是等 M1 全网格与 M2。
5. **81 路这个可选项**：按实测 RSS 1.27 GiB，81 路只要 102.9 GiB 内存、
   四折约 17.4 h（比 41 路省约 8.5 h）。本轮没有改用，理由：81 路降级未实测
   （而杀线是 1.5×，留不出诊断余量）、共享宿主上 81/104 核属于要点头的资源占用、
   且重启会丢掉在飞进度（搜索记录按配置落盘，未完成的配置救不回来）。

## 11. 红线（继承，仍然有效）

* **零 DB 写入**；不碰 `materialization_status.{json,md}`、`/tmp/frozen_chain_hold.txt`、2026 locked 段。
* 提交只加相关文件；`docs/research/` 保持未跟踪（属因子侧那个对话）。
* 推送：`https_proxy=http://127.0.0.1:7890 git push origin codex/publish-quant-platform`。
* HMAC key：`cat /data/zhangyudong/qrv2_research_artifacts/secrets/selection_evidence_hmac.key`
  （`QRV2_SELECTION_EVIDENCE_HMAC_KEY` 与 `QRV2_PREDICTION_EVIDENCE_HMAC_KEY` 同值）；
  registry root `QRV2_MODEL_REGISTRY_ROOT=/data/zhangyudong/qrv2_research_artifacts/model_registry`。
* Python：`/data/zhangyudong/.conda/envs/quantyd/bin/python`；测试从 `research_v2/` 跑。
* 负结果如实报告，不美化；治理决定归所有者。

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


1. **等 `m1_full_v1/report.json`**（预计 09-13 06:45）。监控三项：
   `ls m1_full_v1/search/F1 | wc -l` 应到 27；`find m1_full_v1/checkpoints -maxdepth 1 -type d -name 'm1-f1-c*'`
   的最大编号应到 26；日志 0 错误。
2. **做验收 A**：同工作区 `--resume` 再跑一次，比对两份报告 md5（第 6 节）。
3. **做 §3.5 判定归档**，并把全网格的 G4 与 `m1_parallel_v1` 的 +0.005845、
   `m0r_v2` 的 +0.009149 并列——三者可比性已核实（第 6 节验收 B）。
4. **若发 M2**：先读 `MemAvailable`（需 ≥467.1 GiB），确认 M1 已结束（互斥，第 10 节第 3 项），
   首折用 **相邻 epoch 检查点时间戳之差**实测降级（不要用 epoch-1 墙钟），
   以及实测每格 epoch 数——这是 M2 工期 4.4–6.3 天这个区间的唯一来源。
