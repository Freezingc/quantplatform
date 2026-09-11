# 2026-09-11：两条 M1 运行都死在评估阶段、四折完整的谱系对照、以及一次被实测否掉的预期

本交接覆盖 2026-09-11 00:31–19:10 的工作，六个提交，按时间顺序：`898ca4b`（四处「前向线程稳定」的勘误）、
`c016271`（F1 单折的谱系实测）、`2d58a7b`（时序批次构造重写）、`c64cbac`（M2 发射包按实测改写）、
`e1e9c09`（内容哈希流式化）、`b86cdb0`（评估器 card dataset_id 改成员归属，即第 1 节那次崩溃的修复）。

**第 3 节的四折谱系对照在数值上取代了 `c016271` 里的单折结果，第 5 节的足迹实测部分否掉了 `e1e9c09` 提交时
的预期收益——两处都以本交接为准。**

## 1. 最要紧的一条：两条真实 M1 运行都训练完了四折，然后在评估阶段崩掉，都没有判定

`m1_v2`（串行、未钉线程、约 44 小时）与 `m1_parallel_v1`（钉 1 线程、约 20 小时）都是同一个结局：F1–F4 全部
`complete: model=m1-fN`、四个 bundle 已冻结并签名、八个签名预测工件已入索引，然后在四折**之后**的评估阶段抛

```
ValueError: model bundle training dataset does not match frozen model selection
  evaluation/model_walk_forward.py:167  _require_verified_model_bundle
```

`report.json` 因此从未写出，两个 watcher 的审计与判定都失败了——**失败原因是「报告不存在」，不是复现失败**，
这两件事在转录里长得很像，不要混。

**根因是 card 里 `dataset_id` 有两种合法写法，而评估器只认其中一种。** 签名选择证据里
`training_dataset_id` 一律是 `<dataset>:train`（所有运行、所有家族都一样，已逐一核对 m0_v3 / m0r_v2 /
m1_v2 / m1_parallel_v1）。LightGBM 的 card 写 `<dataset>:train` → 严格相等成立 → M0/M0R 的评估通过并出了
判定；torch 的 card 写**裸** `<dataset>`（因为复现审计随后要用它去 `load_split(..., "test")`）→ 不相等 →
评估拒绝。**所以这不是某次运行的偶发问题，而是每一个 torch 家族都必然在训练完成之后、最贵的那一刻失败。**

**这件事代码库里早已裁决过，只是评估器没跟上。** 另外两处都按成员归属处理：

- `inference/reproduction.py:412 _checked_card_dataset_id`，文档字符串明写「两个写入方**有意**不一致……
  **两者都合法**，所以检查是成员归属而不是相等」，接受 `{report_dataset_id, report_dataset_id + ":train"}`；
- 驱动自己的 `_fold_adoption`（`run_real_candidate_training.py:1571`），`recorded_dataset not in
  (dataset_id, f"{dataset_id}:train")`。

**修法（`b86cdb0`）**：评估器接受本折签名选择所点名的三个 id —— `training_dataset_id`、`test_dataset_id`、
`test_dataset_id + ":train"`。**这是旧接受集的严格超集**，所以过去能过的一定还能过，没有任何既有测试改变
预期；而集合只由本折自己签名的证据构成，且 bundle 摘要在此之前已由 `load_model_card` 比对过，所以来自别的折
或别的运行的 bundle 仍然报名另一个哈希、仍然被拒。

**测试夹具本身就是这个 bug 能藏住的原因**：它把 `training_dataset_id` 写成 `"train-F1"`——一个没有任何写入方
会产出、且与该折真实 dataset id 无关的字符串，于是整套测试只跑过了一种写法并且全绿。夹具现在写
`f"{test_dataset_id}:train"`（与所有真实运行一致），并加了 `card_form` 参数，使两种约定与「无关数据集」三种
情况都被覆盖。新增两条测试：裸 id 的 bundle 必须通过；报名无关数据集的 bundle 必须**仍然**被同一条检查拒掉
——后一条是前一条的非恒真性守卫，放宽验证器时尤其需要它。

门禁：`ruff check src tests` 干净、`mypy src` 169 文件干净、`pytest tests` **2078 passed / 45 skipped**。

## 2. 恢复路径：并行运行可恢复，串行运行不可（原因是它早于搜索记录逐条落盘）

`_adopt_completed_fold` 的采纳要求里有一条是「每个配置的搜索记录都可采纳」。实测：

- `m1_parallel_v1` 有四条搜索记录（`search/F1..F4/c0-9ef4cccccd3fdf78.json`），`--resume` 重跑后
  **F1 已被采纳**：`F1 config 0: resumed from c0-9ef4cccccd3fdf78.json (ensemble validation RankIC=0.0561)` /
  `F1 adopted from verified artifacts: model=m1-f1`。四折采纳完即进入评估并写 `report.json`。
- `m1_v2` 的 `search/` 目录**为空**——它启动于 `142f8ea`（搜索记录逐条落盘）之前。采纳会失败，而
  `_adopt_completed_fold` 在失败时是**立即 SystemExit、不做任何拟合**（「an operator deserves the reason at
  the start of the run rather than hours into it」），所以试一次不会浪费算力，但也**恢复不出判定**。要它的判定
  只有两条路：新工作区重跑约 44 小时；或者只读取数（见下）。

**只读取数这条路是存在的，且不伪造任何证据。** 评估只需要 `ModelOosFold`（三个 id）加签名工件，而这些都在；
缺的只是 `report.json` 里那份由搜索记录重建的 `search_results` / `best_config`。**不建议事后补造搜索记录**
——那是伪造来源。可行的做法是直接调 `evaluate_model_walk_forward` 取 OOS 指标与六道门结果，只打印、不落盘、
不签发；它回答的是科学问题（M1@52 线程过不过门），产出的不是可归档的判定文档。

## 3. 四折完整的谱系对照（裁决 2 的输入，比 `c016271` 里的单折严重得多）

`m1_parallel_v1` 跑完 F4 之后，两条运行在**四折上全部可比**：同数据集工件、同配置
（`hidden_dim=64, num_blocks=1, dropout=0.0`）、同三 seed、同 15 因子、同 `residual-mlp-1`、同早停预算，
唯一受控差别是 `torch_intraop_threads`（串行 52 未钉 vs 并行 1 显式钉住）。数值取自两侧**签名的** model
选择证据的 `validation_rank_ic`：

| 折 | 串行 52 线程 | 并行 1 线程 | 差（并行−串行） | 占 G4(5e-3) |
|---|---|---|---|---|
| F1 | 0.05441212880837507 | 0.05605766632279820 | +1.645538e-03 | 32.9% |
| F2 | 0.09434474279284134 | 0.09030626306078236 | **−4.038480e-03** | **80.8%** |
| F3 | 0.08209953348378562 | 0.08220521556268619 | +1.056821e-04 | 2.1% |
| F4 | 0.09934948816614481 | 0.09967804374342573 | +3.285556e-04 | 6.6% |

四条性质，逐条都是实测：

1. **符号 +−++、带符号均值 −4.9e-04 ≈ 0**，所以谱系差不是可事后校正的系统偏移。
2. **平均 |差| = 1.53e-03（G4 的 30.6%），最大 4.04e-03（G4 的 80.8%，F2）**。
3. **四折之间差值的极差是 5.684e-03，已经超过 G4 门槛本身（5e-3）**——即谱系选择能造成的摆动比它要喂进的
   那道门还大。这是本节最该被记住的一个数。
4. 两条谱系**各自内部重拟合都是确定的**：串行与并行的 `c0` 搜索拟合与 `final` 最终拟合，三个 seed 的最佳值
   与最佳 epoch 都逐一相同（串行 6/8/14、并行 6/13/16，值到 16 位小数一致）。所以发散严格是**跨**谱系性质，
   与「同宿主两次全量跑逐字节相同」那条既有性质一致。

**对裁决 2 的含义**：钉线程的一侧不仅身份可移植，而且下面第 4 节证明它在任意宿主上可被精确审计；未钉的一侧
身份依赖宿主核数。但两侧在四折上的验证 RankIC 差别已达到门槛的 80.8%（单折）与超过门槛（极差），所以
**「选哪条谱系作为生产口径」会实质影响 M1 是否过门的结论**，不能当作形式选择。

## 4. 可移植性主张已被实测证明，不再是推论

`m1_parallel_v1` 的 F1（bundle 记录的 `torch_intraop_threads = 1`）在**默认 52 线程**的宿主上审计：

```
scoring under torch intra-op threads=1 (the run recorded 1; the host default was 52)
identical: true   max_absolute_difference: 0.0   row_count: 803572   exit=0
```

即「钉到 1 线程的谱系既身份可移植、又能在任意宿主上被精确审计」现在是证据。反向也仍然成立：`m1_v2`（记录 52）
自动钉到 52 → `exit=0`、`max|diff| 0.0`；钉到 1 → `exit=1`、差 4.923682914836913e-05，**那不是缺陷**。
另：`recorded_run_threads` 对 `m1_v2` 读出 52、对 `m1_parallel_v1` 读出 1、对 LightGBM 基线工作区返回 `None`
（回落宿主默认，因其 `num_threads` 早已钉死）。

## 5. 流式内容哈希：真实收益，以及一个被实测否掉的预期

`e1e9c09` 把 `build_dataset_manifest` 的规范 JSON 改为增量喂给 sha256，不再把整个数据集的每个单元变成 Python
对象、也不再构造出整段 JSON 文本。**在真实工件上逐位不变**：改后 `load_split` 对 `m1_v2` 的 F1 重算出的
四个 content_hash 与改前完全一致（test `28c12cd9a23d32c9`、train `19888596dbec7503`、validation
`8afda36f080df01e`、combined `226a8dfc497c2de6`，最后这个就是已发布的 dataset_id
`226a8dfc497c2de6b07516711f058f26a9270675916c9e67dbf7f087a07d2538`），而 store 只在完全相等时才接受。
改后复现审计重跑 `m1_v2` F1 仍是 `exit=0`、`max|diff| 0.0`、803,572 行。

| 量 | 改前 | 改后 |
|---|---|---|
| `load_split(train)` 峰值 RSS | 8.62 GiB | **4.28 GiB** |
| `load_split(train)` 耗时 | 189.7 s | **168.8 s（−11%）** |
| 其中 combined manifest 构建 | 185.7 s / +5.61 GiB | 165.5 s / +1.26 GiB |
| `temporal_fold_batches` worker 峰值 | 13.61 GiB | **13.62 GiB（无变化）** |
| `temporal_fold_batches` 全程 | 455.9 s | **400.1 s** |
| **父进程**峰值（折批次 + TEST 面板批次） | 19.15 GiB | **17.21 GiB** |
| 父进程 fold / test 批次耗时 | 446.0 s / 621.6 s | 399.6 s / 552.1 s |

**必须纠正一个预期（负面结果）**：改动前我推断它会把时序 worker 足迹从 13.61 GiB 再降到约 9 GiB、从而把
M2 从 24 路解锁到 36 路。**实测否掉了**——worker 峰值仍是 13.62 GiB。原因是 worker 峰值从来不由加载阶段
决定：加载峰值（8.62 GiB）本来就低于后面的 panel + 4.47 GiB 批次张量所形成的峰（13.6 GiB）。削掉的是一个
本来不是瓶颈的峰。父进程降了 1.94 GiB，是因为它要加载三次（TEST 面板路径），加载峰在它这里更接近总峰。

**所以 M2 的 worker 数仍是 24、仍是约 5.8 天，36 路没有解锁**，`c64cbac` 写进发射包的数字不需要改。流式哈希
的真实收益落在**以加载为峰值的那些路径**：静态家族（M1/M3）、复现审计、数据集构建，以及 11% 的加载时间。

**顺带实测否掉的另一个优化**：`_load_verified_snapshot` 里那次在 split 口径下用不到的 `combined` 全量拼接，
用 spy 包住 `pd.concat` 实测只占 **0.27 s / 0.54 GiB**（整个 `load_split(train)` 是 189.7 s / 8.63 GiB，
5 次 concat 合计 0.7 s）。所以「顺手删掉多余 concat」不是杠杆，没有做。

**下一个真杠杆的位置也随之变了**：worker 那 13.6 GiB 里，4.47 GiB 是批次张量（不做惰性批次就无法降），
其余是帧与分配器开销；内容哈希已经不在峰值里了。要继续降足迹，方向是惰性/分批构造批次，或帧的 dtype
（数据集以 `date32[day]` 与 `string` 存储，读进 pandas 后是 object 列，即逐单元 Python 对象），
两者都比本次的改动大，且都要求「输出逐位不变」这条同样的纪律。

## 6. 采纳检查已上提到构造折输入之前（`554b3d0`）

第 1 节那三次恢复重跑把这条从"看起来划算"变成了"实测在付费"：`_run_full_cross_section` 与 `run_full_m2` 都在
调用 `_adopt_completed_fold` **之前**就先 `FoldPreprocessor.fit(train)` 并构造了 train 与 validation 两套批次
（M2 还多一套 TEST 面板），于是一个**已经冻结、马上就要被采纳跳过**的折，仍要先付出这些工作再由 `continue`
全部丢弃。实测 F1：M1 约 6 分钟/折；M2 的三套面板是 400 s + 552 s，约 16 分钟/折。而 `--resume` 是多日运行
**计划内**的跨折手段，不是意外——今天一天就走了三次。

采纳需要的全部输入（`model_id`、三个 dataset id、`factor_order`、网格、预算）在预处理拟合之前就都已知，基线证据
的写入更在前面，M2 绑定的窗口契约来自 `_temporal_search_budget` 读的冻结常量而不是任何面板，所以没有任何东西
强制原来的顺序。LightGBM 那条循环本来就把采纳放在基线证据之后、构造之前——**是两条 torch 循环没对齐**，现在三条
一致了。改的是位置不是内容：同样的调用、同样的顺序、同样的参数；数 3 个采纳点的既有 wiring 测试仍然数到 3。新增
的是一条钉住**顺序本身**的测试（把驱动按顶层函数切开，要求在两条 torch 循环里采纳调用先于 `preprocessor.fit(`，
并断言被检查的恰好是这两条，以免它哪天悄悄什么都不覆盖了）；它在改动前以 3406 对 2378 的偏移失败。

上提之后再跑恢复，每折从约 6 分钟降到约 4.5 分钟：省掉的是预处理拟合与批次构造，剩下的主要是 `extract` 与
`publish_fold_dataset` 的内容哈希校验（约 170 s），而那一部分采纳确实需要——`dataset_id` 是采纳要比对的东西。

## 7. 评估路径上一共是**三个**阻塞点，不是一个；三个都属于同一类

第 1 节写的是第一个。修掉它之后评估往前走了一步，又停在第二个；再修，又停在第三个。**同一个根因重复了三次：
评估路径把 LightGBM 载体的行集/命名约定当成了普适规则，而此前没有任何 torch 家族走到过这么远**，所以这三条
都没有被任何测试或任何真实运行触碰过。

| # | 位置 | 症状 | 真实成因（均为实测） | 修法 | 提交 |
|---|---|---|---|---|---|
| 1 | `model_walk_forward._require_verified_model_bundle` | `model bundle training dataset does not match frozen model selection` | card 的 `dataset_id` 有两种合法写法：LightGBM 记 `<dataset>:train`，torch 记裸 `<dataset>`（复现审计随后要用它加载 TEST 分片）。评估器做严格相等，只认前者 | 改为成员归属：接受本折签名选择点名的 `training_dataset_id` / `test_dataset_id` / `test_dataset_id + ":train"`。**是旧接受集的严格超集** | `b86cdb0` |
| 2 | `walk_forward._assemble_oos_frames` | `F1 candidate predictions and labels do not contain identical business keys` | LightGBM 候选覆盖整个 TEST 截面（F1 为 804,575 行），torch 只覆盖有可用标签的行（**803,572**，差的 1,003 行是 `label_valid=False`）。该检查要求候选的键与**未过滤**的标签帧完全相同 | 改为双向覆盖：候选不得有 TEST 帧之外的键，且不得漏掉任何**可打分**行；`_scoreable_label_keys` 照抄指标自己的有效性表达式 | `6a19ef2` |
| 3 | `walk_forward._require_baseline_covered_by_candidate` | `F1 candidate and baseline predictions: baseline rows are not all covered by the candidate` | 冻结的 best_simple 基线覆盖 747,461 行，torch 候选覆盖 803,572 行，基线里候选没有的那 **940 行中可打分的行数是 0**；指标的可用交集两种情况下都是 **746,521 行** | 覆盖要求收窄到**可打分**行；同时把"基线不得有 TEST 帧之外的键"显式写出（它原先只是恒等检查的副作用，收窄后不再被隐含保证） | `a950e42` |

三条都各自带了非恒真性守卫测试，因为放宽验证器正是最需要这种守卫的改动：第 1 条配"报名无关数据集的 bundle 仍被
拒"，第 2 条配"漏掉一个可打分行仍被拒"与"候选出现 TEST 帧之外的键仍被拒"，第 3 条配"基线出现 TEST 帧之外的键
仍被拒"，并且被接受的两种情形都断言 `delta_rank_ic` 仍为 2.0（即数值没有被动过）。

另外两条值得记下的连带事实：

- **测试夹具本身就是这三个 bug 能藏住的原因。** `test_model_walk_forward.py` 的夹具把 `training_dataset_id` 写成
  `"train-F1"`（没有任何写入方会产出的字符串），并且所有 TEST 行都 `label_valid=True`、候选与基线行集完全相同。
  于是整套测试只跑过一种 card 写法、一种行集约定，并且全绿。夹具现在两者都参数化了。
- **`m1_v2` 的 `--resume` 是安全失败的**：它在**任何拟合之前**就 SystemExit，理由是
  `config 0 has no adoptable search record (no record at c0-9ef4cccccd3fdf78.json)`——该运行早于 `142f8ea`
  （搜索记录逐条落盘）。所以试它不花算力，但也恢复不出判定。
