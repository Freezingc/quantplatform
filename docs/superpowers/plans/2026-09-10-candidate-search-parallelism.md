# Candidate Search Parallelism Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run independent M1/M3 validation configuration searches in spawned processes while preserving exact selection and evidence contracts.

**Architecture:** A reusable process scheduler accepts index-addressed work and returns index-addressed results in canonical order. The candidate driver uses it only for M1/M3 after explicitly pinning Torch threads; serial search, final fits, artifacts, and OOS evaluation remain unchanged.

**Tech Stack:** Python 3.11, `concurrent.futures.ProcessPoolExecutor`, PyTorch CPU, pandas, pytest.

**Spec:** `docs/superpowers/specs/2026-09-10-candidate-search-parallelism-design.md`

## Global Constraints

- Scope is M1/M3 cross-sectional config search only.
- `--parallel-workers` defaults to `1`; this legacy path does not configure Torch threads.
- `--parallel-workers > 1` requires a positive `--torch-threads` value and uses a spawned context.
- `SearchResume.persist` is the sole durable config-result writer.
- Report ordering is increasing config index, never worker completion order.
- Final fits, bundle/evidence/prediction writes, and G1-G6 aggregation stay in the parent.

---

### Task 1: Declare the execution contract

**Files:**

- Modify: `research_v2/src/research_v2/training/seeds.py`
- Modify: `research_v2/scripts/run_real_candidate_training.py`
- Test: `research_v2/tests/unit/training/test_reproducibility_seeds.py`
- Test: `research_v2/tests/unit/scripts/test_run_real_candidate_training.py`

**Interfaces:**

- `configure_torch_threads(thread_count: int) -> None` rejects zero and configures both Torch thread pools.
- `_parallel_execution(args) -> ParallelExecution` parses/validates explicit opt-in settings.

- [x] Write failing tests that a parallel run without `--torch-threads` is rejected and a zero thread count is invalid.
- [x] Run the two targeted test files and confirm the APIs are absent.
- [x] Add the CLI flags, validation helper, and thread setter.
- [x] Re-run the targeted tests (`70 passed`) and commit with `feat(training): declare parallel torch execution settings` (`05d3922`).

### Task 2: Add an ordered spawned scheduler

**Files:**

- Create: `research_v2/src/research_v2/training/parallel_search.py`
- Test: `research_v2/tests/unit/training/test_parallel_search.py`

**Interfaces:**

- `SearchTask(index: int, payload: Mapping[str, object])` binds every result to one grid position.
- `run_parallel_search(tasks, workers, run_task) -> list[tuple[int, dict[str, object]]]` uses `multiprocessing.get_context("spawn")`, caps workers to task count, propagates failures, and returns index order.

- [x] Write failing tests for reverse completion ordering, worker-cap calculation, and worker exception propagation.
- [x] Run `python -m pytest tests/unit/training/test_parallel_search.py -q` and confirm the missing-module failure.
- [x] Implement the scheduler with `ProcessPoolExecutor` and a spawned context.
- [x] Re-run the scheduler tests (`3 passed`) and commit with `feat(training): add ordered spawned config scheduler` (`0f77b95`).

### Task 3: Route M1/M3 search trials through the scheduler

**Files:**

- Modify: `research_v2/scripts/run_real_candidate_training.py`
- Test: `research_v2/tests/unit/scripts/test_run_real_candidate_training.py`

**Interfaces:**

- `CrossSectionConfigTask` contains all three-seed search inputs and a config/seed-unique checkpoint prefix.
- `run_cross_section_config_task(payload) -> dict[str, object]` is package-level and writes no bundle, selection evidence, prediction, or report.
- `_ordered_search_results(configs, completed) -> list[dict[str, Any]]` restores grid order before best-config selection.

- [x] Write a failing test that reverse-order scheduler results become increasing-index search results.
- [x] Run that test and confirm the ordering helper is absent.
- [x] Extract the existing three-seed config body into the worker and call the scheduler only for M1/M3 when workers exceed one. Preserve the serial loop for one worker.
- [x] Run the complete candidate-driver suite (`69 passed`, including a real spawned three-seed worker) and commit with `feat(training): parallelize cross-sectional config search` (`77e8401`).

### Task 4: Verify the supported surface

**Files:**

- Modify: this plan to check completed steps and record exact commands/results.

- [x] Run `python -m pytest tests/unit/training/test_reproducibility_seeds.py tests/unit/training/test_parallel_search.py tests/unit/training/test_search_resume.py tests/unit/scripts/test_run_real_candidate_training.py -q` (`108 passed`).
- [x] Run `python -m pytest tests/unit/evaluation -q` (`72 passed`; two pre-existing pandas FutureWarnings).
- [x] Run `ruff check src tests scripts/run_real_candidate_training.py` (`All checks passed!`).
- [x] Run `git diff --check`, record exact results, and commit with `docs(training): record parallel search verification`.

### Task 5: Route the M2 temporal search through the same scheduler (2026-09-10 increment)

**Why:** Tasks 1-4 left M2 serial because of its temporal-batch contract. M2 is also the family
that cannot afford to stay serial -- measured ~8.6 h per fit and ~161 days for the family, so it is
the one search that will never finish without concurrency. The blocker was never the scheduler
(index-addressed, family-agnostic); it was the payload. A temporal batch holds a 20-day window per
row and the validation panel is train+validation concatenated, so per-task batches would be
gigabytes each.

**Files:**

- Add: `research_v2/src/research_v2/training/search_metrics.py`, `research_v2/src/research_v2/training/temporal_search.py`, `research_v2/tests/unit/training/test_temporal_search.py`.
- Modify: `research_v2/src/research_v2/training/cross_section_search.py`, `research_v2/scripts/run_real_candidate_training.py`, `research_v2/tests/unit/scripts/test_run_real_candidate_training.py`, the design spec.

**Interfaces:**

- `search_metrics.validation_rank_ic(predictions, labels) -> float` -- the number the winning config is selected by, in one place. It had two copies (the driver's serial loops and the M1/M3 worker); three would have been three chances for "the model the search selected" to mean something different from "the model the parent fits".
- `temporal_search.TemporalFoldSpec / TemporalFoldBatches / TemporalConfigTask`, `temporal_fold_batches(spec)` (cached per worker), `temporal_panel_batches`, `temporal_model`, `run_temporal_config_task(payload)`.
- `HISTORY_LENGTH = 20` / `MINIMUM_VALID_HISTORY_LENGTH = 15` -- the window contract, imported by the driver's serial `_temporal_panel_batches` too.
- `_temporal_search_budget(max_epochs)` -- M2's record budget: fit length **plus the window**, because a 20-day and a 10-day window are different models and neither is in the config payload.

**Steps:**

- [x] Write failing tests: the worker rebuilds bitwise the batches the parent would build (against a real published dataset artifact), the validation window reaches back into train (day 21, the split's first day, has a batch only if the panel was concatenated -- and day 15 as the first eligible train date pins that the 15-day minimum counts the signal date itself), one fold is built once per worker, the task payload stays under a 100 KB budget, GRU/TCN follow the config, and a real spawned worker returns a complete three-seed result.
- [x] Run them and confirm the missing-module failure.
- [x] Implement `search_metrics` and `temporal_search`; point `cross_section_search` at the shared metric and delete its private copy.
- [x] Route the M2 search through `run_parallel_search` when `--parallel-workers > 1`, adopting and persisting search records exactly as the M1/M3 parallel path does, and restore grid order through the existing `_ordered_search_results` before the existing `max` selection.
- [x] Bind the window contract into M2's budget and make the driver's serial panel builder read the same constants, so serial loop, spawned worker and records cannot disagree about the window.
- [x] Allow M2 in `_parallel_execution`; keep M0/M0R refused, with the reason in the message.
- [x] Mutation-check: hardcoding the driver's window back to a literal, and dropping `history_length` from the budget, each kill exactly the test that pins them.
- [x] Full gate: `ruff check src tests` clean, `mypy src` **169 files** clean, `pytest tests -q` **1997 passed / 45 skipped**.

**Still open after Task 5:** the FINAL three-seed fit was still serial, and on a fold whose grid
already saturates the workers it was the only serial part left -- three fits out of a 24-config M2
fold's ~52 hours, i.e. half the fold at M2's measured ~8.6 h per fit. Taken by Task 6 below.

### Task 6: Parallelise the final three-seed fit (2026-09-10 increment)

**Why:** config-level parallelism accelerates the search only. The three final fits that produce the
frozen bundle ran serially afterwards in every family, so a fold whose 24-config grid already filled
the workers still paid three full fit-times at the end. They are independent by construction -- same
config, three seeds, `REQUIRED_SEEDS` is a fail-closed contract -- and the parent does not need the
models, only their checkpoint paths and their validation scoring.

**Interfaces:**

- `cross_section_search.CrossSectionSeedTask` / `run_cross_section_seed_task`, `temporal_search.TemporalSeedTask` / `run_temporal_seed_task`, both returning `{seed, checkpoint_path, predictions, validation_rank_ic}`.
- `_fit_one_seed` / `_fit_one_temporal_seed`: one seeded fit, defined ONCE and used by both the config-level worker (which loops the three seeds itself) and the seed-level worker (which the parent loops across processes). Returning the checkpoint path is what lets the parent freeze a bundle from a fit it never ran.

**Steps:**

- [x] Write failing tests: a spawned seed worker hands back a checkpoint that exists on disk under the same `<prefix>-seed-<seed>-epoch-<n>` name the serial path produces, and three spawned seed fits reassemble through `build_seed_ensemble` into a real ensemble with three distinct checkpoints -- for the static families and for M2.
- [x] Refactor both config-level workers onto the shared single-seed fit; their existing spawned-worker tests stay green unchanged, which is the evidence the refactor preserved behaviour.
- [x] Route the final fits of `_run_full_cross_section` (M1/M3) and `run_full_m2` through `run_parallel_search` when `--parallel-workers > 1`, indexed by seed; the serial path stays the default and keeps the identical checkpoint prefixes, so a bundle's provenance cannot depend on the worker count.
- [x] Mutation-check: renaming the parallel branch's checkpoint prefix kills exactly the test that pins serial/parallel prefix equality.
- [x] Full gate: `ruff check src tests` clean, `mypy src` 169 files clean, `pytest tests -q` **2004 passed / 45 skipped**.

**Taken by Task 7 below:** seed-level granularity for the SEARCH.

### Task 7: Seed-level search granularity for narrow grids (2026-09-10 increment)

**Why:** with `--max-configs 1` -- the shape of every zero-tuning comparison run, and what M0R's
+0.009149 was measured at -- there is one config per fold, so `capped_worker_count` returned 1 and
the search stayed serial even with `--parallel-workers 48`. Task 6 parallelised the final fits, which
took such a run from six serial fits per fold to four; splitting the search's unit of work to
(config, seed) takes it to two parallel rounds, i.e. **3x**. At the measured ~2.05 h per fit
(m1_v2: 18 fits in 37 h) that is 8 fit-times per run instead of 24 -- **~16 h instead of ~49 h** --
which is what makes an end-to-end validation of the parallel path one overnight run rather than two
days. Three workers is the ceiling here: a one-config fold holds only three seeds, so the remaining
lever for this run shape is fold-level parallelism, not more workers.

**Rule:** per fold, adopt what the search records cover, then use config-level tasks when the
remaining configs can fill the workers (`len(unadopted) >= workers`) and seed-level tasks otherwise.
Config-level keeps the smaller return payload (scalars only); seed-level returns each seed's
validation predictions so the parent can build the ensemble, which is affordable precisely because
it is chosen only when there are few configs. Both write the same records and the same checkpoint
names, so a workspace does not record which granularity produced it.

**Interfaces:**

- `_ensemble_from_seed_fits(fits, config_payload, validation_labels) -> dict` -- rebuilds one config's
  search result from seed fits that ran in other processes, in the serial loop's exact shape, and
  refuses a seed set that is not exactly `REQUIRED_SEEDS`.
- `_constant_budget(budget)` -- a named factory for the torch families' config-independent budget,
  replacing a lambda whose type mypy could not infer at a binding site.

**Steps:**

- [x] Write failing tests for `_ensemble_from_seed_fits`: the result shape and its four keys, `worst_seed`
      is the lowest-RankIC seed, an incomplete seed set is refused, an unknown seed is refused.
- [x] Implement the seed-level branch in both torch search loops, indexed `config_index * len(REQUIRED_SEEDS) + slot`
      and decoded by integer division, so grid order is restored by the existing `_ordered_search_results`.
- [x] Mutation-check: weakening the seed-set guard to a subset test turns the refusal into a downstream
      `KeyError`, and the incomplete-seed test fails.
- [x] Clean the driver's mypy output while in it: `MYPYPATH=src mypy scripts/run_real_candidate_training.py`
      now reports **zero errors in the driver** (was 13 -- six from today's parallel wiring, six pre-existing
      `ModelWalkForwardResult` name reuse, plus the third-party `dolphindb` stub note which is not ours).
      The gate is `mypy src`, which never covered `scripts/`, so none of these were being caught.
- [x] Full gate: `ruff check src tests` clean, `mypy src` 169 files clean, `pytest tests -q`
      **2008 passed / 45 skipped**.

**End-to-end validation launched 2026-09-10 21:52.** The library level had real spawned workers
(static and temporal, config-level and seed-level, including three-seed reassembly through
`build_seed_ensemble`) and the driver level had source-wiring assertions, but neither is a run. So a
full real-data M1 run was started on the parallel path: `--family M1 --parallel-workers 3
--torch-threads 1 --factor-source frozen_v2 --max-configs 1`, workspace `m1_parallel_v1`, with
`QRV2_MODEL_REGISTRY_ROOT` pointed at a scratch directory under `/tmp` so nothing it produces can
reach the real registry, and the serial `m1_v2` run in flight left untouched. Three spawned workers
were observed training F1's three search seeds concurrently within six minutes of launch.

What that run does and does not settle. It validates the machinery end to end on real data. It also
starts a **new numeric lineage** -- threads pinned to 1 and recorded in every search record's
runtime descriptor, which is the substance of adjudication 2 in the compute brief -- so its numbers
are comparable to themselves and NOT to `m1_v2`'s, and promoting anything from it still needs that
adjudication recorded. The serial run remains the one whose verdict answers the M1-vs-M0R question.

## 实测记录（2026-09-10 22:00–22:30，`m1_parallel_v1` 与串行 `m1_v2` 同时在跑）

**1. 线程数对速度中性，在真实 v2 数据上复现。** 串行运行（52 线程）相邻 epoch 检查点间隔 5m37s
（21:08:28 → 21:14:05 → 21:19:41）；并行运行的三个 worker（各 1 线程、三路并发）epoch-1 检查点落在
22:04:08 / 22:04:38 / 22:06:41，即单 epoch 约 4–6 分钟。两者同量级，所以 3× 加速不是靠降线程换来的，
也没有被降线程吃掉。计划 §3.4 早前的基准是在旧的小折上测的（1/4/8 线程 81.56–81.80 s），这是第一次
在 230 万行的 v2 折上验证同一条结论。

**2. 种子级并行在真实数据上跑通。** 启动 6 分钟内观测到三个 spawned worker 并发训练 F1 的三个搜索 seed，
各自写出自己的检查点（`m1-f1-c0-<seed>-seed-<seed>-epoch-N`），互不覆盖。

**2b. 三路并发没有逐拟合速度：零降级（2026-09-10 23:05，直接比对检查点时间戳）。** 同一个折 F1、
同一个 seed 20260801、同一个配置：并行运行（3 个 worker、各 1 线程）相邻改进检查点间隔为 5m31s / 5m32s /
5m33s / 5m33s（epoch 4→6 为两个 epoch）；串行运行（52 线程）在同一折同一 seed 上为 6m / 5m / 6m /
5m（epoch 4→6 同为两个 epoch）。两者同为约 **5.5 分钟/epoch**，即 **3 路并发、每路 1 线程的逐步拟合
降级约为 1.00×**。这条与计划 §3.4 早在旧的小折上标定的 16 路 1.14×、48 路 1.30×）一致，
并且是 M2 敢开到内存约束上限（20 路）而不只是"看起来能开"的前提——若若逐步拟合随并发度显著变慢，
20 路的收益就要被降级吃掉一部分。而 M2 单折面板更大（约 21.6 GiB）、40 分钟重建），其降级需要在更高并发下
仍**未实测**，应在 M2 首跑前用小规模实测标定，而不是直接按静态家族的曲线外推。

**3. 钉线程确实生效，并且写进了证据工件。** worker 的 checkpoint manifest 记录
`torch_intraop_threads: 1`、`torch_interop_threads: 1`；串行运行同一位置记录 52。

**4. 两个谱系的差异被量化了——这是裁决 2 的关键输入。** 同一折、同一配置、同一 seed、同一 epoch：

| 谱系 | epoch-1 验证 RankIC | checkpoint sha256（前 12 位） |
|---|---|---|
| 并行，threads=1 | 0.049795006597773174 | `85e9b4a35585` |
| 串行，threads=52 | 0.04979499815492084 | `f1205743b37a` |

指标相差 **8.4e-9**（G4 门槛 5e-3，小 6 个数量级），而**权重摘要完全不同**。三条含义：科学结论上两者
等价；工件上两者不可互换（派生 model_id 不同，跨谱系不存在逐位复现）；所以搜索记录绑定线程数是必要的
而非过度谨慎。反过来也说明钉线程的代价是**一次性且单向的**：一旦选定某个值，此后所有运行都必须用同一
个值，否则「两次全量跑逐字节相同」这条既有性质（`m0r_v2`/`m0r_v3`/三次 M0R 复验共 5 份同 md5 报告）
就不再成立。

**5. 折级并行不是下一个杠杆（修正 Task 7 末尾的说法）。** 存储层已核实可并发写：选择证据与冻结预测都是
**内容寻址 + `tempfile.mkdtemp` 后 `rename` 原子落盘**，不同折写的是互不相交的 id，所以并发写在存储层是
安全的。但吞吐上不成立：总工作量与 worker 数都不变，配置级并行已经吃满机器时再按折拆分不会更快——24 配置
× 3 seed = 72 次拟合，无论切成 24 个配置任务还是 72 个种子任务交给 24 个 worker，都是 3 轮。折级并行只对
**窄网格**有意义（1 配置时一折只有 3 个 seed，3 个 worker 就是天花板，四折仍一个接一个），而窄网格是载体
对比运行的形状，不是 M2/M3 正式运行的形状。因此 M2 时长的真正杠杆是**能开多少 worker**，即每个时序 worker
的内存占用与重建成本。

**6. 时序 worker 的足迹与重建成本已实测（2026-09-10 22:03–22:45，只读 `m1_v2` 已发布的真实 F1 数据集）。**
`temporal_fold_batches` 在 230 万行 train / 70 万行 validation 的真实折上：**峰值 RSS 16.80 GiB**、
**一次性重建 2409 秒（40.2 分钟）**、产出 953 个训练批次与 237 个验证批次，单个训练批次的特征张量为
`(1, 1610, 20, 15)` float32。两条结论：重建相对 8.6 小时的单次拟合是 2.5%，且**每 worker 只做一次、
跨该折所有配置复用**——这正是当初选择"worker 侧自建"而不是逐任务 pickle 的原因；内存才是 worker 数的约束。

按各折已发布数据集的总行数等比折算（F1 实测 16.8 GiB 为基准），在 497 GiB 可用、90% 余量下：

| 折 | 数据集总行数 | 估算单 worker 足迹 | 安全 worker 数 |
|---|---|---|---|
| F1 | 3,798,054 | 16.8 GiB | 26 |
| F2 | 4,205,787 | 18.6 GiB | 24 |
| F3 | 4,582,910 | 20.3 GiB | 22 |
| F4 | 4,893,642 | 21.6 GiB | **20** |

`--parallel-workers` 是运行级参数、进程池按折创建，所以**约束取最大的折：20 路**。这也说明为什么需要一个
启动前的内存前置条件（Task 8）：48 路 × 21.6 GiB = 1037 GiB，在这台机器上不是启动失败，而是跑到第几小时
被 OOM 挑中。

### Task 8: 内存前置条件与按 makespan 选择任务粒度（2026-09-10 增量）

**为什么：** 第 6 条的实测把两件事从猜测变成算术。其一，worker 数受内存约束，而超限的失败方式是最贵的
那种——不是启动时被拒，而是数小时后被内核 OOM 挑中。其二，M2 的 24 个配置在 20 路下**不能整除**，于是
Task 7 里"未采纳配置数 ≥ worker 数就用配置级"这条规则是错的：配置级跑 `ceil(24/20)=2` 轮、每轮 3 次拟合
= **6 个拟合时间**；种子级跑 `ceil(72/20)=4` 轮、每轮 1 次 = **4 个拟合时间**。差的这一轮是因为一个配置
任务一旦派发就不能再拆，而多出来的 4 个配置照样占满一整轮。

**接口：**

- `parallel_search.available_memory_gb(path=/proc/meminfo)` —— 读 **MemAvailable 而非 MemFree**：本机 MemFree
  约 125 GiB 而实际可用约 497 GiB，读错那一行会把跑得下的配置判成跑不下，而一个总喊狼来了的护栏会被关掉，
  关掉之后它就什么也不护。
- `parallel_search.worker_memory_budget(...)` —— 返回判定**以及最大的安全 worker 数**，因为被拒之后操作者的
  下一个问题永远是"那到底能开几个"。
- `parallel_search.granularity_makespans / search_granularity` —— 两个候选 makespan（以拟合时间计）与其择优，
  打平时选粗粒度（任务少三分之二、返回标量而非每 seed 的预测帧、每配置一个 worker）。
- 驱动 `--worker-memory-gb`：声明每个 worker 的足迹即成为启动前的硬前置条件；不声明则只把余量打印进转录，
  **不让既有调用方式开始失败**（这是一条新增前置条件，不能顺手把别人的运行弄崩）。

**步骤：**

- [x] 失败测试先行：MemAvailable 与 MemFree 的区分、宿主不提供时为 None、预算的 fit/不 fit/最大安全数、
  90% 余量的边界、退化输入拒绝。
- [x] 实现 `available_memory_gb` / `worker_memory_budget` / `granularity_makespans` / `search_granularity`。
- [x] 驱动接 `--worker-memory-gb` 与 `_check_worker_memory`（`main()` 内、钉线程之前，一次运行只查一次），
  并把两个 torch 搜索循环的粒度判据从"填满 worker"换成 makespan 择优。
- [x] 一条测试预期被实现推翻后**改的是测试而不是实现**：24 配置 / 48 worker 时配置级是 1 轮×3 次拟合，
  种子级是 `ceil(72/48)=2` 轮×1 次——种子级更快，因为配置级会让 24 个 worker 闲着。配置级只在打平时凭
  tie-break 胜出。
- [x] 全量门禁：`ruff check src tests` 干净、`mypy src` 169 文件干净、驱动脚本 mypy 0 错误、
  `pytest tests -q` **2039 passed / 45 skipped**。

**据此重算 M2 的时长**（单次拟合取计划实测的 8.60 h，worker 数取上表的 20）：搜索 4 轮 × 8.6 h ≈ 34.4 h，
加每 worker 一次 40 分钟重建；最终三次拟合 1 轮 ≈ 8.6 h，加新一批 worker 的 40 分钟重建 → **单折约 44 h，
四折约 7.4 天**。若仍按"填满 worker"选粒度则是单折约 61 h、四折约 10 天——**这条规则本身值 2.6 天**。
对照计划里的串行估算（约 161 天），20 路是约 20 倍。

**顺带否掉一个看似划算的优化：** 一折内复用同一个进程池（让最终拟合的 worker 命中搜索阶段的面板缓存）只省
每折约 40 分钟——因为最终拟合只有 3 个任务、只会用到 3 个 worker——占单折 44 小时的 1.5%。已测得重建成本后
就不该做，而不是先做了再解释。
