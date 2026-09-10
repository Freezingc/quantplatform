# 证据目录：条件性因子研究审查（2026-09-10）

本目录保存 [研究审查报告 v0.1](../../2026-09-10-conditional-factor-idea-review-v0.1.md) 所依据的原始工件，使该分支在另一台机器上 clone 之后即可独立复核，不依赖任何临时目录或仓库外的工件库。

## 为什么要入库

报告引用的数值来自两组当时存放在机器本地的文件：

- `/tmp/es_metric_diag/` —— 只读诊断实验的脚本与结果。临时目录会被系统清理。
- `/data/zhangyudong/qrv2_research_artifacts/` —— 仓库外的模型工件库。不会随 clone 转移。

两者都已复制到此处。复制前后的 SHA-256 记录在 [provenance.json](provenance.json)，原始机器路径一并保留，便于追溯这些数字是哪一次运行产生的。

## 目录内容

| 文件 | 内容 | 对应报告章节 |
|---|---|---|
| [experiment/rankreg_subsets.json](experiment/rankreg_subsets.json) | 同一载体、同一固定超参数下 all15 / dense9 / sparse6 三组子集的池化与逐折结果，是报告全部核心数值的直接来源 | §1.3、§6.1 |
| [experiment/run_rankreg_subsets.py](experiment/run_rankreg_subsets.py) | 产生上述结果的脚本。其 `_verdict` 使用固定的 ±0.001 阈值，不是显著性检验 | §6.1 |
| [experiment/run_config_robustness.py](experiment/run_config_robustness.py) | 四配置稳健性脚本。固定使用全部 15 列，因此不能单独证明稀疏因子的边际贡献与配置无关 | §6.2 |
| [experiment/ablation_result.json](experiment/ablation_result.json) | 载体与因子子集的归因消融输出 | §5、§6 |
| [experiment/carrier_result.json](experiment/carrier_result.json) | LambdaRank → rank-regression 的载体对照输出 | §5 |
| [production/m0r_v2_report.json](production/m0r_v2_report.json) | M0R 家族生产链运行报告，用于与 all15 主链指标交叉核对 | §1.3 |
| [literature/manifest.json](literature/manifest.json) | §5 全部 14 篇文献的版本号 URL、出版状态、阅读深度、检索日期与内容 SHA-256 | §5 |
| [literature/url_checks.json](literature/url_checks.json) | 交付当日对 14 个来源的可达性复核（全部 HTTP 200） | §5 |
| [provenance.json](provenance.json) | 每个入库文件的原始机器路径、SHA-256 与字节数 | §9.2 |
| [verify_evidence.py](verify_evidence.py) | 自校验脚本，见下 | 全部 |

## 如何复核

```bash
python3 docs/research/evidence/2026-09-10-conditional-factor/verify_evidence.py
```

在仓库任意位置运行均可（脚本按自身位置解析路径）。它做三件事，任何一项不成立即以非零码退出：

1. **完整性** —— 每个入库文件的 SHA-256 仍与 `provenance.json` 记录一致。
2. **数值** —— 报告 §1.3 的每个数字（三组池化 RankIC、相对参考模型的差值、ICIR、945 个有效交易日、0.000317821 的边际贡献、四个年度的逐年差值）从 `rankreg_subsets.json` **重新计算**，而不是当作已核对的文本；同时确认 §6.1、§6.2 的两条审计判断在脚本源码中仍然成立。
3. **文献** —— 14 篇来源均具备版本号 URL、出版状态、阅读深度和 64 位内容哈希，且每个 URL 确实在报告正文中被引用。

交付时该脚本输出 `PASS: 155 checks held`。

## 没有入库的内容

论文全文（arXiv 与出版方的 PDF/HTML，约 11MB）**未**提交到仓库，原因是第三方版权与仓库体积。改为在 `literature/manifest.json` 中记录带版本号的稳定 URL 与内容 SHA-256：接手者重新抓取该 URL，比对哈希即可确认读到的是同一份文本。arXiv 的版本号 URL（例如 `.../2306.09862v3`）不会随时间改变所指内容。

`manifest.json` 中的 `review_depth` 字段区分了两种阅读深度：读取了正文具体章节的，与仅核验摘要和元数据的（目前只有 AlphaG-OPD 一篇属于后者，报告 §5.2 已标注其为待补充审读项）。

## 边界

本目录只证明**工件与报告一致**，不证明结论正确。报告 §6、§7 明确列出了尚未完成的检验：

- 没有对 all15 与 dense9 的差值做过正式的显著性或等效性检验（§6.1）。
- 没有在四个配置下分别重做该差值消融（§6.2）。
- 没有复现任何论文基线，没有启动新训练，没有新的市场数据实验。
- 因子准入使用了后来被划为外层测试年份的历史区间，这一局限记录在报告 §6.4，不因证据入库而消除。
