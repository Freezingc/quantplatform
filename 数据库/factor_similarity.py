"""
factor_similarity.py

计算新因子与数据库中所有现有因子的相似度，输出报告。

用法示例：
  # 新因子未入库，现场计算：
  python factor_similarity.py \
      --factor_name "3个月下影线均值" \
      --method Vol_DownShadow_Avg_3M_Calculator \
      --lookback_days 63 \
      --start_time 2023.01.01 \
      --end_time 2024.01.01

  # 新因子已入库，直接从 DB 拉取：
  python factor_similarity.py \
      --factor_name "3个月下影线均值" \
      --start_time 2023.01.01 \
      --end_time 2024.01.01
"""

import argparse
import os
import importlib
import inspect
import pkgutil
from pathlib import Path
from datetime import datetime

import dolphindb as ddb
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from tqdm import tqdm

from get_data import DataFetcher
from factor_calculators.base import FactorCalculator


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def discover_factor_calculators():
    calculator_classes = {}
    package_path = Path(__file__).parent / "factor_calculators"
    for _, name, _ in pkgutil.iter_modules([str(package_path)]):
        if name == "base":
            continue
        try:
            module = importlib.import_module(f"factor_calculators.{name}")
            for member_name, member in inspect.getmembers(module):
                if (inspect.isclass(member)
                        and issubclass(member, FactorCalculator)
                        and member is not FactorCalculator):
                    calculator_classes[member_name] = member
        except Exception as e:
            print(f"  [warn] 加载模块 {name} 失败: {e}")
    return calculator_classes


def fetch_factor_from_db(fetcher: DataFetcher, factor_name: str,
                         start_time: str, end_time: str) -> pd.DataFrame:
    """从 DB 拉取单个因子，返回 (trade_date × ts_code) DataFrame。"""
    raw = fetcher.get_factor_matrix_new(
        factor_names=[factor_name],
        start_time=start_time,
        end_time=end_time,
    )
    if raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        return pd.DataFrame()
    # raw: MultiIndex columns (factor_name, trade_date)，index = ts_code
    df = raw.xs(factor_name, level=0, axis=1).T  # → index=trade_date, columns=ts_code
    return df


def factor_in_db(s, factor_name: str) -> bool:
    result = s.run(f"""
        db = database("dfs://YesLab")
        t = loadTable(db, "factor_metadata")
        select count(*) as cnt from t where factor_name = '{factor_name}'
    """)
    return int(result.iloc[0, 0]) > 0


def get_all_factor_names(s, category_filter: str = "") -> list:
    """
    获取所有因子名称。
    category_filter 非空时，做子串模糊匹配，并排除 category 为空的因子。
    """
    result = s.run("""
        db = database("dfs://YesLab")
        t = loadTable(db, "factor_metadata")
        select factor_name, category from t
    """)
    if category_filter:
        # 排除空 category，再做子串匹配（strip 去掉不可见字符）
        mask = (
            result["category"].str.strip().str.len() > 0
        ) & (
            result["category"].str.strip().str.contains(category_filter, na=False)
        )
        matched = result[mask]
        matched_categories = matched["category"].str.strip().unique().tolist()
        print(f"  模糊匹配 '{category_filter}' 命中的 category: {matched_categories}")
        print(f"  共 {len(matched)} 个因子")
        return matched["factor_name"].tolist()
    return result["factor_name"].tolist()


# ---------------------------------------------------------------------------
# 核心相似度计算
# ---------------------------------------------------------------------------

def spearman_cross_section(a: pd.Series, b: pd.Series) -> float:
    """
    对单个截面计算 Spearman 秩相关（取绝对值）。
    a, b 已对齐（同一组股票），不含 NaN。
    """
    n = len(a)
    if n < 2:
        return np.nan
    rank_a = a.rank()
    rank_b = b.rank()
    d = rank_a - rank_b
    rho = 1 - 6 * (d ** 2).sum() / (n * (n ** 2 - 1))
    return abs(float(rho))


def compute_similarity(new_factor: pd.DataFrame,
                       existing_factor: pd.DataFrame,
                       min_common_stocks: int,
                       min_valid_dates: int,
                       sample_dates=None):
    """
    计算两个因子矩阵的平均截面 Spearman 相似度。
    返回 (similarity, diag_info)：
      - similarity: float 或 None（有效截面不足时）
      - diag_info: dict，包含诊断信息
    """
    diag = {}

    if new_factor.empty or existing_factor.empty:
        diag["reason"] = "因子矩阵为空"
        diag["new_factor_empty"] = new_factor.empty
        diag["existing_factor_empty"] = existing_factor.empty
        return None, diag

    if sample_dates is not None:
        dates_to_use = pd.DatetimeIndex(sample_dates).intersection(
            new_factor.index.intersection(existing_factor.index)
        )
        diag["sample_dates_count"] = len(sample_dates)
        diag["new_factor_dates"] = len(new_factor.index)
        diag["existing_factor_dates"] = len(existing_factor.index)
        diag["overlapping_dates"] = len(new_factor.index.intersection(existing_factor.index))
        diag["dates_in_sample_and_overlap"] = len(dates_to_use)
    else:
        dates_to_use = new_factor.index.intersection(existing_factor.index)
        diag["new_factor_dates"] = len(new_factor.index)
        diag["existing_factor_dates"] = len(existing_factor.index)
        diag["overlapping_dates"] = len(dates_to_use)

    if len(dates_to_use) == 0:
        diag["reason"] = "无重叠日期"
        return None, diag

    scores = []
    skipped_dates = 0
    common_stock_counts = []
    for date in dates_to_use:
        a = new_factor.loc[date].dropna()
        b = existing_factor.loc[date].dropna()
        common_stocks = a.index.intersection(b.index)
        common_stock_counts.append(len(common_stocks))
        if len(common_stocks) < min_common_stocks:
            skipped_dates += 1
            continue
        rho = spearman_cross_section(a[common_stocks], b[common_stocks])
        if not np.isnan(rho):
            scores.append(rho)

    diag["dates_evaluated"] = len(dates_to_use)
    diag["dates_skipped_by_min_stocks"] = skipped_dates
    diag["dates_with_valid_score"] = len(scores)
    diag["min_valid_dates_required"] = min_valid_dates
    diag["min_common_stocks_required"] = min_common_stocks
    if common_stock_counts:
        diag["common_stocks_min"] = int(min(common_stock_counts))
        diag["common_stocks_max"] = int(max(common_stock_counts))
        diag["common_stocks_mean"] = round(float(np.mean(common_stock_counts)), 1)

    if len(scores) < min_valid_dates:
        diag["reason"] = f"有效截面数 {len(scores)} < 要求的 {min_valid_dates}"
        return None, diag

    return float(np.mean(scores)), diag


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="因子相似度比较工具")
    parser.add_argument("--factor_name", required=True, help="新因子名称")
    parser.add_argument("--method", default="", help="计算器类名（新因子未入库时必填）")
    parser.add_argument("--lookback_days", type=int, default=0,
                        help="计算器 lookback（新因子未入库时必填）")
    parser.add_argument("--start_time", required=True, help="比较区间开始，格式 YYYY.MM.DD")
    parser.add_argument("--end_time", required=True, help="比较区间结束，格式 YYYY.MM.DD")
    parser.add_argument("--threshold", type=float, default=0.7, help="相似度阈值，默认 0.7")
    parser.add_argument("--min_common_stocks", type=int, default=50,
                        help="截面最少共同有值股票数，默认 50")
    parser.add_argument("--min_valid_dates", type=int, default=20,
                        help="最少有效截面数，不足则跳过该因子，默认 20（采样模式下自动取 min(20, sample_days)）")
    parser.add_argument("--sample", action="store_true", default=True,
                        help="启用随机采样模式（默认启用）")
    parser.add_argument("--no_sample", action="store_true", default=False,
                        help="禁用随机采样，使用全部截面")
    parser.add_argument("--sample_days", type=int, default=30,
                        help="采样天数，默认 30（不超过区间内实际交易日数）")
    parser.add_argument("--output", default="",
                        help="输出报告路径，默认 similarity_report_{factor_name}.txt")
    parser.add_argument("--category", default="",
                        help="只与指定类别的因子比较（子串模糊匹配），不填则比较所有因子")
    parser.add_argument("--verbose_skip", action="store_true", default=False,
                        help="输出跳过因子的详细诊断信息，默认关闭")
    args = parser.parse_args()

    load_dotenv()
    username = os.getenv("dolphinDB_USERNAME")
    password = os.getenv("dolphinDB_PASSWORD")

    s = ddb.session()
    s.connect("10.176.55.211", 8848, username, password)
    fetcher = DataFetcher(s=s)

    # ------------------------------------------------------------------
    # 1. 获取新因子数据
    # ------------------------------------------------------------------
    print(f"\n[1/4] 获取新因子: {args.factor_name}")
    in_db = factor_in_db(s, args.factor_name)

    if in_db:
        print(f"  → 因子已在 DB 中，直接拉取")
        new_factor = fetch_factor_from_db(fetcher, args.factor_name,
                                          args.start_time, args.end_time)
    else:
        if not args.method or args.lookback_days <= 0:
            print("  [error] 因子不在 DB 中，需要提供 --method 和 --lookback_days")
            s.close()
            return

        calculators = discover_factor_calculators()
        if args.method not in calculators:
            print(f"  [error] 未找到计算器: {args.method}")
            s.close()
            return

        print(f"  → 因子未入库，使用 {args.method} 现场计算")
        calc = calculators[args.method](s, args.lookback_days)
        new_factor = calc.calculate(args.start_time, args.end_time)

    print(f"  新因子形状: {new_factor.shape}  (dates × stocks)")

    # ------------------------------------------------------------------
    # 2. 获取所有现有因子名称（排除自身）
    # ------------------------------------------------------------------
    print("\n[2/4] 获取现有因子列表")
    all_factor_names = get_all_factor_names(s, category_filter=args.category)
    compare_names = [n for n in all_factor_names if n != args.factor_name]
    if args.category:
        print(f"  按 category='{args.category}' 过滤后，排除自身，比较 {len(compare_names)} 个因子")
    else:
        print(f"  共 {len(all_factor_names)} 个因子，排除自身后比较 {len(compare_names)} 个")

    # ------------------------------------------------------------------
    # 2.5 确定采样日期
    # ------------------------------------------------------------------
    use_sample = (not args.no_sample)
    sample_dates = None

    if use_sample:
        all_trade_days = fetcher.get_trade_days(
            exchange="SSE",
            start_time=args.start_time,
            end_time=args.end_time,
        )
        n_trade_days = len(all_trade_days)
        actual_sample = min(args.sample_days, n_trade_days)
        rng = np.random.default_rng()
        sample_dates = pd.DatetimeIndex(
            sorted(rng.choice(all_trade_days, size=actual_sample, replace=False))
        )
        # min_valid_dates 联动：不能超过采样天数
        effective_min_valid = min(args.min_valid_dates, actual_sample)
        print(f"  采样模式: 区间内共 {n_trade_days} 个交易日，随机抽取 {actual_sample} 天")
        print(f"  min_valid_dates 调整为: {effective_min_valid}")
    else:
        effective_min_valid = args.min_valid_dates
        print(f"  全量模式: 使用全部截面，min_valid_dates={effective_min_valid}")

    # ------------------------------------------------------------------
    # 3. 逐因子计算相似度
    # ------------------------------------------------------------------
    print("\n[3/4] 逐因子计算相似度...")
    results = []   # list of (factor_name, similarity)
    skipped = []   # list of (factor_name, reason)

    for fname in tqdm(compare_names):
        try:
            existing = fetch_factor_from_db(fetcher, fname,
                                            args.start_time, args.end_time)
            sim, diag = compute_similarity(
                new_factor, existing,
                min_common_stocks=args.min_common_stocks,
                min_valid_dates=effective_min_valid,
                sample_dates=sample_dates,
            )
            if sim is None:
                skipped.append((fname, "有效截面数不足", diag))
            else:
                results.append((fname, sim))
        except Exception as e:
            skipped.append((fname, f"拉取/计算出错: {e}", {}))

    s.close()

    # ------------------------------------------------------------------
    # 4. 整理结果并输出报告
    # ------------------------------------------------------------------
    print("\n[4/4] 生成报告")
    results.sort(key=lambda x: x[1], reverse=True)
    above_threshold = [(n, v) for n, v in results if v >= args.threshold]

    output_path = args.output or f"similarity_report_{args.factor_name}.txt"

    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = []
    lines.append("=" * 60)
    lines.append(f"因子相似度报告")
    lines.append(f"生成时间: {now_str}")
    lines.append("=" * 60)
    lines.append(f"新因子名称  : {args.factor_name}")
    lines.append(f"比较时间段  : {args.start_time} ~ {args.end_time}")
    lines.append(f"比较范围    : {('category 含 ' + args.category) if args.category else '全部因子'}")
    lines.append(f"相似度阈值  : {args.threshold}")
    lines.append(f"最少共同股票: {args.min_common_stocks}")
    if use_sample:
        lines.append(f"采样模式    : 启用，抽取 {actual_sample} 天")
        lines.append(f"最少有效截面: {effective_min_valid}（自动调整）")
    else:
        lines.append(f"采样模式    : 禁用（全量截面）")
        lines.append(f"最少有效截面: {effective_min_valid}")
    lines.append("-" * 60)
    lines.append(f"总比较因子数: {len(compare_names)}")
    lines.append(f"成功比较    : {len(results)}")
    lines.append(f"跳过        : {len(skipped)}")
    lines.append(f"超过阈值    : {len(above_threshold)}")
    lines.append("=" * 60)

    if above_threshold:
        lines.append(f"\n【相似因子列表（相似度 ≥ {args.threshold}，从高到低）】\n")
        lines.append(f"{'排名':<6}{'因子名称':<40}{'相似度':>8}")
        lines.append("-" * 56)
        for rank, (fname, sim) in enumerate(above_threshold, 1):
            lines.append(f"{rank:<6}{fname:<40}{sim:>8.4f}")
    else:
        lines.append(f"\n未发现相似度超过 {args.threshold} 的因子。")

    if skipped:
        lines.append("\n【跳过的因子】\n")
        for fname, reason, diag in skipped:
            lines.append(f"  {fname}: {reason}")
            if args.verbose_skip and diag:
                for k, v in diag.items():
                    lines.append(f"      {k}: {v}")

    lines.append("\n" + "=" * 60)

    report = "\n".join(lines)
    print(report)

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\n报告已保存至: {output_path}")


if __name__ == "__main__":
    main()