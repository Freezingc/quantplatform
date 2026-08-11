import argparse
import inspect
import importlib
import os
import pkgutil
from datetime import datetime
from pathlib import Path

import dolphindb as ddb
import pandas as pd


def discover_factor_calculators(db_dir: Path):
    calculators = {}
    package_path = db_dir / "factor_calculators"
    package_name = "factor_calculators"

    for _, module_name, _ in pkgutil.iter_modules([str(package_path)]):
        if module_name == "base":
            continue
        try:
            module = importlib.import_module(f"{package_name}.{module_name}")
        except Exception as exc:
            print(f"[WARN] 加载模块 {module_name} 失败: {exc}")
            continue

        for member_name, member in inspect.getmembers(module):
            if inspect.isclass(member) and member_name.endswith("Calculator"):
                calculators[member_name] = member

    return calculators


def build_parser():
    parser = argparse.ArgumentParser(description="导出单个因子为因子检验 CSV")
    parser.add_argument("--method", required=True, help="因子计算器类名，例如 AttentionSpilloverCalculator")
    parser.add_argument("--lookback", type=int, default=60, help="lookback_days 参数")
    parser.add_argument("--start", default="2020.01.01", help="开始日期，格式 YYYY.MM.DD")
    parser.add_argument("--end", default=datetime.now().strftime("%Y.%m.%d"), help="结束日期，格式 YYYY.MM.DD")
    parser.add_argument("--factor-name", default="", help="导出文件中的因子名（默认由 method 去掉 Calculator）")
    parser.add_argument("--output-folder", default="", help="输出目录，默认 因子检验/factor_data_files")
    parser.add_argument("--host", default="10.176.55.211", help="DolphinDB host")
    parser.add_argument("--port", type=int, default=8848, help="DolphinDB port")
    parser.add_argument("--username", default="zhangyudong", help="DolphinDB username，默认读取环境变量")
    parser.add_argument("--password", default="123456", help="DolphinDB password，默认读取环境变量")
    parser.add_argument("--dropna", action="store_true", help="是否丢弃 factor 为 NaN 的行")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()

    script_path = Path(__file__).resolve()
    eval_dir = script_path.parents[1]
    repo_root = script_path.parents[2]
    db_dir = repo_root / "数据库"

    if str(db_dir) not in os.sys.path:
        os.sys.path.insert(0, str(db_dir))

    output_folder = Path(args.output_folder) if args.output_folder else (eval_dir / "factor_data_files")
    output_folder.mkdir(parents=True, exist_ok=True)

    factor_name = args.factor_name or args.method.replace("Calculator", "")
    output_file = output_folder / f"factor_data_{factor_name}.csv"

    username = args.username or os.getenv("dolphinDB_USERNAME", "")
    password = args.password or os.getenv("dolphinDB_PASSWORD", "")

    if not username or not password:
        raise ValueError("缺少数据库用户名或密码，请通过参数或环境变量 dolphinDB_USERNAME/dolphinDB_PASSWORD 提供")

    calculators = discover_factor_calculators(db_dir)
    if args.method not in calculators:
        names = ", ".join(sorted(calculators.keys())[:30])
        raise ValueError(f"未找到因子计算器 {args.method}。可用示例: {names}")

    session = ddb.session()
    try:
        print(f"[INFO] 连接 DolphinDB: {args.host}:{args.port}")
        session.connect(args.host, args.port, username, password)

        calc_cls = calculators[args.method]
        calculator = calc_cls(session, args.lookback)

        print(
            f"[INFO] 开始计算因子 method={args.method}, lookback={args.lookback}, "
            f"start={args.start}, end={args.end}"
        )
        factor_df = calculator.calculate(args.start, args.end)

        if not isinstance(factor_df, pd.DataFrame):
            raise TypeError("因子计算结果不是 DataFrame")

        long_df = factor_df.stack(dropna=False).rename("factor").reset_index()
        if long_df.shape[1] != 3:
            raise ValueError("转换后的长表列数异常，预期为3列")

        long_df.columns = ["trade_date", "ts_code", "factor"]
        if args.dropna:
            long_df = long_df.dropna(subset=["factor"])

        long_df["trade_date"] = pd.to_datetime(long_df["trade_date"]).dt.strftime("%Y-%m-%d")
        long_df["ts_code"] = long_df["ts_code"].astype(str)

        long_df.to_csv(output_file, index=False)

        print(f"[DONE] 导出完成: {output_file}")
        print(f"[DONE] 行数={len(long_df)}, 日期范围={long_df['trade_date'].min()}~{long_df['trade_date'].max()}")
    finally:
        try:
            session.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
