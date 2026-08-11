import argparse
import json
import os
import re

import matplotlib.pyplot as plt
import pandas as pd

from layer_backtest import LayerBacktester


def run_one(name, cfg, args):
    score_path = os.path.join(args.input_dir, f"{name}.csv")
    if not os.path.exists(score_path):
        raise FileNotFoundError(f"未找到打分文件: {score_path}")

    dta = pd.read_csv(score_path)
    output_dir = os.path.join(args.output_dir, name, "layer_backtest")

    bt = LayerBacktester(
        score_df=dta,
        begin_date=cfg["begin_date"],
        end_date=cfg["end_date"],
        hold_days=cfg.get("position_cycle", args.hold_days),
        layers=cfg.get("layers", args.layers),
        output_dir=output_dir,
        host=args.host,
        port=args.port,
        user=args.user,
        password=args.password,
        commission=cfg.get("commission", args.commission),
        stamptax=cfg.get("stamptax", args.stamptax),
        slip=cfg.get("slip", args.slip),
    )

    _, nav_df, summary_df = bt.run()
    summary_df.insert(0, "策略", name)
    print(name + " 分层回测已完成")
    return summary_df, nav_df


def _build_compare_layers(nav_map):
    if not nav_map:
        return []

    common_cols = None
    for nav_df in nav_map.values():
        cols = set(nav_df.columns) - {"signal_date", "date"}
        common_cols = cols if common_cols is None else (common_cols & cols)

    if not common_cols:
        return []

    compare_layers = []
    for c in ["L1", "L_top_bottom"]:
        if c in common_cols:
            compare_layers.append(c)

    numeric_layers = []
    for c in common_cols:
        m = re.fullmatch(r"L(\d+)", c)
        if m is not None:
            numeric_layers.append(int(m.group(1)))
    if numeric_layers:
        top_layer = f"L{max(numeric_layers)}"
        if top_layer not in compare_layers:
            compare_layers.append(top_layer)

    return compare_layers


def save_strategy_compare_charts(nav_map, output_dir):
    compare_layers = _build_compare_layers(nav_map)
    if not compare_layers:
        print("未找到可对比的分层列，已跳过策略分层对比图输出")
        return

    os.makedirs(output_dir, exist_ok=True)

    for layer_col in compare_layers:
        plt.figure(figsize=(12, 6))
        merged = None
        for strategy_name, nav_df in nav_map.items():
            if layer_col not in nav_df.columns:
                continue
            date_col = "date" if "date" in nav_df.columns else "signal_date"
            x = pd.to_datetime(nav_df[date_col], format="%Y%m%d", errors="coerce")
            y = nav_df[layer_col].astype(float)
            plt.plot(x, y, label=strategy_name)

            tmp = pd.DataFrame({"date": nav_df[date_col], strategy_name: y})
            merged = tmp if merged is None else merged.merge(tmp, on="date", how="outer")

        plt.title(f"Strategy Layer Comparison - {layer_col}")
        plt.xlabel("Date")
        plt.ylabel("Net Value")
        plt.xticks(rotation=45)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"策略分层对比_{layer_col}.png"), dpi=150)
        plt.close()

        if merged is not None:
            merged = merged.sort_values("date")
            merged.to_csv(
                os.path.join(output_dir, f"策略分层对比_{layer_col}.csv"),
                index=False,
                encoding="utf_8_sig",
            )

    print("策略分层对比图已输出:", output_dir)


def main():
    parser = argparse.ArgumentParser(description="分层回测一键入口（对齐 Test.py 风格）")
    parser.add_argument("--config", default="config.json", help="配置文件路径")
    parser.add_argument("--input-dir", default="input", help="打分CSV目录")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--name", default="", help="只运行一个策略名，留空表示全部")
    parser.add_argument("--layers", type=int, default=10, help="分层数量")
    parser.add_argument("--hold-days", type=int, default=5, help="默认持有天数")
    parser.add_argument("--host", default="10.176.55.211", help="DolphinDB host")
    parser.add_argument("--port", type=int, default=8848, help="DolphinDB port")
    parser.add_argument("--user", default="zhangyudong", help="DolphinDB 用户")
    parser.add_argument("--password", default="123456", help="DolphinDB 密码")
    parser.add_argument("--commission", type=float, default=0.0004, help="买卖手续费率")
    parser.add_argument("--stamptax", type=float, default=0.001, help="卖出印花税率")
    parser.add_argument("--slip", type=float, default=0.001, help="买卖滑点率")
    parser.add_argument(
        "--cross-strategy-compare",
        action="store_true",
        help="是否额外生成跨策略分层对比图（默认关闭）",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    summary_list = []
    nav_map = {}
    if args.name:
        if args.name not in config:
            raise KeyError(f"策略 {args.name} 不在 {args.config} 中")
        summary_df, nav_df = run_one(args.name, config[args.name], args)
        summary_list.append(summary_df)
        nav_map[args.name] = nav_df
    else:
        for name, cfg in config.items():
            summary_df, nav_df = run_one(name, cfg, args)
            summary_list.append(summary_df)
            nav_map[name] = nav_df

    if summary_list:
        final_df = pd.concat(summary_list, ignore_index=True)
        out_path = os.path.join(args.output_dir, "分层回测汇总.csv")
        final_df.to_csv(out_path, index=False, encoding="utf_8_sig")
        print("分层回测汇总已输出:", out_path)

    if nav_map and args.cross_strategy_compare:
        save_strategy_compare_charts(nav_map, args.output_dir)


if __name__ == "__main__":
    main()
