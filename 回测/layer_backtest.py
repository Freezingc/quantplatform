import argparse
import json
import os

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

import get_data


class LayerBacktester:
    def __init__(
        self,
        score_df,
        begin_date,
        end_date,
        hold_days,
        layers=10,
        output_dir="output",
        host="10.176.55.211",
        port=8848,
        user="zhangyudong",
        password="123456",
        drop_bj=True,
        commission=0.0004,
        stamptax=0.001,
        slip=0.001,
    ):
        self.score_df = score_df.copy()
        self.begin_date = str(begin_date)
        self.end_date = str(end_date)
        self.hold_days = int(hold_days)
        self.layers = int(layers)
        self.output_dir = output_dir
        self.drop_bj = drop_bj
        self.commission = float(commission)
        self.stamptax = float(stamptax)
        self.slip = float(slip)

        self.fetcher = get_data.DataFetcher(host=host, port=port, user=user, password=password)

        if self.layers < 2:
            raise ValueError("layers 必须 >= 2")
        if self.hold_days < 1:
            raise ValueError("hold_days 必须 >= 1")
        if min(self.commission, self.stamptax, self.slip) < 0:
            raise ValueError("交易成本参数不能为负数")

    def _prepare_score(self):
        if "ts_code" not in self.score_df.columns:
            raise ValueError("输入打分表必须包含 ts_code 列")

        if self.drop_bj:
            self.score_df = self.score_df.loc[
                ~self.score_df["ts_code"].astype(str).str.endswith(".BJ")
            ].copy()

        date_cols = [c for c in self.score_df.columns if c != "ts_code"]
        date_cols = sorted([str(c) for c in date_cols])

        mask = [(c >= self.begin_date) and (c <= self.end_date) for c in date_cols]
        use_cols = [c for c, keep in zip(date_cols, mask) if keep]

        if len(use_cols) < self.hold_days + 2:
            raise ValueError(
                f"打分日期不足，至少需要 hold_days+2={self.hold_days + 2} 个交易日，当前={len(use_cols)}"
            )

        self.score_df = self.score_df[["ts_code"] + use_cols].copy()

    def _prepare_open_st(self):
        open_data = self.fetcher.get_stock_daily_matrix(
            factor_names=["open_hfq"],
            start_time=pd.to_datetime(self.begin_date).strftime("%Y.%m.%d"),
        )
        open_df = open_data["open_hfq"].copy()

        if self.drop_bj:
            open_df = open_df.loc[[idx for idx in open_df.index if not str(idx).endswith(".BJ")], :]

        open_df.columns = [d.strftime("%Y%m%d") for d in open_df.columns]

        st_df = self.fetcher.get_stock_is_ST(
            start_time=pd.to_datetime(self.begin_date).strftime("%Y.%m.%d")
        )
        if self.drop_bj:
            st_df = st_df.loc[[idx for idx in st_df.index if not str(idx).endswith(".BJ")], :]
        st_df.columns = [d.strftime("%Y%m%d") for d in st_df.columns]

        self.open_df = open_df
        self.st_df = st_df

    def _build_common_dates(self):
        score_dates = set(self.score_df.columns[1:])
        open_dates = set(self.open_df.columns)
        valid_dates = sorted(score_dates & open_dates)

        valid_dates = [d for d in valid_dates if self.begin_date <= d <= self.end_date]
        if len(valid_dates) < self.hold_days + 2:
            raise ValueError(
                f"score/open 对齐后交易日不足，至少需要 {self.hold_days + 2} 天，当前 {len(valid_dates)} 天"
            )

        self.valid_dates = valid_dates
        self.score_df = self.score_df[["ts_code"] + valid_dates].copy()
        self.open_df = self.open_df.reindex(columns=valid_dates)
        self.st_df = self.st_df.reindex(columns=valid_dates)

    def _calc_one_day_layer_return(self, signal_date, entry_date, exit_date):
        tmp = self.score_df[["ts_code", signal_date]].copy()
        tmp.columns = ["ts_code", "score"]

        tmp = tmp.merge(
            self.open_df[[entry_date]].reset_index().rename(columns={"index": "ts_code", entry_date: "entry_open"}),
            on="ts_code",
            how="left",
        )
        tmp = tmp.merge(
            self.open_df[[exit_date]].reset_index().rename(columns={"index": "ts_code", exit_date: "exit_open"}),
            on="ts_code",
            how="left",
        )

        if entry_date in self.st_df.columns:
            st_tmp = self.st_df[[entry_date]].reset_index().rename(
                columns={"index": "ts_code", entry_date: "is_st"}
            )
            tmp = tmp.merge(st_tmp, on="ts_code", how="left")
            tmp = tmp[tmp["is_st"].fillna(0) != 1]

        tmp = tmp.replace([np.inf, -np.inf], np.nan)
        tmp = tmp.dropna(subset=["score", "entry_open", "exit_open"])

        if len(tmp) < self.layers:
            return None

        buy_cost = self.commission + self.slip
        sell_cost = self.commission + self.slip + self.stamptax
        tmp["ret"] = (1.0 - buy_cost) * (tmp["exit_open"] / tmp["entry_open"]) * (1.0 - sell_cost) - 1.0
        tmp = tmp.replace([np.inf, -np.inf], np.nan).dropna(subset=["ret"])

        if len(tmp) < self.layers:
            return None

        rank_pct = tmp["score"].rank(method="first", pct=True)
        tmp["layer"] = np.ceil(rank_pct * self.layers).clip(1, self.layers).astype(int)

        layer_ret = tmp.groupby("layer")["ret"].mean()
        out = {f"L{l}": float(layer_ret.get(l, np.nan)) for l in range(1, self.layers + 1)}
        out["L_top_bottom"] = out[f"L{self.layers}"] - out["L1"]
        out["样本数"] = int(len(tmp))
        out["signal_date"] = signal_date
        out["entry_date"] = entry_date
        out["exit_date"] = exit_date
        return out

    def run(self):
        self._prepare_score()
        self._prepare_open_st()
        self._build_common_dates()

        rows = []
        max_i = len(self.valid_dates) - self.hold_days - 1
        for i in range(0, max_i, self.hold_days):
            signal_date = self.valid_dates[i]
            entry_date = self.valid_dates[i + 1]
            exit_date = self.valid_dates[i + self.hold_days + 1]

            row = self._calc_one_day_layer_return(signal_date, entry_date, exit_date)
            if row is not None:
                rows.append(row)

        if not rows:
            raise ValueError("没有可用分层样本，请检查分数数据覆盖、层数设置和交易日范围")

        daily_ret_df = pd.DataFrame(rows).sort_values("signal_date")
        layer_cols = [f"L{l}" for l in range(1, self.layers + 1)] + ["L_top_bottom"]

        nav_df = daily_ret_df[["exit_date"] + layer_cols].copy()
        nav_df = nav_df.rename(columns={"exit_date": "date"})
        for c in layer_cols:
            nav_df[c] = (1.0 + nav_df[c]).cumprod()

        summary = []
        n = len(daily_ret_df)
        for c in layer_cols:
            r = daily_ret_df[c].dropna()
            if len(r) == 0:
                summary.append({"分层": c, "年化收益": np.nan, "年化波动": np.nan, "夏普": np.nan, "累计收益": np.nan})
                continue
            periods_per_year = 252.0 / self.hold_days
            ann_ret = (1.0 + r).prod() ** (252.0 / (len(r) * self.hold_days)) - 1.0
            ann_vol = r.std(ddof=1) * np.sqrt(periods_per_year) if len(r) > 1 else np.nan
            sharpe = ann_ret / ann_vol if pd.notna(ann_vol) and abs(ann_vol) > 1e-12 else np.nan
            cum_ret = (1.0 + r).prod() - 1.0
            summary.append({"分层": c, "年化收益": ann_ret, "年化波动": ann_vol, "夏普": sharpe, "累计收益": cum_ret, "样本天数": n})
        summary_df = pd.DataFrame(summary)

        os.makedirs(self.output_dir, exist_ok=True)
        summary_df.to_csv(os.path.join(self.output_dir, "分层指标汇总.csv"), index=False, encoding="utf_8_sig")

        plt.figure(figsize=(12, 6))
        x = pd.to_datetime(nav_df["date"], format="%Y%m%d", errors="coerce")
        for c in layer_cols:
            plt.plot(x, nav_df[c], label=c)
        plt.title(f"Layered Backtest NAV (layers={self.layers}, hold_days={self.hold_days})")
        plt.xlabel("Date")
        plt.ylabel("NAV")
        ax = plt.gca()
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=10))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        plt.xticks(rotation=45)
        plt.legend(ncol=2, fontsize=9)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "layer_nav.png"), dpi=150)
        plt.close()

        return daily_ret_df, nav_df, summary_df


def run_one_strategy(args, strategy_name, cfg):
    score_path = os.path.join(args.input_dir, f"{strategy_name}.csv")
    if not os.path.exists(score_path):
        raise FileNotFoundError(f"未找到打分文件: {score_path}")

    score_df = pd.read_csv(score_path)

    begin_date = cfg["begin_date"]
    end_date = cfg["end_date"]
    hold_days = cfg.get("position_cycle", args.hold_days)

    output_dir = os.path.join(args.output_dir, strategy_name, "layer_backtest")
    backtester = LayerBacktester(
        score_df=score_df,
        begin_date=begin_date,
        end_date=end_date,
        hold_days=hold_days,
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

    _, _, summary_df = backtester.run()
    summary_df.insert(0, "策略", strategy_name)
    return summary_df


def main():
    parser = argparse.ArgumentParser(description="基于现有回测输入格式的分层回测脚本")
    parser.add_argument("--config", default="config.json", help="配置文件路径")
    parser.add_argument("--input-dir", default="input", help="打分CSV目录")
    parser.add_argument("--output-dir", default="output", help="输出目录")
    parser.add_argument("--name", default="", help="仅运行某个策略名，留空则跑 config 全部")
    parser.add_argument("--layers", type=int, default=10, help="分层数")
    parser.add_argument("--hold-days", type=int, default=5, help="默认持有天数（可被 config.position_cycle 覆盖）")
    parser.add_argument("--host", default="10.176.55.211", help="DolphinDB host")
    parser.add_argument("--port", type=int, default=8848, help="DolphinDB port")
    parser.add_argument("--user", default="zhangyudong", help="DolphinDB 用户")
    parser.add_argument("--password", default="123456", help="DolphinDB 密码")
    parser.add_argument("--commission", type=float, default=0.0004, help="买卖手续费率")
    parser.add_argument("--stamptax", type=float, default=0.001, help="卖出印花税率")
    parser.add_argument("--slip", type=float, default=0.001, help="买卖滑点率")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if args.name:
        if args.name not in cfg:
            raise KeyError(f"策略 {args.name} 不在配置中")
        all_summary = [run_one_strategy(args, args.name, cfg[args.name])]
    else:
        all_summary = []
        for strategy_name, strategy_cfg in cfg.items():
            print(f"开始分层回测: {strategy_name}")
            all_summary.append(run_one_strategy(args, strategy_name, strategy_cfg))

    if all_summary:
        final_summary = pd.concat(all_summary, ignore_index=True)
        final_path = os.path.join(args.output_dir, "分层回测汇总.csv")
        final_summary.to_csv(final_path, index=False, encoding="utf_8_sig")
        print(f"分层回测完成，汇总输出: {final_path}")


if __name__ == "__main__":
    main()
