import pandas as pd
import os
import time
from datetime import datetime
import channel
import get_data
import math
import numpy as np
import dolphindb as ddb
import matplotlib.pyplot as plt
from matplotlib.dates import AutoDateLocator

class backtest:
    def __init__(self, dta, begin_date, end_date, base, commission = 0.0004, stamptax = 0.001, slip = 0.001, use_open = True, init_cash = 200000, result_path = 'output', mode = 0):
        # 打分表
        self.dta = dta
        # 起始日期
        self.begin_date = datetime.strptime(begin_date, "%Y%m%d")
        # 结束日期，回测区间左闭右闭
        self.end_date = datetime.strptime(end_date, "%Y%m%d")
        # 基准
        self.base = base
        # 成本commission = 0.0004, stamptax = 0.001, slip = 0.001
        self.commission = commission
        self.stamptax = stamptax
        self.slip = slip
        # 是否第二天开盘时交易
        self.use_open = use_open
        # 初始资金
        self.init_cash = init_cash
        # 结果目录
        self.result_path = result_path
        # 模式
        self.mode = mode
        # 交易记录
        self.recode = pd.DataFrame(columns=['channel', 'ts_code', '时间', '买入1/卖出0', '净值'])

    def _calc_cross_section_ic(self, y_pred, y_true):
        pred = np.asarray(y_pred).reshape(-1)
        true = np.asarray(y_true).reshape(-1)
        df = pd.DataFrame({'score': pred, 'label': true})
        df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=['score', 'label'])
        if len(df) < 2:
            return np.nan, np.nan
        if df['score'].nunique() < 2 or df['label'].nunique() < 2:
            return np.nan, np.nan
        ic = df['score'].corr(df['label'], method='pearson')
        rank_ic = df['score'].corr(df['label'], method='spearman')
        return ic, rank_ic

    def _summarize_ic_metrics(self, daily_metrics):
        if daily_metrics is None or daily_metrics.empty:
            return {
                'IC均值': np.nan,
                'ICIR': np.nan,
                'RankIC均值': np.nan,
                'RankICIR': np.nan,
            }

        ic_series = daily_metrics['IC'].dropna()
        rank_ic_series = daily_metrics['RankIC'].dropna()

        ic_std = ic_series.std(ddof=1) if len(ic_series) > 1 else np.nan
        rank_ic_std = rank_ic_series.std(ddof=1) if len(rank_ic_series) > 1 else np.nan
        ic_mean = ic_series.mean() if len(ic_series) > 0 else np.nan
        rank_ic_mean = rank_ic_series.mean() if len(rank_ic_series) > 0 else np.nan
        icir = ic_mean / ic_std if pd.notna(ic_std) and abs(ic_std) > 1e-12 else np.nan
        rank_icir = rank_ic_mean / rank_ic_std if pd.notna(rank_ic_std) and abs(rank_ic_std) > 1e-12 else np.nan

        return {
            'IC均值': ic_mean,
            'ICIR': icir,
            'RankIC均值': rank_ic_mean,
            'RankICIR': rank_icir,
        }

    def _evaluate_signal_ic(self):
        rows = []
        if len(self.dates) <= self.position_cycle + 1:
            empty_daily = pd.DataFrame(columns=['日期', 'IC', 'RankIC', '样本数'])
            return self._summarize_ic_metrics(empty_daily), empty_daily

        for i in range(len(self.dates) - self.position_cycle - 1):
            signal_date = self.dates[i]
            entry_date = self.dates[i + 1]
            exit_date = self.dates[i + self.position_cycle + 1]

            signal_rows = []
            for _, row in self.dta[['ts_code', signal_date]].iterrows():
                stock = row['ts_code']
                score = row[signal_date]

                entry_px = self.open_map.get(entry_date, {}).get(stock, np.nan)
                exit_px = self.open_map.get(exit_date, {}).get(stock, np.nan)
                if pd.isna(score) or pd.isna(entry_px) or pd.isna(exit_px):
                    continue

                forward_return = exit_px / entry_px - 1
                signal_rows.append({
                    '日期': signal_date,
                    'ts_code': stock,
                    'score': score,
                    'label': forward_return,
                })

            if len(signal_rows) < 2:
                rows.append({'日期': signal_date, 'IC': np.nan, 'RankIC': np.nan, '样本数': len(signal_rows)})
                continue

            signal_df = pd.DataFrame(signal_rows)

            if len(signal_df) < 2:
                rows.append({'日期': signal_date, 'IC': np.nan, 'RankIC': np.nan, '样本数': len(signal_df)})
                continue

            ic, rank_ic = self._calc_cross_section_ic(signal_df['score'].values, signal_df['label'].values)
            rows.append({'日期': signal_date, 'IC': ic, 'RankIC': rank_ic, '样本数': len(signal_df)})

        daily_metrics = pd.DataFrame(rows)
        return self._summarize_ic_metrics(daily_metrics), daily_metrics

    def calc_all_net_worth(self, price_table, date):
        sum = 0
        for i in range(self.channels_num):
            sum = sum + self.channels[i].calc_net_worth(price_table, date)
        return sum
    
    def fetch_data(self):
        self.dta = self.dta.loc[[i for i in self.dta.index if not self.dta.iat[i, 0].endswith('.BJ')], :]
        # 数据获取
        self.fetcher = get_data.DataFetcher(user='zhangyudong', password='123456')
        data = self.fetcher.get_stock_daily_matrix(factor_names = ['open_hfq', 'close_hfq'], start_time = self.begin_date.date().strftime("%Y.%m.%d"))
        # 开盘价
        tmp = []
        self.open_df = data['open_hfq']
        self.open_df = self.open_df.loc[[i for i in self.open_df.index if not i.endswith('.BJ')], :]
        for i in range(len(self.open_df.columns)):
            tmp.append(self.open_df.columns[i].strftime("%Y%m%d"))
        self.open_df.columns = tmp
        self.open_df.to_csv('data/open_hfq.csv')
        # 收盘价
        tmp = []
        self.close_df = data['close_hfq']
        self.close_df = self.close_df.loc[[i for i in self.close_df.index if not i.endswith('.BJ')], :]
        for i in range(len(self.close_df.columns)):
            tmp.append(self.close_df.columns[i].strftime("%Y%m%d"))
        self.close_df.columns = tmp

        self.exp_df = self.fetcher.get_index_daily_matrix(factor_names = ['open'], start_time = self.begin_date.date().strftime("%Y.%m.%d"))['open']
        self.exp_df.to_csv('data/exp.csv')

        # ST数据
        tmp = []
        self.ST_df = self.fetcher.get_stock_is_ST(start_time = self.begin_date.date().strftime("%Y.%m.%d"))
        self.ST_df = self.ST_df.loc[[i for i in self.ST_df.index if not i.endswith('.BJ')], :]
        for i in range(len(self.ST_df.columns)):
            tmp.append(self.ST_df.columns[i].strftime("%Y%m%d"))
        self.ST_df.columns = tmp
        self.ST_df.to_csv('data/ST.csv')
        self.ST_list = self.ST_df.stack().loc[lambda x: x == 1].reset_index()[['level_0', 'level_1']].values.tolist()
        self.ST_list = [tuple(row) for row in self.ST_list]
        self.ST_set = set(self.ST_list)

        # 停牌数据
        tmp = []
        self.suspended_df = self.fetcher.get_stock_is_suspended(start_time = self.begin_date.date().strftime("%Y.%m.%d"))
        self.suspended_df = self.suspended_df.loc[[i for i in self.suspended_df.index if not i.endswith('.BJ')], :]
        for i in range(len(self.suspended_df.columns)):
            tmp.append(self.suspended_df.columns[i].strftime("%Y%m%d"))
        self.suspended_df.columns = tmp
        self.suspended_df.to_csv('data/suspended.csv')
        self.suspended_list = self.suspended_df.stack().loc[lambda x: x == 1].reset_index()[['level_0', 'level_1']].values.tolist()
        self.suspended_list = [tuple(row) for row in self.suspended_list]
        self.suspended_set = set(self.suspended_list)
    def gen_pic(self, df, directory, use_delta = 0): 
        returns = df.divide(df.iloc[:, 0], axis=0) - 1
        if use_delta == 1:
            returns.iloc[-1] = df.iloc[-1]
        plt.figure(figsize=(12, 6))
        plt.gca().xaxis.set_major_locator(AutoDateLocator())
        # Plot each line
        for stock in returns.index:
            plt.plot(returns.columns, returns.loc[stock], label=stock, marker='o', markersize=2)
        # Add chart elements
        plt.title(str(self.channels_num) + ' Channels - ' + str(self.stocks_num) + ' Stocks - ' + str(self.position_cycle) + ' Days Cycle - Return Comparison', fontsize=15)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Return', fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle='--', alpha=0.7)

        # Rotate x-axis date labels
        plt.xticks(rotation=45)

        # Display chart
        plt.tight_layout()
        plt.savefig(directory)

    def run_backtest(self, channels_num, stocks_num, position_cycle):
        # 通道数量
        self.channels_num = channels_num
        # 每个通道购买股票数量
        self.stocks_num = stocks_num
        # 持仓周期
        self.position_cycle = position_cycle
        # 总买入金额与卖出金额，用于计算换手率
        self.total_buy = 0
        self.total_sell = 0

        self.recode = pd.DataFrame(columns=['channel', 'ts_code', '时间', '买入1/卖出0', '净值'])
        # 初始化channels
        self.channels = []
        for i in range(self.channels_num):
            self.channels.append(channel.channel(stocks_num = self.stocks_num, position_cycle = self.position_cycle, cd = i + 1, commission = self.commission, stamptax = self.stamptax, slip = self.slip, cash = self.init_cash / self.channels_num, mode = self.mode))

        # 开始模拟
        # print(self.dta.columns)
        # for i in self.open_map:
            # print(i)
        # dta 含一列 ts_code，日期列数量应与 open_map 的日期数量一致
        assert(len(self.dta.columns) - 1 == len(self.open_map))

        net_worth_map = {}
        net_worth_map[self.dates[0]] = self.init_cash
        stocks_net_worth_df = []
        all_stocks_net_worth_df = pd.DataFrame(0.0, index = self.dta['ts_code'], columns = self.dates)
        for i in range(self.channels_num):
            stocks_net_worth_df.append(pd.DataFrame(0.0, index = self.dta['ts_code'], columns = self.dates))

        for i in range(len(self.dates) - 1):
            now = self.dates[i]
            next_day = self.dates[i + 1]
            # print(next_day)
            # print(self.calc_all_net_worth(self.open_map, next_day))

            for j in range(self.channels_num):
                self.channels[j].next_day()
                # if self.channels[j].ready() and datetime.strptime(next_day, "%Y%m%d").date().weekday() == 4:
                if self.channels[j].ready():
                    if self.channels[j].empty():
                        self.total_buy += self.channels[j].buy(self.dta, self.open_map, now, next_day, self.stocks_num, j, next_day, self.recode)
                    else:
                        tmp_buy, tmp_sell = self.channels[j].sell_and_buy(self.dta, self.open_map, now, next_day, j, next_day, self.recode)
                        self.total_buy += tmp_buy
                        self.total_sell += tmp_sell
                    self.channels[j].reset()

            net_worth_map[next_day] = self.calc_all_net_worth(self.open_map, next_day)   
            
            for j in range(self.channels_num):
                self.channels[j].update_net_worth_df(self.open_map, next_day, stocks_net_worth_df[j], all_stocks_net_worth_df)
        
        # 用于汇总后的收益率曲线
        self.result_list.append(net_worth_map)
        self.result_index_list.append(str(self.channels_num) + '-' + str(self.stocks_num) + '-' + str(self.position_cycle))
        # 生成当前策略的收益率曲线
        net_worth_delta = {}
        for i in range(0, len(self.dates)):
            net_worth_delta[self.dates[i]] = net_worth_map[self.dates[i]] / net_worth_map[self.dates[0]] - self.exp_map[self.base][self.dates[i]] / self.exp_map[self.base][self.dates[0]]
        net_worth_df = pd.DataFrame([net_worth_map, self.exp_map[self.base], net_worth_delta], index=['Strategy', self.base, 'Excess'])

        # 生成net_worth记录
        # net_worth_df.to_csv(self.result_path + '/net_worth' + str(self.channels_num) + '_' + str(self.stocks_num) + '_' + str(self.position_cycle) + '_' + '.csv', encoding='utf_8_sig')
        # for i in range(self.channels_num):
        #     stocks_net_worth_df[i].to_csv(self.result_path + '/channel_' + str(i) + '_stocks_net_worth_' + str(self.channels_num) + '_' + str(self.stocks_num) + '_' + str(self.position_cycle) + '_' + '.csv')


        self.gen_pic(net_worth_df, self.result_path + '/' + str(self.channels_num) + '_' + str(self.stocks_num) + '_' + str(self.position_cycle) + '_' + '.png', use_delta = 1)
        plt.close(1)
        # 生成持仓表
        tmp_df = all_stocks_net_worth_df.reset_index().melt(
            id_vars='ts_code',
            var_name='日期',
            value_name='净值'
        ).query('净值 != 0')
        # tmp_df.to_csv(self.result_path + '/持仓表_' + str(self.channels_num) + '_' + str(self.stocks_num) + '_' + str(self.position_cycle) + '.csv', encoding='utf_8_sig')

        # self.recode.to_csv(self.result_path + '交易记录' + str(self.channels_num) + '_' + str(self.stocks_num) + '_' + str(self.position_cycle) + '.csv', encoding='utf_8_sig')
        
        # 生成IC评估记录
        ic_summary, daily_ic_df = self._evaluate_signal_ic()
        # pd.DataFrame([ic_summary]).to_csv(
        #     self.result_path + '/IC汇总_' + str(self.channels_num) + '_' + str(self.stocks_num) + '_' + str(self.position_cycle) + '.csv',
        #     encoding='utf_8_sig',
        #     index=False,
        # )
        # daily_ic_df.to_csv(
        #     self.result_path + '/每日IC_' + str(self.channels_num) + '_' + str(self.stocks_num) + '_' + str(self.position_cycle) + '.csv',
        #     encoding='utf_8_sig',
        #     index=False,
        # )

        # 计算投资组合换手率
        avg_asset = 0
        for i in range(1, len(self.dates)):
            avg_asset += net_worth_map[self.dates[i]]
        avg_asset /= (len(self.dates) - 1)
        total_turnover = (self.total_buy + self.total_sell) / avg_asset / 2
        annual_turnover = total_turnover / (len(self.dates) - 1) * 252
        monthly_turnover = total_turnover / (len(self.dates) - 1) * 20
        weekly_turnover = total_turnover / (len(self.dates) - 1) * 5
        daily_turnover = total_turnover / (len(self.dates) - 1)
        # 计算指标
        return_map = {}
        # 计算年化收益率
        annualized_return = (net_worth_map[self.dates[len(self.dates) - 1]] / net_worth_map[self.dates[0]]) ** (252 / (len(self.dates) - 1)) - 1
        # 计算超额收益率
        excess_return = net_worth_map[self.dates[len(self.dates) - 1]] / net_worth_map[self.dates[0]] - self.exp_map[self.base][self.dates[len(self.dates) - 1]] / self.exp_map[self.base][self.dates[0]]
        # 计算年化超额收益率
        annualized_excess_return = (1 + excess_return) ** (252 / (len(self.dates) - 1)) - 1
        # 计算最大回撤
        maximum_drawdown = 0
        history_max = net_worth_map[self.dates[0]]
        for i in range(1, len(self.dates)):
            current_value = net_worth_map[self.dates[i]]
            if current_value > history_max:
                history_max = current_value
            drawdown = (history_max - current_value) / history_max
            maximum_drawdown = max(maximum_drawdown, drawdown)
        # 计算波动率
        daily_volatility = np.std([net_worth_map[self.dates[i]] / net_worth_map[self.dates[i - 1]] - 1 for i in range(1, len(self.dates))], ddof=1)
        annualized_volatility = daily_volatility * np.sqrt(252)
        # 计算日胜率
        win_time = 0
        for i in range(1, len(self.dates)):
            if net_worth_map[self.dates[i]] / net_worth_map[self.dates[i - 1]] > self.exp_map[self.base][self.dates[i]] / self.exp_map[self.base][self.dates[i - 1]]:
                win_time = win_time + 1
        win_rate_daily = win_time / (len(self.dates) - 1) 
        # 计算周胜率（5个交易日）   
        win_time = 0
        all_time = 0
        for i in range(1, len(self.dates)):
            if i % 5 == 0:
                all_time = all_time + 1
                if net_worth_map[self.dates[i]] / net_worth_map[self.dates[i - 5]] > self.exp_map[self.base][self.dates[i]] / self.exp_map[self.base][self.dates[i - 5]]:
                    win_time = win_time + 1
        win_rate_weekly = win_time / all_time
        # 计算月胜率（21个交易日）
        win_time = 0
        all_time = 0
        for i in range(1, len(self.dates)):
            if i % 21 == 0:
                all_time = all_time + 1
                if net_worth_map[self.dates[i]] / net_worth_map[self.dates[i - 21]] > self.exp_map[self.base][self.dates[i]] / self.exp_map[self.base][self.dates[i - 21]]:
                    win_time = win_time + 1
        win_rate_monthly = win_time / all_time
        # 计算累计胜率
        win_time = 0
        for i in range(1, len(self.dates)):
            if net_worth_map[self.dates[i]] / net_worth_map[self.dates[0]] > self.exp_map[self.base][self.dates[i]] / self.exp_map[self.base][self.dates[0]]:
                win_time = win_time + 1
        win_rate_cumulative = win_time / (len(self.dates) - 1)

        return_map['年化收益率'] = annualized_return
        return_map['年化超额收益率'] = annualized_excess_return
        return_map['最大回撤'] = maximum_drawdown
        return_map['年化波动率'] = annualized_volatility
        return_map['日胜率'] = win_rate_daily
        return_map['周胜率'] = win_rate_weekly
        return_map['月胜率'] = win_rate_monthly
        return_map['累计胜率'] = win_rate_cumulative
        return_map['年换手率'] = annual_turnover
        return_map['月换手率'] = monthly_turnover
        return_map['周换手率'] = weekly_turnover
        return_map['日换手率'] = daily_turnover
        return_map['夏普比率'] = (annualized_return - 0.02) / annualized_volatility
        return_map.update(ic_summary)
        return return_map
    def run_backtest_list(self, channels_num_list, stocks_num_list, position_cycle_list):
        self.fetch_data()

        self.Stock_ID = self.dta['ts_code'].tolist()
        self.dates = self.dta.columns.tolist()
        dates_num = len(self.dates) - 1
        begin_id = 1
        end_id = 0

        #确保回测区间在打分表时间范围内
        assert(self.begin_date >= datetime.strptime(self.dates[1], "%Y%m%d"))
        assert(self.end_date <= datetime.strptime(self.dates[dates_num], "%Y%m%d"))

        for i in range(1, dates_num + 1):
            if self.begin_date > datetime.strptime(self.dates[i], "%Y%m%d"):
                begin_id = begin_id + 1
            if self.end_date >= datetime.strptime(self.dates[i], "%Y%m%d"):
                end_id = end_id + 1
        
        assert(begin_id <= end_id) #确保回测区间非空
        self.dta = self.dta.iloc[:, [True if i == 0 or (begin_id <= i <= end_id) else False for i in range(len(self.dta.columns))]]
        # open_map[日期][股票代码]->股价
        # dates 需要用到的日期  
        self.open_map = {}
        self.dates = []
        for i in range(len(self.open_df.columns)):
            if datetime.strptime(self.open_df.columns[i], "%Y%m%d") >= self.begin_date and (datetime.strptime(self.open_df.columns[i], "%Y%m%d") <= self.end_date or datetime.strptime(self.open_df.columns[i - 1], "%Y%m%d") <= self.end_date):
                self.open_map[self.open_df.columns[i]] = dict(zip(self.open_df.index.tolist(), self.open_df.iloc[:, i].tolist()))
                self.dates.append(self.open_df.columns[i])

        for i in range(len(self.dates) - 1):
            now = self.dates[i]
            next_day = self.dates[i + 1]
            #处理不可交易的股票（股价为nan的股票）
            for j in range(len(self.dta)):
                if not self.Stock_ID[j] in self.open_map[next_day] or math.isnan(self.open_map[next_day][self.Stock_ID[j]]):
                    self.dta.iat[j, i + 1] = 0
                if (self.Stock_ID[j], next_day) in self.ST_set:
                    self.dta.iat[j, i + 1] = 0
                # if (self.Stock_ID[j], next_day) in self.suspended_set:
                #     self.dta.iat[j, i + 1] = 0
        # exp_map[指数代码][日期]
        self.exp_map = {}
        for i in self.exp_df.index:
            self.exp_map[i] = {}
            for j in range(len(self.exp_df.columns)):
                if not math.isnan(self.exp_df.at[i, self.exp_df.columns[j]]) and self.exp_df.columns[j] >= self.begin_date and (self.exp_df.columns[j] <= self.end_date or self.exp_df.columns[j - 1] <= self.end_date):
                    self.exp_map[i][self.exp_df.columns[j].strftime("%Y%m%d")] = self.exp_df.at[i, self.exp_df.columns[j]]

        # 日期对齐：评分表/开盘价/基准指数三者取交集，避免 KeyError: 'YYYYMMDD'
        if self.base not in self.exp_map:
            raise ValueError(f"基准指数 {self.base} 不在指数数据中，请检查 base 配置。")

        score_dates = set(self.dta.columns.tolist()[1:])  # 去掉 ts_code
        open_dates = set(self.open_map.keys())
        bench_dates = set(self.exp_map[self.base].keys())
        valid_dates = sorted(list(score_dates & open_dates & bench_dates))

        if len(valid_dates) < 2:
            raise ValueError(
                f"可回测日期不足。score/open/bench 交集天数={len(valid_dates)}，"
                f"请检查配置区间与基准数据覆盖范围（base={self.base}）。"
            )

        removed_dates = [d for d in self.dates if d not in set(valid_dates)]
        if len(removed_dates) > 0:
            print(
                f"[WARN] 日期未对齐，已自动剔除 {len(removed_dates)} 个交易日。"
                f"示例: {removed_dates[:3]} ... {removed_dates[-3:]}"
            )

        self.dta = self.dta[['ts_code'] + valid_dates]
        self.open_map = {d: self.open_map[d] for d in valid_dates}
        self.dates = valid_dates
        for idx in self.exp_map:
            self.exp_map[idx] = {d: self.exp_map[idx][d] for d in valid_dates if d in self.exp_map[idx]}

        self.result_list = []
        self.result_index_list = []
        return_map = {}
        for channels_num in channels_num_list:
            for stocks_num in stocks_num_list:
                for position_cycle in position_cycle_list:
                    return_map[(channels_num, stocks_num, position_cycle)] = self.run_backtest(channels_num = channels_num, stocks_num = stocks_num, position_cycle = position_cycle)

        self.result_list.append(self.exp_map[self.base])
        self.result_index_list.append(self.base)
        result_df = pd.DataFrame(self.result_list, index = self.result_index_list)

        self.gen_pic(result_df, self.result_path + '/result.png')
        plt.close(1)
        return return_map