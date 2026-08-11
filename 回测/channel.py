import math
import pandas as pd

class channel:
    def __init__(self, stocks_num, position_cycle, cd, commission, stamptax, slip, mode, cash = 0):
        self.stock_list = []
        self.stocks_num = stocks_num
        self.position_cycle = position_cycle
        self.cd = cd
        self.commission = commission
        self.stamptax = stamptax
        self.slip = slip
        self.cash = cash
        self.mode = mode
        self.eps = 1e-6

    def _get_price(self, price_table, date, stock):
        day_map = price_table.get(date, None)
        if day_map is None:
            return None
        px = day_map.get(stock, None)
        if px is None:
            return None
        if isinstance(px, float) and math.isnan(px):
            return None
        return px

    def _select_top_tradable(self, dta, dta_date, buy_num, price_table, price_date):
        ranked = dta.sort_values(by=dta_date, ascending=False).iloc[:, 0].tolist()
        picked = []
        for stock in ranked:
            if self._get_price(price_table, price_date, stock) is None:
                continue
            picked.append(stock)
            if len(picked) >= buy_num:
                break
        return picked
    
    def calc_net_worth(self, price_table, date):
        sum = self.cash
        for i in range(len(self.stock_list)):
            px = self._get_price(price_table, date, self.stock_list[i][0])
            if px is not None:
                self.stock_list[i][2] = px
            sum = sum + self.stock_list[i][2] * self.stock_list[i][1]
        return sum
    
    def buy(self, dta, price_table, dta_date, price_date, buy_num, channel, date, recode):
        total_buy = 0

        top_stocks = self._select_top_tradable(dta, dta_date, buy_num, price_table, price_date)
        if len(top_stocks) == 0:
            return 0
        cash_pre_stock = self.cash / len(top_stocks)
        for i in range(len(top_stocks)):
            stock = top_stocks[i]
            px = self._get_price(price_table, price_date, stock)
            if px is None:
                continue
            # num = math.floor(cash_pre_stock / (price_table[price_date][stock] * (1 + self.commission + self.slip) * 100)) * 100
            num = math.floor(cash_pre_stock / (px * (1 + self.commission + self.slip))) #暂时不考虑整100股买
            if num <= 0:
                continue
            self.stock_list.append([stock, num, px])
            self.cash = self.cash - px * num * (1 + self.commission + self.slip) # 手续费，滑点
            total_buy += num * (px * (1 + self.commission + self.slip))
            recode.loc[len(recode) + 1] = [channel, stock, date, 1, num * (px * (1 + self.commission + self.slip))]

        # assert(abs(self.cash) < self.eps) # 暂时不考虑整数/手买，因此不应该有剩余现金
        return total_buy
    
    def sell_and_buy(self, dta, open_map, dta_date, price_date, channel, date, recode):
        total_buy = 0
        total_sell = 0

        # 允许持仓数量小于目标值（退市/无行情时自动跳过）
        assert(len(self.stock_list) <= self.stocks_num)
        worst_set = set(dta.sort_values(by = dta_date).head(int(len(dta) * 0.8)).iloc[:, 0].tolist())
        # 模式1，调仓重叠部分省手续费，保持仓位平衡
        if self.mode == 1 or self.mode == 2:
            buy_num = 0
            tmp_list = []
            for i in range(len(self.stock_list)):
                px = self._get_price(open_map, price_date, self.stock_list[i][0])
                if px is not None and (self.mode == 1 or self.mode == 2 and self.stock_list[i][0] in worst_set):
                    buy_num = buy_num + 1
                else:
                    tmp_list.append(self.stock_list[i])
                    # 输出无法出售的股票，可能退市
                    # print(self.stock_list[i][0])
            if buy_num == 0:
                return 0, 0
            top_stocks = self._select_top_tradable(dta, dta_date, buy_num, open_map, price_date)
            buy_num = len(top_stocks)
            if buy_num == 0:
                self.stock_list = tmp_list
                return 0, 0
            position_map = {}
            for i in range(buy_num):
                position_map[top_stocks[i]] = 0
            
            d_list = []

            #Ax = B
            B = 0
            for i in range(len(self.stock_list)):
                px = self._get_price(open_map, price_date, self.stock_list[i][0])
                if px is not None and (self.mode == 1 or self.mode == 2 and self.stock_list[i][0] in worst_set):
                    if self.stock_list[i][0] in top_stocks:
                        position_map[self.stock_list[i][0]] = self.stock_list[i][1] * px
                        d_list.append(self.stock_list[i][1] * px)
                        B = B + px * self.stock_list[i][1]
                    else:
                        self.cash = self.cash + px * self.stock_list[i][1] * (1 - self.commission - self.slip - self.stamptax)
                        B = B + px * self.stock_list[i][1] * (1 - self.commission - self.slip - self.stamptax)
                        total_sell += px * self.stock_list[i][1]
            for i in range(buy_num):
                if abs(position_map[top_stocks[i]]) < self.eps:
                    d_list.append(0)
            d_list.sort()
            # print('d_list:')
            # print(d_list)
            # print('top:')
            # print(top_stocks)
            A = buy_num
            for i in range(buy_num):
                A = A - (self.commission + self.slip + self.stamptax)
                B = B - position_map[top_stocks[i]] * (self.commission + self.slip + self.stamptax)
            assert(A != 0)
            x = B / A
            v = -1
            if len(d_list) == 0 or d_list[0] - x >= -self.eps:
                # print('x:', x, '<0')
                v = x 
            for i in range(len(d_list)):
                A = A + (self.commission + self.slip + self.stamptax) + (self.commission + self.slip)
                B = B + d_list[i] * (self.commission + self.slip + self.stamptax) + d_list[i] * (self.commission + self.slip)
                x = B / A
                if x - d_list[i] >= -self.eps and (i == len(d_list) - 1 or d_list[i + 1] - x >= -self.eps):
                    v = x
                    # if i == len(d_list) - 1:
                    #     print(d_list[i], '<=', 'x:', x)
                    # else:
                    #     print(d_list[i], '<=', 'x:', x, '<=', d_list[i + 1])
            
            self.stock_list = tmp_list
            for i in range(buy_num):
                px = self._get_price(open_map, price_date, top_stocks[i])
                if px is None:
                    continue
                self.stock_list.append([top_stocks[i], v / px, px])
                if position_map[top_stocks[i]] < v:
                    total_buy += v - position_map[top_stocks[i]]
                else:
                    total_sell += position_map[top_stocks[i]] - v
            self.cash = 0
            return total_buy, total_sell
        
        buy_num = 0
        tmp_list = []
        for i in range(len(self.stock_list)):
            px = self._get_price(open_map, price_date, self.stock_list[i][0])
            if px is not None: # 暂未考虑无法出售的情况
                self.cash = self.cash + px * self.stock_list[i][1] * (1 - self.commission - self.slip - self.stamptax)
                buy_num = buy_num + 1
                total_sell += px * self.stock_list[i][1]
                recode.loc[len(recode) + 1] = [channel, self.stock_list[i][0], date, 0, px * self.stock_list[i][1]]
            else:
                tmp_list.append(self.stock_list[i])

        # print(tmp_list)
        self.stock_list = tmp_list
        if buy_num > 0:
            total_buy += self.buy(dta, open_map, dta_date, price_date, buy_num, channel, date, recode)
        return total_buy, total_sell

    def next_day(self):
        self.cd = self.cd - 1
    
    def ready(self):
        if self.cd <= 0:
            return True
        else:
            return False
    
    def reset(self):
        self.cd = self.position_cycle

    def empty(self):
        if len(self.stock_list) == 0:
            return True
        else:
            return False
        
    def update_net_worth_df(self, price_table, date, df1, df2):
        updates = {}
        for stock in self.stock_list:
            stock_id, stock_qty, stock_price = stock
            tmp = stock_qty * stock_price
            if stock_id not in updates:
                updates[stock_id] = 0
            updates[stock_id] += tmp
        update_series = pd.Series(updates)
        df1.loc[update_series.index, date] += update_series
        df2.loc[update_series.index, date] += update_series
