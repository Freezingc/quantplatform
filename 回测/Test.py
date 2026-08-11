import backtest
import pandas as pd
import os
import json

with open('config.json', 'r', encoding='utf-8') as file:
    config = json.load(file)

return_list = []

for name in config:

    dta = pd.read_csv('input/' + name + '.csv')
    if not os.path.exists('output/' + name):
        os.makedirs('output/' + name)

    base = '000001.SH'
    if 'base' in config[name]:
        base = config[name]['base']

    test = backtest.backtest(dta = dta, begin_date = config[name]['begin_date'], end_date = config[name]['end_date'], base = base, init_cash=1000000, result_path = 'output/' + name, commission = 0.0004, stamptax = 0.001, slip = 0.001, mode =0)
    # test = backtest.backtest(dta = dta, begin_date = config[name]['begin_date'], end_date = config[name]['end_date'], base = base, init_cash=1000000, result_path = 'output/' + name, commission = 0, stamptax = 0, slip = 0, mode = 0)
    # test = backtest.backtest(dta = dta, begin_date = config[name]['begin_date'], end_date = config[name]['end_date'], base = base, init_cash=10000000, result_path = 'output/' + name, commission = 0.0004, stamptax = 0.001, slip = 0.001, mode = 0)

    return_map = test.run_backtest_list(channels_num_list = [config[name]['channel_num']], stocks_num_list = [3,5,7,15,30,50], position_cycle_list = [config[name]['position_cycle']])
    # return_map = test.run_backtest_list(channels_num_list = [config[name]['channel_num']], stocks_num_list = [5], position_cycle_list = [config[name]['position_cycle']])
    for key in return_map:
        tmp = {}
        tmp['策略'] = name
        tmp['通道数量'] = key[0]
        tmp['股票数量'] = key[1]
        tmp['持仓周期'] = key[2]
        for i in return_map[key]:
            tmp[i] = return_map[key][i]
        return_list.append(tmp)

    print(name + '已完成')

return_df = pd.DataFrame(return_list)
return_df.to_csv('output/策略汇总.csv', encoding='utf_8_sig')


