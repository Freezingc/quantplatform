使用前需要在Test.py修改相关参数，配置config.json，并将数据库模块的get_data.py复制到当前目录下，随后运行Test.py。

### 配置config.json

Xgboost系列数据可以运行auto_config.py自动生存config.json，要求文件名格式形如Xgboost_10_5_predictions，其中10表示回看窗口，5表示持仓周期/通道数量。

config.json格式如下：

{
    "Xgboost_10_5_predictions": {
        "begin_date": "20200703",
        "end_date": "20250529",
        "channel_num": 5,
        "position_cycle": 5
    }
}

其中“Xgboost_10_5_predictions”为文件名，“begin_date”与“end_date”为回测区间，“channel_num”为通道数，“position_cycle”为持仓周期，还可以通过“base”来指定基准。

如有多组数据则用逗号隔开。

### 修改Test.py中的参数

主要参数在代码

```
test = backtest.backtest(dta = dta, begin_date = config[name]['begin_date'], end_date = config[name]['end_date'], base = base, init_cash=1000000, result_path = 'output/' + name, commission = 0.0004, stamptax = 0.001, slip = 0, mode = 0)
```

其中init_cash为起始资金，result_path为结果保存的目录，commission为手续费，stamptax为印花税，slip为滑点，mode为预设模式。

mode=0为每次交易先全部卖出，再进行买入。

mode=1为对于原有股票再次被选中时，仅交易差额，节约部分交易成本。

mode=2为在mode=1的基础上，仅对排名靠后的股票进行换股（此策略不成熟）。

### 路径说明

打分表应放在input文件夹内。

回测需要用到的相关数据，会自动获取并保存在data文件夹内。

回测结果默认保存至output文件夹内。
