"""
测试 get_financial_metrics_matrix_anndate 函数
该函数按照公告日期(ann_date)获取财报数据，与 get_financial_metrics_matrix 按报告期(end_date)不同
"""

import dolphindb as ddb
import os
from dotenv import load_dotenv
from get_data import DataFetcher

def test_financial_metrics_anndate():
    """测试按公告日期获取财报数据"""
    
    # 加载环境变量
  
    username = "zhangyudong"
    password = "123456"
    
    # 连接数据库
    s = ddb.session()
    s.connect("10.176.55.211", 8848, username, password)
    
    try:
        fetcher = DataFetcher(s=s)
        
        print("=" * 80)
        print("测试 get_financial_metrics_matrix_anndate 函数")
        print("=" * 80)
        
        # 测试参数
        factor_names = ["roa", "roe"]  # 测试两个常用的财务指标
        start_time = "2023.01.01"
        end_time = "2023.12.31"
        
        print(f"\n测试参数:")
        print(f"  因子名称: {factor_names}")
        print(f"  开始时间: {start_time}")
        print(f"  结束时间: {end_time}")
        print(f"  注意: 函数会自动提前1年获取数据用于前向填充")
        
        # 调用函数
        print(f"\n正在获取数据...")
        data = fetcher.get_financial_metrics_matrix_anndate(
            factor_names=factor_names,
            start_time=start_time,
            end_time=end_time
        )
        
        # 输出结果
        print(f"\n✅ 数据获取成功!")
        print(f"\n数据形状: {data.shape}")
        print(f"  - 股票数量: {data.shape[0]}")
        print(f"  - 总列数: {data.shape[1]}")
        
        # 查看列的结构（多级索引）
        if not data.empty:
            print(f"\n因子列表:")
            for factor in factor_names:
                factor_cols = data[factor].columns if factor in data.columns.levels[0] else []
                print(f"  - {factor}: {len(factor_cols)} 个公告日期")
                if len(factor_cols) > 0:
                    print(f"    日期范围: {factor_cols[0]} 到 {factor_cols[-1]}")
            
            # 显示前几行数据
            print(f"\n前5个股票的数据预览:")
            print(data.head())
            
            # 查看某个特定股票的数据
            sample_stock = data.index[0]
            print(f"\n示例股票 {sample_stock} 的数据:")
            for factor in factor_names:
                if factor in data.columns.levels[0]:
                    stock_data = data.loc[sample_stock, factor]
                    non_nan = stock_data.notna().sum()
                    print(f"  {factor}: {non_nan}/{len(stock_data)} 个非空数据")
                    # 显示最近几个非空数据
                    valid_data = stock_data.dropna()
                    if len(valid_data) > 0:
                        print(f"    最近的值: {valid_data.iloc[-3:].to_dict()}")
            
            # 统计数据覆盖率
            print(f"\n数据覆盖率统计:")
            for factor in factor_names:
                if factor in data.columns.levels[0]:
                    factor_data = data[factor]
                    total_cells = factor_data.size
                    non_nan_cells = factor_data.notna().sum().sum()
                    coverage = non_nan_cells / total_cells * 100 if total_cells > 0 else 0
                    print(f"  {factor}: {coverage:.2f}% ({non_nan_cells}/{total_cells})")
        
        else:
            print("⚠️  返回的数据为空")
        
        print(f"\n" + "=" * 80)
        print("测试完成!")
        print("=" * 80)
        
        return data
        
    except Exception as e:
        print(f"\n❌ 发生错误: {e}")
        import traceback
        traceback.print_exc()
        return None
        
    finally:
        s.close()
        print("\n数据库连接已关闭")




if __name__ == "__main__":
    # 基本测试
    data = test_financial_metrics_anndate()
    
    # 对比测试（可选）
    print("\n" + "#" * 80)
    print("# 是否要进行对比测试? (需要额外时间)")
    print("#" * 80)
    # compare_anndate_vs_enddate()  # 取消注释以运行对比测试

