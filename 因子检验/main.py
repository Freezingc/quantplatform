import logging
import sys

from src.batch_factor_analysis_pipline import run_factor_analysis_from_files


# 配置日志
def setup_logging():
    """配置日志系统"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('factor_analysis.log', encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )


if __name__ == "__main__":
    # 参数设置，不设置就用默认的
    params = {
        'group_num': 10,
        'factor_direction': 1,
        'benchmark': '中证500',
        'data_preprocess': True,
        'ic_horizons': [1, 3, 5, 10, 20]
    }

    setup_logging()
    summary_results = run_factor_analysis_from_files(input_folder='factor_data_files', params=params)

    # 显示结果摘要
    if 'ic_summary' in summary_results:
        ic_df = summary_results['ic_summary']

        for _, row in ic_df.iterrows():
            print(f"\n{'='*50}")
            print(f"多持有期IC分析结果汇总：{row.get('factor_name', 'Unknown')}")
            print(f"{'='*50}")
        
            for key, value in row.items():
                if key != 'factor_name':
                    if isinstance(value, float):
                        print(f"{key}: {value:.4f}")
                    else:
                        print(f"{key}: {value}")
        
        print(f"{'='*50}")

    if 'full_summary' in summary_results:
        full_df = summary_results['full_summary']

        for _, row in full_df.iterrows():
            print(f"\n{'='*50}")
            print(f"1D完整效果分析结果汇总：{row.get('factor_name', 'Unknown')}")
            print(f"{'='*50}")
        
            for key, value in row.items():
                if key != 'factor_name':
                    if isinstance(value, float):
                        print(f"{key}: {value:.4f}")
                    else:
                        print(f"{key}: {value}")
        
        print(f"{'='*50}")


# nohup python main.py > main.log 2>&1 &