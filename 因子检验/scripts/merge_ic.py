import os
import pandas as pd

def batch_read_and_merge(input_folder: str) -> pd.DataFrame:
    all_ic_data = pd.DataFrame()
    for factor_name in os.listdir(input_folder):
        file_path = os.path.join(input_folder, factor_name, '1D', 'ic_time_series.csv')
        ic_data = pd.read_csv(file_path)
        ic_data.insert(0, "factor_name", factor_name)
        all_ic_data = pd.concat([all_ic_data, ic_data], axis=0)

    return all_ic_data

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.dirname(current_dir)
    output_dir = os.path.join(parent_dir, "visualization")

    all_ic_data = batch_read_and_merge(output_dir)
    all_ic_data.to_csv(os.path.join(current_dir, "all_ic.csv"), index=False)
