import timstof_PASEF_20250506
from copy import deepcopy
bruker_d_folder_name = '/Users/augustsirius/Desktop/DIABERT_test_code_lib/DIA_sample.d'
timstof_data = timstof_PASEF_20250506.TimsTOF(bruker_d_folder_name)
df = deepcopy(timstof_data[:,:,:,:,:])


import pandas as pd
import numpy as np
def process_mixed_grid_data(temp_df: pd.DataFrame, bin_sizes: dict) -> pd.DataFrame:
    print("开始处理混合坐标数据（m/z, im离散; rt连续）...")
    if temp_df.empty:
        print("输入的DataFrame为空，返回一个空的DataFrame。")
        return pd.DataFrame(columns=['mz_values', 'rt_values_min', 'mobility_values', 'intensity_values'])
    temp_df['mz_coord'] = np.ceil(temp_df['mz_values'] * 1000).astype(int)
    temp_df['im_coord'] = np.ceil(temp_df['mobility_values'] * 1000).astype(int)
    # --- 1. 离散维度分箱 (m/z, im) ---
    print("步骤1: 为m/z和im计算分箱ID...")
    mz_bin_size = bin_sizes['mz_values']
    im_bin_size = bin_sizes['mobility_values']
    temp_df['mz_bin_id'] = temp_df['mz_coord'] // mz_bin_size
    temp_df['im_bin_id'] = temp_df['im_coord'] // im_bin_size
    # temp_df['mz_bin_id'] = temp_df['mz_values'].astype(int) // mz_bin_size
    # temp_df['im_bin_id'] = temp_df['mobility_values'].astype(int) // im_bin_size
    # --- 2. 连续维度分组 (rt) ---
    print("步骤2: 在每个(m/z, im)箱内按rt排序并分组...")
    rt_group_size = bin_sizes['rt_values_min']
    # 首先按分箱ID和rt排序，确保后续cumcount的顺序正确
    temp_df = temp_df.sort_values(by=['mz_bin_id', 'im_bin_id', 'rt_values_min'])
    # 在每个(m/z, im)箱内，计算rt的排名，然后除以rt组大小得到rt分组ID
    # cumcount() 会计算每个元素在分组内的序号 (0, 1, 2, ...)
    temp_df['rt_group_id'] = temp_df.groupby(['mz_bin_id', 'im_bin_id']).cumcount() // rt_group_size
    # --- 3. 最终聚合 ---
    print("步骤3: 按所有ID进行最终分组和聚合...")
    # 定义聚合操作：intensity求和，坐标取中位数
    # 中位数对于找到中心点和处理不完整分组非常稳健
    agg_operations = {
        'intensity_values': 'sum',
        'mz_values': 'median',
        'rt_values_min': 'median',
        'mobility_values': 'median'
    }
    # 按所有ID进行分组
    final_groups = temp_df.groupby(['mz_bin_id', 'im_bin_id', 'rt_group_id'])
    # 执行聚合
    result_df = final_groups.agg(agg_operations)
    # 过滤掉intensity为0的行
    result_df = result_df[result_df['intensity_values'] > 1e-9].copy()
    # 将m/z和im转换为整数，因为中位数可能会产生.5的浮点数
    result_df['mz_values'] = np.round(result_df['mz_values']).astype(int)
    result_df['mobility_values'] = np.round(result_df['mobility_values']).astype(int)
    print(f"处理完成！原始数据 {len(df)} 行，最终结果 {len(result_df)} 行。")
    return result_df.reset_index(drop=True)

binning_parameters = {'mz_values': 5, 'mobility_values': 3, 'rt_values_min': 3}
# 3. 调用函数进行处理
result_df = process_mixed_grid_data(df, binning_parameters)
