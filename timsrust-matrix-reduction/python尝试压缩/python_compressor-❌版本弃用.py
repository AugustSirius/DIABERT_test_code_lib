# import timstof_PASEF_20250506

# # Load TimsTOF data
# bruker_d_folder_name = '/Users/augustsirius/Desktop/DIABERT_test_code_lib/DIA_sample.d'
# timstof_data = timstof_PASEF_20250506.TimsTOF(bruker_d_folder_name)

# total = timstof_data[:, :, :, :][['rt_values_min', 'mobility_values', 'mz_values', 'intensity_values']]

# print(total)

import timstof_PASEF_20250506
from copy import deepcopy
bruker_d_folder_name = '/Users/augustsirius/Desktop/DIABERT_test_code_lib/DIA_sample.d'
timstof_data = timstof_PASEF_20250506.TimsTOF(bruker_d_folder_name)
df = deepcopy(timstof_data[:,:,:,:,:])

import pandas as pd
import numpy as np

def process_data_with_groupby(temp_df: pd.DataFrame, bin_sizes: dict) -> pd.DataFrame:
    print("开始使用 pandas.groupby 方法处理数据...")
    if temp_df.empty:
        print("输入的DataFrame为空，返回一个空的DataFrame。")
        return pd.DataFrame(columns=['mz_values', 'rt_values_min', 'mobility_values', 'intensity_values'])
    # 创建整数坐标
    temp_df['mz_coord'] = np.ceil(temp_df['mz_values'] * 1000).astype(int)
    temp_df['im_coord'] = np.ceil(temp_df['mobility_values'] * 1000).astype(int)
    temp_df['rt_coord'] = np.round(temp_df['rt_values_min']).astype(int)
    # --- 2. 计算分箱ID ---
    mz_bin_size = bin_sizes['mz_values']
    rt_bin_size = bin_sizes['rt_values_min']
    im_bin_size = bin_sizes['mobility_values']
    temp_df['mz_bin_id'] = temp_df['mz_coord'] // mz_bin_size
    temp_df['rt_bin_id'] = temp_df['rt_coord'] // rt_bin_size
    temp_df['im_bin_id'] = temp_df['im_coord'] // im_bin_size
    print("分箱ID计算完成。")
    # --- 3. 分组与聚合 ---
    # 按分箱ID对intensity求和
    # .sum()会自动处理重复的坐标点
    grouped = temp_df.groupby(['mz_bin_id', 'rt_bin_id', 'im_bin_id'])
    summed_intensity = grouped['intensity_values'].sum()
    # 过滤掉加和后强度为0或接近0的分箱 (可选，但推荐)
    summed_intensity = summed_intensity[summed_intensity > 1e-9]
    if summed_intensity.empty:
        print("处理后没有发现任何非零强度的点。")
        return pd.DataFrame(columns=['mz_values', 'rt_values_min', 'mobility_values', 'intensity_values'])
    # 将结果转换为DataFrame，并将多重索引重置为列
    result_df = summed_intensity.reset_index()
    print(f"分组与聚合完成。发现 {len(result_df)} 个非空分箱。")
    # --- 4. 计算新中心坐标 ---
    # 计算每个维度的中心偏移量
    mz_center_offset = mz_bin_size // 2
    rt_center_offset = rt_bin_size // 2
    im_center_offset = im_bin_size // 2
    # 根据分箱ID计算新的中心整数坐标
    result_df['new_mz_coord'] = result_df['mz_bin_id'] * mz_bin_size + mz_center_offset
    result_df['new_rt_coord'] = result_df['rt_bin_id'] * rt_bin_size + rt_center_offset
    result_df['new_im_coord'] = result_df['im_bin_id'] * im_bin_size + im_center_offset
    # --- 5. 构建最终结果 ---
    # 创建最终的DataFrame，并转换回原始单位
    final_df = pd.DataFrame({
        'mz_values': result_df['new_mz_coord'] / 1000.0,
        'rt_values_min': result_df['new_rt_coord'].astype(float), # 保持rt为浮点数
        'mobility_values': result_df['new_im_coord'] / 1000.0,
        'intensity_values': result_df['intensity_values']
    })
    # 按照题目要求的顺序排序（可选，但推荐）
    final_df = final_df.sort_values(by=['mz_values', 'rt_values_min', 'mobility_values']).reset_index(drop=True)
    print(f"处理完成！原始数据 {len(df)} 行，最终结果 {len(final_df)} 行。")
    return final_df
binning_parameters = {'mz_values': 5, 'rt_values_min': 3, 'mobility_values': 3}
# 3. 调用函数进行处理

result_df = process_data_with_groupby(df, binning_parameters)
