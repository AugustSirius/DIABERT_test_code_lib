import timstof_PASEF_20250506
import pandas as pd
import numpy as np
from copy import deepcopy

# 加载数据
bruker_d_folder_name = '/Users/augustsirius/Desktop/DIABERT_test_code_lib/DIA_sample.d'
timstof_data = timstof_PASEF_20250506.TimsTOF(bruker_d_folder_name)

# 提取所有数据
df = deepcopy(timstof_data[:,:,:,:,:])

# 分离 MS1 和 MS2 数据
print("分离 MS1 和 MS2 数据...")

# MS1: precursor_indices == 0
ms1_df = timstof_data[:, :, 0, :][['rt_values_min', 'mobility_values', 'mz_values', 'intensity_values']]
print(f"MS1 数据点数: {len(ms1_df)}")

# MS2: precursor_indices > 0  
ms2_mask = df['precursor_indices'] > 0
ms2_df = df[ms2_mask].copy()
print(f"MS2 数据点数: {len(ms2_df)}")

# 定义压缩函数
def process_mixed_grid_data(temp_df: pd.DataFrame, bin_sizes: dict) -> pd.DataFrame:
    print("开始处理混合坐标数据（m/z, im离散; rt连续）...")
    if temp_df.empty:
        print("输入的DataFrame为空，返回一个空的DataFrame。")
        return pd.DataFrame(columns=['mz_values', 'rt_values_min', 'mobility_values', 'intensity_values'])
    
    # 创建坐标的整数版本用于分箱
    temp_df['mz_coord'] = np.ceil(temp_df['mz_values'] * 1000).astype(int)
    temp_df['im_coord'] = np.ceil(temp_df['mobility_values'] * 1000).astype(int)
    
    # --- 1. 离散维度分箱 (m/z, im) ---
    print("步骤1: 为m/z和im计算分箱ID...")
    mz_bin_size = bin_sizes['mz_values']
    im_bin_size = bin_sizes['mobility_values']
    temp_df['mz_bin_id'] = temp_df['mz_coord'] // mz_bin_size
    temp_df['im_bin_id'] = temp_df['im_coord'] // im_bin_size
    
    # --- 2. 连续维度分组 (rt) ---
    print("步骤2: 在每个(m/z, im)箱内按rt排序并分组...")
    rt_group_size = bin_sizes['rt_values_min']
    
    # 首先按分箱ID和rt排序，确保后续cumcount的顺序正确
    temp_df = temp_df.sort_values(by=['mz_bin_id', 'im_bin_id', 'rt_values_min'])
    
    # 在每个(m/z, im)箱内，计算rt的排名，然后除以rt组大小得到rt分组ID
    temp_df['rt_group_id'] = temp_df.groupby(['mz_bin_id', 'im_bin_id']).cumcount() // rt_group_size
    
    # --- 3. 最终聚合 ---
    print("步骤3: 按所有ID进行最终分组和聚合...")
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
    
    # 将m/z和im转换为整数
    result_df['mz_values'] = np.round(result_df['mz_values']).astype(int)
    result_df['mobility_values'] = np.round(result_df['mobility_values']).astype(int)
    
    print(f"处理完成！原始数据 {len(temp_df)} 行，压缩后 {len(result_df)} 行。")
    print(f"压缩率: {(1 - len(result_df)/len(temp_df)) * 100:.2f}%")
    
    return result_df.reset_index(drop=True)

# 定义压缩参数
binning_parameters = {'mz_values': 5, 'mobility_values': 3, 'rt_values_min': 3}

# 分别压缩 MS1 和 MS2 数据
print("\n=== 处理 MS1 数据 ===")
ms1_compressed = process_mixed_grid_data(ms1_df, binning_parameters)

print("\n=== 处理 MS2 数据 ===")
ms2_compressed = process_mixed_grid_data(ms2_df, binning_parameters)

# 输出压缩结果统计
print("\n=== 压缩结果统计 ===")
print(f"MS1 原始数据点: {len(ms1_df):,}")
print(f"MS1 压缩后数据点: {len(ms1_compressed):,}")
print(f"MS1 压缩率: {(1 - len(ms1_compressed)/len(ms1_df)) * 100:.2f}%")

print(f"\nMS2 原始数据点: {len(ms2_df):,}")
print(f"MS2 压缩后数据点: {len(ms2_compressed):,}")
print(f"MS2 压缩率: {(1 - len(ms2_compressed)/len(ms2_df)) * 100:.2f}%")

# 保存压缩后的数据
print("\n保存压缩后的数据...")
ms1_compressed.to_csv('ms1_compressed.csv', index=False)
ms2_compressed.to_csv('ms2_compressed.csv', index=False)

# 可选：查看数据分布
print("\n=== MS1 数据分布 ===")
print(f"m/z 范围: {ms1_compressed['mz_values'].min()} - {ms1_compressed['mz_values'].max()}")
print(f"RT 范围: {ms1_compressed['rt_values_min'].min():.2f} - {ms1_compressed['rt_values_min'].max():.2f} min")
print(f"Mobility 范围: {ms1_compressed['mobility_values'].min()} - {ms1_compressed['mobility_values'].max()}")
print(f"Intensity 范围: {ms1_compressed['intensity_values'].min():.2e} - {ms1_compressed['intensity_values'].max():.2e}")

print("\n=== MS2 数据分布 ===")
print(f"m/z 范围: {ms2_compressed['mz_values'].min()} - {ms2_compressed['mz_values'].max()}")
print(f"RT 范围: {ms2_compressed['rt_values_min'].min():.2f} - {ms2_compressed['rt_values_min'].max():.2f} min")
print(f"Mobility 范围: {ms2_compressed['mobility_values'].min()} - {ms2_compressed['mobility_values'].max()}")
print(f"Intensity 范围: {ms2_compressed['intensity_values'].min():.2e} - {ms2_compressed['intensity_values'].max():.2e}")
