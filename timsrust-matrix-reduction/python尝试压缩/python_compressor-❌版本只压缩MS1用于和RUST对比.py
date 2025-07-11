import timstof_PASEF_20250506
from copy import deepcopy
import pandas as pd
import numpy as np
import time

# 修改数据读取部分，只获取MS1数据
bruker_d_folder_name = '/Users/augustsirius/Desktop/DIABERT_test_code_lib/DIA_sample.d'
print(f"开始读取TimsTOF数据文件: {bruker_d_folder_name}")
start_time = time.time()

timstof_data = timstof_PASEF_20250506.TimsTOF(bruker_d_folder_name)

# 获取所有数据，但只筛选MS1（precursor_indices == 0 表示MS1）
print("开始筛选MS1数据...")
# 使用第三个维度为0来获取MS1数据（0表示没有前体，即MS1）
df = timstof_data[:, :, 0, :][['rt_values_min', 'mobility_values', 'mz_values', 'intensity_values']]

# 转换为DataFrame以便处理
if isinstance(df, pd.DataFrame):
    ms1_df = df.copy()
else:
    # 如果返回的不是DataFrame，手动创建
    ms1_df = pd.DataFrame(df)

print(f"\nMS1数据读取完成:")
print(f"  数据点数: {len(ms1_df)}")
print(f"  读取耗时: {time.time() - start_time:.2f}秒")

# 打印数据统计信息
print("\nMS1数据统计:")
print(f"  m/z范围: {ms1_df['mz_values'].min():.4f} - {ms1_df['mz_values'].max():.4f}")
print(f"  RT范围: {ms1_df['rt_values_min'].min():.2f} - {ms1_df['rt_values_min'].max():.2f} 分钟")
print(f"  IM范围: {ms1_df['mobility_values'].min():.4f} - {ms1_df['mobility_values'].max():.4f}")
print(f"  强度范围: {ms1_df['intensity_values'].min()} - {ms1_df['intensity_values'].max()}")

# 定义压缩函数
def process_data_with_groupby(temp_df: pd.DataFrame, bin_sizes: dict) -> pd.DataFrame:
    print("\n开始使用 pandas.groupby 方法处理数据...")
    compress_start = time.time()
    
    if temp_df.empty:
        print("输入的DataFrame为空，返回一个空的DataFrame。")
        return pd.DataFrame(columns=['mz_values', 'rt_values_min', 'mobility_values', 'intensity_values'])
    
    # 创建整数坐标
    temp_df = temp_df.copy()  # 避免修改原始数据
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
    grouped = temp_df.groupby(['mz_bin_id', 'rt_bin_id', 'im_bin_id'])
    summed_intensity = grouped['intensity_values'].sum()
    
    # 过滤掉加和后强度为0或接近0的分箱
    summed_intensity = summed_intensity[summed_intensity > 0]  # 改为>0，与Rust保持一致
    
    if summed_intensity.empty:
        print("处理后没有发现任何非零强度的点。")
        return pd.DataFrame(columns=['mz_values', 'rt_values_min', 'mobility_values', 'intensity_values'])
    
    # 将结果转换为DataFrame，并将多重索引重置为列
    result_df = summed_intensity.reset_index()
    print(f"分箱与聚合完成。发现 {len(result_df)} 个非空分箱。")
    
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
        'rt_values_min': result_df['new_rt_coord'].astype(float),
        'mobility_values': result_df['new_im_coord'] / 1000.0,
        'intensity_values': result_df['intensity_values']
    })
    
    # 按照要求的顺序排序
    final_df = final_df.sort_values(by=['mz_values', 'rt_values_min', 'mobility_values']).reset_index(drop=True)
    
    compress_time = time.time() - compress_start
    compression_rate = (1.0 - len(final_df) / len(temp_df)) * 100.0
    
    print(f"压缩完成！原始数据 {len(temp_df)} 行，压缩后 {len(final_df)} 行。")
    print(f"压缩率: {compression_rate:.2f}%")
    print(f"压缩耗时: {compress_time:.2f}秒")
    
    return final_df

# 定义分箱参数
binning_parameters = {'mz_values': 5, 'rt_values_min': 3, 'mobility_values': 3}

print("\n分箱参数:")
print(f"  m/z分箱大小: {binning_parameters['mz_values']}")
print(f"  RT分箱大小: {binning_parameters['rt_values_min']}")
print(f"  Mobility分箱大小: {binning_parameters['mobility_values']}")

# 调用函数进行处理
result_df = process_data_with_groupby(ms1_df, binning_parameters)

# 打印压缩后的统计信息
print("\n压缩数据统计:")
print(f"  数据点数: {len(result_df)}")
print(f"  m/z范围: {result_df['mz_values'].min():.4f} - {result_df['mz_values'].max():.4f}")
print(f"  RT范围: {result_df['rt_values_min'].min():.2f} - {result_df['rt_values_min'].max():.2f} 分钟")
print(f"  IM范围: {result_df['mobility_values'].min():.4f} - {result_df['mobility_values'].max():.4f}")
print(f"  强度范围: {result_df['intensity_values'].min()} - {result_df['intensity_values'].max()}")
print(f"  总强度: {result_df['intensity_values'].sum()}")

# 打印前10个压缩后的数据点作为示例
print("\n前10个压缩后的数据点:")
for i, row in result_df.head(10).iterrows():
    print(f"  {i+1}: m/z={row['mz_values']:.4f}, RT={row['rt_values_min']:.2f}, "
          f"IM={row['mobility_values']:.4f}, Intensity={row['intensity_values']}")

# 总耗时
total_time = time.time() - start_time
print(f"\n总耗时: {total_time:.2f}秒")

# 可选：保存结果以便与Rust版本对比
# result_df.to_csv('python_compressed_ms1.csv', index=False)
# print("\n结果已保存到 python_compressed_ms1.csv")