import os
import torch
import utils
import timstof_PASEF_20250506
from copy import deepcopy
# import model_handler
# from score_model import DIArtModel
import torch.nn as nn
# from score_model import FeatureEngineer
import pandas as pd
import numpy as np
import time
import bisect

# ----------------------------------------- 第一部分 -------------------------------------------
# ----------------20250711/timstof/稳定版/main1-实现python第一部分.rs ----------------------------

class FastChunkFinderViaDict:
    def __init__(self, chunks_dict):
        if not chunks_dict:
            raise ValueError("输入的字典不能为空。")
        self.chunks_dict = chunks_dict
        self.sorted_keys = sorted(self.chunks_dict.keys())
        # 提取所有下限值，用于 bisect 模块
        self.low_mz_values = [key[0] for key in self.sorted_keys]
        print("查找器已初始化，范围已排序。")
    def find(self, value):
        index = bisect.bisect_right(self.low_mz_values, value) - 1
        if index < 0:
            print(f"未找到包含值 {value} 的范围。")
            return None
        candidate_key = self.sorted_keys[index]
        low_mz, high_mz = candidate_key
        if low_mz <= value <= high_mz:
            print(f"值 {value} 在范围 [{low_mz}, {high_mz}] 内。")
            return self.chunks_dict[candidate_key]
        else:
            print(f"未找到包含值 {value} 的范围。")
            return None

bruker_d_folder_name = '/Users/augustsirius/Desktop/DIABERT_test_code_lib/DIA_sample.d'

timstof_data = timstof_PASEF_20250506.TimsTOF(bruker_d_folder_name)
df = deepcopy(timstof_data[:,:,:,:,:])
df_index = df.set_index('mz_values', drop=False)
df_index.sort_index(inplace=True)

df1_index = df_index[df_index['precursor_indices'] == 0] # MS1 BIG MATRIX
df2_index = df_index[df_index['precursor_indices'] != 0] # MS2 BIG MATRIX

# 这是alphatims解析出的isolation window 的下限和上限；对应到timsrust中是应该一个 precursor_mz +- 1/2 isolation_width
grouped_df2 = df2_index.groupby(['quad_low_mz_values', 'quad_high_mz_values']) 
dataframe_chunks = {name: group for name, group in grouped_df2}
finder = FastChunkFinderViaDict(dataframe_chunks)

# ----------------------------------------- 第二部分 -------------------------------------------

library = pd.read_csv(r"/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/lib/TPHPlib_frag1025_swissprot_final_all_from_Yueliang.tsv", sep="\t")

report_diann = pd.read_csv(r'/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/report/report.parquet', sep="\t")

report_diann['transition_group_id'] = report_diann['Precursor.Id']

# col mapping
lib_col_dict = utils.get_lib_col_dict()
for col in set(library.columns) & set(lib_col_dict.keys()):
    library.loc[:, lib_col_dict[col]] = library.loc[:, col]

replacement_dict = {'b': 1, 'y': 2, 'p': 3}
# 生成decoy lib
library['transition_group_id'] = library['FullUniModPeptideName'] + library['PrecursorCharge'].astype(str)
library['FragmentType'] = library['FragmentType'].replace(replacement_dict)
library['decoy'] = 0

# 注意这里从diann中获取，每个precursor的rt数据，按照字典形式传入
diann_result = pd.merge(library[['transition_group_id', 'PrecursorMz', 'ProductMz']], report_diann[['transition_group_id', 'RT', 'IM','iIM']], on='transition_group_id', how='left').dropna(subset=['RT'])
diann_precursor_id_all = diann_result.drop_duplicates(subset=['transition_group_id'])[['transition_group_id', 'RT', 'IM']].reset_index(drop=True)
assay_rt_kept_dict = dict(zip(diann_precursor_id_all['transition_group_id'], diann_precursor_id_all['RT']))
assay_im_kept_dict = dict(zip(diann_precursor_id_all['transition_group_id'], diann_precursor_id_all['IM']))

# precursor_id_all = library['transition_group_id'].unique()
precursor_id_all = list(assay_rt_kept_dict.keys())

device = 'cpu'
frag_repeat_num = 5

# ----------------------------------------- 第三部分 -------------------------------------------

def get_rt_list(lst, target):
    lst.sort()
    if not lst:
        return [0] * 48
    if len(lst) <= 48:
        return lst + [0] * (48 - len(lst))
    closest_idx = min(range(len(lst)), key=lambda i: abs(lst[i] - target))
    start = max(0, closest_idx - 24)
    end = start + 48
    if end > len(lst):
        start = len(lst) - 48
        end = len(lst)
    return lst[start:end]

start_time = time.perf_counter()
precursor_id_list = ["VAFSAVR2"]
each_lib_data = library[library['transition_group_id'].isin(precursor_id_list)]
precursors_list, ms1_data_list, ms2_data_list, precursor_info_list = utils.build_lib_matrix(
            each_lib_data,
            utils.lib_cols,
            None,
            None,
            5,
            1801,
            20,
            None)
ms1_data_tensor, ms2_data_tensor = utils.build_precursors_matrix_step1(ms1_data_list, ms2_data_list, device)
ms2_data_tensor = utils.build_precursors_matrix_step2(ms2_data_tensor)
ms1_range_list, ms2_range_list = utils.build_range_matrix_step3(ms1_data_tensor, ms2_data_tensor, frag_repeat_num, device=device)
ms1_data_tensor, ms2_data_tensor, ms1_extract_width_range_list, ms2_extract_width_range_list = utils.build_precursors_matrix_step3(ms1_data_tensor, ms2_data_tensor, frag_repeat_num, device=device)
precursor_info_np_org = np.array(precursor_info_list)
precursor_info_choose = precursor_info_np_org[:, 0: 5]
delta_rt_kept = np.array([0] * len(precursor_info_choose)).reshape(-1, 1)
assay_rt_kept = np.array([assay_rt_kept_dict[ee[0]] for ee in precursors_list]).reshape(-1, 1)
assay_im_kept = np.array([assay_im_kept_dict[ee[0]] for ee in precursors_list]).reshape(-1, 1)

# ----------------------------------------- 第四部分 -------------------------------------------

# precursor_feat
precursor_feat = np.column_stack([precursor_info_choose, assay_im_kept, assay_rt_kept, delta_rt_kept])
# frag info
frag_info = utils.build_frag_info(ms1_data_tensor, ms2_data_tensor, frag_repeat_num,
                                device)
i = 0
IM = precursor_feat[i][5]
RT = precursor_feat[i][6]
precursor_mz = precursor_feat[i][1]
df2_index_final = finder.find(precursor_mz)
ms1_range_min = (ms1_range_list[i].min().item()-1)/1000
ms1_range_max = (ms1_range_list[i].max().item()+1)/1000

precursor_result = df1_index.loc[ms1_range_min:ms1_range_max][['rt_values_min', 'mobility_values', 'mz_values', 'intensity_values']]
precursor_result['mz_values'] = np.ceil(precursor_result['mz_values'] * 1000)
# 根据IM值筛选precursor_result
precursor_result = precursor_result[(precursor_result['mobility_values'] <= IM + 0.05) & (precursor_result['mobility_values'] >= IM - 0.05)]
frag_result = []
for j in range(0, 66):
    ms2_range_min = (ms2_range_list[i][j].min().item()-1)/1000
    ms2_range_max = (ms2_range_list[i][j].max().item()+1)/1000
    frag_result.append(df2_index_final.loc[ms2_range_min:ms2_range_max][['rt_values_min', 'mobility_values',  'mz_values', 'intensity_values']])
frag_result = pd.concat(frag_result, ignore_index=True)
frag_result['mz_values'] = np.ceil(frag_result['mz_values'] * 1000)
# 根据IM值筛选frag_result
frag_result = frag_result[(frag_result['mobility_values'] <= IM + 0.05) & (frag_result['mobility_values'] >= IM - 0.05)]

# ----------------------------------------- 第五部分 -------------------------------------------

#### 获取ms1和ms2的frag_moz的0/1矩阵
search_ms1_tensor = torch.tensor(list(precursor_result['mz_values']))
search_ms2_tensor = torch.tensor(list(frag_result['mz_values']))
mask_ms1 = torch.isin(ms1_extract_width_range_list[i], search_ms1_tensor)
mask_ms2 = torch.isin(ms2_extract_width_range_list[i], search_ms2_tensor)
ms1_frag_moz_matrix = torch.where(mask_ms1, 1., 0.)
ms2_frag_moz_matrix = torch.where(mask_ms2, 1., 0.)

all_rt = get_rt_list(list(set(list(precursor_result['rt_values_min'].unique())+list(frag_result['rt_values_min'].unique()))), RT)

rsm_matrix = []
#### 获取ms1的frag_rt的intensity矩阵
precursor_result.index.name = None
precursor_pivot = precursor_result.pivot_table(
    index='mz_values',
    columns='rt_values_min',
    values='intensity_values',
    aggfunc='sum',
    fill_value=0)
precursor_pivot = precursor_pivot.reindex(all_rt, axis=1, fill_value=0)
ms1_frag_rt_matrix = []
for a in range(len(ms1_extract_width_range_list[i])):
    # 3. 高效提取: 直接从预处理好的透视表中，根据 mz 值列表批量提取子矩阵
    moz_list = ms1_extract_width_range_list[i][a].tolist()
    ms1_moz_rt_df = precursor_pivot.reindex(moz_list, fill_value=0)
    # 4. 转换为 Tensor: 将提取的 DataFrame 高效转换为 PyTorch Tensor
    ms1_moz_rt_matrix = torch.tensor(ms1_moz_rt_df.values, dtype=torch.float32)
    # 5. 矩阵相乘 (与原逻辑相同)
    ms1_frag_rt = ms1_frag_moz_matrix[a] @ ms1_moz_rt_matrix
    ms1_frag_rt_matrix.append(ms1_frag_rt)
# 6. 堆叠结果 (与原逻辑相同)
ms1_frag_rt_matrix = torch.stack(ms1_frag_rt_matrix, dim=0)

#### 获取ms2的frag_rt的intensity矩阵
frag_result.index.name = None
frag_pivot = frag_result.pivot_table(
    index='mz_values',
    columns='rt_values_min',
    values='intensity_values',
    aggfunc='sum',
    fill_value=0)
# 2. 对齐列: 确保透视表的列与 all_rt 的顺序和内容完全一致。
#    这一步至关重要，它保证了我们后续提取的矩阵的列维度是正确的。
frag_pivot = frag_pivot.reindex(all_rt, axis=1, fill_value=0)
ms2_frag_rt_matrix = []
for b in range(len(ms2_extract_width_range_list[i])):
    # 3. 高效提取: 直接从预处理好的透视表中，根据 mz 值列表批量提取出所需的子矩阵。
    #    用一次 reindex 调用替代了整个内层 for 循环。
    moz_list = ms2_extract_width_range_list[i][b].tolist()
    ms2_moz_rt_df = frag_pivot.reindex(moz_list, fill_value=0)
    # 4. 转换为 Tensor: 将提取出的 DataFrame 高效地转换为 PyTorch Tensor。
    ms2_moz_rt_matrix = torch.tensor(ms2_moz_rt_df.values, dtype=torch.float32)
    # 5. 矩阵相乘 (与原逻辑相同)
    ms2_frag_rt = ms2_frag_moz_matrix[b] @ ms2_moz_rt_matrix
    ms2_frag_rt_matrix.append(ms2_frag_rt)
# 6. 堆叠结果 (与原逻辑相同)
ms2_frag_rt_matrix = torch.stack(ms2_frag_rt_matrix, dim=0)

# ----------------------------------------- 第六部分 -------------------------------------------

ms1_frag_rt_matrix_shape = ms1_frag_rt_matrix.shape
ms1_frag_rt_matrix1 = ms1_frag_rt_matrix.reshape(frag_repeat_num, ms1_frag_rt_matrix_shape[0] // frag_repeat_num, ms1_frag_rt_matrix_shape[1])
ms2_frag_rt_matrix_shape = ms2_frag_rt_matrix.shape
ms2_frag_rt_matrix1 = ms2_frag_rt_matrix.reshape(frag_repeat_num, ms2_frag_rt_matrix_shape[0] // frag_repeat_num, ms2_frag_rt_matrix_shape[1])

full_frag_rt_matrix = torch.cat([ms1_frag_rt_matrix1, ms2_frag_rt_matrix1], dim=1)
rsm_matrix.append(full_frag_rt_matrix)
rsm_matrix = torch.stack(rsm_matrix, dim=0)
aggregated_x_sum = torch.sum(rsm_matrix, dim=1)

header = all_rt+['ProductMz', 'LibraryIntensity', 'frag_type', 'FragmentType']
merged_tensor = torch.cat((aggregated_x_sum[0], frag_info[i]), dim=1)
numpy_array = merged_tensor.cpu().numpy()
data = pd.DataFrame(numpy_array)
data.columns = header

end_time = time.perf_counter()
elapsed_time_seconds = end_time - start_time
print(f"提取一个precursor所花费时间: {elapsed_time_seconds} 秒")