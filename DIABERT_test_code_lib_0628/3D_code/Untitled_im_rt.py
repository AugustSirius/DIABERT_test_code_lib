import os
import torch
import utils
import timstof_PASEF_20250506
# from copy import deepcopy
# import model_handler
# from score_model import DIArtModel
import torch.nn as nn
# from score_model import FeatureEngineer
import pandas as pd
import numpy as np

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

  
bruker_d_folder_name = 'D:\\00.Westlake\\00.Phd_candidate\\01.LiuZW\\test\\CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d'
timstof_data = timstof_PASEF_20250506.TimsTOF(bruker_d_folder_name)
# df = deepcopy(timstof_data[:,:,:,:,:])

# **************** 下面两个按照实际传入************************
library = pd.read_csv(r"D:\\00.Westlake\\00.Phd_candidate\\01.LiuZW\\66.DIA-NN_library\\TPHPlib_frag1025_swissprot_final_all_from_Yueliang.tsv", sep="\t")

report_diann = pd.read_parquet('Z:\\members\\wangshuaiyao\\CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382\\report.parquet')
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

precursor_id_all = list(assay_rt_kept_dict.keys())
precursor_id_list_arr = utils.list_split(precursor_id_all, 500)

device = 'cpu'
frag_repeat_num = 5


save_precursor_list = []
save_precursors_info = []
save_rsm = []
save_frag_info = []
save_score = []
for each_num, precursor_id_list in enumerate(precursor_id_list_arr):
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
    ms1_data_tensor, ms2_data_tensor = utils.build_precursors_matrix_step1(ms1_data_list,
                                                                                        ms2_data_list,
                                                                                        device)
    ms2_data_tensor = utils.build_precursors_matrix_step2(ms2_data_tensor)
    ms1_range_list, ms2_range_list = utils.build_range_matrix_step3(ms1_data_tensor, ms2_data_tensor, frag_repeat_num, device=device)
    ms1_data_tensor, ms2_data_tensor, ms1_extract_width_range_list, ms2_extract_width_range_list = utils.build_precursors_matrix_step3(ms1_data_tensor, ms2_data_tensor, frag_repeat_num, device=device)

    precursor_info_np_org = np.array(precursor_info_list)
    precursor_info_choose = precursor_info_np_org[:, 0: 5]
    delta_rt_kept = np.array([0] * len(precursor_info_choose)).reshape(-1, 1)
    
    assay_rt_kept = np.array([assay_rt_kept_dict[ee[0]] for ee in precursors_list]).reshape(-1, 1)
    assay_im_kept = np.array([assay_im_kept_dict[ee[0]] for ee in precursors_list]).reshape(-1, 1)


    # precursor_feat
    precursor_feat = np.column_stack([precursor_info_choose, assay_im_kept, assay_rt_kept, delta_rt_kept])
    # frag info
    frag_info = utils.build_frag_info(ms1_data_tensor, ms2_data_tensor, frag_repeat_num,
                                    device)
    
    rsm_matrix = []
    for i in range(len(precursor_feat)):
        IM = precursor_feat[i][5]
        RT = precursor_feat[i][6]
       
        #### 获取ms1和对应ms2的spectrum信息
        ms1_range = slice((ms1_range_list[i].min().item()-1)/1000, (ms1_range_list[i].max().item()+1)/1000)
        precursor_result = timstof_data[:, :, 0, ms1_range][['rt_values_min', 'mobility_values', 'mz_values', 'intensity_values']]
        precursor_result['mz_values'] = np.ceil(precursor_result['mz_values'] * 1000)
        # 根据IM和RT筛选precursor_result
        precursor_result = precursor_result[(precursor_result['mobility_values'] <= IM + 0.05) & (precursor_result['mobility_values'] >= IM - 0.05)]
        frag_result = []
        for j in range(0, 66):
            ms2_range = slice((ms2_range_list[i][j].min().item()-1)/1000, (ms2_range_list[i][j].max().item()+1)/1000)
            frag_result.append(timstof_data[:, :, ms1_range, ms2_range][['rt_values_min', 'mobility_values',  'mz_values', 'intensity_values']])
        frag_result = pd.concat(frag_result, ignore_index=True)
        frag_result['mz_values'] = np.ceil(frag_result['mz_values'] * 1000)
        # 根据IM和RT筛选frag_result
        frag_result = frag_result[(frag_result['mobility_values'] <= IM + 0.05) & (frag_result['mobility_values'] >= IM - 0.05)]

        #### 获取ms1和ms2的frag_moz的0/1矩阵
        search_ms1_tensor = torch.tensor(list(precursor_result['mz_values']))
        search_ms2_tensor = torch.tensor(list(frag_result['mz_values']))
        mask_ms1 = torch.isin(ms1_extract_width_range_list[i], search_ms1_tensor)
        mask_ms2 = torch.isin(ms2_extract_width_range_list[i], search_ms2_tensor)
        ms1_frag_moz_matrix = torch.where(mask_ms1, 1., 0.)
        ms2_frag_moz_matrix = torch.where(mask_ms2, 1., 0.)

        #### 获取ms1和ms2的16个rt列表（不够的补0）
        all_rt = get_rt_list(list(set(list(precursor_result['rt_values_min'].unique())+list(frag_result['rt_values_min'].unique()))), RT)

        #### 获取ms1的frag_rt的intensity矩阵
        ms1_frag_rt_matrix = []
        for a in range(len(ms1_extract_width_range_list[i])):
            #### 获取ms1的moz_rt的intensity矩阵
            ms1_moz_rt_matrix = []
            for rt in all_rt:
                moz_to_intensity = dict(zip(precursor_result[precursor_result['rt_values_min'] == rt]['mz_values'], 
                                            precursor_result[precursor_result['rt_values_min'] == rt]['intensity_values']))
                mapped_intensities = torch.tensor([moz_to_intensity.get(moz.item(), 0) for moz in ms1_extract_width_range_list[i][a]], 
                                                dtype=torch.float32)
                ms1_moz_rt_matrix.append(mapped_intensities)
            ms1_moz_rt_matrix = torch.stack(ms1_moz_rt_matrix, dim=1)
            ### 矩阵相乘
            ms1_frag_rt = ms1_frag_moz_matrix[a] @ ms1_moz_rt_matrix
            ms1_frag_rt_matrix.append(ms1_frag_rt)
        ms1_frag_rt_matrix = torch.stack(ms1_frag_rt_matrix, dim=0)

        #### 获取ms2的frag_rt的intensity矩阵
        ms2_frag_rt_matrix = []
        for b in range(len(ms2_extract_width_range_list[i])):
            #### 获取ms2的moz_rt的intensity矩阵
            ms2_moz_rt_matrix = []
            for rt in all_rt:
                moz_to_intensity = dict(zip(frag_result[frag_result['rt_values_min'] == rt]['mz_values'], 
                                            frag_result[frag_result['rt_values_min'] == rt]['intensity_values']))
                mapped_intensities = torch.tensor([moz_to_intensity.get(moz.item(), 0) for moz in ms2_extract_width_range_list[i][b]], 
                                                dtype=torch.float32)
                ms2_moz_rt_matrix.append(mapped_intensities)
            ms2_moz_rt_matrix = torch.stack(ms2_moz_rt_matrix, dim=1)
            ### 矩阵相乘
            ms2_frag_rt = ms2_frag_moz_matrix[b] @ ms2_moz_rt_matrix
            ms2_frag_rt_matrix.append(ms2_frag_rt)
        ms2_frag_rt_matrix = torch.stack(ms2_frag_rt_matrix, dim=0)

        full_frag_rt_matrix = torch.cat([ms1_frag_rt_matrix, ms2_frag_rt_matrix], dim=0)
        full_frag_rt_matrix_shape = full_frag_rt_matrix.shape
        full_frag_rt_matrix = full_frag_rt_matrix.reshape(frag_repeat_num, full_frag_rt_matrix_shape[0] // frag_repeat_num, full_frag_rt_matrix_shape[1])
        rsm_matrix.append(full_frag_rt_matrix)
    rsm_matrix = torch.stack(rsm_matrix, dim=0)

    save_precursor_list.append(precursor_id_list)
    save_precursors_info.append(precursor_feat)
    save_rsm.append(rsm_matrix)
    save_frag_info.append(frag_info)

