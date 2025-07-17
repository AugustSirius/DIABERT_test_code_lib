import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import groupby
import pickle
import torch
import os
import timstof_PASEF_20250506
import matplotlib.pyplot as plt
import time  # 添加时间模块

def get_lib_col_dict():
    lib_col_dict = defaultdict(str)
    for key in ['transition_group_id', 'PrecursorID']:lib_col_dict[key] = 'transition_group_id'
    for key in ['PeptideSequence', 'Sequence', 'StrippedPeptide']:lib_col_dict[key] = 'PeptideSequence'
    for key in ['FullUniModPeptideName', 'ModifiedPeptide', 'LabeledSequence', 'modification_sequence','ModifiedPeptideSequence']:lib_col_dict[key] = 'FullUniModPeptideName'
    for key in ['PrecursorCharge', 'Charge', 'prec_z']:lib_col_dict[key] = 'PrecursorCharge'
    for key in ['PrecursorMz', 'Q1']:lib_col_dict[key] = 'PrecursorMz'
    for key in ['Tr_recalibrated', 'iRT', 'RetentionTime', 'NormalizedRetentionTime', 'RT_detected']:lib_col_dict[key] = 'Tr_recalibrated'
    for key in ['ProductMz', 'FragmentMz', 'Q3']:lib_col_dict[key] = 'ProductMz'
    for key in ['FragmentType', 'FragmentIonType', 'ProductType', 'ProductIonType', 'frg_type']:lib_col_dict[key] = 'FragmentType'
    for key in ['FragmentCharge', 'FragmentIonCharge', 'ProductCharge', 'ProductIonCharge', 'frg_z']:lib_col_dict[key] = 'FragmentCharge'
    for key in ['FragmentNumber', 'frg_nr', 'FragmentSeriesNumber']:lib_col_dict[key] = 'FragmentNumber'
    for key in ['LibraryIntensity', 'RelativeIntensity', 'RelativeFragmentIntensity', 'RelativeFragmentIonIntensity','relative_intensity']:lib_col_dict[key] = 'LibraryIntensity'
    for key in ['FragmentLossType', 'FragmentIonLossType', 'ProductLossType', 'ProductIonLossType']:lib_col_dict[key] = 'FragmentLossType'
    for key in ['ProteinID', 'ProteinId', 'UniprotID', 'uniprot_id', 'UniProtIds']:lib_col_dict[key] = 'ProteinID'
    for key in ['ProteinName', 'Protein Name', 'Protein_name', 'protein_name']:lib_col_dict[key] = 'ProteinName'
    for key in ['Gene', 'Genes', 'GeneName']:lib_col_dict[key] = 'Gene'
    for key in ['Decoy', 'decoy']:lib_col_dict[key] = 'decoy'
    for key in ['ExcludeFromAssay', 'ExcludeFromQuantification']:lib_col_dict[key] = 'ExcludeFromAssay'
    return lib_col_dict

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

# ============== START: FIRST CODE VERSION ==============

# 1. Load library
library = pd.read_csv("/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/lib/TPHPlib_frag1025_swissprot_final_all_from_Yueliang.tsv", sep="\t")

# 2. Col mapping
lib_col_dict = get_lib_col_dict()
for col in set(library.columns) & set(lib_col_dict.keys()):
    library.loc[:, lib_col_dict[col]] = library.loc[:, col]

# 3. Library preprocessing
library['transition_group_id'] = library['FullUniModPeptideName'] + library['PrecursorCharge'].astype(str)
replacement_dict = {'b': 1, 'y': 2, 'p': 3}
library['FragmentType'] = library['FragmentType'].replace(replacement_dict)
library['decoy'] = 0

# 4. SINGLE PRECURSOR - same as first code
precursor_id_list = ["LLIYGASTR2"]

# 5. Load report
report_diann = pd.read_parquet('/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/report/report.parquet')
report_diann['transition_group_id'] = report_diann['Precursor.Id']

# 6. Build assay dictionaries
diann_result = pd.merge(library[['transition_group_id', 'PrecursorMz', 'ProductMz']], report_diann[['transition_group_id', 'RT', 'IM','iIM']], on='transition_group_id', how='left').dropna(subset=['RT'])
diann_precursor_id_all = diann_result.drop_duplicates(subset=['transition_group_id'])[['transition_group_id', 'RT', 'IM']].reset_index(drop=True)
assay_rt_kept_dict = dict(zip(diann_precursor_id_all['transition_group_id'], diann_precursor_id_all['RT']))
assay_im_kept_dict = dict(zip(diann_precursor_id_all['transition_group_id'], diann_precursor_id_all['IM']))

# 7. Filter library data
each_lib_data = library[library['transition_group_id'].isin(precursor_id_list)]

# ============== ALL THE BUILD FUNCTIONS FROM FIRST CODE ==============

lib_cols = {'PRECURSOR_MZ_COL': 'PrecursorMz','IRT_COL': 'Tr_recalibrated','PRECURSOR_ID_COL': 'transition_group_id','FULL_SEQUENCE_COL': 'FullUniModPeptideName','PURE_SEQUENCE_COL': 'PeptideSequence','PRECURSOR_CHARGE_COL': 'PrecursorCharge','FRAGMENT_MZ_COL': 'ProductMz','FRAGMENT_SERIES_COL': 'FragmentNumber','FRAGMENT_CHARGE_COL': 'FragmentCharge','FRAGMENT_TYPE_COL': 'FragmentType','LIB_INTENSITY_COL': 'LibraryIntensity','PROTEIN_NAME_COL': 'ProteinName','DECOY_OR_NOT_COL': 'decoy'}
MS1_ISOTOPE_COUNT = 6      
FRAGMENT_VARIANTS = 3       
MS1_TYPE_MARKER = 5         
MS1_FRAGMENT_TYPE = 1       
VARIANT_ORIGINAL = 2        
VARIANT_LIGHT = 3          
VARIANT_HEAVY = 4          

def get_precursor_indice(precursor_ids):
    precursor_indice = []
    for key, group in groupby(enumerate(precursor_ids), key=lambda x: x[1]):
        indices = [idx for idx, _ in group]
        precursor_indice.append(indices)
    return precursor_indice

def intercept_frags_sort(fragment_list, max_length):
    fragment_list.sort(reverse=True)
    return fragment_list[:max_length]

def build_ms1_data(fragment_list, isotope_range, max_mz):
    first_fragment = fragment_list[0]
    charge = first_fragment[1]  
    precursor_mz = first_fragment[5]  
    available_range = (max_mz - precursor_mz) * charge
    iso_shift_max = int(min(isotope_range, available_range)) + 1
    isotope_mz_list = [precursor_mz + iso_shift / charge 
                       for iso_shift in range(iso_shift_max)]
    isotope_mz_list = intercept_frags_sort(isotope_mz_list, MS1_ISOTOPE_COUNT)
    ms1_data = []
    for mz in isotope_mz_list:
        row = [mz,first_fragment[1],first_fragment[2],first_fragment[3],3,first_fragment[5],MS1_TYPE_MARKER,0,MS1_FRAGMENT_TYPE]
        ms1_data.append(row)
    while len(ms1_data) < MS1_ISOTOPE_COUNT:
        ms1_data.append([0, 0, 0, 0, 0, 0, 0, 0, 0])
    return np.array(ms1_data)

def build_ms2_data(fragment_list, max_fragment_num):
    total_count = max_fragment_num * FRAGMENT_VARIANTS
    fragment_num = fragment_list.shape[0]
    tripled_fragments = np.vstack([fragment_list] * FRAGMENT_VARIANTS)
    total_rows = fragment_num * FRAGMENT_VARIANTS
    type_column = np.zeros(total_rows)
    type_column[fragment_num:fragment_num * 2] = -1 
    type_column[fragment_num * 2:] = 1              
    window_id_column = np.zeros(total_rows)
    variant_type_column = np.zeros(total_rows)
    variant_type_column[:fragment_num] = VARIANT_ORIGINAL  
    variant_type_column[fragment_num:fragment_num * 2] = VARIANT_LIGHT  
    variant_type_column[fragment_num * 2:] = VARIANT_HEAVY  
    complete_data = np.hstack((
        tripled_fragments,
        type_column[:, np.newaxis],
        window_id_column[:, np.newaxis],
        variant_type_column[:, np.newaxis]
    ))
    if len(complete_data) >= total_count:
        return complete_data[:total_count]
    else:
        result = np.zeros((total_count, complete_data.shape[1]))
        result[:len(complete_data)] = complete_data
        return result

def build_precursor_info(fragment_list):
    first_fragment = fragment_list[0]
    return np.array([len(first_fragment[7]), first_fragment[5], first_fragment[1], first_fragment[6], len(fragment_list), 0])

def format_ms_data(fragment_list, isotope_range, max_mz, max_fragment):
    ms1_data = build_ms1_data(fragment_list, isotope_range, max_mz)
    ms2_data = build_ms2_data(fragment_list[:, :6], max_fragment)
    ms1_copy = np.copy(ms1_data)
    ms1_copy[:, 8] = 5
    combined_ms2 = np.concatenate([ms2_data, ms1_copy], axis=0)
    precursor_info = build_precursor_info(fragment_list)
    return ms1_data, combined_ms2, precursor_info

def build_lib_matrix(lib_data, lib_cols, iso_range=5, mz_max=1801, max_fragment=20):
    precursor_groups = get_precursor_indice(lib_data[lib_cols["PRECURSOR_ID_COL"]])
    all_precursors = []
    all_ms1_data = []
    all_ms2_data = []
    all_precursor_info = []
    first_indices = [group[0] for group in precursor_groups]
    cols = [lib_cols['PRECURSOR_ID_COL'], 'decoy']
    precursors_list = lib_data.iloc[first_indices][cols].values.tolist()
    all_indices = []
    for group in precursor_groups:
        all_indices.extend(group)
    group_data_col = ['ProductMz', 'PrecursorCharge', 'FragmentCharge', 'LibraryIntensity', 
                      'FragmentType', 'PrecursorMz','Tr_recalibrated', 'PeptideSequence', 
                      'decoy', 'transition_group_id']
    fragment_data = lib_data.iloc[all_indices][group_data_col].values
    for i, indices in enumerate(precursor_groups):
        group_fragments = fragment_data[indices]
        ms1, ms2, info = format_ms_data(
            group_fragments, iso_range, mz_max, max_fragment
        )
        all_precursors.append(precursors_list[i])
        all_ms1_data.append(ms1)
        all_ms2_data.append(ms2)
        all_precursor_info.append(info)
    return all_precursors, all_ms1_data, all_ms2_data, all_precursor_info

# ============== TENSOR PROCESSING FUNCTIONS ==============

device = 'cpu'
frag_repeat_num = 5

def build_precursors_matrix_step1(ms1_data_list, ms2_data_list, device='cpu'):
    ms1_data_tensor = torch.tensor(ms1_data_list, dtype=torch.float32, device=device)
    ms2_data_tensor = torch.tensor(ms2_data_list, dtype=torch.float32, device=device)
    return ms1_data_tensor, ms2_data_tensor

def build_precursors_matrix_step2(ms2_data_tensor):
    ms2_data_tensor[:, :, 0] = ms2_data_tensor[:, :, 0] + ms2_data_tensor[:, :, 6] / ms2_data_tensor[:, :, 2]
    ms2_data_tensor[torch.isinf(ms2_data_tensor)] = 0
    ms2_data_tensor[torch.isnan(ms2_data_tensor)] = 0
    return ms2_data_tensor

def extract_width_2(mz_to_extract, mz_unit, mz_tol, max_extract_len=20, frag_repeat_num=5, max_moz_num=50, device='cpu'):
    if mz_to_extract.eq(0).all():
        return torch.zeros(mz_to_extract.size() + (max_extract_len,))
    if mz_unit == "Da":
        mz_tol_full = (mz_to_extract / mz_to_extract) * mz_tol
    elif mz_unit == "ppm":
        mz_tol_full = mz_to_extract * mz_tol * 0.000001
    else:
        raise Exception("Invalid mz_unit format: %s. Only Da and ppm are supported." % mz_unit)
    mz_tol_full[torch.isnan(mz_tol_full)] = 0
    mz_tol_full_num = (max_moz_num / 1000)
    condition = mz_tol_full[:, :] > mz_tol_full_num
    mz_tol_full[condition] = mz_tol_full_num
    mz_tol_full = torch.ceil(mz_tol_full * 1000 / frag_repeat_num) * frag_repeat_num
    extract_width_range_list = torch.stack((mz_to_extract * 1000 - mz_tol_full, mz_to_extract * 1000 + mz_tol_full),
                                     dim=-1).floor()
    return extract_width_range_list

def build_range_matrix_step3(ms1_data_tensor, ms2_data_tensor, frag_repeat_num=5, mz_unit='ppm', mz_tol_ms1=20,
                                  mz_tol_ms2=50, device='cpu'):
    re_ms1_data_tensor = ms1_data_tensor.repeat(1, frag_repeat_num, 1)
    re_ms2_data_tensor = ms2_data_tensor.repeat(1, frag_repeat_num, 1)
    ms1_extract_width_range_list = extract_width_2(re_ms1_data_tensor[:, :, 0], mz_unit, mz_tol_ms1,
                                                            device=device)
    ms2_extract_width_range_list = extract_width_2(re_ms2_data_tensor[:, :, 0], mz_unit, mz_tol_ms2,
                                                            device=device)
    return ms1_extract_width_range_list, ms2_extract_width_range_list

def extract_width(mz_to_extract, mz_unit='ppm', mz_tol=50, max_extract_len=20, frag_repeat_num=5, max_moz_num=50, device='cpu'):
    if mz_to_extract.eq(0).all():
        return torch.zeros(mz_to_extract.size() + (max_extract_len,))
    if mz_unit == "Da":
        mz_tol_half = (mz_to_extract / mz_to_extract) * mz_tol / 2 
    elif mz_unit == "ppm":
        mz_tol_half = mz_to_extract * mz_tol * 0.000001 / 2
    else:
        raise Exception("Invalid mz_unit format: %s. Only Da and ppm are supported." % mz_unit)
    mz_tol_half[torch.isnan(mz_tol_half)] = 0
    mz_tol_half_num = (max_moz_num / 1000) / 2
    condition = mz_tol_half[:, :] > mz_tol_half_num
    mz_tol_half[condition] = mz_tol_half_num
    mz_tol_half = torch.ceil(mz_tol_half * 1000 / frag_repeat_num) * frag_repeat_num
    extract_width_list = torch.stack((mz_to_extract * 1000 - mz_tol_half, mz_to_extract * 1000 + mz_tol_half),
                                     dim=-1).floor()
    batch_num = int(mz_to_extract.shape[1] / frag_repeat_num)
    cha_tensor = (extract_width_list[:, 0:batch_num, 1] - extract_width_list[:, 0:batch_num, 0]) / frag_repeat_num
    extract_width_list[:, 0:batch_num, 0] = extract_width_list[:, 0:batch_num, 0]
    extract_width_list[:, 0:batch_num, 1] = extract_width_list[:, 0:batch_num, 0] + cha_tensor - 1
    extract_width_list[:, batch_num:batch_num * 2, 0] = extract_width_list[:, 0:batch_num, 0] + cha_tensor
    extract_width_list[:, batch_num:batch_num * 2, 1] = extract_width_list[:, 0:batch_num, 0] + 2 * cha_tensor - 1
    extract_width_list[:, batch_num * 2:batch_num * 3, 0] = extract_width_list[:, 0:batch_num, 0] + 2 * cha_tensor
    extract_width_list[:, batch_num * 2:batch_num * 3, 1] = extract_width_list[:, 0:batch_num, 0] + 3 * cha_tensor - 1
    extract_width_list[:, batch_num * 3:batch_num * 4, 0] = extract_width_list[:, 0:batch_num, 0] + 3 * cha_tensor
    extract_width_list[:, batch_num * 3:batch_num * 4, 1] = extract_width_list[:, 0:batch_num, 0] + 4 * cha_tensor - 1
    extract_width_list[:, batch_num * 4:batch_num * 5, 0] = extract_width_list[:, 0:batch_num, 0] + 4 * cha_tensor
    extract_width_list[:, batch_num * 4:batch_num * 5, 1] = extract_width_list[:, 0:batch_num, 0] + 5 * cha_tensor - 1
    new_tensor = torch.zeros(mz_to_extract.shape[0], mz_to_extract.shape[1], max_moz_num, dtype=torch.float32,
                             device=device)
    for i in range(new_tensor.shape[2]):
        new_tensor[:, :, i] = extract_width_list[:, :, 0] + i * 1
        condition = new_tensor[:, :, i] > extract_width_list[:, :, 1]                                   
        new_tensor[:, :, i][condition] = 0
    return new_tensor

def build_precursors_matrix_step3(ms1_data_tensor, ms2_data_tensor, frag_repeat_num=5, mz_unit='ppm', mz_tol_ms1=20,
                                  mz_tol_ms2=50, device='cpu'):
    re_ms1_data_tensor = ms1_data_tensor.repeat(1, frag_repeat_num, 1)
    re_ms2_data_tensor = ms2_data_tensor.repeat(1, frag_repeat_num, 1)
    ms1_extract_width_range_list = extract_width(re_ms1_data_tensor[:, :, 0], mz_unit, mz_tol_ms1,
                                                            device=device)
    ms2_extract_width_range_list = extract_width(re_ms2_data_tensor[:, :, 0], mz_unit, mz_tol_ms2,
                                                            device=device)
    return re_ms1_data_tensor, re_ms2_data_tensor, ms1_extract_width_range_list, ms2_extract_width_range_list

def build_ext_ms1_matrix(ms1_data_tensor, device):
    ext_matrix = ms1_data_tensor[:, :, [0, 3, 8, 4]].to(device)
    return ext_matrix

def build_ext_ms2_matrix(ms2_data_tensor, device):
    ext_matrix = ms2_data_tensor[:, :, [0, 3, 8, 4]].to(device)
    return ext_matrix

def build_frag_info(ms1_data_tensor, ms2_data_tensor, frag_repeat_num, device):
    ext_ms1_precursors_frag_rt_matrix = build_ext_ms1_matrix(ms1_data_tensor, device)
    ext_ms2_precursors_frag_rt_matrix = build_ext_ms2_matrix(ms2_data_tensor, device)
    ms1_ext_shape = ext_ms1_precursors_frag_rt_matrix.shape
    ms2_ext_shape = ext_ms2_precursors_frag_rt_matrix.shape
    ext_ms1_precursors_frag_rt_matrix = ext_ms1_precursors_frag_rt_matrix.reshape(ms1_ext_shape[0],
                                                                                  frag_repeat_num,
                                                                                  ms1_ext_shape[
                                                                                      1] // frag_repeat_num,
                                                                                  ms1_ext_shape[2]).cpu()
    ext_ms2_precursors_frag_rt_matrix = ext_ms2_precursors_frag_rt_matrix.reshape(ms2_ext_shape[0],
                                                                                  frag_repeat_num,
                                                                                  ms2_ext_shape[
                                                                                      1] // frag_repeat_num,
                                                                                  ms2_ext_shape[2]).cpu()
    frag_info = torch.cat([ext_ms1_precursors_frag_rt_matrix, ext_ms2_precursors_frag_rt_matrix], dim=2)
    frag_info = frag_info[:, 0, :, :]

    print(f"frag_info shape: {frag_info.shape}")
    return frag_info

# ============== EXECUTE FIRST CODE PROCESSING ==============

# 8. Build library matrix
precursors_list, ms1_data_list, ms2_data_list, precursor_info_list = build_lib_matrix(each_lib_data,lib_cols,iso_range=5,mz_max=1801,max_fragment=20)

# 9. Build tensor matrices
ms1_data_tensor, ms2_data_tensor = build_precursors_matrix_step1(ms1_data_list, ms2_data_list, device)
ms2_data_tensor = build_precursors_matrix_step2(ms2_data_tensor)
ms1_range_list, ms2_range_list = build_range_matrix_step3(ms1_data_tensor, ms2_data_tensor, frag_repeat_num, device=device)
ms1_data_tensor, ms2_data_tensor, ms1_extract_width_range_list, ms2_extract_width_range_list = build_precursors_matrix_step3(ms1_data_tensor, ms2_data_tensor, frag_repeat_num, device=device)

# 10. Build precursor features
precursor_info_np_org = np.array(precursor_info_list)
precursor_info_choose = precursor_info_np_org[:, 0: 5]
delta_rt_kept = np.array([0] * len(precursor_info_choose)).reshape(-1, 1)
assay_rt_kept = np.array([assay_rt_kept_dict[ee[0]] for ee in precursors_list]).reshape(-1, 1)
assay_im_kept = np.array([assay_im_kept_dict[ee[0]] for ee in precursors_list]).reshape(-1, 1)

# 11. Create precursor_feat and frag_info
precursor_feat = np.column_stack([precursor_info_choose, assay_im_kept, assay_rt_kept, delta_rt_kept])
frag_info = build_frag_info(ms1_data_tensor, ms2_data_tensor, frag_repeat_num, device)

print("First code processing completed!")
print(f"precursor_feat shape: {precursor_feat.shape}")
print(f"Processing precursor: {precursors_list[0][0]}")

# ============== ADD LATER PROCESSING FROM SECOND CODE ==============

# Load TimsTOF data
bruker_d_folder_name = '/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d'
timstof_data = timstof_PASEF_20250506.TimsTOF(bruker_d_folder_name)

# Since we only have 1 precursor, use index 0 instead of 490
i = 0  # IMPORTANT: Changed from 490 to 0 since we only have 1 precursor
IM = precursor_feat[i][5]
RT = precursor_feat[i][6]
print(f"Processing precursor at index {i}: IM={IM}, RT={RT}") # Processing precursor at index 0: IM=0.8748147487640381, RT=49.73412322998047

# Extract MS1 range
ms1_range = slice((ms1_range_list[i].min().item()-1)/1000, (ms1_range_list[i].max().item()+1)/1000)
precursor_result = timstof_data[:, :, 0, ms1_range][['rt_values_min', 'mobility_values', 'mz_values', 'intensity_values']]
precursor_result['mz_values'] = np.ceil(precursor_result['mz_values'] * 1000)
print(ms1_range)

# 第一个导出点：过滤IM之前的数据
precursor_result.to_csv('python_precursor_result_before_IM_filter.csv', index=False)
print(f"Exported precursor_result before IM filter: {len(precursor_result)} rows")

# Filter by IM
precursor_result = precursor_result[(precursor_result['mobility_values'] <= IM + 0.05) & (precursor_result['mobility_values'] >= IM - 0.05)]
print(f"After IM filter: {len(precursor_result)} rows")

# 第二个导出点：过滤IM之后的数据
precursor_result.to_csv('python_precursor_result_after_IM_filter.csv', index=False)
print(f"Exported precursor_result after IM filter: {len(precursor_result)} rows")

# ============== 修改的MS2提取部分 ==============
print("\n========== 开始提取MS2碎片数据 ==========")
start_time = time.time()

# 打印MS2范围信息
print(f"MS2范围矩阵形状: {ms2_range_list[i].shape}")
print(f"将提取 66 个MS2碎片")

# 初始化结果列表和计时
frag_result = []
fragment_times = []

# 提取每个碎片
for j in range(0, 66):
    fragment_start = time.time()
    
    # 获取该碎片的m/z范围
    ms2_range_min = (ms2_range_list[i][j].min().item()-1)/1000
    ms2_range_max = (ms2_range_list[i][j].max().item()+1)/1000
    ms2_range = slice(ms2_range_min, ms2_range_max)
    
    # 提取数据
    fragment_data = timstof_data[:, :, ms1_range, ms2_range][['rt_values_min', 'mobility_values', 'mz_values', 'intensity_values']]
    frag_result.append(fragment_data)
    
    fragment_time = time.time() - fragment_start
    fragment_times.append(fragment_time)
    
    # 进度报告
    if j % 10 == 0:
        print(f"  已处理 {j+1}/66 个碎片，最近10个碎片平均耗时: {np.mean(fragment_times[-10:]):.2f}秒")
    
    # 打印前几个碎片的详细信息
    if j < 5:
        print(f"    碎片 {j}: m/z范围 {ms2_range_min:.4f} - {ms2_range_max:.4f}, 数据点数: {len(fragment_data)}")

# 合并所有碎片数据
concat_start = time.time()
frag_result = pd.concat(frag_result, ignore_index=True)
concat_time = time.time() - concat_start
print(f"\n合并数据耗时: {concat_time:.2f}秒")

# 转换m/z值为整数
frag_result['mz_values'] = np.ceil(frag_result['mz_values'] * 1000)

# 导出IM过滤前的MS2数据
frag_result.to_csv('python_frag_result_before_IM_filter.csv', index=False)
print(f"导出IM过滤前的MS2数据: {len(frag_result)} 行")

# 统计信息
print(f"\nIM过滤前的MS2数据统计:")
print(f"  总数据点数: {len(frag_result)}")
print(f"  m/z范围: {frag_result['mz_values'].min():.0f} - {frag_result['mz_values'].max():.0f}")
print(f"  RT范围: {frag_result['rt_values_min'].min():.2f} - {frag_result['rt_values_min'].max():.2f}")
print(f"  IM范围: {frag_result['mobility_values'].min():.4f} - {frag_result['mobility_values'].max():.4f}")

# Filter by IM
im_filter_start = time.time()
frag_result_filtered = frag_result[(frag_result['mobility_values'] <= IM + 0.05) & (frag_result['mobility_values'] >= IM - 0.05)]
im_filter_time = time.time() - im_filter_start

# 导出IM过滤后的MS2数据
frag_result_filtered.to_csv('python_frag_result_after_IM_filter.csv', index=False)
print(f"\nIM过滤耗时: {im_filter_time:.2f}秒")
print(f"IM过滤后的MS2数据: {len(frag_result_filtered)} 行 (保留比例: {len(frag_result_filtered)/len(frag_result)*100:.2f}%)")

# 更新frag_result为过滤后的结果
frag_result = frag_result_filtered

# 总耗时
total_time = time.time() - start_time
print(f"\n========== MS2提取完成 ==========")
print(f"总耗时: {total_time:.2f}秒")
print(f"平均每个碎片耗时: {np.mean(fragment_times):.2f}秒")

# ============== Build masks ==============
print("\n========== 构建Mask矩阵 ==========")
mask_start = time.time()

# 创建搜索张量
search_ms1_tensor = torch.tensor(list(precursor_result['mz_values']))
search_ms2_tensor = torch.tensor(list(frag_result['mz_values']))

print(f"MS1搜索集合大小: {len(search_ms1_tensor)}")
print(f"MS2搜索集合大小: {len(search_ms2_tensor)}")

# 构建mask
print("\n构建MS1 mask...")
mask_ms1_start = time.time()
mask_ms1 = torch.isin(ms1_extract_width_range_list[i], search_ms1_tensor)
ms1_frag_moz_matrix = torch.where(mask_ms1, 1., 0.)
mask_ms1_time = time.time() - mask_ms1_start
print(f"MS1 mask构建耗时: {mask_ms1_time:.2f}秒")

print("\n构建MS2 mask...")
mask_ms2_start = time.time()
mask_ms2 = torch.isin(ms2_extract_width_range_list[i], search_ms2_tensor)
ms2_frag_moz_matrix = torch.where(mask_ms2, 1., 0.)
mask_ms2_time = time.time() - mask_ms2_start
print(f"MS2 mask构建耗时: {mask_ms2_time:.2f}秒")

mask_total_time = time.time() - mask_start
print(f"\nMask构建总耗时: {mask_total_time:.2f}秒")

# 统计非零元素
ms1_nonzero = (ms1_frag_moz_matrix > 0).sum().item()
ms2_nonzero = (ms2_frag_moz_matrix > 0).sum().item()

print(f"\nMS1碎片矩阵形状: {ms1_frag_moz_matrix.shape}")
print(f"MS1碎片矩阵非零元素: {ms1_nonzero} / {ms1_frag_moz_matrix.numel()} ({ms1_nonzero/ms1_frag_moz_matrix.numel()*100:.2f}%)")

print(f"\nMS2碎片矩阵形状: {ms2_frag_moz_matrix.shape}")
print(f"MS2碎片矩阵非零元素: {ms2_nonzero} / {ms2_frag_moz_matrix.numel()} ({ms2_nonzero/ms2_frag_moz_matrix.numel()*100:.2f}%)")

# 保存mask矩阵
np.savetxt('python_ms1_frag_moz_matrix.csv', ms1_frag_moz_matrix.numpy(), delimiter=',', fmt='%.0f')
np.savetxt('python_ms2_frag_moz_matrix.csv', ms2_frag_moz_matrix.numpy(), delimiter=',', fmt='%.0f')
print("\n已保存mask矩阵到CSV文件")

# 保存矩阵摘要信息
with open('python_mask_matrices_summary.txt', 'w') as f:
    f.write("=== Python Mask Matrices Summary ===\n\n")
    f.write(f"MS1 Fragment Matrix:\n")
    f.write(f"  Shape: {ms1_frag_moz_matrix.shape}\n")
    f.write(f"  Non-zero elements: {ms1_nonzero} / {ms1_frag_moz_matrix.numel()}\n")
    f.write(f"  Density: {ms1_nonzero/ms1_frag_moz_matrix.numel()*100:.2f}%\n\n")
    
    f.write(f"MS2 Fragment Matrix:\n")
    f.write(f"  Shape: {ms2_frag_moz_matrix.shape}\n")
    f.write(f"  Non-zero elements: {ms2_nonzero} / {ms2_frag_moz_matrix.numel()}\n")
    f.write(f"  Density: {ms2_nonzero/ms2_frag_moz_matrix.numel()*100:.2f}%\n\n")
    
    f.write(f"Timing Information:\n")
    f.write(f"  MS2 extraction total time: {total_time:.2f}s\n")
    f.write(f"  Average time per fragment: {np.mean(fragment_times):.2f}s\n")
    f.write(f"  Mask building total time: {mask_total_time:.2f}s\n")

print("\n========== 程序完成 ==========")
print(f"总体执行时间: {time.time() - start_time:.2f}秒")

# =========================================================

# Get RT list
all_rt = get_rt_list(list(set(list(precursor_result['rt_values_min'].unique())+list(frag_result['rt_values_min'].unique()))), RT)

# Build intensity matrices
rsm_matrix = []

# MS1 intensity matrix
ms1_frag_rt_matrix = []
for a in range(len(ms1_extract_width_range_list[i])):
    ms1_moz_rt_matrix = []
    for rt in all_rt:
        grouped = precursor_result[precursor_result['rt_values_min'] == rt].groupby('mz_values')['intensity_values'].sum()
        moz_to_intensity = grouped.to_dict()
        mapped_intensities = torch.tensor([moz_to_intensity.get(moz.item(), 0) for moz in ms1_extract_width_range_list[i][a]], 
                                        dtype=torch.float32)
        ms1_moz_rt_matrix.append(mapped_intensities)
    ms1_moz_rt_matrix = torch.stack(ms1_moz_rt_matrix, dim=1)
    ms1_frag_rt = ms1_frag_moz_matrix[a] @ ms1_moz_rt_matrix
    ms1_frag_rt_matrix.append(ms1_frag_rt)
ms1_frag_rt_matrix = torch.stack(ms1_frag_rt_matrix, dim=0)

# MS2 intensity matrix
ms2_frag_rt_matrix = []
for b in range(len(ms2_extract_width_range_list[i])):
    ms2_moz_rt_matrix = []
    for rt in all_rt:
        grouped = frag_result[frag_result['rt_values_min'] == rt].groupby('mz_values')['intensity_values'].sum()
        moz_to_intensity = grouped.to_dict()
        mapped_intensities = torch.tensor([moz_to_intensity.get(moz.item(), 0) for moz in ms2_extract_width_range_list[i][b]], 
                                        dtype=torch.float32)
        ms2_moz_rt_matrix.append(mapped_intensities)
    ms2_moz_rt_matrix = torch.stack(ms2_moz_rt_matrix, dim=1)
    ms2_frag_rt = ms2_frag_moz_matrix[b] @ ms2_moz_rt_matrix
    ms2_frag_rt_matrix.append(ms2_frag_rt)
ms2_frag_rt_matrix = torch.stack(ms2_frag_rt_matrix, dim=0)

# Reshape and combine
ms1_frag_rt_matrix_shape = ms1_frag_rt_matrix.shape
ms1_frag_rt_matrix1 = ms1_frag_rt_matrix.reshape(frag_repeat_num, ms1_frag_rt_matrix_shape[0] // frag_repeat_num, ms1_frag_rt_matrix_shape[1])
ms2_frag_rt_matrix_shape = ms2_frag_rt_matrix.shape
ms2_frag_rt_matrix1 = ms2_frag_rt_matrix.reshape(frag_repeat_num, ms2_frag_rt_matrix_shape[0] // frag_repeat_num, ms2_frag_rt_matrix_shape[1])

full_frag_rt_matrix = torch.cat([ms1_frag_rt_matrix1, ms2_frag_rt_matrix1], dim=1)
rsm_matrix.append(full_frag_rt_matrix)
rsm_matrix = torch.stack(rsm_matrix, dim=0)
aggregated_x_sum = torch.sum(rsm_matrix, dim=1)

# Create final data frame
header = all_rt+['ProductMz', 'LibraryIntensity', 'frag_type', 'FragmentType']
merged_tensor = torch.cat((aggregated_x_sum[0], frag_info[i]), dim=1)
numpy_array = merged_tensor.cpu().numpy()
data = pd.DataFrame(numpy_array)
data.columns = header

print("Data processing completed!")
print(f"Final data shape: {data.shape}")

# ============== VISUALIZATION ==============

# Plot
filtered_df = data[data['frag_type'].isin([1, 2])]
columns_to_plot = filtered_df.drop(columns=['frag_type','ProductMz','LibraryIntensity','FragmentType']).columns
values_to_plot = filtered_df[columns_to_plot].values
row_indices = filtered_df.index

plt.figure(figsize=(10, 6))
for t, row_values in enumerate(values_to_plot):
    original_row_index = row_indices[t]
    category_value = filtered_df.loc[original_row_index, 'frag_type']
    row_series = pd.Series(row_values, index=columns_to_plot)
    non_zero_series = row_series[row_series != 0]
    plt.plot(non_zero_series.index, non_zero_series.values, marker='o',
             label=f'Row {original_row_index} (frag_type: {category_value})')

plt.xlabel('Features')
plt.ylabel('Values')
plt.title(f'Line Chart for Precursor {precursors_list[0][0]} - Rows with frag_type 1 or 2')
plt.xticks(rotation=45, ha='right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

print("Visualization completed!")

# ============== EXPORT TO CSV FOR COMPARISON ==============

# # Export the final dataframe to CSV (same as Rust output)
# output_csv_path = 'precursor_result_python.csv'

# # Reorder columns to match expected format:
# # RT columns first, then ProductMz, LibraryIntensity, frag_type, FragmentType
# rt_columns = [col for col in data.columns if col not in ['ProductMz', 'LibraryIntensity', 'frag_type', 'FragmentType']]
# other_columns = ['ProductMz', 'LibraryIntensity', 'frag_type', 'FragmentType']
# ordered_columns = rt_columns + other_columns

# # Create reordered dataframe
# data_reordered = data[ordered_columns]

# # Save to CSV
# data_reordered.to_csv(output_csv_path, index=False)
# print(f"\nData exported to CSV: {output_csv_path}")
# print(f"CSV file contains {len(data_reordered)} rows and {len(data_reordered.columns)} columns")

# # Print first few rows for verification
# print("\nFirst 5 rows of exported data:")
# print(data_reordered.head())

# # Print column names for comparison
# print("\nColumn names in order:")
# for i, col in enumerate(data_reordered.columns):
#     print(f"  {i}: {col}")

# # Print data types for debugging
# print("\nData types:")
# print(data_reordered.dtypes)

# # Additional verification: check for any differences in data format
# print("\nData summary:")
# print(f"Number of RT columns: {len(rt_columns)}")
# print(f"Number of rows with frag_type 1: {len(data_reordered[data_reordered['frag_type'] == 1])}")
# print(f"Number of rows with frag_type 2: {len(data_reordered[data_reordered['frag_type'] == 2])}")
# print(f"Number of rows with frag_type 5: {len(data_reordered[data_reordered['frag_type'] == 5])}")

# # Export filtered data (frag_type 1 or 2 only) for plotting comparison
# filtered_for_plot = data_reordered[data_reordered['frag_type'].isin([1, 2])]
# filtered_for_plot.to_csv('precursor_result_python_filtered.csv', index=False)
# print(f"\nFiltered data (frag_type 1 or 2) exported to: precursor_result_python_filtered.csv")
# print(f"Filtered CSV contains {len(filtered_for_plot)} rows")