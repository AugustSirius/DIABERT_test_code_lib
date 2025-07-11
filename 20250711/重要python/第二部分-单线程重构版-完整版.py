import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import groupby
import pickle

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

# 1
library = pd.read_csv("/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/lib/TPHPlib_frag1025_swissprot_final_all_from_Yueliang.tsv", sep="\t")

# 2
# col mapping
lib_col_dict = get_lib_col_dict()

for col in set(library.columns) & set(lib_col_dict.keys()):
    library.loc[:, lib_col_dict[col]] = library.loc[:, col]

# 3
library['transition_group_id'] = library['FullUniModPeptideName'] + library['PrecursorCharge'].astype(str)
replacement_dict = {'b': 1, 'y': 2, 'p': 3}
library['FragmentType'] = library['FragmentType'].replace(replacement_dict)
library['decoy'] = 0
# precursor_id_list = ["AAAAAAALQAK2"]
precursor_id_list = ["LLIYGASTR2"]

# 4
report_diann = pd.read_parquet('/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/report/report.parquet')
report_diann['transition_group_id'] = report_diann['Precursor.Id']

# 5
diann_result = pd.merge(library[['transition_group_id', 'PrecursorMz', 'ProductMz']], report_diann[['transition_group_id', 'RT', 'IM','iIM']], on='transition_group_id', how='left').dropna(subset=['RT'])

# 6
diann_precursor_id_all = diann_result.drop_duplicates(subset=['transition_group_id'])[['transition_group_id', 'RT', 'IM']].reset_index(drop=True)

# 7
assay_rt_kept_dict = dict(zip(diann_precursor_id_all['transition_group_id'], diann_precursor_id_all['RT']))
assay_im_kept_dict = dict(zip(diann_precursor_id_all['transition_group_id'], diann_precursor_id_all['IM']))

# 8
each_lib_data = library[library['transition_group_id'].isin(precursor_id_list)]

# =======================================================================================
lib_cols = {'PRECURSOR_MZ_COL': 'PrecursorMz','IRT_COL': 'Tr_recalibrated','PRECURSOR_ID_COL': 'transition_group_id','FULL_SEQUENCE_COL': 'FullUniModPeptideName','PURE_SEQUENCE_COL': 'PeptideSequence','PRECURSOR_CHARGE_COL': 'PrecursorCharge','FRAGMENT_MZ_COL': 'ProductMz','FRAGMENT_SERIES_COL': 'FragmentNumber','FRAGMENT_CHARGE_COL': 'FragmentCharge','FRAGMENT_TYPE_COL': 'FragmentType','LIB_INTENSITY_COL': 'LibraryIntensity','PROTEIN_NAME_COL': 'ProteinName','DECOY_OR_NOT_COL': 'decoy'}
MS1_ISOTOPE_COUNT = 6      # MS1同位素峰的数量
FRAGMENT_VARIANTS = 3       # 每个碎片的变体数量（原始、轻、重）
MS1_TYPE_MARKER = 5         # MS1类型标识符
MS1_FRAGMENT_TYPE = 1       # MS1碎片类型
VARIANT_ORIGINAL = 2        # 原始碎片
VARIANT_LIGHT = 3          # 轻同位素
VARIANT_HEAVY = 4          # 重同位素

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
    charge = first_fragment[1]  # PrecursorCharge
    precursor_mz = first_fragment[5]  # PrecursorMz
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
    variant_type_column[:fragment_num] = VARIANT_ORIGINAL  # 2
    variant_type_column[fragment_num:fragment_num * 2] = VARIANT_LIGHT  # 3
    variant_type_column[fragment_num * 2:] = VARIANT_HEAVY  # 4
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
    # 返回一个numpy数组，与原始版本保持一致
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
    # 获取前体分组
    precursor_groups = get_precursor_indice(lib_data[lib_cols["PRECURSOR_ID_COL"]])
    
    # 准备结果列表
    all_precursors = []
    all_ms1_data = []
    all_ms2_data = []
    all_precursor_info = []
    
    # 获取前体信息
    first_indices = [group[0] for group in precursor_groups]
    cols = [lib_cols['PRECURSOR_ID_COL'], 'decoy']
    precursors_list = lib_data.iloc[first_indices][cols].values.tolist()
    
    # 获取所有碎片数据
    all_indices = []
    for group in precursor_groups:
        all_indices.extend(group)
    
    group_data_col = ['ProductMz', 'PrecursorCharge', 'FragmentCharge', 'LibraryIntensity', 
                      'FragmentType', 'PrecursorMz','Tr_recalibrated', 'PeptideSequence', 
                      'decoy', 'transition_group_id']
    fragment_data = lib_data.iloc[all_indices][group_data_col].values
    
    # 处理每个前体组
    for i, indices in enumerate(precursor_groups):
        group_fragments = fragment_data[indices]
        ms1, ms2, info = format_ms_data(
            group_fragments, iso_range, mz_max, max_fragment
        )
        all_precursors.append(precursors_list[i])
        all_ms1_data.append(ms1)
        all_ms2_data.append(ms2)
        all_precursor_info.append(info)
    
    # 非常重要：返回列表，不要转换为numpy数组！
    # 保持与原始版本完全一致
    return all_precursors, all_ms1_data, all_ms2_data, all_precursor_info

# 9
precursors_list, ms1_data_list, ms2_data_list, precursor_info_list = build_lib_matrix(each_lib_data,lib_cols,iso_range=5,mz_max=1801,max_fragment=20)

# =========================================================================================
import torch

device = 'cpu'
frag_repeat_num = 5

def build_precursors_matrix_step1(ms1_data_list, ms2_data_list, device='cpu'):
    ms1_data_tensor = torch.tensor(ms1_data_list, dtype=torch.float32, device=device)
    ms2_data_tensor = torch.tensor(ms2_data_list, dtype=torch.float32, device=device)
    return ms1_data_tensor, ms2_data_tensor

def build_precursors_matrix_step2(ms2_data_tensor):
    # times = time.time()
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
        mz_tol_half = (mz_to_extract / mz_to_extract) * mz_tol / 2 # 获取mz_to_extract的维度
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
    # times = time.time()
    re_ms1_data_tensor = ms1_data_tensor.repeat(1, frag_repeat_num, 1)

    re_ms2_data_tensor = ms2_data_tensor.repeat(1, frag_repeat_num, 1)
    ms1_extract_width_range_list = extract_width(re_ms1_data_tensor[:, :, 0], mz_unit, mz_tol_ms1,
                                                            device=device)
    ms2_extract_width_range_list = extract_width(re_ms2_data_tensor[:, :, 0], mz_unit, mz_tol_ms2,
                                                            device=device)
    return re_ms1_data_tensor, re_ms2_data_tensor, ms1_extract_width_range_list, ms2_extract_width_range_list

# 10
ms1_data_tensor, ms2_data_tensor = build_precursors_matrix_step1(ms1_data_list, ms2_data_list, device)
ms2_data_tensor = build_precursors_matrix_step2(ms2_data_tensor)
ms1_range_list, ms2_range_list = build_range_matrix_step3(ms1_data_tensor, ms2_data_tensor, frag_repeat_num, device=device)
ms1_data_tensor, ms2_data_tensor, ms1_extract_width_range_list, ms2_extract_width_range_list = build_precursors_matrix_step3(ms1_data_tensor, ms2_data_tensor, frag_repeat_num, device=device)

# 11
precursor_info_np_org = np.array(precursor_info_list)
precursor_info_choose = precursor_info_np_org[:, 0: 5]
delta_rt_kept = np.array([0] * len(precursor_info_choose)).reshape(-1, 1)

# 12
assay_rt_kept = np.array([assay_rt_kept_dict[ee[0]] for ee in precursors_list]).reshape(-1, 1)
assay_im_kept = np.array([assay_im_kept_dict[ee[0]] for ee in precursors_list]).reshape(-1, 1)

def build_ext_ms1_matrix(ms1_data_tensor, device):
    ext_matrix = ms1_data_tensor[:, :, [0, 3, 8, 4]].to(device)
    return ext_matrix

def build_ext_ms2_matrix(ms2_data_tensor, device):
    ext_matrix = ms2_data_tensor[:, :, [0, 3, 8, 4]].to(device)
    return ext_matrix

def build_frag_info(ms1_data_tensor, ms2_data_tensor, frag_repeat_num, device):
    ext_ms1_precursors_frag_rt_matrix = build_ext_ms1_matrix(
        ms1_data_tensor, device)
    ext_ms2_precursors_frag_rt_matrix = build_ext_ms2_matrix(
        ms2_data_tensor, device)

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
    return frag_info

# 13
# precursor_feat
precursor_feat = np.column_stack([precursor_info_choose, assay_im_kept, assay_rt_kept, delta_rt_kept])

# 14
# frag info
frag_info = build_frag_info(ms1_data_tensor, ms2_data_tensor, frag_repeat_num,device)

print(frag_info)