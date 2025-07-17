from collections import defaultdict
import numpy as np
import torch

qt3_length = 6

type_column_qt3 = 5
type_column_lib = 0
type_column_light = -1
type_column_iso = 1

frag_type_qt3 = 1
frag_type_lib = 2
frag_type_light = 3
frag_type_iso = 4

frag_type_num = 3

frag_type_dict = {'qt3': 1, 'lib': 2, 'light': 3, 'iso': 4}

group_data_col = ['ProductMz', 'PrecursorCharge', 'FragmentCharge', 'LibraryIntensity', 'FragmentType', 'PrecursorMz',
                  'Tr_recalibrated', 'PeptideSequence', 'decoy', 'transition_group_id']

'''
分隔frag，超过最大值截取，小于暂时不处理
'''

lib_cols = {'PRECURSOR_MZ_COL': 'PrecursorMz',
                'IRT_COL': 'Tr_recalibrated',
                'PRECURSOR_ID_COL': 'transition_group_id',
                'FULL_SEQUENCE_COL': 'FullUniModPeptideName',
                'PURE_SEQUENCE_COL': 'PeptideSequence',
                'PRECURSOR_CHARGE_COL': 'PrecursorCharge',
                'FRAGMENT_MZ_COL': 'ProductMz',
                'FRAGMENT_SERIES_COL': 'FragmentNumber',
                'FRAGMENT_CHARGE_COL': 'FragmentCharge',
                'FRAGMENT_TYPE_COL': 'FragmentType',
                'LIB_INTENSITY_COL': 'LibraryIntensity',
                'PROTEIN_NAME_COL': 'ProteinName',
                'DECOY_OR_NOT_COL': 'decoy'}

def list_split(data_arr, each_num):
    return [data_arr[i: i + each_num] for i in range(0, len(data_arr), each_num)]



def intercept_frags_sort(frag_list, length):
    frag_list.sort(reverse=True)
    if len(frag_list) > length:
        frag_list = frag_list[0: length]
    return frag_list


def build_ms1_data(frag_list, iso_range, mz_max):
    eg_frag = frag_list[0]
    charge = eg_frag[1]
    precursor_mz = eg_frag[5]
    iso_shift_max = int(min(iso_range, (mz_max - precursor_mz) * charge)) + 1
    qt3_frags = [precursor_mz + iso_shift / charge for iso_shift in range(iso_shift_max)]
    qt3_frags = intercept_frags_sort(qt3_frags, qt3_length)
    qt3_data = [
        [qt3_frag, eg_frag[1], eg_frag[2], eg_frag[3], 3, eg_frag[5], type_column_qt3, 0, frag_type_qt3] for
        qt3_frag in qt3_frags]
    if len(qt3_data) < qt3_length:
        qt3_data.extend([[0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(qt3_length - len(qt3_data))])
    return np.array(qt3_data)


'''
'''


def build_ms2_data(frag_list, max_fragment_num):
    frag_count = max_fragment_num * frag_type_num
    frag_num = frag_list.shape[0]
    frag_list = np.vstack([frag_list] * frag_type_num)
    win_id_column = np.array([0] * frag_num * frag_type_num)
    type_column = np.array([0] * frag_num * frag_type_num)
    type_column[frag_num: frag_num * (frag_type_num - 1)] = -1
    type_column[(frag_type_num - 1) * frag_num: frag_type_num * frag_num] = 1

    frag_type_column = np.array([0] * frag_num * frag_type_num)
    frag_type_column[:frag_num] = 2
    frag_type_column[frag_num: frag_num * (frag_type_num - 1)] = 3
    frag_type_column[(frag_type_num - 1) * frag_num: frag_type_num * frag_num] = 4

    frag_list = np.hstack(
        (frag_list, type_column[:, np.newaxis], win_id_column[:, np.newaxis], frag_type_column[:, np.newaxis]))
    if len(frag_list) >= frag_count:
        return frag_list[:frag_count]
    frag_moz = np.zeros((frag_count, frag_list.shape[1]))
    frag_moz[:len(frag_list)] = frag_list
    return frag_moz


def build_precursor_info(frag_list, diann_raw_rt_map):
    precursor_data = frag_list[0]
    if diann_raw_rt_map is not None:
        return [len(precursor_data[7]), precursor_data[5], precursor_data[1], precursor_data[6], len(frag_list),
                diann_raw_rt_map[precursor_data[9]]]
    else:
        return [len(precursor_data[7]), precursor_data[5], precursor_data[1], precursor_data[6], len(frag_list),
                0]

def format_ms_data(frag_list, iso_range, mz_max, max_fragment, diann_raw_rt_map):
    qt3_moz = build_ms1_data(frag_list, iso_range, mz_max)
    frag_moz = build_ms2_data(frag_list[:, 0:6], max_fragment)
    ms1_moz = np.copy(qt3_moz)
    ms1_moz[:, 8] = 5
    frag_moz = np.concatenate([frag_moz, ms1_moz], axis=0)
    precursor_info = build_precursor_info(frag_list, diann_raw_rt_map)
    return qt3_moz, frag_moz, precursor_info


def precursors_data_group_thread(lib_cols, library, diann_raw_rt_map, precursor_index_arr, iso_range, mz_max,
                                 max_fragment, chunk_index, process_result_arr=None):
    # t1 = time.time()
    first_index_list = [idx[0] for idx in precursor_index_arr]
    precursors_list = library.iloc[first_index_list, :][[lib_cols['PRECURSOR_ID_COL'], 'decoy']].values.tolist()
    # t2 = time.time()

    all_index_list = []
    for idx in precursor_index_arr:
        all_index_list.extend(idx)
    group_data_col_values = library.iloc[all_index_list, :][group_data_col].values

    ms_moz_list = [
        format_ms_data(group_data_col_values[idx], iso_range, mz_max, max_fragment, diann_raw_rt_map)
        for idx in
        precursor_index_arr]
    # t3 = time.time()

    ms1_data_list = np.array([d[0] for d in ms_moz_list])
    ms2_data_list = np.array([d[1] for d in ms_moz_list])
    precursor_info_list = np.array([d[2] for d in ms_moz_list])

    if process_result_arr is not None:
        process_result_arr.append((precursors_list, ms1_data_list, ms2_data_list, precursor_info_list))
    # t4 = time.time()
    return precursors_list, ms1_data_list, ms2_data_list, precursor_info_list


def build_lib_matrix(lib_data, lib_cols, run_env, diann_raw_rt_map, iso_range,
                     mz_max, max_fragment, thread_num=None):
    # logger.info('start build lib matrix')
    # times = time.time()
    # logger.info('start calc tear library')
    precursor_indice, chunk_indice = tear_library(lib_data, lib_cols, 1)
    # logger.info('end calc tear library, time: {}, chunk_indice len: {}'.format(time.time() - times, len(chunk_indice)))

    precursors_list = []
    ms1_data_list = []
    ms2_data_list = []
    prec_data_list = []

    for i, chunk_index in enumerate(chunk_indice):
        precursor_index = [precursor_indice[idx] for idx in chunk_index]
        each_process_result = precursors_data_group_thread(lib_cols, lib_data, diann_raw_rt_map, precursor_index,
                                                           iso_range, mz_max, max_fragment, chunk_index, None)
        precursors_list.extend(each_process_result[0])
        ms1_data_list.extend(each_process_result[1])
        ms2_data_list.extend(each_process_result[2])
        prec_data_list.extend(each_process_result[3])
    # t4 = time.time()
    # logger.debug('build lib matrix time: {}'.format(t4 - times))
    return precursors_list, ms1_data_list, ms2_data_list, prec_data_list


def get_lib_col_dict():
    lib_col_dict = defaultdict(str)

    for key in ['transition_group_id', 'PrecursorID']:
        lib_col_dict[key] = 'transition_group_id'

    for key in ['PeptideSequence', 'Sequence', 'StrippedPeptide']:
        lib_col_dict[key] = 'PeptideSequence'

    for key in ['FullUniModPeptideName', 'ModifiedPeptide', 'LabeledSequence', 'modification_sequence',
                'ModifiedPeptideSequence']:
        lib_col_dict[key] = 'FullUniModPeptideName'

    for key in ['PrecursorCharge', 'Charge', 'prec_z']:
        lib_col_dict[key] = 'PrecursorCharge'

    for key in ['PrecursorMz', 'Q1']:
        lib_col_dict[key] = 'PrecursorMz'

    for key in ['Tr_recalibrated', 'iRT', 'RetentionTime', 'NormalizedRetentionTime', 'RT_detected']:
        lib_col_dict[key] = 'Tr_recalibrated'

    for key in ['ProductMz', 'FragmentMz', 'Q3']:
        lib_col_dict[key] = 'ProductMz'

    for key in ['FragmentType', 'FragmentIonType', 'ProductType', 'ProductIonType', 'frg_type']:
        lib_col_dict[key] = 'FragmentType'

    for key in ['FragmentCharge', 'FragmentIonCharge', 'ProductCharge', 'ProductIonCharge', 'frg_z']:
        lib_col_dict[key] = 'FragmentCharge'

    for key in ['FragmentNumber', 'frg_nr', 'FragmentSeriesNumber']:
        lib_col_dict[key] = 'FragmentNumber'

    for key in ['LibraryIntensity', 'RelativeIntensity', 'RelativeFragmentIntensity', 'RelativeFragmentIonIntensity',
                'relative_intensity']:
        lib_col_dict[key] = 'LibraryIntensity'

    # exclude
    for key in ['FragmentLossType', 'FragmentIonLossType', 'ProductLossType', 'ProductIonLossType']:
        lib_col_dict[key] = 'FragmentLossType'

    for key in ['ProteinID', 'ProteinId', 'UniprotID', 'uniprot_id', 'UniProtIds']:
        lib_col_dict[key] = 'ProteinID'

    for key in ['ProteinName', 'Protein Name', 'Protein_name', 'protein_name']:
        lib_col_dict[key] = 'ProteinName'

    for key in ['Gene', 'Genes', 'GeneName']:
        lib_col_dict[key] = 'Gene'

    for key in ['Decoy', 'decoy']:
        lib_col_dict[key] = 'decoy'

    for key in ['ExcludeFromAssay', 'ExcludeFromQuantification']:
        lib_col_dict[key] = 'ExcludeFromAssay'
    return lib_col_dict


def get_precursor_indice(precursor_ids):
    precursor_indice = []
    last_record = ""
    prec_index = [0]
    for i, prec in enumerate(precursor_ids):
        if last_record != prec:
            if i:
                precursor_indice.append(prec_index)
                prec_index = [i]
        else:
            prec_index.append(i)
        last_record = prec
    precursor_indice.append(prec_index)
    return precursor_indice


def tear_library(library, lib_cols, n_threads):
    precursor_indice = get_precursor_indice(library[lib_cols["PRECURSOR_ID_COL"]])
    n_precursors = len(precursor_indice)
    n_each_chunk = n_precursors // n_threads
    chunk_indice = [[k + i * n_each_chunk for k in range(n_each_chunk)] for i in range(n_threads)]
    chunk_indice[-1].extend([i for i in range(n_each_chunk * n_threads, n_precursors)])

    return precursor_indice, chunk_indice


def build_precursors_matrix_step1(ms1_data_list, ms2_data_list, device='cpu'):
    # times = time.time()
    ms1_data_tensor = torch.tensor(ms1_data_list, dtype=torch.float32, device=device)
    ms2_data_tensor = torch.tensor(ms2_data_list, dtype=torch.float32, device=device)
    # timee = time.time()
    # logger.debug('step 1 time: {}'.format(timee - times))
    return ms1_data_tensor, ms2_data_tensor

def build_precursors_matrix_step2(ms2_data_tensor):
    # times = time.time()
    ms2_data_tensor[:, :, 0] = ms2_data_tensor[:, :, 0] + ms2_data_tensor[:, :, 6] / ms2_data_tensor[:, :, 2]
    ms2_data_tensor[torch.isinf(ms2_data_tensor)] = 0
    ms2_data_tensor[torch.isnan(ms2_data_tensor)] = 0
    # timee = time.time()
    # logger.debug('step 2 time: {}'.format(timee - times))
    return ms2_data_tensor


def build_precursors_matrix_step3(ms1_data_tensor, ms2_data_tensor, frag_repeat_num=5, mz_unit='ppm', mz_tol_ms1=20,
                                  mz_tol_ms2=50, device='cpu'):
    # times = time.time()
    re_ms1_data_tensor = ms1_data_tensor.repeat(1, frag_repeat_num, 1)

    re_ms2_data_tensor = ms2_data_tensor.repeat(1, frag_repeat_num, 1)
    ms1_extract_width_range_list = extract_width(re_ms1_data_tensor[:, :, 0], mz_unit, mz_tol_ms1,
                                                            device=device)
    ms2_extract_width_range_list = extract_width(re_ms2_data_tensor[:, :, 0], mz_unit, mz_tol_ms2,
                                                            device=device)

    # timee = time.time()
    # logger.debug('step 3 time: {}'.format(timee - times))
    return re_ms1_data_tensor, re_ms2_data_tensor, ms1_extract_width_range_list, ms2_extract_width_range_list
    # return ms1_data_tensor, ms2_data_tensor



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

    # t1 = time.time()
    # batch_num = int(mz_to_extract.shape[1] / frag_repeat_num)
    # cha_tensor = (extract_width_list[:, 0:batch_num, 1] - extract_width_list[:, 0:batch_num, 0]) / frag_repeat_num
    #
    # extract_width_list[:, 0:batch_num, 0] = extract_width_list[:, 0:batch_num, 0]
    # extract_width_list[:, 0:batch_num, 1] = extract_width_list[:, 0:batch_num, 0] + cha_tensor - 1
    #
    # extract_width_list[:, batch_num:batch_num * 2, 0] = extract_width_list[:, 0:batch_num, 0] + cha_tensor
    # extract_width_list[:, batch_num:batch_num * 2, 1] = extract_width_list[:, 0:batch_num, 0] + 2 * cha_tensor - 1
    #
    # extract_width_list[:, batch_num * 2:batch_num * 3, 0] = extract_width_list[:, 0:batch_num, 0] + 2 * cha_tensor
    # extract_width_list[:, batch_num * 2:batch_num * 3, 1] = extract_width_list[:, 0:batch_num, 0] + 3 * cha_tensor - 1
    #
    # extract_width_list[:, batch_num * 3:batch_num * 4, 0] = extract_width_list[:, 0:batch_num, 0] + 3 * cha_tensor
    # extract_width_list[:, batch_num * 3:batch_num * 4, 1] = extract_width_list[:, 0:batch_num, 0] + 4 * cha_tensor - 1
    #
    # extract_width_list[:, batch_num * 4:batch_num * 5, 0] = extract_width_list[:, 0:batch_num, 0] + 4 * cha_tensor
    # extract_width_list[:, batch_num * 4:batch_num * 5, 1] = extract_width_list[:, 0:batch_num, 0] + 5 * cha_tensor - 1
    #
    # new_tensor = torch.zeros(mz_to_extract.shape[0], mz_to_extract.shape[1], max_moz_num, dtype=torch.float32,
    #                          device=device)
    # for i in range(new_tensor.shape[2]):
    #     new_tensor[:, :, i] = extract_width_list[:, :, 0] + i * 1
    #     condition = new_tensor[:, :, i] > extract_width_list[:, :, 1]
    #     new_tensor[:, :, i][condition] = 0
    # return new_tensor, mz_tol_half
    return extract_width_range_list

def build_range_matrix_step3(ms1_data_tensor, ms2_data_tensor, frag_repeat_num=5, mz_unit='ppm', mz_tol_ms1=20,
                                  mz_tol_ms2=50, device='cpu'):
    # times = time.time()
    re_ms1_data_tensor = ms1_data_tensor.repeat(1, frag_repeat_num, 1)

    re_ms2_data_tensor = ms2_data_tensor.repeat(1, frag_repeat_num, 1)
    ms1_extract_width_range_list = extract_width_2(re_ms1_data_tensor[:, :, 0], mz_unit, mz_tol_ms1,
                                                            device=device)
    ms2_extract_width_range_list = extract_width_2(re_ms2_data_tensor[:, :, 0], mz_unit, mz_tol_ms2,
                                                            device=device)

    # timee = time.time()
    # logger.debug('step 3 time: {}'.format(timee - times))
    return ms1_extract_width_range_list, ms2_extract_width_range_list
    # return ms1_data_tensor, ms2_data_tensor

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
    # 不能超过50，两边各一半就是0.025（需要*1000）
    condition = mz_tol_half[:, :] > mz_tol_half_num
    mz_tol_half[condition] = mz_tol_half_num

    mz_tol_half = torch.ceil(mz_tol_half * 1000 / frag_repeat_num) * frag_repeat_num

    # 把每个数据取[-half, +half]范围，取整就是总的数据个数，最大取值50个
    extract_width_list = torch.stack((mz_to_extract * 1000 - mz_tol_half, mz_to_extract * 1000 + mz_tol_half),
                                     dim=-1).floor()
    # 再处理一下，因为是要均匀分为5份，所以每一份非0的数据都要一致 0-60,60-120,120-180,180-340,240-300，各份的起始值和结束值不一样，需要设定好范围
    # output_dir = 'E:\data\diart\chen\\raw_csv\\P20190100166-DS-TB11'
    # import pandas as pd
    # dd = extract_width_list[0].numpy()
    # if len(dd) == 30:
    #     csv_filename = os.path.join(output_dir, 'ms1_mz.csv')
    # else:
    #     csv_filename = os.path.join(output_dir, 'ms2_mz.csv')
    # data_df = pd.DataFrame(dd, columns=['min', 'max'])
    # data_df.to_csv(csv_filename, index=False)

    # t1 = time.time()
    # 每5个批次的等差
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



def build_ext_ms1_matrix(ms1_data_tensor, device):
    ext_matrix = ms1_data_tensor[:, :, [0, 3, 8, 4]].to(device)
    return ext_matrix


'''
'''


def build_ext_ms2_matrix(ms2_data_tensor, device):
    ext_matrix = ms2_data_tensor[:, :, [0, 3, 8, 4]].to(device)
    return ext_matrix

