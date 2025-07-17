import pandas as pd
import numpy as np
from collections import defaultdict
from itertools import groupby
import pickle
import torch
import os
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

# 5. Load report
report_diann = pd.read_parquet('/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/report/report.parquet')
report_diann['transition_group_id'] = report_diann['Precursor.Id']

# help me save first 5000 transition group id of report_diann to a txt file
with open('report_diann_transition_group_id.txt', 'w') as f:
    for transition_group_id in report_diann['transition_group_id'].iloc[:5000]:
        f.write(transition_group_id + '\n')
