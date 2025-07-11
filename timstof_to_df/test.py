# -*- coding: utf-8 -*-
# 从 Untitled_im_rt_test.ipynb 转换而来

# ==================== Cell 1 ====================
import os
import torch
import timstof_PASEF_20250506
from copy import deepcopy
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
  
bruker_d_folder_name = '/Users/augustsirius/Desktop/DIABERT_test_code_lib/timstof_to_df/DIA_sample.d'
timstof_data = timstof_PASEF_20250506.TimsTOF(bruker_d_folder_name)
df = deepcopy(timstof_data[:,:,:,:,:])