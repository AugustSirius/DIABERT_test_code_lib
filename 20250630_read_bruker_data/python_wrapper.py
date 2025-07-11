# python_wrapper.py
import json
import sys
import os
import numpy as np
import pandas as pd
import torch

# Add the utils directory to path
sys.path.append('/Users/augustsirius/Desktop/DIABERT_test_code_lib/read_bruker_data')
import utils

def numpy_to_list(obj):
    """Convert numpy arrays to lists for JSON serialization"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, torch.Tensor):
        return obj.cpu().numpy().tolist()
    elif isinstance(obj, (list, tuple)):
        return [numpy_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: numpy_to_list(value) for key, value in obj.items()}
    else:
        return obj

def process_library_data(inputs):
    """
    Main processing function that wraps your Python code
    
    Args:
        inputs: Dictionary containing:
            - library_data: DataFrame data as dict
            - precursor_id_list: List of precursor IDs
            - report_path: Path to report parquet file
    
    Returns:
        Dictionary containing all processed results
    """
    try:
        # Extract inputs
        library_dict = inputs['library_data']
        precursor_id_list = inputs['precursor_id_list']
        report_path = inputs['report_path']
        
        # Convert library dict back to DataFrame
        library = pd.DataFrame(library_dict)
        
        # Your original Python code starts here - completely unchanged
        device = 'cpu'
        frag_repeat_num = 5
        
        report_diann = pd.read_parquet(report_path)
        report_diann['transition_group_id'] = report_diann['Precursor.Id']
        
        diann_result = pd.merge(
            library[['transition_group_id', 'PrecursorMz', 'ProductMz']], 
            report_diann[['transition_group_id', 'RT', 'IM','iIM']], 
            on='transition_group_id', 
            how='left'
        ).dropna(subset=['RT'])
        
        diann_precursor_id_all = diann_result.drop_duplicates(subset=['transition_group_id'])[
            ['transition_group_id', 'RT', 'IM']
        ].reset_index(drop=True)
        
        assay_rt_kept_dict = dict(zip(
            diann_precursor_id_all['transition_group_id'], 
            diann_precursor_id_all['RT']
        ))
        assay_im_kept_dict = dict(zip(
            diann_precursor_id_all['transition_group_id'], 
            diann_precursor_id_all['IM']
        ))
        
        each_lib_data = library[library['transition_group_id'].isin(precursor_id_list)]
        
        precursors_list, ms1_data_list, ms2_data_list, precursor_info_list = utils.build_lib_matrix(
            each_lib_data,
            utils.lib_cols,
            None,
            None,
            5,
            1801,
            20,
            None
        )
        
        ms1_data_tensor, ms2_data_tensor = utils.build_precursors_matrix_step1(
            ms1_data_list, ms2_data_list, device
        )
        ms2_data_tensor = utils.build_precursors_matrix_step2(ms2_data_tensor)
        
        ms1_range_list, ms2_range_list = utils.build_range_matrix_step3(
            ms1_data_tensor, ms2_data_tensor, frag_repeat_num, device=device
        )
        
        ms1_data_tensor, ms2_data_tensor, ms1_extract_width_range_list, ms2_extract_width_range_list = \
            utils.build_precursors_matrix_step3(
                ms1_data_tensor, ms2_data_tensor, frag_repeat_num, device=device
            )
        
        precursor_info_np_org = np.array(precursor_info_list)
        precursor_info_choose = precursor_info_np_org[:, 0: 5]
        delta_rt_kept = np.array([0] * len(precursor_info_choose)).reshape(-1, 1)
        
        assay_rt_kept = np.array([assay_rt_kept_dict.get(ee[0], 0) for ee in precursors_list]).reshape(-1, 1)
        assay_im_kept = np.array([assay_im_kept_dict.get(ee[0], 0) for ee in precursors_list]).reshape(-1, 1)
        
        # precursor_feat
        precursor_feat = np.column_stack([
            precursor_info_choose, assay_im_kept, assay_rt_kept, delta_rt_kept
        ])
        
        # frag info
        frag_info = utils.build_frag_info(
            ms1_data_tensor, ms2_data_tensor, frag_repeat_num, device
        )
        
        # Package all results for return
        results = {
            'precursors_list': precursors_list,
            'ms1_data': numpy_to_list(ms1_data_list),
            'ms2_data': numpy_to_list(ms2_data_list),
            'precursor_info': numpy_to_list(precursor_info_list),
            'precursor_feat': numpy_to_list(precursor_feat),
            'frag_info': numpy_to_list(frag_info),
            'ms1_range_list': numpy_to_list(ms1_range_list),
            'ms2_range_list': numpy_to_list(ms2_range_list),
        }
        
        return results
        
    except Exception as e:
        print(f"Error in Python processing: {str(e)}")
        import traceback
        traceback.print_exc()
        raise