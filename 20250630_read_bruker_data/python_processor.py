import pandas as pd
import numpy as np
import json
import sys

def process_second_part(library_data, precursor_id_list, report_path, device='cpu', frag_repeat_num=5):
    """
    Process the second part of the pipeline
    
    Args:
        library_data: Dictionary containing library data
        precursor_id_list: List of precursor IDs to process
        report_path: Path to the report parquet file
        device: Device to use (default: 'cpu')
        frag_repeat_num: Fragment repeat number (default: 5)
    
    Returns:
        Dictionary containing processed results
    """
    try:
        # Import utils module - make sure it's in the Python path
        import utils
        
        # Convert library data dict to DataFrame
        library = pd.DataFrame(library_data)
        
        # Read report
        report_diann = pd.read_parquet(report_path)
        report_diann['transition_group_id'] = report_diann['Precursor.Id']
        
        # Merge and process
        diann_result = pd.merge(
            library[['transition_group_id', 'PrecursorMz', 'ProductMz']], 
            report_diann[['transition_group_id', 'RT', 'IM','iIM']], 
            on='transition_group_id', 
            how='left'
        ).dropna(subset=['RT'])
        
        diann_precursor_id_all = diann_result.drop_duplicates(subset=['transition_group_id'])[
            ['transition_group_id', 'RT', 'IM']
        ].reset_index(drop=True)
        
        assay_rt_kept_dict = dict(zip(diann_precursor_id_all['transition_group_id'], diann_precursor_id_all['RT']))
        assay_im_kept_dict = dict(zip(diann_precursor_id_all['transition_group_id'], diann_precursor_id_all['IM']))
        
        # Process library data
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
        
        # Build tensors
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
        
        # Process precursor info
        precursor_info_np_org = np.array(precursor_info_list)
        precursor_info_choose = precursor_info_np_org[:, 0: 5]
        delta_rt_kept = np.array([0] * len(precursor_info_choose)).reshape(-1, 1)
        
        assay_rt_kept = np.array([assay_rt_kept_dict[ee[0]] for ee in precursors_list]).reshape(-1, 1)
        assay_im_kept = np.array([assay_im_kept_dict[ee[0]] for ee in precursors_list]).reshape(-1, 1)
        
        # Build precursor feat
        precursor_feat = np.column_stack([precursor_info_choose, assay_im_kept, assay_rt_kept, delta_rt_kept])
        
        # Build frag info
        frag_info = utils.build_frag_info(ms1_data_tensor, ms2_data_tensor, frag_repeat_num, device)
        
        # Convert numpy arrays to lists for JSON serialization
        result = {
            'precursor_feat': precursor_feat.tolist(),
            'frag_info': frag_info.tolist() if hasattr(frag_info, 'tolist') else frag_info,
            'ms1_range_list': [x.tolist() if hasattr(x, 'tolist') else x for x in ms1_range_list],
            'ms2_range_list': [x.tolist() if hasattr(x, 'tolist') else x for x in ms2_range_list],
            'precursors_list': precursors_list,
            'status': 'success'
        }
        
        return result
        
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e)
        }

if __name__ == "__main__":
    # This allows the script to be called standalone for testing
    pass