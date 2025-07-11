import sys
import json
import pandas as pd
import numpy as np
import os
import traceback

# Add the python directory to path so we can import utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
import utils

def process_diann(library_csv_path, report_path, precursor_ids_json, output_dir):
    """
    Process DIANN report with library data
    
    Args:
        library_csv_path: Path to CSV file exported by Rust
        report_path: Path to report.parquet file
        precursor_ids_json: JSON string containing precursor IDs list
        output_dir: Directory to save output files
    
    Returns:
        JSON string with results
    """
    try:
        device = 'cpu'
        frag_repeat_num = 5
        
        # Read library from CSV (exported by Rust)
        print(f"Reading library from: {library_csv_path}")
        library = pd.read_csv(library_csv_path)
        print(f"Library shape: {library.shape}")
        
        # Parse precursor IDs
        precursor_id_list = json.loads(precursor_ids_json)
        print(f"Processing {len(precursor_id_list)} precursor IDs")
        
        # Read report
        print(f"Reading report from: {report_path}")
        report_diann = pd.read_parquet(report_path)
        report_diann['transition_group_id'] = report_diann['Precursor.Id']
        print(f"Report shape: {report_diann.shape}")
        
        # Merge
        print("Performing merge operation...")
        diann_result = pd.merge(
            library[['transition_group_id', 'PrecursorMz', 'ProductMz']], 
            report_diann[['transition_group_id', 'RT', 'IM', 'iIM']], 
            on='transition_group_id', 
            how='left'
        ).dropna(subset=['RT'])
        print(f"Merge result shape: {diann_result.shape}")
        
        # Extract unique precursor data
        diann_precursor_id_all = diann_result.drop_duplicates(
            subset=['transition_group_id']
        )[['transition_group_id', 'RT', 'IM']].reset_index(drop=True)
        
        # Create dictionaries
        assay_rt_kept_dict = dict(
            zip(diann_precursor_id_all['transition_group_id'], 
                diann_precursor_id_all['RT'])
        )
        assay_im_kept_dict = dict(
            zip(diann_precursor_id_all['transition_group_id'], 
                diann_precursor_id_all['IM'])
        )
        
        # Process specific precursors if provided
        if precursor_id_list:
            each_lib_data = library[library['transition_group_id'].isin(precursor_id_list)]
        else:
            each_lib_data = library
            
        print(f"Processing {len(each_lib_data)} library entries...")
        
        # Build matrices using utils
        precursors_list, ms1_data_list, ms2_data_list, precursor_info_list = utils.build_lib_matrix(
            each_lib_data,
            utils.lib_cols,
            None,  # run_env
            None,  # diann_raw_rt_map
            5,     # iso_range
            1801,  # mz_max
            20,    # max_fragment
            None   # thread_num
        )
        
        print("Building tensors...")
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
        
        # Build precursor features
        precursor_info_np_org = np.array(precursor_info_list)
        precursor_info_choose = precursor_info_np_org[:, 0:5]
        delta_rt_kept = np.array([0] * len(precursor_info_choose)).reshape(-1, 1)
        
        assay_rt_kept = np.array([assay_rt_kept_dict.get(ee[0], 0) for ee in precursors_list]).reshape(-1, 1)
        assay_im_kept = np.array([assay_im_kept_dict.get(ee[0], 0) for ee in precursors_list]).reshape(-1, 1)
        
        # Precursor features
        precursor_feat = np.column_stack([precursor_info_choose, assay_im_kept, assay_rt_kept, delta_rt_kept])
        
        # Fragment info
        frag_info = utils.build_frag_info(ms1_data_tensor, ms2_data_tensor, frag_repeat_num, device)
        
        # Save intermediate results
        os.makedirs(output_dir, exist_ok=True)
        
        # Save diann_result
        diann_result_path = os.path.join(output_dir, 'diann_result.csv')
        diann_result.to_csv(diann_result_path, index=False)
        
        # Save precursor features
        precursor_feat_path = os.path.join(output_dir, 'precursor_feat.npy')
        np.save(precursor_feat_path, precursor_feat)
        
        # Save fragment info
        frag_info_path = os.path.join(output_dir, 'frag_info.npy')
        np.save(frag_info_path, frag_info.numpy() if hasattr(frag_info, 'numpy') else frag_info)
        
        # Prepare result summary
        result = {
            'status': 'success',
            'diann_result_shape': list(diann_result.shape),
            'num_precursors': len(precursors_list),
            'precursor_feat_shape': list(precursor_feat.shape),
            'frag_info_shape': list(frag_info.shape),
            'output_files': {
                'diann_result': diann_result_path,
                'precursor_feat': precursor_feat_path,
                'frag_info': frag_info_path
            },
            'sample_data': {
                'first_5_precursors': precursors_list[:5] if precursors_list else [],
                'rt_dict_sample': dict(list(assay_rt_kept_dict.items())[:5]),
                'im_dict_sample': dict(list(assay_im_kept_dict.items())[:5])
            }
        }
        
        return json.dumps(result)
        
    except Exception as e:
        error_result = {
            'status': 'error',
            'error': str(e),
            'traceback': traceback.format_exc()
        }
        return json.dumps(error_result)

if __name__ == "__main__":
    if len(sys.argv) != 5:
        print("Usage: python process_diann.py <library_csv> <report_parquet> <precursor_ids_json> <output_dir>")
        sys.exit(1)
    
    library_csv_path = sys.argv[1]
    report_path = sys.argv[2]
    precursor_ids_json = sys.argv[3]
    output_dir = sys.argv[4]
    
    result = process_diann(library_csv_path, report_path, precursor_ids_json, output_dir)
    print(result)