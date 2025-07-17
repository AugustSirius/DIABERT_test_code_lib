#!/usr/bin/env python3
"""
Compare outputs between new and old Rust versions.
"""

import json
import numpy as np
import pandas as pd
import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import sys

def load_json(filepath: str) -> Dict[str, Any]:
    """Load JSON file."""
    if not os.path.exists(filepath):
        print(f"Warning: File not found: {filepath}")
        return {}
    
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_timstof_data(new_data: Dict, old_data: Dict, step_name: str) -> bool:
    """Compare TimsTOF data structures."""
    print(f"\n=== Comparing {step_name} ===")
    
    if not new_data or not old_data:
        print(f"❌ Missing data files for {step_name}")
        return False
    
    # Check lengths
    new_len = new_data.get('length', 0)
    old_len = old_data.get('length', 0)
    
    print(f"Data length - New: {new_len}, Old: {old_len}")
    
    if new_len != old_len:
        print(f"❌ Length mismatch: {new_len} vs {old_len}")
        return False
    
    # Check each field
    fields = ['rt_values_min', 'mobility_values', 'mz_values', 'intensity_values', 'frame_indices', 'scan_indices']
    all_match = True
    
    for field in fields:
        if field not in new_data or field not in old_data:
            print(f"❌ Missing field: {field}")
            all_match = False
            continue
        
        new_vals = np.array(new_data[field])
        old_vals = np.array(old_data[field])
        
        if new_vals.shape != old_vals.shape:
            print(f"❌ Shape mismatch in {field}: {new_vals.shape} vs {old_vals.shape}")
            all_match = False
            continue
        
        if field in ['rt_values_min', 'mobility_values', 'mz_values']:
            # For float fields, use tolerance
            if np.allclose(new_vals, old_vals, rtol=1e-9, atol=1e-9):
                print(f"✅ {field}: Match (tolerance check)")
            else:
                max_diff = np.max(np.abs(new_vals - old_vals))
                print(f"❌ {field}: Mismatch (max diff: {max_diff:.2e})")
                all_match = False
        else:
            # For integer fields, exact match
            if np.array_equal(new_vals, old_vals):
                print(f"✅ {field}: Exact match")
            else:
                diff_count = np.sum(new_vals != old_vals)
                print(f"❌ {field}: {diff_count} differences")
                all_match = False
    
    return all_match

def compare_matrix_data(new_data: Dict, old_data: Dict, step_name: str) -> bool:
    """Compare matrix data structures."""
    print(f"\n=== Comparing {step_name} ===")
    
    if not new_data or not old_data:
        print(f"❌ Missing data files for {step_name}")
        return False
    
    # Check shapes
    new_shape = new_data.get('shape', [])
    old_shape = old_data.get('shape', [])
    
    print(f"Matrix shape - New: {new_shape}, Old: {old_shape}")
    
    if new_shape != old_shape:
        print(f"❌ Shape mismatch: {new_shape} vs {old_shape}")
        return False
    
    # Check data
    new_arr = np.array(new_data['data'])
    old_arr = np.array(old_data['data'])
    
    if np.allclose(new_arr, old_arr, rtol=1e-6, atol=1e-6):
        print(f"✅ Matrix data: Match (tolerance check)")
        return True
    else:
        max_diff = np.max(np.abs(new_arr - old_arr))
        print(f"❌ Matrix data: Mismatch (max diff: {max_diff:.2e})")
        
        # Show some statistics
        diff_mask = ~np.isclose(new_arr, old_arr, rtol=1e-6, atol=1e-6)
        diff_count = np.sum(diff_mask)
        total_count = new_arr.size
        print(f"   {diff_count}/{total_count} ({diff_count/total_count*100:.2f}%) elements differ")
        
        return False

def compare_vector_data(new_data: Dict, old_data: Dict, step_name: str) -> bool:
    """Compare vector data structures."""
    print(f"\n=== Comparing {step_name} ===")
    
    if not new_data or not old_data:
        print(f"❌ Missing data files for {step_name}")
        return False
    
    # Check lengths
    new_len = new_data.get('length', 0)
    old_len = old_data.get('length', 0)
    
    print(f"Vector length - New: {new_len}, Old: {old_len}")
    
    if new_len != old_len:
        print(f"❌ Length mismatch: {new_len} vs {old_len}")
        return False
    
    # Check data
    new_arr = np.array(new_data['data'])
    old_arr = np.array(old_data['data'])
    
    if np.allclose(new_arr, old_arr, rtol=1e-9, atol=1e-9):
        print(f"✅ Vector data: Match (tolerance check)")
        return True
    else:
        max_diff = np.max(np.abs(new_arr - old_arr))
        print(f"❌ Vector data: Mismatch (max diff: {max_diff:.2e})")
        return False

def compare_library_records(new_data: Dict, old_data: Dict, step_name: str) -> bool:
    """Compare library records data structures."""
    print(f"\n=== Comparing {step_name} ===")
    
    if not new_data or not old_data:
        print(f"❌ Missing data files for {step_name}")
        return False
    
    # Check counts
    new_count = new_data.get('count', 0)
    old_count = old_data.get('count', 0)
    
    print(f"Library records count - New: {new_count}, Old: {old_count}")
    
    if new_count != old_count:
        print(f"❌ Count mismatch: {new_count} vs {old_count}")
        return False
    
    # Check records
    new_records = new_data.get('records', [])
    old_records = old_data.get('records', [])
    
    if len(new_records) != len(old_records):
        print(f"❌ Records length mismatch: {len(new_records)} vs {len(old_records)}")
        return False
    
    # Compare each record
    all_match = True
    field_names = ['transition_group_id', 'decoy', 'product_mz', 'precursor_mz', 
                   'tr_recalibrated', 'library_intensity', 'fragment_type', 'fragment_series_number']
    
    for i, (new_record, old_record) in enumerate(zip(new_records, old_records)):
        for field in field_names:
            if field not in new_record or field not in old_record:
                print(f"❌ Missing field {field} in record {i}")
                all_match = False
                continue
            
            new_val = new_record[field]
            old_val = old_record[field]
            
            if field in ['product_mz', 'precursor_mz', 'tr_recalibrated', 'library_intensity']:
                # For float fields, use tolerance
                try:
                    # Convert to float in case they are strings
                    new_float = float(new_val) if new_val is not None else 0.0
                    old_float = float(old_val) if old_val is not None else 0.0
                    if not np.isclose(new_float, old_float, rtol=1e-9, atol=1e-9):
                        print(f"❌ Record {i}, field {field}: {new_val} vs {old_val}")
                        all_match = False
                except (ValueError, TypeError):
                    # If conversion fails, do exact string comparison
                    if new_val != old_val:
                        print(f"❌ Record {i}, field {field}: {new_val} vs {old_val} (string comparison)")
                        all_match = False
            else:
                # For other fields, exact match
                if new_val != old_val:
                    print(f"❌ Record {i}, field {field}: {new_val} vs {old_val}")
                    all_match = False
    
    if all_match:
        print("✅ Library records: Match")
    
    return all_match

def compare_general_json_data(new_data: Dict, old_data: Dict, step_name: str) -> bool:
    """Compare general JSON data structures."""
    print(f"\n=== Comparing {step_name} ===")
    
    if not new_data or not old_data:
        print(f"❌ Missing data files for {step_name}")
        return False
    
    # Check if both have the same keys
    new_keys = set(new_data.keys())
    old_keys = set(old_data.keys())
    
    if new_keys != old_keys:
        print(f"❌ Key mismatch - New: {new_keys}, Old: {old_keys}")
        return False
    
    # Compare each key
    all_match = True
    for key in new_keys:
        new_val = new_data[key]
        old_val = old_data[key]
        
        if isinstance(new_val, list) and isinstance(old_val, list):
            if len(new_val) != len(old_val):
                print(f"❌ {key} length mismatch: {len(new_val)} vs {len(old_val)}")
                all_match = False
                continue
            
            # Check if it's a list of numbers
            if new_val and isinstance(new_val[0], (int, float)):
                new_arr = np.array(new_val)
                old_arr = np.array(old_val)
                if not np.allclose(new_arr, old_arr, rtol=1e-9, atol=1e-9):
                    max_diff = np.max(np.abs(new_arr - old_arr))
                    print(f"❌ {key} data mismatch (max diff: {max_diff:.2e})")
                    all_match = False
                else:
                    print(f"✅ {key}: Match")
            else:
                # For non-numeric lists, compare directly
                if new_val != old_val:
                    print(f"❌ {key}: Direct comparison failed")
                    all_match = False
                else:
                    print(f"✅ {key}: Match")
        
        elif isinstance(new_val, dict) and isinstance(old_val, dict):
            # Recursively compare dictionaries
            if not compare_general_json_data(new_val, old_val, f"{step_name}.{key}"):
                all_match = False
        
        elif isinstance(new_val, (int, float)) and isinstance(old_val, (int, float)):
            # Compare numbers
            if not np.isclose(new_val, old_val, rtol=1e-9, atol=1e-9):
                print(f"❌ {key}: {new_val} vs {old_val}")
                all_match = False
            else:
                print(f"✅ {key}: Match")
        
        else:
            # Direct comparison for other types
            if new_val != old_val:
                print(f"❌ {key}: {new_val} vs {old_val}")
                all_match = False
            else:
                print(f"✅ {key}: Match")
    
    return all_match

def compare_csv_files(new_file: str, old_file: str, step_name: str) -> bool:
    """Compare CSV files."""
    print(f"\n=== Comparing {step_name} ===")
    
    if not os.path.exists(new_file) or not os.path.exists(old_file):
        print(f"❌ Missing CSV files for {step_name}")
        return False
    
    try:
        new_df = pd.read_csv(new_file)
        old_df = pd.read_csv(old_file)
        
        print(f"CSV shape - New: {new_df.shape}, Old: {old_df.shape}")
        
        if new_df.shape != old_df.shape:
            print(f"❌ Shape mismatch: {new_df.shape} vs {old_df.shape}")
            return False
        
        # Compare column names
        if list(new_df.columns) != list(old_df.columns):
            print(f"❌ Column names mismatch")
            print(f"   New: {list(new_df.columns)[:5]}...")
            print(f"   Old: {list(old_df.columns)[:5]}...")
            return False
        
        # Compare data
        for col in new_df.columns:
            if new_df[col].dtype in ['float64', 'float32']:
                # For float columns, use tolerance
                if not np.allclose(new_df[col], old_df[col], rtol=1e-6, atol=1e-6, equal_nan=True):
                    diff_mask = ~np.isclose(new_df[col], old_df[col], rtol=1e-6, atol=1e-6, equal_nan=True)
                    diff_count = np.sum(diff_mask)
                    print(f"❌ Column {col}: {diff_count} differences")
                    return False
            else:
                # For other columns, exact match
                if not new_df[col].equals(old_df[col]):
                    print(f"❌ Column {col}: Exact mismatch")
                    return False
        
        print("✅ CSV files: Match")
        return True
        
    except Exception as e:
        print(f"❌ Error comparing CSV files: {e}")
        return False

def main():
    """Main comparison function."""
    base_dir = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/20250711/rust_comparison_results"
    
    if not os.path.exists(base_dir):
        print(f"❌ Base directory not found: {base_dir}")
        return
    
    print("🔍 Comparing Rust versions outputs...")
    print("=" * 60)
    
    all_passed = True
    first_failure_step = None
    
    # Define comparison steps in logical order
    file_patterns = [
        # Step 0: Initial library processing
        ("Step 0: Library Records", "step0_each_lib_data.json", "library_records"),
        ("Step 0A: Precursors List", "step0a_precursors_list.json", "general_json"),
        ("Step 0A: MS1 Data List", "step0a_ms1_data_list.json", "general_json"),
        ("Step 0A: MS2 Data List", "step0a_ms2_data_list.json", "general_json"),
        ("Step 0A: Precursor Info List", "step0a_precursor_info_list.json", "general_json"),
        
        # Step 0B-0E: Tensor building steps
        ("Step 0B: MS1 Tensor Step 1", "step0b_ms1_tensor_step1.json", "matrix"),
        ("Step 0B: MS2 Tensor Step 1", "step0b_ms2_tensor_step1.json", "matrix"),
        ("Step 0C: MS2 Tensor Step 2", "step0c_ms2_tensor_step2.json", "matrix"),
        ("Step 0D: MS1 Range List", "step0d_ms1_range_list.json", "matrix"),
        ("Step 0D: MS2 Range List", "step0d_ms2_range_list.json", "matrix"),
        ("Step 0E: MS1 Extract Width Range", "step0e_ms1_extract_width_range_list.json", "matrix"),
        ("Step 0E: MS2 Extract Width Range", "step0e_ms2_extract_width_range_list.json", "matrix"),
        
        # Step 0F-0H: Precursor processing
        ("Step 0F: Assay RT Dict", "step0f_assay_rt_kept_dict.json", "general_json"),
        ("Step 0F: Assay IM Dict", "step0f_assay_im_kept_dict.json", "general_json"),
        ("Step 0F: Precursor Info Choose", "step0f_precursor_info_choose.json", "general_json"),
        ("Step 0G: Precursor Feature Matrix", "step0g_precursor_feat.json", "matrix"),
        ("Step 0H: Fragment Info", "step0h_frag_info.json", "matrix"),
        
        # Step 0I-0J: TimsTOF data processing
        # ("Step 0I: TimsTOF Summary", "step0i_timstof_summary.json", "general_json"),
        # ("Step 0J: M/Z Range Details", "step0j_mz_range_details.json", "general_json"),
        
        # Step 1-12: Final processing steps
        # ("Step 1: Precursor Result", "step1_precursor_result.json", "timstof_data"),
        # ("Step 2: Precursor Result (Int)", "step2_precursor_result_int.json", "timstof_data"),
        # ("Step 3: Precursor Result (Filtered)", "step3_precursor_result_filtered.json", "timstof_data"),
        # ("Step 4: Merged Fragment Result", "step4_merged_frag_result.json", "timstof_data"),
        # ("Step 5: Fragment Result (Filtered)", "step5_frag_result_filtered.json", "timstof_data"),
        ("Step 6: MS1 Mask Matrix", "step6_ms1_mask_matrix.json", "matrix"),
        ("Step 6: MS2 Mask Matrix", "step6_ms2_mask_matrix.json", "matrix"),
        ("Step 7: RT List", "step7_rt_list.json", "vector"),
        ("Step 8: MS1 Intensity Matrix", "step8_ms1_intensity_matrix.json", "matrix"),
        ("Step 8: MS2 Intensity Matrix", "step8_ms2_intensity_matrix.json", "matrix"),
        ("Step 9: MS1 Reshaped", "step9_ms1_reshaped.json", "matrix"),
        ("Step 9: MS2 Reshaped", "step9_ms2_reshaped.json", "matrix"),
        ("Step 10: Combined Matrix", "step10_combined_matrix.json", "matrix"),
        ("Step 11: Aggregated Matrix", "step11_aggregated_matrix.json", "matrix"),
        ("Step 12: Final DataFrame", "step12_final_dataframe.csv", "csv"),
    ]
    
    # Perform comparisons
    for step_name, file_pattern, data_type in file_patterns:
        new_file = os.path.join(base_dir, f"new_version_{file_pattern}")
        old_file = os.path.join(base_dir, f"old_version_{file_pattern}")
        
        # Check if files exist
        if not os.path.exists(new_file) or not os.path.exists(old_file):
            print(f"\n⚠️  Skipping {step_name} - missing files")
            continue
        
        # Perform comparison based on data type
        if data_type == "timstof_data":
            new_data = load_json(new_file)
            old_data = load_json(old_file)
            result = compare_timstof_data(new_data, old_data, step_name)
        elif data_type == "matrix":
            new_data = load_json(new_file)
            old_data = load_json(old_file)
            result = compare_matrix_data(new_data, old_data, step_name)
        elif data_type == "vector":
            new_data = load_json(new_file)
            old_data = load_json(old_file)
            result = compare_vector_data(new_data, old_data, step_name)
        elif data_type == "library_records":
            new_data = load_json(new_file)
            old_data = load_json(old_file)
            result = compare_library_records(new_data, old_data, step_name)
        elif data_type == "general_json":
            new_data = load_json(new_file)
            old_data = load_json(old_file)
            result = compare_general_json_data(new_data, old_data, step_name)
        elif data_type == "csv":
            result = compare_csv_files(new_file, old_file, step_name)
        else:
            print(f"❌ Unknown data type: {data_type}")
            result = False
        
        if not result:
            all_passed = False
            if first_failure_step is None:
                first_failure_step = step_name
            print(f"\n🚨 FIRST FAILURE DETECTED AT: {step_name}")
            print(f"   All steps before this were consistent.")
            print(f"   The problem starts here!")
            break  # Stop at first failure
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL COMPARISONS PASSED! The versions are consistent.")
    else:
        print("❌ VERSIONS HAVE DIFFERENCES!")
        if first_failure_step:
            print(f"🎯 FIRST FAILURE AT: {first_failure_step}")
            print(f"   → This is where the implementations diverge.")
            print(f"   → Check the logic in this step between the two versions.")
        else:
            print("   → Could not identify the first failure point.")
    
    print("=" * 60)

if __name__ == "__main__":
    main() 