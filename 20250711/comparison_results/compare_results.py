#!/usr/bin/env python3
"""
Comparison script to verify that Python and Rust implementations produce identical results.
"""

import json
import numpy as np
import pandas as pd
import os
from pathlib import Path

def load_json_results(filepath):
    """Load JSON results file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def compare_basic_metrics(python_results, rust_results):
    """Compare basic metrics between Python and Rust results."""
    print("=" * 60)
    print("📊 BASIC METRICS COMPARISON")
    print("=" * 60)
    
    metrics_to_compare = [
        'ms1_data_list_len',
        'ms2_data_list_len', 
        'precursor_info_list_len',
        'ms1_data_tensor_shape',
        'ms2_data_tensor_shape',
        'ms1_range_list_shape',
        'ms2_range_list_shape',
        'ms1_extract_width_range_list_shape',
        'ms2_extract_width_range_list_shape'
    ]
    
    all_match = True
    for metric in metrics_to_compare:
        python_val = python_results.get(metric, 'N/A')
        rust_val = rust_results.get(metric, 'N/A')
        
        if python_val == rust_val:
            status = "✅ MATCH"
        else:
            status = "❌ DIFF"
            all_match = False
        
        # Convert values to strings for formatting
        python_str = str(python_val)
        rust_str = str(rust_val)
        
        print(f"{metric:35} | Python: {python_str:20} | Rust: {rust_str:20} | {status}")
    
    return all_match

def compare_arrays(python_results, rust_results):
    """Compare array values between Python and Rust results."""
    print("\n" + "=" * 60)
    print("📊 ARRAY VALUES COMPARISON")
    print("=" * 60)
    
    arrays_to_compare = [
        'delta_rt_kept',
        'assay_rt_kept', 
        'assay_im_kept'
    ]
    
    all_match = True
    for array_name in arrays_to_compare:
        python_arr = np.array(python_results.get(array_name, []))
        rust_arr = np.array(rust_results.get(array_name, []))
        
        if python_arr.shape == rust_arr.shape:
            if np.allclose(python_arr, rust_arr, rtol=1e-10, atol=1e-10):
                status = "✅ MATCH"
            else:
                status = "❌ VALUES DIFFER"
                all_match = False
                # Show some statistics about the differences
                diff = np.abs(python_arr - rust_arr)
                print(f"  Max difference: {np.max(diff):.2e}")
                print(f"  Mean difference: {np.mean(diff):.2e}")
                print(f"  Non-zero differences: {np.count_nonzero(diff)}")
        else:
            status = "❌ SHAPE DIFF"
            all_match = False
            
        print(f"{array_name:20} | Python: {python_arr.shape} | Rust: {rust_arr.shape} | {status}")
    
    return all_match

def compare_precursor_lists(python_results, rust_results):
    """Compare precursor lists between Python and Rust results."""
    print("\n" + "=" * 60)
    print("📊 PRECURSOR LISTS COMPARISON")
    print("=" * 60)
    
    python_precursors = python_results.get('precursors_list', [])
    rust_precursors = rust_results.get('precursors_list', [])
    
    if python_precursors == rust_precursors:
        print("✅ Precursor lists are identical")
        return True
    else:
        print("❌ Precursor lists differ")
        print(f"Python length: {len(python_precursors)}")
        print(f"Rust length: {len(rust_precursors)}")
        
        # Show first few differences
        min_len = min(len(python_precursors), len(rust_precursors))
        for i in range(min_len):
            if python_precursors[i] != rust_precursors[i]:
                print(f"  Difference at index {i}:")
                print(f"    Python: {python_precursors[i]}")
                print(f"    Rust: {rust_precursors[i]}")
                break
        
        return False

def compare_csv_files(output_dir):
    """Compare CSV files generated by both implementations."""
    print("\n" + "=" * 60)
    print("📊 CSV FILES COMPARISON")
    print("=" * 60)
    
    csv_files = [
        'ms1_data_tensor.csv',
        'ms2_data_tensor.csv',
        'ms1_range_list.csv',
        'ms2_range_list.csv',
        'ms1_extract_width_range_list.csv',
        'ms2_extract_width_range_list.csv',
        'precursor_info_choose.csv'
    ]
    
    all_match = True
    for csv_file in csv_files:
        python_file = os.path.join(output_dir, f"python_{csv_file}")
        rust_file = os.path.join(output_dir, f"rust_{csv_file}")
        
        if not os.path.exists(python_file) or not os.path.exists(rust_file):
            print(f"❌ {csv_file:30} | Missing files")
            all_match = False
            continue
        
        try:
            # Read CSV files (skip header with shape info)
            python_data = pd.read_csv(python_file, comment='#', header=None)
            rust_data = pd.read_csv(rust_file, comment='#', header=None)
            
            if python_data.shape == rust_data.shape:
                # Compare numerical values
                if python_data.select_dtypes(include=[np.number]).shape[1] > 0:
                    numerical_cols = python_data.select_dtypes(include=[np.number]).columns
                    if np.allclose(python_data[numerical_cols].values, 
                                 rust_data[numerical_cols].values, 
                                 rtol=1e-10, atol=1e-10):
                        status = "✅ MATCH"
                    else:
                        status = "❌ VALUES DIFFER"
                        all_match = False
                else:
                    # For non-numerical data, do exact comparison
                    if python_data.equals(rust_data):
                        status = "✅ MATCH"
                    else:
                        status = "❌ DIFFER"
                        all_match = False
            else:
                status = "❌ SHAPE DIFF"
                all_match = False
                
        except Exception as e:
            status = f"❌ ERROR: {str(e)}"
            all_match = False
            
        print(f"{csv_file:30} | {status}")
    
    return all_match

def compare_numpy_files(output_dir):
    """Compare numpy files generated by Python implementation."""
    print("\n" + "=" * 60)
    print("📊 NUMPY FILES ANALYSIS")
    print("=" * 60)
    
    numpy_files = [
        'python_ms1_data_tensor.npy',
        'python_ms2_data_tensor.npy',
        'python_ms1_range_list.npy',
        'python_ms2_range_list.npy',
        'python_ms1_extract_width_range_list.npy',
        'python_ms2_extract_width_range_list.npy',
        'python_precursor_info_choose.npy',
        'python_delta_rt_kept.npy',
        'python_assay_rt_kept.npy',
        'python_assay_im_kept.npy'
    ]
    
    for npy_file in numpy_files:
        filepath = os.path.join(output_dir, npy_file)
        if os.path.exists(filepath):
            try:
                data = np.load(filepath)
                shape_str = str(data.shape)
                dtype_str = str(data.dtype)
                print(f"{npy_file:40} | Shape: {shape_str:15} | Dtype: {dtype_str}")
                if data.size > 0:
                    if np.issubdtype(data.dtype, np.number):
                        print(f"{'':42} | Min: {np.min(data):10.4f} | Max: {np.max(data):10.4f} | Mean: {np.mean(data):10.4f}")
                    else:
                        print(f"{'':42} | Non-numerical data")
                else:
                    print(f"{'':42} | Empty array")
            except Exception as e:
                print(f"{npy_file:40} | ❌ Error loading: {str(e)}")
        else:
            print(f"{npy_file:40} | ❌ File not found")

def main():
    """Main comparison function."""
    output_dir = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/20250711/comparison_results"
    
    print("🔍 PYTHON vs RUST IMPLEMENTATION COMPARISON")
    print("=" * 60)
    
    # Load JSON results
    python_json = os.path.join(output_dir, "python_results.json")
    rust_json = os.path.join(output_dir, "rust_results.json")
    
    if not os.path.exists(python_json):
        print(f"❌ Python results not found: {python_json}")
        return
    
    if not os.path.exists(rust_json):
        print(f"❌ Rust results not found: {rust_json}")
        return
    
    python_results = load_json_results(python_json)
    rust_results = load_json_results(rust_json)
    
    # Compare basic metrics
    basic_match = compare_basic_metrics(python_results, rust_results)
    
    # Compare arrays
    arrays_match = compare_arrays(python_results, rust_results)
    
    # Compare precursor lists
    precursors_match = compare_precursor_lists(python_results, rust_results)
    
    # Compare CSV files
    csv_match = compare_csv_files(output_dir)
    
    # Analyze numpy files
    compare_numpy_files(output_dir)
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎯 FINAL COMPARISON SUMMARY")
    print("=" * 60)
    
    overall_match = basic_match and arrays_match and precursors_match and csv_match
    
    if overall_match:
        print("🎉 ✅ ALL RESULTS MATCH - Python and Rust implementations are identical!")
    else:
        print("⚠️  ❌ DIFFERENCES FOUND - Python and Rust implementations differ")
        print("\nStatus breakdown:")
        print(f"  Basic metrics: {'✅ MATCH' if basic_match else '❌ DIFFER'}")
        print(f"  Array values:  {'✅ MATCH' if arrays_match else '❌ DIFFER'}")
        print(f"  Precursor lists: {'✅ MATCH' if precursors_match else '❌ DIFFER'}")
        print(f"  CSV files:     {'✅ MATCH' if csv_match else '❌ DIFFER'}")
    
    # Show processing time comparison
    python_time = python_results.get('processing_time_seconds', 0)
    rust_time = rust_results.get('processing_time_seconds', 0)
    
    print(f"\n⏱️  Processing time comparison:")
    print(f"  Python: {python_time:.6f} seconds")
    print(f"  Rust:   {rust_time:.6f} seconds")
    
    if rust_time > 0 and python_time > 0:
        speedup = python_time / rust_time
        print(f"  Speedup: {speedup:.2f}x ({'Rust is faster' if speedup > 1 else 'Python is faster'})")

if __name__ == "__main__":
    main() 