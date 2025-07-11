import numpy as np
import pandas as pd
from timstof_PASEF_20250506 import TimsTOF
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

def extract_ms1_ms2_dataframes(bruker_d_folder_path):
    """
    Extract MS1 and MS2 dataframes from a Bruker .d folder.
    
    Parameters
    ----------
    bruker_d_folder_path : str
        Path to the Bruker .d folder
        
    Returns
    -------
    ms1_df : pd.DataFrame
        DataFrame containing MS1 data
    ms2_df : pd.DataFrame
        DataFrame containing MS2 data
    """
    
    # Load the TimsTOF data
    print(f"Loading data from {bruker_d_folder_path}...")
    data = TimsTOF(bruker_d_folder_path)
    
    # Extract MS1 data (precursor_index = 0)
    print("Extracting MS1 data...")
    ms1_indices = data[:, :, 0, :, "raw"]  # All frames, all scans, precursor_index=0, all tof indices
    
    # Convert MS1 indices to dataframe
    ms1_df = data.as_dataframe(
        ms1_indices,
        raw_indices=True,
        frame_indices=True,
        scan_indices=True,
        tof_indices=True,
        precursor_indices=True,
        rt_values=True,
        rt_values_min=True,
        mobility_values=True,
        quad_mz_values=True,
        mz_values=True,
        intensity_values=True,
        corrected_intensity_values=True,
        push_indices=False,
        quad_indices=False,
        raw_indices_sorted=False
    )
    
    # Filter to ensure we only have MS1 data
    ms1_df = ms1_df[ms1_df['precursor_indices'] == 0].copy()
    
    # Extract MS2 data (precursor_index > 0)
    print("Extracting MS2 data...")
    # For MS2, we want all precursors except 0
    ms2_indices = data[:, :, 1:, :, "raw"]  # All frames, all scans, precursor_indices > 0, all tof indices
    
    # Convert MS2 indices to dataframe
    if len(ms2_indices) > 0:
        ms2_df = data.as_dataframe(
            ms2_indices,
            raw_indices=True,
            frame_indices=True,
            scan_indices=True,
            tof_indices=True,
            precursor_indices=True,
            rt_values=True,
            rt_values_min=True,
            mobility_values=True,
            quad_mz_values=True,
            mz_values=True,
            intensity_values=True,
            corrected_intensity_values=True,
            push_indices=False,
            quad_indices=False,
            raw_indices_sorted=False
        )
        
        # Filter to ensure we only have MS2 data
        ms2_df = ms2_df[ms2_df['precursor_indices'] > 0].copy()
    else:
        # Create empty dataframe with the same columns if no MS2 data
        ms2_df = pd.DataFrame(columns=ms1_df.columns)
    
    # Add some useful columns
    ms1_df['ms_level'] = 1
    ms2_df['ms_level'] = 2
    
    # Sort by retention time and m/z for better organization
    ms1_df = ms1_df.sort_values(['rt_values', 'mz_values']).reset_index(drop=True)
    ms2_df = ms2_df.sort_values(['rt_values', 'precursor_indices', 'mz_values']).reset_index(drop=True)
    
    print(f"MS1 DataFrame shape: {ms1_df.shape}")
    print(f"MS2 DataFrame shape: {ms2_df.shape}")
    
    return ms1_df, ms2_df


def summarize_dataframes(ms1_df, ms2_df):
    """
    Print summary statistics for MS1 and MS2 dataframes.
    
    Parameters
    ----------
    ms1_df : pd.DataFrame
        MS1 dataframe
    ms2_df : pd.DataFrame
        MS2 dataframe
    """
    print("\n=== MS1 Summary ===")
    print(f"Total MS1 peaks: {len(ms1_df):,}")
    print(f"RT range: {ms1_df['rt_values'].min():.2f} - {ms1_df['rt_values'].max():.2f} seconds")
    print(f"m/z range: {ms1_df['mz_values'].min():.2f} - {ms1_df['mz_values'].max():.2f}")
    print(f"Mobility range: {ms1_df['mobility_values'].min():.4f} - {ms1_df['mobility_values'].max():.4f}")
    print(f"Intensity range: {ms1_df['intensity_values'].min()} - {ms1_df['intensity_values'].max()}")
    
    print("\n=== MS2 Summary ===")
    print(f"Total MS2 peaks: {len(ms2_df):,}")
    if len(ms2_df) > 0:
        print(f"Number of unique precursors: {ms2_df['precursor_indices'].nunique()}")
        print(f"RT range: {ms2_df['rt_values'].min():.2f} - {ms2_df['rt_values'].max():.2f} seconds")
        print(f"Fragment m/z range: {ms2_df['mz_values'].min():.2f} - {ms2_df['mz_values'].max():.2f}")
        print(f"Precursor m/z range: {ms2_df['quad_low_mz_values'].min():.2f} - {ms2_df['quad_high_mz_values'].max():.2f}")
        print(f"Mobility range: {ms2_df['mobility_values'].min():.4f} - {ms2_df['mobility_values'].max():.4f}")
        print(f"Intensity range: {ms2_df['intensity_values'].min()} - {ms2_df['intensity_values'].max()}")


# Example usage
if __name__ == "__main__":
    # Path to your .d file
    d_file_path = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d"
    
    try:
        # Extract MS1 and MS2 dataframes
        ms1_df, ms2_df = extract_ms1_ms2_dataframes(d_file_path)
        
        # Print summary statistics
        summarize_dataframes(ms1_df, ms2_df)
        
        # Display first few rows of each dataframe
        print("\n=== First 5 rows of MS1 DataFrame ===")
        print(ms1_df.head())
        
        print("\n=== First 5 rows of MS2 DataFrame ===")
        print(ms2_df.head())
        
        # Optional: Save to CSV files
        # ms1_df.to_csv("ms1_data.csv", index=False)
        # ms2_df.to_csv("ms2_data.csv", index=False)
        
        # Optional: Save to parquet files (more efficient for large datasets)
        # ms1_df.to_parquet("ms1_data.parquet", index=False)
        # ms2_df.to_parquet("ms2_data.parquet", index=False)
        
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()