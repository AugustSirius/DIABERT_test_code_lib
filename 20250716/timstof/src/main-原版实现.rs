mod utils;
mod cache;
mod processing;

use cache::CacheManager;
use utils::{
    read_timstof_data, build_indexed_data, read_parquet_with_polars,
    library_records_to_dataframe, merge_library_and_report, get_unique_precursor_ids, 
    process_library_fast, create_rt_im_dicts, build_lib_matrix, build_precursors_matrix_step1, 
    build_precursors_matrix_step2, build_range_matrix_step3, build_precursors_matrix_step3, 
    build_frag_info, LibCols,
};
use processing::{
    FastChunkFinder, build_intensity_matrix_optimized, prepare_precursor_features,
    calculate_mz_range, extract_ms2_data, build_mask_matrices, extract_aligned_rt_values,
    reshape_and_combine_matrices, create_final_dataframe,
};

use rayon::prelude::*;
use std::{error::Error, path::Path, time::Instant, env, fs::File};
use ndarray::{Array2, Array3, Array4, s, Axis};
use polars::prelude::*;

fn main() -> Result<(), Box<dyn Error>> {
    // Initialize global thread pool with all available CPU cores
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .unwrap();
    
    let args: Vec<String> = env::args().collect();
    
    // Handle command-line arguments for cache operations
    if let Some(arg) = args.get(1) {
        match arg.as_str() {
            "--clear-cache" => {
                CacheManager::new().clear_cache()?;
                return Ok(());
            }
            "--cache-info" => {
                let cache_manager = CacheManager::new();
                let info = cache_manager.get_cache_info()?;
                if info.is_empty() {
                    println!("Cache is empty");
                } else {
                    println!("Cache files:");
                    for (name, _, size_str) in info {
                        println!("  {} - {}", name, size_str);
                    }
                }
                return Ok(());
            }
            _ => {}
        }
    }
    
    // Set data folder path
    let d_folder = args.get(1).cloned().unwrap_or_else(|| {
        "/Users/augustsirius/Desktop/DIABERT_test_code_lib/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d".to_string()
    });
    
    let d_path = Path::new(&d_folder);
    if !d_path.exists() {
        return Err(format!("folder {:?} not found", d_path).into());
    }
    
    // ================================ DATA LOADING AND INDEXING ================================
    let cache_manager = CacheManager::new();
    
    println!("\n========== DATA PREPARATION PHASE ==========");
    let total_start = Instant::now();
    
    let (ms1_indexed, ms2_indexed_pairs) = if cache_manager.is_cache_valid(d_path) {
        println!("Found valid cache, loading indexed data directly...");
        let cache_load_start = Instant::now();
        let result = cache_manager.load_indexed_data(d_path)?;
        println!("Cache loading time: {:.5} seconds", cache_load_start.elapsed().as_secs_f32());
        result
    } else {
        println!("Cache invalid or non-existent, reading TimsTOF data...");
        
        // Read raw data
        let raw_data_start = Instant::now();
        let raw_data = read_timstof_data(d_path)?;
        println!("Raw data reading time: {:.5} seconds", raw_data_start.elapsed().as_secs_f32());
        println!("  - MS1 data points: {}", raw_data.ms1_data.mz_values.len());
        println!("  - MS2 windows: {}", raw_data.ms2_windows.len());
        
        // Build indexed data
        println!("\nBuilding indexed data structures...");
        let index_start = Instant::now();
        let (ms1_indexed, ms2_indexed_pairs) = build_indexed_data(raw_data)?;
        println!("Index building time: {:.5} seconds", index_start.elapsed().as_secs_f32());
        
        // Save to cache
        let cache_save_start = Instant::now();
        cache_manager.save_indexed_data(d_path, &ms1_indexed, &ms2_indexed_pairs)?;
        println!("Cache saving time: {:.5} seconds", cache_save_start.elapsed().as_secs_f32());
        
        (ms1_indexed, ms2_indexed_pairs)
    };
    
    println!("Total data preparation time: {:.5} seconds", total_start.elapsed().as_secs_f32());
    
    // Create MS2 finder for fast chunk lookup
    let finder = FastChunkFinder::new(ms2_indexed_pairs)?;
    
    // ================================ LIBRARY AND REPORT LOADING ================================
    println!("\n========== LIBRARY AND REPORT PROCESSING ==========");
    let lib_processing_start = Instant::now();
    
    let lib_file_path = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/lib/TPHPlib_frag1025_swissprot_final_all_from_Yueliang.tsv";
    let library_records = process_library_fast(lib_file_path)?;
    let library_df = library_records_to_dataframe(library_records.clone())?;
    
    let report_file_path = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/report/report.parquet";
    let report_df = read_parquet_with_polars(report_file_path)?;
    
    let diann_result = merge_library_and_report(library_df, report_df)?;
    let diann_precursor_id_all = get_unique_precursor_ids(&diann_result)?;
    let (assay_rt_kept_dict, assay_im_kept_dict) = create_rt_im_dicts(&diann_precursor_id_all)?;
    
    println!("Library and report processing time: {:.5} seconds", lib_processing_start.elapsed().as_secs_f32());
    
    // Set processing parameters
    let device = "cpu";
    let frag_repeat_num = 5;
    
    // ================================ SINGLE-PRECURSOR PIPELINE ================================
    
    // Step 1: Precursor Selection and Library Data Filtering
    println!("\n[Step 1] Precursor Selection and Library Data Filtering");
    let step1_start = Instant::now();
    
    let precursor_id_list = vec!["LLIYGASTR2"];
    let each_lib_data: Vec<_> = library_records
        .iter()
        .filter(|record| precursor_id_list.contains(&record.transition_group_id.as_str()))
        .cloned()
        .collect();
    
    if each_lib_data.is_empty() {
        println!("Specified precursor not found");
        return Ok(());
    }
    
    println!("  - Found {} library entries for precursor", each_lib_data.len());
    println!("  - Time elapsed: {:.5} seconds", step1_start.elapsed().as_secs_f32());
    
    // Step 2: Build Library Matrices

    println!("\n========== SINGLE-PRECURSOR PIPELINE ==========");
    let pipeline_total_start = Instant::now();

    println!("\n[Step 2] Building Library Matrices");
    let step2_start = Instant::now();
    
    let lib_cols = LibCols::default();
    let (precursors_list, ms1_data_list, ms2_data_list, precursor_info_list) = 
        build_lib_matrix(&each_lib_data, &lib_cols, 5.0, 1801.0, 20)?;
    
    println!("  - Precursors processed: {}", precursors_list.len());
    println!("  - Time elapsed: {:.5} seconds", step2_start.elapsed().as_secs_f32());
    
    // Step 3: Build Tensor Representations
    println!("\n[Step 3] Building Tensor Representations");
    let step3_start = Instant::now();
    
    let (ms1_data_tensor, ms2_data_tensor) = 
        build_precursors_matrix_step1(&ms1_data_list, &ms2_data_list, device)?;
    let ms2_data_tensor_processed = build_precursors_matrix_step2(ms2_data_tensor);
    
    println!("  - MS1 tensor shape: {:?}", ms1_data_tensor.shape());
    println!("  - MS2 tensor shape: {:?}", ms2_data_tensor_processed.shape());
    println!("  - Time elapsed: {:.5} seconds", step3_start.elapsed().as_secs_f32());
    
    // Step 4: Build Range Matrices for Extraction
    println!("\n[Step 4] Building Range Matrices for Extraction");
    let step4_start = Instant::now();
    
    let (ms1_range_list, ms2_range_list) = 
        build_range_matrix_step3(&ms1_data_tensor, &ms2_data_tensor_processed, 
                                 frag_repeat_num, "ppm", 20.0, 50.0, device)?;
    
    let (re_ms1_data_tensor, re_ms2_data_tensor, ms1_extract_width_range_list, ms2_extract_width_range_list) = 
        build_precursors_matrix_step3(&ms1_data_tensor, &ms2_data_tensor_processed, 
                                      frag_repeat_num, "ppm", 20.0, 50.0, device)?;
    
    println!("  - MS1 range shape: {:?}", ms1_range_list.shape());
    println!("  - MS2 range shape: {:?}", ms2_range_list.shape());
    println!("  - Time elapsed: {:.5} seconds", step4_start.elapsed().as_secs_f32());
    
    // Step 5: Prepare Precursor Features
    println!("\n[Step 5] Preparing Precursor Features");
    let step5_start = Instant::now();
    
    let precursor_features = prepare_precursor_features(
        &precursors_list,
        &precursor_info_list,
        &assay_rt_kept_dict,
        &assay_im_kept_dict,
    )?;
    
    println!("  - Feature matrix shape: {:?}", precursor_features.shape());
    println!("  - Time elapsed: {:.5} seconds", step5_start.elapsed().as_secs_f32());
    
    // Step 6: Build Fragment Information
    println!("\n[Step 6] Building Fragment Information");
    let step6_start = Instant::now();
    
    let frag_info = build_frag_info(&ms1_data_tensor, &ms2_data_tensor_processed, 
                                    frag_repeat_num, device);
    
    println!("  - Fragment info shape: {:?}", frag_info.shape());
    println!("  - Time elapsed: {:.5} seconds", step6_start.elapsed().as_secs_f32());
    
    // Process single precursor (index 0)
    let i = 0;
    let im = precursor_features[[i, 5]];
    let rt = precursor_features[[i, 6]];
    let precursor_mz = precursor_features[[i, 1]];
    
    println!("\nProcessing precursor: {}", precursors_list[i][0]);
    println!("  - m/z: {:.4}", precursor_mz);
    println!("  - RT: {:.2}", rt);
    println!("  - IM: {:.4}", im);
    
    // Step 7: Calculate extraction ranges
    println!("\n[Step 7] Calculating Extraction Ranges");
    let step7_start = Instant::now();
    
    let (ms1_range_min, ms1_range_max) = calculate_mz_range(&ms1_range_list, i);
    let im_tolerance = 0.05f32;
    let im_min = im - im_tolerance;
    let im_max = im + im_tolerance;
    
    println!("  - MS1 m/z range: [{:.4}, {:.4}]", ms1_range_min, ms1_range_max);
    println!("  - IM range: [{:.4}, {:.4}]", im_min, im_max);
    println!("  - Time elapsed: {:.5} seconds", step7_start.elapsed().as_secs_f32());   
    
    // Step 8: MS1 Data Extraction
    println!("\n[Step 8] MS1 Data Extraction");
    let step8_start = Instant::now();
    
    let mut precursor_result_filtered = ms1_indexed.slice_by_mz_im_range(
        ms1_range_min, ms1_range_max, im_min, im_max
    );
    
    // Convert m/z values to integers in-place
    precursor_result_filtered.mz_values.iter_mut()
        .for_each(|mz| *mz = (*mz * 1000.0).ceil());
    
    println!("  - MS1 points extracted: {}", precursor_result_filtered.mz_values.len());
    println!("  - Time elapsed: {:.5} seconds", step8_start.elapsed().as_secs_f32());
    
    // Step 9: MS2 Data Extraction
    println!("\n[Step 9] MS2 Data Extraction");
    let step9_start = Instant::now();
    
    let mut frag_result_filtered = extract_ms2_data(
        &finder,
        precursor_mz,
        &ms2_range_list,
        i,
        im_min,
        im_max,
    )?;
    
    println!("  - MS2 points extracted: {}", frag_result_filtered.mz_values.len());
    println!("  - Time elapsed: {:.5} seconds", step9_start.elapsed().as_secs_f32());
    
    // Step 10: Build Mask Matrices
    println!("\n[Step 10] Building Mask Matrices");
    let step10_start = Instant::now();
    
    let (ms1_frag_moz_matrix, ms2_frag_moz_matrix) = build_mask_matrices(
        &precursor_result_filtered,
        &frag_result_filtered,
        &ms1_extract_width_range_list,
        &ms2_extract_width_range_list,
        i,
    )?;
    
    println!("  - MS1 mask shape: {:?}", ms1_frag_moz_matrix.shape());
    println!("  - MS2 mask shape: {:?}", ms2_frag_moz_matrix.shape());
    println!("  - Time elapsed: {:.5} seconds", step10_start.elapsed().as_secs_f32());
    
    // Step 11: Extract and Align RT Values
    println!("\n[Step 11] Extracting and Aligning RT Values");
    let step11_start = Instant::now();
    
    let all_rt = extract_aligned_rt_values(
        &precursor_result_filtered,
        &frag_result_filtered,
        rt,
    );
    
    println!("  - RT values aligned: {}", all_rt.len());
    println!("  - Time elapsed: {:.5} seconds", step11_start.elapsed().as_secs_f32());
    
    // Step 12: Build Intensity Matrices
    println!("\n[Step 12] Building Intensity Matrices");
    let step12_start = Instant::now();
    
    let ms1_extract_slice = ms1_extract_width_range_list.slice(s![i, .., ..]).to_owned();
    let ms2_extract_slice = ms2_extract_width_range_list.slice(s![i, .., ..]).to_owned();
    
    let ms1_frag_rt_matrix = build_intensity_matrix_optimized(
        &precursor_result_filtered,
        &ms1_extract_slice,
        &ms1_frag_moz_matrix,
        &all_rt,
    )?;
    
    let ms2_frag_rt_matrix = build_intensity_matrix_optimized(
        &frag_result_filtered,
        &ms2_extract_slice,
        &ms2_frag_moz_matrix,
        &all_rt,
    )?;
    
    println!("  - MS1 intensity matrix shape: {:?}", ms1_frag_rt_matrix.shape());
    println!("  - MS2 intensity matrix shape: {:?}", ms2_frag_rt_matrix.shape());
    println!("  - Time elapsed: {:.5} seconds", step12_start.elapsed().as_secs_f32());
    
    // Step 13: Reshape and Combine Matrices
    println!("\n[Step 13] Reshaping and Combining Matrices");
    let step13_start = Instant::now();
    
    let rsm_matrix = reshape_and_combine_matrices(
        ms1_frag_rt_matrix,
        ms2_frag_rt_matrix,
        frag_repeat_num,
    )?;
    
    println!("  - Combined matrix shape: {:?}", rsm_matrix.shape());
    println!("  - Time elapsed: {:.5} seconds", step13_start.elapsed().as_secs_f32());
    
    // Step 14: Create Final DataFrame
    println!("\n[Step 14] Creating Final DataFrame");
    let step14_start = Instant::now();
    
    let final_df = create_final_dataframe(
        &rsm_matrix,
        &frag_info,
        &all_rt,
        i,
    )?;
    
    println!("  - DataFrame shape: {} rows × {} columns", 
             final_df.height(), final_df.width());
    println!("  - Time elapsed: {:.5} seconds", step14_start.elapsed().as_secs_f32());

    let pipeline_total_elapsed = pipeline_total_start.elapsed();
    println!("\n========== PIPELINE SUMMARY ==========");
    println!("Total single-precursor pipeline time: {:.5} seconds", 
             pipeline_total_elapsed.as_secs_f32());
    
    // Step 15: Save Results
    println!("\n[Step 15] Saving Results");
    let step15_start = Instant::now();
    
    let output_path = format!("{}_final_dataframe.csv", precursors_list[0][0]);
    let mut file = File::create(&output_path)?;
    CsvWriter::new(&mut file)
        .include_header(true)
        .finish(&mut final_df.clone())?;
    
    println!("  - Output saved to: {}", output_path);
    println!("  - Time elapsed: {:.5} seconds", step15_start.elapsed().as_secs_f32());
    
    // Print summary
    println!("\nProcessing completed successfully!");
    
    Ok(())
}