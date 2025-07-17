mod utils;
mod cache;
mod processing;

use cache::CacheManager;
use utils::{
    read_timstof_data, build_indexed_data, read_parquet_with_polars,
    library_records_to_dataframe, merge_library_and_report, get_unique_precursor_ids, 
    process_library_fast, create_rt_im_dicts, build_lib_matrix, build_precursors_matrix_step1, 
    build_precursors_matrix_step2, build_range_matrix_step3, build_precursors_matrix_step3, 
    build_frag_info, LibCols, IndexedTimsTOFData,
};
use processing::{
    FastChunkFinder, build_intensity_matrix_optimized, prepare_precursor_features,
    calculate_mz_range, extract_ms2_data, build_mask_matrices, extract_aligned_rt_values,
    reshape_and_combine_matrices, create_final_dataframe,
};

use std::{error::Error, path::Path, time::Instant, env, fs::File, sync::Arc};
use ndarray::s;
use polars::prelude::*;

#[derive(Clone)]
struct PrecursorResult {
    id: String,
    processing_time: f32,
    dataframe: Option<DataFrame>,
    rt: Option<f32>,
    im: Option<f32>,
}

fn process_single_precursor(
    precursor_id: &str,
    library_records: &[utils::LibraryRecord],
    ms1_indexed: &Arc<IndexedTimsTOFData>,
    finder: &Arc<FastChunkFinder>,
    assay_rt_kept_dict: &Arc<std::collections::HashMap<String, f32>>,
    assay_im_kept_dict: &Arc<std::collections::HashMap<String, f32>>,
    frag_repeat_num: usize,
) -> Result<PrecursorResult, Box<dyn Error + Send + Sync>> {

    let start_time = Instant::now();
    
    // Filter library data for this precursor
    let each_lib_data: Vec<_> = library_records
        .iter()
        .filter(|record| record.transition_group_id == precursor_id)
        .cloned()
        .collect();
    
    if each_lib_data.is_empty() {
        return Ok(PrecursorResult {
            id: precursor_id.to_string(),
            processing_time: 0.0,
            dataframe: None,
            rt: None,
            im: None,
        });
    }

    // let start_time = Instant::now();
    
    // Build library matrices
    let lib_cols = LibCols::default();
    let (precursors_list, ms1_data_list, ms2_data_list, precursor_info_list) = 
        build_lib_matrix(&each_lib_data, &lib_cols, 5.0, 1801.0, 20)
            .map_err(|e| -> Box<dyn Error + Send + Sync> { format!("build_lib_matrix error: {}", e).into() })?;
    
    // Build tensor representations
    let device = "cpu";
    let (ms1_data_tensor, ms2_data_tensor) = 
        build_precursors_matrix_step1(&ms1_data_list, &ms2_data_list, device)
            .map_err(|e| -> Box<dyn Error + Send + Sync> { format!("build_precursors_matrix_step1 error: {}", e).into() })?;
    let ms2_data_tensor_processed = build_precursors_matrix_step2(ms2_data_tensor);
    
    // Build range matrices
    let (ms1_range_list, ms2_range_list) = 
        build_range_matrix_step3(&ms1_data_tensor, &ms2_data_tensor_processed, 
                                 frag_repeat_num, "ppm", 20.0, 50.0, device)
            .map_err(|e| -> Box<dyn Error + Send + Sync> { format!("build_range_matrix_step3 error: {}", e).into() })?;
    
    let (_re_ms1_data_tensor, _re_ms2_data_tensor, ms1_extract_width_range_list, ms2_extract_width_range_list) = 
        build_precursors_matrix_step3(&ms1_data_tensor, &ms2_data_tensor_processed, 
                                      frag_repeat_num, "ppm", 20.0, 50.0, device)
            .map_err(|e| -> Box<dyn Error + Send + Sync> { format!("build_precursors_matrix_step3 error: {}", e).into() })?;
    
    // Prepare precursor features
    let precursor_features = prepare_precursor_features(
        &precursors_list,
        &precursor_info_list,
        &assay_rt_kept_dict,
        &assay_im_kept_dict,
    ).map_err(|e| -> Box<dyn Error + Send + Sync> { format!("prepare_precursor_features error: {}", e).into() })?;
    
    // Build fragment information
    let frag_info = build_frag_info(&ms1_data_tensor, &ms2_data_tensor_processed, 
                                    frag_repeat_num, device);
    
    // Process precursor (index 0)
    let i = 0;
    let im = precursor_features[[i, 5]];
    let rt = precursor_features[[i, 6]];
    let precursor_mz = precursor_features[[i, 1]];
    // println!("precursor_mz: {}", precursor_mz);
    // println!("im: {}", im);
    // println!("rt: {}", rt);
    
    // Calculate extraction ranges
    let (ms1_range_min, ms1_range_max) = calculate_mz_range(&ms1_range_list, i);
    let im_tolerance = 0.05f32;
    let im_min = im - im_tolerance;
    let im_max = im + im_tolerance;
    
    // MS1 data extraction
    let mut precursor_result_filtered = ms1_indexed.slice_by_mz_im_range(
        ms1_range_min, ms1_range_max, im_min, im_max
    );
    
    // Convert m/z values to integers in-place
    precursor_result_filtered.mz_values.iter_mut()
        .for_each(|mz| *mz = (*mz * 1000.0).ceil());
    
    // MS2 data extraction
    let frag_result_filtered = extract_ms2_data(
        &finder,
        precursor_mz,
        &ms2_range_list,
        i,
        im_min,
        im_max,
    ).map_err(|e| -> Box<dyn Error + Send + Sync> { format!("extract_ms2_data error: {}", e).into() })?;
    
    // Build mask matrices
    let (ms1_frag_moz_matrix, ms2_frag_moz_matrix) = build_mask_matrices(
        &precursor_result_filtered,
        &frag_result_filtered,
        &ms1_extract_width_range_list,
        &ms2_extract_width_range_list,
        i,
    ).map_err(|e| -> Box<dyn Error + Send + Sync> { format!("build_mask_matrices error: {}", e).into() })?;
    
    // Extract and align RT values
    let all_rt = extract_aligned_rt_values(
        &precursor_result_filtered,
        &frag_result_filtered,
        rt,
    );
    
    // Build intensity matrices
    let ms1_extract_slice = ms1_extract_width_range_list.slice(s![i, .., ..]).to_owned();
    let ms2_extract_slice = ms2_extract_width_range_list.slice(s![i, .., ..]).to_owned();
    
    let ms1_frag_rt_matrix = build_intensity_matrix_optimized(
        &precursor_result_filtered,
        &ms1_extract_slice,
        &ms1_frag_moz_matrix,
        &all_rt,
    ).map_err(|e| -> Box<dyn Error + Send + Sync> { format!("build_intensity_matrix_optimized MS1 error: {}", e).into() })?;
    
    let ms2_frag_rt_matrix = build_intensity_matrix_optimized(
        &frag_result_filtered,
        &ms2_extract_slice,
        &ms2_frag_moz_matrix,
        &all_rt,
    ).map_err(|e| -> Box<dyn Error + Send + Sync> { format!("build_intensity_matrix_optimized MS2 error: {}", e).into() })?;
    
    // Reshape and combine matrices
    let rsm_matrix = reshape_and_combine_matrices(
        ms1_frag_rt_matrix,
        ms2_frag_rt_matrix,
        frag_repeat_num,
    ).map_err(|e| -> Box<dyn Error + Send + Sync> { format!("reshape_and_combine_matrices error: {}", e).into() })?;
    
    // Create final DataFrame
    let final_df = create_final_dataframe(
        &rsm_matrix,
        &frag_info,
        &all_rt,
        i,
    ).map_err(|e| -> Box<dyn Error + Send + Sync> { format!("create_final_dataframe error: {}", e).into() })?;
    
    Ok(PrecursorResult {
        id: precursor_id.to_string(),
        processing_time: start_time.elapsed().as_secs_f32(),
        dataframe: Some(final_df),
        rt: Some(rt),
        im: Some(im),
    })
}

fn main() -> Result<(), Box<dyn Error>> {
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
    
    // Wrap shared data in Arc for consistency with the existing function signature
    let ms1_indexed = Arc::new(ms1_indexed);
    let finder = Arc::new(FastChunkFinder::new(ms2_indexed_pairs)?);
    
    // ================================ LIBRARY AND REPORT LOADING ================================
    println!("\n========== LIBRARY AND REPORT PROCESSING ==========");
    let lib_processing_start = Instant::now();
    
    let lib_file_path = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/lib/TPHPlib_frag1025_swissprot_final_all_from_Yueliang.tsv";
    let library_records = process_library_fast(lib_file_path)?;
    let library_records = Arc::new(library_records);
    
    let library_df = library_records_to_dataframe((*library_records).clone())?;
    
    let report_file_path = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/report/report.parquet";
    let report_df = read_parquet_with_polars(report_file_path)?;
    
    // Extract precursor IDs before moving report_df
    let precursor_id_list: Vec<String> = report_df
        .column("transition_group_id")?
        .str()?
        .into_no_null_iter()
        .take(100)
        .map(|s| s.to_string())
        .collect();
    
    let diann_result = merge_library_and_report(library_df, report_df)?;
    let diann_precursor_id_all = get_unique_precursor_ids(&diann_result)?;
    let (assay_rt_kept_dict, assay_im_kept_dict) = create_rt_im_dicts(&diann_precursor_id_all)?;
    
    let assay_rt_kept_dict = Arc::new(assay_rt_kept_dict);
    let assay_im_kept_dict = Arc::new(assay_im_kept_dict);
    
    println!("Library and report processing time: {:.5} seconds", lib_processing_start.elapsed().as_secs_f32());
    
    // Set processing parameters
    let frag_repeat_num: usize = 5;

    
    // ================================ SEQUENTIAL PRECURSOR PROCESSING ================================
    println!("\n========== SEQUENTIAL PRECURSOR PROCESSING ==========");
    println!("Processing {} precursors sequentially", precursor_id_list.len());
    
    let sequential_start = Instant::now();
    let mut results = Vec::with_capacity(precursor_id_list.len());
    
    // Process precursors sequentially with progress reporting
    for (index, precursor_id) in precursor_id_list.iter().enumerate() {
        let progress = index + 1;
        let total = precursor_id_list.len();
        let percentage = (progress as f32 / total as f32) * 100.0;
        
        println!("\n[{:2}/{:2}] ({:5.1}%) Processing: {}", 
                 progress, total, percentage, precursor_id);
        
        let precursor_start = Instant::now();
        
        let result = process_single_precursor(
            precursor_id,
            &library_records,
            &ms1_indexed,
            &finder,
            &assay_rt_kept_dict,
            &assay_im_kept_dict,
            frag_repeat_num,
        ).unwrap_or_else(|e| {
            eprintln!("         ERROR: {}", e);
            PrecursorResult {
                id: precursor_id.to_string(),
                processing_time: precursor_start.elapsed().as_secs_f32(),
                dataframe: None,
                rt: None,
                im: None,
            }
        });
        
        let status = if result.dataframe.is_some() { "✓ SUCCESS" } else { "✗ FAILED" };
        println!("         {} - {:.3}s", status, result.processing_time);
        
        results.push(result);
    }
    
    let total_sequential_time = sequential_start.elapsed().as_secs_f32();
    
    // Save results and print statistics
    println!("\n========== RESULTS ==========");
    
    let mut successful_count = 0;
    let mut total_individual_time = 0.0;
    
    // Save results with RT and IM values in filename
    for result in &results {
        if let Some(df) = &result.dataframe {
            successful_count += 1;
            
            // Create filename with RT and IM values
            let filename = if let (Some(rt), Some(im)) = (result.rt, result.im) {
                format!("{}_rt{:.2}_im{:.3}_final_dataframe.csv", result.id, rt, im)
            } else {
                format!("{}_final_dataframe.csv", result.id)
            };
            
            // Save individual results
            let mut file = File::create(&filename)?;
            CsvWriter::new(&mut file)
                .include_header(true)
                .finish(&mut df.clone())?;
            
            println!("Saved: {}", filename);
        }
        
        total_individual_time += result.processing_time;
    }
    
    // Print performance statistics
    println!("\n========== PERFORMANCE SUMMARY ==========");
    println!("Total precursors processed: {}", precursor_id_list.len());
    println!("Successfully processed: {}", successful_count);
    println!("Failed: {}", precursor_id_list.len() - successful_count);
    println!("Success rate: {:.1}%", (successful_count as f32 / precursor_id_list.len() as f32) * 100.0);
    println!("\nTiming Statistics:");
    println!("  - Total sequential processing time: {:.3} seconds", total_sequential_time);
    println!("  - Average time per precursor: {:.3} seconds", total_individual_time / precursor_id_list.len() as f32);
    println!("  - Total time for all precursors: {:.3} seconds", total_individual_time);
    
    // Show detailed results
    println!("\n========== DETAILED RESULTS ==========");
    for (i, result) in results.iter().enumerate() {
        let status = if result.dataframe.is_some() { "SUCCESS" } else { "FAILED" };
        println!("[{:2}] {:<30} | {:.3}s | {}", 
                 i + 1, 
                 result.id, 
                 result.processing_time, 
                 status);
    }
    
    Ok(())
}