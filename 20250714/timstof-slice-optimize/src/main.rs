// 使用说明：
// cargo run /path/to/your/data.d
// 1. 运行 `cargo run --release -- --clear-cache` 清理缓存
// 2. 运行 `cargo run --release -- --cache-info` 查看缓存信息
// 3. 运行 `cargo run --release -- <d_folder>` 进行切片处理

// cargo build --profile=fast-release
// cargo run --profile=fast-release -- --cache-info
// cargo run --profile=fast-release -- ...d_folder

mod utils;
mod cache;

use cache::CacheManager;
use utils::{
    // TimsTOF reading related
    read_timstof_data, TimsTOFRawData,
    // Indexing related
    IndexedTimsTOFData, build_indexed_data,
    // Other utility functions
    find_scan_for_index, read_parquet_with_polars, library_records_to_dataframe,
    merge_library_and_report, get_unique_precursor_ids, process_library_fast, create_rt_im_dicts,
    build_lib_matrix, build_precursors_matrix_step1, build_precursors_matrix_step2, build_range_matrix_step3,
    build_precursors_matrix_step3, build_frag_info, get_rt_list, LibCols, quantize, FrameSplit, MergeFrom,
};
use rayon::prelude::*;
use std::{collections::HashMap, error::Error, path::Path, time::Instant, env, cmp::Ordering, sync::Arc, fs::File};
use ndarray::{Array1, Array2, Array3, Array4, s, Axis, concatenate};
use polars::prelude::*;

pub struct FastChunkFinder {
    low_bounds: Vec<f32>,
    high_bounds: Vec<f32>,
    chunks: Vec<IndexedTimsTOFData>,
}

impl FastChunkFinder {
    pub fn new(mut pairs: Vec<((f32, f32), IndexedTimsTOFData)>) -> Result<Self, Box<dyn Error>> {
        if pairs.is_empty() { return Err("no MS2 windows collected".into()); }
        pairs.sort_by(|a, b| a.0 .0.partial_cmp(&b.0 .0).unwrap());
        
        let n = pairs.len();
        let mut low = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        for ((l, h), _) in &pairs {
            low.push(*l);
            high.push(*h);
        }
        
        let chunks: Vec<IndexedTimsTOFData> = pairs.into_iter().map(|(_, data)| data).collect();
        Ok(Self { low_bounds: low, high_bounds: high, chunks })
    }
    
    #[inline]
    pub fn find(&self, mz: f32) -> Option<&IndexedTimsTOFData> {
        match self.low_bounds.binary_search_by(|probe| probe.partial_cmp(&mz).unwrap()) {
            Ok(idx) => Some(&self.chunks[idx]),
            Err(0) => None,
            Err(pos) => {
                let idx = pos - 1;
                if mz <= self.high_bounds[idx] { Some(&self.chunks[idx]) } else { None }
            }
        }
    }
}

// Helper function to build intensity matrix (Part 5)
fn build_intensity_matrix_optimized(
    data: &utils::TimsTOFData,
    extract_width_range: &Array2<f32>,
    frag_moz_matrix: &Array2<f32>,
    all_rt: &[f32],
) -> Result<Array2<f32>, Box<dyn Error>> {
    let n_frags = extract_width_range.shape()[0];
    let n_rt = all_rt.len();
    
    // Create pivot table: mz -> rt -> intensity
    let mut pivot: HashMap<i32, HashMap<i32, f32>> = HashMap::new();
    
    // Build pivot table
    for i in 0..data.mz_values.len() {
        let mz_key = data.mz_values[i] as i32;
        let rt_key = (data.rt_values_min[i] * 1e6) as i32;
        let intensity = data.intensity_values[i] as f32;
        
        pivot.entry(mz_key)
            .or_insert_with(HashMap::new)
            .entry(rt_key)
            .and_modify(|e| *e += intensity)
            .or_insert(intensity);
    }
    
    // Convert all_rt to keys
    let rt_keys: Vec<i32> = all_rt.iter()
        .map(|&rt| (rt * 1e6) as i32)
        .collect();
    
    // Build intensity matrix in parallel
    let results: Vec<Vec<f32>> = (0..n_frags)
        .into_par_iter()
        .map(|frag_idx| {
            let mut row = vec![0.0f32; n_rt];
            
            for mz_idx in 0..extract_width_range.shape()[1] {
                let mz_val = extract_width_range[[frag_idx, mz_idx]] as i32;
                let mask_val = frag_moz_matrix[[frag_idx, mz_idx]];
                
                if mz_val > 0 && mask_val > 0.0 {
                    if let Some(rt_map) = pivot.get(&mz_val) {
                        for (rt_idx, &rt_key) in rt_keys.iter().enumerate() {
                            if let Some(&intensity) = rt_map.get(&rt_key) {
                                row[rt_idx] += mask_val * intensity;
                            }
                        }
                    }
                }
            }
            
            row
        })
        .collect();
    
    // Convert to ndarray
    let mut frag_rt_matrix = Array2::<f32>::zeros((n_frags, n_rt));
    for (frag_idx, row) in results.into_iter().enumerate() {
        for (rt_idx, val) in row.into_iter().enumerate() {
            frag_rt_matrix[[frag_idx, rt_idx]] = val;
        }
    }
    
    Ok(frag_rt_matrix)
}

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
    println!("\n========== SINGLE-PRECURSOR PIPELINE ==========");
    let pipeline_total_start = Instant::now();
    
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
    
    // Step 9: MS2 Data Extraction (Parallel)
    println!("\n[Step 9] MS2 Data Extraction (Parallel Processing)");
    let step9_start = Instant::now();
    
    let mut frag_result_filtered = extract_ms2_data_parallel(
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
    let pipeline_total_elapsed = pipeline_total_start.elapsed();
    println!("\n========== PIPELINE SUMMARY ==========");
    println!("Total single-precursor pipeline time: {:.5} seconds", 
             pipeline_total_elapsed.as_secs_f32());
    println!("\nProcessing completed successfully!");
    
    Ok(())
}

// Helper function implementations

fn prepare_precursor_features(
    precursors_list: &[Vec<String>],
    precursor_info_list: &[Vec<f32>],
    assay_rt_kept_dict: &std::collections::HashMap<String, f32>,
    assay_im_kept_dict: &std::collections::HashMap<String, f32>,
) -> Result<Array2<f32>, Box<dyn Error>> {
    let n_precursors = precursors_list.len();
    let n_cols = 8; // 5 info columns + im + rt + delta_rt
    
    let mut precursor_feat = Array2::<f32>::zeros((n_precursors, n_cols));
    
    for i in 0..n_precursors {
        // Copy first 5 columns from precursor_info
        let info_len = precursor_info_list[i].len().min(5);
        for j in 0..info_len {
            precursor_feat[[i, j]] = precursor_info_list[i][j];
        }
        
        // Add assay IM and RT
        precursor_feat[[i, 5]] = assay_im_kept_dict
            .get(&precursors_list[i][0])
            .copied()
            .unwrap_or(0.0);
        
        precursor_feat[[i, 6]] = assay_rt_kept_dict
            .get(&precursors_list[i][0])
            .copied()
            .unwrap_or(0.0);
        
        // Delta RT is 0 for all
        precursor_feat[[i, 7]] = 0.0;
    }
    
    Ok(precursor_feat)
}

fn calculate_mz_range(ms1_range_list: &Array3<f32>, i: usize) -> (f32, f32) {
    let ms1_range_slice = ms1_range_list.slice(s![i, .., ..]);
    
    let ms1_range_min_val = ms1_range_slice
        .iter()
        .filter(|&&v| v > 0.0)
        .fold(f32::INFINITY, |a, &b| a.min(b));
    
    let ms1_range_max_val = ms1_range_slice
        .iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    let ms1_range_min = (ms1_range_min_val - 1.0) / 1000.0;
    let ms1_range_max = (ms1_range_max_val + 1.0) / 1000.0;
    
    (ms1_range_min, ms1_range_max)
}

fn extract_ms2_data_parallel(
    finder: &FastChunkFinder,
    precursor_mz: f32,
    ms2_range_list: &Array3<f32>,
    i: usize,
    im_min: f32,
    im_max: f32,
) -> Result<utils::TimsTOFData, Box<dyn Error>> {
    let mut result = if let Some(ms2_indexed) = finder.find(precursor_mz) {
        // Process all 66 MS2 ranges in parallel
        let frag_results: Vec<utils::TimsTOFData> = (0..66)
            .into_par_iter()
            .map(|j| {
                let ms2_range_min_val = ms2_range_list[[i, j, 0]];
                let ms2_range_max_val = ms2_range_list[[i, j, 1]];
                
                let ms2_range_min = (ms2_range_min_val - 1.0) / 1000.0;
                let ms2_range_max = (ms2_range_max_val + 1.0) / 1000.0;
                
                if ms2_range_min <= 0.0 || ms2_range_max <= 0.0 || ms2_range_min >= ms2_range_max {
                    utils::TimsTOFData::new()
                } else {
                    ms2_indexed.slice_by_mz_im_range(
                        ms2_range_min, ms2_range_max, im_min, im_max
                    )
                }
            })
            .collect();
        
        utils::TimsTOFData::merge(frag_results)
    } else {
        println!("  Warning: No MS2 data found for precursor m/z {:.4}", precursor_mz);
        utils::TimsTOFData::new()
    };
    
    // Convert m/z values to integers
    result.mz_values.iter_mut()
        .for_each(|mz| *mz = (*mz * 1000.0).ceil());
    
    Ok(result)
}

fn build_mask_matrices(
    precursor_result_filtered: &utils::TimsTOFData,
    frag_result_filtered: &utils::TimsTOFData,
    ms1_extract_width_range_list: &Array3<f32>,
    ms2_extract_width_range_list: &Array3<f32>,
    i: usize,
) -> Result<(Array2<f32>, Array2<f32>), Box<dyn Error>> {
    use std::collections::HashSet;
    
    // Create hash sets for fast lookup
    let search_ms1_set: HashSet<i32> = precursor_result_filtered.mz_values
        .iter()
        .map(|&mz| mz as i32)
        .collect();
    
    let search_ms2_set: HashSet<i32> = frag_result_filtered.mz_values
        .iter()
        .map(|&mz| mz as i32)
        .collect();
    
    // Extract slices
    let ms1_extract_slice = ms1_extract_width_range_list.slice(s![i, .., ..]);
    let ms2_extract_slice = ms2_extract_width_range_list.slice(s![i, .., ..]);
    
    // Build MS1 mask matrix
    let (n_frags_ms1, n_mz_ms1) = (ms1_extract_slice.shape()[0], ms1_extract_slice.shape()[1]);
    let mut ms1_frag_moz_matrix = Array2::<f32>::zeros((n_frags_ms1, n_mz_ms1));
    
    for j in 0..n_frags_ms1 {
        for k in 0..n_mz_ms1 {
            let val = ms1_extract_slice[[j, k]] as i32;
            if val > 0 && search_ms1_set.contains(&val) {
                ms1_frag_moz_matrix[[j, k]] = 1.0;
            }
        }
    }
    
    // Build MS2 mask matrix
    let (n_frags_ms2, n_mz_ms2) = (ms2_extract_slice.shape()[0], ms2_extract_slice.shape()[1]);
    let mut ms2_frag_moz_matrix = Array2::<f32>::zeros((n_frags_ms2, n_mz_ms2));
    
    for j in 0..n_frags_ms2 {
        for k in 0..n_mz_ms2 {
            let val = ms2_extract_slice[[j, k]] as i32;
            if val > 0 && search_ms2_set.contains(&val) {
                ms2_frag_moz_matrix[[j, k]] = 1.0;
            }
        }
    }
    
    Ok((ms1_frag_moz_matrix, ms2_frag_moz_matrix))
}

fn extract_aligned_rt_values(
    precursor_result_filtered: &utils::TimsTOFData,
    frag_result_filtered: &utils::TimsTOFData,
    target_rt: f32,
) -> Vec<f32> {
    use std::collections::HashSet;
    
    let mut all_rt_set = HashSet::new();
    
    // Collect all unique RT values
    for &rt_val in &precursor_result_filtered.rt_values_min {
        all_rt_set.insert((rt_val * 1e6) as i32);
    }
    
    for &rt_val in &frag_result_filtered.rt_values_min {
        all_rt_set.insert((rt_val * 1e6) as i32);
    }
    
    // Convert to sorted vector
    let mut all_rt_vec: Vec<f32> = all_rt_set
        .iter()
        .map(|&rt_int| rt_int as f32 / 1e6)
        .collect();
    
    all_rt_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    // Get RT list with target RT in the center
    get_rt_list(all_rt_vec, target_rt)
}

fn reshape_and_combine_matrices(
    ms1_frag_rt_matrix: Array2<f32>,
    ms2_frag_rt_matrix: Array2<f32>,
    frag_repeat_num: usize,
) -> Result<Array4<f32>, Box<dyn Error>> {
    // Reshape MS1 matrix
    let (ms1_rows, ms1_cols) = ms1_frag_rt_matrix.dim();
    let ms1_reshaped = ms1_frag_rt_matrix.into_shape((
        frag_repeat_num,
        ms1_rows / frag_repeat_num,
        ms1_cols
    ))?;
    
    // Reshape MS2 matrix
    let (ms2_rows, ms2_cols) = ms2_frag_rt_matrix.dim();
    let ms2_reshaped = ms2_frag_rt_matrix.into_shape((
        frag_repeat_num,
        ms2_rows / frag_repeat_num,
        ms2_cols
    ))?;
    
    // Combine matrices
    let ms1_frags = ms1_reshaped.shape()[1];
    let ms2_frags = ms2_reshaped.shape()[1];
    let total_frags = ms1_frags + ms2_frags;
    let n_rt = ms1_reshaped.shape()[2];
    
    let mut full_frag_rt_matrix = Array3::<f32>::zeros((frag_repeat_num, total_frags, n_rt));
    
    // Copy MS1 data
    full_frag_rt_matrix.slice_mut(s![.., ..ms1_frags, ..])
        .assign(&ms1_reshaped);
    
    // Copy MS2 data
    full_frag_rt_matrix.slice_mut(s![.., ms1_frags.., ..])
        .assign(&ms2_reshaped);
    
    // Add batch dimension
    Ok(full_frag_rt_matrix.insert_axis(Axis(0)))
}

fn create_final_dataframe(
    rsm_matrix: &Array4<f32>,
    frag_info: &Array3<f32>,
    all_rt: &[f32],
    i: usize,
) -> Result<DataFrame, Box<dyn Error>> {
    // Aggregate across repeat dimension
    let aggregated_x_sum = rsm_matrix.sum_axis(Axis(1));
    
    // Extract data for this precursor
    let precursor_data = aggregated_x_sum.slice(s![0, .., ..]);
    let precursor_frag_info = frag_info.slice(s![i, .., ..]);
    
    let total_frags = precursor_data.shape()[0];
    let mut columns = Vec::new();
    
    // Add RT columns
    for (rt_idx, &rt_val) in all_rt.iter().enumerate() {
        let col_data: Vec<f32> = (0..total_frags)
            .map(|frag_idx| precursor_data[[frag_idx, rt_idx]])
            .collect();
        columns.push(Series::new(&format!("{:.6}", rt_val), col_data));
    }
    
    // Add fragment info columns
    let info_names = ["ProductMz", "LibraryIntensity", "frag_type", "FragmentType"];
    for col_idx in 0..4.min(precursor_frag_info.shape()[1]) {
        let col_data: Vec<f32> = (0..total_frags)
            .map(|row_idx| precursor_frag_info[[row_idx, col_idx]])
            .collect();
        columns.push(Series::new(info_names[col_idx], col_data));
    }
    
    Ok(DataFrame::new(columns)?)
}