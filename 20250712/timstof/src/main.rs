// Updated src/main.rs with timing only
//-------------------------------------------------------------
//  src/main.rs   --  with Parallel Indexed Slicing and Parts 5+6 (timing only)
//-------------------------------------------------------------
mod utils;
mod output;

use utils::{
    TimsTOFData, find_scan_for_index,
    read_parquet_with_polars, library_records_to_dataframe,
    merge_library_and_report, get_unique_precursor_ids,
    process_library_fast, create_rt_im_dicts,
    build_lib_matrix, build_precursors_matrix_step1,
    build_precursors_matrix_step2, build_range_matrix_step3,
    build_precursors_matrix_step3,
    build_frag_info, LibCols, get_rt_list,
};

use rayon::prelude::*;
use std::{
    collections::HashMap,
    error::Error,
    path::Path,
    time::Instant,
    env,
    cmp::Ordering,
    sync::Arc,
};
use ndarray::{Array1, Array2, Array3, Array4, s, Axis, concatenate};
use polars::prelude::*;

use timsrust::{
    converters::ConvertableDomain,
    readers::{FrameReader, MetadataReader},
    MSLevel,
};

//-------------------------------------------------------------
//  IndexedTimsTOFData - Optimized with parallel sorting
//-------------------------------------------------------------
#[derive(Clone)]
pub struct IndexedTimsTOFData {
    // Original data
    pub rt_values_min: Vec<f64>,
    pub mobility_values: Vec<f64>,
    pub mz_values: Vec<f64>,
    pub intensity_values: Vec<u32>,
    pub frame_indices: Vec<usize>,
    pub scan_indices: Vec<usize>,
    
    // Index sorted by m/z
    mz_sorted_indices: Vec<usize>,
}

impl IndexedTimsTOFData {
    pub fn new() -> Self {
        Self {
            rt_values_min: Vec::new(),
            mobility_values: Vec::new(),
            mz_values: Vec::new(),
            intensity_values: Vec::new(),
            frame_indices: Vec::new(),
            scan_indices: Vec::new(),
            mz_sorted_indices: Vec::new(),
        }
    }
    
    pub fn from_timstof_data(data: TimsTOFData) -> Self {
        let n = data.mz_values.len();
        
        // Create index-value pairs for parallel sorting
        let mut index_pairs: Vec<(usize, f64)> = (0..n)
            .into_par_iter()
            .map(|i| (i, data.mz_values[i]))
            .collect();
        
        // Use parallel sort from rayon
        index_pairs.par_sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
        });
        
        // Extract just the indices
        let indices: Vec<usize> = index_pairs.into_par_iter()
            .map(|(idx, _)| idx)
            .collect();
        
        Self {
            rt_values_min: data.rt_values_min,
            mobility_values: data.mobility_values,
            mz_values: data.mz_values,
            intensity_values: data.intensity_values,
            frame_indices: data.frame_indices,
            scan_indices: data.scan_indices,
            mz_sorted_indices: indices,
        }
    }
    
    pub fn len(&self) -> usize {
        self.mz_values.len()
    }
    
    pub fn is_empty(&self) -> bool {
        self.mz_values.is_empty()
    }
    
    // Binary search for m/z value, returns index in sorted order
    fn binary_search_mz(&self, target_mz: f64) -> usize {
        let mut left = 0;
        let mut right = self.mz_sorted_indices.len();
        
        while left < right {
            let mid = left + (right - left) / 2;
            let idx = self.mz_sorted_indices[mid];
            
            if self.mz_values[idx] < target_mz {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        
        left
    }
    
    // Python-style .loc[mz_min:mz_max] slicing
    pub fn slice_by_mz_range(&self, mz_min: f64, mz_max: f64) -> TimsTOFData {
        let start_pos = self.binary_search_mz(mz_min);
        let end_pos = self.binary_search_mz(mz_max + 1e-9);
        
        // Pre-allocate with estimated capacity
        let estimated_size = end_pos.saturating_sub(start_pos);
        let mut result = TimsTOFData::with_capacity(estimated_size);
        
        for pos in start_pos..end_pos {
            if pos >= self.mz_sorted_indices.len() {
                break;
            }
            
            let idx = self.mz_sorted_indices[pos];
            let mz = self.mz_values[idx];
            
            if mz > mz_max {
                break;
            }
            
            result.rt_values_min.push(self.rt_values_min[idx]);
            result.mobility_values.push(self.mobility_values[idx]);
            result.mz_values.push(mz);
            result.intensity_values.push(self.intensity_values[idx]);
            result.frame_indices.push(self.frame_indices[idx]);
            result.scan_indices.push(self.scan_indices[idx]);
        }
        
        result
    }
    
    // Convert m/z values to integers and update index
    pub fn convert_mz_to_integer(&mut self) {
        // Parallel conversion
        self.mz_values.par_iter_mut().for_each(|mz| {
            *mz = (*mz * 1000.0).ceil();
        });
        
        // Parallel re-sort
        let mut index_pairs: Vec<(usize, f64)> = self.mz_sorted_indices
            .par_iter()
            .map(|&idx| (idx, self.mz_values[idx]))
            .collect();
        
        index_pairs.par_sort_unstable_by(|a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal)
        });
        
        self.mz_sorted_indices = index_pairs.into_par_iter()
            .map(|(idx, _)| idx)
            .collect();
    }
    
    // Filter by IM range (parallelized)
    pub fn filter_by_im_range(&self, im_min: f64, im_max: f64) -> TimsTOFData {
        // Parallel filter to get valid indices
        let valid_indices: Vec<usize> = (0..self.mobility_values.len())
            .into_par_iter()
            .filter(|&i| {
                let im = self.mobility_values[i];
                im >= im_min && im <= im_max
            })
            .collect();
        
        let mut result = TimsTOFData::with_capacity(valid_indices.len());
        
        for i in valid_indices {
            result.rt_values_min.push(self.rt_values_min[i]);
            result.mobility_values.push(self.mobility_values[i]);
            result.mz_values.push(self.mz_values[i]);
            result.intensity_values.push(self.intensity_values[i]);
            result.frame_indices.push(self.frame_indices[i]);
            result.scan_indices.push(self.scan_indices[i]);
        }
        
        result
    }
}

//-------------------------------------------------------------
//  FastChunkFinder with parallel indexed data creation
//-------------------------------------------------------------
pub struct FastChunkFinder {
    low_bounds: Vec<f64>,
    high_bounds: Vec<f64>,
    chunks: Vec<IndexedTimsTOFData>,
}

impl FastChunkFinder {
    pub fn new(mut pairs: Vec<((f64, f64), TimsTOFData)>) -> Result<Self, Box<dyn Error>> {
        if pairs.is_empty() {
            return Err("no MS2 windows collected".into());
        }
        
        pairs.sort_by(|a, b| a.0 .0.partial_cmp(&b.0 .0).unwrap());
        
        let n = pairs.len();
        let mut low = Vec::with_capacity(n);
        let mut high = Vec::with_capacity(n);
        
        // Extract bounds
        for ((l, h), _) in &pairs {
            low.push(*l);
            high.push(*h);
        }
        
        // Parallel conversion to indexed data
        let chunks: Vec<IndexedTimsTOFData> = pairs
            .into_par_iter()
            .map(|(_, data)| IndexedTimsTOFData::from_timstof_data(data))
            .collect();
        
        Ok(Self { low_bounds: low, high_bounds: high, chunks })
    }
    
    #[inline]
    pub fn find(&self, mz: f64) -> Option<&IndexedTimsTOFData> {
        match self.low_bounds.binary_search_by(|probe| probe.partial_cmp(&mz).unwrap()) {
            Ok(idx) => Some(&self.chunks[idx]),
            Err(0) => None,
            Err(pos) => {
                let idx = pos - 1;
                if mz <= self.high_bounds[idx] {
                    Some(&self.chunks[idx])
                } else {
                    None
                }
            }
        }
    }
    
    pub fn range_count(&self) -> usize {
        self.low_bounds.len()
    }
}

//-------------------------------------------------------------
//  helpers
//-------------------------------------------------------------
#[inline]
fn quantize(x: f64) -> u64 { (x * 10_000.0).round() as u64 }

struct FrameSplit {
    ms1: TimsTOFData,
    ms2: Vec<((u64, u64), TimsTOFData)>,
}

// Add with_capacity method to TimsTOFData
impl TimsTOFData {
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            rt_values_min: Vec::with_capacity(capacity),
            mobility_values: Vec::with_capacity(capacity),
            mz_values: Vec::with_capacity(capacity),
            intensity_values: Vec::with_capacity(capacity),
            frame_indices: Vec::with_capacity(capacity),
            scan_indices: Vec::with_capacity(capacity),
        }
    }
}

/// Optimized Bruker reader with better memory allocation
fn read_timstof_grouped(
    d_folder: &Path,
) -> Result<(IndexedTimsTOFData, Vec<((f64, f64), TimsTOFData)>), Box<dyn Error>> {
    let tdf_path = d_folder.join("analysis.tdf");
    let meta = MetadataReader::new(&tdf_path)?;
    let mz_cv = Arc::new(meta.mz_converter);
    let im_cv = Arc::new(meta.im_converter);
    
    let frames = FrameReader::new(d_folder)?;
    let n_frames = frames.len();
    
    // Process frames in parallel with shared converters
    let splits: Vec<FrameSplit> = (0..n_frames)
        .into_par_iter()
        .map(|idx| {
            let frame = frames.get(idx).expect("frame read");
            let rt_min = frame.rt_in_seconds / 60.0;
            let mut ms1 = TimsTOFData::new();
            let mut ms2_pairs: Vec<((u64,u64), TimsTOFData)> = Vec::new();
            
            match frame.ms_level {
                MSLevel::MS1 => {
                    let n_peaks = frame.tof_indices.len();
                    ms1 = TimsTOFData::with_capacity(n_peaks);
                    
                    for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                        .zip(frame.intensities.iter()).enumerate() 
                    {
                        let mz = mz_cv.convert(tof as f64);
                        let scan = find_scan_for_index(p_idx, &frame.scan_offsets);
                        let im = im_cv.convert(scan as f64);
                        ms1.rt_values_min.push(rt_min);
                        ms1.mobility_values.push(im);
                        ms1.mz_values.push(mz);
                        ms1.intensity_values.push(intensity);
                        ms1.frame_indices.push(frame.index);
                        ms1.scan_indices.push(scan);
                    }
                }
                MSLevel::MS2 => {
                    let qs = &frame.quadrupole_settings;
                    ms2_pairs.reserve(qs.isolation_mz.len());
                    
                    for win in 0..qs.isolation_mz.len() {
                        if win >= qs.isolation_width.len() { break; }
                        let prec_mz = qs.isolation_mz[win];
                        let width = qs.isolation_width[win];
                        let low = prec_mz - width * 0.5;
                        let high = prec_mz + width * 0.5;
                        let key = (quantize(low), quantize(high));
                        
                        let mut td = TimsTOFData::new();
                        for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                            .zip(frame.intensities.iter()).enumerate() 
                        {
                            let scan = find_scan_for_index(p_idx, &frame.scan_offsets);
                            if scan < qs.scan_starts[win] || scan > qs.scan_ends[win] { continue; }
                            let mz = mz_cv.convert(tof as f64);
                            let im = im_cv.convert(scan as f64);
                            td.rt_values_min.push(rt_min);
                            td.mobility_values.push(im);
                            td.mz_values.push(mz);
                            td.intensity_values.push(intensity);
                            td.frame_indices.push(frame.index);
                            td.scan_indices.push(scan);
                        }
                        ms2_pairs.push((key, td));
                    }
                }
                _ => {}
            }
            
            FrameSplit { ms1, ms2: ms2_pairs }
        })
        .collect();
    
    // Estimate MS1 size for pre-allocation
    let ms1_size_estimate: usize = splits.par_iter()
        .map(|s| s.ms1.mz_values.len())
        .sum();
    
    let mut global_ms1 = TimsTOFData::with_capacity(ms1_size_estimate);
    let mut ms2_hash: HashMap<(u64,u64), TimsTOFData> = HashMap::new();
    
    for split in splits {
        // MS1
        global_ms1.rt_values_min.extend(split.ms1.rt_values_min);
        global_ms1.mobility_values.extend(split.ms1.mobility_values);
        global_ms1.mz_values.extend(split.ms1.mz_values);
        global_ms1.intensity_values.extend(split.ms1.intensity_values);
        global_ms1.frame_indices.extend(split.ms1.frame_indices);
        global_ms1.scan_indices.extend(split.ms1.scan_indices);
        
        // MS2
        for (key, mut td) in split.ms2 {
            ms2_hash.entry(key).or_insert_with(TimsTOFData::new)
                    .merge_from(&mut td);
        }
    }
    
    // Create indexed MS1 data with parallel sorting
    let indexed_ms1 = IndexedTimsTOFData::from_timstof_data(global_ms1);
    
    let mut ms2_vec = Vec::with_capacity(ms2_hash.len());
    for ((q_low, q_high), td) in ms2_hash {
        let low = q_low as f64 / 10_000.0;
        let high = q_high as f64 / 10_000.0;
        ms2_vec.push(((low, high), td));
    }
    
    Ok((indexed_ms1, ms2_vec))
}

// helper to merge two TimsTOFData quickly
trait MergeFrom { fn merge_from(&mut self, other: &mut Self); }
impl MergeFrom for TimsTOFData {
    fn merge_from(&mut self, other: &mut Self) {
        self.rt_values_min.append(&mut other.rt_values_min);
        self.mobility_values.append(&mut other.mobility_values);
        self.mz_values.append(&mut other.mz_values);
        self.intensity_values.append(&mut other.intensity_values);
        self.frame_indices.append(&mut other.frame_indices);
        self.scan_indices.append(&mut other.scan_indices);
    }
}

// Helper function to build intensity matrix (Part 5)
fn build_intensity_matrix_optimized(
    data: &TimsTOFData,
    extract_width_range: &Array2<f32>,
    frag_moz_matrix: &Array2<f32>,
    all_rt: &[f64],
) -> Result<Array2<f32>, Box<dyn Error>> {
    let n_frags = extract_width_range.shape()[0];
    let n_rt = all_rt.len();
    
    // Create pivot table: mz -> rt -> intensity
    let mut pivot: HashMap<i64, HashMap<i64, f32>> = HashMap::new();
    
    // Build pivot table
    for i in 0..data.mz_values.len() {
        let mz_key = data.mz_values[i] as i64;
        let rt_key = (data.rt_values_min[i] * 1e6) as i64;
        let intensity = data.intensity_values[i] as f32;
        
        pivot.entry(mz_key)
            .or_insert_with(HashMap::new)
            .entry(rt_key)
            .and_modify(|e| *e += intensity)
            .or_insert(intensity);
    }
    
    // Convert all_rt to keys
    let rt_keys: Vec<i64> = all_rt.iter()
        .map(|&rt| (rt * 1e6) as i64)
        .collect();
    
    // Build intensity matrix in parallel
    let results: Vec<Vec<f32>> = (0..n_frags)
        .into_par_iter()
        .map(|frag_idx| {
            let mut row = vec![0.0f32; n_rt];
            
            for mz_idx in 0..extract_width_range.shape()[1] {
                let mz_val = extract_width_range[[frag_idx, mz_idx]] as i64;
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

//-------------------------------------------------------------
//  main
//-------------------------------------------------------------
fn main() -> Result<(), Box<dyn Error>> {
    // Set rayon thread pool to use all available cores
    rayon::ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build_global()
        .unwrap();
    
    let args: Vec<String> = env::args().collect();
    let d_folder = if args.len() > 1 {
        args[1].clone()
    } else {
        "/Users/augustsirius/Desktop/DIABERT_test_code_lib/DIA_sample.d".to_string()
    };
    
    let d_path = Path::new(&d_folder);
    if !d_path.exists() {
        return Err(format!("folder {:?} not found", d_path).into());
    }
    
    // PART 1: TimsTOF Data Processing
    let t0 = std::time::Instant::now();
    let (ms1_indexed, ms2_pairs) = read_timstof_grouped(d_path)?;
    let finder = FastChunkFinder::new(ms2_pairs)?;
    
    // PART 2: Library and Report Processing
    let lib_file_path = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/lib/TPHPlib_frag1025_swissprot_final_all_from_Yueliang.tsv";
    let library_records = process_library_fast(lib_file_path)?;
    
    let library_df = library_records_to_dataframe(library_records.clone())?;
    
    let report_file_path = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/report/report.parquet";
    let report_df = read_parquet_with_polars(report_file_path)?;
    
    let diann_result = merge_library_and_report(library_df, report_df)?;
    
    let diann_precursor_id_all = get_unique_precursor_ids(&diann_result)?;
    
    let (assay_rt_kept_dict, assay_im_kept_dict) = create_rt_im_dicts(&diann_precursor_id_all)?;
    
    let device = "cpu";
    let frag_repeat_num = 5;
    
    // PART 3: Single Precursor Extraction Timing
    
    // Start timing from Part 3
    let precursor_start_time = Instant::now();
    
    let precursor_id_list = vec!["LLIYGASTR2"];
    
    let each_lib_data: Vec<_> = library_records.iter()
        .filter(|record| precursor_id_list.contains(&record.transition_group_id.as_str()))
        .cloned()
        .collect();
    
    if each_lib_data.is_empty() {
        return Ok(());
    }
    
    let lib_cols = LibCols::default();
    let (precursors_list, ms1_data_list, ms2_data_list, precursor_info_list) = 
        build_lib_matrix(
            &each_lib_data,
            &lib_cols,
            5.0,
            1801.0,
            20,
        )?;
    
    let (ms1_data_tensor, ms2_data_tensor) = 
        build_precursors_matrix_step1(&ms1_data_list, &ms2_data_list, device)?;
    
    let ms2_data_tensor_processed = build_precursors_matrix_step2(ms2_data_tensor);
    
    let (ms1_range_list, ms2_range_list) = 
        build_range_matrix_step3(&ms1_data_tensor, &ms2_data_tensor_processed, frag_repeat_num, "ppm", 20.0, 50.0, device)?;
    
    let (re_ms1_data_tensor, re_ms2_data_tensor, ms1_extract_width_range_list, ms2_extract_width_range_list) = 
        build_precursors_matrix_step3(&ms1_data_tensor, &ms2_data_tensor_processed, frag_repeat_num, "ppm", 20.0, 50.0, device)?;

    let precursor_info_np_org: Vec<Vec<f64>> = precursor_info_list.iter()
        .map(|info| info.clone())
        .collect();
    
    let precursor_info_choose: Vec<Vec<f64>> = precursor_info_np_org.iter()
        .map(|row| row[..5.min(row.len())].to_vec())
        .collect();
    
    let delta_rt_kept: Vec<f64> = vec![0.0; precursors_list.len()];
    
    let assay_rt_kept: Vec<f64> = precursors_list.iter()
        .map(|prec| assay_rt_kept_dict.get(&prec[0]).copied().unwrap_or(0.0))
        .collect();
    
    let assay_im_kept: Vec<f64> = precursors_list.iter()
        .map(|prec| assay_im_kept_dict.get(&prec[0]).copied().unwrap_or(0.0))
        .collect();
    
    // PART 4: Precursor Feature Matrix and Fast Data Extraction
    
    let n_precursors = precursors_list.len();
    let n_cols = precursor_info_choose[0].len() + 3;
    
    let mut precursor_feat = Array2::<f64>::zeros((n_precursors, n_cols));
    
    for i in 0..n_precursors {
        for j in 0..precursor_info_choose[i].len() {
            precursor_feat[[i, j]] = precursor_info_choose[i][j];
        }
        precursor_feat[[i, 5]] = assay_im_kept[i];
        precursor_feat[[i, 6]] = assay_rt_kept[i];
        precursor_feat[[i, 7]] = delta_rt_kept[i];
    }
    
    let frag_info = build_frag_info(&ms1_data_tensor, &ms2_data_tensor_processed, frag_repeat_num, device);
    
    let i = 0;
    let im = precursor_feat[[i, 5]];
    let rt = precursor_feat[[i, 6]];
    let precursor_mz = precursor_feat[[i, 1]];
    
    // Find MS2 data chunk for this precursor
    let df2_index_final = finder.find(precursor_mz);
    
    // Calculate MS1 m/z range
    let ms1_range_slice = ms1_range_list.slice(s![i, .., ..]);
    let ms1_range_min_val = ms1_range_slice.iter()
        .filter(|&&v| v > 0.0)
        .fold(f32::INFINITY, |a, &b| a.min(b));
    let ms1_range_max_val = ms1_range_slice.iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    let ms1_range_min = ((ms1_range_min_val - 1.0) / 1000.0) as f64;
    let ms1_range_max = ((ms1_range_max_val + 1.0) / 1000.0) as f64;
    
    // Use indexed slicing for MS1 data (Python-style .loc[min:max])
    let precursor_result = ms1_indexed.slice_by_mz_range(ms1_range_min, ms1_range_max);
    
    // Convert m/z values to integers
    let mut precursor_result_indexed = IndexedTimsTOFData::from_timstof_data(precursor_result);
    precursor_result_indexed.convert_mz_to_integer();
    
    // Filter by IM range
    let im_tolerance = 0.05;
    let im_min = im - im_tolerance;
    let im_max = im + im_tolerance;
    let precursor_result_filtered = precursor_result_indexed.filter_by_im_range(im_min, im_max);
    
    // Process MS2 fragments with fast slicing
    let mut frag_results = Vec::new();
    let mut valid_ms2_ranges = Vec::new();
    
    for j in 0..66 {
        let ms2_range_min_val = ms2_range_list[[i, j, 0]];
        let ms2_range_max_val = ms2_range_list[[i, j, 1]];
        
        let ms2_range_min = ((ms2_range_min_val - 1.0) / 1000.0) as f64;
        let ms2_range_max = ((ms2_range_max_val + 1.0) / 1000.0) as f64;
        
        if ms2_range_min <= 0.0 || ms2_range_max <= 0.0 || ms2_range_min >= ms2_range_max {
            frag_results.push(TimsTOFData::new());
            continue;
        }
        
        valid_ms2_ranges.push((ms2_range_min, ms2_range_max));
        
        if let Some(ms2_indexed) = df2_index_final {
            // Use indexed slicing instead of linear filtering
            let frag_data = ms2_indexed.slice_by_mz_range(ms2_range_min, ms2_range_max);
            frag_results.push(frag_data);
        } else {
            frag_results.push(TimsTOFData::new());
        }
    }
    
    // Merge all fragment results
    let merged_frag_result = TimsTOFData::merge(frag_results);
    
    // Convert and filter fragments
    let mut frag_result_indexed = IndexedTimsTOFData::from_timstof_data(merged_frag_result);
    frag_result_indexed.convert_mz_to_integer();
    let frag_result_filtered = frag_result_indexed.filter_by_im_range(im_min, im_max);
    
    // ===== 添加输出功能 =====
    // 需要在文件顶部添加: mod output;
    use output::{OutputManager, SliceResult, create_ms1_data, create_ms2_data};
    
    let output_manager = OutputManager::new();
    let slice_result = SliceResult {
        version: "original".to_string(),
        timestamp: chrono::Local::now().format("%Y-%m-%d %H:%M:%S").to_string(),
        precursor_id: precursors_list[i][0].clone(),
        ms1_data: create_ms1_data(
            &precursor_result_filtered,
            (ms1_range_min, ms1_range_max),
            (im_min, im_max),
        ),
        ms2_data: create_ms2_data(&frag_result_filtered, valid_ms2_ranges.len()),
    };
    
    output_manager.save_slice_result(&slice_result)?;
    // ===== 输出功能结束 =====
    
    // PART 5: Building Mask and Intensity Matrices
    
    // Build MS1/MS2 mask matrices
    
    // Convert filtered data m/z values to sets for fast lookup
    let search_ms1_set: std::collections::HashSet<i64> = precursor_result_filtered.mz_values.iter()
        .map(|&mz| mz as i64)
        .collect();
    
    let search_ms2_set: std::collections::HashSet<i64> = frag_result_filtered.mz_values.iter()
        .map(|&mz| mz as i64)
        .collect();
    
    // Extract slices for this precursor
    let ms1_extract_slice = ms1_extract_width_range_list.slice(s![i, .., ..]);
    let ms2_extract_slice = ms2_extract_width_range_list.slice(s![i, .., ..]);
    
    // Build MS1 mask matrix
    let (n_frags_ms1, n_mz_ms1) = (ms1_extract_slice.shape()[0], ms1_extract_slice.shape()[1]);
    let mut ms1_frag_moz_matrix = Array2::<f32>::zeros((n_frags_ms1, n_mz_ms1));
    
    for j in 0..n_frags_ms1 {
        for k in 0..n_mz_ms1 {
            let val = ms1_extract_slice[[j, k]] as i64;
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
            let val = ms2_extract_slice[[j, k]] as i64;
            if val > 0 && search_ms2_set.contains(&val) {
                ms2_frag_moz_matrix[[j, k]] = 1.0;
            }
        }
    }
    
    // Get unique RT values from both MS1 and MS2 data
    let mut all_rt_set = std::collections::HashSet::new();
    
    for &rt_val in &precursor_result_filtered.rt_values_min {
        all_rt_set.insert((rt_val * 1e6) as i64);
    }
    
    for &rt_val in &frag_result_filtered.rt_values_min {
        all_rt_set.insert((rt_val * 1e6) as i64);
    }
    
    let mut all_rt_vec: Vec<f64> = all_rt_set.iter()
        .map(|&rt_int| rt_int as f64 / 1e6)
        .collect();
    all_rt_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    // Get RT list with target RT in the center (48 values)
    let all_rt = get_rt_list(all_rt_vec, rt);
    
    // Build intensity matrices
    let ms1_frag_rt_matrix = build_intensity_matrix_optimized(
        &precursor_result_filtered,
        &ms1_extract_slice.to_owned(),
        &ms1_frag_moz_matrix,
        &all_rt,
    )?;
    
    let ms2_frag_rt_matrix = build_intensity_matrix_optimized(
        &frag_result_filtered,
        &ms2_extract_slice.to_owned(),
        &ms2_frag_moz_matrix,
        &all_rt,
    )?;
    
    // PART 6: Reshaping and Combining Matrices
    
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
    
    // Combine MS1 and MS2 matrices
    let ms1_frags = ms1_reshaped.shape()[1];
    let ms2_frags = ms2_reshaped.shape()[1];
    let total_frags = ms1_frags + ms2_frags;
    let n_rt = all_rt.len();
    
    let mut full_frag_rt_matrix = Array3::<f32>::zeros((frag_repeat_num, total_frags, n_rt));
    
    // Copy MS1 data
    for rep in 0..frag_repeat_num {
        for frag in 0..ms1_frags {
            for rt_idx in 0..n_rt {
                full_frag_rt_matrix[[rep, frag, rt_idx]] = ms1_reshaped[[rep, frag, rt_idx]];
            }
        }
    }
    
    // Copy MS2 data
    for rep in 0..frag_repeat_num {
        for frag in 0..ms2_frags {
            for rt_idx in 0..n_rt {
                full_frag_rt_matrix[[rep, ms1_frags + frag, rt_idx]] = ms2_reshaped[[rep, frag, rt_idx]];
            }
        }
    }
    
    // Create RSM matrix (add batch dimension)
    let rsm_matrix = full_frag_rt_matrix.insert_axis(Axis(0));
    
    // Aggregate across repeat dimension
    let aggregated_x_sum = rsm_matrix.sum_axis(Axis(1));
    
    // Create final DataFrame
    let precursor_data = aggregated_x_sum.slice(s![0, .., ..]);
    let precursor_frag_info = frag_info.slice(s![i, .., ..]);
    
    let mut columns = Vec::new();
    
    // Add RT columns
    for (rt_idx, &rt_val) in all_rt.iter().enumerate() {
        let mut col_data = Vec::with_capacity(total_frags);
        for frag_idx in 0..total_frags {
            col_data.push(precursor_data[[frag_idx, rt_idx]] as f64);
        }
        columns.push(Series::new(&format!("{:.6}", rt_val), col_data));
    }
    
    // Add fragment info columns
    let info_names = ["ProductMz", "LibraryIntensity", "frag_type", "FragmentType"];
    for col_idx in 0..4.min(precursor_frag_info.shape()[1]) {
        let mut col_data = Vec::with_capacity(total_frags);
        for row_idx in 0..total_frags {
            col_data.push(precursor_frag_info[[row_idx, col_idx]] as f64);
        }
        columns.push(Series::new(info_names[col_idx], col_data));
    }
    
    let _final_df = DataFrame::new(columns)?;
    
    // Calculate and display timing
    let precursor_elapsed_time = precursor_start_time.elapsed();
    println!("Single precursor processing time: {:.6} seconds", precursor_elapsed_time.as_secs_f64());
    
    Ok(())
}