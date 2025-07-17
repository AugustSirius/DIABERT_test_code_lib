// Updated src/main.rs with parts 5 and 6
//-------------------------------------------------------------
//  src/main.rs   --  with Parallel Indexed Slicing and Parts 5+6
//-------------------------------------------------------------
mod utils;

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
    fs::File,
    io::Write,
};
use indicatif::{ProgressBar, ProgressStyle, ParallelProgressIterator};
use ndarray::{Array1, Array2, Array3, Array4, s, Axis, concatenate};
use polars::prelude::*;
use serde_json::json;

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
        eprintln!("Creating {} indexed MS2 windows in parallel...", n);
        let pb = ProgressBar::new(n as u64);
        pb.set_style(
            ProgressStyle::default_bar()
                .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} indexing MS2...")
                .unwrap()
                .progress_chars("#>-"),
        );
        
        let chunks: Vec<IndexedTimsTOFData> = pairs
            .into_par_iter()
            .progress_with(pb)
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
    eprintln!("Processing {} frames...", n_frames);
    
    let pb = ProgressBar::new(n_frames as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    
    // Process frames in parallel with shared converters
    let splits: Vec<FrameSplit> = (0..n_frames)
        .into_par_iter()
        .progress_with(pb)
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
    
    eprintln!("Merging results from {} frame splits...", splits.len());
    
    // Estimate MS1 size for pre-allocation
    let ms1_size_estimate: usize = splits.par_iter()
        .map(|s| s.ms1.mz_values.len())
        .sum();
    
    let merge_pb = ProgressBar::new(splits.len() as u64);
    merge_pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.yellow/red}] {pos}/{len} merging...")
            .unwrap()
            .progress_chars("#>-"),
    );
    
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
        
        merge_pb.inc(1);
    }
    
    merge_pb.finish_with_message("Merging complete!");
    
    // Create indexed MS1 data with parallel sorting
    eprintln!("Creating indexed MS1 data structure (parallel sort)...");
    let index_start = Instant::now();
    let indexed_ms1 = IndexedTimsTOFData::from_timstof_data(global_ms1);
    let index_time = index_start.elapsed();
    eprintln!("✓ Indexed {} MS1 peaks in {:.2}s", indexed_ms1.len(), index_time.as_secs_f32());
    
    eprintln!("Converting {} MS2 windows...", ms2_hash.len());
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

// Helper function to save TimsTOFData to JSON
fn save_timstof_data_to_json(data: &TimsTOFData, path: &str) -> Result<(), Box<dyn Error>> {
    let json_data = json!({
        "rt_values_min": data.rt_values_min,
        "mobility_values": data.mobility_values,
        "mz_values": data.mz_values,
        "intensity_values": data.intensity_values,
        "frame_indices": data.frame_indices,
        "scan_indices": data.scan_indices,
        "length": data.mz_values.len()
    });
    
    std::fs::create_dir_all(Path::new(path).parent().unwrap())?;
    let mut file = File::create(path)?;
    file.write_all(serde_json::to_string_pretty(&json_data)?.as_bytes())?;
    Ok(())
}

// Helper function to save Array2 to JSON
fn save_array2_to_json(array: &Array2<f32>, path: &str) -> Result<(), Box<dyn Error>> {
    let shape = array.shape();
    let data: Vec<Vec<f32>> = (0..shape[0])
        .map(|i| (0..shape[1]).map(|j| array[[i, j]]).collect())
        .collect();
    
    let json_data = json!({
        "shape": shape,
        "data": data
    });
    
    std::fs::create_dir_all(Path::new(path).parent().unwrap())?;
    let mut file = File::create(path)?;
    file.write_all(serde_json::to_string_pretty(&json_data)?.as_bytes())?;
    Ok(())
}

// Helper function to save Array3 to JSON
fn save_array3_to_json(array: &Array3<f32>, path: &str) -> Result<(), Box<dyn Error>> {
    let shape = array.shape();
    let data: Vec<Vec<Vec<f32>>> = (0..shape[0])
        .map(|i| (0..shape[1]).map(|j| 
            (0..shape[2]).map(|k| array[[i, j, k]]).collect()
        ).collect())
        .collect();
    
    let json_data = json!({
        "shape": shape,
        "data": data
    });
    
    std::fs::create_dir_all(Path::new(path).parent().unwrap())?;
    let mut file = File::create(path)?;
    file.write_all(serde_json::to_string_pretty(&json_data)?.as_bytes())?;
    Ok(())
}

// Helper function to save Vec<f64> to JSON
fn save_vec_f64_to_json(vec: &Vec<f64>, path: &str) -> Result<(), Box<dyn Error>> {
    let json_data = json!({
        "data": vec,
        "length": vec.len()
    });
    
    std::fs::create_dir_all(Path::new(path).parent().unwrap())?;
    let mut file = File::create(path)?;
    file.write_all(serde_json::to_string_pretty(&json_data)?.as_bytes())?;
    Ok(())
}

// Helper function to save library records to JSON
fn save_library_records_to_json(records: &[utils::LibraryRecord], path: &str) -> Result<(), Box<dyn Error>> {
    let json_records: Vec<_> = records.iter().map(|record| {
        json!({
            "transition_group_id": record.transition_group_id,
            "decoy": record.decoy,
            "product_mz": record.product_mz,
            "precursor_mz": record.precursor_mz,
            "tr_recalibrated": record.tr_recalibrated,
            "library_intensity": record.library_intensity,
            "fragment_type": record.fragment_type,
            "fragment_series_number": record.fragment_number
        })
    }).collect();
    
    let json_data = json!({
        "records": json_records,
        "count": records.len()
    });
    
    std::fs::create_dir_all(Path::new(path).parent().unwrap())?;
    let mut file = File::create(path)?;
    file.write_all(serde_json::to_string_pretty(&json_data)?.as_bytes())?;
    Ok(())
}

// Helper function to save Vec<Vec<String>> to JSON
fn save_precursors_list_to_json(precursors: &[Vec<String>], path: &str) -> Result<(), Box<dyn Error>> {
    let json_data = json!({
        "precursors": precursors,
        "count": precursors.len()
    });
    
    std::fs::create_dir_all(Path::new(path).parent().unwrap())?;
    let mut file = File::create(path)?;
    file.write_all(serde_json::to_string_pretty(&json_data)?.as_bytes())?;
    Ok(())
}

// Helper function to save Vec<Vec<Vec<f64>>> to JSON (3D vector)
fn save_ms_data_list_to_json(ms_data: &[Vec<Vec<f64>>], path: &str) -> Result<(), Box<dyn Error>> {
    let json_data = json!({
        "ms_data": ms_data,
        "count": ms_data.len()
    });
    
    std::fs::create_dir_all(Path::new(path).parent().unwrap())?;
    let mut file = File::create(path)?;
    file.write_all(serde_json::to_string_pretty(&json_data)?.as_bytes())?;
    Ok(())
}

// Helper function to save Vec<Vec<f64>> to JSON
fn save_precursor_info_to_json(precursor_info: &[Vec<f64>], path: &str) -> Result<(), Box<dyn Error>> {
    let json_data = json!({
        "precursor_info": precursor_info,
        "count": precursor_info.len()
    });
    
    std::fs::create_dir_all(Path::new(path).parent().unwrap())?;
    let mut file = File::create(path)?;
    file.write_all(serde_json::to_string_pretty(&json_data)?.as_bytes())?;
    Ok(())
}

// Helper function to save HashMap<String, f64> to JSON
fn save_dict_to_json(dict: &HashMap<String, f64>, path: &str) -> Result<(), Box<dyn Error>> {
    let json_data = json!({
        "dict": dict,
        "count": dict.len()
    });
    
    std::fs::create_dir_all(Path::new(path).parent().unwrap())?;
    let mut file = File::create(path)?;
    file.write_all(serde_json::to_string_pretty(&json_data)?.as_bytes())?;
    Ok(())
}

// Helper function to save Array4 to JSON
fn save_array4_to_json(array: &Array4<f32>, path: &str) -> Result<(), Box<dyn Error>> {
    let shape = array.shape();
    let data: Vec<Vec<Vec<Vec<f32>>>> = (0..shape[0])
        .map(|i| (0..shape[1]).map(|j| 
            (0..shape[2]).map(|k| 
                (0..shape[3]).map(|l| array[[i, j, k, l]]).collect()
            ).collect()
        ).collect())
        .collect();
    
    let json_data = json!({
        "shape": shape,
        "data": data
    });
    
    std::fs::create_dir_all(Path::new(path).parent().unwrap())?;
    let mut file = File::create(path)?;
    file.write_all(serde_json::to_string_pretty(&json_data)?.as_bytes())?;
    Ok(())
}

// Helper function to save precursor feature matrix to JSON
fn save_precursor_feat_to_json(array: &Array2<f64>, path: &str) -> Result<(), Box<dyn Error>> {
    let shape = array.shape();
    let data: Vec<Vec<f64>> = (0..shape[0])
        .map(|i| (0..shape[1]).map(|j| array[[i, j]]).collect())
        .collect();
    
    let json_data = json!({
        "shape": shape,
        "data": data
    });
    
    std::fs::create_dir_all(Path::new(path).parent().unwrap())?;
    let mut file = File::create(path)?;
    file.write_all(serde_json::to_string_pretty(&json_data)?.as_bytes())?;
    Ok(())
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
    
    println!("Using d_folder: {}", d_folder);
    println!("Using {} CPU threads", rayon::current_num_threads());
    
    let d_path = Path::new(&d_folder);
    if !d_path.exists() {
        return Err(format!("folder {:?} not found", d_path).into());
    }
    
    // Create output directory
    let output_dir = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/20250711/rust_comparison_results";
    std::fs::create_dir_all(output_dir)?;
    
    println!("========== PART 1: TimsTOF Data Processing ==========");
    eprintln!("Starting TimsTOF data processing...");
    let t0 = std::time::Instant::now();
    let (ms1_indexed, ms2_pairs) = read_timstof_grouped(d_path)?;
    let decode_time = t0.elapsed();
    
    println!("✓ Total decode time: {:.2}s", decode_time.as_secs_f32());
    println!("✓ MS1 peaks (indexed): {}", ms1_indexed.len());
    println!("✓ MS2 windows: {}", ms2_pairs.len());
    
    // OUTPUT STEP 0I: Save TimsTOF data summary
    let timstof_summary = json!({
        "ms1_peaks_count": ms1_indexed.len(),
        "ms2_windows_count": ms2_pairs.len(),
        "decode_time_seconds": decode_time.as_secs_f64(),
        "ms1_sample_data": {
            "rt_values_min": ms1_indexed.rt_values_min.iter().take(10).collect::<Vec<_>>(),
            "mobility_values": ms1_indexed.mobility_values.iter().take(10).collect::<Vec<_>>(),
            "mz_values": ms1_indexed.mz_values.iter().take(10).collect::<Vec<_>>(),
            "intensity_values": ms1_indexed.intensity_values.iter().take(10).collect::<Vec<_>>()
        },
        "ms2_sample_windows": ms2_pairs.iter().take(3).map(|((low, high), data)| {
            json!({
                "mz_range": [low, high],
                "peaks_count": data.mz_values.len()
            })
        }).collect::<Vec<_>>()
    });
    std::fs::create_dir_all(output_dir)?;
    let mut file = File::create(format!("{}/new_version_step0i_timstof_summary.json", output_dir))?;
    file.write_all(serde_json::to_string_pretty(&timstof_summary)?.as_bytes())?;
    
    eprintln!("Building FastChunkFinder with parallel indexing...");
    let finder_start = std::time::Instant::now();
    let finder = FastChunkFinder::new(ms2_pairs)?;
    eprintln!("✓ FastChunkFinder built ({} ranges) in {:.2}s", 
             finder.range_count(), finder_start.elapsed().as_secs_f32());
    
    // Test queries
    eprintln!("Running test queries...");
    for q in [350.0, 550.0, 750.0, 950.0] {
        match finder.find(q) {
            None => println!("m/z {:.1} → not in any window", q),
            Some(d) => println!("m/z {:.1} → window with {} peaks", q, d.len()),
        }
    }
    
    println!("\n========== PART 2: Library and Report Processing ==========");
    
    let lib_file_path = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/lib/TPHPlib_frag1025_swissprot_final_all_from_Yueliang.tsv";
    let library_records = process_library_fast(lib_file_path)?;
    println!("✓ Loaded {} library records", library_records.len());
    
    eprintln!("Converting library to DataFrame...");
    let library_df = library_records_to_dataframe(library_records.clone())?;
    println!("✓ Library DataFrame created with {} rows", library_df.height());
    
    let report_file_path = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/report/report.parquet";
    eprintln!("Reading DIA-NN report...");
    let report_df = read_parquet_with_polars(report_file_path)?;
    println!("✓ Report loaded with {} rows", report_df.height());
    
    eprintln!("Merging library and report data...");
    let diann_result = merge_library_and_report(library_df, report_df)?;
    println!("✓ Merged data: {} rows", diann_result.height());
    
    eprintln!("Extracting unique precursor IDs...");
    let diann_precursor_id_all = get_unique_precursor_ids(&diann_result)?;
    println!("✓ Found {} unique precursors", diann_precursor_id_all.height());
    
    eprintln!("Creating RT and IM lookup dictionaries...");
    let (assay_rt_kept_dict, assay_im_kept_dict) = create_rt_im_dicts(&diann_precursor_id_all)?;
    println!("✓ RT dictionary: {} entries", assay_rt_kept_dict.len());
    println!("✓ IM dictionary: {} entries", assay_im_kept_dict.len());
    
    let precursor_id_all: Vec<String> = assay_rt_kept_dict.keys().cloned().collect();
    println!("✓ Total precursors with RT: {}", precursor_id_all.len());
    
    let device = "cpu";
    let frag_repeat_num = 5;
    println!("\n✓ Configuration:");
    println!("  - Device: {}", device);
    println!("  - Fragment repeat num: {}", frag_repeat_num);
    
    println!("\n========== PART 3: Single Precursor Extraction Timing ==========");
    
    // Start timing from Part 3
    let precursor_start_time = Instant::now();
    
    let precursor_id_list = vec!["LLIYGASTR2"];
    println!("Processing precursor: {:?}", precursor_id_list);
    
    let each_lib_data: Vec<_> = library_records.iter()
        .filter(|record| precursor_id_list.contains(&record.transition_group_id.as_str()))
        .cloned()
        .collect();
    
    if each_lib_data.is_empty() {
        println!("Warning: No matching precursor data found for {:?}", precursor_id_list);
        return Ok(());
    }
    
    println!("✓ Found {} library records for precursor", each_lib_data.len());
    
    // OUTPUT STEP 0: Save each_lib_data (library records for precursor)
    save_library_records_to_json(&each_lib_data, &format!("{}/new_version_step0_each_lib_data.json", output_dir))?;
    
    let lib_cols = LibCols::default();
    let (precursors_list, ms1_data_list, ms2_data_list, precursor_info_list) = 
        build_lib_matrix(
            &each_lib_data,
            &lib_cols,
            5.0,
            1801.0,
            20,
        )?;
    
    println!("✓ Built library matrix:");
    println!("  - Precursors: {}", precursors_list.len());
    println!("  - MS1 data entries: {}", ms1_data_list.len());
    println!("  - MS2 data entries: {}", ms2_data_list.len());
    
    // OUTPUT STEP 0A: Save build_lib_matrix results
    save_precursors_list_to_json(&precursors_list, &format!("{}/new_version_step0a_precursors_list.json", output_dir))?;
    save_ms_data_list_to_json(&ms1_data_list, &format!("{}/new_version_step0a_ms1_data_list.json", output_dir))?;
    save_ms_data_list_to_json(&ms2_data_list, &format!("{}/new_version_step0a_ms2_data_list.json", output_dir))?;
    save_precursor_info_to_json(&precursor_info_list, &format!("{}/new_version_step0a_precursor_info_list.json", output_dir))?;
    
    let (ms1_data_tensor, ms2_data_tensor) = 
        build_precursors_matrix_step1(&ms1_data_list, &ms2_data_list, device)?;
    println!("✓ Built tensor step 1:");
    println!("  - MS1 tensor shape: {:?}", ms1_data_tensor.shape());
    println!("  - MS2 tensor shape: {:?}", ms2_data_tensor.shape());
    
    // OUTPUT STEP 0B: Save tensor step 1 results
    save_array3_to_json(&ms1_data_tensor, &format!("{}/new_version_step0b_ms1_tensor_step1.json", output_dir))?;
    save_array3_to_json(&ms2_data_tensor, &format!("{}/new_version_step0b_ms2_tensor_step1.json", output_dir))?;
    
    let ms2_data_tensor_processed = build_precursors_matrix_step2(ms2_data_tensor);
    println!("✓ Processed MS2 tensor in step 2");
    
    // OUTPUT STEP 0C: Save tensor step 2 results
    save_array3_to_json(&ms2_data_tensor_processed, &format!("{}/new_version_step0c_ms2_tensor_step2.json", output_dir))?;
    
    let (ms1_range_list, ms2_range_list) = 
        build_range_matrix_step3(&ms1_data_tensor, &ms2_data_tensor_processed, frag_repeat_num, "ppm", 20.0, 50.0, device)?;
    println!("✓ Built range matrices:");
    println!("  - MS1 range shape: {:?}", ms1_range_list.shape());
    println!("  - MS2 range shape: {:?}", ms2_range_list.shape());
    
    // OUTPUT STEP 0D: Save range matrices
    save_array3_to_json(&ms1_range_list, &format!("{}/new_version_step0d_ms1_range_list.json", output_dir))?;
    save_array3_to_json(&ms2_range_list, &format!("{}/new_version_step0d_ms2_range_list.json", output_dir))?;
    
    let (re_ms1_data_tensor, re_ms2_data_tensor, ms1_extract_width_range_list, ms2_extract_width_range_list) = 
        build_precursors_matrix_step3(&ms1_data_tensor, &ms2_data_tensor_processed, frag_repeat_num, "ppm", 20.0, 50.0, device)?;

    println!("✓ Built extract width range lists:");
    println!("  - MS1 extract width shape: {:?}", ms1_extract_width_range_list.shape());
    println!("  - MS2 extract width shape: {:?}", ms2_extract_width_range_list.shape());
    
    // OUTPUT STEP 0E: Save extract width range lists
    save_array3_to_json(&ms1_extract_width_range_list, &format!("{}/new_version_step0e_ms1_extract_width_range_list.json", output_dir))?;
    save_array3_to_json(&ms2_extract_width_range_list, &format!("{}/new_version_step0e_ms2_extract_width_range_list.json", output_dir))?;
    
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
    
    println!("✓ Prepared precursor metadata:");
    println!("  - Delta RT values: {}", delta_rt_kept.len());
    println!("  - Assay RT values: {}", assay_rt_kept.len());
    println!("  - Assay IM values: {}", assay_im_kept.len());
    
    // OUTPUT STEP 0F: Save precursor metadata
    save_dict_to_json(&assay_rt_kept_dict, &format!("{}/new_version_step0f_assay_rt_kept_dict.json", output_dir))?;
    save_dict_to_json(&assay_im_kept_dict, &format!("{}/new_version_step0f_assay_im_kept_dict.json", output_dir))?;
    save_precursor_info_to_json(&precursor_info_choose, &format!("{}/new_version_step0f_precursor_info_choose.json", output_dir))?;
    
    println!("\n========== PART 4: Precursor Feature Matrix and Fast Data Extraction ==========");
    
    println!("Building precursor feature matrix...");
    
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
    
    println!("✓ Precursor feature matrix shape: {:?}", precursor_feat.shape());
    
    // OUTPUT STEP 0G: Save precursor feature matrix
    save_precursor_feat_to_json(&precursor_feat, &format!("{}/new_version_step0g_precursor_feat.json", output_dir))?;
    
    println!("Building fragment info...");
    let frag_info = build_frag_info(&ms1_data_tensor, &ms2_data_tensor_processed, frag_repeat_num, device);
    println!("✓ Fragment info shape: {:?}", frag_info.shape());
    
    // OUTPUT STEP 0H: Save fragment info
    save_array3_to_json(&frag_info, &format!("{}/new_version_step0h_frag_info.json", output_dir))?;
    
    let i = 0;
    let im = precursor_feat[[i, 5]];
    let rt = precursor_feat[[i, 6]];
    let precursor_mz = precursor_feat[[i, 1]];
    
    println!("\nProcessing precursor {} with:", i);
    println!("  - Precursor m/z: {:.4}", precursor_mz);
    println!("  - IM: {:.4}", im);
    println!("  - RT: {:.4}", rt);
    
    // Find MS2 data chunk for this precursor
    let df2_index_final = finder.find(precursor_mz);
    
    if df2_index_final.is_none() {
        println!("Warning: No MS2 window found for precursor m/z {:.4}", precursor_mz);
    } else {
        println!("✓ Found MS2 window for precursor");
    }
    
    // Calculate MS1 m/z range
    let ms1_range_slice = ms1_range_list.slice(s![i, .., ..]);
    let ms1_range_min_val = ms1_range_slice.iter()
        .filter(|&&v| v > 0.0)
        .fold(f32::INFINITY, |a, &b| a.min(b));
    let ms1_range_max_val = ms1_range_slice.iter()
        .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    
    let ms1_range_min = ((ms1_range_min_val - 1.0) / 1000.0) as f64;
    let ms1_range_max = ((ms1_range_max_val + 1.0) / 1000.0) as f64;
    
    println!("MS1 m/z range: [{:.4}, {:.4}]", ms1_range_min, ms1_range_max);
    
    // OUTPUT STEP 0J: Save m/z range calculation details
    let mz_range_details = json!({
        "precursor_index": i,
        "precursor_mz": precursor_mz,
        "im": im,
        "rt": rt,
        "ms1_range_min_val": ms1_range_min_val,
        "ms1_range_max_val": ms1_range_max_val,
        "ms1_range_min": ms1_range_min,
        "ms1_range_max": ms1_range_max,
        "ms1_range_slice_sample": ms1_range_slice.slice(s![..3, ..]).to_owned().into_raw_vec(),
        "ms1_range_slice_shape": ms1_range_slice.shape().to_vec()
    });
    let mut file = File::create(format!("{}/new_version_step0j_mz_range_details.json", output_dir))?;
    file.write_all(serde_json::to_string_pretty(&mz_range_details)?.as_bytes())?;
    
    // Use indexed slicing for MS1 data (Python-style .loc[min:max])
    let slice_start = Instant::now();
    let precursor_result = ms1_indexed.slice_by_mz_range(ms1_range_min, ms1_range_max);
    let slice_time = slice_start.elapsed();
    println!("✓ MS1 slice time: {:.6}s ({} peaks)", slice_time.as_secs_f64(), precursor_result.mz_values.len());
    
    // OUTPUT STEP 1: Save MS1 precursor result
    save_timstof_data_to_json(&precursor_result, &format!("{}/new_version_step1_precursor_result.json", output_dir))?;
    
    // Convert m/z values to integers
    let mut precursor_result_indexed = IndexedTimsTOFData::from_timstof_data(precursor_result);
    precursor_result_indexed.convert_mz_to_integer();
    
    // OUTPUT STEP 2: Save MS1 integer converted result
    let precursor_result_int = TimsTOFData {
        rt_values_min: precursor_result_indexed.rt_values_min.clone(),
        mobility_values: precursor_result_indexed.mobility_values.clone(),
        mz_values: precursor_result_indexed.mz_values.clone(),
        intensity_values: precursor_result_indexed.intensity_values.clone(),
        frame_indices: precursor_result_indexed.frame_indices.clone(),
        scan_indices: precursor_result_indexed.scan_indices.clone(),
    };
    save_timstof_data_to_json(&precursor_result_int, &format!("{}/new_version_step2_precursor_result_int.json", output_dir))?;
    
    // Filter by IM range
    let im_tolerance = 0.05;
    let im_min = im - im_tolerance;
    let im_max = im + im_tolerance;
    let precursor_result_filtered = precursor_result_indexed.filter_by_im_range(im_min, im_max);
    println!("✓ Filtered by IM [{:.4}, {:.4}]: {} peaks", 
             im_min, im_max, precursor_result_filtered.mz_values.len());
    
    // OUTPUT STEP 3: Save MS1 IM filtered result
    save_timstof_data_to_json(&precursor_result_filtered, &format!("{}/new_version_step3_precursor_result_filtered.json", output_dir))?;
    
    // Process MS2 fragments with fast slicing
    println!("\nProcessing MS2 fragments with indexed slicing...");
    let mut frag_results = Vec::new();
    let ms2_slice_start = Instant::now();
    
    for j in 0..66 {
        let ms2_range_min_val = ms2_range_list[[i, j, 0]];
        let ms2_range_max_val = ms2_range_list[[i, j, 1]];
        
        let ms2_range_min = ((ms2_range_min_val - 1.0) / 1000.0) as f64;
        let ms2_range_max = ((ms2_range_max_val + 1.0) / 1000.0) as f64;
        
        if ms2_range_min <= 0.0 || ms2_range_max <= 0.0 || ms2_range_min >= ms2_range_max {
            frag_results.push(TimsTOFData::new());
            continue;
        }
        
        if let Some(ms2_indexed) = df2_index_final {
            // Use indexed slicing instead of linear filtering
            let frag_data = ms2_indexed.slice_by_mz_range(ms2_range_min, ms2_range_max);
            frag_results.push(frag_data);
        } else {
            frag_results.push(TimsTOFData::new());
        }
    }
    
    let ms2_slice_time = ms2_slice_start.elapsed();
    println!("✓ MS2 fragments slice time: {:.6}s", ms2_slice_time.as_secs_f64());
    
    // Merge all fragment results
    let merged_frag_result = TimsTOFData::merge(frag_results);
    println!("✓ Merged MS2 fragments: {} peaks", merged_frag_result.mz_values.len());
    
    // OUTPUT STEP 4: Save MS2 merged result
    save_timstof_data_to_json(&merged_frag_result, &format!("{}/new_version_step4_merged_frag_result.json", output_dir))?;
    
    // Convert and filter fragments
    let mut frag_result_indexed = IndexedTimsTOFData::from_timstof_data(merged_frag_result);
    frag_result_indexed.convert_mz_to_integer();
    let frag_result_filtered = frag_result_indexed.filter_by_im_range(im_min, im_max);
    println!("✓ Filtered MS2 by IM: {} peaks", frag_result_filtered.mz_values.len());
    
    // OUTPUT STEP 5: Save MS2 filtered result
    save_timstof_data_to_json(&frag_result_filtered, &format!("{}/new_version_step5_frag_result_filtered.json", output_dir))?;
    
    // Summary
    println!("\n📊 Part 4 Summary:");
    println!("  - Precursor feature matrix: {:?}", precursor_feat.shape());
    println!("  - Fragment info matrix: {:?}", frag_info.shape());
    println!("  - MS1 peaks (filtered): {}", precursor_result_filtered.mz_values.len());
    println!("  - MS2 peaks (filtered): {}", frag_result_filtered.mz_values.len());
    
    println!("\n========== PART 5: Building Mask and Intensity Matrices ==========");
    
    // Build MS1/MS2 mask matrices
    println!("Building mask matrices...");
    
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
    
    println!("✓ MS1 mask matrix shape: {:?}", ms1_frag_moz_matrix.shape());
    
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
    
    println!("✓ MS2 mask matrix shape: {:?}", ms2_frag_moz_matrix.shape());
    
    // OUTPUT STEP 6: Save mask matrices
    save_array2_to_json(&ms1_frag_moz_matrix, &format!("{}/new_version_step6_ms1_mask_matrix.json", output_dir))?;
    save_array2_to_json(&ms2_frag_moz_matrix, &format!("{}/new_version_step6_ms2_mask_matrix.json", output_dir))?;
    
    // Get unique RT values from both MS1 and MS2 data
    println!("\nPreparing RT list...");
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
    println!("✓ RT list length: {} (centered on RT: {:.4})", all_rt.len(), rt);
    
    // OUTPUT STEP 7: Save RT list
    save_vec_f64_to_json(&all_rt, &format!("{}/new_version_step7_rt_list.json", output_dir))?;
    
    // Build intensity matrices
    println!("\nBuilding intensity matrices...");
    
    let ms1_intensity_start = Instant::now();
    let ms1_frag_rt_matrix = build_intensity_matrix_optimized(
        &precursor_result_filtered,
        &ms1_extract_slice.to_owned(),
        &ms1_frag_moz_matrix,
        &all_rt,
    )?;
    println!("✓ MS1 intensity matrix shape: {:?} (built in {:.3}s)", 
             ms1_frag_rt_matrix.shape(), ms1_intensity_start.elapsed().as_secs_f64());
    
    let ms2_intensity_start = Instant::now();
    let ms2_frag_rt_matrix = build_intensity_matrix_optimized(
        &frag_result_filtered,
        &ms2_extract_slice.to_owned(),
        &ms2_frag_moz_matrix,
        &all_rt,
    )?;
    println!("✓ MS2 intensity matrix shape: {:?} (built in {:.3}s)", 
             ms2_frag_rt_matrix.shape(), ms2_intensity_start.elapsed().as_secs_f64());
    
    // OUTPUT STEP 8: Save intensity matrices
    save_array2_to_json(&ms1_frag_rt_matrix, &format!("{}/new_version_step8_ms1_intensity_matrix.json", output_dir))?;
    save_array2_to_json(&ms2_frag_rt_matrix, &format!("{}/new_version_step8_ms2_intensity_matrix.json", output_dir))?;
    
    println!("\n========== PART 6: Reshaping and Combining Matrices ==========");
    
    // Reshape MS1 matrix
    let (ms1_rows, ms1_cols) = ms1_frag_rt_matrix.dim();
    let ms1_reshaped = ms1_frag_rt_matrix.into_shape((
        frag_repeat_num,
        ms1_rows / frag_repeat_num,
        ms1_cols
    ))?;
    println!("✓ MS1 reshaped to: {:?}", ms1_reshaped.shape());
    
    // Reshape MS2 matrix
    let (ms2_rows, ms2_cols) = ms2_frag_rt_matrix.dim();
    let ms2_reshaped = ms2_frag_rt_matrix.into_shape((
        frag_repeat_num,
        ms2_rows / frag_repeat_num,
        ms2_cols
    ))?;
    println!("✓ MS2 reshaped to: {:?}", ms2_reshaped.shape());
    
    // OUTPUT STEP 9: Save reshaped matrices
    save_array3_to_json(&ms1_reshaped, &format!("{}/new_version_step9_ms1_reshaped.json", output_dir))?;
    save_array3_to_json(&ms2_reshaped, &format!("{}/new_version_step9_ms2_reshaped.json", output_dir))?;
    
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
    
    println!("✓ Combined matrix shape: {:?}", full_frag_rt_matrix.shape());
    
    // OUTPUT STEP 10: Save combined matrix
    save_array3_to_json(&full_frag_rt_matrix, &format!("{}/new_version_step10_combined_matrix.json", output_dir))?;
    
    // Create RSM matrix (add batch dimension)
    let rsm_matrix = full_frag_rt_matrix.insert_axis(Axis(0));
    println!("✓ RSM matrix shape: {:?}", rsm_matrix.shape());
    
    // Aggregate across repeat dimension
    let aggregated_x_sum = rsm_matrix.sum_axis(Axis(1));
    println!("✓ Aggregated matrix shape: {:?}", aggregated_x_sum.shape());
    
    // OUTPUT STEP 11: Save aggregated matrix
    let aggregated_2d = aggregated_x_sum.slice(s![0, .., ..]).to_owned();
    save_array2_to_json(&aggregated_2d, &format!("{}/new_version_step11_aggregated_matrix.json", output_dir))?;
    
    // Create final DataFrame
    println!("\nCreating final DataFrame...");
    
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
    
    let final_df = DataFrame::new(columns)?;
    println!("✓ Final DataFrame shape: {} rows × {} columns", final_df.height(), final_df.width());

    // OUTPUT STEP 12: Save final DataFrame
    let output_path = format!("{}/new_version_step12_final_dataframe.csv", output_dir);
    let mut file = File::create(&output_path)?;
    CsvWriter::new(&mut file)
        .include_header(true)
        .finish(&mut final_df.clone())?;
    
    // Calculate and display timing
    let precursor_elapsed_time = precursor_start_time.elapsed();
    println!("\n⏱️  Single precursor processing time (Parts 3-6): {:.6} seconds", precursor_elapsed_time.as_secs_f64());
    println!("   ({:.3} ms)", precursor_elapsed_time.as_millis());
    
    println!("\n🎉 Processing complete! Total time: {:.2}s", t0.elapsed().as_secs_f32());
    
    println!("\n📁 Output files saved to: {}", output_dir);
    println!("   - Step 0: each_lib_data.json");
    println!("   - Step 0A: precursors_list.json, ms1_data_list.json, ms2_data_list.json, precursor_info_list.json");
    println!("   - Step 0B: ms1_tensor_step1.json, ms2_tensor_step1.json");
    println!("   - Step 0C: ms2_tensor_step2.json");
    println!("   - Step 0D: ms1_range_list.json, ms2_range_list.json");
    println!("   - Step 0E: ms1_extract_width_range_list.json, ms2_extract_width_range_list.json");
    println!("   - Step 0F: assay_rt_kept_dict.json, assay_im_kept_dict.json, precursor_info_choose.json");
    println!("   - Step 0G: precursor_feat.json");
    println!("   - Step 0H: frag_info.json");
    println!("   - Step 0I: timstof_summary.json");
    println!("   - Step 0J: mz_range_details.json");
    println!("   - Step 1: precursor_result.json");
    println!("   - Step 2: precursor_result_int.json");
    println!("   - Step 3: precursor_result_filtered.json");
    println!("   - Step 4: merged_frag_result.json");
    println!("   - Step 5: frag_result_filtered.json");
    println!("   - Step 6: ms1_mask_matrix.json, ms2_mask_matrix.json");
    println!("   - Step 7: rt_list.json");
    println!("   - Step 8: ms1_intensity_matrix.json, ms2_intensity_matrix.json");
    println!("   - Step 9: ms1_reshaped.json, ms2_reshaped.json");
    println!("   - Step 10: combined_matrix.json");
    println!("   - Step 11: aggregated_matrix.json");
    println!("   - Step 12: final_dataframe.csv");
    
    Ok(())
}