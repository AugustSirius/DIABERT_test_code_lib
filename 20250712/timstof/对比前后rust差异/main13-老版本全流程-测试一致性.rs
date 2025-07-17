mod utils;

use std::fs::File;
use std::error::Error;
use std::time::Instant;
use std::io::Write;
use rayon::prelude::*;
use csv::{ReaderBuilder, Writer};
use polars::prelude::*;
use std::collections::{HashMap, HashSet};
use ndarray::{Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, Axis, s};
use std::f64::{INFINITY, NAN};
use timsrust::readers::{FrameReader, MetadataReader};
use timsrust::converters::ConvertableDomain;
use timsrust::MSLevel;
use std::path::Path;
use serde_json::json;

use utils::*;

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

fn read_timstof_data(bruker_d_folder_path: &str) -> Result<TimsTOFData, Box<dyn Error>> {
    let tdf_path = Path::new(bruker_d_folder_path).join("analysis.tdf");
    let metadata = MetadataReader::new(&tdf_path)?;
    let mz_converter = metadata.mz_converter;
    let im_converter = metadata.im_converter;
    
    let frame_reader = FrameReader::new(bruker_d_folder_path)?;
    
    let frame_results: Vec<TimsTOFData> = (0..frame_reader.len())
        .into_par_iter()
        .filter_map(|idx| frame_reader.get(idx).ok())
        .filter(|frame| frame.ms_level == MSLevel::MS1)
        .map(|frame| {
            let mut frame_data = TimsTOFData::new();
            let rt_min = frame.rt_in_seconds / 60.0;
            
            let n_peaks = frame.tof_indices.len();
            frame_data.rt_values_min.reserve(n_peaks);
            frame_data.mobility_values.reserve(n_peaks);
            frame_data.mz_values.reserve(n_peaks);
            frame_data.intensity_values.reserve(n_peaks);
            frame_data.frame_indices.reserve(n_peaks);
            frame_data.scan_indices.reserve(n_peaks);
            
            for (peak_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                .zip(frame.intensities.iter())
                .enumerate() 
            {
                let scan = find_scan_for_index(peak_idx, &frame.scan_offsets);
                let mz = mz_converter.convert(tof as f64);
                let im = im_converter.convert(scan as f64);
                
                frame_data.rt_values_min.push(rt_min);
                frame_data.mobility_values.push(im);
                frame_data.mz_values.push(mz);
                frame_data.intensity_values.push(intensity);
                frame_data.frame_indices.push(frame.index);
                frame_data.scan_indices.push(scan);
            }
            
            frame_data
        })
        .collect();
    
    let timstof_data = TimsTOFData::merge(frame_results);
    Ok(timstof_data)
}

fn read_timstof_data_with_full_ms2(
    bruker_d_folder_path: &str,
    ms1_mz_min: f64,
    ms1_mz_max: f64,
) -> Result<HashMap<String, TimsTOFData>, Box<dyn Error>> {
    let tdf_path = Path::new(bruker_d_folder_path).join("analysis.tdf");
    let metadata = MetadataReader::new(&tdf_path)?;
    let mz_converter = metadata.mz_converter;
    let im_converter = metadata.im_converter;
    
    let frame_reader = FrameReader::new(bruker_d_folder_path)?;
    
    let (ms1_data_vec, ms2_data_vec): (Vec<TimsTOFData>, Vec<TimsTOFData>) = 
        (0..frame_reader.len())
        .into_par_iter()
        .filter_map(|idx| frame_reader.get(idx).ok())
        .map(|frame| {
            let rt_min = frame.rt_in_seconds / 60.0;
            let mut ms1_frame_data = TimsTOFData::new();
            let mut ms2_frame_data = TimsTOFData::new();
            
            match frame.ms_level {
                MSLevel::MS1 => {
                    let n_peaks = frame.tof_indices.len();
                    ms1_frame_data.rt_values_min.reserve(n_peaks);
                    ms1_frame_data.mobility_values.reserve(n_peaks);
                    ms1_frame_data.mz_values.reserve(n_peaks);
                    ms1_frame_data.intensity_values.reserve(n_peaks);
                    ms1_frame_data.frame_indices.reserve(n_peaks);
                    ms1_frame_data.scan_indices.reserve(n_peaks);
                    
                    for (peak_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                        .zip(frame.intensities.iter())
                        .enumerate() 
                    {
                        let mz = mz_converter.convert(tof as f64);
                        
                        if mz >= ms1_mz_min && mz <= ms1_mz_max {
                            let scan = find_scan_for_index(peak_idx, &frame.scan_offsets);
                            let im = im_converter.convert(scan as f64);
                            
                            ms1_frame_data.rt_values_min.push(rt_min);
                            ms1_frame_data.mobility_values.push(im);
                            ms1_frame_data.mz_values.push(mz);
                            ms1_frame_data.intensity_values.push(intensity);
                            ms1_frame_data.frame_indices.push(frame.index);
                            ms1_frame_data.scan_indices.push(scan);
                        }
                    }
                }
                MSLevel::MS2 => {
                    let quad_settings = &frame.quadrupole_settings;
                    
                    for i in 0..quad_settings.isolation_mz.len() {
                        if i >= quad_settings.isolation_width.len() {
                            break;
                        }
                        
                        let precursor_mz = quad_settings.isolation_mz[i];
                        let isolation_width = quad_settings.isolation_width[i];
                        
                        if precursor_mz < ms1_mz_min - isolation_width/2.0 || 
                           precursor_mz > ms1_mz_max + isolation_width/2.0 {
                            continue;
                        }
                        
                        for (peak_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                            .zip(frame.intensities.iter())
                            .enumerate() 
                        {
                            let scan = find_scan_for_index(peak_idx, &frame.scan_offsets);
                            
                            if scan >= quad_settings.scan_starts[i] && scan <= quad_settings.scan_ends[i] {
                                let mz = mz_converter.convert(tof as f64);
                                let im = im_converter.convert(scan as f64);
                                
                                ms2_frame_data.rt_values_min.push(rt_min);
                                ms2_frame_data.mobility_values.push(im);
                                ms2_frame_data.mz_values.push(mz);
                                ms2_frame_data.intensity_values.push(intensity);
                                ms2_frame_data.frame_indices.push(frame.index);
                                ms2_frame_data.scan_indices.push(scan);
                            }
                        }
                    }
                }
                _ => {}
            }
            
            (ms1_frame_data, ms2_frame_data)
        })
        .fold(|| (Vec::new(), Vec::new()), 
            |(mut ms1_acc, mut ms2_acc), (ms1_data, ms2_data)| {
                if !ms1_data.mz_values.is_empty() {
                    ms1_acc.push(ms1_data);
                }
                if !ms2_data.mz_values.is_empty() {
                    ms2_acc.push(ms2_data);
                }
                (ms1_acc, ms2_acc)
            })
        .reduce(|| (Vec::new(), Vec::new()),
            |(mut ms1_acc1, mut ms2_acc1), (mut ms1_acc2, mut ms2_acc2)| {
                ms1_acc1.append(&mut ms1_acc2);
                ms2_acc1.append(&mut ms2_acc2);
                (ms1_acc1, ms2_acc1)
            });
    
    let mut data_map: HashMap<String, TimsTOFData> = HashMap::new();
    data_map.insert("ms1".to_string(), TimsTOFData::merge(ms1_data_vec));
    data_map.insert("ms2".to_string(), TimsTOFData::merge(ms2_data_vec));
    
    Ok(data_map)
}

fn build_intensity_matrix_optimized_parallel(
    data: &TimsTOFData,
    extract_width_range_list: &Array2<f32>,
    frag_moz_matrix: &Array2<f32>,
    all_rt: &[f64],
) -> Result<Array2<f32>, Box<dyn Error>> {
    use rayon::prelude::*;
    use std::time::Instant;
    let start = Instant::now();
    
    let n_frags = extract_width_range_list.shape()[0];
    let n_rt = all_rt.len();
    
    let mut unique_mz_set = HashSet::new();
    for &mz in &data.mz_values {
        unique_mz_set.insert(mz as i64);
    }
    for i in 0..n_frags {
        for j in 0..extract_width_range_list.shape()[1] {
            let mz = extract_width_range_list[[i, j]] as i64;
            if mz > 0 {
                unique_mz_set.insert(mz);
            }
        }
    }
    
    let mut unique_mz: Vec<i64> = unique_mz_set.into_iter().collect();
    unique_mz.sort_unstable();
    
    let mz_to_idx: HashMap<i64, usize> = unique_mz.iter()
        .enumerate()
        .map(|(idx, &mz)| (mz, idx))
        .collect();
    
    let rt_to_idx: HashMap<i64, usize> = all_rt.iter()
        .enumerate()
        .map(|(idx, &rt)| ((rt * 1e6) as i64, idx))
        .collect();
    
    let mut pivot_matrix = Array2::<f32>::zeros((unique_mz.len(), n_rt));
    
    for i in 0..data.rt_values_min.len() {
        let rt_key = (data.rt_values_min[i] * 1e6) as i64;
        let mz = data.mz_values[i] as i64;
        let intensity = data.intensity_values[i] as f32;
        
        if let (Some(&rt_idx), Some(&mz_idx)) = (rt_to_idx.get(&rt_key), mz_to_idx.get(&mz)) {
            pivot_matrix[[mz_idx, rt_idx]] += intensity;
        }
    }
    
    let results: Vec<Vec<f32>> = (0..n_frags)
        .into_par_iter()
        .map(|a| {
            let mut row_result = vec![0.0f32; n_rt];
            
            let moz_list: Vec<i64> = (0..extract_width_range_list.shape()[1])
                .map(|j| extract_width_range_list[[a, j]] as i64)
                .collect();
            
            for (j, &mz) in moz_list.iter().enumerate() {
                if let Some(&mz_idx) = mz_to_idx.get(&mz) {
                    let mask_val = frag_moz_matrix[[a, j]];
                    if mask_val > 0.0 {
                        for k in 0..n_rt {
                            row_result[k] += mask_val * pivot_matrix[[mz_idx, k]];
                        }
                    }
                }
            }
            
            row_result
        })
        .collect();
    
    let mut frag_rt_matrix = Array2::<f32>::zeros((n_frags, n_rt));
    for (a, row) in results.into_iter().enumerate() {
        for (k, val) in row.into_iter().enumerate() {
            frag_rt_matrix[[a, k]] = val;
        }
    }
    
    Ok(frag_rt_matrix)
}

fn create_final_dataframe(
    rsm_matrix: &Array4<f32>,
    frag_info: &Array3<f32>,
    all_rt: &[f64],
    precursor_idx: usize,
) -> Result<DataFrame, Box<dyn Error>> {
    let aggregated_x_sum = rsm_matrix.sum_axis(Axis(1));
    let precursor_data = aggregated_x_sum.slice(s![precursor_idx, .., ..]);
    let precursor_frag_info = frag_info.slice(s![precursor_idx, .., ..]);
    
    let n_frags = precursor_data.shape()[0];
    let n_rt = all_rt.len();
    let n_info_cols = precursor_frag_info.shape()[1];
    
    let mut columns = Vec::new();
    
    for (i, &rt) in all_rt.iter().enumerate() {
        let mut col_data = Vec::with_capacity(n_frags);
        for j in 0..n_frags {
            col_data.push(precursor_data[[j, i]] as f64);
        }
        columns.push(Series::new(&format!("{:.6}", rt), col_data));
    }
    
    let info_names = ["ProductMz", "LibraryIntensity", "frag_type", "FragmentType"];
    for col_idx in 0..n_info_cols {
        let mut col_data = Vec::with_capacity(n_frags);
        for row_idx in 0..n_frags {
            col_data.push(precursor_frag_info[[row_idx, col_idx]] as f64);
        }
        if col_idx < info_names.len() {
            columns.push(Series::new(info_names[col_idx], col_data));
        }
    }
    
    Ok(DataFrame::new(columns)?)
}

fn extract_ms2_fragments_for_ranges(
    all_data: &HashMap<String, TimsTOFData>,
    ms1_range_min: f64,
    ms1_range_max: f64,
    ms2_range_list: &Array3<f32>,
    precursor_idx: usize,
    n_fragments: usize,
) -> Result<Vec<TimsTOFData>, Box<dyn Error>> {
    let ms2_data = match all_data.get("ms2") {
        Some(data) => data,
        None => {
            return Ok(vec![TimsTOFData::new(); n_fragments]);
        }
    };
    
    let mut frag_results = Vec::new();
    
    for j in 0..n_fragments {
        let ms2_range_slice = ms2_range_list.slice(s![precursor_idx, j, ..]);
        
        if ms2_range_slice.len() < 2 {
            frag_results.push(TimsTOFData::new());
            continue;
        }
        
        let ms2_min_val = ms2_range_slice[0];
        let ms2_max_val = ms2_range_slice[1];
        
        let ms2_range_min = (ms2_min_val - 1.0) / 1000.0;
        let ms2_range_max = (ms2_max_val + 1.0) / 1000.0;
        
        if ms2_range_min <= 0.0 || ms2_range_max <= 0.0 || ms2_range_min >= ms2_range_max {
            frag_results.push(TimsTOFData::new());
            continue;
        }
        
        let mut fragment_data = TimsTOFData::new();
        
        for i in 0..ms2_data.mz_values.len() {
            let mz = ms2_data.mz_values[i];
            
            if mz >= ms2_range_min as f64 && mz <= ms2_range_max as f64 {
                fragment_data.rt_values_min.push(ms2_data.rt_values_min[i]);
                fragment_data.mobility_values.push(ms2_data.mobility_values[i]);
                fragment_data.mz_values.push(mz);
                fragment_data.intensity_values.push(ms2_data.intensity_values[i]);
                fragment_data.frame_indices.push(ms2_data.frame_indices[i]);
                fragment_data.scan_indices.push(ms2_data.scan_indices[i]);
            }
        }
        
        frag_results.push(fragment_data);
    }
    
    Ok(frag_results)
}

fn filter_by_im_range(data: &TimsTOFData, im_min: f64, im_max: f64) -> TimsTOFData {
    let mut filtered = TimsTOFData::new();
    
    for i in 0..data.mz_values.len() {
        let mobility = data.mobility_values[i];
        if mobility >= im_min && mobility <= im_max {
            filtered.rt_values_min.push(data.rt_values_min[i]);
            filtered.mobility_values.push(mobility);
            filtered.mz_values.push(data.mz_values[i]);
            filtered.intensity_values.push(data.intensity_values[i]);
            filtered.frame_indices.push(data.frame_indices[i]);
            filtered.scan_indices.push(data.scan_indices[i]);
        }
    }
    
    filtered
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("启动程序...");
    let program_start = Instant::now();
    
    // Create output directory
    let output_dir = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/20250711/rust_comparison_results";
    std::fs::create_dir_all(output_dir)?;
    
    // 读取库文件
    println!("\n步骤1: 读取库文件");
    let lib_file_path = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/lib/TPHPlib_frag1025_swissprot_final_all_from_Yueliang.tsv";
    let library_records = process_library_fast(lib_file_path)?;
    
    // 转换为DataFrame
    println!("\n步骤2: 转换库数据为DataFrame");
    let library_df = library_records_to_dataframe(library_records.clone())?;
    
    // 读取DIA-NN报告文件
    println!("\n步骤3: 读取DIA-NN报告文件");
    let report_file_path = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/report/report.parquet";
    let report_df = read_parquet_with_polars(report_file_path)?;
    
    // 合并数据
    println!("\n步骤4: 合并库数据和报告数据");
    let diann_result = merge_library_and_report(library_df, report_df)?;
    
    // 提取唯一前体ID
    println!("\n步骤5: 提取唯一前体ID");
    let diann_precursor_id_all = get_unique_precursor_ids(&diann_result)?;
    
    // 创建RT和IM字典
    println!("\n步骤6: 创建RT和IM查找字典");
    let (assay_rt_kept_dict, assay_im_kept_dict) = create_rt_im_dicts(&diann_precursor_id_all)?;
    
    // 筛选特定前体
    println!("\n步骤7: 筛选特定前体数据");
    let precursor_id_list = vec!["LLIYGASTR2".to_string()];
    let each_lib_data = filter_library_by_precursor_ids(&library_records, &precursor_id_list);
    
    if each_lib_data.is_empty() {
        println!("警告：没有找到匹配的前体数据");
        return Ok(());
    }
    
    // OUTPUT STEP 0: Save each_lib_data (library records for precursor)
    save_library_records_to_json(&each_lib_data, &format!("{}/old_version_step0_each_lib_data.json", output_dir))?;
    
    // 构建库矩阵
    println!("\n步骤8: 构建库矩阵");
    let lib_cols = LibCols::default();
    let (precursors_list, ms1_data_list, ms2_data_list, precursor_info_list) = 
        build_lib_matrix(&each_lib_data, &lib_cols, 5.0, 1801.0, 20)?;
    
    // OUTPUT STEP 0A: Save build_lib_matrix results
    save_precursors_list_to_json(&precursors_list, &format!("{}/old_version_step0a_precursors_list.json", output_dir))?;
    save_ms_data_list_to_json(&ms1_data_list, &format!("{}/old_version_step0a_ms1_data_list.json", output_dir))?;
    save_ms_data_list_to_json(&ms2_data_list, &format!("{}/old_version_step0a_ms2_data_list.json", output_dir))?;
    save_precursor_info_to_json(&precursor_info_list, &format!("{}/old_version_step0a_precursor_info_list.json", output_dir))?;
    
    // 构建前体特征矩阵
    println!("\n步骤9: 构建前体特征矩阵");
    let precursor_feat = create_precursor_feat(
        &precursor_info_list,
        &precursors_list,
        &assay_rt_kept_dict,
        &assay_im_kept_dict
    )?;
    
    // OUTPUT STEP 0F: Save precursor metadata
    save_dict_to_json(&assay_rt_kept_dict, &format!("{}/old_version_step0f_assay_rt_kept_dict.json", output_dir))?;
    save_dict_to_json(&assay_im_kept_dict, &format!("{}/old_version_step0f_assay_im_kept_dict.json", output_dir))?;
    
    // OUTPUT STEP 0G: Save precursor feature matrix
    save_precursor_feat_to_json(&precursor_feat, &format!("{}/old_version_step0g_precursor_feat.json", output_dir))?;
    
    // 构建张量
    println!("\n步骤10: 构建张量");
    let device = "cpu";
    let frag_repeat_num = 5;
    
    let (ms1_tensor, ms2_tensor) = build_precursors_matrix_step1(&ms1_data_list, &ms2_data_list, device)?;
    
    // OUTPUT STEP 0B: Save tensor step 1 results
    save_array3_to_json(&ms1_tensor, &format!("{}/old_version_step0b_ms1_tensor_step1.json", output_dir))?;
    save_array3_to_json(&ms2_tensor, &format!("{}/old_version_step0b_ms2_tensor_step1.json", output_dir))?;
    
    let ms2_tensor_processed = build_precursors_matrix_step2(ms2_tensor);
    
    // OUTPUT STEP 0C: Save tensor step 2 results
    save_array3_to_json(&ms2_tensor_processed, &format!("{}/old_version_step0c_ms2_tensor_step2.json", output_dir))?;
    
    let (ms1_range_list, ms2_range_list) = build_range_matrix_step3(
        &ms1_tensor, 
        &ms2_tensor_processed, 
        frag_repeat_num,
        "ppm",
        20.0,
        50.0,
        device
    )?;
    
    // OUTPUT STEP 0D: Save range matrices
    save_array3_to_json(&ms1_range_list, &format!("{}/old_version_step0d_ms1_range_list.json", output_dir))?;
    save_array3_to_json(&ms2_range_list, &format!("{}/old_version_step0d_ms2_range_list.json", output_dir))?;
    
    let (re_ms1_tensor, re_ms2_tensor, ms1_extract_width_range_list, ms2_extract_width_range_list) = 
        build_precursors_matrix_step3(
            &ms1_tensor,
            &ms2_tensor_processed,
            frag_repeat_num,
            "ppm",
            20.0,
            50.0,
            device
        )?;
    
    // OUTPUT STEP 0E: Save extract width range lists
    save_array3_to_json(&ms1_extract_width_range_list, &format!("{}/old_version_step0e_ms1_extract_width_range_list.json", output_dir))?;
    save_array3_to_json(&ms2_extract_width_range_list, &format!("{}/old_version_step0e_ms2_extract_width_range_list.json", output_dir))?;
    
    let frag_info = build_frag_info(
        &ms1_tensor,
        &ms2_tensor_processed,
        frag_repeat_num,
        device
    );
    
    // OUTPUT STEP 0H: Save fragment info
    println!("✓ Fragment info shape: {:?}", frag_info.shape());
    save_array3_to_json(&frag_info, &format!("{}/old_version_step0h_frag_info.json", output_dir))?;
    
    // 读取TimsTOF数据
    println!("\n步骤11: 读取TimsTOF数据");
    let bruker_d_folder_name = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d";
    let timstof_data = read_timstof_data(bruker_d_folder_name)?;
    
    // OUTPUT STEP 0I: Save TimsTOF data summary
    let timstof_summary = json!({
        "ms1_peaks_count": timstof_data.mz_values.len(),
        "ms1_sample_data": {
            "rt_values_min": timstof_data.rt_values_min.iter().take(10).collect::<Vec<_>>(),
            "mobility_values": timstof_data.mobility_values.iter().take(10).collect::<Vec<_>>(),
            "mz_values": timstof_data.mz_values.iter().take(10).collect::<Vec<_>>(),
            "intensity_values": timstof_data.intensity_values.iter().take(10).collect::<Vec<_>>()
        }
    });
    let mut file = File::create(format!("{}/old_version_step0i_timstof_summary.json", output_dir))?;
    file.write_all(serde_json::to_string_pretty(&timstof_summary)?.as_bytes())?;
    
    // 处理前体数据
    println!("\n步骤12: 处理前体数据");
    let i = 0;
    let im = precursor_feat[[i, 5]];
    let rt = precursor_feat[[i, 6]];
    
    // 计算m/z范围
    let ms1_range_slice = ms1_range_list.slice(s![i, .., ..]);
    let min_val = ms1_range_slice.iter()
        .filter(|&&v| v > 0.0)
        .fold(f64::INFINITY, |a, &b| a.min(b as f64));
    let max_val = ms1_range_slice.iter()
        .fold(f64::NEG_INFINITY, |a, &b| a.max(b as f64));
    
    let ms1_range_min = (min_val - 1.0) / 1000.0;
    let ms1_range_max = (max_val + 1.0) / 1000.0;
    
    // OUTPUT STEP 0J: Save m/z range calculation details
    let mz_range_details = json!({
        "precursor_index": i,
        "im": im,
        "rt": rt,
        "min_val": min_val,
        "max_val": max_val,
        "ms1_range_min": ms1_range_min,
        "ms1_range_max": ms1_range_max,
        "ms1_range_slice_sample": ms1_range_slice.slice(s![..3, ..]).to_owned().into_raw_vec(),
        "ms1_range_slice_shape": ms1_range_slice.shape().to_vec()
    });
    let mut file = File::create(format!("{}/old_version_step0j_mz_range_details.json", output_dir))?;
    file.write_all(serde_json::to_string_pretty(&mz_range_details)?.as_bytes())?;
    
    // 筛选和处理MS1数据
    let precursor_result = timstof_data.filter_by_mz_range(ms1_range_min, ms1_range_max);
    
    // OUTPUT STEP 1: Save MS1 precursor result
    save_timstof_data_to_json(&precursor_result, &format!("{}/old_version_step1_precursor_result.json", output_dir))?;
    
    let precursor_result_int = convert_mz_to_integer(&precursor_result);
    
    // OUTPUT STEP 2: Save MS1 integer converted result
    save_timstof_data_to_json(&precursor_result_int, &format!("{}/old_version_step2_precursor_result_int.json", output_dir))?;
    
    // IM过滤
    let im_tolerance = 0.05;
    let im_min = im - im_tolerance;
    let im_max = im + im_tolerance;
    let filtered_result = filter_by_im_range(&precursor_result_int, im_min, im_max);
    
    // OUTPUT STEP 3: Save MS1 IM filtered result
    save_timstof_data_to_json(&filtered_result, &format!("{}/old_version_step3_precursor_result_filtered.json", output_dir))?;
    
    // 读取完整MS数据
    println!("\n步骤13: 读取完整MS数据");
    let all_data = read_timstof_data_with_full_ms2(
        bruker_d_folder_name,
        ms1_range_min,
        ms1_range_max
    )?;
    
    // 提取MS2碎片
    println!("\n步骤14: 提取MS2碎片");
    let frag_results = extract_ms2_fragments_for_ranges(
        &all_data,
        ms1_range_min,
        ms1_range_max,
        &ms2_range_list,
        i,
        66,
    )?;
    
    // 处理MS2数据
    let merged_frag_data = TimsTOFData::merge(frag_results);
    
    // OUTPUT STEP 4: Save MS2 merged result
    save_timstof_data_to_json(&merged_frag_data, &format!("{}/old_version_step4_merged_frag_result.json", output_dir))?;
    
    let merged_frag_data_int = convert_mz_to_integer(&merged_frag_data);
    let filtered_frag_data = filter_by_im_range(&merged_frag_data_int, im_min, im_max);
    
    // OUTPUT STEP 5: Save MS2 filtered result
    save_timstof_data_to_json(&filtered_frag_data, &format!("{}/old_version_step5_frag_result_filtered.json", output_dir))?;
    
    // 构建Mask矩阵
    println!("\n步骤15: 构建Mask矩阵");
    let search_ms1_set: HashSet<i64> = filtered_result.mz_values.iter()
        .map(|&mz| mz as i64)
        .collect();
    let search_ms2_set: HashSet<i64> = filtered_frag_data.mz_values.iter()
        .map(|&mz| mz as i64)
        .collect();
    
    let ms1_extract_slice = ms1_extract_width_range_list.slice(s![i, .., ..]);
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
    
    let ms2_extract_slice = ms2_extract_width_range_list.slice(s![i, .., ..]);
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
    
    // OUTPUT STEP 6: Save mask matrices
    save_array2_to_json(&ms1_frag_moz_matrix, &format!("{}/old_version_step6_ms1_mask_matrix.json", output_dir))?;
    save_array2_to_json(&ms2_frag_moz_matrix, &format!("{}/old_version_step6_ms2_mask_matrix.json", output_dir))?;
    
    // 构建强度矩阵
    println!("\n步骤16: 构建强度矩阵");
    let mut all_rt_set = HashSet::new();
    
    for &rt in &filtered_result.rt_values_min {
        all_rt_set.insert((rt * 1e6) as i64);
    }
    
    for &rt in &filtered_frag_data.rt_values_min {
        all_rt_set.insert((rt * 1e6) as i64);
    }
    
    let mut all_rt_vec: Vec<f64> = all_rt_set.iter()
        .map(|&rt_int| rt_int as f64 / 1e6)
        .collect();
    all_rt_vec.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    let rt_value = precursor_feat[[i, 6]];
    let all_rt = get_rt_list(all_rt_vec, rt_value);
    
    // OUTPUT STEP 7: Save RT list
    save_vec_f64_to_json(&all_rt, &format!("{}/old_version_step7_rt_list.json", output_dir))?;
    
    let ms1_frag_rt_matrix = build_intensity_matrix_optimized_parallel(
        &filtered_result,
        &ms1_extract_width_range_list.slice(s![i, .., ..]).to_owned(),
        &ms1_frag_moz_matrix,
        &all_rt,
    )?;
    
    let ms2_frag_rt_matrix = build_intensity_matrix_optimized_parallel(
        &filtered_frag_data,
        &ms2_extract_width_range_list.slice(s![i, .., ..]).to_owned(),
        &ms2_frag_moz_matrix,
        &all_rt,
    )?;
    
    // OUTPUT STEP 8: Save intensity matrices
    save_array2_to_json(&ms1_frag_rt_matrix, &format!("{}/old_version_step8_ms1_intensity_matrix.json", output_dir))?;
    save_array2_to_json(&ms2_frag_rt_matrix, &format!("{}/old_version_step8_ms2_intensity_matrix.json", output_dir))?;
    
    // 重塑和合并矩阵
    println!("\n步骤17: 重塑和合并矩阵");
    let (ms1_rows, ms1_cols) = {
        let shape = ms1_frag_rt_matrix.shape();
        (shape[0], shape[1])
    };
    let ms1_reshaped = ms1_frag_rt_matrix.into_shape((
        frag_repeat_num,
        ms1_rows / frag_repeat_num,
        ms1_cols
    ))?;
    
    let (ms2_rows, ms2_cols) = {
        let shape = ms2_frag_rt_matrix.shape();
        (shape[0], shape[1])
    };
    let ms2_reshaped = ms2_frag_rt_matrix.into_shape((
        frag_repeat_num,
        ms2_rows / frag_repeat_num,
        ms2_cols
    ))?;
    
    // OUTPUT STEP 9: Save reshaped matrices
    save_array3_to_json(&ms1_reshaped, &format!("{}/old_version_step9_ms1_reshaped.json", output_dir))?;
    save_array3_to_json(&ms2_reshaped, &format!("{}/old_version_step9_ms2_reshaped.json", output_dir))?;
    
    let ms1_frags = ms1_reshaped.shape()[1];
    let ms2_frags = ms2_reshaped.shape()[1];
    let total_frags = ms1_frags + ms2_frags;
    let n_rt = all_rt.len();
    
    let mut full_frag_rt_matrix = Array3::<f32>::zeros((frag_repeat_num, total_frags, n_rt));
    
    for rep in 0..frag_repeat_num {
        for frag in 0..ms1_frags {
            for rt in 0..n_rt {
                full_frag_rt_matrix[[rep, frag, rt]] = ms1_reshaped[[rep, frag, rt]];
            }
        }
    }
    
    for rep in 0..frag_repeat_num {
        for frag in 0..ms2_frags {
            for rt in 0..n_rt {
                full_frag_rt_matrix[[rep, ms1_frags + frag, rt]] = ms2_reshaped[[rep, frag, rt]];
            }
        }
    }
    
    // OUTPUT STEP 10: Save combined matrix
    save_array3_to_json(&full_frag_rt_matrix, &format!("{}/old_version_step10_combined_matrix.json", output_dir))?;
    
    let rsm_matrix = full_frag_rt_matrix.insert_axis(Axis(0));
    
    // 创建最终数据框
    println!("\n步骤18: 创建最终数据框");
    let aggregated_x_sum = rsm_matrix.sum_axis(Axis(1));
    
    // OUTPUT STEP 11: Save aggregated matrix
    let aggregated_2d = aggregated_x_sum.slice(s![0, .., ..]).to_owned();
    save_array2_to_json(&aggregated_2d, &format!("{}/old_version_step11_aggregated_matrix.json", output_dir))?;
    
    let final_df = create_final_dataframe(
        &rsm_matrix,
        &frag_info,
        &all_rt,
        0,
    )?;
    
    // 导出最终结果
    println!("\n步骤19: 导出最终结果");
    let output_path = format!("{}/old_version_step12_final_dataframe.csv", output_dir);
    let mut df_file = File::create(output_path)?;
    CsvWriter::new(&mut df_file)
        .include_header(true)
        .finish(&mut final_df.clone())?;
    
    let program_total_time = program_start.elapsed();
    println!("\n程序执行完成！总运行时间: {:.2}秒", program_total_time.as_secs_f64());
    
    println!("\n📁 老版本输出文件已保存到: {}", output_dir);
    println!("   - Step 0: old_version_step0_each_lib_data.json");
    println!("   - Step 0A: old_version_step0a_precursors_list.json, old_version_step0a_ms1_data_list.json, old_version_step0a_ms2_data_list.json, old_version_step0a_precursor_info_list.json");
    println!("   - Step 0B: old_version_step0b_ms1_tensor_step1.json, old_version_step0b_ms2_tensor_step1.json");
    println!("   - Step 0C: old_version_step0c_ms2_tensor_step2.json");
    println!("   - Step 0D: old_version_step0d_ms1_range_list.json, old_version_step0d_ms2_range_list.json");
    println!("   - Step 0E: old_version_step0e_ms1_extract_width_range_list.json, old_version_step0e_ms2_extract_width_range_list.json");
    println!("   - Step 0F: old_version_step0f_assay_rt_kept_dict.json, old_version_step0f_assay_im_kept_dict.json, old_version_step0f_precursor_info_choose.json");
    println!("   - Step 0G: old_version_step0g_precursor_feat.json");
    println!("   - Step 0H: old_version_step0h_frag_info.json");
    println!("   - Step 0I: old_version_step0i_timstof_summary.json");
    println!("   - Step 0J: old_version_step0j_mz_range_details.json");
    println!("   - Step 1: old_version_step1_precursor_result.json");
    println!("   - Step 2: old_version_step2_precursor_result_int.json");
    println!("   - Step 3: old_version_step3_precursor_result_filtered.json");
    println!("   - Step 4: old_version_step4_merged_frag_result.json");
    println!("   - Step 5: old_version_step5_frag_result_filtered.json");
    println!("   - Step 6: old_version_step6_ms1_mask_matrix.json, old_version_step6_ms2_mask_matrix.json");
    println!("   - Step 7: old_version_step7_rt_list.json");
    println!("   - Step 8: old_version_step8_ms1_intensity_matrix.json, old_version_step8_ms2_intensity_matrix.json");
    println!("   - Step 9: old_version_step9_ms1_reshaped.json, old_version_step9_ms2_reshaped.json");
    println!("   - Step 10: old_version_step10_combined_matrix.json");
    println!("   - Step 11: old_version_step11_aggregated_matrix.json");
    println!("   - Step 12: old_version_step12_final_dataframe.csv");
    
    Ok(())
}