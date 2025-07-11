use std::error::Error;
use std::collections::HashMap;
use timsrust::readers::{FrameReader, MetadataReader};
use timsrust::converters::ConvertableDomain;
use timsrust::MSLevel;
use std::time::Instant;
use std::path::Path;
use rayon::prelude::*;

/// TimsTOF数据结构，用于存储从.d文件读取的数据
#[derive(Debug, Clone)]
pub struct TimsTOFData {
    pub rt_values_min: Vec<f64>,
    pub mobility_values: Vec<f64>,
    pub mz_values: Vec<f64>,
    pub intensity_values: Vec<u32>,
    pub frame_indices: Vec<usize>,
    pub scan_indices: Vec<usize>,
}

impl TimsTOFData {
    pub fn new() -> Self {
        TimsTOFData {
            rt_values_min: Vec::new(),
            mobility_values: Vec::new(),
            mz_values: Vec::new(),
            intensity_values: Vec::new(),
            frame_indices: Vec::new(),
            scan_indices: Vec::new(),
        }
    }
    
    /// 筛选指定m/z范围的数据
    pub fn filter_by_mz_range(&self, min_mz: f64, max_mz: f64) -> Self {
        let mut filtered = TimsTOFData::new();
        
        for (i, &mz) in self.mz_values.iter().enumerate() {
            if mz >= min_mz && mz <= max_mz {
                filtered.rt_values_min.push(self.rt_values_min[i]);
                filtered.mobility_values.push(self.mobility_values[i]);
                filtered.mz_values.push(mz);
                filtered.intensity_values.push(self.intensity_values[i]);
                filtered.frame_indices.push(self.frame_indices[i]);
                filtered.scan_indices.push(self.scan_indices[i]);
            }
        }
        
        filtered
    }
    
    /// 合并多个TimsTOFData实例
    pub fn merge(data_list: Vec<TimsTOFData>) -> Self {
        let mut merged = TimsTOFData::new();
        
        for data in data_list {
            merged.rt_values_min.extend(data.rt_values_min);
            merged.mobility_values.extend(data.mobility_values);
            merged.mz_values.extend(data.mz_values);
            merged.intensity_values.extend(data.intensity_values);
            merged.frame_indices.extend(data.frame_indices);
            merged.scan_indices.extend(data.scan_indices);
        }
        
        merged
    }
}

/// 辅助函数：找到数据点所属的扫描
fn find_scan_for_index(index: usize, scan_offsets: &[usize]) -> usize {
    for (scan, window) in scan_offsets.windows(2).enumerate() {
        if index >= window[0] && index < window[1] {
            return scan;
        }
    }
    scan_offsets.len() - 1
}

/// ===== 方法1: 原始串行方法（作为基准） =====
pub fn read_timstof_data_original(bruker_d_folder_path: &str) -> Result<TimsTOFData, Box<dyn Error>> {
    println!("\n[方法1 - 原始串行] 开始读取TimsTOF数据...");
    
    let tdf_path = Path::new(bruker_d_folder_path).join("analysis.tdf");
    let metadata = MetadataReader::new(&tdf_path)?;
    let mz_converter = metadata.mz_converter;
    let im_converter = metadata.im_converter;
    
    let frame_reader = FrameReader::new(bruker_d_folder_path)?;
    let mut timstof_data = TimsTOFData::new();
    
    let mut ms1_count = 0;
    let mut ms2_count = 0;
    
    for frame_idx in 0..frame_reader.len() {
        if let Ok(frame) = frame_reader.get(frame_idx) {
            match frame.ms_level {
                MSLevel::MS1 => {
                    ms1_count += 1;
                    let rt_min = frame.rt_in_seconds / 60.0;
                    
                    for (peak_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                        .zip(frame.intensities.iter())
                        .enumerate() 
                    {
                        let scan = find_scan_for_index(peak_idx, &frame.scan_offsets);
                        let mz = mz_converter.convert(tof as f64);
                        let im = im_converter.convert(scan as f64);
                        
                        timstof_data.rt_values_min.push(rt_min);
                        timstof_data.mobility_values.push(im);
                        timstof_data.mz_values.push(mz);
                        timstof_data.intensity_values.push(intensity);
                        timstof_data.frame_indices.push(frame.index);
                        timstof_data.scan_indices.push(scan);
                    }
                }
                MSLevel::MS2 => ms2_count += 1,
                _ => {}
            }
        }
    }
    
    println!("  MS1帧数: {}, MS2帧数: {}", ms1_count, ms2_count);
    println!("  提取的数据点数: {}", timstof_data.mz_values.len());
    
    Ok(timstof_data)
}

/// ===== 方法2: PyO3风格的并行过滤（借鉴自 timsrust_pyo3） =====
pub fn read_timstof_data_pyo3_style(bruker_d_folder_path: &str) -> Result<TimsTOFData, Box<dyn Error>> {
    println!("\n[方法2 - PyO3风格并行] 开始读取TimsTOF数据...");
    
    let tdf_path = Path::new(bruker_d_folder_path).join("analysis.tdf");
    let metadata = MetadataReader::new(&tdf_path)?;
    let mz_converter = metadata.mz_converter;
    let im_converter = metadata.im_converter;
    
    let frame_reader = FrameReader::new(bruker_d_folder_path)?;
    
    // 这是 timsrust_pyo3 中 read_ms1_frames 的实现方式
    let frame_results: Vec<TimsTOFData> = (0..frame_reader.len())
        .into_par_iter()
        .filter_map(|idx| frame_reader.get(idx).ok())
        .filter(|frame| frame.ms_level == MSLevel::MS1)
        .map(|frame| {
            let mut frame_data = TimsTOFData::new();
            let rt_min = frame.rt_in_seconds / 60.0;
            
            // 预分配容量以提高性能
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
    
    let ms1_count = frame_results.len();
    let timstof_data = TimsTOFData::merge(frame_results);
    
    println!("  MS1帧数: {}", ms1_count);
    println!("  提取的数据点数: {}", timstof_data.mz_values.len());
    
    Ok(timstof_data)
}

/// ===== 完整数据读取 - 原始方法 =====
pub fn read_full_data_original(
    bruker_d_folder_path: &str,
    ms1_mz_min: f64,
    ms1_mz_max: f64,
) -> Result<HashMap<String, TimsTOFData>, Box<dyn Error>> {
    println!("\n[完整数据-原始] 开始读取MS1+MS2数据...");
    
    let tdf_path = Path::new(bruker_d_folder_path).join("analysis.tdf");
    let metadata = MetadataReader::new(&tdf_path)?;
    let mz_converter = metadata.mz_converter;
    let im_converter = metadata.im_converter;
    
    let frame_reader = FrameReader::new(bruker_d_folder_path)?;
    
    let mut data_map: HashMap<String, TimsTOFData> = HashMap::new();
    data_map.insert("ms1".to_string(), TimsTOFData::new());
    data_map.insert("ms2".to_string(), TimsTOFData::new());
    
    let mut ms1_count = 0;
    let mut ms2_count = 0;
    
    for frame_idx in 0..frame_reader.len() {
        if let Ok(frame) = frame_reader.get(frame_idx) {
            let rt_min = frame.rt_in_seconds / 60.0;
            
            match frame.ms_level {
                MSLevel::MS1 => {
                    ms1_count += 1;
                    let ms1_data = data_map.get_mut("ms1").unwrap();
                    
                    for (peak_idx, (&tof, &intensity)) in frame.tof_indices.iter()
                        .zip(frame.intensities.iter())
                        .enumerate() 
                    {
                        let mz = mz_converter.convert(tof as f64);
                        
                        if mz >= ms1_mz_min && mz <= ms1_mz_max {
                            let scan = find_scan_for_index(peak_idx, &frame.scan_offsets);
                            let im = im_converter.convert(scan as f64);
                            
                            ms1_data.rt_values_min.push(rt_min);
                            ms1_data.mobility_values.push(im);
                            ms1_data.mz_values.push(mz);
                            ms1_data.intensity_values.push(intensity);
                            ms1_data.frame_indices.push(frame.index);
                            ms1_data.scan_indices.push(scan);
                        }
                    }
                }
                MSLevel::MS2 => {
                    ms2_count += 1;
                    let ms2_data = data_map.get_mut("ms2").unwrap();
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
                                
                                ms2_data.rt_values_min.push(rt_min);
                                ms2_data.mobility_values.push(im);
                                ms2_data.mz_values.push(mz);
                                ms2_data.intensity_values.push(intensity);
                                ms2_data.frame_indices.push(frame.index);
                                ms2_data.scan_indices.push(scan);
                            }
                        }
                    }
                }
                _ => {}
            }
        }
    }
    
    println!("  MS1帧数: {}, MS2帧数: {}", ms1_count, ms2_count);
    println!("  MS1数据点数: {}", data_map["ms1"].mz_values.len());
    println!("  MS2数据点数: {}", data_map["ms2"].mz_values.len());
    
    Ok(data_map)
}

/// ===== 完整数据读取 - PyO3风格优化 =====
pub fn read_full_data_pyo3_style(
    bruker_d_folder_path: &str,
    ms1_mz_min: f64,
    ms1_mz_max: f64,
) -> Result<HashMap<String, TimsTOFData>, Box<dyn Error>> {
    println!("\n[完整数据-PyO3风格] 开始并行读取MS1+MS2数据...");
    
    let tdf_path = Path::new(bruker_d_folder_path).join("analysis.tdf");
    let metadata = MetadataReader::new(&tdf_path)?;
    let mz_converter = metadata.mz_converter;
    let im_converter = metadata.im_converter;
    
    let frame_reader = FrameReader::new(bruker_d_folder_path)?;
    
    // 使用 parallel_filter 风格同时处理 MS1 和 MS2
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
    
    println!("  MS1数据点数: {}", data_map["ms1"].mz_values.len());
    println!("  MS2数据点数: {}", data_map["ms2"].mz_values.len());
    
    Ok(data_map)
}

/// 根据IM范围过滤TimsTOF数据
pub fn filter_by_im_range(data: &TimsTOFData, im_min: f64, im_max: f64) -> TimsTOFData {
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

/// 将m/z值转换为整数格式（乘以1000并向上取整）
pub fn convert_mz_to_integer(data: &TimsTOFData) -> TimsTOFData {
    let mut converted = TimsTOFData::new();
    
    converted.rt_values_min = data.rt_values_min.clone();
    converted.mobility_values = data.mobility_values.clone();
    converted.intensity_values = data.intensity_values.clone();
    converted.frame_indices = data.frame_indices.clone();
    converted.scan_indices = data.scan_indices.clone();
    
    converted.mz_values = data.mz_values.iter()
        .map(|&mz| (mz * 1000.0).ceil())
        .collect();
    
    converted
}

/// 打印数据统计信息
pub fn print_data_stats(data: &TimsTOFData, label: &str) {
    if data.mz_values.is_empty() {
        println!("  {}数据为空", label);
        return;
    }
    
    let min_mz = data.mz_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_mz = data.mz_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    
    println!("  {}统计:", label);
    println!("    数据点数: {}", data.mz_values.len());
    println!("    m/z范围: {:.4} - {:.4}", min_mz, max_mz);
}

/// 比较两个数据集的一致性
fn compare_data_consistency(data1: &TimsTOFData, data2: &TimsTOFData, label1: &str, label2: &str) -> bool {
    println!("\n比较 {} vs {}", label1, label2);
    
    if data1.mz_values.len() != data2.mz_values.len() {
        println!("  ❌ 数据点数不同: {} vs {}", 
            data1.mz_values.len(), data2.mz_values.len());
        return false;
    }
    
    // 检查前1000个数据点的一致性
    let n_compare = std::cmp::min(1000, data1.mz_values.len());
    let mut max_mz_diff: f64 = 0.0;
    let mut max_im_diff: f64 = 0.0;
    let mut intensity_mismatch = 0;
    
    for i in 0..n_compare {
        max_mz_diff = max_mz_diff.max((data1.mz_values[i] - data2.mz_values[i]).abs());
        max_im_diff = max_im_diff.max((data1.mobility_values[i] - data2.mobility_values[i]).abs());
        if data1.intensity_values[i] != data2.intensity_values[i] {
            intensity_mismatch += 1;
        }
    }
    
    let consistent = max_mz_diff < 1e-10 && max_im_diff < 1e-10 && intensity_mismatch == 0;
    
    if consistent {
        println!("  ✅ 数据完全一致");
    } else {
        println!("  ❌ 数据存在差异:");
        println!("    最大m/z差异: {:.10}", max_mz_diff);
        println!("    最大IM差异: {:.10}", max_im_diff);
        println!("    强度不匹配数: {}", intensity_mismatch);
    }
    
    println!("  数据点数: {}", data1.mz_values.len());
    
    consistent
}

fn main() -> Result<(), Box<dyn Error>> {
    let total_start = Instant::now();
    println!("========== TimsTOF 数据读取性能测试 (PyO3方法对比) ==========");
    
    // 设置文件路径
    let bruker_d_folder = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/DIA_sample.d";
    
    // 检查文件是否存在
    if !Path::new(bruker_d_folder).exists() {
        eprintln!("错误: 文件夹不存在: {}", bruker_d_folder);
        return Ok(());
    }
    
    // ========== MS1数据读取测试 ==========
    println!("\n========== MS1数据读取测试 ==========");
    
    // 方法1: 原始串行
    let start = Instant::now();
    let data_original = read_timstof_data_original(bruker_d_folder)?;
    let time_original = start.elapsed();
    
    // 方法2: PyO3风格并行
    let start = Instant::now();
    let data_pyo3 = read_timstof_data_pyo3_style(bruker_d_folder)?;
    let time_pyo3 = start.elapsed();
    
    // 性能对比
    println!("\n========== MS1读取性能对比 ==========");
    println!("原始方法: {:.3}秒", time_original.as_secs_f64());
    println!("PyO3风格: {:.3}秒 (加速比: {:.2}x)", 
        time_pyo3.as_secs_f64(),
        time_original.as_secs_f64() / time_pyo3.as_secs_f64());
    
    // 数据一致性检查
    println!("\n========== 数据一致性检查 ==========");
    let consistent = compare_data_consistency(&data_original, &data_pyo3, "原始方法", "PyO3风格");
    
    if !consistent {
        println!("\n⚠️  警告：数据一致性检查失败！");
    }
    
    // ========== 数据处理功能测试 ==========
    println!("\n========== 数据处理功能测试 ==========");
    
    // 使用PyO3方法的数据进行后续测试
    println!("\n使用PyO3风格方法的数据进行处理测试");
    
    // m/z范围过滤
    println!("\n1. m/z范围过滤测试");
    let filter_start = Instant::now();
    let ms1_mz_min = 400.0;
    let ms1_mz_max = 1200.0;
    let filtered_data = data_pyo3.filter_by_mz_range(ms1_mz_min, ms1_mz_max);
    let filter_time = filter_start.elapsed();
    println!("  过滤耗时: {:.3}秒", filter_time.as_secs_f64());
    println!("  过滤前: {} 点, 过滤后: {} 点", 
        data_pyo3.mz_values.len(), filtered_data.mz_values.len());
    println!("  保留比例: {:.1}%", 
        filtered_data.mz_values.len() as f64 / data_pyo3.mz_values.len() as f64 * 100.0);
    
    // m/z转换
    println!("\n2. m/z转换为整数测试");
    let convert_start = Instant::now();
    let integer_data = convert_mz_to_integer(&filtered_data);
    let convert_time = convert_start.elapsed();
    println!("  转换耗时: {:.3}秒", convert_time.as_secs_f64());
    if integer_data.mz_values.len() >= 5 {
        println!("  转换示例（前5个）:");
        for i in 0..5 {
            println!("    {:.4} -> {:.0}", 
                filtered_data.mz_values[i], 
                integer_data.mz_values[i]);
        }
    }
    
    // IM过滤
    println!("\n3. IM范围过滤测试");
    let im_start = Instant::now();
    let im_filtered = filter_by_im_range(&integer_data, 0.75, 0.85);
    let im_time = im_start.elapsed();
    println!("  过滤耗时: {:.3}秒", im_time.as_secs_f64());
    println!("  过滤前: {} 点, 过滤后: {} 点", 
        integer_data.mz_values.len(), im_filtered.mz_values.len());
    println!("  保留比例: {:.1}%", 
        im_filtered.mz_values.len() as f64 / integer_data.mz_values.len() as f64 * 100.0);
    
    // ========== 完整数据（MS1+MS2）读取测试 ==========
    println!("\n========== 完整数据（MS1+MS2）读取测试 ==========");
    
    // 原始方法
    let start = Instant::now();
    let full_data_original = read_full_data_original(bruker_d_folder, ms1_mz_min, ms1_mz_max)?;
    let time_full_original = start.elapsed();
    
    // PyO3风格方法
    let start = Instant::now();
    let full_data_pyo3 = read_full_data_pyo3_style(bruker_d_folder, ms1_mz_min, ms1_mz_max)?;
    let time_full_pyo3 = start.elapsed();
    
    println!("\n完整数据读取性能对比:");
    println!("  原始方法: {:.2}秒", time_full_original.as_secs_f64());
    println!("  PyO3风格: {:.2}秒 (加速比: {:.2}x)", 
        time_full_pyo3.as_secs_f64(),
        time_full_original.as_secs_f64() / time_full_pyo3.as_secs_f64());
    
    // 检查MS1和MS2数据一致性
    let ms1_consistent = compare_data_consistency(
        &full_data_original["ms1"], 
        &full_data_pyo3["ms1"], 
        "原始MS1", 
        "PyO3风格MS1"
    );
    
    let ms2_consistent = compare_data_consistency(
        &full_data_original["ms2"], 
        &full_data_pyo3["ms2"], 
        "原始MS2", 
        "PyO3风格MS2"
    );
    
    if !ms1_consistent || !ms2_consistent {
        println!("\n⚠️  警告：完整数据一致性检查失败！");
    }
    
    // ========== 最终总结 ==========
    let total_time = total_start.elapsed();
    println!("\n========== 最终总结 ==========");
    println!("总运行时间: {:.2}秒", total_time.as_secs_f64());
    
    println!("\n性能提升总结:");
    println!("  MS1读取: {:.2}x 加速", time_original.as_secs_f64() / time_pyo3.as_secs_f64());
    println!("  完整数据读取: {:.2}x 加速", time_full_original.as_secs_f64() / time_full_pyo3.as_secs_f64());
    
    if let Some(ms1_data) = full_data_pyo3.get("ms1") {
        let total_points = ms1_data.mz_values.len() + 
            full_data_pyo3.get("ms2").map(|d| d.mz_values.len()).unwrap_or(0);
        println!("\nPyO3风格方法处理速度:");
        println!("  总数据点数: {}", total_points);
        println!("  处理速度: {:.0} 点/秒", total_points as f64 / time_full_pyo3.as_secs_f64());
    }
    
    println!("\n✅ 程序执行完成！");
    
    Ok(())
}