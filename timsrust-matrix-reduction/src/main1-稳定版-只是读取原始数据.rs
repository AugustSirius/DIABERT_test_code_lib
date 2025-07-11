use std::error::Error;
use std::path::Path;
use std::time::Instant;
use rayon::prelude::*;
use timsrust::readers::{FrameReader, MetadataReader};
use timsrust::converters::ConvertableDomain;
use timsrust::MSLevel;

// 定义TimsTOF数据结构体
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
        
        for i in 0..self.mz_values.len() {
            let mz = self.mz_values[i];
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
    
    /// 转换为DataFrame用于展示
    pub fn to_dataframe(&self) -> Result<(), Box<dyn Error>> {
        // 在实际应用中，这里会使用polars创建DataFrame
        // 但为了简化，我们只打印统计信息
        self.print_stats();
        Ok(())
    }
    
    /// 打印数据统计信息
    pub fn print_stats(&self) {
        if self.mz_values.is_empty() {
            println!("\n数据为空");
            return;
        }
        
        let min_mz = self.mz_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_mz = self.mz_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_rt = self.rt_values_min.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_rt = self.rt_values_min.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_im = self.mobility_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_im = self.mobility_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let min_intensity = self.intensity_values.iter().min().copied().unwrap_or(0);
        let max_intensity = self.intensity_values.iter().max().copied().unwrap_or(0);
        
        println!("\n数据统计:");
        println!("  数据点数: {}", self.mz_values.len());
        println!("  m/z范围: {:.4} - {:.4}", min_mz, max_mz);
        println!("  RT范围: {:.2} - {:.2} 分钟", min_rt, max_rt);
        println!("  IM范围: {:.4} - {:.4}", min_im, max_im);
        println!("  强度范围: {} - {}", min_intensity, max_intensity);
    }
}

// 辅助函数：找到数据点所属的扫描
// fn find_scan_for_index(index: usize, scan_offsets: &[usize]) -> usize {
//     for (scan, window) in scan_offsets.windows(2).enumerate() {
//         if index >= window[0] && index < window[1] {
//             return scan;
//         }
//     }
//     scan_offsets.len() - 1
// }

// 目前是 O(#scan_offsets) ≈ 4000 的线性搜索 → 每个 peak 要执行一次。
// 重写为二分：
fn find_scan_for_index(idx: usize, scan_offsets: &[usize]) -> usize {
    scan_offsets.binary_search(&idx).unwrap_or_else(|i| i - 1)
}

/// 读取Bruker .d文件并提取MS1数据
fn read_timstof_data_ms1(bruker_d_folder_path: &str) -> Result<TimsTOFData, Box<dyn Error>> {
    println!("\n开始读取TimsTOF数据文件: {}", bruker_d_folder_path);
    
    let tdf_path = Path::new(bruker_d_folder_path).join("analysis.tdf");
    let metadata = MetadataReader::new(&tdf_path)?;
    let mz_converter = metadata.mz_converter;
    let im_converter = metadata.im_converter;
    
    println!("元数据加载完成:");
    println!("  RT范围: {:.2} - {:.2} 秒 ({:.2} - {:.2} 分钟)", 
        metadata.lower_rt, metadata.upper_rt,
        metadata.lower_rt / 60.0, metadata.upper_rt / 60.0);
    println!("  IM范围: {:.2} - {:.2}", metadata.lower_im, metadata.upper_im);
    println!("  m/z范围: {:.2} - {:.2}", metadata.lower_mz, metadata.upper_mz);
    
    let frame_reader = FrameReader::new(bruker_d_folder_path)?;
    
    println!("\n开始并行处理MS1数据...");
    
    // 并行处理每一帧
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
    let mut timstof_data = TimsTOFData::new();
    for data in frame_results {
        timstof_data.rt_values_min.extend(data.rt_values_min);
        timstof_data.mobility_values.extend(data.mobility_values);
        timstof_data.mz_values.extend(data.mz_values);
        timstof_data.intensity_values.extend(data.intensity_values);
        timstof_data.frame_indices.extend(data.frame_indices);
        timstof_data.scan_indices.extend(data.scan_indices);
    }
    
    println!("\n数据读取完成:");
    println!("  MS1帧数: {}", ms1_count);
    println!("  提取的数据点数: {}", timstof_data.mz_values.len());
    
    Ok(timstof_data)
}

/// 读取完整的TimsTOF数据（包括MS1和MS2）
fn read_timstof_data_with_full_ms2(
    bruker_d_folder_path: &str,
    ms1_mz_min: f64,
    ms1_mz_max: f64,
) -> Result<Vec<TimsTOFData>, Box<dyn Error>> {
    println!("\n开始并行读取完整的TimsTOF数据（包括MS1和MS2）: {}", bruker_d_folder_path);
    
    let tdf_path = Path::new(bruker_d_folder_path).join("analysis.tdf");
    let metadata = MetadataReader::new(&tdf_path)?;
    let mz_converter = metadata.mz_converter;
    let im_converter = metadata.im_converter;
    
    let frame_reader = FrameReader::new(bruker_d_folder_path)?;
    
    // 并行处理所有帧
    let frame_results: Vec<(TimsTOFData, TimsTOFData)> = (0..frame_reader.len())
        .into_par_iter()
        .filter_map(|idx| frame_reader.get(idx).ok())
        .map(|frame| {
            let rt_min = frame.rt_in_seconds / 60.0;
            let mut ms1_frame_data = TimsTOFData::new();
            let mut ms2_frame_data = TimsTOFData::new();
            
            match frame.ms_level {
                MSLevel::MS1 => {
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
        .collect();
    
    // 分离并合并MS1和MS2数据
    let (ms1_data_vec, ms2_data_vec): (Vec<_>, Vec<_>) = frame_results.into_iter().unzip();
    
    let ms1_data = TimsTOFData::merge(ms1_data_vec);
    let ms2_data = TimsTOFData::merge(ms2_data_vec);
    
    println!("\n数据读取完成:");
    println!("  MS1数据点数: {}", ms1_data.mz_values.len());
    println!("  MS2数据点数: {}", ms2_data.mz_values.len());
    
    Ok(vec![ms1_data, ms2_data])
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("请提供Bruker .d文件夹路径作为参数");
        std::process::exit(1);
    }
    let bruker_d_path = &args[1];
    
    println!("开始读取Bruker .d文件: {}", bruker_d_path);
    let start = Instant::now();
    
    // // 读取MS1数据
    let ms1_data = read_timstof_data_ms1(bruker_d_path)?;
    ms1_data.print_stats();
    
    // 读取完整数据（MS1和MS2）
    let full_data = read_timstof_data_with_full_ms2(bruker_d_path, 0, 2400)?;
    
    println!("\n完整数据读取结果:");
    for (i, data) in full_data.iter().enumerate() {
        println!("\n数据集 {}:", i + 1);
        data.print_stats();
    }
    
    let duration = start.elapsed();
    println!("\n总耗时: {:.2}秒", duration.as_secs_f64());
    
    Ok(())
}