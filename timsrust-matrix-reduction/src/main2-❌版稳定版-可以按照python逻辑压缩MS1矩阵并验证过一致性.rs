use std::collections::HashMap;
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

// 定义分箱参数结构体
#[derive(Debug, Clone)]
pub struct BinningParameters {
    pub mz_bin_size: i32,
    pub rt_bin_size: i32,
    pub mobility_bin_size: i32,
}

// 定义压缩后的数据点
#[derive(Debug, Clone)]
pub struct CompressedPoint {
    pub mz_value: f64,
    pub rt_value_min: f64,
    pub mobility_value: f64,
    pub intensity_value: u64,  // 使用u64以防止溢出
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
    
    /// 使用分箱压缩数据
    pub fn compress_with_binning(&self, bin_params: &BinningParameters) -> Vec<CompressedPoint> {
        println!("\n开始使用分箱方法压缩数据...");
        let start = Instant::now();
        
        if self.mz_values.is_empty() {
            println!("输入数据为空，返回空结果。");
            return Vec::new();
        }
        
        // 使用HashMap来累加相同分箱的强度
        let mut bin_map: HashMap<(i32, i32, i32), u64> = HashMap::new();
        
        // 处理每个数据点
        for i in 0..self.mz_values.len() {
            // 创建整数坐标
            let mz_coord = (self.mz_values[i] * 1000.0).ceil() as i32;
            let im_coord = (self.mobility_values[i] * 1000.0).ceil() as i32;
            let rt_coord = self.rt_values_min[i].round() as i32;
            
            // 计算分箱ID
            let mz_bin_id = mz_coord / bin_params.mz_bin_size;
            let rt_bin_id = rt_coord / bin_params.rt_bin_size;
            let im_bin_id = im_coord / bin_params.mobility_bin_size;
            
            let bin_key = (mz_bin_id, rt_bin_id, im_bin_id);
            
            // 累加强度
            *bin_map.entry(bin_key).or_insert(0) += self.intensity_values[i] as u64;
        }
        
        println!("分箱与聚合完成。发现 {} 个非空分箱。", bin_map.len());
        
        // 将HashMap转换为向量并计算中心坐标
        let mut result: Vec<CompressedPoint> = bin_map
            .into_iter()
            .filter(|(_, intensity)| *intensity > 0)  // 过滤掉强度为0的点
            .map(|((mz_bin_id, rt_bin_id, im_bin_id), intensity)| {
                // 计算每个维度的中心偏移量
                let mz_center_offset = bin_params.mz_bin_size / 2;
                let rt_center_offset = bin_params.rt_bin_size / 2;
                let im_center_offset = bin_params.mobility_bin_size / 2;
                
                // 计算新的中心坐标
                let new_mz_coord = mz_bin_id * bin_params.mz_bin_size + mz_center_offset;
                let new_rt_coord = rt_bin_id * bin_params.rt_bin_size + rt_center_offset;
                let new_im_coord = im_bin_id * bin_params.mobility_bin_size + im_center_offset;
                
                CompressedPoint {
                    mz_value: new_mz_coord as f64 / 1000.0,
                    rt_value_min: new_rt_coord as f64,
                    mobility_value: new_im_coord as f64 / 1000.0,
                    intensity_value: intensity,
                }
            })
            .collect();
        
        // 按照mz, rt, mobility的顺序排序
        result.sort_by(|a, b| {
            a.mz_value.partial_cmp(&b.mz_value).unwrap()
                .then(a.rt_value_min.partial_cmp(&b.rt_value_min).unwrap())
                .then(a.mobility_value.partial_cmp(&b.mobility_value).unwrap())
        });
        
        let duration = start.elapsed();
        println!("压缩完成！原始数据 {} 行，压缩后 {} 行。", self.mz_values.len(), result.len());
        println!("压缩率: {:.2}%", (1.0 - result.len() as f64 / self.mz_values.len() as f64) * 100.0);
        println!("压缩耗时: {:.2}秒", duration.as_secs_f64());
        
        result
    }
    
    /// 使用并行处理进行分箱压缩
    pub fn compress_with_binning_parallel(&self, bin_params: &BinningParameters) -> Vec<CompressedPoint> {
        println!("\n开始使用并行分箱方法压缩数据...");
        let start = Instant::now();
        
        if self.mz_values.is_empty() {
            println!("输入数据为空，返回空结果。");
            return Vec::new();
        }
        
        // 并行计算每个点的分箱键
        let bin_entries: Vec<((i32, i32, i32), u64)> = (0..self.mz_values.len())
            .into_par_iter()
            .map(|i| {
                // 创建整数坐标
                let mz_coord = (self.mz_values[i] * 1000.0).ceil() as i32;
                let im_coord = (self.mobility_values[i] * 1000.0).ceil() as i32;
                let rt_coord = self.rt_values_min[i].round() as i32;
                
                // 计算分箱ID
                let mz_bin_id = mz_coord / bin_params.mz_bin_size;
                let rt_bin_id = rt_coord / bin_params.rt_bin_size;
                let im_bin_id = im_coord / bin_params.mobility_bin_size;
                
                ((mz_bin_id, rt_bin_id, im_bin_id), self.intensity_values[i] as u64)
            })
            .collect();
        
        // 聚合相同分箱的强度
        let mut bin_map: HashMap<(i32, i32, i32), u64> = HashMap::new();
        for (bin_key, intensity) in bin_entries {
            *bin_map.entry(bin_key).or_insert(0) += intensity;
        }
        
        println!("分箱与聚合完成。发现 {} 个非空分箱。", bin_map.len());
        
        // 并行转换为最终结果
        let mut result: Vec<CompressedPoint> = bin_map
            .into_par_iter()
            .filter(|(_, intensity)| *intensity > 0)
            .map(|((mz_bin_id, rt_bin_id, im_bin_id), intensity)| {
                // 计算每个维度的中心偏移量
                let mz_center_offset = bin_params.mz_bin_size / 2;
                let rt_center_offset = bin_params.rt_bin_size / 2;
                let im_center_offset = bin_params.mobility_bin_size / 2;
                
                // 计算新的中心坐标
                let new_mz_coord = mz_bin_id * bin_params.mz_bin_size + mz_center_offset;
                let new_rt_coord = rt_bin_id * bin_params.rt_bin_size + rt_center_offset;
                let new_im_coord = im_bin_id * bin_params.mobility_bin_size + im_center_offset;
                
                CompressedPoint {
                    mz_value: new_mz_coord as f64 / 1000.0,
                    rt_value_min: new_rt_coord as f64,
                    mobility_value: new_im_coord as f64 / 1000.0,
                    intensity_value: intensity,
                }
            })
            .collect();
        
        // 排序
        result.par_sort_by(|a, b| {
            a.mz_value.partial_cmp(&b.mz_value).unwrap()
                .then(a.rt_value_min.partial_cmp(&b.rt_value_min).unwrap())
                .then(a.mobility_value.partial_cmp(&b.mobility_value).unwrap())
        });
        
        let duration = start.elapsed();
        println!("压缩完成！原始数据 {} 行，压缩后 {} 行。", self.mz_values.len(), result.len());
        println!("压缩率: {:.2}%", (1.0 - result.len() as f64 / self.mz_values.len() as f64) * 100.0);
        println!("压缩耗时: {:.2}秒", duration.as_secs_f64());
        
        result
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

// 打印压缩结果统计
fn print_compressed_stats(compressed_data: &[CompressedPoint]) {
    if compressed_data.is_empty() {
        println!("\n压缩数据为空");
        return;
    }
    
    let min_mz = compressed_data.iter().map(|p| p.mz_value).fold(f64::INFINITY, f64::min);
    let max_mz = compressed_data.iter().map(|p| p.mz_value).fold(f64::NEG_INFINITY, f64::max);
    let min_rt = compressed_data.iter().map(|p| p.rt_value_min).fold(f64::INFINITY, f64::min);
    let max_rt = compressed_data.iter().map(|p| p.rt_value_min).fold(f64::NEG_INFINITY, f64::max);
    let min_im = compressed_data.iter().map(|p| p.mobility_value).fold(f64::INFINITY, f64::min);
    let max_im = compressed_data.iter().map(|p| p.mobility_value).fold(f64::NEG_INFINITY, f64::max);
    let min_intensity = compressed_data.iter().map(|p| p.intensity_value).min().unwrap_or(0);
    let max_intensity = compressed_data.iter().map(|p| p.intensity_value).max().unwrap_or(0);
    let total_intensity: u64 = compressed_data.iter().map(|p| p.intensity_value).sum();
    
    println!("\n压缩数据统计:");
    println!("  数据点数: {}", compressed_data.len());
    println!("  m/z范围: {:.4} - {:.4}", min_mz, max_mz);
    println!("  RT范围: {:.2} - {:.2} 分钟", min_rt, max_rt);
    println!("  IM范围: {:.4} - {:.4}", min_im, max_im);
    println!("  强度范围: {} - {}", min_intensity, max_intensity);
    println!("  总强度: {}", total_intensity);
}

// 二分查找扫描索引
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

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("请提供Bruker .d文件夹路径作为参数");
        std::process::exit(1);
    }
    let bruker_d_path = &args[1];
    
    println!("开始读取Bruker .d文件: {}", bruker_d_path);
    let start = Instant::now();
    
    // 读取MS1数据
    let ms1_data = read_timstof_data_ms1(bruker_d_path)?;
    ms1_data.print_stats();
    
    // 定义分箱参数（与Python代码相同）
    let binning_params = BinningParameters {
        mz_bin_size: 5,
        rt_bin_size: 3,
        mobility_bin_size: 3,
    };
    
    println!("\n分箱参数:");
    println!("  m/z分箱大小: {}", binning_params.mz_bin_size);
    println!("  RT分箱大小: {}", binning_params.rt_bin_size);
    println!("  Mobility分箱大小: {}", binning_params.mobility_bin_size);
    
    // 执行压缩
    let compressed_data = ms1_data.compress_with_binning_parallel(&binning_params);
    
    // 打印压缩后的统计信息
    print_compressed_stats(&compressed_data);
    
    // 可选：打印前10个压缩后的数据点作为示例
    println!("\n前10个压缩后的数据点:");
    for (i, point) in compressed_data.iter().take(10).enumerate() {
        println!("  {}: m/z={:.4}, RT={:.2}, IM={:.4}, Intensity={}", 
                 i+1, point.mz_value, point.rt_value_min, point.mobility_value, point.intensity_value);
    }
    
    let total_duration = start.elapsed();
    println!("\n总耗时: {:.2}秒", total_duration.as_secs_f64());
    
    Ok(())
}