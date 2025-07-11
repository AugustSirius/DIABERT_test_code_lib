use std::error::Error;
use std::path::Path;
use std::time::Instant;
use std::collections::HashMap;
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
    pub precursor_indices: Vec<usize>,  // 新增：用于区分MS1和MS2
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
            precursor_indices: Vec::new(),
        }
    }
    
    // 分离MS1和MS2数据
    pub fn separate_ms1_ms2(self) -> (TimsTOFData, TimsTOFData) {
        let mut ms1_data = TimsTOFData::new();
        let mut ms2_data = TimsTOFData::new();
        
        for i in 0..self.precursor_indices.len() {
            if self.precursor_indices[i] == 0 {
                // MS1数据
                ms1_data.rt_values_min.push(self.rt_values_min[i]);
                ms1_data.mobility_values.push(self.mobility_values[i]);
                ms1_data.mz_values.push(self.mz_values[i]);
                ms1_data.intensity_values.push(self.intensity_values[i]);
                ms1_data.frame_indices.push(self.frame_indices[i]);
                ms1_data.scan_indices.push(self.scan_indices[i]);
                ms1_data.precursor_indices.push(0);
            } else {
                // MS2数据
                ms2_data.rt_values_min.push(self.rt_values_min[i]);
                ms2_data.mobility_values.push(self.mobility_values[i]);
                ms2_data.mz_values.push(self.mz_values[i]);
                ms2_data.intensity_values.push(self.intensity_values[i]);
                ms2_data.frame_indices.push(self.frame_indices[i]);
                ms2_data.scan_indices.push(self.scan_indices[i]);
                ms2_data.precursor_indices.push(self.precursor_indices[i]);
            }
        }
        
        (ms1_data, ms2_data)
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

// 定义压缩参数
#[derive(Debug, Clone)]
pub struct BinningParameters {
    pub mz_bin_size: u32,
    pub mobility_bin_size: u32,
    pub rt_group_size: usize,
}

// 定义数据点结构体，用于压缩处理
#[derive(Debug, Clone)]
struct DataPoint {
    rt_values_min: f64,
    mobility_values: f64,
    mz_values: f64,
    intensity_values: u32,
    mz_bin_id: u32,
    im_bin_id: u32,
    rt_group_id: usize,
}

// 压缩函数
pub fn process_mixed_grid_data(
    data: &TimsTOFData,
    bin_params: &BinningParameters,
) -> TimsTOFData {
    println!("开始处理混合坐标数据（m/z, im离散; rt连续）...");
    
    if data.mz_values.is_empty() {
        println!("输入数据为空，返回空数据。");
        return TimsTOFData::new();
    }
    
    let start = Instant::now();
    
    // 步骤1: 创建数据点并计算bin ID
    println!("步骤1: 为m/z和im计算分箱ID...");
    let mut data_points: Vec<DataPoint> = data.mz_values
        .par_iter()
        .enumerate()
        .map(|(i, &mz)| {
            let mz_coord = (mz * 1000.0).ceil() as u32;
            let im_coord = (data.mobility_values[i] * 1000.0).ceil() as u32;
            
            DataPoint {
                rt_values_min: data.rt_values_min[i],
                mobility_values: data.mobility_values[i],
                mz_values: mz,
                intensity_values: data.intensity_values[i],
                mz_bin_id: mz_coord / bin_params.mz_bin_size,
                im_bin_id: im_coord / bin_params.mobility_bin_size,
                rt_group_id: 0, // 稍后计算
            }
        })
        .collect();
    
    // 步骤2: 按分箱ID和RT排序
    println!("步骤2: 在每个(m/z, im)箱内按rt排序并分组...");
    data_points.par_sort_unstable_by(|a, b| {
        a.mz_bin_id.cmp(&b.mz_bin_id)
            .then(a.im_bin_id.cmp(&b.im_bin_id))
            .then(a.rt_values_min.partial_cmp(&b.rt_values_min).unwrap())
    });
    
    // 计算RT组ID
    let mut current_mz_bin = u32::MAX;
    let mut current_im_bin = u32::MAX;
    let mut rt_counter = 0;
    
    for point in data_points.iter_mut() {
        if point.mz_bin_id != current_mz_bin || point.im_bin_id != current_im_bin {
            current_mz_bin = point.mz_bin_id;
            current_im_bin = point.im_bin_id;
            rt_counter = 0;
        }
        point.rt_group_id = rt_counter / bin_params.rt_group_size;
        rt_counter += 1;
    }
    
    // 步骤3: 聚合数据
    println!("步骤3: 按所有ID进行最终分组和聚合...");
    
    // 使用HashMap进行分组聚合
    let grouped_data: HashMap<(u32, u32, usize), Vec<&DataPoint>> = 
        data_points.iter()
            .fold(HashMap::new(), |mut acc, point| {
                let key = (point.mz_bin_id, point.im_bin_id, point.rt_group_id);
                acc.entry(key).or_insert_with(Vec::new).push(point);
                acc
            });
    
    // 并行计算聚合结果
    let aggregated: Vec<(f64, f64, f64, u32)> = grouped_data
        .par_iter()
        .filter_map(|(_, points)| {
            if points.is_empty() {
                return None;
            }
            
            // 计算总强度
            let total_intensity: u32 = points.iter()
                .map(|p| p.intensity_values)
                .sum();
            
            if total_intensity == 0 {
                return None;
            }
            
            // 计算中位数
            let mut mz_values: Vec<f64> = points.iter().map(|p| p.mz_values).collect();
            let mut rt_values: Vec<f64> = points.iter().map(|p| p.rt_values_min).collect();
            let mut im_values: Vec<f64> = points.iter().map(|p| p.mobility_values).collect();
            
            mz_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            rt_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            im_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
            
            let median_mz = median(&mz_values);
            let median_rt = median(&rt_values);
            let median_im = median(&im_values);
            
            Some((median_mz, median_rt, median_im, total_intensity))
        })
        .collect();
    
    // 构建结果
    let mut result = TimsTOFData::new();
    for (mz, rt, im, intensity) in aggregated {
        result.mz_values.push(mz.round());
        result.rt_values_min.push(rt);
        result.mobility_values.push(im.round());
        result.intensity_values.push(intensity);
        // 其他字段可根据需要填充
        result.frame_indices.push(0);
        result.scan_indices.push(0);
        result.precursor_indices.push(0);
    }
    
    let duration = start.elapsed();
    println!("处理完成！原始数据 {} 行，压缩后 {} 行。", 
             data.mz_values.len(), result.mz_values.len());
    println!("压缩率: {:.2}%", 
             (1.0 - result.mz_values.len() as f64 / data.mz_values.len() as f64) * 100.0);
    println!("压缩耗时: {:.2}秒", duration.as_secs_f64());
    
    result
}

// 计算中位数的辅助函数
fn median(values: &[f64]) -> f64 {
    let len = values.len();
    if len == 0 {
        return 0.0;
    }
    
    if len % 2 == 0 {
        (values[len / 2 - 1] + values[len / 2]) / 2.0
    } else {
        values[len / 2]
    }
}

// 二分查找函数
fn find_scan_for_index(idx: usize, scan_offsets: &[usize]) -> usize {
    scan_offsets.binary_search(&idx).unwrap_or_else(|i| i - 1)
}

// 读取完整的TimsTOF数据（包括MS1和MS2）
fn read_timstof_data_full(bruker_d_folder_path: &str) -> Result<TimsTOFData, Box<dyn Error>> {
    println!("\n开始读取完整的TimsTOF数据: {}", bruker_d_folder_path);
    
    let tdf_path = Path::new(bruker_d_folder_path).join("analysis.tdf");
    let metadata = MetadataReader::new(&tdf_path)?;
    let mz_converter = metadata.mz_converter;
    let im_converter = metadata.im_converter;
    
    let frame_reader = FrameReader::new(bruker_d_folder_path)?;
    
    // 并行处理所有帧
    let frame_results: Vec<TimsTOFData> = (0..frame_reader.len())
        .into_par_iter()
        .filter_map(|idx| frame_reader.get(idx).ok())
        .map(|frame| {
            let mut frame_data = TimsTOFData::new();
            let rt_min = frame.rt_in_seconds / 60.0;
            
            let precursor_index = match frame.ms_level {
                MSLevel::MS1 => 0,
                MSLevel::MS2 => 1, // 简化处理，实际应该根据具体的precursor信息
                _ => 0,
            };
            
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
                frame_data.precursor_indices.push(precursor_index);
            }
            
            frame_data
        })
        .collect();
    
    // 合并所有帧的数据
    let mut full_data = TimsTOFData::new();
    for data in frame_results {
        full_data.rt_values_min.extend(data.rt_values_min);
        full_data.mobility_values.extend(data.mobility_values);
        full_data.mz_values.extend(data.mz_values);
        full_data.intensity_values.extend(data.intensity_values);
        full_data.frame_indices.extend(data.frame_indices);
        full_data.scan_indices.extend(data.scan_indices);
        full_data.precursor_indices.extend(data.precursor_indices);
    }
    
    println!("数据读取完成，总数据点数: {}", full_data.mz_values.len());
    
    Ok(full_data)
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("请提供Bruker .d文件夹路径作为参数");
        std::process::exit(1);
    }
    let bruker_d_path = &args[1];
    
    println!("开始处理Bruker .d文件: {}", bruker_d_path);
    let total_start = Instant::now();
    
    // 读取完整数据
    let full_data = read_timstof_data_full(bruker_d_path)?;
    
    // 分离MS1和MS2数据
    println!("\n分离 MS1 和 MS2 数据...");
    let (ms1_data, ms2_data) = full_data.separate_ms1_ms2();
    println!("MS1 数据点数: {}", ms1_data.mz_values.len());
    println!("MS2 数据点数: {}", ms2_data.mz_values.len());
    
    // 定义压缩参数
    let bin_params = BinningParameters {
        mz_bin_size: 5,
        mobility_bin_size: 3,
        rt_group_size: 3,
    };
    
    // 分别压缩MS1和MS2数据
    println!("\n=== 处理 MS1 数据 ===");
    let ms1_compressed = process_mixed_grid_data(&ms1_data, &bin_params);
    
    println!("\n=== 处理 MS2 数据 ===");
    let ms2_compressed = process_mixed_grid_data(&ms2_data, &bin_params);
    
    // 输出压缩结果统计
    println!("\n=== 压缩结果统计 ===");
    println!("MS1 原始数据点: {}", ms1_data.mz_values.len());
    println!("MS1 压缩后数据点: {}", ms1_compressed.mz_values.len());
    println!("MS1 压缩率: {:.2}%", 
             (1.0 - ms1_compressed.mz_values.len() as f64 / ms1_data.mz_values.len() as f64) * 100.0);
    
    println!("\nMS2 原始数据点: {}", ms2_data.mz_values.len());
    println!("MS2 压缩后数据点: {}", ms2_compressed.mz_values.len());
    println!("MS2 压缩率: {:.2}%", 
             (1.0 - ms2_compressed.mz_values.len() as f64 / ms2_data.mz_values.len() as f64) * 100.0);
    
    let total_duration = total_start.elapsed();
    println!("\n总耗时: {:.2}秒", total_duration.as_secs_f64());
    
    Ok(())
}