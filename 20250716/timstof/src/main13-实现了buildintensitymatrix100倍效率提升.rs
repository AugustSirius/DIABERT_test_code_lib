use std::fs::File;
use std::error::Error;
use std::time::Instant;
use rayon::prelude::*;
use csv::{ReaderBuilder, Writer};
use polars::prelude::*;
use std::collections::{HashMap, HashSet};
use ndarray::{Array1, Array2, Array3, Array4, ArrayView1, ArrayView2, Axis, s};
use std::f64::{INFINITY, NAN};
use timsrust::readers::{FrameReader, MetadataReader};
use timsrust::converters::ConvertableDomain;
use timsrust::MSLevel;
use std::io::Write;
use std::path::Path;

// 设备类型枚举（虽然在Rust中我们只使用CPU，但保持接口一致）
#[derive(Debug, Clone, Copy)]
enum Device {
    Cpu,
}

impl Device {
    fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "cpu" => Device::Cpu,
            _ => Device::Cpu, // 默认使用CPU
        }
    }
}

// ========== 新增的常量定义 ==========
// 这些常量对应Python中的全局常量，用于定义MS数据的各种参数
const MS1_ISOTOPE_COUNT: usize = 6;      // MS1同位素峰的数量
const FRAGMENT_VARIANTS: usize = 3;       // 每个碎片的变体数量（原始、轻、重）
const MS1_TYPE_MARKER: f64 = 5.0;        // MS1类型标识符
const MS1_FRAGMENT_TYPE: f64 = 1.0;      // MS1碎片类型
const VARIANT_ORIGINAL: f64 = 2.0;       // 原始碎片标识
const VARIANT_LIGHT: f64 = 3.0;          // 轻同位素标识
const VARIANT_HEAVY: f64 = 4.0;          // 重同位素标识

// ========== 新增的数据结构 ==========
// 定义库列名映射结构体，对应Python中的lib_cols字典
#[derive(Debug, Clone)]
struct LibCols {
    precursor_mz_col: &'static str,
    irt_col: &'static str,
    precursor_id_col: &'static str,
    full_sequence_col: &'static str,
    pure_sequence_col: &'static str,
    precursor_charge_col: &'static str,
    fragment_mz_col: &'static str,
    fragment_series_col: &'static str,
    fragment_charge_col: &'static str,
    fragment_type_col: &'static str,
    lib_intensity_col: &'static str,
    protein_name_col: &'static str,
    decoy_or_not_col: &'static str,
}

// 实现默认的LibCols，对应Python中的lib_cols字典
impl Default for LibCols {
    fn default() -> Self {
        LibCols {
            precursor_mz_col: "PrecursorMz",
            irt_col: "Tr_recalibrated",
            precursor_id_col: "transition_group_id",
            full_sequence_col: "FullUniModPeptideName",
            pure_sequence_col: "PeptideSequence",
            precursor_charge_col: "PrecursorCharge",
            fragment_mz_col: "ProductMz",
            fragment_series_col: "FragmentNumber",
            fragment_charge_col: "FragmentCharge",
            fragment_type_col: "FragmentType",
            lib_intensity_col: "LibraryIntensity",
            protein_name_col: "ProteinName",
            decoy_or_not_col: "decoy",
        }
    }
}

// 用于存储MS数据的类型别名
// 在Python中使用numpy数组，在Rust中我们使用Vec<Vec<f64>>来表示二维数组
type MSDataArray = Vec<Vec<f64>>;

#[derive(Debug, Clone)]
struct LibraryRecord {
    transition_group_id: String, // 转换组ID
    peptide_sequence: String, // 肽段序列
    full_unimod_peptide_name: String, // 完整的UniMod肽段名称
    precursor_charge: String, // 前体电荷
    precursor_mz: String, // 前体质荷比
    tr_recalibrated: String, // 重新校准的保留时间
    product_mz: String, // 产物质荷比
    fragment_type: String, // 片段类型
    fragment_charge: String, // 片段电荷
    fragment_number: String, // 片段编号
    library_intensity: String, // 库强度
    protein_id: String, // 蛋白质ID
    protein_name: String, // 蛋白质名称
    gene: String, // 基因名
    decoy: String, // 诱饵标记
    other_columns: HashMap<String, String>, // 存储其他未映射的列
}

/// ========== 优化的TimsTOF数据结构（借鉴PyO3） ==========
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
    
    /// 转换为DataFrame用于展示（支持整数m/z值）
    fn to_dataframe(&self) -> PolarsResult<DataFrame> {
        // 检查m/z值是否都是整数（通过检查是否都能被1整除）
        let all_integers = self.mz_values.iter().all(|&mz| mz.fract() == 0.0);
        
        if all_integers {
            // 如果都是整数，转换为i64类型
            let mz_integers: Vec<i64> = self.mz_values.iter()
                .map(|&mz| mz as i64)
                .collect();
            
            let df = DataFrame::new(vec![
                Series::new("rt_values_min", &self.rt_values_min),
                Series::new("mobility_values", &self.mobility_values),
                Series::new("mz_values", mz_integers),  // 使用整数类型
                Series::new("intensity_values", self.intensity_values.iter().map(|&v| v as f64).collect::<Vec<_>>()),
            ])?;
            Ok(df)
        } else {
            // 如果不都是整数，使用浮点数
            let df = DataFrame::new(vec![
                Series::new("rt_values_min", &self.rt_values_min),
                Series::new("mobility_values", &self.mobility_values),
                Series::new("mz_values", &self.mz_values),
                Series::new("intensity_values", self.intensity_values.iter().map(|&v| v as f64).collect::<Vec<_>>()),
            ])?;
            Ok(df)
        }
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

/// ========== 使用PyO3风格优化的TimsTOF数据读取函数 ==========

/// 读取Bruker .d文件并提取MS1数据（PyO3风格优化）
fn read_timstof_data(bruker_d_folder_path: &str) -> Result<TimsTOFData, Box<dyn Error>> {
    println!("\n[PyO3优化] 开始读取TimsTOF数据文件: {}", bruker_d_folder_path);
    
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
    
    // 使用PyO3风格的并行处理
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
    
    println!("\n数据读取完成:");
    println!("  MS1帧数: {}", ms1_count);
    println!("  提取的数据点数: {}", timstof_data.mz_values.len());
    
    Ok(timstof_data)
}

/// 处理MS1帧
fn process_ms1_frame(
    frame: &timsrust::Frame,
    rt_min: f64,
    ms1_mz_min: f64,
    ms1_mz_max: f64,
    mz_converter: &timsrust::converters::Tof2MzConverter,
    im_converter: &timsrust::converters::Scan2ImConverter,
    ms1_data: &mut TimsTOFData,
) {
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

/// 处理MS2帧（处理所有隔离窗口）
fn process_ms2_frame(
    frame: &timsrust::Frame,
    rt_min: f64,
    ms1_mz_min: f64,
    ms1_mz_max: f64,
    mz_converter: &timsrust::converters::Tof2MzConverter,
    im_converter: &timsrust::converters::Scan2ImConverter,
    ms2_windows: &mut HashMap<String, TimsTOFData>,
) {
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
        
        let window_key = format!("{:.2}_{:.2}", precursor_mz, isolation_width);
        let window_data = ms2_windows.entry(window_key).or_insert_with(TimsTOFData::new);
        
        for (peak_idx, (&tof, &intensity)) in frame.tof_indices.iter()
            .zip(frame.intensities.iter())
            .enumerate() 
        {
            let scan = find_scan_for_index(peak_idx, &frame.scan_offsets);
            
            if scan >= quad_settings.scan_starts[i] && scan <= quad_settings.scan_ends[i] {
                let mz = mz_converter.convert(tof as f64);
                let im = im_converter.convert(scan as f64);
                
                window_data.rt_values_min.push(rt_min);
                window_data.mobility_values.push(im);
                window_data.mz_values.push(mz);
                window_data.intensity_values.push(intensity);
                window_data.frame_indices.push(frame.index);
                window_data.scan_indices.push(scan);
            }
        }
    }
}

/// 读取完整的TimsTOF数据（包括MS1和MS2）- PyO3风格优化
fn read_timstof_data_with_full_ms2(
    bruker_d_folder_path: &str,
    ms1_mz_min: f64,
    ms1_mz_max: f64,
) -> Result<HashMap<String, TimsTOFData>, Box<dyn Error>> {
    println!("\n[PyO3优化] 开始并行读取完整的TimsTOF数据（包括MS1和MS2）: {}", bruker_d_folder_path);
    
    let tdf_path = Path::new(bruker_d_folder_path).join("analysis.tdf");
    let metadata = MetadataReader::new(&tdf_path)?;
    let mz_converter = metadata.mz_converter;
    let im_converter = metadata.im_converter;
    
    let frame_reader = FrameReader::new(bruker_d_folder_path)?;
    
    // 使用PyO3风格的并行处理
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
    
    println!("\n数据读取完成:");
    println!("  MS1数据点数: {}", data_map["ms1"].mz_values.len());
    println!("  MS2数据点数: {}", data_map["ms2"].mz_values.len());
    
    Ok(data_map)
}

// ========== 保持原有的其他函数不变 ==========

/// 获取RT列表：返回最接近目标值的48个RT值
fn get_rt_list(mut lst: Vec<f64>, target: f64) -> Vec<f64> {
    lst.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    
    if lst.is_empty() {
        return vec![0.0; 48];
    }
    
    if lst.len() <= 48 {
        let mut result = lst;
        result.resize(48, 0.0);
        return result;
    }
    
    let closest_idx = lst.iter()
        .enumerate()
        .min_by_key(|(_, &val)| ((val - target).abs() * 1e9) as i64)
        .map(|(idx, _)| idx)
        .unwrap_or(0);
    
    let start = if closest_idx >= 24 {
        (closest_idx - 24).min(lst.len() - 48)
    } else {
        0
    };
    
    lst[start..start + 48].to_vec()
}

/// 构建强度矩阵
fn build_intensity_matrix(
    data: &TimsTOFData,
    extract_width_range_list: &Array2<f32>,
    frag_moz_matrix: &Array2<f32>,
    all_rt: &[f64],
) -> Result<Array2<f32>, Box<dyn Error>> {
    let n_frags = extract_width_range_list.shape()[0];
    let n_rt = all_rt.len();
    
    let mut frag_rt_matrix = Array2::<f32>::zeros((n_frags, n_rt));
    
    for a in 0..n_frags {
        for (rt_idx, &rt) in all_rt.iter().enumerate() {
            let mut moz_to_intensity: HashMap<i64, f64> = HashMap::new();
            
            for i in 0..data.rt_values_min.len() {
                if (data.rt_values_min[i] - rt).abs() < 1e-6 {
                    let mz = data.mz_values[i] as i64;
                    let intensity = data.intensity_values[i] as f64;
                    *moz_to_intensity.entry(mz).or_insert(0.0) += intensity;
                }
            }
            
            let mut mapped_intensities = Array1::<f32>::zeros(extract_width_range_list.shape()[1]);
            for j in 0..extract_width_range_list.shape()[1] {
                let moz = extract_width_range_list[[a, j]] as i64;
                if let Some(&intensity) = moz_to_intensity.get(&moz) {
                    mapped_intensities[j] = intensity as f32;
                }
            }
            
            let frag_moz_row = frag_moz_matrix.slice(s![a, ..]);
            let intensity_sum: f32 = frag_moz_row.iter()
                .zip(mapped_intensities.iter())
                .map(|(&mask, &intensity)| mask * intensity)
                .sum();
            
            frag_rt_matrix[[a, rt_idx]] = intensity_sum;
        }
    }
    
    Ok(frag_rt_matrix)
}

// 优化后的构建强度矩阵函数 - 与Python pivot_table版本对应
fn build_intensity_matrix_optimized(
    data: &TimsTOFData,
    extract_width_range_list: &Array2<f32>,
    frag_moz_matrix: &Array2<f32>,
    all_rt: &[f64],
) -> Result<Array2<f32>, Box<dyn Error>> {
    use std::time::Instant;
    let start = Instant::now();
    
    let n_frags = extract_width_range_list.shape()[0];
    let n_rt = all_rt.len();
    
    println!("构建强度矩阵 - 优化版本（对应Python pivot_table）");
    println!("  碎片数: {}, RT数: {}", n_frags, n_rt);
    
    // Step 1: 收集所有唯一的mz值（与Python的pivot_table索引对应）
    let unique_mz_start = Instant::now();
    let mut unique_mz_set = HashSet::new();
    
    // 从数据中收集所有mz值
    for &mz in &data.mz_values {
        unique_mz_set.insert(mz as i64);
    }
    
    // 从extract_width_range_list中收集所有需要的mz值
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
    
    // 创建mz到索引的映射
    let mz_to_idx: HashMap<i64, usize> = unique_mz.iter()
        .enumerate()
        .map(|(idx, &mz)| (mz, idx))
        .collect();
    
    println!("  唯一mz值数量: {}", unique_mz.len());
    println!("  收集唯一mz值耗时: {:.3}秒", unique_mz_start.elapsed().as_secs_f64());
    
    // Step 2: 创建RT到索引的映射
    let rt_to_idx: HashMap<i64, usize> = all_rt.iter()
        .enumerate()
        .map(|(idx, &rt)| ((rt * 1e6) as i64, idx))
        .collect();
    
    // Step 3: 构建pivot表（mz × rt的强度矩阵）
    let pivot_start = Instant::now();
    let mut pivot_matrix = Array2::<f32>::zeros((unique_mz.len(), n_rt));
    
    // 填充pivot矩阵（对应Python的pivot_table操作）
    for i in 0..data.rt_values_min.len() {
        let rt_key = (data.rt_values_min[i] * 1e6) as i64;
        let mz = data.mz_values[i] as i64;
        let intensity = data.intensity_values[i] as f32;
        
        if let (Some(&rt_idx), Some(&mz_idx)) = (rt_to_idx.get(&rt_key), mz_to_idx.get(&mz)) {
            pivot_matrix[[mz_idx, rt_idx]] += intensity; // aggfunc='sum'
        }
    }
    
    println!("  构建pivot矩阵耗时: {:.3}秒", pivot_start.elapsed().as_secs_f64());
    
    // Step 4: 对每个碎片进行计算（对应Python的循环）
    let calc_start = Instant::now();
    let mut frag_rt_matrix = Array2::<f32>::zeros((n_frags, n_rt));
    
    for a in 0..n_frags {
        // 获取当前碎片需要的mz列表
        let mut moz_list: Vec<i64> = Vec::new();
        for j in 0..extract_width_range_list.shape()[1] {
            let mz = extract_width_range_list[[a, j]] as i64;
            moz_list.push(mz);
        }
        
        // 构建mz_rt矩阵（对应Python的reindex操作）
        let mut mz_rt_matrix = Array2::<f32>::zeros((moz_list.len(), n_rt));
        for (j, &mz) in moz_list.iter().enumerate() {
            if let Some(&mz_idx) = mz_to_idx.get(&mz) {
                for k in 0..n_rt {
                    mz_rt_matrix[[j, k]] = pivot_matrix[[mz_idx, k]];
                }
            }
            // 如果mz不在pivot_matrix中，保持为0（对应Python的fill_value=0）
        }
        
        // 执行矩阵乘法：frag_moz @ mz_rt（对应Python的矩阵乘法）
        for j in 0..moz_list.len() {
            for k in 0..n_rt {
                frag_rt_matrix[[a, k]] += frag_moz_matrix[[a, j]] * mz_rt_matrix[[j, k]];
            }
        }
    }
    
    println!("  矩阵计算耗时: {:.3}秒", calc_start.elapsed().as_secs_f64());
    println!("  总耗时: {:.3}秒", start.elapsed().as_secs_f64());
    
    // 打印一些统计信息用于验证
    let non_zero_count = frag_rt_matrix.iter().filter(|&&v| v > 0.0).count();
    println!("  结果矩阵非零元素: {} / {} ({:.2}%)", 
        non_zero_count, 
        frag_rt_matrix.len(),
        non_zero_count as f64 / frag_rt_matrix.len() as f64 * 100.0
    );
    
    Ok(frag_rt_matrix)
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
    
    println!("构建强度矩阵 - 并行优化版本");
    
    // Step 1-3: 与上面相同，构建pivot矩阵
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
    
    // Step 4: 并行计算每个碎片
    let results: Vec<Vec<f32>> = (0..n_frags)
        .into_par_iter()
        .map(|a| {
            let mut row_result = vec![0.0f32; n_rt];
            
            // 获取当前碎片需要的mz列表
            let moz_list: Vec<i64> = (0..extract_width_range_list.shape()[1])
                .map(|j| extract_width_range_list[[a, j]] as i64)
                .collect();
            
            // 执行矩阵乘法
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
    
    // 组装结果
    let mut frag_rt_matrix = Array2::<f32>::zeros((n_frags, n_rt));
    for (a, row) in results.into_iter().enumerate() {
        for (k, val) in row.into_iter().enumerate() {
            frag_rt_matrix[[a, k]] = val;
        }
    }
    
    println!("  并行计算完成，总耗时: {:.3}秒", start.elapsed().as_secs_f64());
    
    Ok(frag_rt_matrix)
}


/// 创建最终的数据框
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

// 对碎片列表进行排序并截取前max_length个
fn intercept_frags_sort(mut fragment_list: Vec<f64>, max_length: usize) -> Vec<f64> {
    fragment_list.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
    fragment_list.truncate(max_length);
    fragment_list
}

// 获取前体分组的索引
fn get_precursor_indices(precursor_ids: &[String]) -> Vec<Vec<usize>> {
    let mut precursor_indices = Vec::new();
    let mut current_group = Vec::new();
    let mut last_id = "";
    
    for (idx, id) in precursor_ids.iter().enumerate() {
        if idx == 0 || id == last_id {
            current_group.push(idx);
        } else {
            if !current_group.is_empty() {
                precursor_indices.push(current_group.clone());
                current_group.clear();
            }
            current_group.push(idx);
        }
        last_id = id;
    }
    
    if !current_group.is_empty() {
        precursor_indices.push(current_group);
    }
    
    precursor_indices
}

// 构建MS1数据
fn build_ms1_data(fragment_list: &[Vec<f64>], isotope_range: f64, max_mz: f64) -> MSDataArray {
    let first_fragment = &fragment_list[0];
    let charge = first_fragment[1];
    let precursor_mz = first_fragment[5];
    
    let available_range = (max_mz - precursor_mz) * charge;
    let iso_shift_max = (isotope_range.min(available_range) as i32) + 1;
    
    let mut isotope_mz_list: Vec<f64> = (0..iso_shift_max)
        .map(|iso_shift| precursor_mz + (iso_shift as f64) / charge)
        .collect();
    
    isotope_mz_list = intercept_frags_sort(isotope_mz_list, MS1_ISOTOPE_COUNT);
    
    let mut ms1_data = Vec::new();
    for mz in isotope_mz_list {
        let row = vec![
            mz,
            first_fragment[1],
            first_fragment[2],
            first_fragment[3],
            3.0,
            first_fragment[5],
            MS1_TYPE_MARKER,
            0.0,
            MS1_FRAGMENT_TYPE,
        ];
        ms1_data.push(row);
    }
    
    while ms1_data.len() < MS1_ISOTOPE_COUNT {
        ms1_data.push(vec![0.0; 9]);
    }
    
    ms1_data
}

// 构建MS2数据
fn build_ms2_data(fragment_list: &[Vec<f64>], max_fragment_num: usize) -> MSDataArray {
    let total_count = max_fragment_num * FRAGMENT_VARIANTS;
    let fragment_num = fragment_list.len();
    
    let mut tripled_fragments = Vec::new();
    for _ in 0..FRAGMENT_VARIANTS {
        for fragment in fragment_list {
            tripled_fragments.push(fragment.clone());
        }
    }
    
    let total_rows = fragment_num * FRAGMENT_VARIANTS;
    
    let mut type_column = vec![0.0; total_rows];
    for i in fragment_num..(fragment_num * 2) {
        type_column[i] = -1.0;
    }
    for i in (fragment_num * 2)..total_rows {
        type_column[i] = 1.0;
    }
    
    let window_id_column = vec![0.0; total_rows];
    
    let mut variant_type_column = vec![0.0; total_rows];
    for i in 0..fragment_num {
        variant_type_column[i] = VARIANT_ORIGINAL;
    }
    for i in fragment_num..(fragment_num * 2) {
        variant_type_column[i] = VARIANT_LIGHT;
    }
    for i in (fragment_num * 2)..total_rows {
        variant_type_column[i] = VARIANT_HEAVY;
    }
    
    let mut complete_data = Vec::new();
    for i in 0..total_rows {
        let mut row = tripled_fragments[i].clone();
        row.push(type_column[i]);
        row.push(window_id_column[i]);
        row.push(variant_type_column[i]);
        complete_data.push(row);
    }
    
    if complete_data.len() >= total_count {
        complete_data.truncate(total_count);
    } else {
        let row_size = if !complete_data.is_empty() { complete_data[0].len() } else { 9 };
        while complete_data.len() < total_count {
            complete_data.push(vec![0.0; row_size]);
        }
    }
    
    complete_data
}

// 构建前体信息
fn build_precursor_info(fragment_list: &[Vec<f64>]) -> Vec<f64> {
    let first_fragment = &fragment_list[0];
    vec![
        first_fragment[7],
        first_fragment[5],
        first_fragment[1],
        first_fragment[6],
        fragment_list.len() as f64,
        0.0,
    ]
}

// 格式化MS数据
fn format_ms_data(
    fragment_list: &[Vec<f64>], 
    isotope_range: f64, 
    max_mz: f64, 
    max_fragment: usize
) -> (MSDataArray, MSDataArray, Vec<f64>) {
    let ms1_data = build_ms1_data(fragment_list, isotope_range, max_mz);
    
    let fragment_list_subset: Vec<Vec<f64>> = fragment_list.iter()
        .map(|row| row[..6].to_vec())
        .collect();
    
    let mut ms2_data = build_ms2_data(&fragment_list_subset, max_fragment);
    
    let mut ms1_copy = ms1_data.clone();
    for row in &mut ms1_copy {
        if row.len() > 8 {
            row[8] = 5.0;
        }
    }
    
    ms2_data.extend(ms1_copy);
    
    let precursor_info = build_precursor_info(fragment_list);
    
    (ms1_data, ms2_data, precursor_info)
}

// 构建库矩阵 - 主函数
fn build_lib_matrix(
    lib_data: &[LibraryRecord],
    lib_cols: &LibCols,
    iso_range: f64,
    mz_max: f64,
    max_fragment: usize,
) -> Result<(Vec<Vec<String>>, Vec<MSDataArray>, Vec<MSDataArray>, Vec<Vec<f64>>), Box<dyn Error>> {
    println!("\n开始构建库矩阵...");
    println!("  同位素范围: {}", iso_range);
    println!("  最大质荷比: {}", mz_max);
    println!("  最大碎片数: {}", max_fragment);
    
    let precursor_ids: Vec<String> = lib_data.iter()
        .map(|record| record.transition_group_id.clone())
        .collect();
    
    let precursor_groups = get_precursor_indices(&precursor_ids);
    println!("  找到 {} 个前体组", precursor_groups.len());
    
    let mut all_precursors = Vec::new();
    let mut all_ms1_data = Vec::new();
    let mut all_ms2_data = Vec::new();
    let mut all_precursor_info = Vec::new();
    
    for (group_idx, indices) in precursor_groups.iter().enumerate() {
        if indices.is_empty() {
            continue;
        }
        
        let first_idx = indices[0];
        let first_record = &lib_data[first_idx];
        
        let precursor_info = vec![
            first_record.transition_group_id.clone(),
            first_record.decoy.clone(),
        ];
        all_precursors.push(precursor_info);
        
        let mut group_fragments = Vec::new();
        for &idx in indices {
            let record = &lib_data[idx];
            
            let fragment_row = vec![
                record.product_mz.parse::<f64>().unwrap_or(0.0),
                record.precursor_charge.parse::<f64>().unwrap_or(0.0),
                record.fragment_charge.parse::<f64>().unwrap_or(0.0),
                record.library_intensity.parse::<f64>().unwrap_or(0.0),
                record.fragment_type.parse::<f64>().unwrap_or(0.0),
                record.precursor_mz.parse::<f64>().unwrap_or(0.0),
                record.tr_recalibrated.parse::<f64>().unwrap_or(0.0),
                record.peptide_sequence.len() as f64,
                record.decoy.parse::<f64>().unwrap_or(0.0),
                record.transition_group_id.len() as f64,
            ];
            group_fragments.push(fragment_row);
        }
        
        let (ms1, ms2, info) = format_ms_data(&group_fragments, iso_range, mz_max, max_fragment);
        
        all_ms1_data.push(ms1);
        all_ms2_data.push(ms2);
        all_precursor_info.push(info);
        
        if (group_idx + 1) % 100 == 0 {
            println!("  已处理 {} 个前体组...", group_idx + 1);
        }
    }
    
    println!("库矩阵构建完成！");
    println!("  前体数量: {}", all_precursors.len());
    println!("  MS1数据数量: {}", all_ms1_data.len());
    println!("  MS2数据数量: {}", all_ms2_data.len());
    println!("  前体信息数量: {}", all_precursor_info.len());
    
    Ok((all_precursors, all_ms1_data, all_ms2_data, all_precursor_info))
}

// 辅助函数：打印MS数据用于调试
fn print_ms_data_sample(name: &str, ms_data: &[MSDataArray], max_items: usize) {
    println!("\n{}示例（前{}个）:", name, max_items.min(ms_data.len()));
    for (i, data) in ms_data.iter().take(max_items).enumerate() {
        println!("  {}[{}]: {} 行", name, i, data.len());
        if !data.is_empty() && !data[0].is_empty() {
            println!("    第一行: {:?}", data[0]);
            if data.len() > 1 {
                println!("    第二行: {:?}", data[1]);
            }
        }
    }
}

// [保持原有的其他函数不变...]
fn get_lib_col_dict() -> HashMap<&'static str, &'static str> {
    let mut lib_col_dict = HashMap::new();
    for key in ["transition_group_id", "PrecursorID"] { lib_col_dict.insert(key, "transition_group_id"); }
    for key in ["PeptideSequence", "Sequence", "StrippedPeptide"] { lib_col_dict.insert(key, "PeptideSequence"); }
    for key in ["FullUniModPeptideName", "ModifiedPeptide", "LabeledSequence", "modification_sequence", "ModifiedPeptideSequence"] { lib_col_dict.insert(key, "FullUniModPeptideName"); }
    for key in ["PrecursorCharge", "Charge", "prec_z"] { lib_col_dict.insert(key, "PrecursorCharge"); }
    for key in ["PrecursorMz", "Q1"] { lib_col_dict.insert(key, "PrecursorMz"); }
    for key in ["Tr_recalibrated", "iRT", "RetentionTime", "NormalizedRetentionTime", "RT_detected"] { lib_col_dict.insert(key, "Tr_recalibrated"); }
    for key in ["ProductMz", "FragmentMz", "Q3"] { lib_col_dict.insert(key, "ProductMz"); }
    for key in ["FragmentType", "FragmentIonType", "ProductType", "ProductIonType", "frg_type"] { lib_col_dict.insert(key, "FragmentType"); }
    for key in ["FragmentCharge", "FragmentIonCharge", "ProductCharge", "ProductIonCharge", "frg_z"] { lib_col_dict.insert(key, "FragmentCharge"); }
    for key in ["FragmentNumber", "frg_nr", "FragmentSeriesNumber"] { lib_col_dict.insert(key, "FragmentNumber"); }
    for key in ["LibraryIntensity", "RelativeIntensity", "RelativeFragmentIntensity", "RelativeFragmentIonIntensity", "relative_intensity"] { lib_col_dict.insert(key, "LibraryIntensity"); }
    for key in ["ProteinID", "ProteinId", "UniprotID", "uniprot_id", "UniProtIds"] { lib_col_dict.insert(key, "ProteinID"); }
    for key in ["ProteinName", "Protein Name", "Protein_name", "protein_name"] { lib_col_dict.insert(key, "ProteinName"); }
    for key in ["Gene", "Genes", "GeneName"] { lib_col_dict.insert(key, "Gene"); }
    for key in ["Decoy", "decoy"] { lib_col_dict.insert(key, "decoy"); }
    lib_col_dict
}

fn read_parquet_with_polars(file_path: &str) -> PolarsResult<DataFrame> {
    use std::time::Instant;
    let start = Instant::now();
    println!("正在使用Polars读取Parquet文件: {}", file_path);
    let file = File::open(file_path)?;
    let mut df = ParquetReader::new(file).finish()?;
    let new_col = df.column("Precursor.Id")?.clone().with_name("transition_group_id");
    df.with_column(new_col)?;
    let elapsed = start.elapsed();
    println!("Parquet文件读取完成！共 {} 条记录，耗时: {:.2}秒", df.height(), elapsed.as_secs_f64());
    println!("DataFrame列名: {:?}", df.get_column_names());
    Ok(df)
}

fn process_library_fast(file_path: &str) -> Result<Vec<LibraryRecord>, Box<dyn Error>> {
    use std::time::Instant;
    let start = Instant::now();
    println!("正在使用高性能模式读取文件: {}", file_path);
    let file = File::open(file_path)?;
    let mut reader = ReaderBuilder::new().delimiter(b'\t').has_headers(true).from_reader(file);
    let headers = reader.headers()?.clone();
    println!("成功读取表头，共 {} 列", headers.len());
    let mut column_indices = HashMap::new();
    for (i, header) in headers.iter().enumerate() {
        column_indices.insert(header, i);
    }
    let lib_col_dict = get_lib_col_dict();
    let mut mapped_indices: HashMap<&str, usize> = HashMap::new();
    for (old_col, new_col) in &lib_col_dict {
        if let Some(&idx) = column_indices.get(old_col) {
            mapped_indices.insert(new_col, idx);
        }
    }
    
    let fragment_number_idx = column_indices.get("FragmentNumber").copied();
    
    let mut byte_records = Vec::new();
    for result in reader.byte_records() {
        byte_records.push(result?);
    }
    let total_records = byte_records.len();
    println!("已读取 {} 条原始记录，开始并行处理...", total_records);
    let processed = std::sync::atomic::AtomicUsize::new(0);
    let records: Vec<LibraryRecord> = byte_records.par_iter().map(|record| {
        let current = processed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if current % 100000 == 0 && current > 0 {
            println!("已处理 {} 条记录...", current);
        }
        let mut rec = LibraryRecord {
            transition_group_id: String::new(),
            peptide_sequence: String::new(),
            full_unimod_peptide_name: String::new(),
            precursor_charge: String::new(),
            precursor_mz: String::new(),
            tr_recalibrated: String::new(),
            product_mz: String::new(),
            fragment_type: String::new(),
            fragment_charge: String::new(),
            fragment_number: String::new(),
            library_intensity: String::new(),
            protein_id: String::new(),
            protein_name: String::new(),
            gene: String::new(),
            decoy: "0".to_string(),
            other_columns: HashMap::new(),
        };
        if let Some(&idx) = mapped_indices.get("PeptideSequence") { if let Some(val) = record.get(idx) { rec.peptide_sequence = String::from_utf8_lossy(val).into_owned(); } }
        if let Some(&idx) = mapped_indices.get("FullUniModPeptideName") { if let Some(val) = record.get(idx) { rec.full_unimod_peptide_name = String::from_utf8_lossy(val).into_owned(); } }
        if let Some(&idx) = mapped_indices.get("PrecursorCharge") { if let Some(val) = record.get(idx) { rec.precursor_charge = String::from_utf8_lossy(val).into_owned(); } }
        if let Some(&idx) = mapped_indices.get("PrecursorMz") { if let Some(val) = record.get(idx) { rec.precursor_mz = String::from_utf8_lossy(val).into_owned(); } }
        if let Some(&idx) = mapped_indices.get("ProductMz") { if let Some(val) = record.get(idx) { rec.product_mz = String::from_utf8_lossy(val).into_owned(); } }
        if let Some(&idx) = mapped_indices.get("FragmentType") {
            if let Some(val) = record.get(idx) {
                let fragment_str = String::from_utf8_lossy(val);
                rec.fragment_type = match fragment_str.as_ref() { "b" => "1".to_string(), "y" => "2".to_string(), "p" => "3".to_string(), _ => fragment_str.into_owned() };
            }
        }
        if let Some(&idx) = mapped_indices.get("FragmentCharge") { if let Some(val) = record.get(idx) { rec.fragment_charge = String::from_utf8_lossy(val).into_owned(); } }
        if let Some(&idx) = mapped_indices.get("LibraryIntensity") { if let Some(val) = record.get(idx) { rec.library_intensity = String::from_utf8_lossy(val).into_owned(); } }
        if let Some(&idx) = mapped_indices.get("Tr_recalibrated") { if let Some(val) = record.get(idx) { rec.tr_recalibrated = String::from_utf8_lossy(val).into_owned(); } }
        if let Some(&idx) = mapped_indices.get("ProteinID") { if let Some(val) = record.get(idx) { rec.protein_id = String::from_utf8_lossy(val).into_owned(); } }
        if let Some(&idx) = mapped_indices.get("Gene") { if let Some(val) = record.get(idx) { rec.gene = String::from_utf8_lossy(val).into_owned(); } }
        if let Some(&idx) = mapped_indices.get("ProteinName") { if let Some(val) = record.get(idx) { rec.protein_name = String::from_utf8_lossy(val).into_owned(); } }
        
        if let Some(idx) = fragment_number_idx {
            if let Some(val) = record.get(idx) {
                rec.fragment_number = String::from_utf8_lossy(val).into_owned();
            }
        }
        
        rec.transition_group_id = format!("{}{}", rec.full_unimod_peptide_name, rec.precursor_charge);
        rec
    }).collect();
    let elapsed = start.elapsed();
    println!("处理完成！共 {} 条记录，耗时: {:.2}秒", records.len(), elapsed.as_secs_f64());
    Ok(records)
}

fn export_to_csv(records: &[LibraryRecord], output_path: &str) -> Result<(), Box<dyn Error>> {
    let file = File::create(output_path)?;
    let mut wtr = Writer::from_writer(file);
    wtr.write_record(&[
        "transition_group_id", "PeptideSequence", "FullUniModPeptideName", "PrecursorCharge", "PrecursorMz",
        "ProductMz", "FragmentType", "LibraryIntensity", "ProteinID", "Gene", "decoy"
    ])?;
    for record in records {
        wtr.write_record(&[
            &record.transition_group_id, &record.peptide_sequence, &record.full_unimod_peptide_name,
            &record.precursor_charge, &record.precursor_mz, &record.product_mz, &record.fragment_type,
            &record.library_intensity, &record.protein_id, &record.gene, &record.decoy,
        ])?;
    }
    wtr.flush()?;
    println!("结果已导出到: {}", output_path);
    Ok(())
}

fn export_polars_to_csv(df: &mut DataFrame, output_path: &str) -> PolarsResult<()> {
    let mut file = File::create(output_path)?;
    CsvWriter::new(&mut file).include_header(true).finish(df)?;
    println!("Polars DataFrame已导出到: {}", output_path);
    Ok(())
}

fn library_records_to_dataframe(records: Vec<LibraryRecord>) -> PolarsResult<DataFrame> {
    use std::time::Instant;
    let start = Instant::now();
    println!("正在将Library记录转换为DataFrame...");
    let mut transition_group_ids = Vec::with_capacity(records.len());
    let mut precursor_mzs = Vec::with_capacity(records.len());
    let mut product_mzs = Vec::with_capacity(records.len());
    for record in records {
        transition_group_ids.push(record.transition_group_id);
        precursor_mzs.push(record.precursor_mz.parse::<f64>().unwrap_or(f64::NAN));
        product_mzs.push(record.product_mz.parse::<f64>().unwrap_or(f64::NAN));
    }
    let df = DataFrame::new(vec![
        Series::new("transition_group_id", transition_group_ids),
        Series::new("PrecursorMz", precursor_mzs),
        Series::new("ProductMz", product_mzs),
    ])?;
    let elapsed = start.elapsed();
    println!("转换完成！DataFrame包含 {} 行，耗时: {:.2}秒", df.height(), elapsed.as_secs_f64());
    Ok(df)
}

fn merge_library_and_report(library_df: DataFrame, report_df: DataFrame) -> PolarsResult<DataFrame> {
    use std::time::Instant;
    let start = Instant::now();
    println!("\n开始执行merge操作...");
    println!("Library DataFrame: {} 行", library_df.height());
    println!("Report DataFrame: {} 行", report_df.height());
    let report_selected = report_df.select(["transition_group_id", "RT", "IM", "iIM"])?;
    let merged = library_df.join(&report_selected, ["transition_group_id"], ["transition_group_id"], JoinArgs::new(JoinType::Left))?;
    println!("Merge完成，结果包含 {} 行", merged.height());
    let rt_col = merged.column("RT")?;
    let mask = rt_col.is_not_null();
    let filtered = merged.filter(&mask)?;
    let elapsed = start.elapsed();
    println!("过滤完成！最终结果包含 {} 行，耗时: {:.2}秒", filtered.height(), elapsed.as_secs_f64());
    println!("最终DataFrame列名: {:?}", filtered.get_column_names());
    let reordered = filtered.select(["transition_group_id", "PrecursorMz", "ProductMz", "RT", "IM", "iIM"])?;
    Ok(reordered)
}

fn get_unique_precursor_ids(diann_result: &DataFrame) -> PolarsResult<DataFrame> {
    use std::time::Instant;
    let start = Instant::now();
    println!("\n开始提取唯一的前体ID数据...");
    println!("输入DataFrame: {} 行", diann_result.height());
    let unique_df = diann_result.unique(Some(&["transition_group_id".to_string()]), UniqueKeepStrategy::First, None)?;
    println!("去重后剩余 {} 行", unique_df.height());
    let selected_df = unique_df.select(["transition_group_id", "RT", "IM"])?;
    let elapsed = start.elapsed();
    println!("提取完成！最终结果包含 {} 行，耗时: {:.2}秒", selected_df.height(), elapsed.as_secs_f64());
    println!("\n前10条唯一前体ID记录:");
    println!("{}", selected_df.head(Some(10)));
    Ok(selected_df)
}

fn create_rt_im_dicts(df: &DataFrame) -> PolarsResult<(HashMap<String, f64>, HashMap<String, f64>)> {
    use std::time::Instant;
    let start = Instant::now();
    println!("\n开始创建RT和IM查找字典...");
    println!("DataFrame schema:");
    for (name, dtype) in df.schema().iter() { println!("  列 '{}': {:?}", name, dtype); }
    let id_col = df.column("transition_group_id")?;
    let id_vec = id_col.str()?.into_iter().map(|opt| opt.unwrap_or("").to_string()).collect::<Vec<String>>();
    let rt_col = df.column("RT")?;
    let rt_vec: Vec<f64> = match rt_col.dtype() {
        DataType::Float32 => rt_col.f32()?.into_iter().map(|opt| opt.map(|v| v as f64).unwrap_or(f64::NAN)).collect(),
        DataType::Float64 => rt_col.f64()?.into_iter().map(|opt| opt.unwrap_or(f64::NAN)).collect(),
        _ => return Err(PolarsError::SchemaMismatch(format!("RT列的类型不是浮点数: {:?}", rt_col.dtype()).into())),
    };
    let im_col = df.column("IM")?;
    let im_vec: Vec<f64> = match im_col.dtype() {
        DataType::Float32 => im_col.f32()?.into_iter().map(|opt| opt.map(|v| v as f64).unwrap_or(f64::NAN)).collect(),
        DataType::Float64 => im_col.f64()?.into_iter().map(|opt| opt.unwrap_or(f64::NAN)).collect(),
        _ => return Err(PolarsError::SchemaMismatch(format!("IM列的类型不是浮点数: {:?}", im_col.dtype()).into())),
    };
    let mut rt_dict = HashMap::new();
    let mut im_dict = HashMap::new();
    for ((id, rt), im) in id_vec.iter().zip(rt_vec.iter()).zip(im_vec.iter()) {
        rt_dict.insert(id.clone(), *rt);
        im_dict.insert(id.clone(), *im);
    }
    let elapsed = start.elapsed();
    println!("字典创建完成！");
    println!("  RT字典包含 {} 个条目", rt_dict.len());
    println!("  IM字典包含 {} 个条目", im_dict.len());
    println!("  耗时: {:.2}秒", elapsed.as_secs_f64());
    Ok((rt_dict, im_dict))
}

fn filter_library_by_precursor_ids(library: &[LibraryRecord], precursor_id_list: &[String]) -> Vec<LibraryRecord> {
    use std::time::Instant;
    let start = Instant::now();
    println!("\n开始筛选库记录...");
    println!("输入库记录数: {}", library.len());
    println!("筛选条件: transition_group_id 在 {:?} 中", precursor_id_list);
    let id_set: std::collections::HashSet<&String> = precursor_id_list.iter().collect();
    let filtered: Vec<LibraryRecord> = library.par_iter().filter(|record| id_set.contains(&record.transition_group_id)).cloned().collect();
    let elapsed = start.elapsed();
    println!("筛选完成！找到 {} 条匹配记录，耗时: {:.2}秒", filtered.len(), elapsed.as_secs_f64());
    filtered
}

// ========== 张量构建函数 ==========

/// 构建前体矩阵步骤1：将MS数据列表转换为张量
fn build_precursors_matrix_step1(
    ms1_data_list: &[MSDataArray], 
    ms2_data_list: &[MSDataArray], 
    device: &str
) -> Result<(Array3<f32>, Array3<f32>), Box<dyn Error>> {
    println!("\n构建前体矩阵步骤1：转换数据为张量...");
    
    if ms1_data_list.is_empty() || ms2_data_list.is_empty() {
        return Err("MS1或MS2数据列表为空".into());
    }
    
    let batch_size = ms1_data_list.len();
    let ms1_rows = ms1_data_list[0].len();
    let ms1_cols = if !ms1_data_list[0].is_empty() { ms1_data_list[0][0].len() } else { 0 };
    let ms2_rows = ms2_data_list[0].len();
    let ms2_cols = if !ms2_data_list[0].is_empty() { ms2_data_list[0][0].len() } else { 0 };
    
    println!("  MS1张量维度: [{}, {}, {}]", batch_size, ms1_rows, ms1_cols);
    println!("  MS2张量维度: [{}, {}, {}]", batch_size, ms2_rows, ms2_cols);
    
    let mut ms1_tensor = Array3::<f32>::zeros((batch_size, ms1_rows, ms1_cols));
    for (i, ms1_data) in ms1_data_list.iter().enumerate() {
        for (j, row) in ms1_data.iter().enumerate() {
            for (k, &val) in row.iter().enumerate() {
                ms1_tensor[[i, j, k]] = val as f32;
            }
        }
    }
    
    let mut ms2_tensor = Array3::<f32>::zeros((batch_size, ms2_rows, ms2_cols));
    for (i, ms2_data) in ms2_data_list.iter().enumerate() {
        for (j, row) in ms2_data.iter().enumerate() {
            for (k, &val) in row.iter().enumerate() {
                ms2_tensor[[i, j, k]] = val as f32;
            }
        }
    }
    
    println!("  张量创建完成！");
    Ok((ms1_tensor, ms2_tensor))
}

/// 构建前体矩阵步骤2：处理MS2张量中的特殊值
fn build_precursors_matrix_step2(mut ms2_data_tensor: Array3<f32>) -> Array3<f32> {
    println!("\n构建前体矩阵步骤2：处理特殊值...");
    
    let shape = ms2_data_tensor.shape();
    let (batch, rows, cols) = (shape[0], shape[1], shape[2]);
    
    for i in 0..batch {
        for j in 0..rows {
            if cols > 6 {
                let val0 = ms2_data_tensor[[i, j, 0]];
                let val6 = ms2_data_tensor[[i, j, 6]];
                let val2 = ms2_data_tensor[[i, j, 2]];
                
                if val2 != 0.0 {
                    ms2_data_tensor[[i, j, 0]] = val0 + val6 / val2;
                }
            }
        }
    }
    
    let mut inf_count = 0;
    let mut nan_count = 0;
    
    for i in 0..batch {
        for j in 0..rows {
            for k in 0..cols {
                let val = ms2_data_tensor[[i, j, k]];
                if val.is_infinite() {
                    ms2_data_tensor[[i, j, k]] = 0.0;
                    inf_count += 1;
                } else if val.is_nan() {
                    ms2_data_tensor[[i, j, k]] = 0.0;
                    nan_count += 1;
                }
            }
        }
    }
    
    println!("  处理了 {} 个无穷大值和 {} 个NaN值", inf_count, nan_count);
    ms2_data_tensor
}

/// 提取宽度函数（版本2）
fn extract_width_2(
    mz_to_extract: &Array3<f32>,
    mz_unit: &str,
    mz_tol: f32,
    max_extract_len: usize,
    frag_repeat_num: usize,
    max_moz_num: f32,
    device: &str
) -> Result<Array3<f32>, Box<dyn Error>> {
    let shape = mz_to_extract.shape();
    let (batch, rows, _) = (shape[0], shape[1], shape[2]);
    
    let is_all_zero = mz_to_extract.iter().all(|&v| v == 0.0);
    if is_all_zero {
        return Ok(Array3::<f32>::zeros((batch, rows, 2)));
    }
    
    let mut mz_tol_full = Array3::<f32>::zeros((batch, rows, 1));
    
    match mz_unit {
        "Da" => {
            for i in 0..batch {
                for j in 0..rows {
                    mz_tol_full[[i, j, 0]] = mz_tol;
                }
            }
        },
        "ppm" => {
            for i in 0..batch {
                for j in 0..rows {
                    mz_tol_full[[i, j, 0]] = mz_to_extract[[i, j, 0]] * mz_tol * 0.000001;
                }
            }
        },
        _ => return Err(format!("Invalid mz_unit format: {}. Only Da and ppm are supported.", mz_unit).into()),
    }
    
    for i in 0..batch {
        for j in 0..rows {
            if mz_tol_full[[i, j, 0]].is_nan() {
                mz_tol_full[[i, j, 0]] = 0.0;
            }
        }
    }
    
    let mz_tol_full_num = max_moz_num / 1000.0;
    for i in 0..batch {
        for j in 0..rows {
            if mz_tol_full[[i, j, 0]] > mz_tol_full_num {
                mz_tol_full[[i, j, 0]] = mz_tol_full_num;
            }
        }
    }
    
    for i in 0..batch {
        for j in 0..rows {
            let val = mz_tol_full[[i, j, 0]];
            mz_tol_full[[i, j, 0]] = ((val * 1000.0 / frag_repeat_num as f32).ceil()) * frag_repeat_num as f32;
        }
    }
    
    let mut extract_width_range_list = Array3::<f32>::zeros((batch, rows, 2));
    
    for i in 0..batch {
        for j in 0..rows {
            let mz_val = mz_to_extract[[i, j, 0]] * 1000.0;
            let tol_val = mz_tol_full[[i, j, 0]];
            extract_width_range_list[[i, j, 0]] = (mz_val - tol_val).floor();
            extract_width_range_list[[i, j, 1]] = (mz_val + tol_val).floor();
        }
    }
    
    Ok(extract_width_range_list)
}

/// 构建范围矩阵步骤3
fn build_range_matrix_step3(
    ms1_data_tensor: &Array3<f32>,
    ms2_data_tensor: &Array3<f32>,
    frag_repeat_num: usize,
    mz_unit: &str,
    mz_tol_ms1: f32,
    mz_tol_ms2: f32,
    device: &str
) -> Result<(Array3<f32>, Array3<f32>), Box<dyn Error>> {
    println!("\n构建范围矩阵步骤3...");
    
    let shape1 = ms1_data_tensor.shape();
    let shape2 = ms2_data_tensor.shape();
    
    let mut re_ms1_data_tensor = Array3::<f32>::zeros((shape1[0], shape1[1] * frag_repeat_num, shape1[2]));
    let mut re_ms2_data_tensor = Array3::<f32>::zeros((shape2[0], shape2[1] * frag_repeat_num, shape2[2]));
    
    for i in 0..shape1[0] {
        for rep in 0..frag_repeat_num {
            for j in 0..shape1[1] {
                for k in 0..shape1[2] {
                    re_ms1_data_tensor[[i, rep * shape1[1] + j, k]] = ms1_data_tensor[[i, j, k]];
                }
            }
        }
    }
    
    for i in 0..shape2[0] {
        for rep in 0..frag_repeat_num {
            for j in 0..shape2[1] {
                for k in 0..shape2[2] {
                    re_ms2_data_tensor[[i, rep * shape2[1] + j, k]] = ms2_data_tensor[[i, j, k]];
                }
            }
        }
    }
    
    let ms1_col0 = re_ms1_data_tensor.slice(s![.., .., 0..1]).to_owned();
    let ms2_col0 = re_ms2_data_tensor.slice(s![.., .., 0..1]).to_owned();
    
    let ms1_extract_width_range_list = extract_width_2(
        &ms1_col0, mz_unit, mz_tol_ms1, 20, frag_repeat_num, 50.0, device
    )?;
    
    let ms2_extract_width_range_list = extract_width_2(
        &ms2_col0, mz_unit, mz_tol_ms2, 20, frag_repeat_num, 50.0, device
    )?;
    
    println!("  MS1范围矩阵形状: {:?}", ms1_extract_width_range_list.shape());
    println!("  MS2范围矩阵形状: {:?}", ms2_extract_width_range_list.shape());
    
    Ok((ms1_extract_width_range_list, ms2_extract_width_range_list))
}

/// 提取宽度函数（主版本）
fn extract_width(
    mz_to_extract: &Array3<f32>,
    mz_unit: &str,
    mz_tol: f32,
    max_extract_len: usize,
    frag_repeat_num: usize,
    max_moz_num: f32,
    device: &str
) -> Result<Array3<f32>, Box<dyn Error>> {
    let shape = mz_to_extract.shape();
    let (batch, rows, _) = (shape[0], shape[1], shape[2]);
    
    let is_all_zero = mz_to_extract.iter().all(|&v| v == 0.0);
    if is_all_zero {
        return Ok(Array3::<f32>::zeros((batch, rows, max_moz_num as usize)));
    }
    
    let mut mz_tol_half = Array3::<f32>::zeros((batch, rows, 1));
    
    match mz_unit {
        "Da" => {
            for i in 0..batch {
                for j in 0..rows {
                    mz_tol_half[[i, j, 0]] = mz_tol / 2.0;
                }
            }
        },
        "ppm" => {
            for i in 0..batch {
                for j in 0..rows {
                    mz_tol_half[[i, j, 0]] = mz_to_extract[[i, j, 0]] * mz_tol * 0.000001 / 2.0;
                }
            }
        },
        _ => return Err(format!("Invalid mz_unit format: {}. Only Da and ppm are supported.", mz_unit).into()),
    }
    
    for i in 0..batch {
        for j in 0..rows {
            if mz_tol_half[[i, j, 0]].is_nan() {
                mz_tol_half[[i, j, 0]] = 0.0;
            }
        }
    }
    
    let mz_tol_half_num = (max_moz_num / 1000.0) / 2.0;
    for i in 0..batch {
        for j in 0..rows {
            if mz_tol_half[[i, j, 0]] > mz_tol_half_num {
                mz_tol_half[[i, j, 0]] = mz_tol_half_num;
            }
        }
    }
    
    for i in 0..batch {
        for j in 0..rows {
            let val = mz_tol_half[[i, j, 0]];
            mz_tol_half[[i, j, 0]] = ((val * 1000.0 / frag_repeat_num as f32).ceil()) * frag_repeat_num as f32;
        }
    }
    
    let mut extract_width_list = Array3::<f32>::zeros((batch, rows, 2));
    
    for i in 0..batch {
        for j in 0..rows {
            let mz_val = mz_to_extract[[i, j, 0]] * 1000.0;
            let tol_val = mz_tol_half[[i, j, 0]];
            extract_width_list[[i, j, 0]] = (mz_val - tol_val).floor();
            extract_width_list[[i, j, 1]] = (mz_val + tol_val).floor();
        }
    }
    
    let batch_num = rows / frag_repeat_num;
    
    let mut cha_tensor = Array2::<f32>::zeros((batch, batch_num));
    for i in 0..batch {
        for j in 0..batch_num {
            cha_tensor[[i, j]] = (extract_width_list[[i, j, 1]] - extract_width_list[[i, j, 0]]) / frag_repeat_num as f32;
        }
    }
    
    for i in 0..batch {
        for j in 0..batch_num {
            extract_width_list[[i, j, 1]] = extract_width_list[[i, j, 0]] + cha_tensor[[i, j]] - 1.0;
        }
        
        for j in 0..batch_num {
            let idx = batch_num + j;
            if idx < rows {
                extract_width_list[[i, idx, 0]] = extract_width_list[[i, j, 0]] + cha_tensor[[i, j]];
                extract_width_list[[i, idx, 1]] = extract_width_list[[i, j, 0]] + 2.0 * cha_tensor[[i, j]] - 1.0;
            }
        }
        
        for j in 0..batch_num {
            let idx = batch_num * 2 + j;
            if idx < rows {
                extract_width_list[[i, idx, 0]] = extract_width_list[[i, j, 0]] + 2.0 * cha_tensor[[i, j]];
                extract_width_list[[i, idx, 1]] = extract_width_list[[i, j, 0]] + 3.0 * cha_tensor[[i, j]] - 1.0;
            }
        }
        
        for j in 0..batch_num {
            let idx = batch_num * 3 + j;
            if idx < rows {
                extract_width_list[[i, idx, 0]] = extract_width_list[[i, j, 0]] + 3.0 * cha_tensor[[i, j]];
                extract_width_list[[i, idx, 1]] = extract_width_list[[i, j, 0]] + 4.0 * cha_tensor[[i, j]] - 1.0;
            }
        }
        
        for j in 0..batch_num {
            let idx = batch_num * 4 + j;
            if idx < rows {
                extract_width_list[[i, idx, 0]] = extract_width_list[[i, j, 0]] + 4.0 * cha_tensor[[i, j]];
                extract_width_list[[i, idx, 1]] = extract_width_list[[i, j, 0]] + 5.0 * cha_tensor[[i, j]] - 1.0;
            }
        }
    }
    
    let mut new_tensor = Array3::<f32>::zeros((batch, rows, max_moz_num as usize));
    
    for i in 0..batch {
        for j in 0..rows {
            for k in 0..(max_moz_num as usize) {
                new_tensor[[i, j, k]] = extract_width_list[[i, j, 0]] + k as f32;
                if new_tensor[[i, j, k]] > extract_width_list[[i, j, 1]] {
                    new_tensor[[i, j, k]] = 0.0;
                }
            }
        }
    }
    
    Ok(new_tensor)
}

/// 构建前体矩阵步骤3（主版本）
fn build_precursors_matrix_step3(
    ms1_data_tensor: &Array3<f32>,
    ms2_data_tensor: &Array3<f32>,
    frag_repeat_num: usize,
    mz_unit: &str,
    mz_tol_ms1: f32,
    mz_tol_ms2: f32,
    device: &str
) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>, Array3<f32>), Box<dyn Error>> {
    println!("\n构建前体矩阵步骤3（主版本）...");
    
    let shape1 = ms1_data_tensor.shape();
    let shape2 = ms2_data_tensor.shape();
    
    let mut re_ms1_data_tensor = Array3::<f32>::zeros((shape1[0], shape1[1] * frag_repeat_num, shape1[2]));
    let mut re_ms2_data_tensor = Array3::<f32>::zeros((shape2[0], shape2[1] * frag_repeat_num, shape2[2]));
    
    for i in 0..shape1[0] {
        for rep in 0..frag_repeat_num {
            for j in 0..shape1[1] {
                for k in 0..shape1[2] {
                    re_ms1_data_tensor[[i, rep * shape1[1] + j, k]] = ms1_data_tensor[[i, j, k]];
                }
            }
        }
    }
    
    for i in 0..shape2[0] {
        for rep in 0..frag_repeat_num {
            for j in 0..shape2[1] {
                for k in 0..shape2[2] {
                    re_ms2_data_tensor[[i, rep * shape2[1] + j, k]] = ms2_data_tensor[[i, j, k]];
                }
            }
        }
    }
    
    let ms1_col0 = re_ms1_data_tensor.slice(s![.., .., 0..1]).to_owned();
    let ms2_col0 = re_ms2_data_tensor.slice(s![.., .., 0..1]).to_owned();
    
    let ms1_extract_width_range_list = extract_width(
        &ms1_col0, mz_unit, mz_tol_ms1, 20, frag_repeat_num, 50.0, device
    )?;
    
    let ms2_extract_width_range_list = extract_width(
        &ms2_col0, mz_unit, mz_tol_ms2, 20, frag_repeat_num, 50.0, device
    )?;
    
    println!("  重复后的MS1张量形状: {:?}", re_ms1_data_tensor.shape());
    println!("  重复后的MS2张量形状: {:?}", re_ms2_data_tensor.shape());
    println!("  MS1提取宽度范围形状: {:?}", ms1_extract_width_range_list.shape());
    println!("  MS2提取宽度范围形状: {:?}", ms2_extract_width_range_list.shape());
    
    Ok((re_ms1_data_tensor, re_ms2_data_tensor, ms1_extract_width_range_list, ms2_extract_width_range_list))
}

/// 构建扩展MS1矩阵
fn build_ext_ms1_matrix(ms1_data_tensor: &Array3<f32>, device: &str) -> Array3<f32> {
    let shape = ms1_data_tensor.shape();
    let (batch, rows, _) = (shape[0], shape[1], shape[2]);
    
    let mut ext_matrix = Array3::<f32>::zeros((batch, rows, 4));
    
    for i in 0..batch {
        for j in 0..rows {
            ext_matrix[[i, j, 0]] = ms1_data_tensor[[i, j, 0]];
            if shape[2] > 3 {
                ext_matrix[[i, j, 1]] = ms1_data_tensor[[i, j, 3]];
            }
            if shape[2] > 8 {
                ext_matrix[[i, j, 2]] = ms1_data_tensor[[i, j, 8]];
            }
            if shape[2] > 4 {
                ext_matrix[[i, j, 3]] = ms1_data_tensor[[i, j, 4]];
            }
        }
    }
    
    ext_matrix
}

/// 构建扩展MS2矩阵
fn build_ext_ms2_matrix(ms2_data_tensor: &Array3<f32>, device: &str) -> Array3<f32> {
    let shape = ms2_data_tensor.shape();
    let (batch, rows, _) = (shape[0], shape[1], shape[2]);
    
    let mut ext_matrix = Array3::<f32>::zeros((batch, rows, 4));
    
    for i in 0..batch {
        for j in 0..rows {
            ext_matrix[[i, j, 0]] = ms2_data_tensor[[i, j, 0]];
            if shape[2] > 3 {
                ext_matrix[[i, j, 1]] = ms2_data_tensor[[i, j, 3]];
            }
            if shape[2] > 8 {
                ext_matrix[[i, j, 2]] = ms2_data_tensor[[i, j, 8]];
            }
            if shape[2] > 4 {
                ext_matrix[[i, j, 3]] = ms2_data_tensor[[i, j, 4]];
            }
        }
    }
    
    ext_matrix
}

/// 构建碎片信息
fn build_frag_info(
    ms1_data_tensor: &Array3<f32>,
    ms2_data_tensor: &Array3<f32>,
    frag_repeat_num: usize,
    device: &str
) -> Array3<f32> {
    println!("\n构建碎片信息...");
    
    println!("  输入MS1张量形状: {:?}", ms1_data_tensor.shape());
    println!("  输入MS2张量形状: {:?}", ms2_data_tensor.shape());
    println!("  frag_repeat_num: {}", frag_repeat_num);
    
    let ext_ms1_precursors_frag_rt_matrix = build_ext_ms1_matrix(ms1_data_tensor, device);
    let ext_ms2_precursors_frag_rt_matrix = build_ext_ms2_matrix(ms2_data_tensor, device);
    
    println!("  扩展MS1矩阵形状: {:?}", ext_ms1_precursors_frag_rt_matrix.shape());
    println!("  扩展MS2矩阵形状: {:?}", ext_ms2_precursors_frag_rt_matrix.shape());
    
    let ms1_shape = ext_ms1_precursors_frag_rt_matrix.shape().to_vec();
    let ms2_shape = ext_ms2_precursors_frag_rt_matrix.shape().to_vec();
    
    let batch = ms1_shape[0];
    let ms1_rows = ms1_shape[1];
    let ms2_rows = ms2_shape[1];
    
    println!("  MS1行数: {}, 是否能被{}整除: {}", ms1_rows, frag_repeat_num, ms1_rows % frag_repeat_num == 0);
    println!("  MS2行数: {}, 是否能被{}整除: {}", ms2_rows, frag_repeat_num, ms2_rows % frag_repeat_num == 0);
    
    let orig_ms1_shape = ms1_data_tensor.shape();
    let orig_ms2_shape = ms2_data_tensor.shape();
    let ms1_frag_count = orig_ms1_shape[1];
    let ms2_frag_count = orig_ms2_shape[1];
    
    println!("  原始MS1碎片数: {}", ms1_frag_count);
    println!("  原始MS2碎片数: {}", ms2_frag_count);
    
    let total_frag_count = ms1_frag_count + ms2_frag_count;
    let mut frag_info = Array3::<f32>::zeros((batch, total_frag_count, 4));
    
    for i in 0..batch {
        for j in 0..ms1_frag_count {
            for k in 0..4 {
                frag_info[[i, j, k]] = ext_ms1_precursors_frag_rt_matrix[[i, j, k]];
            }
        }
        
        for j in 0..ms2_frag_count {
            for k in 0..4 {
                frag_info[[i, ms1_frag_count + j, k]] = ext_ms2_precursors_frag_rt_matrix[[i, j, k]];
            }
        }
    }
    
    println!("  碎片信息矩阵形状: {:?}", frag_info.shape());
    
    if frag_info.shape()[0] > 0 && frag_info.shape()[1] > 0 {
        println!("  碎片信息第一行: {:?}", frag_info.slice(s![0, 0, ..]));
        if frag_info.shape()[1] > ms1_frag_count {
            println!("  MS2部分第一行: {:?}", frag_info.slice(s![0, ms1_frag_count, ..]));
        }
    }
    
    frag_info
}

/// 创建前体特征矩阵
fn create_precursor_feat(
    precursor_info_list: &[Vec<f64>],
    precursors_list: &[Vec<String>],
    assay_rt_kept_dict: &HashMap<String, f64>,
    assay_im_kept_dict: &HashMap<String, f64>,
) -> Result<Array2<f64>, Box<dyn Error>> {
    println!("\n构建前体特征矩阵...");
    
    let n_precursors = precursor_info_list.len();
    if n_precursors == 0 {
        return Err("前体信息列表为空".into());
    }
    
    let mut precursor_feat = Array2::<f64>::zeros((n_precursors, 8));
    
    for (i, (info, precursor)) in precursor_info_list.iter().zip(precursors_list.iter()).enumerate() {
        for j in 0..5.min(info.len()) {
            precursor_feat[[i, j]] = info[j];
        }
        
        if let Some(&im) = assay_im_kept_dict.get(&precursor[0]) {
            precursor_feat[[i, 5]] = im;
        } else {
            precursor_feat[[i, 5]] = 0.0;
        }
        
        if let Some(&rt) = assay_rt_kept_dict.get(&precursor[0]) {
            precursor_feat[[i, 6]] = rt;
        } else {
            precursor_feat[[i, 6]] = 0.0;
        }
        
        precursor_feat[[i, 7]] = 0.0;
    }
    
    println!("  前体特征矩阵形状: {:?}", precursor_feat.shape());
    Ok(precursor_feat)
}

/// 提取指定m/z范围的MS2碎片
fn extract_ms2_fragments_for_ranges(
    all_data: &HashMap<String, TimsTOFData>,
    ms1_range_min: f64,
    ms1_range_max: f64,
    ms2_range_list: &Array3<f32>,
    precursor_idx: usize,
    n_fragments: usize,
) -> Result<Vec<TimsTOFData>, Box<dyn Error>> {
    println!("\n========== 开始提取MS2碎片数据 ==========");
    let start_time = Instant::now();
    
    let ms2_data = match all_data.get("ms2") {
        Some(data) => data,
        None => {
            println!("警告：没有找到MS2数据");
            return Ok(vec![TimsTOFData::new(); n_fragments]);
        }
    };
    
    println!("MS2数据总点数: {}", ms2_data.mz_values.len());
    println!("MS2范围矩阵形状: {:?}", ms2_range_list.shape());
    println!("将提取 {} 个MS2碎片", n_fragments);
    
    let mut frag_results = Vec::new();
    let mut fragment_times = Vec::new();
    
    for j in 0..n_fragments {
        let fragment_start = Instant::now();
        
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
        
        let fragment_count = fragment_data.mz_values.len();
        frag_results.push(fragment_data);
        
        let fragment_time = fragment_start.elapsed();
        fragment_times.push(fragment_time.as_secs_f64());
        
        if j % 10 == 0 {
            let recent_avg = if fragment_times.len() >= 10 {
                fragment_times[fragment_times.len()-10..].iter().sum::<f64>() / 10.0
            } else {
                fragment_times.iter().sum::<f64>() / fragment_times.len() as f64
            };
            println!("  已处理 {}/{} 个碎片，最近10个碎片平均耗时: {:.2}秒", 
                j + 1, n_fragments, recent_avg);
        }
        
        if j < 5 {
            println!("    碎片 {}: m/z范围 {:.4} - {:.4}, 数据点数: {}", 
                j, ms2_range_min, ms2_range_max, fragment_count);
        }
    }
    
    let extract_time = start_time.elapsed();
    println!("\n碎片提取完成，总耗时: {:.2}秒", extract_time.as_secs_f64());
    
    Ok(frag_results)
}

// ========== 数据处理函数 ==========

/// 根据IM范围过滤TimsTOF数据
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

/// 将m/z值转换为整数格式（乘以1000并向上取整）
fn convert_mz_to_integer(data: &TimsTOFData) -> TimsTOFData {
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

/// 导出TimsTOF数据的统计信息
fn print_timstof_data_stats(data: &TimsTOFData, label: &str) {
    if data.mz_values.is_empty() {
        println!("\n{}数据为空", label);
        return;
    }
    
    let min_mz = data.mz_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_mz = data.mz_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_rt = data.rt_values_min.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_rt = data.rt_values_min.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_im = data.mobility_values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max_im = data.mobility_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let min_intensity = data.intensity_values.iter().min().copied().unwrap_or(0);
    let max_intensity = data.intensity_values.iter().max().copied().unwrap_or(0);
    
    println!("\n{}数据统计:", label);
    println!("  数据点数: {}", data.mz_values.len());
    println!("  m/z范围: {:.4} - {:.4}", min_mz, max_mz);
    println!("  RT范围: {:.2} - {:.2} 分钟", min_rt, max_rt);
    println!("  IM范围: {:.4} - {:.4}", min_im, max_im);
    println!("  强度范围: {} - {}", min_intensity, max_intensity);
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("程序开始运行...");
    let program_start = Instant::now();
    
    // ========== 优化1：只读取一次TSV文件 ==========
    println!("\n========== 读取库文件（只读取一次） ==========");
    let lib_file_path = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/lib/TPHPlib_frag1025_swissprot_final_all_from_Yueliang.tsv";
    
    let library_records = match process_library_fast(lib_file_path) {
        Ok(lib) => {
            println!("\n总共加载了 {} 条库记录", lib.len());
            println!("\n前5条库记录（选定列）:");
            for (i, record) in lib.iter().take(5).enumerate() {
                println!("记录 {}:", i + 1);
                println!("  transition_group_id: {}", record.transition_group_id);
                println!("  PeptideSequence: {}", record.peptide_sequence);
                println!("  FullUniModPeptideName: {}", record.full_unimod_peptide_name);
                println!("  PrecursorCharge: {}", record.precursor_charge);
                println!("  FragmentType: {}", record.fragment_type);
                println!("  decoy: {}", record.decoy);
                println!();
            }
            lib
        }
        Err(e) => {
            eprintln!("读取库文件错误: {}", e);
            return Err(e);
        }
    };
    
    // 2. 将library转换为DataFrame
    let step2_start = Instant::now();
    let library_df = match library_records_to_dataframe(library_records.clone()) {
        Ok(df) => {
            println!("\nLibrary DataFrame前5行:");
            println!("{}", df.head(Some(5)));
            println!("步骤2耗时: {:.2}秒", step2_start.elapsed().as_secs_f64());
            df
        }
        Err(e) => {
            eprintln!("转换Library为DataFrame时出错: {}", e);
            return Err(Box::new(e));
        }
    };
    
    // 3. 读取DIA-NN报告文件（Parquet格式）
    let step3_start = Instant::now();
    println!("\n开始读取DIA-NN报告文件...");
    let report_file_path = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/report/report.parquet";
    let report_df = match read_parquet_with_polars(report_file_path) {
        Ok(df) => {
            println!("\n总共加载了 {} 条DIA-NN报告记录", df.height());
            println!("\n前5条DIA-NN报告记录:");
            println!("{}", df.head(Some(5)));
            println!("步骤3耗时: {:.2}秒", step3_start.elapsed().as_secs_f64());
            df
        }
        Err(e) => {
            eprintln!("读取Parquet文件错误: {}", e);
            return Err(Box::new(e));
        }
    };
    
    // 4. 执行merge操作
    let step4_start = Instant::now();
    let mut diann_result = match merge_library_and_report(library_df, report_df) {
        Ok(df) => {
            println!("\nMerge结果前10行:");
            println!("{}", df.head(Some(10)));
            println!("步骤4耗时: {:.2}秒", step4_start.elapsed().as_secs_f64());
            df
        }
        Err(e) => {
            eprintln!("Merge操作失败: {}", e);
            return Err(Box::new(e));
        }
    };
    
    // 5. 导出diann_result到CSV文件
    let step5_start = Instant::now();
    export_polars_to_csv(&mut diann_result, "diann_result_rust.csv")?;
    println!("步骤5耗时: {:.2}秒", step5_start.elapsed().as_secs_f64());
    
    // 6. 提取唯一的前体ID数据
    let step6_start = Instant::now();
    let mut diann_precursor_id_all = match get_unique_precursor_ids(&diann_result) {
        Ok(df) => {
            println!("\n成功提取唯一前体ID数据！");
            println!("步骤6耗时: {:.2}秒", step6_start.elapsed().as_secs_f64());
            df
        }
        Err(e) => {
            eprintln!("提取唯一前体ID失败: {}", e);
            return Err(Box::new(e));
        }
    };
    
    // 7. 导出唯一前体ID数据到CSV文件
    let step7_start = Instant::now();
    export_polars_to_csv(&mut diann_precursor_id_all, "diann_precursor_id_all_rust.csv")?;
    println!("步骤7耗时: {:.2}秒", step7_start.elapsed().as_secs_f64());
    
    // 8. 创建RT和IM的查找字典
    let step8_start = Instant::now();
    let (assay_rt_kept_dict, assay_im_kept_dict) = create_rt_im_dicts(&diann_precursor_id_all)?;
    println!("步骤8耗时: {:.2}秒", step8_start.elapsed().as_secs_f64());
    
    println!("\nRT字典示例（前5个条目）:");
    for (i, (key, value)) in assay_rt_kept_dict.iter().take(5).enumerate() {
        println!("  {}: {} -> {}", i + 1, key, value);
    }
    
    println!("\nIM字典示例（前5个条目）:");
    for (i, (key, value)) in assay_im_kept_dict.iter().take(5).enumerate() {
        println!("  {}: {} -> {}", i + 1, key, value);
    }
    
    // 9. 根据前体ID列表筛选库数据（使用已有的library_records）
    let step9_start = Instant::now();
    let precursor_id_list = vec!["LLIYGASTR2".to_string()];
    
    println!("\n正在筛选前体ID列表: {:?}", precursor_id_list);
    
    // 优化：直接使用已有的library_records，而不是重新读取
    let each_lib_data = filter_library_by_precursor_ids(&library_records, &precursor_id_list);
    println!("步骤9耗时: {:.2}秒", step9_start.elapsed().as_secs_f64());
    
    println!("\n筛选结果：找到 {} 条匹配的库记录", each_lib_data.len());
    
    if !each_lib_data.is_empty() {
        println!("\n筛选后的库记录详情:");
        for (i, record) in each_lib_data.iter().enumerate() {
            println!("\n记录 {}:", i + 1);
            println!("  transition_group_id: {}", record.transition_group_id);
            println!("  PeptideSequence: {}", record.peptide_sequence);
            println!("  FullUniModPeptideName: {}", record.full_unimod_peptide_name);
            println!("  PrecursorCharge: {}", record.precursor_charge);
            println!("  PrecursorMz: {}", record.precursor_mz);
            println!("  ProductMz: {}", record.product_mz);
            println!("  FragmentType: {}", record.fragment_type);
            println!("  FragmentNumber: {}", record.fragment_number);
            println!("  FragmentCharge: {}", record.fragment_charge);
            println!("  LibraryIntensity: {}", record.library_intensity);
            
            if let Some(rt) = assay_rt_kept_dict.get(&record.transition_group_id) {
                println!("  对应的RT值: {}", rt);
            }
            if let Some(im) = assay_im_kept_dict.get(&record.transition_group_id) {
                println!("  对应的IM值: {}", im);
            }
        }
        
        // 10. 将筛选后的数据导出到CSV
        let step10_start = Instant::now();
        export_to_csv(&each_lib_data, "each_lib_data_rust.csv")?;
        println!("\n筛选后的库数据已保存到: each_lib_data_rust.csv");
        println!("步骤10耗时: {:.2}秒", step10_start.elapsed().as_secs_f64());
    } else {
        println!("警告：没有找到匹配的库记录！");
    }
    
    // ========== 读取TimsTOF数据 ==========
    println!("\n========== 开始读取TimsTOF数据 ==========");
    let timstof_start = Instant::now();
    
    let bruker_d_folder_name = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d";
    
    // 读取TimsTOF数据（使用PyO3优化方法）
    let timstof_data = match read_timstof_data(bruker_d_folder_name) {
        Ok(data) => {
            println!("\nTimsTOF数据读取成功！");
            println!("TimsTOF数据读取耗时: {:.2}秒", timstof_start.elapsed().as_secs_f64());
            data
        }
        Err(e) => {
            eprintln!("读取TimsTOF数据失败: {}", e);
            return Err(e);
        }
    };
    
    // ========== 处理特定前体的MS1数据 ==========
    println!("\n========== 开始处理特定前体的MS1数据 ==========");
    let ms1_processing_start = Instant::now();
    
    // 优化：直接使用已筛选的each_lib_data，而不是重新筛选
    if !each_lib_data.is_empty() {
        // 创建LibCols实例
        let lib_cols = LibCols::default();
        
        // 构建库矩阵
        let lib_matrix_start = Instant::now();
        match build_lib_matrix(&each_lib_data, &lib_cols, 5.0, 1801.0, 20) {
            Ok((precursors_list, ms1_data_list, ms2_data_list, precursor_info_list)) => {
                println!("\n成功构建库矩阵");
                println!("构建库矩阵耗时: {:.2}秒", lib_matrix_start.elapsed().as_secs_f64());
                
                // 构建前体特征矩阵
                let feat_matrix_start = Instant::now();
                let precursor_feat = create_precursor_feat(
                    &precursor_info_list,
                    &precursors_list,
                    &assay_rt_kept_dict,
                    &assay_im_kept_dict
                )?;
                println!("构建前体特征矩阵耗时: {:.2}秒", feat_matrix_start.elapsed().as_secs_f64());
                
                // 构建张量
                let tensor_start = Instant::now();
                let device = "cpu";
                let frag_repeat_num = 5;
                
                // 步骤1：构建前体矩阵
                let step1_tensor_start = Instant::now();
                let (ms1_tensor, ms2_tensor) = build_precursors_matrix_step1(&ms1_data_list, &ms2_data_list, device)?;
                println!("张量步骤1耗时: {:.2}秒", step1_tensor_start.elapsed().as_secs_f64());
                
                // 步骤2：处理MS2张量
                let step2_tensor_start = Instant::now();
                let ms2_tensor_processed = build_precursors_matrix_step2(ms2_tensor);
                println!("张量步骤2耗时: {:.2}秒", step2_tensor_start.elapsed().as_secs_f64());
                
                // 步骤3：构建范围矩阵
                let step3_tensor_start = Instant::now();
                let (ms1_range_list, ms2_range_list) = build_range_matrix_step3(
                    &ms1_tensor, 
                    &ms2_tensor_processed, 
                    frag_repeat_num,
                    "ppm",
                    20.0,
                    50.0,
                    device
                )?;
                println!("张量步骤3（范围矩阵）耗时: {:.2}秒", step3_tensor_start.elapsed().as_secs_f64());
                
                // 构建前体矩阵步骤3（主版本）
                let step3_main_start = Instant::now();
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
                println!("张量步骤3（主版本）耗时: {:.2}秒", step3_main_start.elapsed().as_secs_f64());
                
                // 构建碎片信息
                let frag_info_start = Instant::now();
                let frag_info = build_frag_info(
                    &ms1_tensor,
                    &ms2_tensor_processed,
                    frag_repeat_num,
                    device
                );
                println!("构建碎片信息耗时: {:.2}秒", frag_info_start.elapsed().as_secs_f64());
                
                println!("\n碎片信息构建完成:");
                println!("  碎片信息形状: {:?}", frag_info.shape());
                println!("张量构建总耗时: {:.2}秒", tensor_start.elapsed().as_secs_f64());
                
                // 使用第一个前体（索引0，因为我们只有一个前体）
                let i = 0;  // 对应Python中的 i = 0
                
                // 获取IM和RT值
                let im = precursor_feat[[i, 5]];  // IM在第6列（索引5）
                let rt = precursor_feat[[i, 6]];  // RT在第7列（索引6）
                
                println!("\n处理前体索引 {}: IM={}, RT={}", i, im, rt);

                // 计算MS1的m/z范围
                let mz_range_start = Instant::now();
                let ms1_range_slice = ms1_range_list.slice(s![i, .., ..]);
                let min_val = ms1_range_slice.iter()
                    .filter(|&&v| v > 0.0)
                    .fold(f64::INFINITY, |a, &b| a.min(b as f64));
                let max_val = ms1_range_slice.iter()
                    .fold(f64::NEG_INFINITY, |a, &b| a.max(b as f64));

                let ms1_range_min = (min_val - 1.0) / 1000.0;
                let ms1_range_max = (max_val + 1.0) / 1000.0;
                println!("\nMS1范围: {:.4} - {:.4}", ms1_range_min, ms1_range_max);
                println!("计算m/z范围耗时: {:.2}秒", mz_range_start.elapsed().as_secs_f64());

                // 从TimsTOF数据中提取指定m/z范围的数据
                let filter_mz_start = Instant::now();
                let precursor_result = timstof_data.filter_by_mz_range(ms1_range_min, ms1_range_max);
                println!("\n筛选m/z范围后的数据点数: {}", precursor_result.mz_values.len());
                println!("筛选m/z范围耗时: {:.2}秒", filter_mz_start.elapsed().as_secs_f64());

                // 将m/z值转换为整数格式
                let convert_mz_start = Instant::now();
                let precursor_result_int = convert_mz_to_integer(&precursor_result);
                print_timstof_data_stats(&precursor_result_int, "m/z整数转换后");
                println!("m/z整数转换耗时: {:.2}秒", convert_mz_start.elapsed().as_secs_f64());

                // 第一个导出点：过滤IM之前的数据
                let export1_start = Instant::now();
                println!("\n========== 导出IM过滤前的数据 ==========");
                match precursor_result_int.to_dataframe() {
                    Ok(mut df) => {
                        println!("\nIM过滤前的数据（前10行）:");
                        println!("{}", df.head(Some(10)));
                        
                        let mut file = File::create("precursor_result_before_IM_filter.csv")?;
                        CsvWriter::new(&mut file).include_header(true).finish(&mut df)?;

                        println!("\n已导出到: precursor_result_before_IM_filter.csv");
                        println!("总行数: {}", precursor_result_int.mz_values.len());
                        println!("导出IM过滤前数据耗时: {:.2}秒", export1_start.elapsed().as_secs_f64());
                    }
                    Err(e) => {
                        eprintln!("创建DataFrame失败: {}", e);
                        return Err(Box::new(e));
                    }
                }

                // 根据IM值进行过滤
                let im_filter_start = Instant::now();
                let im_tolerance = 0.05;
                let im_min = im - im_tolerance;
                let im_max = im + im_tolerance;

                println!("\n========== 应用IM过滤 ==========");
                println!("IM过滤条件: {:.4} <= IM <= {:.4}", im_min, im_max);
                println!("（中心值: {:.4}, 容差: ±{:.4}）", im, im_tolerance);

                // 执行IM过滤
                let filtered_result = filter_by_im_range(&precursor_result_int, im_min, im_max);

                println!("\nIM过滤结果:");
                println!("  过滤前数据点数: {}", precursor_result_int.mz_values.len());
                println!("  过滤后数据点数: {}", filtered_result.mz_values.len());
                println!("  保留比例: {:.2}%", 
                    filtered_result.mz_values.len() as f64 / precursor_result_int.mz_values.len() as f64 * 100.0);
                println!("IM过滤耗时: {:.2}秒", im_filter_start.elapsed().as_secs_f64());

                print_timstof_data_stats(&filtered_result, "IM过滤后");

                // 第二个导出点：过滤IM之后的数据
                let export2_start = Instant::now();
                println!("\n========== 导出IM过滤后的数据 ==========");
                match filtered_result.to_dataframe() {
                    Ok(mut df) => {
                        println!("\nIM过滤后的数据（前20行）:");
                        println!("{}", df.head(Some(20)));
                        
                        let mut file = File::create("precursor_result_after_IM_filter.csv")?;
                        CsvWriter::new(&mut file).include_header(true).finish(&mut df)?;
                        println!("\n已导出到: precursor_result_after_IM_filter.csv");
                        println!("总行数: {}", filtered_result.mz_values.len());
                        println!("导出IM过滤后数据耗时: {:.2}秒", export2_start.elapsed().as_secs_f64());
                        
                        // [保持原有的统计信息代码...]
                    }
                    Err(e) => {
                        eprintln!("创建DataFrame失败: {}", e);
                        return Err(Box::new(e));
                    }
                }

                println!("\n========== IM过滤处理完成 ==========");
                println!("MS1数据处理总耗时: {:.2}秒", ms1_processing_start.elapsed().as_secs_f64());

                // ========== 开始处理MS2碎片数据 ==========
                println!("\n========== 开始处理MS2碎片数据 ==========");
                let ms2_processing_start = Instant::now();

                // 读取完整的TimsTOF数据（MS1和MS2）
                let read_full_start = Instant::now();
                let all_data = match read_timstof_data_with_full_ms2(
                    bruker_d_folder_name,
                    ms1_range_min,
                    ms1_range_max
                ) {
                    Ok(data) => {
                        println!("成功读取完整的MS1和MS2数据");
                        println!("读取完整数据耗时: {:.2}秒", read_full_start.elapsed().as_secs_f64());
                        data
                    }
                    Err(e) => {
                        eprintln!("读取MS1/MS2数据失败: {}", e);
                        return Err(e);
                    }
                };

                // 提取MS2碎片
                let extract_ms2_start = Instant::now();
                let frag_results = match extract_ms2_fragments_for_ranges(
                    &all_data,
                    ms1_range_min,
                    ms1_range_max,
                    &ms2_range_list,
                    i,
                    66,  // 提取66个碎片
                ) {
                    Ok(results) => {
                        println!("MS2碎片提取耗时: {:.2}秒", extract_ms2_start.elapsed().as_secs_f64());
                        results
                    }
                    Err(e) => {
                        eprintln!("提取MS2碎片失败: {}", e);
                        return Err(e);
                    }
                };

                // 合并所有碎片数据
                println!("\n合并碎片数据...");
                let concat_start = Instant::now();
                let merged_frag_data = TimsTOFData::merge(frag_results);
                println!("合并数据耗时: {:.2}秒", concat_start.elapsed().as_secs_f64());

                // 转换m/z值为整数
                let convert_ms2_start = Instant::now();
                let merged_frag_data_int = convert_mz_to_integer(&merged_frag_data);
                println!("MS2 m/z整数转换耗时: {:.2}秒", convert_ms2_start.elapsed().as_secs_f64());

                // 导出IM过滤前的MS2数据
                let export_ms2_1_start = Instant::now();
                println!("\n导出IM过滤前的MS2数据...");
                match merged_frag_data_int.to_dataframe() {
                    Ok(mut df) => {
                        let mut file = File::create("frag_result_before_IM_filter.csv")?;
                        CsvWriter::new(&mut file).include_header(true).finish(&mut df)?;
                        println!("导出IM过滤前的MS2数据: {} 行", merged_frag_data_int.mz_values.len());
                        println!("导出MS2过滤前数据耗时: {:.2}秒", export_ms2_1_start.elapsed().as_secs_f64());
                    }
                    Err(e) => {
                        eprintln!("创建MS2 DataFrame失败: {}", e);
                    }
                }

                print_timstof_data_stats(&merged_frag_data_int, "IM过滤前的MS2");

                // 应用IM过滤
                let im_filter_ms2_start = Instant::now();
                let filtered_frag_data = filter_by_im_range(&merged_frag_data_int, im_min, im_max);
                println!("\nMS2 IM过滤耗时: {:.2}秒", im_filter_ms2_start.elapsed().as_secs_f64());

                if !merged_frag_data_int.mz_values.is_empty() {
                    let retain_ratio = filtered_frag_data.mz_values.len() as f64 / 
                                    merged_frag_data_int.mz_values.len() as f64 * 100.0;
                    println!("IM过滤后的MS2数据: {} 行 (保留比例: {:.2}%)", 
                        filtered_frag_data.mz_values.len(), retain_ratio);
                }

                // 导出IM过滤后的MS2数据
                let export_ms2_2_start = Instant::now();
                println!("\n导出IM过滤后的MS2数据...");
                match filtered_frag_data.to_dataframe() {
                    Ok(mut df) => {
                        let mut file = File::create("frag_result_after_IM_filter.csv")?;
                        CsvWriter::new(&mut file).include_header(true).finish(&mut df)?;
                        println!("已导出到: frag_result_after_IM_filter.csv");
                        println!("导出MS2过滤后数据耗时: {:.2}秒", export_ms2_2_start.elapsed().as_secs_f64());
                        
                        println!("\nIM过滤后的MS2数据（前20行）:");
                        println!("{}", df.head(Some(20)));
                    }
                    Err(e) => {
                        eprintln!("创建过滤后MS2 DataFrame失败: {}", e);
                    }
                }

                println!("\n========== MS2碎片数据处理完成 ==========");
                println!("MS2处理总耗时: {:.2}秒", ms2_processing_start.elapsed().as_secs_f64());

                // ========== 构建Mask矩阵 ==========
                println!("\n========== 构建Mask矩阵 ==========");
                let mask_start = Instant::now();

                // 创建搜索集合
                let search_set_start = Instant::now();
                let search_ms1_set: HashSet<i64> = filtered_result.mz_values.iter()
                    .map(|&mz| mz as i64)
                    .collect();
                let search_ms2_set: HashSet<i64> = filtered_frag_data.mz_values.iter()
                    .map(|&mz| mz as i64)
                    .collect();
                println!("创建搜索集合耗时: {:.2}秒", search_set_start.elapsed().as_secs_f64());

                println!("MS1搜索集合大小: {}", search_ms1_set.len());
                println!("MS2搜索集合大小: {}", search_ms2_set.len());

                // 构建MS1 mask
                println!("\n构建MS1 mask...");
                let mask_ms1_start = Instant::now();

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

                println!("MS1 mask构建耗时: {:.2}秒", mask_ms1_start.elapsed().as_secs_f64());

                // 构建MS2 mask
                println!("\n构建MS2 mask...");
                let mask_ms2_start = Instant::now();

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

                println!("MS2 mask构建耗时: {:.2}秒", mask_ms2_start.elapsed().as_secs_f64());

                println!("\nMask构建总耗时: {:.2}秒", mask_start.elapsed().as_secs_f64());

                // 统计非零元素
                let ms1_nonzero = ms1_frag_moz_matrix.iter()
                    .filter(|&&v| v > 0.0)
                    .count();
                let ms1_total = ms1_frag_moz_matrix.len();

                let ms2_nonzero = ms2_frag_moz_matrix.iter()
                    .filter(|&&v| v > 0.0)
                    .count();
                let ms2_total = ms2_frag_moz_matrix.len();

                println!("\nMS1碎片矩阵形状: {:?}", ms1_frag_moz_matrix.shape());
                println!("MS1碎片矩阵非零元素: {} / {} ({:.2}%)", 
                    ms1_nonzero, ms1_total, 
                    ms1_nonzero as f64 / ms1_total as f64 * 100.0);

                println!("\nMS2碎片矩阵形状: {:?}", ms2_frag_moz_matrix.shape());
                println!("MS2碎片矩阵非零元素: {} / {} ({:.2}%)", 
                    ms2_nonzero, ms2_total, 
                    ms2_nonzero as f64 / ms2_total as f64 * 100.0);

                // 保存Mask矩阵
                let save_mask_start = Instant::now();
                println!("\n保存MS1 mask矩阵到CSV文件...");
                let mut ms1_file = File::create("rust_ms1_frag_moz_matrix.csv")?;
                for row in 0..n_frags_ms1 {
                    let row_str: String = (0..n_mz_ms1)
                        .map(|col| format!("{:.0}", ms1_frag_moz_matrix[[row, col]]))
                        .collect::<Vec<_>>()
                        .join(",");
                    writeln!(ms1_file, "{}", row_str)?;
                }

                println!("保存MS2 mask矩阵到CSV文件...");
                let mut ms2_file = File::create("rust_ms2_frag_moz_matrix.csv")?;
                for row in 0..n_frags_ms2 {
                    let row_str: String = (0..n_mz_ms2)
                        .map(|col| format!("{:.0}", ms2_frag_moz_matrix[[row, col]]))
                        .collect::<Vec<_>>()
                        .join(",");
                    writeln!(ms2_file, "{}", row_str)?;
                }

                let mut summary_file = File::create("rust_mask_matrices_summary.txt")?;
                writeln!(summary_file, "=== Rust Mask Matrices Summary ===\n")?;
                writeln!(summary_file, "MS1 Fragment Matrix:")?;
                writeln!(summary_file, "  Shape: {:?}", ms1_frag_moz_matrix.shape())?;
                writeln!(summary_file, "  Non-zero elements: {} / {}", ms1_nonzero, ms1_total)?;
                writeln!(summary_file, "  Density: {:.2}%\n", ms1_nonzero as f64 / ms1_total as f64 * 100.0)?;

                writeln!(summary_file, "MS2 Fragment Matrix:")?;
                writeln!(summary_file, "  Shape: {:?}", ms2_frag_moz_matrix.shape())?;
                writeln!(summary_file, "  Non-zero elements: {} / {}", ms2_nonzero, ms2_total)?;
                writeln!(summary_file, "  Density: {:.2}%\n", ms2_nonzero as f64 / ms2_total as f64 * 100.0)?;

                writeln!(summary_file, "Timing Information:")?;
                writeln!(summary_file, "  MS2 extraction total time: {:.2}s", ms2_processing_start.elapsed().as_secs_f64())?;
                writeln!(summary_file, "  Mask building total time: {:.2}s", mask_start.elapsed().as_secs_f64())?;

                println!("保存mask矩阵耗时: {:.2}秒", save_mask_start.elapsed().as_secs_f64());

                println!("\n已保存mask矩阵到CSV文件");

                println!("\n========== Mask矩阵构建完成 ==========");

                // ========== 构建强度矩阵 ==========
                println!("\n========== 开始构建强度矩阵 ==========");
                let intensity_start = Instant::now();

                // 获取所有唯一的RT值
                let rt_collection_start = Instant::now();
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

                println!("找到 {} 个唯一的RT值", all_rt_vec.len());
                println!("收集RT值耗时: {:.2}秒", rt_collection_start.elapsed().as_secs_f64());

                // 使用get_rt_list获取48个RT值
                let get_rt_list_start = Instant::now();
                let rt_value = precursor_feat[[i, 6]];
                let all_rt = get_rt_list(all_rt_vec, rt_value);

                println!("选择了 {} 个RT值用于构建强度矩阵", all_rt.len());
                println!("RT范围: {:.2} - {:.2}", 
                    all_rt.first().unwrap_or(&0.0), 
                    all_rt.last().unwrap_or(&0.0));
                println!("获取RT列表耗时: {:.2}秒", get_rt_list_start.elapsed().as_secs_f64());

                // 构建MS1强度矩阵
                println!("\n构建MS1强度矩阵...");
                let ms1_intensity_start = Instant::now();

                let ms1_frag_rt_matrix = build_intensity_matrix_optimized_parallel(
                    &filtered_result,
                    &ms1_extract_width_range_list.slice(s![i, .., ..]).to_owned(),
                    &ms1_frag_moz_matrix,
                    &all_rt,
                )?;

                println!("MS1强度矩阵构建完成，耗时: {:.2}秒", 
                    ms1_intensity_start.elapsed().as_secs_f64());
                println!("MS1强度矩阵形状: {:?}", ms1_frag_rt_matrix.shape());

                // 构建MS2强度矩阵
                println!("\n构建MS2强度矩阵...");
                let ms2_intensity_start = Instant::now();

                let ms2_frag_rt_matrix = build_intensity_matrix_optimized_parallel(
                    &filtered_frag_data,
                    &ms2_extract_width_range_list.slice(s![i, .., ..]).to_owned(),
                    &ms2_frag_moz_matrix,
                    &all_rt,
                )?;

                println!("MS2强度矩阵构建完成，耗时: {:.2}秒", 
                    ms2_intensity_start.elapsed().as_secs_f64());
                println!("MS2强度矩阵形状: {:?}", ms2_frag_rt_matrix.shape());

                // 重塑矩阵
                println!("\n重塑和合并矩阵...");
                let reshape_start = Instant::now();

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

                // 合并MS1和MS2矩阵
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

                let rsm_matrix = full_frag_rt_matrix.insert_axis(Axis(0));
                println!("RSM矩阵形状: {:?}", rsm_matrix.shape());
                println!("矩阵重塑耗时: {:.2}秒", reshape_start.elapsed().as_secs_f64());

                println!("\n强度矩阵构建总耗时: {:.2}秒", intensity_start.elapsed().as_secs_f64());

                // 创建最终的数据框
                println!("\n创建最终数据框...");
                let dataframe_start = Instant::now();
                let final_df = create_final_dataframe(
                    &rsm_matrix,
                    &frag_info,
                    &all_rt,
                    0,
                )?;
                
                println!("数据框创建完成！");
                println!("数据框形状: {} 行 × {} 列", final_df.height(), final_df.width());
                println!("数据框创建耗时: {:.2}秒", dataframe_start.elapsed().as_secs_f64());
                
                // 导出数据框到CSV
                let export_final_start = Instant::now();
                let mut df_file = File::create("final_intensity_data.csv")?;
                CsvWriter::new(&mut df_file)
                    .include_header(true)
                    .finish(&mut final_df.clone())?;
                println!("数据已导出到: final_intensity_data.csv");
                println!("导出最终数据耗时: {:.2}秒", export_final_start.elapsed().as_secs_f64());
                
                println!("\n========== 强度矩阵处理完成 ==========");
            }
            Err(e) => {
                eprintln!("构建库矩阵失败: {}", e);
            }
        }
    } else {
        println!("警告：没有找到匹配的库记录！");
    }
    
    println!("\n========== 特定前体MS1数据处理完成 ==========");
    
    // 程序总时间统计
    let program_total_time = program_start.elapsed();
    println!("\n程序执行完成！");
    println!("程序总运行时间: {:.2}秒", program_total_time.as_secs_f64());
    
    println!("\n结果已保存到:");
    println!("  - diann_result_rust.csv");
    println!("  - diann_precursor_id_all_rust.csv");
    println!("  - each_lib_data_rust.csv");
    println!("  - precursor_result_before_IM_filter.csv");
    println!("  - precursor_result_after_IM_filter.csv");
    println!("  - frag_result_before_IM_filter.csv");
    println!("  - frag_result_after_IM_filter.csv");
    println!("  - rust_ms1_frag_moz_matrix.csv");
    println!("  - rust_ms2_frag_moz_matrix.csv");
    println!("  - rust_mask_matrices_summary.txt");
    println!("  - final_intensity_data.csv");
    
    Ok(())
}
