use std::collections::HashMap;
use std::error::Error;
use std::path::Path;
use std::time::Instant;

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use timsrust::converters::ConvertableDomain;
use timsrust::readers::{FrameReader, MetadataReader};
use timsrust::MSLevel;

/* --------------------- TimsTOFData 结构体 --------------------- */

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
        Self {
            rt_values_min: Vec::new(),
            mobility_values: Vec::new(),
            mz_values: Vec::new(),
            intensity_values: Vec::new(),
            frame_indices: Vec::new(),
            scan_indices: Vec::new(),
        }
    }

    /* ------------------ 1. 卷积 / 聚合 ------------------ */
    /// mz_win=5, im_win=3, rt_win=3
    pub fn convolve(&self, mz_win: usize, im_win: usize, rt_win: usize) -> Self {
        let mut aggregated = TimsTOFData::new();
        let total_points = self.mz_values.len();
        if total_points == 0 {
            return aggregated;
        }

        /* ---------- 进度条 ---------- */
        let pb = ProgressBar::new(total_points as u64);
        pb.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%)",
            )
            .unwrap(),
        );
        pb.set_message("Convolving");

        /* ---------- m/z 和 IM 的整数索引 ---------- */
        let min_mz_idx = ((self.mz_values.iter().fold(f64::INFINITY, |a, &b| a.min(b)) * 1000.0)
            .ceil()) as i64;
        let min_im_idx = ((self
            .mobility_values
            .iter()
            .fold(f64::INFINITY, |a, &b| a.min(b))
            * 1000.0)
            .ceil()) as i64;

        /* ---------- RT 的顺序编号 ---------- */
        let mut unique_rts: Vec<f64> = self.rt_values_min.clone();
        unique_rts.sort_by(|a, b| a.partial_cmp(b).unwrap());
        unique_rts.dedup();

        // RT(bit pattern) -> ordinal
        let rt2ordinal: HashMap<u64, usize> = unique_rts
            .iter()
            .enumerate()
            .map(|(i, &rt)| (rt.to_bits(), i))
            .collect();

        /* ---------- 聚合强度 ---------- */
        let mut map: HashMap<(i64, i64, i64), u64> = HashMap::new();

        for i in 0..total_points {
            let mz_idx = ((self.mz_values[i] * 1000.0).ceil()) as i64;
            let im_idx = ((self.mobility_values[i] * 1000.0).ceil()) as i64;
            let rt_ord = *rt2ordinal
                .get(&self.rt_values_min[i].to_bits())
                .expect("RT not found") as i64;

            let mz_bin = (mz_idx - min_mz_idx) / mz_win as i64;
            let im_bin = (im_idx - min_im_idx) / im_win as i64;
            let rt_bin = rt_ord / rt_win as i64;

            *map.entry((mz_bin, im_bin, rt_bin))
                .or_insert(0) += self.intensity_values[i] as u64;

            pb.inc(1);
        }
        pb.finish_with_message("Convolution done");

        /* ---------- 反算窗口中心值 ---------- */
        for ((mz_bin, im_bin, rt_bin), intensity_sum) in map {
            let center_mz_idx = min_mz_idx + mz_bin * mz_win as i64 + mz_win as i64 / 2;
            let center_im_idx = min_im_idx + im_bin * im_win as i64 + im_win as i64 / 2;

            let rt_start_ord = rt_bin * rt_win as i64;
            let rt_len =
                ((rt_start_ord + rt_win as i64).min(unique_rts.len() as i64)) - rt_start_ord;
            let center_rt_ord = rt_start_ord + rt_len / 2;
            let center_rt = unique_rts[center_rt_ord as usize];

            aggregated.mz_values.push(center_mz_idx as f64 / 1000.0);
            aggregated.mobility_values.push(center_im_idx as f64 / 1000.0);
            aggregated.rt_values_min.push(center_rt);
            aggregated
                .intensity_values
                .push(intensity_sum.min(u32::MAX as u64) as u32);
            aggregated.frame_indices.push(0);
            aggregated.scan_indices.push(0);
        }

        aggregated
    }

    /* ------------------ 2. 其它辅助函数 ------------------ */

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

    pub fn merge(vec: Vec<TimsTOFData>) -> Self {
        let mut merged = TimsTOFData::new();
        for d in vec {
            merged.rt_values_min.extend(d.rt_values_min);
            merged.mobility_values.extend(d.mobility_values);
            merged.mz_values.extend(d.mz_values);
            merged.intensity_values.extend(d.intensity_values);
            merged.frame_indices.extend(d.frame_indices);
            merged.scan_indices.extend(d.scan_indices);
        }
        merged
    }

    pub fn print_stats(&self) {
        if self.mz_values.is_empty() {
            println!("数据为空");
            return;
        }
        let min_mz = self.mz_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_mz = self.mz_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let min_rt = self.rt_values_min.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_rt = self
            .rt_values_min
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_im = self
            .mobility_values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min);
        let max_im = self
            .mobility_values
            .iter()
            .cloned()
            .fold(f64::NEG_INFINITY, f64::max);
        let min_int = self.intensity_values.iter().min().copied().unwrap_or(0);
        let max_int = self.intensity_values.iter().max().copied().unwrap_or(0);

        println!(
            "点数 {:>8} | m/z {:.3}-{:.3} | RT {:.2}-{:.2} min | IM {:.3}-{:.3} | Int {}-{}",
            self.mz_values.len(),
            min_mz,
            max_mz,
            min_rt,
            max_rt,
            min_im,
            max_im,
            min_int,
            max_int
        );
    }
}

/* --------------------- 工具函数 --------------------- */

fn find_scan_for_index(idx: usize, scan_offsets: &[usize]) -> usize {
    scan_offsets.binary_search(&idx).unwrap_or_else(|i| i - 1)
}

/* --------------------- 读取 MS¹ / MS² --------------------- */

fn read_timstof_data_with_full_ms2(
    bruker_d_folder_path: &str,
    ms1_mz_min: f64,
    ms1_mz_max: f64,
) -> Result<Vec<TimsTOFData>, Box<dyn Error>> {
    let tdf_path = Path::new(bruker_d_folder_path).join("analysis.tdf");
    let metadata = MetadataReader::new(&tdf_path)?;
    let mz_conv = metadata.mz_converter;
    let im_conv = metadata.im_converter;

    let frame_reader = FrameReader::new(bruker_d_folder_path)?;

    let frame_results: Vec<(TimsTOFData, TimsTOFData)> = (0..frame_reader.len())
        .into_par_iter()
        .filter_map(|idx| frame_reader.get(idx).ok())
        .map(|frame| {
            let rt_min = frame.rt_in_seconds / 60.0;
            let mut ms1 = TimsTOFData::new();
            let mut ms2 = TimsTOFData::new();

            match frame.ms_level {
                MSLevel::MS1 => {
                    for (p_idx, (&tof, &intensity)) in
                        frame.tof_indices.iter().zip(frame.intensities.iter()).enumerate()
                    {
                        let mz = mz_conv.convert(tof as f64);
                        if mz < ms1_mz_min || mz > ms1_mz_max {
                            continue;
                        }
                        let scan = find_scan_for_index(p_idx, &frame.scan_offsets);
                        let im = im_conv.convert(scan as f64);

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
                    for i in 0..qs.isolation_mz.len() {
                        if i >= qs.isolation_width.len() {
                            break;
                        }
                        let precursor = qs.isolation_mz[i];
                        let iso_w = qs.isolation_width[i];
                        if precursor < ms1_mz_min - iso_w / 2.0
                            || precursor > ms1_mz_max + iso_w / 2.0
                        {
                            continue;
                        }
                        for (p_idx, (&tof, &intensity)) in
                            frame.tof_indices.iter().zip(frame.intensities.iter()).enumerate()
                        {
                            let scan = find_scan_for_index(p_idx, &frame.scan_offsets);
                            if scan < qs.scan_starts[i] || scan > qs.scan_ends[i] {
                                continue;
                            }
                            let mz = mz_conv.convert(tof as f64);
                            let im = im_conv.convert(scan as f64);

                            ms2.rt_values_min.push(rt_min);
                            ms2.mobility_values.push(im);
                            ms2.mz_values.push(mz);
                            ms2.intensity_values.push(intensity);
                            ms2.frame_indices.push(frame.index);
                            ms2.scan_indices.push(scan);
                        }
                    }
                }
                _ => {}
            }
            (ms1, ms2)
        })
        .collect();

    let (ms1_vec, ms2_vec): (Vec<_>, Vec<_>) = frame_results.into_iter().unzip();
    Ok(vec![TimsTOFData::merge(ms1_vec), TimsTOFData::merge(ms2_vec)])
}

/* --------------------- main --------------------- */

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run --release -- /path/to/file.d");
        std::process::exit(1);
    }
    let bruker_path = &args[1];

    let t0 = Instant::now();
    let mut vec = read_timstof_data_with_full_ms2(bruker_path, 94.9960, 1704.9934)?;
    let (ms1_raw, ms2_raw) = (vec.remove(0), vec.remove(0));

    println!("\n--- Raw MS1 ---");
    ms1_raw.print_stats();
    println!("--- Raw MS2 ---");
    ms2_raw.print_stats();

    println!("\n开始卷积 MS1...");
    let ms1_conv = ms1_raw.convolve(5, 3, 3);
    println!("开始卷积 MS2...");
    let ms2_conv = ms2_raw.convolve(5, 3, 3);

    println!("\n=== Convolved MS1 ===");
    ms1_conv.print_stats();
    println!("=== Convolved MS2 ===");
    ms2_conv.print_stats();

    println!("\n总耗时 {:.2} 秒", t0.elapsed().as_secs_f64());
    Ok(())
}