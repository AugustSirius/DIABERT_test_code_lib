use std::{
    error::Error,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use ahash::{AHashMap, AHashSet};
use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use timsrust::{
    converters::ConvertableDomain,
    readers::{FrameReader, MetadataReader},
    MSLevel,
};

/* ---------------- TimsTOFData ---------------- */

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

    /* ---------- 高效并行卷积 ---------- */
    pub fn convolve(&self, mz_win: i32, im_win: i32, rt_win: i32) -> Self {
        let n_points = self.mz_values.len();
        if n_points == 0 {
            return Self::new();
        }

        /* ------------ 预计算索引常量 ------------ */
        let min_mz_idx = ((self
            .mz_values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            * 1000.0)
            .ceil()) as i32;
        let min_im_idx = ((self
            .mobility_values
            .iter()
            .copied()
            .fold(f64::INFINITY, f64::min)
            * 1000.0)
            .ceil()) as i32;

        /* ------------ RT → ordinal ------------ */
        let mut unique_rt: Vec<f64> = {
            let mut set: AHashSet<u64> = AHashSet::with_capacity(self.rt_values_min.len());
            let mut vec = Vec::with_capacity(self.rt_values_min.len());
            for &rt in &self.rt_values_min {
                let bits = rt.to_bits();
                if set.insert(bits) {
                    vec.push(rt);
                }
            }
            vec.sort_by(|a, b| a.partial_cmp(b).unwrap());
            vec
        };
        let rt_ord_map: AHashMap<u64, usize> = unique_rt
            .iter()
            .enumerate()
            .map(|(i, &rt)| (rt.to_bits(), i))
            .collect();
        let rt_ord_vec: Vec<i32> = self
            .rt_values_min
            .iter()
            .map(|rt| rt_ord_map[&rt.to_bits()] as i32)
            .collect();

        /* ------------ 进度条 ------------ */
        const UPDATE_STEP: u64 = 50_000;
        let progress = ProgressBar::new(n_points as u64);
        progress.set_style(
            ProgressStyle::with_template(
                "[{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%)",
            )
            .unwrap(),
        );
        let counter = AtomicU64::new(0);

        /* ------------ 并行折叠 ------------ */
        type Map = AHashMap<(i32, i32, i32), u64>;
        let merge_maps = |mut a: Map, b: Map| {
            for (k, v) in b {
                *a.entry(k).or_insert(0) += v;
            }
            a
        };

        let aggregated_map: Map = (0..n_points)
            .into_par_iter()
            .fold(
                || Map::with_capacity(8_192),
                |mut local: Map, i| {
                    let mz_idx = ((self.mz_values[i] * 1000.0).ceil()) as i32;
                    let im_idx = ((self.mobility_values[i] * 1000.0).ceil()) as i32;
                    let rt_ord = rt_ord_vec[i];

                    let key = (
                        (mz_idx - min_mz_idx) / mz_win,
                        (im_idx - min_im_idx) / im_win,
                        rt_ord / rt_win,
                    );
                    *local.entry(key).or_insert(0) += self.intensity_values[i] as u64;

                    /* 进度条（降频刷新） */
                    if counter.fetch_add(1, Ordering::Relaxed) % UPDATE_STEP == 0 {
                        progress.inc(UPDATE_STEP);
                    }
                    local
                },
            )
            .reduce(|| Map::with_capacity(8_192), merge_maps);

        progress.finish_with_message("Convolution done");

        /* ------------ 生成聚合结果 ------------ */
        let mut out = TimsTOFData::new();
        out.mz_values.reserve(aggregated_map.len());

        for ((mz_bin, im_bin, rt_bin), sum_int) in aggregated_map {
            let center_mz =
                (min_mz_idx + mz_bin * mz_win + mz_win / 2) as f64 / 1000.0;
            let center_im =
                (min_im_idx + im_bin * im_win + im_win / 2) as f64 / 1000.0;

            let rt_start_ord = rt_bin * rt_win;
            let rt_len =
                ((rt_start_ord + rt_win).min(unique_rt.len() as i32)) - rt_start_ord;
            let center_rt = unique_rt[(rt_start_ord + rt_len / 2) as usize];

            out.mz_values.push(center_mz);
            out.mobility_values.push(center_im);
            out.rt_values_min.push(center_rt);
            out.intensity_values
                .push(sum_int.min(u32::MAX as u64) as u32);
            out.frame_indices.push(0);
            out.scan_indices.push(0);
        }

        out
    }

    /* ---------- merge / print_stats 与之前相同 ---------- */

    pub fn merge(v: Vec<TimsTOFData>) -> Self {
        let mut m = Self::new();
        for d in v {
            m.rt_values_min.extend(d.rt_values_min);
            m.mobility_values.extend(d.mobility_values);
            m.mz_values.extend(d.mz_values);
            m.intensity_values.extend(d.intensity_values);
            m.frame_indices.extend(d.frame_indices);
            m.scan_indices.extend(d.scan_indices);
        }
        m
    }

    pub fn print_stats(&self, name: &str) {
        if self.mz_values.is_empty() {
            println!("{name}: empty");
            return;
        }
        let (min_mz, max_mz) = self
            .mz_values
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        let (min_rt, max_rt) = self
            .rt_values_min
            .iter()
            .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| {
                (lo.min(v), hi.max(v))
            });
        println!(
            "{:<15} {:>8} pts | m/z {:.3}-{:.3} | RT {:.2}-{:.2} min",
            name,
            self.mz_values.len(),
            min_mz,
            max_mz,
            min_rt,
            max_rt
        );
    }
}

/* ------------- 其它代码：读取 .d 文件（与之前一致，略） ------------- */

fn find_scan(idx: usize, offsets: &[usize]) -> usize {
    offsets.binary_search(&idx).unwrap_or_else(|i| i - 1)
}

fn read_timstof_data_with_full_ms2(
    bruker_d_folder_path: &str,
    ms1_mz_min: f64,
    ms1_mz_max: f64,
) -> Result<Vec<TimsTOFData>, Box<dyn Error>> {
    let tdf = Path::new(bruker_d_folder_path).join("analysis.tdf");
    let meta = MetadataReader::new(&tdf)?;
    let mzc = meta.mz_converter;
    let imc = meta.im_converter;
    let fr = FrameReader::new(bruker_d_folder_path)?;

    let frames: Vec<(TimsTOFData, TimsTOFData)> = (0..fr.len())
        .into_par_iter()
        .filter_map(|i| fr.get(i).ok())
        .map(|frame| {
            let rt = frame.rt_in_seconds / 60.0;
            let mut ms1 = TimsTOFData::new();
            let mut ms2 = TimsTOFData::new();

            match frame.ms_level {
                MSLevel::MS1 => {
                    for (p_idx, (&tof, &int)) in
                        frame.tof_indices.iter().zip(frame.intensities.iter()).enumerate()
                    {
                        let mz = mzc.convert(tof as f64);
                        if mz < ms1_mz_min || mz > ms1_mz_max {
                            continue;
                        }
                        let scan = find_scan(p_idx, &frame.scan_offsets);
                        let im = imc.convert(scan as f64);

                        ms1.rt_values_min.push(rt);
                        ms1.mobility_values.push(im);
                        ms1.mz_values.push(mz);
                        ms1.intensity_values.push(int);
                    }
                }
                MSLevel::MS2 => {
                    let qs = &frame.quadrupole_settings;
                    for iso_idx in 0..qs.isolation_mz.len() {
                        if iso_idx >= qs.isolation_width.len() {
                            break;
                        }
                        let prec = qs.isolation_mz[iso_idx];
                        let iso_w = qs.isolation_width[iso_idx];
                        if prec < ms1_mz_min - iso_w / 2.0
                            || prec > ms1_mz_max + iso_w / 2.0
                        {
                            continue;
                        }
                        for (p_idx, (&tof, &int)) in frame
                            .tof_indices
                            .iter()
                            .zip(frame.intensities.iter())
                            .enumerate()
                        {
                            let scan = find_scan(p_idx, &frame.scan_offsets);
                            if scan < qs.scan_starts[iso_idx]
                                || scan > qs.scan_ends[iso_idx]
                            {
                                continue;
                            }
                            let mz = mzc.convert(tof as f64);
                            let im = imc.convert(scan as f64);

                            ms2.rt_values_min.push(rt);
                            ms2.mobility_values.push(im);
                            ms2.mz_values.push(mz);
                            ms2.intensity_values.push(int);
                        }
                    }
                }
                _ => {}
            }
            (ms1, ms2)
        })
        .collect();

    let (ms1_vec, ms2_vec): (Vec<_>, Vec<_>) = frames.into_iter().unzip();
    Ok(vec![TimsTOFData::merge(ms1_vec), TimsTOFData::merge(ms2_vec)])
}

/* ----------------------- main ----------------------- */

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: cargo run --release -- path/to/*.d");
        std::process::exit(1);
    }
    let t0 = Instant::now();
    let mut v = read_timstof_data_with_full_ms2(&args[1], 94.996, 1704.9934)?;
    let (ms1_raw, ms2_raw) = (v.remove(0), v.remove(0));

    ms1_raw.print_stats("Raw MS1");
    ms2_raw.print_stats("Raw MS2");

    println!("Convolving MS1 …");
    let ms1_conv = ms1_raw.convolve(5, 3, 3);
    println!("Convolving MS2 …");
    let ms2_conv = ms2_raw.convolve(5, 3, 3);

    ms1_conv.print_stats("Conv MS1");
    ms2_conv.print_stats("Conv MS2");

    println!("Total elapsed: {:.2} s", t0.elapsed().as_secs_f64());
    Ok(())
}