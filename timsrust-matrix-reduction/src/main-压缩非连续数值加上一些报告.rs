use std::{
    collections::HashMap,
    error::Error,
    path::Path,
    sync::atomic::{AtomicU64, Ordering},
    time::Instant,
};

use indicatif::{ProgressBar, ProgressStyle};
use rayon::prelude::*;
use timsrust::{
    converters::ConvertableDomain,
    readers::{FrameReader, MetadataReader},
    MSLevel,
};

/* === 小工具 === */
use std::collections::HashSet;
fn uniq(v:&[i32]) -> usize {       // i32 vector
    let mut s = v.to_vec(); s.sort_unstable(); s.dedup(); s.len()
}
fn uniq_i32(v:&[i32]) -> usize { uniq(v) }

/* =======================================================================
   数据结构
======================================================================= */
#[derive(Clone, Debug)]
pub struct TimsTOFData {
    pub rt_values_min:   Vec<f64>,
    pub mobility_values: Vec<f64>,
    pub mz_values:       Vec<f64>,
    pub intensity_values: Vec<u32>,
}

impl TimsTOFData {
    pub fn new() -> Self {
        Self {
            rt_values_min:   Vec::new(),
            mobility_values: Vec::new(),
            mz_values:       Vec::new(),
            intensity_values: Vec::new(),
        }
    }

    /* ===================================================================
       Remove zero intensity entries
    =================================================================== */
    pub fn remove_zero_intensities(&self) -> Self {
        let mut result = Self::new();
        
        for i in 0..self.intensity_values.len() {
            if self.intensity_values[i] != 0 {
                result.rt_values_min.push(self.rt_values_min[i]);
                result.mobility_values.push(self.mobility_values[i]);
                result.mz_values.push(self.mz_values[i]);
                result.intensity_values.push(self.intensity_values[i]);
            }
        }
        
        result
    }

    /* ===================================================================
       卷积（按"存在值顺序"分箱）
    =================================================================== */
    pub fn convolve(&self, mz_win: i32, im_win: i32, rt_win: i32) -> Self {
        let n = self.mz_values.len();
        if n == 0 { return Self::new(); }

        /* ---------- 1. 计算离散 index ---------- */
        let mz_idx: Vec<i32> = self.mz_values
            .iter().map(|&v| (v*1000.0).ceil() as i32).collect();
        let im_idx: Vec<i32> = self.mobility_values
            .iter().map(|&v| (v*1000.0).ceil() as i32).collect();

        /* ---------- 2. 构造 ordinal 映射 ---------- */
        let unique_mz = build_unique(&mz_idx);
        let unique_im = build_unique(&im_idx);
        let unique_rt = build_unique_f64(&self.rt_values_min);      // 已按原始数值排序

        let mz2ord: HashMap<i32,i32> = unique_mz.iter().enumerate()
            .map(|(i,&v)| (v,i as i32)).collect();
        let im2ord: HashMap<i32,i32> = unique_im.iter().enumerate()
            .map(|(i,&v)| (v,i as i32)).collect();
        let rt2ord: HashMap<u64,i32> = unique_rt.iter().enumerate()
            .map(|(i,&v)| (v.to_bits(), i as i32)).collect();

        /* ---------- 3. 进度条 ---------- */
        let bar = ProgressBar::new(n as u64);
        bar.set_style(ProgressStyle::with_template(
            "[{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({percent}%) {msg}"
        ).unwrap());

        /* ---------- 4. 并行生成 (key,intensity) ---------- */
        type Pair = (u64,u64);
        const UPDATE: u64 = 50_000;
        let counter = AtomicU64::new(0);

        let mut pairs: Vec<Pair> = (0..n).into_par_iter().fold(
            || Vec::<Pair>::with_capacity(32_000),
            |mut local, i| {
                let mz_ord = mz2ord[&mz_idx[i]];
                let im_ord = im2ord[&im_idx[i]];
                let rt_ord = rt2ord[&self.rt_values_min[i].to_bits()];

                let key: u64 = ((mz_ord / mz_win) as u64) << 42
                             | ((im_ord / im_win) as u64) << 21
                             |  (rt_ord / rt_win) as u64;
                local.push((key, self.intensity_values[i] as u64));

                if counter.fetch_add(1, Ordering::Relaxed) % UPDATE == 0 {
                    bar.inc(UPDATE);
                }
                local
            }
        ).reduce(|| Vec::new(), |mut a, mut b| { a.append(&mut b); a });

        bar.set_position(n as u64);
        bar.set_message("sorting …");
        pairs.par_sort_unstable_by_key(|&(k,_)| k);

        bar.set_message("compressing …");
        let mut out = Self::new();
        out.mz_values.reserve(pairs.len()/4);

        let mut i = 0;
        while i < pairs.len() {
            let key   = pairs[i].0;
            let mut s = pairs[i].1;
            let mut j = i+1;
            while j < pairs.len() && pairs[j].0 == key { s += pairs[j].1; j+=1; }

            let mz_bin = ((key >> 42) & 0x1FFFFF) as i32; // 21 bit
            let im_bin = ((key >> 21) & 0x1FFFFF) as i32;
            let rt_bin =  (key        & 0x1FFFFF) as i32;

            let mz_center = pick_center(&unique_mz, mz_bin, mz_win);
            let im_center = pick_center(&unique_im, im_bin, im_win);
            let rt_center = pick_center_f64(&unique_rt, rt_bin, rt_win);

            out.mz_values.push(mz_center as f64 / 1000.0);
            out.mobility_values.push(im_center as f64 / 1000.0);
            out.rt_values_min.push(rt_center);
            out.intensity_values.push(s.min(u32::MAX as u64) as u32);

            i = j;
        }
        bar.finish_with_message("done");
        out
    }

    pub fn analyze(&self, mz_win:i32, im_win:i32, rt_win:i32, tag:&str) {
        let n = self.mz_values.len() as f64;
        if n==0.0 { println!("{tag}: empty"); return; }

        // 离散索引
        let mz_idx : Vec<i32> = self.mz_values .iter().map(|&v| (v*1000.0).ceil() as i32).collect();
        let im_idx : Vec<i32> = self.mobility_values.iter().map(|&v| (v*1000.0).ceil() as i32).collect();
        let rt_ord: Vec<i32> = {
            let mut u = self.rt_values_min.clone();
            u.sort_by(|a,b| a.partial_cmp(b).unwrap());
            u.dedup();
            let map: HashMap<u64,i32> = u.iter().enumerate().map(|(i,&x)| (x.to_bits(),i as i32)).collect();
            self.rt_values_min.iter().map(|rt| map[&rt.to_bits()]).collect()
        };

        // 唯一计数
        let (n_mz, n_im, n_rt) = (uniq(&mz_idx), uniq(&im_idx), uniq_i32(&rt_ord));
        let (mz_bin, im_bin, rt_bin): (Vec<i32>, Vec<i32>, Vec<i32>) = (
            mz_idx.iter().map(|&v| v / mz_win).collect(),
            im_idx.iter().map(|&v| v / im_win).collect(),
            rt_ord.iter().map(|&v| v / rt_win).collect(),
        );
        let n_key = {
            let mut set = HashSet::with_capacity(self.mz_values.len()/8);
            for ((&a,&b),&c) in mz_bin.iter().zip(&im_bin).zip(&rt_bin) {
                set.insert( (a,b,c) );
            }
            set.len() as f64
        };

        println!("\n===== {tag} Analysis =====");
        println!("Points                         : {:.0}", n);
        println!("Unique m/z idx (*1000, ceil)   : {}  (×{:.1} repeats)", n_mz, n/n_mz as f64);
        println!("Unique IM  idx (*1000, ceil)   : {}  (×{:.1})",          n_im, n/n_im as f64);
        println!("Unique RT values               : {}  (×{:.1})",          n_rt, n/n_rt as f64);
        println!("mz bins (/{})                  : {}", mz_win, uniq(&mz_bin));
        println!("im bins (/{})                  : {}", im_win, uniq(&im_bin));
        println!("rt bins (/{})                  : {}", rt_win, uniq(&rt_bin));
        println!("3-D keys (mz,im,rt)            : {}", n_key as usize);
        println!("Theoretical max compression    : {:.1}×", n / n_key);
    }

    /* ---------- 打印 ---------- */
    pub fn print_stats(&self, tag:&str) {
        let zero_count = self.intensity_values.iter().filter(|&&x| x == 0).count();
        let total_count = self.intensity_values.len();
        let non_zero_count = total_count - zero_count;
        
        println!("{tag:<12} {:>12} points ({} non-zero, {} zero)", 
                 total_count, non_zero_count, zero_count);
    }
}

/* =======================================================================
   工具：unique + center 选择
======================================================================= */
fn build_unique(v: &[i32]) -> Vec<i32> {
    let mut u = v.to_vec();
    u.sort_unstable(); u.dedup(); u
}
fn build_unique_f64(v: &[f64]) -> Vec<f64> {
    let mut u = v.to_vec();
    u.sort_by(|a,b| a.partial_cmp(b).unwrap());
    u.dedup(); u
}
fn pick_center(u: &[i32], bin:i32, win:i32) -> i32 {
    let start = bin * win;
    let len   = ((start+win).min(u.len() as i32)) - start;
    u[(start + len/2) as usize]
}
fn pick_center_f64(u:&[f64], bin:i32, win:i32) -> f64 {
    let start = bin * win;
    let len   = ((start+win).min(u.len() as i32)) - start;
    u[(start + len/2) as usize]
}

/* =======================================================================
   读取 .d 文件（与之前相同，简化保存四列）
======================================================================= */
fn read_timstof(bruker:&str, mz_min:f64, mz_max:f64) -> Result<Vec<TimsTOFData>,Box<dyn Error>>{
    let tdf  = Path::new(bruker).join("analysis.tdf");
    let meta = MetadataReader::new(&tdf)?;
    let mzc  = meta.mz_converter;
    let imc  = meta.im_converter;
    let fr   = FrameReader::new(bruker)?;

    let frames : Vec<(TimsTOFData,TimsTOFData)> = (0..fr.len()).into_par_iter()
        .filter_map(|i| fr.get(i).ok())
        .map(|frame|{
            let rt = frame.rt_in_seconds/60.0;
            let mut ms1=TimsTOFData::new();
            let mut ms2=TimsTOFData::new();
            match frame.ms_level{
                MSLevel::MS1=>{
                    for (p,(&tof,&int)) in frame.tof_indices.iter().zip(frame.intensities.iter()).enumerate(){
                        let mz=mzc.convert(tof as f64);
                        if mz<mz_min||mz>mz_max{continue;}
                        let im=imc.convert(find_scan(p,&frame.scan_offsets) as f64);
                        ms1.rt_values_min.push(rt);
                        ms1.mobility_values.push(im);
                        ms1.mz_values.push(mz);
                        ms1.intensity_values.push(int);
                    }
                }
                MSLevel::MS2=>{
                    let qs=&frame.quadrupole_settings;
                    for iso in 0..qs.isolation_mz.len(){
                        if iso>=qs.isolation_width.len(){break;}
                        let prec=qs.isolation_mz[iso];
                        let w=qs.isolation_width[iso];
                        if prec<mz_min-w/2.0||prec>mz_max+w/2.0{continue;}
                        for (p,(&tof,&int)) in frame.tof_indices.iter().zip(frame.intensities.iter()).enumerate(){
                            let scan=find_scan(p,&frame.scan_offsets);
                            if scan<qs.scan_starts[iso]||scan>qs.scan_ends[iso]{continue;}
                            let mz=mzc.convert(tof as f64);
                            let im=imc.convert(scan as f64);
                            ms2.rt_values_min.push(rt);
                            ms2.mobility_values.push(im);
                            ms2.mz_values.push(mz);
                            ms2.intensity_values.push(int);
                        }
                    }
                }
                _=>{}
            }
            (ms1,ms2)
        }).collect();
    let (ms1,ms2):(Vec<_>,Vec<_>)=frames.into_iter().unzip();
    Ok(vec![merge(ms1),merge(ms2)])
}
fn find_scan(i:usize,off:&[usize])->usize{off.binary_search(&i).unwrap_or_else(|x|x-1)}
fn merge(v:Vec<TimsTOFData>)->TimsTOFData{
    v.into_iter().fold(TimsTOFData::new(),|mut a,b|{
        a.rt_values_min.extend(b.rt_values_min);
        a.mobility_values.extend(b.mobility_values);
        a.mz_values.extend(b.mz_values);
        a.intensity_values.extend(b.intensity_values);
        a
    })
}

/* =======================================================================
   main
======================================================================= */
fn main() -> Result<(),Box<dyn Error>>{
    let args:Vec<String>=std::env::args().collect();
    if args.len()<2{
        eprintln!("cargo run --release -- /path/to/*.d"); std::process::exit(1);
    }
    let t0=Instant::now();
    let mut data=read_timstof(&args[1],94.996,1704.9934)?;

    let (ms1_raw,ms2_raw)=(data.remove(0),data.remove(0));

    // ms1_raw.analyze(5,3,3,"MS1 raw");
    // ms2_raw.analyze(5,3,3,"MS2 raw");

    // ms1_raw.print_stats("Raw MS1");
    // ms2_raw.print_stats("Raw MS2");

    let ms1_conv=ms1_raw.convolve(5,3,3);
    let ms2_conv=ms2_raw.convolve(5,3,3);

    ms1_conv.print_stats("Conv MS1");
    ms2_conv.print_stats("Conv MS2");

    // Remove zero intensity entries
    let ms1_conv_no_zero = ms1_conv.remove_zero_intensities();
    let ms2_conv_no_zero = ms2_conv.remove_zero_intensities();

    println!("\n===== After removing zero intensities =====");
    ms1_conv_no_zero.print_stats("MS1 No-Zero");
    ms2_conv_no_zero.print_stats("MS2 No-Zero");

    // Additional analysis for filtered data
    ms1_conv_no_zero.analyze(5,3,3,"MS1 convolved (no zeros)");
    ms2_conv_no_zero.analyze(5,3,3,"MS2 convolved (no zeros)");

    println!("Elapsed {:.2}s",t0.elapsed().as_secs_f64());
    Ok(())
}