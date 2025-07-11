mod utils;

use utils::{
    TimsTOFData, find_scan_for_index,
    read_parquet_with_polars, library_records_to_dataframe,
    merge_library_and_report, get_unique_precursor_ids,
    process_library_fast, create_rt_im_dicts,
    build_lib_matrix, build_precursors_matrix_step1,
    build_precursors_matrix_step2, build_range_matrix_step3,
    build_precursors_matrix_step3 as build_precursors_matrix_step3_full,
    LibCols, get_rt_list,
};

use rayon::prelude::*;
use std::{
    collections::HashMap,
    error::Error,
    path::Path,
    time::Instant,
};
use indicatif::{ProgressBar, ProgressStyle, ParallelProgressIterator};
use ndarray::{Array1, Array2, s};

use timsrust::{
    converters::ConvertableDomain,
    readers::{FrameReader, MetadataReader},
    MSLevel,
};

//-------------------------------------------------------------
//  FastChunkFinder (same public API, faster internals)
//-------------------------------------------------------------
pub struct FastChunkFinder {
    low_bounds: Vec<f64>,          // sorted ascending
    high_bounds: Vec<f64>,         // same length
    chunks:     Vec<TimsTOFData>,  // same order
}

impl FastChunkFinder {
    pub fn new(mut pairs: Vec<((f64, f64), TimsTOFData)>) -> Result<Self, Box<dyn Error>> {
        if pairs.is_empty() {
            return Err("no MS2 windows collected".into());
        }
        // sort once by low_mz
        pairs.sort_by(|a, b| a.0 .0.partial_cmp(&b.0 .0).unwrap());

        let (mut low, mut high, mut data) = (Vec::new(), Vec::new(), Vec::new());
        low.reserve(pairs.len());
        high.reserve(pairs.len());
        data.reserve(pairs.len());

        for ((l, h), d) in pairs {
            low.push(l);
            high.push(h);
            data.push(d);
        }
        Ok(Self { low_bounds: low, high_bounds: high, chunks: data })
    }

    #[inline]
    pub fn find(&self, mz: f64) -> Option<&TimsTOFData> {
        match self.low_bounds.binary_search_by(|probe| probe.partial_cmp(&mz).unwrap()) {
            Ok(idx) => Some(&self.chunks[idx]),         // exact match on lower bound
            Err(0)  => None,                            // smaller than every window
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

    pub fn range_count(&self) -> usize { self.low_bounds.len() }
}

//-------------------------------------------------------------
//  helpers
//-------------------------------------------------------------
#[inline]
fn quantize(x: f64) -> u64 { (x * 10_000.0).round() as u64 }   // 0.0001 Da grid

struct FrameSplit {
    ms1: TimsTOFData,
    ms2: Vec<((u64, u64), TimsTOFData)>,   // quantised key → data
}

/// read a Bruker run, fully parallel, zero String allocations
fn read_timstof_grouped(
    d_folder: &Path,
) -> Result<(TimsTOFData, Vec<((f64, f64), TimsTOFData)>), Box<dyn Error>> {

    let tdf_path = d_folder.join("analysis.tdf");
    let meta   = MetadataReader::new(&tdf_path)?;
    let mz_cv  = meta.mz_converter;
    let im_cv  = meta.im_converter;

    // ---------- PARALLEL pass over frames -----------------------------------
    let frames = FrameReader::new(d_folder)?;
    eprintln!("Processing {} frames...", frames.len());
    
    let pb = ProgressBar::new(frames.len() as u64);
    pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.cyan/blue}] {pos}/{len} ({eta})")
            .unwrap()
            .progress_chars("#>-"),
    );
    
    let splits: Vec<FrameSplit> = (0..frames.len()).into_par_iter().progress_with(pb).map(|idx| {
        let frame   = frames.get(idx).expect("frame read");
        let rt_min  = frame.rt_in_seconds / 60.0;
        let mut ms1 = TimsTOFData::new();
        let mut ms2_pairs: Vec<((u64,u64), TimsTOFData)> = Vec::new();

        match frame.ms_level {
            MSLevel::MS1 => {
                // one tight loop, no branching
                for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter().zip(frame.intensities.iter()).enumerate() {
                    let mz = mz_cv.convert(tof as f64);
                    let scan = find_scan_for_index(p_idx, &frame.scan_offsets);
                    let im   = im_cv.convert(scan as f64);
                    ms1.rt_values_min.push(rt_min);
                    ms1.mobility_values.push(im);
                    ms1.mz_values.push(mz);
                    ms1.intensity_values.push(intensity);
                    ms1.frame_indices.push(frame.index);
                    ms1.scan_indices.push(scan);
                }
            }
            MSLevel::MS2 => {
                // handle every quadrupole window inside this frame
                let qs = &frame.quadrupole_settings;
                for win in 0..qs.isolation_mz.len() {
                    if win >= qs.isolation_width.len() { break; }
                    let prec_mz = qs.isolation_mz[win];
                    let width   = qs.isolation_width[win];
                    let low     = prec_mz - width * 0.5;
                    let high    = prec_mz + width * 0.5;
                    let key     = (quantize(low), quantize(high));

                    let mut td = TimsTOFData::new();
                    for (p_idx, (&tof, &intensity)) in frame.tof_indices.iter().zip(frame.intensities.iter()).enumerate() {
                        let scan = find_scan_for_index(p_idx, &frame.scan_offsets);
                        if scan < qs.scan_starts[win] || scan > qs.scan_ends[win] { continue; }
                        let mz  = mz_cv.convert(tof as f64);
                        let im  = im_cv.convert(scan as f64);
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
    }).collect();

    // ---------- merge the thread-local structures ---------------------------
    eprintln!("Merging results from {} frame splits...", splits.len());
    
    let merge_pb = ProgressBar::new(splits.len() as u64);
    merge_pb.set_style(
        ProgressStyle::default_bar()
            .template("{spinner:.green} [{elapsed_precise}] [{wide_bar:.yellow/red}] {pos}/{len} merging...")
            .unwrap()
            .progress_chars("#>-"),
    );
    
    let mut global_ms1 = TimsTOFData::new();
    let mut ms2_hash: HashMap<(u64,u64), TimsTOFData> = HashMap::new();

    for split in splits {
        // MS1
        global_ms1.rt_values_min.extend(split.ms1.rt_values_min);
        global_ms1.mobility_values.extend(split.ms1.mobility_values);
        global_ms1.mz_values.extend(split.ms1.mz_values);
        global_ms1.intensity_values.extend(split.ms1.intensity_values);
        global_ms1.frame_indices.extend(split.ms1.frame_indices);
        global_ms1.scan_indices.extend(split.ms1.scan_indices);

        // MS2 windows
        for (key, mut td) in split.ms2 {
            ms2_hash.entry(key).or_insert_with(TimsTOFData::new)
                    .merge_from(&mut td);
        }
        
        merge_pb.inc(1);
    }
    
    merge_pb.finish_with_message("Merging complete!");

    // convert quantised keys back to (f64,f64) + collect into Vec
    eprintln!("Converting {} MS2 windows...", ms2_hash.len());
    let mut ms2_vec = Vec::with_capacity(ms2_hash.len());
    for ((q_low, q_high), td) in ms2_hash {
        let low  = q_low  as f64 / 10_000.0;
        let high = q_high as f64 / 10_000.0;
        ms2_vec.push(((low, high), td));
    }

    Ok((global_ms1, ms2_vec))
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

//-------------------------------------------------------------
//  main
//-------------------------------------------------------------
fn main() -> Result<(), Box<dyn Error>> {
    // Use sample D folder for first part
    let d_folder = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/DIA_sample.d";
    let d_path = Path::new(&d_folder);
    if !d_path.exists() {
        return Err(format!("folder {:?} not found", d_path).into());
    }

    println!("========== PART 1: TimsTOF Data Processing ==========");
    eprintln!("Starting TimsTOF data processing...");
    let t0 = std::time::Instant::now();
    let (ms1, ms2_pairs) = read_timstof_grouped(d_path)?;
    let decode_time = t0.elapsed();
    
    println!("✓ Total decode time: {:.2}s", decode_time.as_secs_f32());
    println!("✓ MS1 peaks  : {}", ms1.mz_values.len());
    println!("✓ MS2 windows: {}", ms2_pairs.len());

    eprintln!("Building FastChunkFinder...");
    let finder_start = std::time::Instant::now();
    let finder = FastChunkFinder::new(ms2_pairs)?;
    eprintln!("✓ FastChunkFinder built ({} ranges) in {:.2}s", 
             finder.range_count(), finder_start.elapsed().as_secs_f32());

    // Test queries
    eprintln!("Running test queries...");
    for q in [350.0, 550.0, 750.0, 950.0] {
        match finder.find(q) {
            None   => println!("m/z {:.1} → not in any window", q),
            Some(d)=> println!("m/z {:.1} → window with {} peaks", q, d.mz_values.len()),
        }
    }

    println!("\n========== PART 2: Library and Report Processing ==========");
    
    // Read library file
    let lib_file_path = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/lib/TPHPlib_frag1025_swissprot_final_all_from_Yueliang.tsv";
    let library_records = process_library_fast(lib_file_path)?;
    println!("✓ Loaded {} library records", library_records.len());
    
    // Convert to DataFrame
    eprintln!("Converting library to DataFrame...");
    let library_df = library_records_to_dataframe(library_records.clone())?;
    println!("✓ Library DataFrame created with {} rows", library_df.height());
    
    // Read DIA-NN report
    let report_file_path = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/report/report.parquet";
    eprintln!("Reading DIA-NN report...");
    let report_df = read_parquet_with_polars(report_file_path)?;
    println!("✓ Report loaded with {} rows", report_df.height());
    
    // Merge library and report
    eprintln!("Merging library and report data...");
    let diann_result = merge_library_and_report(library_df, report_df)?;
    println!("✓ Merged data: {} rows", diann_result.height());
    
    // Get unique precursor IDs
    eprintln!("Extracting unique precursor IDs...");
    let diann_precursor_id_all = get_unique_precursor_ids(&diann_result)?;
    println!("✓ Found {} unique precursors", diann_precursor_id_all.height());
    
    // Create RT and IM dictionaries
    eprintln!("Creating RT and IM lookup dictionaries...");
    let (assay_rt_kept_dict, assay_im_kept_dict) = create_rt_im_dicts(&diann_precursor_id_all)?;
    println!("✓ RT dictionary: {} entries", assay_rt_kept_dict.len());
    println!("✓ IM dictionary: {} entries", assay_im_kept_dict.len());
    
    // Get all precursor IDs with RT values
    let precursor_id_all: Vec<String> = assay_rt_kept_dict.keys().cloned().collect();
    println!("✓ Total precursors with RT: {}", precursor_id_all.len());
    
    // Device and parameters
    let device = "cpu";
    let frag_repeat_num = 5;
    println!("\n✓ Configuration:");
    println!("  - Device: {}", device);
    println!("  - Fragment repeat num: {}", frag_repeat_num);

    println!("\n========== PART 3: Single Precursor Extraction Timing ==========");
    
    // Start timing for single precursor extraction
    let start_time = Instant::now();
    
    // Select a single precursor for testing
    let precursor_id_list = vec!["VAFSAVR2"];
    println!("Processing precursor: {:?}", precursor_id_list);
    
    // Filter library data for the selected precursor
    let each_lib_data: Vec<_> = library_records.iter()
        .filter(|record| precursor_id_list.contains(&record.transition_group_id.as_str()))
        .cloned()
        .collect();
    
    if each_lib_data.is_empty() {
        println!("Warning: No matching precursor data found for {:?}", precursor_id_list);
        return Ok(());
    }
    
    println!("✓ Found {} library records for precursor", each_lib_data.len());
    
    // Build library matrix
    let lib_cols = LibCols::default();
    let (precursors_list, ms1_data_list, ms2_data_list, precursor_info_list) = 
        build_lib_matrix(
            &each_lib_data,
            &lib_cols,
            None,    // delta_rt_dict
            None,    // delta_im_dict
            5,       // frag_repeat_num
            1801.0,  // max_mz
            20,      // max_fragment
            None,    // extra_cols
        )?;
    
    println!("✓ Built library matrix:");
    println!("  - Precursors: {}", precursors_list.len());
    println!("  - MS1 data entries: {}", ms1_data_list.len());
    println!("  - MS2 data entries: {}", ms2_data_list.len());
    
    // Build precursors matrix step 1
    let (ms1_data_tensor, ms2_data_tensor) = 
        build_precursors_matrix_step1(&ms1_data_list, &ms2_data_list, device)?;
    println!("✓ Built tensor step 1:");
    println!("  - MS1 tensor shape: {:?}", ms1_data_tensor.shape());
    println!("  - MS2 tensor shape: {:?}", ms2_data_tensor.shape());
    
    // Build precursors matrix step 2
    let ms2_data_tensor = build_precursors_matrix_step2(ms2_data_tensor);
    println!("✓ Processed MS2 tensor in step 2");
    
    // Build range matrix step 3
    let (ms1_range_list, ms2_range_list) = 
        build_range_matrix_step3(&ms1_data_tensor, &ms2_data_tensor, frag_repeat_num, device)?;
    println!("✓ Built range matrices:");
    println!("  - MS1 range shape: {:?}", ms1_range_list.shape());
    println!("  - MS2 range shape: {:?}", ms2_range_list.shape());
    
    // Build precursors matrix step 3 (full)
    let (ms1_data_tensor, ms2_data_tensor, ms1_extract_width_range_list, ms2_extract_width_range_list) = 
        build_precursors_matrix_step3_full(&ms1_data_tensor, &ms2_data_tensor, frag_repeat_num, device)?;
    println!("✓ Built extract width range lists:");
    println!("  - MS1 extract width shape: {:?}", ms1_extract_width_range_list.shape());
    println!("  - MS2 extract width shape: {:?}", ms2_extract_width_range_list.shape());
    
    // Create precursor info array
    let precursor_info_np_org: Vec<Vec<f64>> = precursor_info_list.iter()
        .map(|info| info.clone())
        .collect();
    
    // Extract precursor info (first 5 columns)
    let precursor_info_choose: Vec<Vec<f64>> = precursor_info_np_org.iter()
        .map(|row| row[..5.min(row.len())].to_vec())
        .collect();
    
    // Create delta RT array (all zeros)
    let delta_rt_kept: Vec<f64> = vec![0.0; precursors_list.len()];
    
    // Get RT values from dictionary
    let assay_rt_kept: Vec<f64> = precursors_list.iter()
        .map(|prec| assay_rt_kept_dict.get(&prec[0]).copied().unwrap_or(0.0))
        .collect();
    
    // Get IM values from dictionary
    let assay_im_kept: Vec<f64> = precursors_list.iter()
        .map(|prec| assay_im_kept_dict.get(&prec[0]).copied().unwrap_or(0.0))
        .collect();
    
    println!("✓ Prepared precursor metadata:");
    println!("  - Delta RT values: {}", delta_rt_kept.len());
    println!("  - Assay RT values: {}", assay_rt_kept.len());
    println!("  - Assay IM values: {}", assay_im_kept.len());
    
    // Calculate elapsed time
    let elapsed_time = start_time.elapsed();
    println!("\n⏱️  Single precursor extraction time: {:.6} seconds", elapsed_time.as_secs_f64());
    println!("   ({:.3} ms)", elapsed_time.as_millis());
    
    println!("\n🎉 Processing complete! Total time: {:.2}s", t0.elapsed().as_secs_f32());
    
    Ok(())
}