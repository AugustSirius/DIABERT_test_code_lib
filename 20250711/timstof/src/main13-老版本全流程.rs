mod utils;

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

use utils::*;

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

fn build_lib_matrix(
    lib_data: &[LibraryRecord],
    lib_cols: &LibCols,
    iso_range: f64,
    mz_max: f64,
    max_fragment: usize,
) -> Result<(Vec<Vec<String>>, Vec<MSDataArray>, Vec<MSDataArray>, Vec<Vec<f64>>), Box<dyn Error>> {
    let precursor_ids: Vec<String> = lib_data.iter()
        .map(|record| record.transition_group_id.clone())
        .collect();
    
    let precursor_groups = get_precursor_indices(&precursor_ids);
    
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
    }
    
    Ok((all_precursors, all_ms1_data, all_ms2_data, all_precursor_info))
}



fn process_library_fast(file_path: &str) -> Result<Vec<LibraryRecord>, Box<dyn Error>> {
    let file = File::open(file_path)?;
    let mut reader = ReaderBuilder::new().delimiter(b'\t').has_headers(true).from_reader(file);
    let headers = reader.headers()?.clone();
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
    
    let records: Vec<LibraryRecord> = byte_records.par_iter().map(|record| {
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
    
    Ok(records)
}









fn create_rt_im_dicts(df: &DataFrame) -> PolarsResult<(HashMap<String, f64>, HashMap<String, f64>)> {
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
    Ok((rt_dict, im_dict))
}



fn build_precursors_matrix_step1(
    ms1_data_list: &[MSDataArray], 
    ms2_data_list: &[MSDataArray], 
    device: &str
) -> Result<(Array3<f32>, Array3<f32>), Box<dyn Error>> {
    if ms1_data_list.is_empty() || ms2_data_list.is_empty() {
        return Err("MS1或MS2数据列表为空".into());
    }
    
    let batch_size = ms1_data_list.len();
    let ms1_rows = ms1_data_list[0].len();
    let ms1_cols = if !ms1_data_list[0].is_empty() { ms1_data_list[0][0].len() } else { 0 };
    let ms2_rows = ms2_data_list[0].len();
    let ms2_cols = if !ms2_data_list[0].is_empty() { ms2_data_list[0][0].len() } else { 0 };
    
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
    
    Ok((ms1_tensor, ms2_tensor))
}

fn build_precursors_matrix_step2(mut ms2_data_tensor: Array3<f32>) -> Array3<f32> {
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
    
    for i in 0..batch {
        for j in 0..rows {
            for k in 0..cols {
                let val = ms2_data_tensor[[i, j, k]];
                if val.is_infinite() || val.is_nan() {
                    ms2_data_tensor[[i, j, k]] = 0.0;
                }
            }
        }
    }
    
    ms2_data_tensor
}

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

fn build_range_matrix_step3(
    ms1_data_tensor: &Array3<f32>,
    ms2_data_tensor: &Array3<f32>,
    frag_repeat_num: usize,
    mz_unit: &str,
    mz_tol_ms1: f32,
    mz_tol_ms2: f32,
    device: &str
) -> Result<(Array3<f32>, Array3<f32>), Box<dyn Error>> {
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
    
    Ok((ms1_extract_width_range_list, ms2_extract_width_range_list))
}

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

fn build_precursors_matrix_step3(
    ms1_data_tensor: &Array3<f32>,
    ms2_data_tensor: &Array3<f32>,
    frag_repeat_num: usize,
    mz_unit: &str,
    mz_tol_ms1: f32,
    mz_tol_ms2: f32,
    device: &str
) -> Result<(Array3<f32>, Array3<f32>, Array3<f32>, Array3<f32>), Box<dyn Error>> {
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
    
    Ok((re_ms1_data_tensor, re_ms2_data_tensor, ms1_extract_width_range_list, ms2_extract_width_range_list))
}

fn build_frag_info(
    ms1_data_tensor: &Array3<f32>,
    ms2_data_tensor: &Array3<f32>,
    frag_repeat_num: usize,
    device: &str
) -> Array3<f32> {
    let ext_ms1_precursors_frag_rt_matrix = build_ext_ms1_matrix(ms1_data_tensor, device);
    let ext_ms2_precursors_frag_rt_matrix = build_ext_ms2_matrix(ms2_data_tensor, device);
    
    let ms1_shape = ext_ms1_precursors_frag_rt_matrix.shape().to_vec();
    let ms2_shape = ext_ms2_precursors_frag_rt_matrix.shape().to_vec();
    
    let batch = ms1_shape[0];
    let ms1_rows = ms1_shape[1];
    let ms2_rows = ms2_shape[1];
    
    let orig_ms1_shape = ms1_data_tensor.shape();
    let orig_ms2_shape = ms2_data_tensor.shape();
    let ms1_frag_count = orig_ms1_shape[1];
    let ms2_frag_count = orig_ms2_shape[1];
    
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
    
    frag_info
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
    
    // 构建库矩阵
    println!("\n步骤8: 构建库矩阵");
    let lib_cols = LibCols::default();
    let (precursors_list, ms1_data_list, ms2_data_list, precursor_info_list) = 
        build_lib_matrix(&each_lib_data, &lib_cols, 5.0, 1801.0, 20)?;
    
    // 构建前体特征矩阵
    println!("\n步骤9: 构建前体特征矩阵");
    let precursor_feat = create_precursor_feat(
        &precursor_info_list,
        &precursors_list,
        &assay_rt_kept_dict,
        &assay_im_kept_dict
    )?;
    
    // 构建张量
    println!("\n步骤10: 构建张量");
    let device = "cpu";
    let frag_repeat_num = 5;
    
    let (ms1_tensor, ms2_tensor) = build_precursors_matrix_step1(&ms1_data_list, &ms2_data_list, device)?;
    let ms2_tensor_processed = build_precursors_matrix_step2(ms2_tensor);
    let (ms1_range_list, ms2_range_list) = build_range_matrix_step3(
        &ms1_tensor, 
        &ms2_tensor_processed, 
        frag_repeat_num,
        "ppm",
        20.0,
        50.0,
        device
    )?;
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
    let frag_info = build_frag_info(
        &ms1_tensor,
        &ms2_tensor_processed,
        frag_repeat_num,
        device
    );
    
    // 读取TimsTOF数据
    println!("\n步骤11: 读取TimsTOF数据");
    let bruker_d_folder_name = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d";
    let timstof_data = read_timstof_data(bruker_d_folder_name)?;
    
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
    
    // 筛选和处理MS1数据
    let precursor_result = timstof_data.filter_by_mz_range(ms1_range_min, ms1_range_max);
    let precursor_result_int = convert_mz_to_integer(&precursor_result);
    
    // IM过滤
    let im_tolerance = 0.05;
    let im_min = im - im_tolerance;
    let im_max = im + im_tolerance;
    let filtered_result = filter_by_im_range(&precursor_result_int, im_min, im_max);
    
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
    let merged_frag_data_int = convert_mz_to_integer(&merged_frag_data);
    let filtered_frag_data = filter_by_im_range(&merged_frag_data_int, im_min, im_max);
    
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
    
    // 创建最终数据框
    println!("\n步骤18: 创建最终数据框");
    let final_df = create_final_dataframe(
        &rsm_matrix,
        &frag_info,
        &all_rt,
        0,
    )?;
    
    // 导出最终结果
    println!("\n步骤19: 导出最终结果");
    let mut df_file = File::create("final_intensity_data.csv")?;
    CsvWriter::new(&mut df_file)
        .include_header(true)
        .finish(&mut final_df.clone())?;
    
    let program_total_time = program_start.elapsed();
    println!("\n程序执行完成！总运行时间: {:.2}秒", program_total_time.as_secs_f64());
    
    Ok(())
}