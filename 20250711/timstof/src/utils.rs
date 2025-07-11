use std::collections::{HashMap, HashSet};
use ndarray::{Array1, Array2, Array3};
use std::cmp::Ordering;
use std::error::Error;
use timsrust::converters::ConvertableDomain;
use polars::prelude::*;
use std::fs::File;
use std::io::Write;
use crate::TimsTOFData;
use rayon::prelude::*;
use csv::Writer;

// 设备类型枚举
#[derive(Debug, Clone, Copy)]
pub enum Device {
    Cpu,
}

impl Device {
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "cpu" => Device::Cpu,
            _ => Device::Cpu,
        }
    }
}

// 常量定义
pub const MS1_ISOTOPE_COUNT: usize = 6;
pub const FRAGMENT_VARIANTS: usize = 3;
pub const MS1_TYPE_MARKER: f64 = 5.0;
pub const MS1_FRAGMENT_TYPE: f64 = 1.0;
pub const VARIANT_ORIGINAL: f64 = 2.0;
pub const VARIANT_LIGHT: f64 = 3.0;
pub const VARIANT_HEAVY: f64 = 4.0;

// 库列名映射结构体
#[derive(Debug, Clone)]
pub struct LibCols {
    pub precursor_mz_col: &'static str,
    pub irt_col: &'static str,
    pub precursor_id_col: &'static str,
    pub full_sequence_col: &'static str,
    pub pure_sequence_col: &'static str,
    pub precursor_charge_col: &'static str,
    pub fragment_mz_col: &'static str,
    pub fragment_series_col: &'static str,
    pub fragment_charge_col: &'static str,
    pub fragment_type_col: &'static str,
    pub lib_intensity_col: &'static str,
    pub protein_name_col: &'static str,
    pub decoy_or_not_col: &'static str,
}

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

pub type MSDataArray = Vec<Vec<f64>>;

#[derive(Debug, Clone)]
pub struct LibraryRecord {
    pub transition_group_id: String,
    pub peptide_sequence: String,
    pub full_unimod_peptide_name: String,
    pub precursor_charge: String,
    pub precursor_mz: String,
    pub tr_recalibrated: String,
    pub product_mz: String,
    pub fragment_type: String,
    pub fragment_charge: String,
    pub fragment_number: String,
    pub library_intensity: String,
    pub protein_id: String,
    pub protein_name: String,
    pub gene: String,
    pub decoy: String,
    pub other_columns: HashMap<String, String>,
}

pub fn find_scan_for_index(index: usize, scan_offsets: &[usize]) -> usize {
    for (scan, window) in scan_offsets.windows(2).enumerate() {
        if index >= window[0] && index < window[1] {
            return scan;
        }
    }
    scan_offsets.len() - 1
}

pub fn get_rt_list(mut lst: Vec<f64>, target: f64) -> Vec<f64> {
    lst.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
    
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

pub fn intercept_frags_sort(mut fragment_list: Vec<f64>, max_length: usize) -> Vec<f64> {
    fragment_list.sort_by(|a, b| b.partial_cmp(a).unwrap_or(Ordering::Equal));
    fragment_list.truncate(max_length);
    fragment_list
}

pub fn get_precursor_indices(precursor_ids: &[String]) -> Vec<Vec<usize>> {
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

pub fn get_lib_col_dict() -> HashMap<&'static str, &'static str> {
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

pub fn build_ext_ms1_matrix(ms1_data_tensor: &Array3<f32>, device: &str) -> Array3<f32> {
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

pub fn build_ext_ms2_matrix(ms2_data_tensor: &Array3<f32>, device: &str) -> Array3<f32> {
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

// 未使用的函数
pub fn build_intensity_matrix(
    data: &crate::TimsTOFData,
    extract_width_range_list: &Array2<f32>,
    frag_moz_matrix: &Array2<f32>,
    all_rt: &[f64],
) -> Result<Array2<f32>, Box<dyn std::error::Error>> {
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
            
            let frag_moz_row = frag_moz_matrix.slice(ndarray::s![a, ..]);
            let intensity_sum: f32 = frag_moz_row.iter()
                .zip(mapped_intensities.iter())
                .map(|(&mask, &intensity)| mask * intensity)
                .sum();
            
            frag_rt_matrix[[a, rt_idx]] = intensity_sum;
        }
    }
    
    Ok(frag_rt_matrix)
}

// 未使用的函数
pub fn process_ms1_frame(
    frame: &timsrust::Frame,
    rt_min: f64,
    ms1_mz_min: f64,
    ms1_mz_max: f64,
    mz_converter: &timsrust::converters::Tof2MzConverter,
    im_converter: &timsrust::converters::Scan2ImConverter,
    ms1_data: &mut crate::TimsTOFData,
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

// 未使用的函数
pub fn process_ms2_frame(
    frame: &timsrust::Frame,
    rt_min: f64,
    ms1_mz_min: f64,
    ms1_mz_max: f64,
    mz_converter: &timsrust::converters::Tof2MzConverter,
    im_converter: &timsrust::converters::Scan2ImConverter,
    ms2_windows: &mut HashMap<String, crate::TimsTOFData>,
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
        let window_data = ms2_windows.entry(window_key).or_insert_with(crate::TimsTOFData::new);
        
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



// ----------------------------- Unused functions -----------------------------


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
    
    let mut frag_rt_matrix = Array2::<f32>::zeros((n_frags, n_rt));
    
    for a in 0..n_frags {
        let mut moz_list: Vec<i64> = Vec::new();
        for j in 0..extract_width_range_list.shape()[1] {
            let mz = extract_width_range_list[[a, j]] as i64;
            moz_list.push(mz);
        }
        
        let mut mz_rt_matrix = Array2::<f32>::zeros((moz_list.len(), n_rt));
        for (j, &mz) in moz_list.iter().enumerate() {
            if let Some(&mz_idx) = mz_to_idx.get(&mz) {
                for k in 0..n_rt {
                    mz_rt_matrix[[j, k]] = pivot_matrix[[mz_idx, k]];
                }
            }
        }
        
        for j in 0..moz_list.len() {
            for k in 0..n_rt {
                frag_rt_matrix[[a, k]] += frag_moz_matrix[[a, j]] * mz_rt_matrix[[j, k]];
            }
        }
    }
    
    Ok(frag_rt_matrix)
}



pub fn export_polars_to_csv(df: &mut DataFrame, output_path: &str) -> PolarsResult<()> {
    let mut file = File::create(output_path)?;
    CsvWriter::new(&mut file).include_header(true).finish(df)?;
    Ok(())
}


// ----------------------------- End of unused functions -----------------------------

// ----------------------------- Moved functions from main.rs -----------------------------

pub fn read_parquet_with_polars(file_path: &str) -> PolarsResult<DataFrame> {
    let file = File::open(file_path)?;
    let mut df = ParquetReader::new(file).finish()?;
    let new_col = df.column("Precursor.Id")?.clone().with_name("transition_group_id");
    df.with_column(new_col)?;
    Ok(df)
}

pub fn export_to_csv(records: &[LibraryRecord], output_path: &str) -> Result<(), Box<dyn Error>> {
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
    Ok(())
}

pub fn library_records_to_dataframe(records: Vec<LibraryRecord>) -> PolarsResult<DataFrame> {
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
    Ok(df)
}

pub fn merge_library_and_report(library_df: DataFrame, report_df: DataFrame) -> PolarsResult<DataFrame> {
    let report_selected = report_df.select(["transition_group_id", "RT", "IM", "iIM"])?;
    let merged = library_df.join(&report_selected, ["transition_group_id"], ["transition_group_id"], JoinArgs::new(JoinType::Left))?;
    let rt_col = merged.column("RT")?;
    let mask = rt_col.is_not_null();
    let filtered = merged.filter(&mask)?;
    let reordered = filtered.select(["transition_group_id", "PrecursorMz", "ProductMz", "RT", "IM", "iIM"])?;
    Ok(reordered)
}

pub fn get_unique_precursor_ids(diann_result: &DataFrame) -> PolarsResult<DataFrame> {
    let unique_df = diann_result.unique(Some(&["transition_group_id".to_string()]), UniqueKeepStrategy::First, None)?;
    let selected_df = unique_df.select(["transition_group_id", "RT", "IM"])?;
    Ok(selected_df)
}

pub fn create_precursor_feat(
    precursor_info_list: &[Vec<f64>],
    precursors_list: &[Vec<String>],
    assay_rt_kept_dict: &HashMap<String, f64>,
    assay_im_kept_dict: &HashMap<String, f64>,
) -> Result<Array2<f64>, Box<dyn Error>> {
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
    
    Ok(precursor_feat)
}

pub fn filter_library_by_precursor_ids(library: &[LibraryRecord], precursor_id_list: &[String]) -> Vec<LibraryRecord> {
    let id_set: HashSet<&String> = precursor_id_list.iter().collect();
    let filtered: Vec<LibraryRecord> = library.par_iter().filter(|record| id_set.contains(&record.transition_group_id)).cloned().collect();
    filtered
}

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

// ----------------------------- End of moved functions -----------------------------