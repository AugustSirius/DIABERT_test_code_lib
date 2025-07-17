// src/output.rs
use std::path::{Path, PathBuf};
use std::fs::{self, File};
use std::io::Write;
use serde::{Serialize, Deserialize};
use chrono::Local;

#[derive(Serialize, Deserialize, Debug)]
pub struct SliceResult {
    pub version: String,
    pub timestamp: String,
    pub precursor_id: String,
    pub ms1_data: MS1Data,
    pub ms2_data: MS2Data,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MS1Data {
    pub mz_range: (f64, f64),
    pub im_range: (f64, f64),
    pub data_points: usize,
    pub mz_values: Vec<f64>,
    pub intensity_values: Vec<u32>,
    pub rt_values: Vec<f64>,
    pub mobility_values: Vec<f64>,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MS2Data {
    pub num_fragments: usize,
    pub data_points: usize,
    pub mz_values: Vec<f64>,
    pub intensity_values: Vec<u32>,
    pub rt_values: Vec<f64>,
    pub mobility_values: Vec<f64>,
}

pub struct OutputManager {
    output_dir: PathBuf,
}

impl OutputManager {
    pub fn new() -> Self {
        let output_dir = PathBuf::from("/Users/augustsirius/Desktop/DIABERT_test_code_lib/20250714/timstof-slice-optimize/MS1MS2切片结果对比");
        Self { output_dir }
    }
    
    pub fn ensure_output_dir(&self) -> std::io::Result<()> {
        fs::create_dir_all(&self.output_dir)?;
        Ok(())
    }
    
    pub fn save_slice_result(&self, result: &SliceResult) -> std::io::Result<PathBuf> {
        self.ensure_output_dir()?;
        
        let filename = format!(
            "{}_{}_{}.json",
            result.version,
            result.precursor_id,
            result.timestamp.replace(":", "-").replace(" ", "_")
        );
        
        let file_path = self.output_dir.join(filename);
        let json_content = serde_json::to_string_pretty(result)?;
        
        let mut file = File::create(&file_path)?;
        file.write_all(json_content.as_bytes())?;
        
        println!("保存切片结果到: {:?}", file_path);
        Ok(file_path)
    }
}

// Helper function to convert TimsTOFData to our output format
pub fn create_ms1_data(
    data: &crate::utils::TimsTOFData,
    mz_range: (f64, f64),
    im_range: (f64, f64),
) -> MS1Data {
    MS1Data {
        mz_range,
        im_range,
        data_points: data.mz_values.len(),
        mz_values: data.mz_values.clone(),
        intensity_values: data.intensity_values.clone(),
        rt_values: data.rt_values_min.clone(),
        mobility_values: data.mobility_values.clone(),
    }
}

pub fn create_ms2_data(
    data: &crate::utils::TimsTOFData,
    num_fragments: usize,
) -> MS2Data {
    MS2Data {
        num_fragments,
        data_points: data.mz_values.len(),
        mz_values: data.mz_values.clone(),
        intensity_values: data.intensity_values.clone(),
        rt_values: data.rt_values_min.clone(),
        mobility_values: data.mobility_values.clone(),
    }
}