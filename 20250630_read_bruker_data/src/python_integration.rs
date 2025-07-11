use std::process::Command;
use std::fs;
use std::path::{Path, PathBuf};
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::env;

#[derive(Debug, Serialize, Deserialize)]
pub struct PythonResult {
    pub status: String,
    pub diann_result_shape: Option<Vec<usize>>,
    pub num_precursors: Option<usize>,
    pub precursor_feat_shape: Option<Vec<usize>>,
    pub frag_info_shape: Option<Vec<usize>>,
    pub output_files: Option<OutputFiles>,
    pub sample_data: Option<SampleData>,
    pub error: Option<String>,
    pub traceback: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct OutputFiles {
    pub diann_result: String,
    pub precursor_feat: String,
    pub frag_info: String,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct SampleData {
    pub first_5_precursors: Vec<Vec<serde_json::Value>>,
    pub rt_dict_sample: std::collections::HashMap<String, f64>,
    pub im_dict_sample: std::collections::HashMap<String, f64>,
}

pub struct PythonIntegration {
    python_script_path: PathBuf,
    temp_dir: String,
    python_executable: String,
}

impl PythonIntegration {
    pub fn new() -> Result<Self, Box<dyn Error>> {
        // Create temp directory if it doesn't exist
        let temp_dir = "temp".to_string();
        fs::create_dir_all(&temp_dir)?;
        
        // Try to find the Python script in different locations
        let possible_paths = vec![
            PathBuf::from("python/process_diann.py"),
            PathBuf::from("./python/process_diann.py"),
            PathBuf::from("../python/process_diann.py"),
        ];
        
        let mut python_script_path = None;
        for path in possible_paths {
            if path.exists() {
                python_script_path = Some(path.canonicalize()?);
                break;
            }
        }
        
        let python_script_path = python_script_path
            .ok_or("Could not find process_diann.py in expected locations")?;
        
        println!("Found Python script at: {}", python_script_path.display());
        
        // Use the specific conda environment "siri"
        let python_executable = "/opt/anaconda3/envs/siri/bin/python".to_string();
        
        // Verify the Python executable exists
        if !Path::new(&python_executable).exists() {
            return Err(format!("Python executable not found at: {}", python_executable).into());
        }
        
        // Verify it has the required packages
        println!("Verifying Python packages in siri environment...");
        let check_output = Command::new(&python_executable)
            .args(&["-c", "
import sys
print(f'Python: {sys.executable}')
print(f'Version: {sys.version}')
try:
    import pandas
    print('pandas: OK')
except ImportError as e:
    print(f'pandas: MISSING - {e}')
try:
    import numpy
    print('numpy: OK')
except ImportError as e:
    print(f'numpy: MISSING - {e}')
try:
    import torch
    print('torch: OK')
except ImportError as e:
    print(f'torch: MISSING - {e}')
try:
    import pyarrow
    print('pyarrow: OK')
except ImportError as e:
    print(f'pyarrow: MISSING - {e}')
"])
            .output()?;
        
        let check_result = String::from_utf8_lossy(&check_output.stdout);
        let check_error = String::from_utf8_lossy(&check_output.stderr);
        
        println!("Package check output:\n{}", check_result);
        if !check_error.is_empty() {
            eprintln!("Package check errors:\n{}", check_error);
        }
        
        // Install missing packages if needed
        if check_result.contains("MISSING") {
            println!("\nSome packages are missing. Installing them in the siri environment...");
            
            let install_output = Command::new(&python_executable)
                .args(&["-m", "pip", "install", "pandas", "numpy", "torch", "pyarrow"])
                .output()?;
            
            if !install_output.status.success() {
                eprintln!("Failed to install packages: {}", String::from_utf8_lossy(&install_output.stderr));
                return Err("Failed to install required Python packages".into());
            } else {
                println!("Successfully installed missing packages");
            }
        }
        
        println!("Using Python executable: {}", python_executable);
        
        Ok(Self {
            python_script_path,
            temp_dir,
            python_executable,
        })
    }
    
    pub fn process_diann_data(
        &self,
        library_csv_path: &str,
        report_path: &str,
        precursor_ids: Vec<String>,
    ) -> Result<PythonResult, Box<dyn Error>> {
        // Convert precursor IDs to JSON
        let precursor_ids_json = serde_json::to_string(&precursor_ids)?;
        
        // Create output directory
        let output_dir = format!("{}/diann_output_{}", self.temp_dir, chrono::Utc::now().timestamp());
        fs::create_dir_all(&output_dir)?;
        
        println!("调用Python脚本: {}", self.python_script_path.display());
        println!("使用Python解释器: {}", self.python_executable);
        println!("输出目录: {}", output_dir);
        
        // Set up the command with the conda environment
        let mut cmd = Command::new(&self.python_executable);
        
        // Make sure to use the conda environment's site-packages
        cmd.env("PYTHONPATH", "");
        
        // Add arguments
        cmd.arg(&self.python_script_path)
            .arg(library_csv_path)
            .arg(report_path)
            .arg(&precursor_ids_json)
            .arg(&output_dir);
        
        // Execute the command
        let output = cmd.output()?;
        
        // Print Python output for debugging
        if !output.stdout.is_empty() {
            println!("Python stdout:\n{}", String::from_utf8_lossy(&output.stdout));
        }
        if !output.stderr.is_empty() {
            eprintln!("Python stderr:\n{}", String::from_utf8_lossy(&output.stderr));
        }
        
        if !output.status.success() {
            let error_msg = String::from_utf8_lossy(&output.stderr);
            return Err(format!("Python脚本执行失败: {}", error_msg).into());
        }
        
        // Parse the JSON output - look for the last line that contains valid JSON
        let stdout = String::from_utf8(output.stdout)?;
        let lines: Vec<&str> = stdout.lines().collect();
        
        let mut result = None;
        for line in lines.iter().rev() {
            if line.trim().starts_with('{') {
                match serde_json::from_str::<PythonResult>(line) {
                    Ok(parsed) => {
                        result = Some(parsed);
                        break;
                    }
                    Err(_) => continue,
                }
            }
        }
        
        let result = result.ok_or("Could not parse Python output as JSON")?;
        
        if result.status == "error" {
            return Err(format!(
                "Python处理出错: {}\n{}",
                result.error.as_ref().unwrap_or(&"Unknown error".to_string()),
                result.traceback.as_ref().unwrap_or(&"".to_string())
            ).into());
        }
        
        Ok(result)
    }
    
    pub fn cleanup_temp_files(&self) -> Result<(), Box<dyn Error>> {
        // Clean up old temp files (older than 1 hour)
        let temp_path = Path::new(&self.temp_dir);
        if temp_path.exists() {
            for entry in fs::read_dir(temp_path)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    // Check if directory name contains timestamp
                    if let Some(name) = path.file_name() {
                        if let Some(name_str) = name.to_str() {
                            if name_str.starts_with("diann_output_") {
                                // You can add timestamp check here if needed
                                // For now, just skip
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }
}