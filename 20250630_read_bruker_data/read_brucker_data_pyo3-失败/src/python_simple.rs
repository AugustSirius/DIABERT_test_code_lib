// src/python_simple.rs
use pyo3::prelude::*;
use std::error::Error;

pub fn test_python_integration() -> Result<(), Box<dyn Error>> {
    Python::with_gil(|py| {
        // Test basic Python functionality
        let result = py.eval("1 + 1", None, None)?;
        println!("Python says 1 + 1 = {}", result);
        
        // Test importing pandas
        match py.import("pandas") {
            Ok(_) => println!("✓ pandas is available"),
            Err(e) => println!("✗ pandas import failed: {}", e),
        }
        
        // Test importing numpy
        match py.import("numpy") {
            Ok(_) => println!("✓ numpy is available"),
            Err(e) => println!("✗ numpy import failed: {}", e),
        }
        
        Ok(())
    })
}

pub fn run_python_script(
    library_data: &[crate::LibraryRecord],
    report_path: &str,
) -> Result<(Vec<f64>, Vec<f64>), Box<dyn Error>> {
    Python::with_gil(|py| {
        // Convert library data to Python list of dicts
        let py_library = pyo3::types::PyList::empty(py);
        
        for record in library_data.iter().take(10) { // Test with first 10 records
            let dict = pyo3::types::PyDict::new(py);
            dict.set_item("transition_group_id", &record.transition_group_id)?;
            dict.set_item("PrecursorMz", &record.precursor_mz)?;
            dict.set_item("ProductMz", &record.product_mz)?;
            dict.set_item("PeptideSequence", &record.peptide_sequence)?;
            py_library.append(dict)?;
        }
        
        // Import pandas and create DataFrame
        let locals = pyo3::types::PyDict::new(py);
        locals.set_item("library_list", py_library)?;
        
        py.run(r#"
import pandas as pd
library = pd.DataFrame(library_list)
print(f"Library shape: {library.shape}")
print(library.head())
        "#, None, Some(locals))?;
        
        // For now, return dummy data
        Ok((vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]))
    })
}