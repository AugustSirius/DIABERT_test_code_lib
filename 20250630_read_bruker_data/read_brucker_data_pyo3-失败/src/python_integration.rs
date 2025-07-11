// src/python_integration.rs
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::collections::HashMap;
use std::error::Error;

pub struct PythonRunner {
    gil: GILPool,
}

impl PythonRunner {
    pub fn new() -> Self {
        // Initialize Python interpreter
        pyo3::prepare_freethreaded_python();
        Self {
            gil: Python::acquire_gil(),
        }
    }

    pub fn run_diann_processing(
        &self,
        library_data: &Vec<HashMap<String, String>>,
        precursor_id_list: Vec<String>,
        report_path: &str,
    ) -> Result<DiannResult, Box<dyn Error>> {
        let py = self.gil.python();
        
        // Import required modules
        let pandas = py.import("pandas")?;
        let numpy = py.import("numpy")?;
        let sys = py.import("sys")?;
        
        // Add utils.py path to Python path
        let path = sys.getattr("path")?;
        path.call_method1("append", (".",))?;
        
        // Import utils module
        let utils = py.import("utils")?;
        
        // Convert Rust data to Python
        let library_df = self.convert_to_dataframe(py, library_data)?;
        
        // Run Python code
        let locals = PyDict::new(py);
        locals.set_item("pd", pandas)?;
        locals.set_item("np", numpy)?;
        locals.set_item("utils", utils)?;
        locals.set_item("library", library_df)?;
        locals.set_item("precursor_id_list", precursor_id_list)?;
        locals.set_item("report_path", report_path)?;
        
        py.run(r#"
device = 'cpu'
frag_repeat_num = 5

# Read parquet file
report_diann = pd.read_parquet(report_path)
report_diann['transition_group_id'] = report_diann['Precursor.Id']

# Merge data
diann_result = pd.merge(
    library[['transition_group_id', 'PrecursorMz', 'ProductMz']], 
    report_diann[['transition_group_id', 'RT', 'IM','iIM']], 
    on='transition_group_id', 
    how='left'
).dropna(subset=['RT'])

# Extract unique precursors
diann_precursor_id_all = diann_result.drop_duplicates(
    subset=['transition_group_id']
)[['transition_group_id', 'RT', 'IM']].reset_index(drop=True)

# Create dictionaries
assay_rt_kept_dict = dict(zip(
    diann_precursor_id_all['transition_group_id'], 
    diann_precursor_id_all['RT']
))
assay_im_kept_dict = dict(zip(
    diann_precursor_id_all['transition_group_id'], 
    diann_precursor_id_all['IM']
))
        "#, None, Some(locals))?;
        
        // Extract results from Python
        let assay_rt_kept_dict: HashMap<String, f64> = locals
            .get_item("assay_rt_kept_dict")?
            .extract()?;
        let assay_im_kept_dict: HashMap<String, f64> = locals
            .get_item("assay_im_kept_dict")?
            .extract()?;
        
        Ok(DiannResult {
            rt_dict: assay_rt_kept_dict,
            im_dict: assay_im_kept_dict,
        })
    }
    
    fn convert_to_dataframe(
        &self,
        py: Python,
        data: &Vec<HashMap<String, String>>,
    ) -> PyResult<PyObject> {
        let pandas = py.import("pandas")?;
        
        // Create dictionary of lists for DataFrame
        let data_dict = PyDict::new(py);
        
        if !data.is_empty() {
            // Get all column names from first record
            let columns: Vec<&str> = data[0].keys().map(|s| s.as_str()).collect();
            
            // Initialize lists for each column
            for col in &columns {
                let list = PyList::empty(py);
                data_dict.set_item(col, list)?;
            }
            
            // Fill lists with data
            for record in data {
                for (key, value) in record {
                    let list = data_dict.get_item(key)?.downcast::<PyList>()?;
                    list.append(value)?;
                }
            }
        }
        
        // Create DataFrame
        let df = pandas.call_method1("DataFrame", (data_dict,))?;
        Ok(df.to_object(py))
    }
}

pub struct DiannResult {
    pub rt_dict: HashMap<String, f64>,
    pub im_dict: HashMap<String, f64>,
}