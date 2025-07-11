mod python_integration;

use std::fs::{self, File};
use std::collections::HashMap;
use std::error::Error;
use rayon::prelude::*;
use csv::{ReaderBuilder, ByteRecord, Writer};
use python_integration::PythonIntegration;

// 使用结构体代替HashMap来存储数据
#[derive(Debug, Clone)]
struct LibraryRecord {
    transition_group_id: String,
    peptide_sequence: String,
    full_unimod_peptide_name: String,
    precursor_charge: String,
    precursor_mz: String,
    tr_recalibrated: String,
    product_mz: String,
    fragment_type: String,
    fragment_charge: String,
    fragment_number: String,
    library_intensity: String,
    protein_id: String,
    protein_name: String,
    gene: String,
    decoy: String,
    other_columns: HashMap<String, String>,
}

fn get_lib_col_dict() -> HashMap<&'static str, &'static str> {
    let mut lib_col_dict = HashMap::new();

    // transition_group_id mappings
    for key in ["transition_group_id", "PrecursorID"] {
        lib_col_dict.insert(key, "transition_group_id");
    }

    // PeptideSequence mappings
    for key in ["PeptideSequence", "Sequence", "StrippedPeptide"] {
        lib_col_dict.insert(key, "PeptideSequence");
    }

    // FullUniModPeptideName mappings
    for key in [
        "FullUniModPeptideName",
        "ModifiedPeptide",
        "LabeledSequence",
        "modification_sequence",
        "ModifiedPeptideSequence",
    ] {
        lib_col_dict.insert(key, "FullUniModPeptideName");
    }

    // PrecursorCharge mappings
    for key in ["PrecursorCharge", "Charge", "prec_z"] {
        lib_col_dict.insert(key, "PrecursorCharge");
    }

    // PrecursorMz mappings
    for key in ["PrecursorMz", "Q1"] {
        lib_col_dict.insert(key, "PrecursorMz");
    }

    // Tr_recalibrated mappings
    for key in [
        "Tr_recalibrated",
        "iRT",
        "RetentionTime",
        "NormalizedRetentionTime",
        "RT_detected",
    ] {
        lib_col_dict.insert(key, "Tr_recalibrated");
    }

    // ProductMz mappings
    for key in ["ProductMz", "FragmentMz", "Q3"] {
        lib_col_dict.insert(key, "ProductMz");
    }

    // FragmentType mappings
    for key in [
        "FragmentType",
        "FragmentIonType",
        "ProductType",
        "ProductIonType",
        "frg_type",
    ] {
        lib_col_dict.insert(key, "FragmentType");
    }

    // FragmentCharge mappings
    for key in [
        "FragmentCharge",
        "FragmentIonCharge",
        "ProductCharge",
        "ProductIonCharge",
        "frg_z",
    ] {
        lib_col_dict.insert(key, "FragmentCharge");
    }

    // FragmentNumber mappings
    for key in ["FragmentNumber", "frg_nr", "FragmentSeriesNumber"] {
        lib_col_dict.insert(key, "FragmentNumber");
    }

    // LibraryIntensity mappings
    for key in [
        "LibraryIntensity",
        "RelativeIntensity",
        "RelativeFragmentIntensity",
        "RelativeFragmentIonIntensity",
        "relative_intensity",
    ] {
        lib_col_dict.insert(key, "LibraryIntensity");
    }

    // ProteinID mappings
    for key in ["ProteinID", "ProteinId", "UniprotID", "uniprot_id", "UniProtIds"] {
        lib_col_dict.insert(key, "ProteinID");
    }

    // ProteinName mappings
    for key in ["ProteinName", "Protein Name", "Protein_name", "protein_name"] {
        lib_col_dict.insert(key, "ProteinName");
    }

    // Gene mappings
    for key in ["Gene", "Genes", "GeneName"] {
        lib_col_dict.insert(key, "Gene");
    }

    // Decoy mappings
    for key in ["Decoy", "decoy"] {
        lib_col_dict.insert(key, "decoy");
    }

    lib_col_dict
}

// 高性能读取函数
fn process_library_fast(file_path: &str) -> Result<Vec<LibraryRecord>, Box<dyn Error>> {
    use std::time::Instant;
    let start = Instant::now();
    
    println!("正在使用高性能模式读取文件: {}", file_path);
    
    let file = File::open(file_path)?;
    let mut reader = ReaderBuilder::new()
        .delimiter(b'\t')
        .has_headers(true)
        .from_reader(file);

    let headers = reader.headers()?.clone();
    println!("成功读取表头，共 {} 列", headers.len());
    
    // 创建列索引映射
    let mut column_indices = HashMap::new();
    for (i, header) in headers.iter().enumerate() {
        column_indices.insert(header, i);
    }
    
    let lib_col_dict = get_lib_col_dict();
    
    // 预先计算需要的列索引
    let mut mapped_indices: HashMap<&str, usize> = HashMap::new();
    for (old_col, new_col) in &lib_col_dict {
        if let Some(&idx) = column_indices.get(old_col) {
            mapped_indices.insert(new_col, idx);
        }
    }

    // 收集所有记录到内存
    let mut byte_records = Vec::new();
    for result in reader.byte_records() {
        byte_records.push(result?);
    }
    
    let total_records = byte_records.len();
    println!("已读取 {} 条原始记录，开始并行处理...", total_records);
    
    // 创建进度跟踪
    let processed = std::sync::atomic::AtomicUsize::new(0);
    
    // 使用rayon并行处理记录
    let records: Vec<LibraryRecord> = byte_records
        .par_iter()
        .map(|record| {
            let current = processed.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            if current % 100000 == 0 && current > 0 {
                println!("已处理 {} 条记录...", current);
            }
            
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

            // 快速填充字段
            if let Some(&idx) = mapped_indices.get("PeptideSequence") {
                if let Some(val) = record.get(idx) {
                    rec.peptide_sequence = String::from_utf8_lossy(val).into_owned();
                }
            }
            
            if let Some(&idx) = mapped_indices.get("FullUniModPeptideName") {
                if let Some(val) = record.get(idx) {
                    rec.full_unimod_peptide_name = String::from_utf8_lossy(val).into_owned();
                }
            }
            
            if let Some(&idx) = mapped_indices.get("PrecursorCharge") {
                if let Some(val) = record.get(idx) {
                    rec.precursor_charge = String::from_utf8_lossy(val).into_owned();
                }
            }
            
            if let Some(&idx) = mapped_indices.get("PrecursorMz") {
                if let Some(val) = record.get(idx) {
                    rec.precursor_mz = String::from_utf8_lossy(val).into_owned();
                }
            }
            
            if let Some(&idx) = mapped_indices.get("ProductMz") {
                if let Some(val) = record.get(idx) {
                    rec.product_mz = String::from_utf8_lossy(val).into_owned();
                }
            }
            
            if let Some(&idx) = mapped_indices.get("FragmentType") {
                if let Some(val) = record.get(idx) {
                    let fragment_str = String::from_utf8_lossy(val);
                    rec.fragment_type = match fragment_str.as_ref() {
                        "b" => "1".to_string(),
                        "y" => "2".to_string(),
                        "p" => "3".to_string(),
                        _ => fragment_str.into_owned(),
                    };
                }
            }
            
            if let Some(&idx) = mapped_indices.get("FragmentCharge") {
                if let Some(val) = record.get(idx) {
                    rec.fragment_charge = String::from_utf8_lossy(val).into_owned();
                }
            }
            
            if let Some(&idx) = mapped_indices.get("FragmentNumber") {
                if let Some(val) = record.get(idx) {
                    rec.fragment_number = String::from_utf8_lossy(val).into_owned();
                }
            }
            
            if let Some(&idx) = mapped_indices.get("LibraryIntensity") {
                if let Some(val) = record.get(idx) {
                    rec.library_intensity = String::from_utf8_lossy(val).into_owned();
                }
            }
            
            if let Some(&idx) = mapped_indices.get("Tr_recalibrated") {
                if let Some(val) = record.get(idx) {
                    rec.tr_recalibrated = String::from_utf8_lossy(val).into_owned();
                }
            }
            
            if let Some(&idx) = mapped_indices.get("ProteinID") {
                if let Some(val) = record.get(idx) {
                    rec.protein_id = String::from_utf8_lossy(val).into_owned();
                }
            }
            
            if let Some(&idx) = mapped_indices.get("ProteinName") {
                if let Some(val) = record.get(idx) {
                    rec.protein_name = String::from_utf8_lossy(val).into_owned();
                }
            }
            
            if let Some(&idx) = mapped_indices.get("Gene") {
                if let Some(val) = record.get(idx) {
                    rec.gene = String::from_utf8_lossy(val).into_owned();
                }
            }
            
            if let Some(&idx) = mapped_indices.get("decoy") {
                if let Some(val) = record.get(idx) {
                    rec.decoy = String::from_utf8_lossy(val).into_owned();
                }
            }

            // 生成 transition_group_id
            rec.transition_group_id = format!("{}{}", rec.full_unimod_peptide_name, rec.precursor_charge);
            
            rec
        })
        .collect();

    let elapsed = start.elapsed();
    println!("处理完成！共 {} 条记录，耗时: {:.2}秒", records.len(), elapsed.as_secs_f64());
    
    Ok(records)
}

// 导出到CSV的函数（用于Python处理）
fn export_library_to_csv(records: &[LibraryRecord], output_path: &str) -> Result<(), Box<dyn Error>> {
    // 确保目录存在
    if let Some(parent) = std::path::Path::new(output_path).parent() {
        fs::create_dir_all(parent)?;
    }
    
    let file = File::create(output_path)?;
    let mut wtr = Writer::from_writer(file);
    
    // 写入表头 - 包含所有Python需要的列
    wtr.write_record(&[
        "transition_group_id",
        "PeptideSequence", 
        "FullUniModPeptideName",
        "PrecursorCharge",
        "PrecursorMz",
        "ProductMz",
        "FragmentType",
        "FragmentCharge",
        "FragmentNumber",
        "LibraryIntensity",
        "Tr_recalibrated",
        "ProteinID",
        "ProteinName",
        "Gene",
        "decoy"
    ])?;
    
    // 写入数据
    for record in records {
        wtr.write_record(&[
            &record.transition_group_id,
            &record.peptide_sequence,
            &record.full_unimod_peptide_name,
            &record.precursor_charge,
            &record.precursor_mz,
            &record.product_mz,
            &record.fragment_type,
            &record.fragment_charge,
            &record.fragment_number,
            &record.library_intensity,
            &record.tr_recalibrated,
            &record.protein_id,
            &record.protein_name,
            &record.gene,
            &record.decoy,
        ])?;
    }
    
    wtr.flush()?;
    println!("库文件已导出到: {}", output_path);
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    println!("程序开始运行...");
    
    // 创建必要的目录
    fs::create_dir_all("temp")?;
    fs::create_dir_all("python")?;
    
    let file_path = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/lib/TPHPlib_frag1025_swissprot_final_all_from_Yueliang.tsv";
    
    match process_library_fast(file_path) {
        Ok(library) => {
            println!("\n总共加载了 {} 条记录", library.len());
            
            // 导出库文件供Python使用
            let library_csv_path = "temp/library_export.csv";
            export_library_to_csv(&library, library_csv_path)?;
            
            // 设置Python处理参数
            let report_path = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/report/report.parquet";
            let precursor_id_list = vec!["AAAAAAALQAK2".to_string()]; // 示例precursor IDs
            
            // 初始化Python集成
            println!("\n初始化Python集成...");
            match PythonIntegration::new() {
                Ok(python_integration) => {
                    // 调用Python处理
                    println!("\n开始Python处理...");
                    match python_integration.process_diann_data(
                        library_csv_path,
                        report_path,
                        precursor_id_list.clone()
                    ) {
                        Ok(result) => {
                            // 显示结果
                            println!("\n处理结果:");
                            println!("状态: {}", result.status);
                            
                            if let Some(shape) = &result.diann_result_shape {
                                println!("DIANN结果形状: {:?}", shape);
                            }
                            
                            if let Some(num) = result.num_precursors {
                                println!("前体数量: {}", num);
                            }
                            
                            if let Some(shape) = &result.precursor_feat_shape {
                                println!("前体特征形状: {:?}", shape);
                            }
                            
                            if let Some(shape) = &result.frag_info_shape {
                                println!("片段信息形状: {:?}", shape);
                            }
                            
                            if let Some(files) = &result.output_files {
                                println!("\n输出文件:");
                                println!("  DIANN结果: {}", files.diann_result);
                                println!("  前体特征: {}", files.precursor_feat);
                                println!("  片段信息: {}", files.frag_info);
                            }
                            
                            if let Some(sample_data) = &result.sample_data {
                                println!("\n样本数据:");
                                println!("  前5个前体: {:?}", sample_data.first_5_precursors);
                                println!("  RT字典样本: {:?}", sample_data.rt_dict_sample);
                                println!("  IM字典样本: {:?}", sample_data.im_dict_sample);
                            }
                        }
                        Err(e) => {
                            eprintln!("Python处理错误: {}", e);
                        }
                    }
                }
                Err(e) => {
                    eprintln!("初始化Python集成错误: {}", e);
                }
            }
        }
        Err(e) => {
            eprintln!("读取库文件错误: {}", e);
            return Err(e);
        }
    }

    Ok(())
}