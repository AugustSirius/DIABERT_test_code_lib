use std::fs::File;
use std::io::{BufRead, BufReader};
use std::collections::HashMap;
use std::error::Error;
use std::time::Instant;
use rayon::prelude::*;

#[derive(Debug)]
struct ComparisonResult {
    total_rows_file1: usize,
    total_rows_file2: usize,
    matching_rows: usize,
    different_rows: usize,
    column_differences: HashMap<String, usize>,
    sample_differences: Vec<DifferenceDetail>,
}

#[derive(Debug)]
struct DifferenceDetail {
    row_number: usize,
    column: String,
    value1: String,
    value2: String,
}

fn compare_csv_files(file1_path: &str, file2_path: &str) -> Result<ComparisonResult, Box<dyn Error>> {
    let start = Instant::now();
    println!("开始比较CSV文件...");
    println!("文件1: {}", file1_path);
    println!("文件2: {}", file2_path);

    // 读取第一个文件
    let file1 = File::open(file1_path)?;
    let reader1 = BufReader::new(file1);
    let mut lines1: Vec<String> = reader1.lines().collect::<Result<_, _>>()?;
    
    // 读取第二个文件
    let file2 = File::open(file2_path)?;
    let reader2 = BufReader::new(file2);
    let mut lines2: Vec<String> = reader2.lines().collect::<Result<_, _>>()?;

    println!("文件读取完成，耗时: {:.2}秒", start.elapsed().as_secs_f64());

    // 获取表头
    let headers1 = lines1.get(0).ok_or("文件1为空")?;
    let headers2 = lines2.get(0).ok_or("文件2为空")?;

    if headers1 != headers2 {
        println!("警告：文件表头不一致!");
        println!("文件1表头: {}", headers1);
        println!("文件2表头: {}", headers2);
    }

    let headers: Vec<&str> = headers1.split(',').collect();
    let column_count = headers.len();

    // 准备结果
    let mut result = ComparisonResult {
        total_rows_file1: lines1.len() - 1,
        total_rows_file2: lines2.len() - 1,
        matching_rows: 0,
        different_rows: 0,
        column_differences: HashMap::new(),
        sample_differences: Vec::new(),
    };

    // 初始化列差异计数
    for header in &headers {
        result.column_differences.insert(header.to_string(), 0);
    }

    // 并行比较数据行
    let min_rows = std::cmp::min(lines1.len(), lines2.len());
    
    println!("开始比较 {} 行数据...", min_rows - 1);

    // 使用并行处理比较行
    let differences: Vec<(usize, Vec<(usize, String, String, String)>)> = (1..min_rows)
        .into_par_iter()
        .filter_map(|i| {
            let row1_parts: Vec<&str> = lines1[i].split(',').collect();
            let row2_parts: Vec<&str> = lines2[i].split(',').collect();
            
            let mut row_differences = Vec::new();
            let mut has_difference = false;

            for j in 0..std::cmp::min(row1_parts.len(), row2_parts.len()) {
                if row1_parts[j] != row2_parts[j] {
                    has_difference = true;
                    if let Some(header) = headers.get(j) {
                        row_differences.push((j, header.to_string(), row1_parts[j].to_string(), row2_parts[j].to_string()));
                    }
                }
            }

            if has_difference {
                Some((i, row_differences))
            } else {
                None
            }
        })
        .collect();

    // 汇总结果
    for (row_num, row_diffs) in differences {
        result.different_rows += 1;
        
        for (col_idx, col_name, val1, val2) in row_diffs {
            *result.column_differences.get_mut(&col_name).unwrap() += 1;
            
            // 保存前100个差异作为样本
            if result.sample_differences.len() < 100 {
                result.sample_differences.push(DifferenceDetail {
                    row_number: row_num,
                    column: col_name,
                    value1: val1,
                    value2: val2,
                });
            }
        }
    }

    result.matching_rows = (min_rows - 1) - result.different_rows;

    println!("比较完成，总耗时: {:.2}秒", start.elapsed().as_secs_f64());
    
    Ok(result)
}

fn print_comparison_report(result: &ComparisonResult) {
    println!("\n========== 比较报告 ==========");
    println!("文件1总行数: {}", result.total_rows_file1);
    println!("文件2总行数: {}", result.total_rows_file2);
    
    if result.total_rows_file1 != result.total_rows_file2 {
        println!("⚠️  警告: 文件行数不一致!");
    }
    
    println!("\n匹配的行数: {}", result.matching_rows);
    println!("不同的行数: {}", result.different_rows);
    
    let total_compared = result.matching_rows + result.different_rows;
    if total_compared > 0 {
        let match_rate = result.matching_rows as f64 / total_compared as f64 * 100.0;
        println!("匹配率: {:.2}%", match_rate);
    }
    
    println!("\n各列差异统计:");
    let mut sorted_cols: Vec<_> = result.column_differences.iter().collect();
    sorted_cols.sort_by_key(|&(_, count)| std::cmp::Reverse(count));
    
    for (col, count) in sorted_cols {
        if *count > 0 {
            println!("  {}: {} 个差异", col, count);
        }
    }
    
    if !result.sample_differences.is_empty() {
        println!("\n差异样本 (最多显示20个):");
        for (i, diff) in result.sample_differences.iter().take(20).enumerate() {
            println!("\n  {}. 行号: {}", i + 1, diff.row_number);
            println!("     列名: {}", diff.column);
            println!("     文件1: '{}'", diff.value1);
            println!("     文件2: '{}'", diff.value2);
        }
    }
    
    // 保存详细报告
    if let Err(e) = save_detailed_report(result) {
        eprintln!("保存详细报告失败: {}", e);
    }
}

fn save_detailed_report(result: &ComparisonResult) -> Result<(), Box<dyn Error>> {
    use std::io::Write;
    
    let mut file = File::create("csv_comparison_report.txt")?;
    
    writeln!(file, "CSV文件比较详细报告")?;
    writeln!(file, "====================")?;
    writeln!(file, "生成时间: {}", chrono::Local::now().format("%Y-%m-%d %H:%M:%S"))?;
    writeln!(file)?;
    writeln!(file, "文件1总行数: {}", result.total_rows_file1)?;
    writeln!(file, "文件2总行数: {}", result.total_rows_file2)?;
    writeln!(file, "匹配的行数: {}", result.matching_rows)?;
    writeln!(file, "不同的行数: {}", result.different_rows)?;
    writeln!(file)?;
    
    writeln!(file, "各列差异详情:")?;
    for (col, count) in &result.column_differences {
        if *count > 0 {
            writeln!(file, "  {}: {} 个差异", col, count)?;
        }
    }
    
    writeln!(file)?;
    writeln!(file, "所有差异详情 (最多1000个):")?;
    for (i, diff) in result.sample_differences.iter().take(1000).enumerate() {
        writeln!(file, "\n{}. 行号: {}, 列: {}", i + 1, diff.row_number, diff.column)?;
        writeln!(file, "   文件1: '{}'", diff.value1)?;
        writeln!(file, "   文件2: '{}'", diff.value2)?;
    }
    
    println!("\n详细报告已保存到: csv_comparison_report.txt");
    
    Ok(())
}

fn main() -> Result<(), Box<dyn Error>> {
    let args: Vec<String> = std::env::args().collect();
    
    let (file1, file2) = if args.len() >= 3 {
        (args[1].clone(), args[2].clone())
    } else {
        // 默认文件路径
        (
            "/Users/augustsirius/Desktop/DIABERT_test_code_lib/read_bruker_data/python_output.csv".to_string(),
            "/Users/augustsirius/Desktop/DIABERT_test_code_lib/read_bruker_data/rust_output.csv".to_string()
        )
    };
    
    match compare_csv_files(&file1, &file2) {
        Ok(result) => {
            print_comparison_report(&result);
            
            if result.different_rows == 0 && result.total_rows_file1 == result.total_rows_file2 {
                println!("\n✅ 两个文件完全一致!");
            } else {
                println!("\n❌ 两个文件存在差异!");
            }
        }
        Err(e) => {
            eprintln!("错误: {}", e);
            std::process::exit(1);
        }
    }
    
    Ok(())
}