import pandas as pd
import numpy as np

def process_and_compare_mz_values(file1_path, file2_path):
    """
    处理两个CSV文件的mz_values列，进行排序、去重并对比差异
    """
    try:
        # 读取两个CSV文件
        df1 = pd.read_csv(file1_path)
        df2 = pd.read_csv(file2_path)
        
        print(f"文件1: {file1_path}")
        print(f"文件1形状: {df1.shape}")
        print(f"文件2: {file2_path}")
        print(f"文件2形状: {df2.shape}")
        print("-" * 50)
        
        # 检查是否存在mz_values列
        if 'mz_values' not in df1.columns:
            print(f"警告: 文件1中未找到'mz_values'列，可用列: {df1.columns.tolist()}")
            return
        if 'mz_values' not in df2.columns:
            print(f"警告: 文件2中未找到'mz_values'列，可用列: {df2.columns.tolist()}")
            return
        
        # 提取mz_values列，去除NaN值，去重并排序
        mz1 = df1['mz_values'].dropna().drop_duplicates().sort_values().reset_index(drop=True)
        mz2 = df2['mz_values'].dropna().drop_duplicates().sort_values().reset_index(drop=True)
        
        print("处理结果:")
        print(f"文件1 - 原始数据量: {len(df1['mz_values'])}, 去重后: {len(mz1)}")
        print(f"文件2 - 原始数据量: {len(df2['mz_values'])}, 去重后: {len(mz2)}")
        print("-" * 50)
        
        # 转换为集合进行比较
        set1 = set(mz1)
        set2 = set(mz2)
        
        # 计算差异
        only_in_file1 = set1 - set2
        only_in_file2 = set2 - set1
        common = set1 & set2
        
        print("对比结果:")
        print(f"共同的mz_values数量: {len(common)}")
        print(f"仅在文件1中的mz_values数量: {len(only_in_file1)}")
        print(f"仅在文件2中的mz_values数量: {len(only_in_file2)}")
        print(f"总的不同mz_values数量: {len(set1.union(set2))}")
        print("-" * 50)
        
        # 显示一些具体的差异值（如果有的话）
        if only_in_file1:
            print(f"仅在文件1中的前10个mz_values: {sorted(list(only_in_file1))[:10]}")
        if only_in_file2:
            print(f"仅在文件2中的前10个mz_values: {sorted(list(only_in_file2))[:10]}")
        
        # 统计信息
        print("\n统计信息:")
        print(f"文件1 mz_values范围: {mz1.min():.4f} - {mz1.max():.4f}")
        print(f"文件2 mz_values范围: {mz2.min():.4f} - {mz2.max():.4f}")
        
        # 保存结果到新文件（可选）
        save_results = input("\n是否要保存处理结果到文件? (y/n): ").lower().strip() == 'y'
        if save_results:
            # 保存排序去重后的结果
            mz1.to_csv('file1_mz_values_processed.csv', index=False, header=['mz_values'])
            mz2.to_csv('file2_mz_values_processed.csv', index=False, header=['mz_values'])
            
            # 保存差异分析结果
            with open('mz_values_comparison_report.txt', 'w') as f:
                f.write(f"文件1: {file1_path}\n")
                f.write(f"文件2: {file2_path}\n")
                f.write(f"共同的mz_values数量: {len(common)}\n")
                f.write(f"仅在文件1中的数量: {len(only_in_file1)}\n")
                f.write(f"仅在文件2中的数量: {len(only_in_file2)}\n")
                f.write(f"\n仅在文件1中的mz_values:\n")
                for val in sorted(only_in_file1):
                    f.write(f"{val}\n")
                f.write(f"\n仅在文件2中的mz_values:\n")
                for val in sorted(only_in_file2):
                    f.write(f"{val}\n")
            
            print("结果已保存到当前目录")
        
        return {
            'file1_processed': mz1,
            'file2_processed': mz2,
            'common': common,
            'only_in_file1': only_in_file1,
            'only_in_file2': only_in_file2
        }
        
    except Exception as e:
        print(f"处理过程中出现错误: {e}")
        return None

# 使用您的文件路径
file1_path = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/20250707/timstof/precursor_result_after_IM_filter.csv"
file2_path = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/python_precursor_result_after_IM_filter.csv"

# 执行处理和对比
result = process_and_compare_mz_values(file1_path, file2_path)