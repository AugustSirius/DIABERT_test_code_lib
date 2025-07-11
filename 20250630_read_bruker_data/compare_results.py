import pandas as pd
import numpy as np
from collections import defaultdict

# 运行原始Python代码
print("运行原始Python代码...")
library = pd.read_csv("/Users/augustsirius/Desktop/DIABERT_test_code_lib/helper/lib/TPHPlib_frag1025_swissprot_final_all_from_Yueliang.tsv", sep="\t")

def get_lib_col_dict():
    lib_col_dict = defaultdict(str)

    for key in ['transition_group_id', 'PrecursorID']:
        lib_col_dict[key] = 'transition_group_id'

    for key in ['PeptideSequence', 'Sequence', 'StrippedPeptide']:
        lib_col_dict[key] = 'PeptideSequence'

    for key in ['FullUniModPeptideName', 'ModifiedPeptide', 'LabeledSequence', 'modification_sequence',
                'ModifiedPeptideSequence']:
        lib_col_dict[key] = 'FullUniModPeptideName'

    for key in ['PrecursorCharge', 'Charge', 'prec_z']:
        lib_col_dict[key] = 'PrecursorCharge'

    for key in ['PrecursorMz', 'Q1']:
        lib_col_dict[key] = 'PrecursorMz'

    for key in ['Tr_recalibrated', 'iRT', 'RetentionTime', 'NormalizedRetentionTime', 'RT_detected']:
        lib_col_dict[key] = 'Tr_recalibrated'

    for key in ['ProductMz', 'FragmentMz', 'Q3']:
        lib_col_dict[key] = 'ProductMz'

    for key in ['FragmentType', 'FragmentIonType', 'ProductType', 'ProductIonType', 'frg_type']:
        lib_col_dict[key] = 'FragmentType'

    for key in ['FragmentCharge', 'FragmentIonCharge', 'ProductCharge', 'ProductIonCharge', 'frg_z']:
        lib_col_dict[key] = 'FragmentCharge'

    for key in ['FragmentNumber', 'frg_nr', 'FragmentSeriesNumber']:
        lib_col_dict[key] = 'FragmentNumber'

    for key in ['LibraryIntensity', 'RelativeIntensity', 'RelativeFragmentIntensity', 'RelativeFragmentIonIntensity',
                'relative_intensity']:
        lib_col_dict[key] = 'LibraryIntensity'

    # exclude
    for key in ['FragmentLossType', 'FragmentIonLossType', 'ProductLossType', 'ProductIonLossType']:
        lib_col_dict[key] = 'FragmentLossType'

    for key in ['ProteinID', 'ProteinId', 'UniprotID', 'uniprot_id', 'UniProtIds']:
        lib_col_dict[key] = 'ProteinID'

    for key in ['ProteinName', 'Protein Name', 'Protein_name', 'protein_name']:
        lib_col_dict[key] = 'ProteinName'

    for key in ['Gene', 'Genes', 'GeneName']:
        lib_col_dict[key] = 'Gene'

    for key in ['Decoy', 'decoy']:
        lib_col_dict[key] = 'decoy'

    for key in ['ExcludeFromAssay', 'ExcludeFromQuantification']:
        lib_col_dict[key] = 'ExcludeFromAssay'
    return lib_col_dict

# col mapping
lib_col_dict = get_lib_col_dict()
for col in set(library.columns) & set(lib_col_dict.keys()):
    library.loc[:, lib_col_dict[col]] = library.loc[:, col]

library['transition_group_id'] = library['FullUniModPeptideName'] + library['PrecursorCharge'].astype(str)
replacement_dict = {'b': 1, 'y': 2, 'p': 3}
library['FragmentType'] = library['FragmentType'].replace(replacement_dict)
library['decoy'] = 0

# 保存Python结果
print(f"Python处理完成，共 {len(library)} 条记录")
python_cols = ['transition_group_id', 'PeptideSequence', 'FullUniModPeptideName', 
               'PrecursorCharge', 'PrecursorMz', 'ProductMz', 'FragmentType', 
               'LibraryIntensity', 'ProteinID', 'Gene', 'decoy']

# 确保所有列都存在
for col in python_cols:
    if col not in library.columns:
        library[col] = ''

library[python_cols].to_csv('python_output.csv', index=False)

# 比较结果
print("\n开始比较Python和Rust的结果...")

# 读取Rust输出
rust_df = pd.read_csv('rust_output.csv')
python_df = pd.read_csv('python_output.csv')

print(f"\nPython记录数: {len(python_df)}")
print(f"Rust记录数: {len(rust_df)}")

# 基本比较
if len(python_df) != len(rust_df):
    print("⚠️ 警告：记录数不一致！")
else:
    print("✓ 记录数一致")

# 比较列
print("\n列比较:")
python_cols_set = set(python_df.columns)
rust_cols_set = set(rust_df.columns)

if python_cols_set == rust_cols_set:
    print("✓ 列名完全一致")
else:
    print("⚠️ 列名不一致")
    print(f"Python独有列: {python_cols_set - rust_cols_set}")
    print(f"Rust独有列: {rust_cols_set - python_cols_set}")

# 详细比较前100条记录
print("\n详细比较前100条记录...")
comparison_cols = list(python_cols_set & rust_cols_set)

for col in comparison_cols:
    # 将两个DataFrame的列都转换为字符串进行比较
    python_col = python_df[col].astype(str).head(100)
    rust_col = rust_df[col].astype(str).head(100)
    
    if python_col.equals(rust_col):
        print(f"✓ {col}: 完全一致")
    else:
        diff_count = (python_col != rust_col).sum()
        print(f"⚠️ {col}: 有 {diff_count} 处不同")
        
        # 显示前5个不同的例子
        diff_indices = python_col[python_col != rust_col].index[:5]
        for idx in diff_indices:
            print(f"   行 {idx}: Python='{python_col[idx]}', Rust='{rust_col[idx]}'")

# 数值列的统计比较
print("\n数值列统计比较:")
numeric_cols = ['PrecursorMz', 'ProductMz', 'LibraryIntensity']
for col in numeric_cols:
    if col in comparison_cols:
        try:
            python_numeric = pd.to_numeric(python_df[col], errors='coerce')
            rust_numeric = pd.to_numeric(rust_df[col], errors='coerce')
            
            print(f"\n{col}:")
            print(f"  Python - 平均值: {python_numeric.mean():.6f}, 标准差: {python_numeric.std():.6f}")
            print(f"  Rust   - 平均值: {rust_numeric.mean():.6f}, 标准差: {rust_numeric.std():.6f}")
            
            # 计算差异
            valid_mask = ~(python_numeric.isna() | rust_numeric.isna())
            if valid_mask.any():
                abs_diff = np.abs(python_numeric[valid_mask] - rust_numeric[valid_mask])
                print(f"  最大绝对差异: {abs_diff.max():.9f}")
                print(f"  平均绝对差异: {abs_diff.mean():.9f}")
        except Exception as e:
            print(f"  无法比较 {col}: {e}")

# 抽样比较
print("\n随机抽样比较 (1000条记录):")
sample_size = min(1000, len(python_df), len(rust_df))
sample_indices = np.random.choice(min(len(python_df), len(rust_df)), sample_size, replace=False)

differences = {}
for col in comparison_cols:
    python_sample = python_df.iloc[sample_indices][col].astype(str)
    rust_sample = rust_df.iloc[sample_indices][col].astype(str)
    
    diff_count = (python_sample != rust_sample).sum()
    if diff_count > 0:
        differences[col] = diff_count

if not differences:
    print("✓ 随机抽样的1000条记录完全一致！")
else:
    print("发现以下列有差异:")
    for col, count in differences.items():
        print(f"  {col}: {count}个不同 ({count/sample_size*100:.1f}%)")

# 特定值检查
print("\n特定值检查:")
print(f"FragmentType值分布 (前1000条):")
print("Python:")
print(python_df['FragmentType'].head(1000).value_counts().sort_index())
print("\nRust:")
print(rust_df['FragmentType'].head(1000).value_counts().sort_index())

print("\ndecoy值分布:")
print(f"Python decoy值: {python_df['decoy'].unique()}")
print(f"Rust decoy值: {rust_df['decoy'].unique()}")

# 保存完整比较报告
print("\n生成详细比较报告...")
with open('comparison_report.txt', 'w') as f:
    f.write("Python vs Rust 数据处理结果比较报告\n")
    f.write("="*50 + "\n\n")
    f.write(f"Python记录数: {len(python_df)}\n")
    f.write(f"Rust记录数: {len(rust_df)}\n")
    f.write(f"记录数是否一致: {'是' if len(python_df) == len(rust_df) else '否'}\n\n")
    
    f.write("列差异:\n")
    for col in comparison_cols:
        python_col = python_df[col].astype(str)
        rust_col = rust_df[col].astype(str)
        diff_count = (python_col != rust_col).sum()
        f.write(f"  {col}: {diff_count} 个不同 ({diff_count/len(python_df)*100:.2f}%)\n")

print("\n比较完成！详细报告已保存到 comparison_report.txt")