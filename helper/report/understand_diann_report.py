import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# 设置显示选项
pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 50)

# 1. 读取DIANN report文件
print("="*80)
print("1. 读取DIANN Report文件")
print("="*80)

report_path = '/Users/augustsirius/Desktop/DIABERT_test_code_lib/report/report.parquet'
df = pd.read_parquet(report_path)

print(f"文件大小: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
print(f"数据维度: {df.shape[0]} 行 × {df.shape[1]} 列")
print()

# 2. 查看所有列名
print("="*80)
print("2. DIANN Report的所有列名")
print("="*80)
for i, col in enumerate(df.columns, 1):
    print(f"{i:2d}. {col}")
print()

# 3. 数据类型分析
print("="*80)
print("3. 各列的数据类型")
print("="*80)
dtype_summary = df.dtypes.value_counts()
print("数据类型汇总:")
print(dtype_summary)
print("\n详细数据类型:")
for col, dtype in df.dtypes.items():
    print(f"{col:<40} {dtype}")
print()

# 4. 关键列的详细解释
print("="*80)
print("4. DIANN Report关键列解释")
print("="*80)

key_columns = {
    'File.Name': '原始质谱数据文件名',
    'Run': '实验运行批次标识',
    'Protein.Group': '蛋白质组（可能包含多个蛋白质）',
    'Protein.Ids': '蛋白质ID列表',
    'Protein.Names': '蛋白质名称',
    'Genes': '基因名称',
    'PG.MaxLFQ': '蛋白质组最大LFQ强度',
    'PG.Quantity': '蛋白质组定量值',
    'PG.Normalised': '标准化后的蛋白质组定量值',
    'Genes.Quantity': '基因水平定量值',
    'Genes.Normalised': '标准化后的基因定量值',
    'Genes.MaxLFQ': '基因最大LFQ强度',
    'Genes.MaxLFQ.Unique': '基因特异性最大LFQ强度',
    'Modified.Sequence': '修饰后的肽段序列',
    'Stripped.Sequence': '未修饰的肽段序列',
    'Precursor.Id': '前体离子ID（肽段+电荷）',
    'Precursor.Charge': '前体离子电荷状态',
    'Q.Value': 'FDR校正后的q值',
    'PEP': '后验错误概率',
    'Global.Q.Value': '全局q值',
    'Protein.Q.Value': '蛋白质水平q值',
    'PG.Q.Value': '蛋白质组q值',
    'GG.Q.Value': '基因组q值',
    'Proteotypic': '是否为蛋白特异性肽段',
    'Precursor.Quantity': '前体离子定量值',
    'Precursor.Normalised': '标准化的前体离子定量值',
    'Label.Ratio': '标记定量比率',
    'RT': '保留时间(秒)',
    'RT.Start': '洗脱开始时间',
    'RT.Stop': '洗脱结束时间',
    'iRT': '索引保留时间',
    'Predicted.RT': 'DIANN预测的保留时间',
    'Predicted.iRT': 'DIANN预测的索引保留时间',
    'First.Protein.Description': '第一个蛋白质的描述',
    'Lib.Q.Value': '谱库匹配q值',
    'Ms1.Profile.Corr': 'MS1谱图相关性',
    'Ms1.Area': 'MS1峰面积',
    'Evidence': '鉴定证据得分',
    'CScore': '置信度得分',
    'Decoy.Evidence': '诱饵库证据得分',
    'Decoy.CScore': '诱饵库置信度得分',
    'Fragment.Quant.Raw': '片段离子原始定量值',
    'Fragment.Quant.Corrected': '校正后的片段离子定量值',
    'Fragment.Correlations': '片段离子相关性',
    'MS2.Scan': 'MS2扫描号',
    'IM': '离子淌度值(1/K0)',
    'iIM': '索引离子淌度',
    'Predicted.IM': 'DIANN预测的离子淌度',
    'Predicted.iIM': 'DIANN预测的索引离子淌度',
    'Mass.Evidence': '质量准确度证据',
    'CCS': '碰撞截面积'
}

print("关键列说明:")
for col, desc in key_columns.items():
    if col in df.columns:
        print(f"\n{col}:")
        print(f"  说明: {desc}")
        print(f"  数据类型: {df[col].dtype}")
        if df[col].dtype in ['float64', 'int64']:
            print(f"  范围: [{df[col].min():.3f}, {df[col].max():.3f}]")
            print(f"  均值: {df[col].mean():.3f}")
        else:
            print(f"  唯一值数量: {df[col].nunique()}")
            if df[col].nunique() < 10:
                print(f"  唯一值: {df[col].unique()[:5]}")

# 5. 数据质量统计
print("\n" + "="*80)
print("5. 数据质量统计")
print("="*80)

# 缺失值分析
missing_stats = df.isnull().sum()
missing_pct = (missing_stats / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    '缺失值数量': missing_stats[missing_stats > 0],
    '缺失比例(%)': missing_pct[missing_stats > 0]
}).sort_values('缺失比例(%)', ascending=False)

if not missing_df.empty:
    print("\n缺失值统计 (仅显示有缺失的列):")
    print(missing_df.head(20))

# 6. 鉴定结果统计
print("\n" + "="*80)
print("6. 鉴定结果统计")
print("="*80)

# Q值分布
print(f"\nQ值分布:")
print(f"  Q.Value < 0.01: {(df['Q.Value'] < 0.01).sum()} ({(df['Q.Value'] < 0.01).sum()/len(df)*100:.1f}%)")
print(f"  Q.Value < 0.05: {(df['Q.Value'] < 0.05).sum()} ({(df['Q.Value'] < 0.05).sum()/len(df)*100:.1f}%)")

# 电荷状态分布
print(f"\n电荷状态分布:")
charge_dist = df['Precursor.Charge'].value_counts().sort_index()
for charge, count in charge_dist.items():
    print(f"  +{charge}: {count} ({count/len(df)*100:.1f}%)")

# 蛋白质和肽段统计
print(f"\n鉴定统计:")
print(f"  总前体离子数: {len(df)}")
print(f"  唯一肽段数: {df['Stripped.Sequence'].nunique()}")
print(f"  唯一修饰肽段数: {df['Modified.Sequence'].nunique()}")
print(f"  蛋白质组数: {df['Protein.Group'].nunique()}")
print(f"  基因数: {df['Genes'].nunique()}")

# 7. RT和IM分布分析
print("\n" + "="*80)
print("7. RT和IM分布分析")
print("="*80)

if 'RT' in df.columns:
    print(f"\n保留时间(RT)分布:")
    print(f"  范围: {df['RT'].min():.1f} - {df['RT'].max():.1f} 秒")
    print(f"  均值: {df['RT'].mean():.1f} 秒")
    print(f"  中位数: {df['RT'].median():.1f} 秒")

if 'IM' in df.columns:
    print(f"\n离子淌度(IM)分布:")
    print(f"  范围: {df['IM'].min():.4f} - {df['IM'].max():.4f}")
    print(f"  均值: {df['IM'].mean():.4f}")
    print(f"  中位数: {df['IM'].median():.4f}")

# 8. 定量信息分析
print("\n" + "="*80)
print("8. 定量信息分析")
print("="*80)

quant_cols = ['Precursor.Quantity', 'Precursor.Normalised', 'PG.Quantity', 'PG.Normalised']
for col in quant_cols:
    if col in df.columns and df[col].notna().any():
        print(f"\n{col}:")
        print(f"  非零值数量: {(df[col] > 0).sum()}")
        print(f"  范围: {df[col][df[col] > 0].min():.2e} - {df[col].max():.2e}")
        print(f"  CV: {(df[col].std() / df[col].mean() * 100):.1f}%")

# 9. 示例数据展示
print("\n" + "="*80)
print("9. 示例数据（前5行主要列）")
print("="*80)

display_cols = ['Precursor.Id', 'Modified.Sequence', 'Precursor.Charge', 
                'RT', 'IM', 'Q.Value', 'Precursor.Quantity', 'Protein.Group']
display_cols = [col for col in display_cols if col in df.columns]

print("\n前5行数据:")
print(df[display_cols].head())

# 10. 特定前体离子分析（如LLIYGASTR2）
print("\n" + "="*80)
print("10. 特定前体离子分析")
print("="*80)

# 查找示例前体离子
example_precursor = 'LLIYGASTR2'
example_data = df[df['Precursor.Id'] == example_precursor]

if not example_data.empty:
    print(f"\n前体离子 '{example_precursor}' 的详细信息:")
    for col in ['RT', 'IM', 'Q.Value', 'Precursor.Quantity', 'Modified.Sequence', 'Protein.Group']:
        if col in example_data.columns:
            print(f"  {col}: {example_data[col].iloc[0]}")
else:
    print(f"未找到前体离子 '{example_precursor}'")
    # 显示一些实际存在的前体离子
    print("\n实际存在的前体离子示例:")
    print(df['Precursor.Id'].head(10).tolist())

# 11. 创建可视化
print("\n" + "="*80)
print("11. 数据分布可视化")
print("="*80)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# RT分布
if 'RT' in df.columns:
    axes[0, 0].hist(df['RT'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_xlabel('Retention Time (s)')
    axes[0, 0].set_ylabel('Count')
    axes[0, 0].set_title('RT Distribution')

# IM分布
if 'IM' in df.columns:
    axes[0, 1].hist(df['IM'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_xlabel('Ion Mobility (1/K0)')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('IM Distribution')

# Q值分布
if 'Q.Value' in df.columns:
    axes[1, 0].hist(df['Q.Value'], bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].set_xlabel('Q Value')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].set_title('Q Value Distribution')
    axes[1, 0].set_yscale('log')

# 电荷状态分布
if 'Precursor.Charge' in df.columns:
    charge_counts = df['Precursor.Charge'].value_counts().sort_index()
    axes[1, 1].bar(charge_counts.index, charge_counts.values, edgecolor='black')
    axes[1, 1].set_xlabel('Precursor Charge')
    axes[1, 1].set_ylabel('Count')
    axes[1, 1].set_title('Charge State Distribution')

plt.tight_layout()
plt.show()

# 12. 数据导出建议
print("\n" + "="*80)
print("12. 数据使用建议")
print("="*80)

print("\n对于您的peak group提取任务，最重要的列是:")
print("1. Precursor.Id - 前体离子标识符")
print("2. RT - 保留时间（用于定位色谱峰）")
print("3. IM - 离子淌度（用于TIMS维度定位）")
print("4. Q.Value - 质量控制（建议只使用Q.Value < 0.01的结果）")
print("5. Precursor.Quantity - 定量信息（可用于验证提取结果）")

print("\n数据筛选建议:")
print("filtered_df = df[df['Q.Value'] < 0.01]  # 高置信度结果")
print("filtered_df = filtered_df[filtered_df['Precursor.Quantity'] > 0]  # 有定量值的结果")