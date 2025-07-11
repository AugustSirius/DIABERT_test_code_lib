import pandas as pd
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ResultComparator:
    def __init__(self, rust_dir, python_dir):
        self.rust_dir = Path(rust_dir)
        self.python_dir = Path(python_dir)
        self.report = []
        
    def add_to_report(self, section, content):
        """添加内容到报告"""
        self.report.append(f"\n{'='*80}")
        self.report.append(f"{section}")
        self.report.append('='*80)
        self.report.append(content)
        
    def compare_csv_files(self, rust_file, python_file, file_type):
        """比较两个CSV文件"""
        print(f"\n比较 {file_type}...")
        
        # 检查文件是否存在
        if not os.path.exists(rust_file):
            self.add_to_report(file_type, f"Rust文件不存在: {rust_file}")
            return
        if not os.path.exists(python_file):
            self.add_to_report(file_type, f"Python文件不存在: {python_file}")
            return
            
        # 读取文件
        try:
            rust_df = pd.read_csv(rust_file)
            python_df = pd.read_csv(python_file)
        except Exception as e:
            self.add_to_report(file_type, f"读取文件时出错: {str(e)}")
            return
        
        report_content = []
        
        # 基本信息
        report_content.append(f"Rust文件: {rust_file}")
        report_content.append(f"Python文件: {python_file}")
        report_content.append("")
        
        # 行数和列数对比
        report_content.append(f"数据维度:")
        report_content.append(f"  Rust:   {rust_df.shape[0]:,} 行 × {rust_df.shape[1]} 列")
        report_content.append(f"  Python: {python_df.shape[0]:,} 行 × {python_df.shape[1]} 列")
        report_content.append(f"  差异:   {rust_df.shape[0] - python_df.shape[0]:,} 行")
        report_content.append("")
        
        # 列名对比
        rust_cols = set(rust_df.columns)
        python_cols = set(python_df.columns)
        
        if rust_cols != python_cols:
            report_content.append("列名差异:")
            only_rust = rust_cols - python_cols
            only_python = python_cols - rust_cols
            if only_rust:
                report_content.append(f"  只在Rust中: {only_rust}")
            if only_python:
                report_content.append(f"  只在Python中: {only_python}")
        else:
            report_content.append("列名: 完全一致")
        report_content.append("")
        
        # 对共同列进行数值统计
        common_cols = rust_cols & python_cols
        numeric_cols = []
        
        for col in common_cols:
            if pd.api.types.is_numeric_dtype(rust_df[col]) and pd.api.types.is_numeric_dtype(python_df[col]):
                numeric_cols.append(col)
        
        if numeric_cols:
            report_content.append("数值列统计对比:")
            for col in numeric_cols:
                report_content.append(f"\n  列 '{col}':")
                
                # Rust统计
                rust_stats = rust_df[col].describe()
                python_stats = python_df[col].describe()
                
                report_content.append(f"    Rust   - 均值: {rust_stats['mean']:.4f}, 标准差: {rust_stats['std']:.4f}, "
                                    f"最小值: {rust_stats['min']:.4f}, 最大值: {rust_stats['max']:.4f}")
                report_content.append(f"    Python - 均值: {python_stats['mean']:.4f}, 标准差: {python_stats['std']:.4f}, "
                                    f"最小值: {python_stats['min']:.4f}, 最大值: {python_stats['max']:.4f}")
                
                # 计算差异
                mean_diff = abs(rust_stats['mean'] - python_stats['mean'])
                if rust_stats['mean'] != 0:
                    mean_diff_pct = mean_diff / abs(rust_stats['mean']) * 100
                    report_content.append(f"    均值差异: {mean_diff:.4f} ({mean_diff_pct:.2f}%)")
        
        # 如果是特定类型的文件，进行特殊分析
        if 'mz_values' in common_cols:
            # 分析m/z值的分布
            report_content.append("\nm/z值分布分析:")
            
            # 找出共同的m/z值
            rust_mz = set(rust_df['mz_values'].astype(int))
            python_mz = set(python_df['mz_values'].astype(int))
            common_mz = rust_mz & python_mz
            
            report_content.append(f"  Rust独有的m/z值数量: {len(rust_mz - python_mz):,}")
            report_content.append(f"  Python独有的m/z值数量: {len(python_mz - rust_mz):,}")
            report_content.append(f"  共同的m/z值数量: {len(common_mz):,}")
            
            if len(common_mz) > 0:
                overlap_ratio = len(common_mz) / len(rust_mz | python_mz) * 100
                report_content.append(f"  重叠率: {overlap_ratio:.2f}%")
        
        self.add_to_report(file_type, '\n'.join(report_content))
        
    def compare_matrix_files(self, rust_file, python_file, matrix_type):
        """比较矩阵文件"""
        print(f"\n比较 {matrix_type} 矩阵...")
        
        if not os.path.exists(rust_file) or not os.path.exists(python_file):
            self.add_to_report(matrix_type, "矩阵文件不存在")
            return
            
        try:
            # 读取矩阵
            rust_matrix = np.loadtxt(rust_file, delimiter=',')
            python_matrix = np.loadtxt(python_file, delimiter=',')
            
            report_content = []
            
            # 基本信息
            report_content.append(f"矩阵维度:")
            report_content.append(f"  Rust:   {rust_matrix.shape}")
            report_content.append(f"  Python: {python_matrix.shape}")
            report_content.append("")
            
            # 非零元素统计
            rust_nonzero = np.count_nonzero(rust_matrix)
            python_nonzero = np.count_nonzero(python_matrix)
            
            report_content.append(f"非零元素:")
            report_content.append(f"  Rust:   {rust_nonzero:,} / {rust_matrix.size:,} ({rust_nonzero/rust_matrix.size*100:.2f}%)")
            report_content.append(f"  Python: {python_nonzero:,} / {python_matrix.size:,} ({python_nonzero/python_matrix.size*100:.2f}%)")
            report_content.append("")
            
            # 如果维度相同，计算差异
            if rust_matrix.shape == python_matrix.shape:
                diff_matrix = rust_matrix - python_matrix
                diff_count = np.count_nonzero(diff_matrix)
                
                report_content.append(f"矩阵差异:")
                report_content.append(f"  不同元素数量: {diff_count:,} / {rust_matrix.size:,} ({diff_count/rust_matrix.size*100:.2f}%)")
                
                if diff_count > 0:
                    report_content.append(f"  最大差异: {np.max(np.abs(diff_matrix)):.4f}")
                    report_content.append(f"  平均差异: {np.mean(np.abs(diff_matrix)):.4f}")
                    
                # 计算相似度（使用Jaccard指数）
                rust_nonzero_mask = rust_matrix > 0
                python_nonzero_mask = python_matrix > 0
                intersection = np.sum(rust_nonzero_mask & python_nonzero_mask)
                union = np.sum(rust_nonzero_mask | python_nonzero_mask)
                
                if union > 0:
                    jaccard = intersection / union
                    report_content.append(f"\n非零元素位置相似度 (Jaccard): {jaccard:.4f}")
            else:
                report_content.append("矩阵维度不同，无法直接比较")
                
            self.add_to_report(matrix_type, '\n'.join(report_content))
            
        except Exception as e:
            self.add_to_report(matrix_type, f"读取矩阵文件时出错: {str(e)}")
            
    def compare_summary_files(self):
        """比较summary文件"""
        print("\n比较 Summary 文件...")
        
        rust_summary = self.rust_dir / "mask_matrices_summary.txt"
        python_summary = self.python_dir / "python_mask_matrices_summary.txt"
        
        report_content = []
        
        # 读取Rust summary
        if rust_summary.exists():
            with open(rust_summary, 'r') as f:
                rust_content = f.read()
            report_content.append("Rust Summary内容:")
            report_content.append("-" * 40)
            report_content.append(rust_content)
            report_content.append("")
        
        # 读取Python summary
        if python_summary.exists():
            with open(python_summary, 'r') as f:
                python_content = f.read()
            report_content.append("Python Summary内容:")
            report_content.append("-" * 40)
            report_content.append(python_content)
            
        self.add_to_report("Summary文件对比", '\n'.join(report_content))
        
    def generate_comparison_plots(self):
        """生成对比图表"""
        print("\n生成对比图表...")
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Rust vs Python 结果对比', fontsize=16)
        
        # 1. 数据量对比
        ax = axes[0, 0]
        
        # 收集数据量信息
        data_counts = {
            'Precursor (before IM)': [],
            'Precursor (after IM)': [],
            'Fragment (before IM)': [],
            'Fragment (after IM)': []
        }
        
        # Rust数据
        for name, file in [
            ('Precursor (before IM)', self.rust_dir / 'precursor_result_before_IM_filter.csv'),
            ('Precursor (after IM)', self.rust_dir / 'precursor_result_after_IM_filter.csv'),
            ('Fragment (before IM)', self.rust_dir / 'frag_result.csv'),
            ('Fragment (after IM)', self.rust_dir / 'frag_result.csv')  # Rust只有过滤后的
        ]:
            if file.exists():
                df = pd.read_csv(file)
                data_counts[name].append(len(df))
            else:
                data_counts[name].append(0)
                
        # Python数据
        for name, file in [
            ('Precursor (before IM)', self.python_dir / 'python_precursor_result_before_IM_filter.csv'),
            ('Precursor (after IM)', self.python_dir / 'python_precursor_result_after_IM_filter.csv'),
            ('Fragment (before IM)', self.python_dir / 'python_frag_result_before_IM_filter.csv'),
            ('Fragment (after IM)', self.python_dir / 'python_frag_result_after_IM_filter.csv')
        ]:
            if file.exists():
                df = pd.read_csv(file)
                data_counts[name].append(len(df))
            else:
                data_counts[name].append(0)
        
        # 绘制条形图
        x = np.arange(len(data_counts))
        width = 0.35
        
        rust_counts = [data_counts[k][0] for k in data_counts]
        python_counts = [data_counts[k][1] for k in data_counts]
        
        ax.bar(x - width/2, rust_counts, width, label='Rust', alpha=0.8)
        ax.bar(x + width/2, python_counts, width, label='Python', alpha=0.8)
        
        ax.set_xlabel('数据类型')
        ax.set_ylabel('数据点数量')
        ax.set_title('数据量对比')
        ax.set_xticks(x)
        ax.set_xticklabels(data_counts.keys(), rotation=45, ha='right')
        ax.legend()
        ax.set_yscale('log')  # 使用对数刻度
        
        # 添加数值标签
        for i, (r, p) in enumerate(zip(rust_counts, python_counts)):
            if r > 0:
                ax.text(i - width/2, r, f'{r:,}', ha='center', va='bottom', fontsize=8)
            if p > 0:
                ax.text(i + width/2, p, f'{p:,}', ha='center', va='bottom', fontsize=8)
        
        # 2. 非零元素对比（留空其他子图）
        for i in range(1, 4):
            ax = axes.flatten()[i]
            ax.text(0.5, 0.5, '更多对比图表待添加', ha='center', va='center', transform=ax.transAxes)
            ax.set_xticks([])
            ax.set_yticks([])
        
        plt.tight_layout()
        plt.savefig('rust_vs_python_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        self.add_to_report("图表生成", "对比图表已保存到: rust_vs_python_comparison.png")
        
    def run_full_comparison(self):
        """运行完整对比"""
        print("开始全面对比分析...")
        
        # 1. 对比precursor结果
        self.compare_csv_files(
            self.rust_dir / 'precursor_result_before_IM_filter.csv',
            self.python_dir / 'python_precursor_result_before_IM_filter.csv',
            'Precursor结果 (IM过滤前)'
        )
        
        self.compare_csv_files(
            self.rust_dir / 'precursor_result_after_IM_filter.csv',
            self.python_dir / 'python_precursor_result_after_IM_filter.csv',
            'Precursor结果 (IM过滤后)'
        )
        
        # 2. 对比fragment结果
        self.compare_csv_files(
            self.rust_dir / 'frag_result.csv',
            self.python_dir / 'python_frag_result_after_IM_filter.csv',
            'Fragment结果 (IM过滤后)'
        )
        
        # 3. 对比矩阵
        self.compare_matrix_files(
            self.rust_dir / 'ms1_frag_moz_matrix_full.csv',
            self.python_dir / 'python_ms1_frag_moz_matrix.csv',
            'MS1 Fragment矩阵'
        )
        
        self.compare_matrix_files(
            self.rust_dir / 'ms2_frag_moz_matrix_full.csv',
            self.python_dir / 'python_ms2_frag_moz_matrix.csv',
            'MS2 Fragment矩阵'
        )
        
        # 4. 对比summary文件
        self.compare_summary_files()
        
        # 5. 生成图表
        self.generate_comparison_plots()
        
        # 6. 保存报告
        self.save_report()
        
    def save_report(self):
        """保存对比报告"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = f'comparison_report_{timestamp}.txt'
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("Rust vs Python 结果对比报告\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("="*80 + "\n")
            
            for line in self.report:
                f.write(line + '\n')
                
        print(f"\n对比报告已保存到: {report_file}")

# 主程序
if __name__ == "__main__":
    # 设置路径
    rust_dir = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/20250707/timstof"
    python_dir = "/Users/augustsirius/Desktop/DIABERT_test_code_lib"
    
    # 创建比较器并运行
    comparator = ResultComparator(rust_dir, python_dir)
    comparator.run_full_comparison()
    
    print("\n对比分析完成！")