import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

def load_matrix(filepath):
    """加载CSV格式的矩阵"""
    try:
        # 使用numpy加载，跳过可能的标题行
        matrix = np.loadtxt(filepath, delimiter=',')
        return matrix
    except Exception as e:
        print(f"加载文件 {filepath} 时出错: {e}")
        return None

def compare_matrices(matrix1, matrix2, name1="Matrix1", name2="Matrix2"):
    """比较两个矩阵的详细信息"""
    print(f"\n{'='*60}")
    print(f"比较 {name1} 和 {name2}")
    print(f"{'='*60}")
    
    # 1. 形状比较
    shape1, shape2 = matrix1.shape, matrix2.shape
    print(f"\n1. 矩阵形状:")
    print(f"   {name1}: {shape1}")
    print(f"   {name2}: {shape2}")
    print(f"   形状一致: {shape1 == shape2}")
    
    if shape1 != shape2:
        print("\n⚠️  警告：矩阵形状不一致，无法进行元素级比较")
        return False
    
    # 2. 非零元素统计
    nonzero1 = np.count_nonzero(matrix1)
    nonzero2 = np.count_nonzero(matrix2)
    total_elements = matrix1.size
    
    print(f"\n2. 非零元素统计:")
    print(f"   {name1}: {nonzero1:,} / {total_elements:,} ({nonzero1/total_elements*100:.2f}%)")
    print(f"   {name2}: {nonzero2:,} / {total_elements:,} ({nonzero2/total_elements*100:.2f}%)")
    print(f"   非零元素数量差异: {abs(nonzero1 - nonzero2):,}")
    
    # 3. 元素值比较
    print(f"\n3. 元素值比较:")
    
    # 完全相等检查
    are_equal = np.array_equal(matrix1, matrix2)
    print(f"   矩阵完全相同: {are_equal}")
    
    if not are_equal:
        # 查找差异
        diff_mask = matrix1 != matrix2
        diff_count = np.sum(diff_mask)
        diff_percentage = (diff_count / total_elements) * 100
        
        print(f"   不同元素数量: {diff_count:,} / {total_elements:,} ({diff_percentage:.4f}%)")
        
        # 显示前10个差异的位置和值
        if diff_count > 0:
            diff_positions = np.argwhere(diff_mask)
            print(f"\n   前10个差异位置:")
            for i, (row, col) in enumerate(diff_positions[:10]):
                val1 = matrix1[row, col]
                val2 = matrix2[row, col]
                print(f"   [{row:4d}, {col:4d}]: {name1}={val1:.1f}, {name2}={val2:.1f}")
            
            if diff_count > 10:
                print(f"   ... 还有 {diff_count - 10:,} 个差异")
    
    # 4. 数值范围
    print(f"\n4. 数值范围:")
    print(f"   {name1}: min={matrix1.min():.1f}, max={matrix1.max():.1f}")
    print(f"   {name2}: min={matrix2.min():.1f}, max={matrix2.max():.1f}")
    
    # 5. 非零元素位置比较
    print(f"\n5. 非零元素位置比较:")
    nonzero_mask1 = matrix1 > 0
    nonzero_mask2 = matrix2 > 0
    
    # 只在matrix1中为非零
    only_in_1 = nonzero_mask1 & ~nonzero_mask2
    only_in_1_count = np.sum(only_in_1)
    
    # 只在matrix2中为非零
    only_in_2 = ~nonzero_mask1 & nonzero_mask2
    only_in_2_count = np.sum(only_in_2)
    
    # 两者都为非零
    both_nonzero = nonzero_mask1 & nonzero_mask2
    both_nonzero_count = np.sum(both_nonzero)
    
    print(f"   只在{name1}中非零: {only_in_1_count:,}")
    print(f"   只在{name2}中非零: {only_in_2_count:,}")
    print(f"   两者都非零: {both_nonzero_count:,}")
    
    return are_equal

def visualize_differences(matrix1, matrix2, name1, name2, output_prefix):
    """可视化两个矩阵的差异"""
    if matrix1.shape != matrix2.shape:
        print("矩阵形状不同，无法可视化差异")
        return
    
    # 创建差异矩阵
    diff_matrix = matrix1 - matrix2
    
    # 创建图形
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Matrix1的热图
    ax1 = axes[0, 0]
    im1 = ax1.imshow(matrix1, aspect='auto', cmap='Blues', interpolation='nearest')
    ax1.set_title(f'{name1} (非零元素: {np.count_nonzero(matrix1):,})')
    ax1.set_xlabel('列索引')
    ax1.set_ylabel('行索引')
    plt.colorbar(im1, ax=ax1)
    
    # 2. Matrix2的热图
    ax2 = axes[0, 1]
    im2 = ax2.imshow(matrix2, aspect='auto', cmap='Blues', interpolation='nearest')
    ax2.set_title(f'{name2} (非零元素: {np.count_nonzero(matrix2):,})')
    ax2.set_xlabel('列索引')
    ax2.set_ylabel('行索引')
    plt.colorbar(im2, ax=ax2)
    
    # 3. 差异矩阵的热图
    ax3 = axes[1, 0]
    im3 = ax3.imshow(diff_matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest', 
                     vmin=-1, vmax=1)
    ax3.set_title(f'差异矩阵 ({name1} - {name2})')
    ax3.set_xlabel('列索引')
    ax3.set_ylabel('行索引')
    plt.colorbar(im3, ax=ax3)
    
    # 4. 差异位置的散点图
    ax4 = axes[1, 1]
    diff_positions = np.argwhere(matrix1 != matrix2)
    if len(diff_positions) > 0:
        ax4.scatter(diff_positions[:, 1], diff_positions[:, 0], 
                   alpha=0.5, s=1, c='red')
        ax4.set_title(f'差异位置 (共 {len(diff_positions):,} 个)')
        ax4.set_xlabel('列索引')
        ax4.set_ylabel('行索引')
        ax4.set_xlim(0, matrix1.shape[1])
        ax4.set_ylim(matrix1.shape[0], 0)  # 反转y轴以匹配矩阵显示
    else:
        ax4.text(0.5, 0.5, '无差异', ha='center', va='center', 
                transform=ax4.transAxes, fontsize=16)
        ax4.set_title('差异位置')
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_comparison.png', dpi=150, bbox_inches='tight')
    print(f"\n差异可视化图已保存到: {output_prefix}_comparison.png")
    plt.close()

def main():
    """主函数"""
    # 文件路径
    base_path = Path("/Users/augustsirius/Desktop/DIABERT_test_code_lib")
    
    files = {
        "rust_ms1": base_path / "20250707/timstof/rust_ms1_frag_moz_matrix.csv",
        "rust_ms2": base_path / "20250707/timstof/rust_ms2_frag_moz_matrix.csv",
        "python_ms1": base_path / "python_ms1_frag_moz_matrix.csv",
        "python_ms2": base_path / "python_ms2_frag_moz_matrix.csv"
    }
    
    # 检查文件是否存在
    print("检查文件是否存在:")
    for name, filepath in files.items():
        exists = filepath.exists()
        print(f"  {name}: {filepath}")
        print(f"    存在: {exists}")
        if exists:
            print(f"    大小: {filepath.stat().st_size:,} bytes")
    
    # 加载矩阵
    print("\n加载矩阵...")
    matrices = {}
    for name, filepath in files.items():
        if filepath.exists():
            matrices[name] = load_matrix(filepath)
            if matrices[name] is not None:
                print(f"  {name}: 成功加载，形状 {matrices[name].shape}")
        else:
            print(f"  {name}: 文件不存在")
    
    # 比较MS1矩阵
    if "rust_ms1" in matrices and "python_ms1" in matrices:
        ms1_equal = compare_matrices(
            matrices["python_ms1"], 
            matrices["rust_ms1"],
            "Python MS1", 
            "Rust MS1"
        )
        
        # 可视化MS1差异
        visualize_differences(
            matrices["python_ms1"],
            matrices["rust_ms1"],
            "Python MS1",
            "Rust MS1",
            "ms1_matrix"
        )
    
    # 比较MS2矩阵
    if "rust_ms2" in matrices and "python_ms2" in matrices:
        ms2_equal = compare_matrices(
            matrices["python_ms2"], 
            matrices["rust_ms2"],
            "Python MS2", 
            "Rust MS2"
        )
        
        # 可视化MS2差异
        visualize_differences(
            matrices["python_ms2"],
            matrices["rust_ms2"],
            "Python MS2",
            "Rust MS2",
            "ms2_matrix"
        )
    
    # 总结
    print(f"\n{'='*60}")
    print("总结")
    print(f"{'='*60}")
    
    if "rust_ms1" in matrices and "python_ms1" in matrices:
        print(f"MS1矩阵完全一致: {ms1_equal}")
    
    if "rust_ms2" in matrices and "python_ms2" in matrices:
        print(f"MS2矩阵完全一致: {ms2_equal}")
    
    # 保存详细比较报告
    report_path = "matrix_comparison_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("Mask矩阵比较报告\n")
        f.write("="*60 + "\n\n")
        
        for matrix_type in ["ms1", "ms2"]:
            rust_key = f"rust_{matrix_type}"
            python_key = f"python_{matrix_type}"
            
            if rust_key in matrices and python_key in matrices:
                f.write(f"\n{matrix_type.upper()}矩阵比较:\n")
                f.write("-"*40 + "\n")
                
                rust_matrix = matrices[rust_key]
                python_matrix = matrices[python_key]
                
                f.write(f"Python形状: {python_matrix.shape}\n")
                f.write(f"Rust形状: {rust_matrix.shape}\n")
                f.write(f"Python非零元素: {np.count_nonzero(python_matrix):,}\n")
                f.write(f"Rust非零元素: {np.count_nonzero(rust_matrix):,}\n")
                
                if python_matrix.shape == rust_matrix.shape:
                    diff_count = np.sum(python_matrix != rust_matrix)
                    f.write(f"不同元素数量: {diff_count:,}\n")
                    f.write(f"一致性: {diff_count == 0}\n")
    
    print(f"\n详细报告已保存到: {report_path}")

if __name__ == "__main__":
    main()