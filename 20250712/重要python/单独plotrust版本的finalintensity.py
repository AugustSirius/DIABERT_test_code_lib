import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

def plot_peak_groups(csv_file_path, precursor_name="Unknown Precursor", 
                     figsize=(12, 8), dpi=100, save_path=None):
    """
    绘制Rust生成的final intensity数据的peak group图
    
    Parameters:
    -----------
    csv_file_path : str
        CSV文件路径
    precursor_name : str
        前体名称，用于图表标题
    figsize : tuple
        图表大小
    dpi : int
        图表分辨率
    save_path : str or None
        如果提供，将保存图表到该路径
    """
    
    # 读取数据
    print(f"读取数据文件: {csv_file_path}")
    data = pd.read_csv(csv_file_path)
    print(f"数据形状: {data.shape}")
    
    # 分离RT列和其他列
    rt_columns = [col for col in data.columns if not col in ['ProductMz', 'LibraryIntensity', 'frag_type', 'FragmentType']]
    rt_values = [float(col) for col in rt_columns]
    
    # 获取碎片信息
    fragment_info = data[['ProductMz', 'LibraryIntensity', 'frag_type', 'FragmentType']].copy()
    intensity_data = data[rt_columns].values
    
    # 过滤出frag_type为1或2的行（b和y离子）
    mask = data['frag_type'].isin([1, 2])
    filtered_indices = np.where(mask)[0]
    
    if len(filtered_indices) == 0:
        print("警告：没有找到frag_type为1或2的碎片")
        return
    
    # 创建图表
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, 
                                   gridspec_kw={'height_ratios': [3, 1]}, 
                                   dpi=dpi)
    
    # 设置颜色映射
    colors = {1: '#FF6B6B', 2: '#4ECDC4'}  # b离子：红色系，y离子：青色系
    frag_labels = {1: 'b ions', 2: 'y ions'}
    
    # 主图：绘制强度曲线
    max_intensity = 0
    for idx in filtered_indices:
        frag_type = int(data.loc[idx, 'frag_type'])
        product_mz = data.loc[idx, 'ProductMz']
        lib_intensity = data.loc[idx, 'LibraryIntensity']
        
        # 获取强度值
        intensities = intensity_data[idx]
        
        # 找到非零值
        non_zero_mask = intensities > 0
        if not any(non_zero_mask):
            continue
            
        non_zero_rt = np.array(rt_values)[non_zero_mask]
        non_zero_intensities = intensities[non_zero_mask]
        
        # 更新最大强度
        max_intensity = max(max_intensity, np.max(non_zero_intensities))
        
        # 绘制曲线
        label = f'{frag_labels[frag_type]} m/z={product_mz:.1f}'
        ax1.plot(non_zero_rt, non_zero_intensities, 
                marker='o', markersize=4, 
                color=colors[frag_type], 
                alpha=0.7, 
                linewidth=1.5,
                label=label if idx in filtered_indices[:4] else "")  # 只显示前4个标签
    
    # 设置主图属性
    ax1.set_xlabel('Retention Time (min)', fontsize=12)
    ax1.set_ylabel('Intensity', fontsize=12)
    ax1.set_title(f'Peak Group Plot - {precursor_name}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max_intensity * 1.1)
    
    # 添加图例
    if len(filtered_indices) <= 10:
        ax1.legend(loc='best', fontsize=8)
    
    # 添加RT范围指示
    rt_min, rt_max = min(rt_values), max(rt_values)
    rt_center = (rt_min + rt_max) / 2
    
    # 在顶部添加RT范围条
    ax1.axvspan(rt_min, rt_max, alpha=0.1, color='gray', ymax=0.05)
    
    # 热图：显示所有碎片的强度分布
    # 准备热图数据
    heatmap_data = []
    heatmap_labels = []
    
    for idx in filtered_indices:
        frag_type = int(data.loc[idx, 'frag_type'])
        product_mz = data.loc[idx, 'ProductMz']
        intensities = intensity_data[idx]
        
        # 归一化强度值
        if np.max(intensities) > 0:
            norm_intensities = intensities / np.max(intensities)
        else:
            norm_intensities = intensities
            
        heatmap_data.append(norm_intensities)
        heatmap_labels.append(f'{frag_labels[frag_type]}-{product_mz:.1f}')
    
    heatmap_data = np.array(heatmap_data)
    
    # 绘制热图
    im = ax2.imshow(heatmap_data, aspect='auto', cmap='viridis', 
                    extent=[rt_min, rt_max, 0, len(heatmap_data)])
    
    # 设置热图属性
    ax2.set_xlabel('Retention Time (min)', fontsize=12)
    ax2.set_ylabel('Fragments', fontsize=10)
    ax2.set_yticks(np.arange(len(heatmap_labels)) + 0.5)
    ax2.set_yticklabels(heatmap_labels, fontsize=8)
    
    # 添加颜色条
    cbar = plt.colorbar(im, ax=ax2, orientation='horizontal', pad=0.1)
    cbar.set_label('Normalized Intensity', fontsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 添加统计信息文本
    stats_text = f"Total fragments: {len(filtered_indices)}\n"
    stats_text += f"b ions: {sum(data.loc[filtered_indices, 'frag_type'] == 1)}\n"
    stats_text += f"y ions: {sum(data.loc[filtered_indices, 'frag_type'] == 2)}\n"
    stats_text += f"RT range: {rt_min:.2f} - {rt_max:.2f} min"
    
    # 在图表右上角添加统计信息
    ax1.text(0.98, 0.97, stats_text, 
             transform=ax1.transAxes, 
             fontsize=9,
             verticalalignment='top',
             horizontalalignment='right',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # 保存或显示图表
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"图表已保存到: {save_path}")
    
    plt.show()
    
    return fig, (ax1, ax2)


def plot_intensity_comparison(csv_file_path, precursor_name="Unknown Precursor", 
                            specific_fragments=None, figsize=(14, 8)):
    """
    绘制特定碎片的强度比较图
    
    Parameters:
    -----------
    csv_file_path : str
        CSV文件路径
    precursor_name : str
        前体名称
    specific_fragments : list or None
        要比较的特定碎片的索引列表，如果为None则显示前10个
    """
    
    # 读取数据
    data = pd.read_csv(csv_file_path)
    
    # 分离RT列
    rt_columns = [col for col in data.columns if not col in ['ProductMz', 'LibraryIntensity', 'frag_type', 'FragmentType']]
    rt_values = [float(col) for col in rt_columns]
    
    # 过滤碎片
    mask = data['frag_type'].isin([1, 2])
    filtered_data = data[mask].reset_index(drop=True)
    
    if specific_fragments is None:
        # 选择强度最高的前10个碎片
        max_intensities = filtered_data[rt_columns].max(axis=1)
        top_indices = max_intensities.nlargest(10).index
    else:
        top_indices = specific_fragments
    
    # 创建子图
    n_fragments = len(top_indices)
    n_cols = 2
    n_rows = (n_fragments + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_fragments > 1 else [axes]
    
    # 绘制每个碎片
    for i, idx in enumerate(top_indices):
        if i >= len(axes):
            break
            
        ax = axes[i]
        
        # 获取碎片信息
        frag_type = int(filtered_data.loc[idx, 'frag_type'])
        product_mz = filtered_data.loc[idx, 'ProductMz']
        lib_intensity = filtered_data.loc[idx, 'LibraryIntensity']
        
        # 获取强度值
        intensities = filtered_data.loc[idx, rt_columns].values
        
        # 绘制条形图和曲线
        bars = ax.bar(rt_values, intensities, width=0.3, alpha=0.5, 
                      color='#FF6B6B' if frag_type == 1 else '#4ECDC4')
        ax.plot(rt_values, intensities, 'o-', markersize=3, 
                color='darkred' if frag_type == 1 else 'darkblue')
        
        # 设置标题和标签
        frag_label = 'b' if frag_type == 1 else 'y'
        ax.set_title(f'{frag_label} ion - m/z {product_mz:.1f}', fontsize=10)
        ax.set_xlabel('RT (min)', fontsize=8)
        ax.set_ylabel('Intensity', fontsize=8)
        ax.tick_params(axis='both', labelsize=7)
        ax.grid(True, alpha=0.3)
        
        # 标记最高点
        max_idx = np.argmax(intensities)
        ax.annotate(f'{intensities[max_idx]:.0f}', 
                   xy=(rt_values[max_idx], intensities[max_idx]),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=7, color='red')
    
    # 隐藏多余的子图
    for i in range(n_fragments, len(axes)):
        axes[i].set_visible(False)
    
    # 设置总标题
    fig.suptitle(f'Fragment Intensity Profiles - {precursor_name}', 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    return fig, axes


# 主函数
if __name__ == "__main__":
    # 设置文件路径
    csv_file = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/20250716/timstof/LLIYGASTR2_final_dataframe.csv"
    
    # 基本的peak group图
    print("生成Peak Group图...")
    fig1, axes1 = plot_peak_groups(
        csv_file, 
        precursor_name="LLIYGASTR2",
        figsize=(12, 8),
        save_path="peak_group_plot.png"  # 可选：保存图片
    )
    
    # 碎片强度比较图
    print("\n生成碎片强度比较图...")
    fig2, axes2 = plot_intensity_comparison(
        csv_file,
        precursor_name="LLIYGASTR2",
        specific_fragments=None  # None表示自动选择强度最高的10个
    )
    
    # 额外的分析：打印数据摘要
    print("\n数据摘要：")
    data = pd.read_csv(csv_file)
    rt_columns = [col for col in data.columns if not col in ['ProductMz', 'LibraryIntensity', 'frag_type', 'FragmentType']]
    
    print(f"RT点数: {len(rt_columns)}")
    print(f"RT范围: {min(float(c) for c in rt_columns):.3f} - {max(float(c) for c in rt_columns):.3f} min")
    print(f"总碎片数: {len(data)}")
    print(f"b离子数: {sum(data['frag_type'] == 1)}")
    print(f"y离子数: {sum(data['frag_type'] == 2)}")
    print(f"其他离子数: {sum(~data['frag_type'].isin([1, 2]))}")