### 1 数据加载与预处理

#### 1.1 原始数据加载
```python
timstof_data = timstof_PASEF_20250506.TimsTOF(bruker_d_folder_name)
```
这一步加载Bruker的原始数据文件，数据结构为4D张量，包含：
- Frame（时间帧）
- Scan（扫描）
- m/z范围
- 强度值

#### 1.2 谱库数据处理
谱库包含了已知肽段的理论碎片离子信息：
- **前体离子信息**：m/z、电荷、肽段序列
- **碎片离子信息**：b/y离子的理论m/z值和相对强度
- **保留时间校准**：iRT值

代码中的列名映射确保了不同格式谱库的兼容性：
```python
lib_col_dict = utils.get_lib_col_dict()
```

### 2 特征提取与映射

#### 2.1 RT和IM信息提取
从DIA-NN的分析结果中提取每个前体离子的：
- **RT（保留时间）**：肽段在色谱柱上的洗脱时间
- **IM（离子淌度）**：离子在电场中的迁移率

```python
assay_rt_kept_dict = dict(zip(diann_precursor_id_all['transition_group_id'], 
                              diann_precursor_id_all['RT']))
assay_im_kept_dict = dict(zip(diann_precursor_id_all['transition_group_id'], 
                              diann_precursor_id_all['IM']))
```

### 3 张量化表示

#### 1 MS1数据构建
MS1数据包含前体离子及其同位素峰：
```python
def build_ms1_data(frag_list, iso_range, mz_max):
    # 计算同位素峰
    iso_shift_max = int(min(iso_range, (mz_max - precursor_mz) * charge)) + 1
    qt3_frags = [precursor_mz + iso_shift / charge for iso_shift in range(iso_shift_max)]
```

#### 2 MS2数据构建
MS2数据包含肽段的碎片离子：
```python
def build_ms2_data(frag_list, max_fragment_num):
    # 构建包含b/y离子的矩阵
    frag_count = max_fragment_num * frag_type_num
```

碎片类型编码：
- b离子：1（N端碎片）
- y离子：2（C端碎片）
- 其他类型：3, 4

### 4 m/z容差窗口计算

#### 4.1 动态容差计算
```python
def extract_width(mz_to_extract, mz_unit='ppm', mz_tol=50):
    if mz_unit == "ppm":
        mz_tol_half = mz_to_extract * mz_tol * 0.000001 / 2
```

这里采用ppm（百万分之一）作为质量精度单位

#### 4.2 窗口分割策略
代码将每个m/z窗口均匀分为5份：
```python
batch_num = int(mz_to_extract.shape[1] / frag_repeat_num)
cha_tensor = (extract_width_list[:, 0:batch_num, 1] - 
              extract_width_list[:, 0:batch_num, 0]) / frag_repeat_num
```
- 提高m/z分辨率
- 减少邻近离子干扰
- 便于并行计算

### 5 色谱峰提取

#### 5.1 数据筛选
基于IM值进行数据筛选：
```python
precursor_result = precursor_result[
    (precursor_result['mobility_values'] <= IM + 0.05) & 
    (precursor_result['mobility_values'] >= IM - 0.05)
]
```

IM窗口设置为±0.05，这个值基于：
- timsTOF仪器的IM分辨率
- 肽段离子的IM值稳定性

#### 5.2 RT窗口选择
```python
def get_rt_list(lst, target):
    # 选择目标RT附近的48个点
    closest_idx = min(range(len(lst)), key=lambda i: abs(lst[i] - target))
    start = max(0, closest_idx - 24)
    end = start + 48
```

选择48个RT点的考虑：
- 典型色谱峰宽度约30-60秒
- 采样频率通常为1-2 Hz
- 48个点可覆盖完整色谱峰

### 6 强度矩阵构建

#### 6.1 稀疏矩阵优化
代码使用了矩阵运算来处理稀疏数据：
```python
# 构建0/1掩码矩阵
mask_ms1 = torch.isin(ms1_extract_width_range_list[i], search_ms1_tensor)
ms1_frag_moz_matrix = torch.where(mask_ms1, 1., 0.)

# 矩阵乘法提取强度
ms1_frag_rt = ms1_frag_moz_matrix[a] @ ms1_moz_rt_matrix
```

这种方法的优势：
- 避免了大量的循环操作
- 利用了PyTorch的GPU加速能力
- 内存效率高

#### 6.2 数据聚合
将相同m/z的信号进行求和：
```python
grouped = precursor_result[precursor_result['rt_values_min'] == rt].groupby('mz_values')['intensity_values'].sum()
```
- 同一m/z可能有多个数据点
- 求和操作提高信噪比

### 7 数据融合与输出

最终将MS1和MS2数据进行融合：
```python
full_frag_rt_matrix = torch.cat([ms1_frag_rt_matrix1, ms2_frag_rt_matrix1], dim=1)
```

输出数据包含：
- RT维度的强度值
- 碎片离子m/z
- 谱库强度
- 碎片类型标识