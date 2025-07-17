from alphatims.bruker import TimsTOF
import numpy as np
import pandas as pd

bruker_file = r"/Users/augustsirius/Desktop/DIABERT_test_code_lib/CAD20220207yuel_TPHP_DIA_pool1_Slot2-54_1_4382.d"        # ← 改成自己的文件
data = TimsTOF(bruker_file, mmap_detector_events=True)

frames = data.frames                         # Frames 表 = 一帧≈一条LC-MS 线
frag_frames = data.fragment_frames           # fragment-frames 表 = MS2 工作参数
meta = data.meta_data                        # GlobalMetaData 表

ms1_frames = frames[frames.MsMsType == 0]
ms2_frames = frames[frames.MsMsType != 0]

n_ms1_points = int(ms1_frames.NumPeaks.sum())     # 每帧 NumPeaks 已经是 detector events 数
n_ms2_points = int(ms2_frames.NumPeaks.sum())

print(f"MS1 数据点: {n_ms1_points:,}")
print(f"MS2 数据点: {n_ms2_points:,}")

# “原始”角度：precursor_index==0 则为 MS1，其余为 MS2
ms1_raw = data[:, :, 0, :, "raw"]
ms2_raw = data[:, :, 1:, :, "raw"]

print(len(ms1_raw), len(ms2_raw))        # 与上面结果一致

# —— MS1 —— #
rt_ms1      = (ms1_frames.Time.min(),   ms1_frames.Time.max())          # 秒
mob_ms1     = (data.mobility_min_value, data.mobility_max_value)        # 1/K0
mz_ms1      = (data.mz_min_value,       data.mz_max_value)              # Th

# —— MS2 —— #
rt_ms2      = (ms2_frames.Time.min(),   ms2_frames.Time.max())
# MS2 的四极杆 (Isolation) 范围：
quad_low  = frag_frames.IsolationMz - frag_frames.IsolationWidth/2
quad_high = frag_frames.IsolationMz + frag_frames.IsolationWidth/2
mz_ms2     = (quad_low.min(), quad_high.max())                          # Th
mob_ms2    = mob_ms1                                                    # MS2 仍然扫描全淌度

print(f"MS1 RT 范围   : {rt_ms1[0]:.2f} – {rt_ms1[1]:.2f} s")
print(f"MS1 Mobility : {mob_ms1[0]:.4f} – {mob_ms1[1]:.4f} 1/K0")
print(f"MS1 m/z      : {mz_ms1[0]:.2f} – {mz_ms1[1]:.2f} Th")

print(f"MS2 RT 范围   : {rt_ms2[0]:.2f} – {rt_ms2[1]:.2f} s")
print(f"MS2 Mobility : {mob_ms2[0]:.4f} – {mob_ms2[1]:.4f} 1/K0")
print(f"MS2 Isolation m/z 范围: {mz_ms2[0]:.2f} – {mz_ms2[1]:.2f} Th")