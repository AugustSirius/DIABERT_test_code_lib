#!/usr/bin/env python3
# tolerance_compare_results.py
# 容差版本的结果比较器，考虑f32/f64浮点精度差异，带进度条

import json
import os
import glob
import numpy as np
from typing import Dict, List, Tuple, Set
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from scipy.spatial import cKDTree


class ToleranceResultComparator:
    def __init__(self, result_dir: str):
        self.result_dir = result_dir
        self.results = {}
        
        # 设置容差参数
        self.tolerances = {
            'mz_ppm': 5.0,           # m/z相对误差 5 ppm
            'rt_seconds': 0.1,       # RT容差 0.1秒  
            'mobility_relative': 0.001,  # mobility相对误差 0.1%
            'intensity_relative': 0.01,  # 强度相对误差 1%
            'range_relative': 0.0001     # 范围相对误差 0.01%
        }
        
    def load_results(self):
        """加载所有JSON结果文件"""
        json_files = glob.glob(os.path.join(self.result_dir, "*.json"))
        
        print("正在加载数据文件...")
        for file_path in tqdm(json_files, desc="加载JSON文件"):
            # 跳过比较报告文件
            if 'comparison_report' in os.path.basename(file_path) or 'tolerance_comparison' in os.path.basename(file_path):
                continue
                
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                
                # 确保是数据文件而不是报告文件
                if not all(key in data for key in ['version', 'precursor_id', 'ms1_data', 'ms2_data']):
                    continue
                    
                key = (data['version'], data['precursor_id'])
                if key not in self.results:
                    self.results[key] = []
                self.results[key].append(data)
                
            except Exception as e:
                print(f"读取文件 {file_path} 时出错: {e}")
                continue
        
        data_files_count = sum(len(files) for files in self.results.values())
        print(f"加载了 {data_files_count} 个数据文件")
        
    def find_matching_pairs(self) -> List[Tuple[Dict, Dict]]:
        """找到原始版本和优化版本的配对结果"""
        pairs = []
        
        precursor_ids = set()
        for (version, precursor_id) in self.results.keys():
            precursor_ids.add(precursor_id)
        
        for precursor_id in precursor_ids:
            original_key = ('original', precursor_id)
            optimized_key = ('optimized', precursor_id)
            
            if original_key in self.results and optimized_key in self.results:
                original = sorted(self.results[original_key], 
                                key=lambda x: x['timestamp'], reverse=True)[0]
                optimized = sorted(self.results[optimized_key], 
                                 key=lambda x: x['timestamp'], reverse=True)[0]
                pairs.append((original, optimized))
        
        return pairs
    
    def _values_match_with_tolerance(self, val1: float, val2: float, 
                                   tolerance_type: str, reference_val: float = None) -> bool:
        """检查两个值是否在容差范围内匹配"""
        if tolerance_type == 'mz_ppm':
            # ppm = |val1 - val2| / reference_val * 1e6
            ref = reference_val if reference_val is not None else max(abs(val1), abs(val2))
            if ref == 0:
                return abs(val1 - val2) < 1e-10
            ppm_diff = abs(val1 - val2) / ref * 1e6
            return ppm_diff <= self.tolerances['mz_ppm']
            
        elif tolerance_type == 'rt_seconds':
            return abs(val1 - val2) <= self.tolerances['rt_seconds']
            
        elif tolerance_type == 'mobility_relative':
            ref = max(abs(val1), abs(val2))
            if ref == 0:
                return abs(val1 - val2) < 1e-10
            return abs(val1 - val2) / ref <= self.tolerances['mobility_relative']
            
        elif tolerance_type == 'intensity_relative':
            ref = max(abs(val1), abs(val2))
            if ref == 0:
                return abs(val1 - val2) < 1e-10
            return abs(val1 - val2) / ref <= self.tolerances['intensity_relative']
            
        elif tolerance_type == 'range_relative':
            ref = max(abs(val1), abs(val2))
            if ref == 0:
                return abs(val1 - val2) < 1e-10
            return abs(val1 - val2) / ref <= self.tolerances['range_relative']
            
        return False
    
    def _ranges_match_with_tolerance(self, range1: List[float], range2: List[float]) -> bool:
        """检查两个范围是否在容差内匹配"""
        if len(range1) != len(range2) or len(range1) != 2:
            return False
        
        return (self._values_match_with_tolerance(range1[0], range2[0], 'range_relative') and
                self._values_match_with_tolerance(range1[1], range2[1], 'range_relative'))
    
    # def _find_matching_data_points(self, data1: Dict, data2: Dict, data_type: str) -> Tuple[Set, Set, Set, List]:
    #     """找到匹配的数据点，返回(匹配的索引对, 仅在data1中的索引, 仅在data2中的索引, 强度差异列表)"""
        
    #     if len(data1['mz_values']) == 0 or len(data2['mz_values']) == 0:
    #         return set(), set(range(len(data1['mz_values']))), set(range(len(data2['mz_values']))), []
        
    #     # 为data2建立索引以加速查找
    #     data2_available = set(range(len(data2['mz_values'])))
        
    #     matched_pairs = set()
    #     intensity_diffs = []
        
    #     # 添加进度条，显示数据点匹配进度
    #     desc = f"匹配{data_type}数据点"
    #     for i in tqdm(range(len(data1['mz_values'])), desc=desc, leave=False):
    #         mz1 = data1['mz_values'][i]
    #         rt1 = data1['rt_values'][i]
    #         mob1 = data1['mobility_values'][i]
    #         int1 = data1['intensity_values'][i]
            
    #         # 在data2中寻找匹配点
    #         best_match = None
    #         best_match_idx = None
            
    #         for j in list(data2_available):  # 复制集合以避免修改时出错
    #             mz2 = data2['mz_values'][j]
    #             rt2 = data2['rt_values'][j]
    #             mob2 = data2['mobility_values'][j]
    #             int2 = data2['intensity_values'][j]
                
    #             # 检查是否匹配
    #             if (self._values_match_with_tolerance(mz1, mz2, 'mz_ppm', mz1) and
    #                 self._values_match_with_tolerance(rt1, rt2, 'rt_seconds') and
    #                 self._values_match_with_tolerance(mob1, mob2, 'mobility_relative')):
                    
    #                 best_match = (i, j)
    #                 best_match_idx = j
                    
    #                 # 检查强度差异
    #                 if not self._values_match_with_tolerance(int1, int2, 'intensity_relative'):
    #                     intensity_diffs.append(abs(int1 - int2))
                    
    #                 break
            
    #         if best_match:
    #             matched_pairs.add(best_match)
    #             data2_available.remove(best_match_idx)
        
    #     # 获取未匹配的索引
    #     matched_data1_indices = {pair[0] for pair in matched_pairs}
    #     matched_data2_indices = {pair[1] for pair in matched_pairs}
        
    #     unmatched_data1 = set(range(len(data1['mz_values']))) - matched_data1_indices
    #     unmatched_data2 = set(range(len(data2['mz_values']))) - matched_data2_indices
        
    #     return matched_pairs, unmatched_data1, unmatched_data2, intensity_diffs

    def _find_matching_data_points(self, data1: Dict, data2: Dict, data_type: str) -> Tuple[Set, Set, Set, List]:
        """使用KD-Tree优化的数据点匹配"""
        
        if len(data1['mz_values']) == 0 or len(data2['mz_values']) == 0:
            return set(), set(range(len(data1['mz_values']))), set(range(len(data2['mz_values']))), []
        
        # 首先转换为NumPy数组
        mz1 = np.array(data1['mz_values'])
        rt1 = np.array(data1['rt_values'])
        mob1 = np.array(data1['mobility_values'])
        int1 = np.array(data1['intensity_values'])
        
        mz2 = np.array(data2['mz_values'])
        rt2 = np.array(data2['rt_values'])
        mob2 = np.array(data2['mobility_values'])
        int2 = np.array(data2['intensity_values'])
        
        n1 = len(mz1)
        n2 = len(mz2)
        
        # 计算归一化尺度
        # 使用数据的平均值来估算尺度
        mz_scale = np.mean(mz1) * self.tolerances['mz_ppm'] / 1e6
        rt_scale = self.tolerances['rt_seconds']
        mob_mean = np.mean(mob1)
        if mob_mean > 0:
            mobility_scale = mob_mean * self.tolerances['mobility_relative']
        else:
            mobility_scale = 0.001  # 防止除零
        
        # 构建特征矩阵
        features1 = np.column_stack([
            mz1 / mz_scale,
            rt1 / rt_scale,
            mob1 / mobility_scale if mobility_scale > 0 else mob1
        ])
        
        features2 = np.column_stack([
            mz2 / mz_scale,
            rt2 / rt_scale,
            mob2 / mobility_scale if mobility_scale > 0 else mob2
        ])
        
        # 构建KD-Tree
        from scipy.spatial import cKDTree
        tree = cKDTree(features2)
        
        matched_pairs = set()
        intensity_diffs = []
        data2_used = set()
        
        # 使用KD-Tree查找最近邻
        desc = f"匹配{data_type}数据点"
        for i in tqdm(range(n1), desc=desc, leave=False):
            # 查找半径内的所有点
            indices = tree.query_ball_point(features1[i], r=1.0)
            
            if indices:
                # 在候选点中找最近的未使用点
                best_j = None
                best_dist = float('inf')
                
                for j in indices:
                    if j not in data2_used:
                        # 再次验证是否真的匹配（因为KD-Tree使用的是欧式距离）
                        if (self._values_match_with_tolerance(mz1[i], mz2[j], 'mz_ppm', mz1[i]) and
                            self._values_match_with_tolerance(rt1[i], rt2[j], 'rt_seconds') and
                            self._values_match_with_tolerance(mob1[i], mob2[j], 'mobility_relative')):
                            
                            dist = np.linalg.norm(features1[i] - features2[j])
                            if dist < best_dist:
                                best_dist = dist
                                best_j = j
                
                if best_j is not None:
                    matched_pairs.add((i, best_j))
                    data2_used.add(best_j)
                    
                    # 检查强度差异
                    if not self._values_match_with_tolerance(int1[i], int2[best_j], 'intensity_relative'):
                        intensity_diffs.append(abs(int1[i] - int2[best_j]))
        
        # 获取未匹配的索引
        matched_data1_indices = {pair[0] for pair in matched_pairs}
        matched_data2_indices = {pair[1] for pair in matched_pairs}
        
        unmatched_data1 = set(range(n1)) - matched_data1_indices
        unmatched_data2 = set(range(n2)) - matched_data2_indices
        
        return matched_pairs, unmatched_data1, unmatched_data2, intensity_diffs

    def _compare_ms_data_with_tolerance(self, data1: Dict, data2: Dict, data_type: str) -> Dict:
        """使用容差比较MS数据"""
        result = {
            'data_type': data_type,
            'data_points_original': data1['data_points'],
            'data_points_optimized': data2['data_points'],
            'data_points_diff': data1['data_points'] - data2['data_points'],
        }
        
        if data_type == 'MS2':
            result['num_fragments_original'] = data1['num_fragments']
            result['num_fragments_optimized'] = data2['num_fragments']
            result['fragments_diff'] = data1['num_fragments'] - data2['num_fragments']
        
        if data1['data_points'] > 0 and data2['data_points'] > 0:
            print(f"  正在比较{data_type}数据...")
            matched_pairs, unmatched_1, unmatched_2, intensity_diffs = \
                self._find_matching_data_points(data1, data2, data_type)
            
            result['matched_points'] = len(matched_pairs)
            result['unmatched_in_original'] = len(unmatched_1)
            result['unmatched_in_optimized'] = len(unmatched_2)
            result['intensity_differences_count'] = len(intensity_diffs)
            result['max_intensity_diff'] = max(intensity_diffs) if intensity_diffs else 0
            result['mean_intensity_diff'] = np.mean(intensity_diffs) if intensity_diffs else 0
            
            # 计算匹配率
            total_points = max(data1['data_points'], data2['data_points'])
            result['match_rate'] = len(matched_pairs) / total_points if total_points > 0 else 0
            
            # 评估数据质量
            if len(matched_pairs) > 0:
                intensity_diff_rate = result['intensity_differences_count'] / len(matched_pairs)
            else:
                intensity_diff_rate = 0
                
            if result['match_rate'] > 0.95 and intensity_diff_rate < 0.01:
                result['quality_assessment'] = '优秀'
            elif result['match_rate'] > 0.90 and intensity_diff_rate < 0.05:
                result['quality_assessment'] = '良好'
            elif result['match_rate'] > 0.80:
                result['quality_assessment'] = '一般'
            else:
                result['quality_assessment'] = '差'
                
            print(f"  {data_type}数据比较完成: 匹配率 {result['match_rate']:.3f}, 质量评估: {result['quality_assessment']}")
        else:
            result['quality_assessment'] = '无数据'
        
        return result
    
    def compare_slice_results(self, original: Dict, optimized: Dict) -> Dict:
        """使用容差比较两个版本的切片结果"""
        comparison = {
            'precursor_id': original['precursor_id'],
            'timestamps': {
                'original': original['timestamp'],
                'optimized': optimized['timestamp']
            },
            'tolerances_used': self.tolerances
        }
        
        # 比较范围
        ms1_orig, ms1_opt = original['ms1_data'], optimized['ms1_data']
        ms2_orig, ms2_opt = original['ms2_data'], optimized['ms2_data']
        
        print(f"正在比较范围...")
        comparison['range_comparison'] = {
            'mz_range_match': self._ranges_match_with_tolerance(
                ms1_orig['mz_range'], ms1_opt['mz_range']),
            'im_range_match': self._ranges_match_with_tolerance(
                ms1_orig['im_range'], ms1_opt['im_range']),
            'mz_range_original': ms1_orig['mz_range'],
            'mz_range_optimized': ms1_opt['mz_range'],
            'im_range_original': ms1_orig['im_range'],
            'im_range_optimized': ms1_opt['im_range']
        }
        
        # 比较MS数据
        comparison['ms1_comparison'] = self._compare_ms_data_with_tolerance(
            ms1_orig, ms1_opt, 'MS1')
        comparison['ms2_comparison'] = self._compare_ms_data_with_tolerance(
            ms2_orig, ms2_opt, 'MS2')
        
        return comparison
    
    def generate_report(self):
        """生成容差比较报告"""
        print("=== 开始生成容差比较报告 ===\n")
        
        self.load_results()
        pairs = self.find_matching_pairs()
        
        if not pairs:
            print("没有找到匹配的原始版本和优化版本结果对")
            return
        
        print(f"\n找到 {len(pairs)} 对匹配结果")
        print(f"使用的容差设置:")
        for key, value in self.tolerances.items():
            print(f"  {key}: {value}")
        print("\n")
        
        all_comparisons = []
        
        # 添加主要比较进度条
        for original, optimized in tqdm(pairs, desc="比较前体切片结果"):
            print(f"\n=== 正在处理 Precursor: {original['precursor_id']} ===")
            comparison = self.compare_slice_results(original, optimized)
            all_comparisons.append(comparison)
            
            print(f"=== Precursor: {comparison['precursor_id']} ===")
            print(f"时间戳:")
            print(f"  原始版本: {comparison['timestamps']['original']}")
            print(f"  优化版本: {comparison['timestamps']['optimized']}")
            
            # 范围比较
            range_comp = comparison['range_comparison']
            print(f"\n范围比较:")
            print(f"  m/z范围匹配: {range_comp['mz_range_match']} "
                  f"({range_comp['mz_range_original']} vs {range_comp['mz_range_optimized']})")
            print(f"  IM范围匹配: {range_comp['im_range_match']} "
                  f"({range_comp['im_range_original']} vs {range_comp['im_range_optimized']})")
            
            # MS1比较
            ms1_comp = comparison['ms1_comparison']
            print(f"\nMS1 数据比较:")
            print(f"  数据点数 - 原始: {ms1_comp['data_points_original']}, "
                  f"优化: {ms1_comp['data_points_optimized']}, "
                  f"差异: {ms1_comp['data_points_diff']}")
            
            if 'matched_points' in ms1_comp:
                print(f"  匹配的数据点: {ms1_comp['matched_points']}")
                print(f"  匹配率: {ms1_comp['match_rate']:.3f}")
                print(f"  仅在原始版本: {ms1_comp['unmatched_in_original']}")
                print(f"  仅在优化版本: {ms1_comp['unmatched_in_optimized']}")
                print(f"  强度差异点数: {ms1_comp['intensity_differences_count']}")
                if ms1_comp['max_intensity_diff'] > 0:
                    print(f"  最大强度差异: {ms1_comp['max_intensity_diff']:.2e}")
                    print(f"  平均强度差异: {ms1_comp['mean_intensity_diff']:.2e}")
                else:
                    print(f"  强度差异: 无")
                print(f"  质量评估: {ms1_comp['quality_assessment']}")
            
            # MS2比较
            ms2_comp = comparison['ms2_comparison']
            print(f"\nMS2 数据比较:")
            print(f"  碎片数 - 原始: {ms2_comp['num_fragments_original']}, "
                  f"优化: {ms2_comp['num_fragments_optimized']}, "
                  f"差异: {ms2_comp['fragments_diff']}")
            print(f"  数据点数 - 原始: {ms2_comp['data_points_original']}, "
                  f"优化: {ms2_comp['data_points_optimized']}, "
                  f"差异: {ms2_comp['data_points_diff']}")
            
            if 'matched_points' in ms2_comp:
                print(f"  匹配的数据点: {ms2_comp['matched_points']}")
                print(f"  匹配率: {ms2_comp['match_rate']:.3f}")
                print(f"  仅在原始版本: {ms2_comp['unmatched_in_original']}")
                print(f"  仅在优化版本: {ms2_comp['unmatched_in_optimized']}")
                print(f"  强度差异点数: {ms2_comp['intensity_differences_count']}")
                if ms2_comp['max_intensity_diff'] > 0:
                    print(f"  最大强度差异: {ms2_comp['max_intensity_diff']:.2e}")
                    print(f"  平均强度差异: {ms2_comp['mean_intensity_diff']:.2e}")
                else:
                    print(f"  强度差异: 无")
                print(f"  质量评估: {ms2_comp['quality_assessment']}")
            
            print("\n" + "="*70 + "\n")
        
        # 保存详细报告
        print("正在保存详细报告...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.result_dir, f"tolerance_comparison_report_{timestamp}.json")
        with open(report_file, 'w') as f:
            json.dump(all_comparisons, f, indent=2)
        
        print(f"详细报告已保存到: {report_file}")
        
        # 生成汇总
        self._generate_summary(all_comparisons)
    
    def _generate_summary(self, comparisons: List[Dict]):
        """生成汇总信息"""
        print("\n=== 汇总评估 ===")
        
        all_excellent = True
        all_good = True
        
        for comp in comparisons:
            range_comp = comp['range_comparison']
            ms1_comp = comp['ms1_comparison']
            ms2_comp = comp['ms2_comparison']
            
            # 检查范围匹配
            if not (range_comp['mz_range_match'] and range_comp['im_range_match']):
                all_excellent = False
                all_good = False
                print(f"⚠️  {comp['precursor_id']}: 范围不匹配")
                continue
            
            # 检查数据质量
            ms1_quality = ms1_comp.get('quality_assessment', '无数据')
            ms2_quality = ms2_comp.get('quality_assessment', '无数据')
            
            if ms1_quality != '优秀' or ms2_quality != '优秀':
                all_excellent = False
            
            if ms1_quality in ['差'] or ms2_quality in ['差']:
                all_good = False
            
            print(f"📊 {comp['precursor_id']}: MS1={ms1_quality}, MS2={ms2_quality}")
        
        print(f"\n最终评估:")
        if all_excellent:
            print("🎉 所有前体的切片结果在容差范围内完全一致，优化版本质量优秀！")
        elif all_good:
            print("✅ 所有前体的切片结果在容差范围内基本一致，优化版本质量良好！")
        else:
            print("⚠️  部分前体的切片结果存在显著差异，建议进一步检查")


if __name__ == "__main__":
    result_dir = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/20250714/timstof-slice-optimize/MS1MS2切片结果对比"
    comparator = ToleranceResultComparator(result_dir)
    comparator.generate_report()