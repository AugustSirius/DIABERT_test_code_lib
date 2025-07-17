#!/usr/bin/env python3
# compare_results.py
# 保存到: /Users/augustsirius/Desktop/DIABERT_test_code_lib/20250714/timstof-slice-optimize/MS1MS2切片结果对比/compare_results.py

import json
import os
import glob
import numpy as np
from typing import Dict, List, Tuple
import pandas as pd
from datetime import datetime

class ResultComparator:
    def __init__(self, result_dir: str):
        self.result_dir = result_dir
        self.results = {}
        
    def load_results(self):
        """加载所有JSON结果文件"""
        json_files = glob.glob(os.path.join(self.result_dir, "*.json"))
        
        for file_path in json_files:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            key = (data['version'], data['precursor_id'])
            if key not in self.results:
                self.results[key] = []
            self.results[key].append(data)
        
        print(f"加载了 {len(json_files)} 个结果文件")
        
    def find_matching_pairs(self) -> List[Tuple[Dict, Dict]]:
        """找到原始版本和优化版本的配对结果"""
        pairs = []
        
        # 获取所有的precursor_id
        precursor_ids = set()
        for (version, precursor_id) in self.results.keys():
            precursor_ids.add(precursor_id)
        
        for precursor_id in precursor_ids:
            original_key = ('original', precursor_id)
            optimized_key = ('optimized', precursor_id)
            
            if original_key in self.results and optimized_key in self.results:
                # 取最新的结果
                original = sorted(self.results[original_key], 
                                key=lambda x: x['timestamp'], reverse=True)[0]
                optimized = sorted(self.results[optimized_key], 
                                 key=lambda x: x['timestamp'], reverse=True)[0]
                pairs.append((original, optimized))
        
        return pairs
    
    def compare_slice_results(self, original: Dict, optimized: Dict) -> Dict:
        """详细比较两个版本的切片结果"""
        comparison = {
            'precursor_id': original['precursor_id'],
            'ms1_comparison': self._compare_ms1_data(
                original['ms1_data'], 
                optimized['ms1_data']
            ),
            'ms2_comparison': self._compare_ms2_data(
                original['ms2_data'], 
                optimized['ms2_data']
            ),
            'timestamps': {
                'original': original['timestamp'],
                'optimized': optimized['timestamp']
            }
        }
        
        return comparison
    
    def _compare_ms1_data(self, ms1_orig: Dict, ms1_opt: Dict) -> Dict:
        """比较MS1数据"""
        result = {
            'mz_range_match': ms1_orig['mz_range'] == ms1_opt['mz_range'],
            'im_range_match': ms1_orig['im_range'] == ms1_opt['im_range'],
            'data_points_original': ms1_orig['data_points'],
            'data_points_optimized': ms1_opt['data_points'],
            'data_points_diff': ms1_orig['data_points'] - ms1_opt['data_points'],
        }
        
        # 比较数据内容
        if ms1_orig['data_points'] > 0 and ms1_opt['data_points'] > 0:
            # 创建数据字典以便比较
            orig_data = {}
            for i in range(len(ms1_orig['mz_values'])):
                key = (
                    round(ms1_orig['mz_values'][i], 6),
                    round(ms1_orig['rt_values'][i], 6),
                    round(ms1_orig['mobility_values'][i], 6)
                )
                orig_data[key] = ms1_orig['intensity_values'][i]
            
            opt_data = {}
            for i in range(len(ms1_opt['mz_values'])):
                key = (
                    round(ms1_opt['mz_values'][i], 6),
                    round(ms1_opt['rt_values'][i], 6),
                    round(ms1_opt['mobility_values'][i], 6)
                )
                opt_data[key] = ms1_opt['intensity_values'][i]
            
            # 检查数据一致性
            common_keys = set(orig_data.keys()) & set(opt_data.keys())
            only_in_original = set(orig_data.keys()) - set(opt_data.keys())
            only_in_optimized = set(opt_data.keys()) - set(orig_data.keys())
            
            result['common_points'] = len(common_keys)
            result['only_in_original'] = len(only_in_original)
            result['only_in_optimized'] = len(only_in_optimized)
            
            # 检查共同点的强度是否一致
            intensity_diffs = []
            for key in common_keys:
                diff = abs(orig_data[key] - opt_data[key])
                if diff > 0:
                    intensity_diffs.append(diff)
            
            result['intensity_differences'] = len(intensity_diffs)
            result['max_intensity_diff'] = max(intensity_diffs) if intensity_diffs else 0
            
        return result
    
    def _compare_ms2_data(self, ms2_orig: Dict, ms2_opt: Dict) -> Dict:
        """比较MS2数据"""
        result = {
            'num_fragments_original': ms2_orig['num_fragments'],
            'num_fragments_optimized': ms2_opt['num_fragments'],
            'fragments_diff': ms2_orig['num_fragments'] - ms2_opt['num_fragments'],
            'data_points_original': ms2_orig['data_points'],
            'data_points_optimized': ms2_opt['data_points'],
            'data_points_diff': ms2_orig['data_points'] - ms2_opt['data_points'],
        }
        
        # 类似MS1的详细比较
        if ms2_orig['data_points'] > 0 and ms2_opt['data_points'] > 0:
            orig_data = {}
            for i in range(len(ms2_orig['mz_values'])):
                key = (
                    round(ms2_orig['mz_values'][i], 6),
                    round(ms2_orig['rt_values'][i], 6),
                    round(ms2_orig['mobility_values'][i], 6)
                )
                orig_data[key] = ms2_orig['intensity_values'][i]
            
            opt_data = {}
            for i in range(len(ms2_opt['mz_values'])):
                key = (
                    round(ms2_opt['mz_values'][i], 6),
                    round(ms2_opt['rt_values'][i], 6),
                    round(ms2_opt['mobility_values'][i], 6)
                )
                opt_data[key] = ms2_opt['intensity_values'][i]
            
            common_keys = set(orig_data.keys()) & set(opt_data.keys())
            only_in_original = set(orig_data.keys()) - set(opt_data.keys())
            only_in_optimized = set(opt_data.keys()) - set(orig_data.keys())
            
            result['common_points'] = len(common_keys)
            result['only_in_original'] = len(only_in_original)
            result['only_in_optimized'] = len(only_in_optimized)
            
            intensity_diffs = []
            for key in common_keys:
                diff = abs(orig_data[key] - opt_data[key])
                if diff > 0:
                    intensity_diffs.append(diff)
            
            result['intensity_differences'] = len(intensity_diffs)
            result['max_intensity_diff'] = max(intensity_diffs) if intensity_diffs else 0
            
        return result
    
    def generate_report(self):
        """生成比较报告"""
        self.load_results()
        pairs = self.find_matching_pairs()
        
        if not pairs:
            print("没有找到匹配的原始版本和优化版本结果对")
            return
        
        print(f"\n找到 {len(pairs)} 对匹配结果\n")
        
        all_comparisons = []
        
        for original, optimized in pairs:
            comparison = self.compare_slice_results(original, optimized)
            all_comparisons.append(comparison)
            
            print(f"=== Precursor: {comparison['precursor_id']} ===")
            print(f"时间戳:")
            print(f"  原始版本: {comparison['timestamps']['original']}")
            print(f"  优化版本: {comparison['timestamps']['optimized']}")
            
            print(f"\nMS1 比较:")
            ms1_comp = comparison['ms1_comparison']
            print(f"  m/z范围匹配: {ms1_comp['mz_range_match']}")
            print(f"  IM范围匹配: {ms1_comp['im_range_match']}")
            print(f"  数据点数 - 原始: {ms1_comp['data_points_original']}, 优化: {ms1_comp['data_points_optimized']}, 差异: {ms1_comp['data_points_diff']}")
            
            if 'common_points' in ms1_comp:
                print(f"  共同数据点: {ms1_comp['common_points']}")
                print(f"  仅在原始版本: {ms1_comp['only_in_original']}")
                print(f"  仅在优化版本: {ms1_comp['only_in_optimized']}")
                print(f"  强度差异点数: {ms1_comp['intensity_differences']}")
                print(f"  最大强度差异: {ms1_comp['max_intensity_diff']}")
            
            print(f"\nMS2 比较:")
            ms2_comp = comparison['ms2_comparison']
            print(f"  碎片数 - 原始: {ms2_comp['num_fragments_original']}, 优化: {ms2_comp['num_fragments_optimized']}, 差异: {ms2_comp['fragments_diff']}")
            print(f"  数据点数 - 原始: {ms2_comp['data_points_original']}, 优化: {ms2_comp['data_points_optimized']}, 差异: {ms2_comp['data_points_diff']}")
            
            if 'common_points' in ms2_comp:
                print(f"  共同数据点: {ms2_comp['common_points']}")
                print(f"  仅在原始版本: {ms2_comp['only_in_original']}")
                print(f"  仅在优化版本: {ms2_comp['only_in_optimized']}")
                print(f"  强度差异点数: {ms2_comp['intensity_differences']}")
                print(f"  最大强度差异: {ms2_comp['max_intensity_diff']}")
            
            print("\n" + "="*50 + "\n")
        
        # 保存详细报告
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.result_dir, f"comparison_report_{timestamp}.json")
        with open(report_file, 'w') as f:
            json.dump(all_comparisons, f, indent=2)
        
        print(f"详细报告已保存到: {report_file}")
        
        # 生成汇总
        self._generate_summary(all_comparisons)
    
    def _generate_summary(self, comparisons: List[Dict]):
        """生成汇总信息"""
        print("\n=== 汇总 ===")
        
        all_match = True
        for comp in comparisons:
            ms1 = comp['ms1_comparison']
            ms2 = comp['ms2_comparison']
            
            # 检查是否完全匹配
            if (ms1['data_points_diff'] != 0 or 
                ms2['data_points_diff'] != 0 or
                ms1.get('only_in_original', 0) > 0 or
                ms1.get('only_in_optimized', 0) > 0 or
                ms2.get('only_in_original', 0) > 0 or
                ms2.get('only_in_optimized', 0) > 0 or
                ms1.get('intensity_differences', 0) > 0 or
                ms2.get('intensity_differences', 0) > 0):
                all_match = False
                break
        
        if all_match:
            print("✅ 所有前体的切片结果完全一致！")
        else:
            print("❌ 发现不一致的结果，请查看详细报告")


if __name__ == "__main__":
    result_dir = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/20250714/timstof-slice-optimize/MS1MS2切片结果对比"
    comparator = ResultComparator(result_dir)
    comparator.generate_report()