#!/usr/bin/env python3
# debug_json_structure.py
# 用于调试JSON文件结构

import json
import os
import glob

def debug_json_files():
    """调试JSON文件结构"""
    result_dir = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/20250714/timstof-slice-optimize/MS1MS2切片结果对比"
    json_files = glob.glob(os.path.join(result_dir, "*.json"))
    
    print(f"找到 {len(json_files)} 个JSON文件:")
    
    for i, file_path in enumerate(json_files):
        print(f"\n=== 文件 {i+1}: {os.path.basename(file_path)} ===")
        
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            print(f"数据类型: {type(data)}")
            
            if isinstance(data, dict):
                print("字典键:")
                for key in data.keys():
                    print(f"  - {key}")
                    
            elif isinstance(data, list):
                print(f"列表长度: {len(data)}")
                if len(data) > 0:
                    print(f"第一个元素类型: {type(data[0])}")
                    if isinstance(data[0], dict):
                        print("第一个元素的键:")
                        for key in data[0].keys():
                            print(f"  - {key}")
                            
            # 显示数据的前几行
            print("\n数据预览:")
            if isinstance(data, dict):
                # 只显示一些关键字段
                preview_keys = ['version', 'precursor_id', 'timestamp']
                for key in preview_keys:
                    if key in data:
                        print(f"  {key}: {data[key]}")
            elif isinstance(data, list) and len(data) > 0:
                print(f"  第一个元素: {data[0]}")
                        
        except Exception as e:
            print(f"读取文件时出错: {e}")

if __name__ == "__main__":
    debug_json_files()