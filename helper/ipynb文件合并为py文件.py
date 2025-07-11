import json
import os

def convert_ipynb_to_py(ipynb_path, output_path=None):
    """
    将 Jupyter notebook 文件转换为 Python 脚本
    
    参数:
    ipynb_path: .ipynb 文件的路径
    output_path: 输出 .py 文件的路径（可选，默认为同目录下同名 .py 文件）
    """
    
    # 如果没有指定输出路径，则使用与输入文件相同的名称但扩展名为 .py
    if output_path is None:
        output_path = os.path.splitext(ipynb_path)[0] + '.py'
    
    try:
        # 读取 ipynb 文件
        with open(ipynb_path, 'r', encoding='utf-8') as f:
            notebook = json.load(f)
        
        # 提取所有代码单元格
        python_code = []
        
        # 添加文件头注释
        python_code.append(f"# -*- coding: utf-8 -*-")
        python_code.append(f"# 从 {os.path.basename(ipynb_path)} 转换而来")
        python_code.append("")
        
        cell_count = 0
        for i, cell in enumerate(notebook['cells']):
            if cell['cell_type'] == 'code':
                cell_count += 1
                
                # 添加单元格分隔注释
                python_code.append(f"# ==================== Cell {cell_count} ====================")
                
                # 获取代码内容
                if 'source' in cell:
                    code_lines = cell['source']
                    if isinstance(code_lines, list):
                        # 如果是列表，直接连接
                        code_content = ''.join(code_lines)
                    else:
                        # 如果是字符串，直接使用
                        code_content = code_lines
                    
                    # 移除末尾的换行符，然后重新添加
                    code_content = code_content.rstrip('\n')
                    if code_content.strip():  # 只有非空代码才添加
                        python_code.append(code_content)
                        python_code.append("")  # 添加空行分隔
        
        # 写入 Python 文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(python_code))
        
        print(f"转换完成！")
        print(f"输入文件: {ipynb_path}")
        print(f"输出文件: {output_path}")
        print(f"共处理了 {cell_count} 个代码单元格")
        
    except FileNotFoundError:
        print(f"错误: 找不到文件 {ipynb_path}")
    except json.JSONDecodeError:
        print(f"错误: {ipynb_path} 不是有效的 JSON 文件")
    except Exception as e:
        print(f"错误: {str(e)}")

# 使用示例
if __name__ == "__main__":
    # 你的文件路径
    ipynb_file = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/DIABERT_test_code_lib_0628/3D_code/Untitled_im_rt_test.ipynb"
    
    # 可以指定输出文件名，或者让程序自动生成
    output_file = "/Users/augustsirius/Desktop/DIABERT_test_code_lib/DIABERT_test_code_lib_0628/3D_code/Untitled_im_rt_test.py"
    
    convert_ipynb_to_py(ipynb_file, output_file)