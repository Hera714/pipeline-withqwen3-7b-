#coding=utf-8
import json

def extract_by_subcategory(input_file, output_file, target_subcategory):
    """
    从 JSON 文件中提取特定 Sub_Category 的样本并保存到新文件。
    
    :param input_file: 原始 JSON 文件路径
    :param output_file: 输出的新 JSON 文件路径
    :param target_subcategory: 想要提取的子类别名称 (字符串)
    """
    try:
        # 1. 加载原始数据
        with open(input_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # 确保数据是一个列表格式
        if not isinstance(data, list):
            print("错误：JSON 根节点必须是一个列表（Array）。")
            return

        # 2. 过滤数据
        # 使用列表推导式提取匹配的样本
        filtered_samples = [
            item for item in data 
            if item.get('Sub_Category') == target_subcategory
        ]

        # 3. 保存结果
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_samples, f, ensure_ascii=False, indent=2)

        print(f"提取完成！")
        print(f"目标类别: {target_subcategory}")
        print(f"找到样本数: {len(filtered_samples)}")
        print(f"已保存至: {output_file}")

    except FileNotFoundError:
        print(f"错误：找不到文件 {input_file}")
    except json.JSONDecodeError:
        print(f"错误：{input_file} 不是有效的 JSON 格式")
    except Exception as e:
        print(f"发生未知错误: {e}")

# ================= 配置区域 =================
if __name__ == "__main__":
    # 输入文件名
    INPUT_PATH = 'Alignment Patterns&Proximity Gradients.json'
    
    # 输出文件名
    OUTPUT_PATH = 'Proximity Gradients.json'
    
    # 你想要提取的具体子类别名 (例如你图中的 "Layering Order")
    TARGET_SUB = 'Proximity Gradients'
    
    extract_by_subcategory(INPUT_PATH, OUTPUT_PATH, TARGET_SUB)