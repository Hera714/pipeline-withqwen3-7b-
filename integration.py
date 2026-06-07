#coding=utf-8
import json
import os

def update_json_with_inference(original_file, replacement_files, output_file):
    # 1. 读取原始 JSON 文件
    try:
        with open(original_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)
    except FileNotFoundError:
        print(f"错误: 找不到原始文件 {original_file}")
        return

    # 将原始数据转换为字典，以 id 为键，方便快速定位和替换
    # 确保 id 统一转换为字符串进行匹配
    data_dict = {str(item['id']): item for item in original_data}
    
    replace_count = 0

    # 2. 遍历所有包含替换内容的 JSON 文件
    for rep_file in replacement_files:
        if not os.path.exists(rep_file):
            print(f"警告: 文件 {rep_file} 不存在，已跳过。")
            continue
            
        with open(rep_file, 'r', encoding='utf-8') as f:
            rep_data = json.load(f)
            
        # 3. 遍历替换文件中的每个样本
        for item in rep_data:
            item_id = str(item.get('id'))
            
            # 判断该样本是否有模型推理过程
            # 这里以是否存在 "Model_Raw_Response" 字段为判断依据
            if 'Model_Raw_Response' in item:
                if item_id in data_dict:
                    data_dict[item_id] = item  # 用新样本替换老样本
                    replace_count += 1
                else:
                    print(f"警告: 在原文件中找不到 id 为 {item_id} 的样本，无法替换。")

    # 4. 重新组装列表，保持原始文件中样本的顺序
    updated_data = []
    for orig_item in original_data:
        orig_id = str(orig_item['id'])
        updated_data.append(data_dict[orig_id])

    # 5. 将更新后的数据写入新的 JSON 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        # indent=2 保证输出的 JSON 格式美观，像你截图里那样带缩进
        json.dump(updated_data, f, ensure_ascii=False, indent=2)

    print(f"处理完成！成功替换了 {replace_count} 个样本。")
    print(f"合并后的文件已保存至: {output_file}")


if __name__ == '__main__':
    # ================= 配置区域 =================
    
    # 原始文件路径
    ORIGINAL_FILE = '3Line/ovis_3d_eval_0_None.json'
    
    # 需要用来替换的多个文件路径列表 (请根据实际情况修改文件名)
    REPLACEMENT_FILES = [
        'Size_Scale_Knowledge/Relative Size Comparison.json',
        'Size_Scale_Knowledge/Scale Consistency.json',
        'Spatial_Navigation_Knowledge/Accessibility Constraints.json',
        'Spatial_Navigation_Knowledge/Viewpoint Visibility.json',
        '3D_Geometry_Knowledge/Gravity Effects(2).json',
        '3D_Geometry_Knowledge/Stability Prediction.json',
        'Orientation_Knowledge/Facing Direction.json',
        'Orientation_Knowledge/Tool Handedness.json',
        'Relative_Positioning_Knowledge/Alignment Patterns.json',
        'Relative_Positioning_Knowledge/Proximity Gradients.json'
        
    ]
    
    # 输出合并后的新文件路径
    OUTPUT_FILE = 'ovis_3d_eval_0_None_merged.json'
    
    # ============================================

    update_json_with_inference(ORIGINAL_FILE, REPLACEMENT_FILES, OUTPUT_FILE)