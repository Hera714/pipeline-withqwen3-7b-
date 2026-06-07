import json

def extract_scenes_by_ids(reference_file, scene_file, output_file):
    """
    根据参考文件中的 id，从场景文件中提取对应的样本。
    
    :param reference_file: 包含所需 id 的参考 JSON 文件路径（例如之前提取好的文件）
    :param scene_file: 原始场景 JSON 文件路径（截图所示格式）
    :param output_file: 输出的新 JSON 文件路径
    """
    # 1. 从参考文件中收集所有需要的 ID
    target_ids = set()
    try:
        with open(reference_file, 'r', encoding='utf-8') as f:
            ref_data = json.load(f)
            
        # 假设参考文件是一个列表，且每个元素都有 'id' 字段
        if isinstance(ref_data, list):
            for item in ref_data:
                if 'id' in item:
                    # 统一转换为字符串，方便后续匹配
                    target_ids.add(str(item['id'])) 
        else:
            print("错误：参考文件格式不是预期的列表格式。")
            return
            
    except FileNotFoundError:
        print(f"错误：找不到参考文件 {reference_file}")
        return

    print(f"从参考文件中获取了 {len(target_ids)} 个需要提取的 ID。")

    # 2. 读取场景文件并提取匹配的数据
    extracted_scenes = {}
    try:
        with open(scene_file, 'r', encoding='utf-8') as f:
            scene_data = json.load(f)
            
        # 确保场景文件最外层是一个字典 (对象)
        if not isinstance(scene_data, dict):
            print("错误：场景文件的根节点应该是一个字典（对象），以 id 为键。")
            return

        # 遍历我们需要提取的 ID 集合
        for target_id in target_ids:
            if target_id in scene_data:
                # 如果场景文件里有这个 id，就把整个对象提取过来
                extracted_scenes[target_id] = scene_data[target_id]
            else:
                # 如果找不到，打印一条警告
                print(f"警告: 场景文件中找不到 ID 为 {target_id} 的数据，已跳过。")

    except FileNotFoundError:
        print(f"错误：找不到场景文件 {scene_file}")
        return

    # 3. 将提取出来的场景数据写入新文件
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extracted_scenes, f, ensure_ascii=False, indent=2)
        print(f"提取完成！成功提取了 {len(extracted_scenes)} 个场景样本。")
        print(f"已保存至: {output_file}")
    except Exception as e:
        print(f"保存文件时发生错误: {e}")


if __name__ == '__main__':
    # ================= 配置区域 =================
    
    # 1. 包含你需要提取的 ID 的文件 (比如你上一步生成的 extracted_samples.json)
    REFERENCE_FILE = 'Proximity Gradients.json' 
    
    # 2. 你的原始场景文件 (包含 "3": {...} 的那个文件)
    SCENE_FILE = 'Alignment Patterns&Proximity Gradients scene.json'
    
    # 3. 提取后想要保存的新文件名
    OUTPUT_FILE = 'Proximity Gradients scene.json'
    
    # ============================================

    extract_scenes_by_ids(REFERENCE_FILE, SCENE_FILE, OUTPUT_FILE)