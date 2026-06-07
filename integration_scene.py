import json
import os

def update_scenes_directly(original_scene_file, replacement_scene_files, output_scene_file):
    """
    直接根据新场景文件中的 ID，去替换原始大场景文件中的对应数据。
    """
    # 1. 加载原始的大场景文件 (底表)
    try:
        with open(original_scene_file, 'r', encoding='utf-8') as f:
            master_scene_data = json.load(f)
            if not isinstance(master_scene_data, dict):
                print("错误: 原始场景文件必须是字典(对象)结构。")
                return
    except FileNotFoundError:
        print(f"错误: 找不到原始场景文件 {original_scene_file}")
        return

    replace_count = 0

    # 2. 遍历你的新场景文件 (也就是那三百多个样本的场景文件)
    for rep_file in replacement_scene_files:
        if not os.path.exists(rep_file):
            print(f"⚠️ 警告: 找不到替换文件 {rep_file}，已跳过。")
            continue
            
        with open(rep_file, 'r', encoding='utf-8') as f:
            new_scene_data = json.load(f)
            
        # 3. 核心逻辑：直接遍历新文件中的所有 ID 进行覆盖
        for new_id, new_content in new_scene_data.items():
            if new_id in master_scene_data:
                # 用新的场景内容覆盖旧的
                master_scene_data[new_id] = new_content
                replace_count += 1
            else:
                # 如果出于某种原因，原始大文件里没有这个 ID，你可以选择直接追加进去
                master_scene_data[new_id] = new_content
                replace_count += 1
                print(f"💡 提示: 原大文件中不存在 ID {new_id}，已作为新数据追加。")

    # 4. 保存最终合并好的场景文件
    try:
        with open(output_scene_file, 'w', encoding='utf-8') as f:
            json.dump(master_scene_data, f, ensure_ascii=False, indent=2)
        print(f"🎉 替换完成！总共成功更新/追加了 {replace_count} 个场景数据。")
        print(f"💾 最终的场景文件已保存至: {output_scene_file}")
    except Exception as e:
        print(f"❌ 保存文件时发生错误: {e}")


if __name__ == '__main__':
    # ================= 配置区域 =================
    
    # 1. 原始的、包含所有数据的场景大文件
    ORIGINAL_SCENE_FILE = '3Line/ovis_3d_scene_graphs_0_None.json'
    
    # 2. 包含那三百多个新场景数据的文件列表 (如果你只有一个文件，列表里写一个就行)
    REPLACEMENT_SCENE_FILES = [
        'Size_Scale_Knowledge/Relative Size Comparison scene.json',
        'Size_Scale_Knowledge/Scale Consistency scene.json',
        'Spatial_Navigation_Knowledge/Accessibility Constraints_Scene.json',
        'Spatial_Navigation_Knowledge/Viewpoint Visibility_Scene.json',
        '3D_Geometry_Knowledge/Gravity Effects(2)_Scene.json',
        '3D_Geometry_Knowledge/Stability Prediction_Scene.json',
        'Orientation_Knowledge/Facing Direction_Scene.json',
        'Orientation_Knowledge/Tool Handedness_Scene.json',
        'Relative_Positioning_Knowledge/Alignment Patterns scene.json',
        'Relative_Positioning_Knowledge/Proximity Gradients scene.json' # 比如你上一步提取出来的场景文件
    ]
    
    # 3. 输出替换完成后的新场景文件路径
    OUTPUT_SCENE_FILE = 'ovis_3d_eval_0_None_merged_scenes.json'
    
    # ============================================

    update_scenes_directly(ORIGINAL_SCENE_FILE, REPLACEMENT_SCENE_FILES, OUTPUT_SCENE_FILE)