# coding=utf-8
import os
os.environ['MODELSCOPE_CACHE'] = '/root/autodl-tmp/modelscope_cache'

import json
import re
import torch
import gc
import traceback
from PIL import Image, ImageDraw
from tqdm import tqdm

from perception import Ovis3DPerceptionModule 
from depth_estimation_3d import DepthPro3DModule
from spatial_relations_generator import generate_spatial_relations
from spatial_relations_bbox import generate_bbox_relations
from knowledge_injector import GeometryKnowledgeInjector

# ================= 0. 消融实验严格控制开关 (Variant C 配置) =================
# Variant C (单一线): 路由=OFF(全走Line1), 细粒度特征=OFF, 符号引擎=ON, 知识注入=OFF
ENABLE_DYNAMIC_ROUTING = False         # ❌ 关闭 (强制所有任务走 Line 1 基础 3D 提取)
ENABLE_FINE_GRAINED_FEATURES = False   # ❌ 关闭 (不提取边界框和铰链点)
ENABLE_SYMBOLIC_ENGINE = True          # ✅ 开启 (基于 Line 1 的基础坐标生成物理距离文本)
ENABLE_EXTERNAL_KNOWLEDGE = False      # ❌ 关闭

# ================= 1. 测试区间与目标控制 =================
START_IDX = 0        
END_IDX = None       # 跑完整个数据集

USE_COMMON_DIMENSIONS = False  

# 消融实验应在全量数据集上运行，以保证 Overall 准确率分母一致
TARGET_TASK_ID = []        
TARGET_CATEGORY = []       
TARGET_SUB_CATEGORY = []

# ================= 2. 路由逻辑 (Variant C 专属) =================
def get_task_route(category, sub_category):
    """
    Variant C 的核心：剥夺路由能力。
    不论题目是什么大类小类，全部强制进入 Line 1 (基础 3D 坐标提取)。
    原本该走 Line 0 (Pure VLM) 的题目也会被强行拉入 3D 管线。
    """
    if not ENABLE_DYNAMIC_ROUTING:
        return "line1_base"
    return "line1_base"

# ================= 3. 配置路径 =================
MODEL_PATH = "AIDC-AI/Ovis2.5-9B" 
JSON_PATH = "/root/autodl-fs/data.json"

OUTPUT_EVAL_JSON = f"VariantC_eval.json" 
OUTPUT_SG_JSON = f"VariantC_scene.json"

IMAGE_DIR = "/root/autodl-fs/img"
OUT_POINTS_DIR = "/root/autodl-fs/img_points_VariantC"
os.makedirs(OUT_POINTS_DIR, exist_ok=True)

def extract_answer(response):
    if not response: return "Unknown"
    match = re.search(r'<answer>\s*([1-4])\s*</answer>', response)
    if match: return match.group(1)
    match = re.search(r'[1-4]', response)
    if match: return match.group(0)
    return "Unknown"

def main():
    vision_module = Ovis3DPerceptionModule(model_path=MODEL_PATH)
    depth_module = DepthPro3DModule()
    knowledge_module = GeometryKnowledgeInjector(kb_dir="/root/autodl-tmp/knowledge_bases", use_common_dimensions=USE_COMMON_DIMENSIONS)

    if os.path.exists(OUTPUT_EVAL_JSON):
        print(f"发现已有进度文件 {OUTPUT_EVAL_JSON}，正在恢复...")
        with open(OUTPUT_EVAL_JSON, "r", encoding="utf-8") as f: full_dataset = json.load(f)
    else:
        print(f"加载全新数据集 {JSON_PATH}...")
        with open(JSON_PATH, "r", encoding="utf-8") as f: full_dataset = json.load(f)

    scene_graphs_dict = {}
    if os.path.exists(OUTPUT_SG_JSON):
        with open(OUTPUT_SG_JSON, "r", encoding="utf-8") as f: scene_graphs_dict = json.load(f)

    correct_count = 0
    total_run_count = 0

    target_task_ids = [TARGET_TASK_ID] if isinstance(TARGET_TASK_ID, str) else TARGET_TASK_ID
    target_task_ids = [str(tid) for tid in target_task_ids if tid]
    target_cats = [TARGET_CATEGORY] if isinstance(TARGET_CATEGORY, str) else TARGET_CATEGORY
    target_cats = [c for c in target_cats if c] 
    target_subcats = [TARGET_SUB_CATEGORY] if isinstance(TARGET_SUB_CATEGORY, str) else TARGET_SUB_CATEGORY
    target_subcats = [sc for sc in target_subcats if sc]
    
    end_val = END_IDX if END_IDX is not None else len(full_dataset)
    test_subset = full_dataset[START_IDX:end_val]
    
    print(f"开始 Variant C (单一线/All Line1) 评估。测试区间: {START_IDX} -> {end_val}")
    
    for item in tqdm(test_subset):
        task_id = str(item.get("id"))
        category = item.get("Category", "").strip()
        sub_category = item.get("Sub_Category", "").strip()
        
        if target_task_ids and task_id not in target_task_ids: continue
        if target_cats and category not in target_cats: continue
        if target_subcats and sub_category not in target_subcats: continue
            
        total_run_count += 1

        try:
            image_path = os.path.join(IMAGE_DIR, item.get("Image_Filename"))
            image = Image.open(image_path).convert("RGB")
            options_text = f"1. {item.get('Option_1', '')}\n2. {item.get('Option_2', '')}\n3. {item.get('Option_3', '')}\n4. {item.get('Option_4', '')}"
            
            # Variant C 会把这里强制锁定为 line1_base
            task_route = get_task_route(category, sub_category)

            # ================= 阶段 1：感知提取 =================
            target_perspective, objects_data = vision_module.extract_objects_3d(
                image, item['Question'], options_text, task_route=task_route
            )
            gc.collect(); torch.cuda.empty_cache()

            # ================= 阶段 2 & 3：深度估计与 3D 反投影 =================
            depth_matrix, focal_length_px = depth_module.estimate_depth(image)
            scene_graph = []
            for obj in objects_data:
                pos_3d = depth_module.get_3d_coordinates(depth_matrix, obj["point_2d"], focal_length_px)
                
                sg_item = {
                    "name": obj.get("name", "Unknown"),
                    "point_2d": obj["point_2d"],
                    "global_3d_position": pos_3d,
                    "reason": obj.get("reason", "")
                }
                # Line 1 仅提取朝向
                sg_item["facing_direction"] = obj.get("facing_direction", "Unknown")
                    
                scene_graph.append(sg_item)
            gc.collect(); torch.cuda.empty_cache()

            # 绘制调试图 (Variant C 用绿色点表示)
            img_with_points = image.copy()
            if scene_graph:
                draw = ImageDraw.Draw(img_with_points)
                for sg_item in scene_graph:
                    x, y = sg_item["point_2d"]
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="green", outline="white") 
                    draw.text((x + 8, y - 10), sg_item['name'], fill="green")
            img_with_points.save(os.path.join(OUT_POINTS_DIR, f"task_{task_id}_VariantC_points.png"))

            # ================= 阶段 3.5：符号引擎 (文本规则生成) =================
            spatial_relations_text = ""
            if ENABLE_SYMBOLIC_ENGINE:
                # 即使是 Line 1，我们也调用 generate_spatial_relations 来生成欧氏距离等基础物理信息
                spatial_relations_text = generate_spatial_relations(scene_graph, target_perspective)
                
                if spatial_relations_text.strip() == "Not enough objects to determine relative positioning.":
                    spatial_relations_text = ""

            # ================= 阶段 3.8：外部先验知识注入 =================
            # Variant C 明确关闭此项
            if ENABLE_EXTERNAL_KNOWLEDGE:
                pass 

            scene_graphs_dict[task_id] = {
                "task_route_used": task_route,
                "target_perspective": target_perspective,
                "objects": scene_graph,
                "spatial_relations_text": spatial_relations_text 
            }

            # ================= 阶段 4：纯净推理 =================
            response = vision_module.perform_final_reasoning(
                image, item['Question'], options_text, scene_graph, spatial_relations_text, task_route=task_route
            )
            prediction = extract_answer(response)

            item["Model_Raw_Response"] = response
            item["Model_Prediction"] = prediction
            item["Is_Correct"] = (prediction == str(item["Answer"]))
            
            if item["Is_Correct"]: correct_count += 1

        except Exception as e:
            print(f"\n[严重崩溃!] 任务 {task_id} 在运行中发生异常: {str(e)}")
            item["Model_Prediction"] = "Error"
            item["Model_Raw_Response"] = f"Runtime Error: {str(e)}\n{traceback.format_exc()}"
            item["Is_Correct"] = False
            scene_graphs_dict[task_id] = {"task_route_used": "Error", "objects": []}

        finally:
            with open(OUTPUT_EVAL_JSON, "w", encoding="utf-8") as f: json.dump(full_dataset, f, ensure_ascii=False, indent=4)
            with open(OUTPUT_SG_JSON, "w", encoding="utf-8") as f: json.dump(scene_graphs_dict, f, ensure_ascii=False, indent=4)

    accuracy = (correct_count / total_run_count if total_run_count > 0 else 0) * 100
    print(f"\n=== Variant C 跑分结束 ===")
    print(f"有效任务数: {total_run_count} | 正确数: {correct_count} | 准确率: {accuracy:.2f}%")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()