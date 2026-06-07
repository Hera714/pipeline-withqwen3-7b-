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

# ================= 0. 消融实验严格控制开关 (Variant B 配置) =================
# Variant B (粗感知): 路由=ON, 细粒度特征=OFF, 符号引擎=ON, 知识注入=OFF
ENABLE_DYNAMIC_ROUTING = True          # ✅ 开启
ENABLE_FINE_GRAINED_FEATURES = False   # ❌ 关闭 (强制降级为基础点坐标提取，无 BBox 和 铰链点)
ENABLE_SYMBOLIC_ENGINE = True          # ✅ 开启 (计算基础 3D 距离和画幅关系)
ENABLE_EXTERNAL_KNOWLEDGE = False      # ❌ 关闭

# ================= 1. 测试区间与目标控制 =================
START_IDX = 0        
END_IDX = None       
USE_COMMON_DIMENSIONS = False  

# 支持列表过滤
TARGET_TASK_ID = []        
TARGET_CATEGORY = []       
TARGET_SUB_CATEGORY = []

# ================= 2. 硬路由字典配置 (Oracle Routing) =================
LINE0_CONFIG = {
    "3D Geometry": ["Volume Comparison"], 
    "Depth & Occlusion": ["Reflective Surfaces"],
    "Orientation":["Cardinal Direction"],
    "Relative Positioning":["Betweenness Relationships"],
    "Size & Scale":["Scale Consistency"],
    "Spatial Navigation":["Accessibility Constraints","Pathway Existence"]
}

LINE2_CONFIG = {
    "Orientation": ["Facing Direction","Object Rotation","Stacking Orientation","Tool Handedness"],
    "3D Geometry":["Spatial Containment","Shape Projection","Gravity Effects"],
    "Depth & Occlusion":["Complete Occlusion Inference","Layering Order","Partial Occlusion"],
    "Relative Positioning": ["Corner/Angle Positioning","Directional Relations","Proximity Gradients"],
}

LINE3_CONFIG = {
    "Relative Positioning": ["Alignment Patterns"],
    "Size & Scale":["*"]
}

def get_task_route(category, sub_category):
    if not ENABLE_DYNAMIC_ROUTING: return "line1_base" # 如果关闭路由，全部降级为 Line 1
    
    if category in LINE0_CONFIG and (sub_category in LINE0_CONFIG[category] or "*" in LINE0_CONFIG[category]):
        return "line0_pure_vlm"
    if category in LINE2_CONFIG and (sub_category in LINE2_CONFIG[category] or "*" in LINE2_CONFIG[category]):
        return "line2_orientation"
    if category in LINE3_CONFIG and (sub_category in LINE3_CONFIG[category] or "*" in LINE3_CONFIG[category]):
        return "line3_bbox"
    return "line1_base"

# ================= 3. 配置路径 =================
MODEL_PATH = "AIDC-AI/Ovis2.5-9B" 
JSON_PATH = "/root/autodl-fs/data.json"

OUTPUT_EVAL_JSON = f"VariantB_eval.json" 
OUTPUT_SG_JSON = f"VariantB_scene.json"

IMAGE_DIR = "/root/autodl-fs/img"
OUT_POINTS_DIR = "/root/autodl-fs/img_points_VariantB"
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
    
    print(f"开始 Variant B (粗感知) 评估。测试区间: {START_IDX} -> {end_val}")
    
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
            
            task_route = get_task_route(category, sub_category)

            # ================= Line 0: 纯原生推理 =================
            if task_route == "line0_pure_vlm":
                system_prompt = "Answer the question directly based on the image.\nOutput a <think> block for step-by-step reasoning, then your exact answer as a single number (1-4) inside <answer> tags."
                
                prompt_text = system_prompt
                if ENABLE_EXTERNAL_KNOWLEDGE:
                    external_knowledge_text = knowledge_module.inject_knowledge([], category, sub_category, item['Question'])
                    if external_knowledge_text:
                        prompt_text += f"\n\n--- EXTERNAL DOMAIN KNOWLEDGE ---\n{external_knowledge_text}"
                        
                prompt_text += f"\n\nQuestion:\n{item['Question']}\nOptions:\n{options_text}\nAnswer (1-4):"

                messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
                response = vision_module._generate_response(messages, max_tokens=2048, enable_thinking=True)
                prediction = extract_answer(response)
                
                item["Model_Raw_Response"] = response
                item["Model_Prediction"] = prediction
                item["Is_Correct"] = (prediction == str(item["Answer"]))
                scene_graphs_dict[task_id] = {"task_route_used": task_route, "objects": [], "spatial_relations_text": "Bypassed via Pure VLM."}
                
                if item["Is_Correct"]: correct_count += 1
                continue

            # ================= 阶段 1：感知提取 =================
            # 【Variant B 核心逻辑】：强制将特征提取降级为 line1_base (只提取中心点，不提取BBox和3D铰链)
            perception_route = task_route if ENABLE_FINE_GRAINED_FEATURES else "line1_base"
            
            target_perspective, objects_data = vision_module.extract_objects_3d(
                image, item['Question'], options_text, task_route=perception_route
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

                # 仅当开启了细粒度特征时，才计算 BBox 尺寸和投影三点铰链
                if ENABLE_FINE_GRAINED_FEATURES:
                    if task_route == "line3_bbox" and "bbox_2d" in obj:
                        sg_item["bbox_2d"] = obj.get("bbox_2d")
                        dims = depth_module.get_3d_bbox_dimensions(depth_matrix, obj["bbox_2d"], obj["point_2d"], focal_length_px)
                        if dims: sg_item.update(dims)
                    elif task_route == "line2_orientation":
                        sg_item["anchor_parts"] = obj.get("anchor_parts", "")
                        sg_item["visible_profile"] = obj.get("visible_profile", "unknown")
                        sg_item["2d_pointing_direction"] = obj.get("2d_pointing_direction", "unknown")
                        if "hinge_points_2d" in obj:
                            sg_item["hinge_points_3d"] = [depth_module.get_3d_coordinates(depth_matrix, hp, focal_length_px) for hp in obj["hinge_points_2d"]]
                elif task_route == "line1_base":
                    sg_item["facing_direction"] = obj.get("facing_direction", "Unknown")
                    
                scene_graph.append(sg_item)
            gc.collect(); torch.cuda.empty_cache()

            # 绘制调试图
            img_with_points = image.copy()
            if scene_graph:
                draw = ImageDraw.Draw(img_with_points)
                for sg_item in scene_graph:
                    x, y = sg_item["point_2d"]
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="blue", outline="white") # Variant B 用蓝色点表示
                    draw.text((x + 8, y - 10), sg_item['name'], fill="blue")
            img_with_points.save(os.path.join(OUT_POINTS_DIR, f"task_{task_id}_VariantB_points.png"))

            # ================= 阶段 3.5：符号引擎 (文本规则生成) =================
            spatial_relations_text = ""
            if ENABLE_SYMBOLIC_ENGINE:
                if task_route == "line2_orientation":
                    spatial_relations_text = generate_spatial_relations(scene_graph, target_perspective)
                elif task_route == "line3_bbox":
                    spatial_relations_text = generate_bbox_relations(scene_graph)
                
                if spatial_relations_text.strip() == "Not enough objects to determine relative positioning.":
                    spatial_relations_text = ""

            # ================= 阶段 3.8：外部先验知识注入 =================
            if ENABLE_EXTERNAL_KNOWLEDGE:
                external_knowledge_text = knowledge_module.inject_knowledge(scene_graph, category, sub_category, item['Question'])
                if external_knowledge_text:
                    spatial_relations_text += "\n\n--- EXTERNAL DOMAIN KNOWLEDGE ---\n" + external_knowledge_text

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
    print(f"\n=== Variant B 跑分结束 ===")
    print(f"有效任务数: {total_run_count} | 正确数: {correct_count} | 准确率: {accuracy:.2f}%")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()