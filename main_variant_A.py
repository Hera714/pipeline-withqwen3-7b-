# coding=utf-8
# main_variant_A.py
import os
os.environ['MODELSCOPE_CACHE'] = '/root/autodl-tmp/modelscope_cache'

import json
import re
import torch
import gc
import traceback
from PIL import Image, ImageDraw
from tqdm import tqdm

# 引入 Variant A 专属的视觉感知模块
from perception_ovis_3d_variant_A import Ovis3DPerceptionModuleVariantA 
from depth_estimation_3d import DepthPro3DModule
# 【核心修改点】：完全注释掉符号文本引擎的引入
# from spatial_relations_generator import generate_spatial_relations
# from spatial_relations_bbox import generate_bbox_relations

# ================= 专项测试控制 (Targeted Testing) =================
TARGET_TASK_ID = ""        
TARGET_CATEGORY = ""       
TARGET_SUB_CATEGORY = ""   

# ================= 硬路由字典配置 (Oracle Routing) 保持原样 =================
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
    "3D Geometry":["Spatial Containment","Shape Projection"],
    "Depth & Occlusion":["Complete Occlusion Inference","Layering Order","Partial Occlusion"],
    "Relative Positioning": ["Corner/Angle Positioning","Directional Relations","Proximity Gradients"],
}

LINE3_CONFIG = {
    "Relative Positioning": ["Alignment Patterns"],
    "Size & Scale":["*"]
}

def get_task_route(category, sub_category):
    if category in LINE0_CONFIG and (sub_category in LINE0_CONFIG[category] or "*" in LINE0_CONFIG[category]):
        return "line0_pure_vlm"
    if category in LINE2_CONFIG and (sub_category in LINE2_CONFIG[category] or "*" in LINE2_CONFIG[category]):
        return "line2_orientation"
    if category in LINE3_CONFIG and (sub_category in LINE3_CONFIG[category] or "*" in LINE3_CONFIG[category]):
        return "line3_bbox"
    return "line1_base"

# ================= 配置路径与测试控制 =================
MODEL_PATH = "AIDC-AI/Ovis2.5-9B" 
JSON_PATH = "/root/autodl-fs/data.json"

START_IDX = 0      
END_IDX = 100 

# 【变体 A 专属文件命名】
OUTPUT_EVAL_JSON = f"ovis_3d_eval_variant_A_{START_IDX}_{END_IDX}.json" 
OUTPUT_SG_JSON = f"ovis_3d_scene_graphs_variant_A_{START_IDX}_{END_IDX}.json"

IMAGE_DIR = "/root/autodl-fs/img"
OUT_POINTS_DIR = "/root/autodl-fs/img_points_VariantA"
os.makedirs(OUT_POINTS_DIR, exist_ok=True)

def extract_answer(response):
    if not response: return "Unknown"
    match = re.search(r'<answer>\s*([1-4])\s*</answer>', response)
    if match: return match.group(1)
    match = re.search(r'[1-4]', response)
    if match: return match.group(0)
    return "Unknown"

def main():
    vision_module = Ovis3DPerceptionModuleVariantA(model_path=MODEL_PATH)
    depth_module = DepthPro3DModule()

    if os.path.exists(OUTPUT_EVAL_JSON):
        print(f"发现已有进度文件 {OUTPUT_EVAL_JSON}，正在恢复...")
        with open(OUTPUT_EVAL_JSON, "r", encoding="utf-8") as f: 
            dataset = json.load(f)
    else:
        print(f"加载全新数据集 {JSON_PATH}...")
        with open(JSON_PATH, "r", encoding="utf-8") as f: 
            dataset = json.load(f)

    if os.path.exists(OUTPUT_SG_JSON):
        with open(OUTPUT_SG_JSON, "r", encoding="utf-8") as f: 
            scene_graphs_dict = json.load(f)
    else: 
        scene_graphs_dict = {}

    correct_count = 0
    total_run_count = 0
    
    target_msg = []
    if TARGET_TASK_ID: target_msg.append(f"TaskID={TARGET_TASK_ID}")
    if TARGET_CATEGORY: target_msg.append(f"Category={TARGET_CATEGORY}")
    if TARGET_SUB_CATEGORY: target_msg.append(f"SubCategory={TARGET_SUB_CATEGORY}")
    target_str = " | ".join(target_msg) if target_msg else "全量测试"
    
    print(f"[Variant A - 裸坐标] 开始评估。当前锁定范围: [{target_str}] (测试前 {END_IDX} 个样本)")
    
    for idx, item in enumerate(tqdm(dataset)):
        if idx < START_IDX: continue
        if END_IDX is not None and idx >= END_IDX: break
            
        task_id = str(item.get("id"))
        category = item.get("Category", "").strip()
        sub_category = item.get("Sub_Category", "").strip()
        
        if TARGET_TASK_ID and task_id != str(TARGET_TASK_ID): continue
        if TARGET_CATEGORY and category != TARGET_CATEGORY: continue
        if TARGET_SUB_CATEGORY and sub_category != TARGET_SUB_CATEGORY: continue
            
        total_run_count += 1

        try:
            image_path = os.path.join(IMAGE_DIR, item.get("Image_Filename"))
            image = Image.open(image_path).convert("RGB")
            options_text = f"1. {item.get('Option_1', '')}\n2. {item.get('Option_2', '')}\n3. {item.get('Option_3', '')}\n4. {item.get('Option_4', '')}"
            
            task_route = get_task_route(category, sub_category)

            # ================= Line 0: 纯原生推理 (Pure VLM) 短路拦截 =================
            if task_route == "line0_pure_vlm":
                system_prompt = "You are a visual AI. Answer the question directly based on the image.\nOutput a <think> block for step-by-step reasoning, then your exact answer as a single number (1-4) inside <answer> tags."
                messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": system_prompt + f"\nQuestion:\n{item['Question']}\nOptions:\n{options_text}\nAnswer (1-4):"}]}]
                
                response = vision_module._generate_response(messages, max_tokens=2048)
                prediction = extract_answer(response)
                
                think_content = ""
                if "<think>" in response and "</think>" in response:
                    think_content = response.split("<think>")[1].split("</think>")[0].strip()

                item["Model_Think"] = think_content
                item["Model_Raw_Response"] = response
                item["Model_Prediction"] = prediction
                item["Is_Correct"] = (prediction == str(item["Answer"]))
                
                scene_graphs_dict[task_id] = {"task_route_used": task_route, "objects": [], "spatial_relations_text": "Bypassed via Pure VLM"}
                if item["Is_Correct"]: correct_count += 1
                continue 

            # ================= 阶段 1：感知提取 =================
            target_perspective, objects_data = vision_module.extract_objects_3d(
                image, item['Question'], options_text, task_route=task_route
            )
            gc.collect(); torch.cuda.empty_cache()

            # ================= 阶段 2：深度估计 =================
            depth_matrix, focal_length_px = depth_module.estimate_depth(image)
            gc.collect(); torch.cuda.empty_cache()

            # ================= 阶段 3：3D 反投影与特征组合 =================
            scene_graph = []
            for obj in objects_data:
                pos_3d = depth_module.get_3d_coordinates(depth_matrix, obj["point_2d"], focal_length_px)
                
                sg_item = {
                    "name": obj.get("name", "Unknown"),
                    "point_2d": obj["point_2d"],
                    "global_3d_position": pos_3d,
                    "reason": obj.get("reason", "")
                }

                if task_route == "line3_bbox":
                    sg_item["bbox_2d"] = obj.get("bbox_2d")
                    if "bbox_2d" in obj:
                        dims = depth_module.get_3d_bbox_dimensions(
                            depth_matrix, obj["bbox_2d"], obj["point_2d"], focal_length_px
                        )
                        if dims: sg_item.update(dims)
                elif task_route == "line2_orientation":
                    sg_item["anchor_parts"] = obj.get("anchor_parts", "")
                    sg_item["visible_profile"] = obj.get("visible_profile", "unknown")
                    sg_item["2d_pointing_direction"] = obj.get("2d_pointing_direction", "unknown")
                elif task_route == "line1_base":
                    sg_item["facing_direction"] = obj.get("facing_direction", "Unknown")
                    
                scene_graph.append(sg_item)

            # 绘制调试图
            img_with_points = image.copy()
            if scene_graph:
                draw = ImageDraw.Draw(img_with_points)
                for sg_item in scene_graph:
                    x, y = sg_item["point_2d"]
                    draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill="red", outline="white")
                    draw.text((x + 8, y - 10), sg_item['name'], fill="red")
                    
                    if "bbox_2d" in sg_item and sg_item["bbox_2d"]:
                        bx1, by1, bx2, by2 = sg_item["bbox_2d"]
                        draw.rectangle([bx1, by1, bx2, by2], outline="green", width=2)
                        
            img_with_points.save(os.path.join(OUT_POINTS_DIR, f"task_{task_id}_points.png"))

            # ================= 阶段 3.5：双路硬路由 (符号引擎强制下线) =================
            spatial_relations_text = "Variant A: Symbolic engine disabled. Naked features provided only."

            scene_graphs_dict[task_id] = {
                "task_route_used": task_route,
                "target_perspective": target_perspective,
                "objects": scene_graph,
                "spatial_relations_text": spatial_relations_text
            }

            # ================= 阶段 4：纯净推理 (使用变体A专属方法) =================
            response = vision_module.perform_final_reasoning(
                image, 
                item['Question'], 
                options_text, 
                scene_graph,
                task_route=task_route  # 注意：这里不再传入 spatial_relations_text
            )
            prediction = extract_answer(response)

            think_content = ""
            if "<think>" in response and "</think>" in response:
                think_content = response.split("<think>")[1].split("</think>")[0].strip()

            item["Model_Raw_Response"] = response
            item["Model_Prediction"] = prediction
            item["Is_Correct"] = (prediction == str(item["Answer"]))
            
            if item["Is_Correct"]: 
                correct_count += 1

        except Exception as e:
            print(f"\n[严重崩溃!] 任务 {task_id} 在运行中发生异常: {str(e)}")
            item["Model_Prediction"] = "Error"
            item["Model_Raw_Response"] = f"Runtime Error: {str(e)}\n{traceback.format_exc()}"
            item["Is_Correct"] = False
            scene_graphs_dict[task_id] = {"task_route_used": "Error", "objects": []}
            gc.collect(); torch.cuda.empty_cache()

        finally:
            with open(OUTPUT_EVAL_JSON, "w", encoding="utf-8") as f: 
                json.dump(dataset, f, ensure_ascii=False, indent=4)
            with open(OUTPUT_SG_JSON, "w", encoding="utf-8") as f: 
                json.dump(scene_graphs_dict, f, ensure_ascii=False, indent=4)

    accuracy = (correct_count / total_run_count if total_run_count > 0 else 0) * 100
    print(f"\n=== Variant A - 裸坐标 [{TARGET_CATEGORY if TARGET_CATEGORY else '全部'}] 测试结束 ===")
    print(f"有效任务数: {total_run_count} | 正确数: {correct_count} | 准确率: {accuracy:.2f}%")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()