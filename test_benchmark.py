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

# ================= 1. 测试区间与模块开关 (可选项) =================
START_IDX = 0        # 起始索引 (从 0 开始)
END_IDX = None       # 截止索引 (设为 None 则跑完后续所有数据)
USE_COMMON_DIMENSIONS = False  # 是否启用通用实体尺寸字典 (True/False)

# ================= 2. 专项测试控制 (Targeted Testing) =================
TARGET_TASK_ID = []
TARGET_CATEGORY = []
TARGET_SUB_CATEGORY = []

# ================= 3. 硬路由字典配置 (Oracle Routing) =================

# 🔴 Line 0: 模型原始推理 (Pure VLM)
LINE0_CONFIG = {
    "3D Geometry": ["Volume Comparison"], 
    "Depth & Occlusion": ["Reflective Surfaces"],
    "Orientation":["Cardinal Direction"],
    "Relative Positioning":["Betweenness Relationships"],
    "Size & Scale":["Scale Consistency"],
    "Spatial Navigation":["Accessibility Constraints","Pathway Existence"]
}

# 🔵 Line 2: 几何与朝向推导
LINE2_CONFIG = {
    "Orientation": ["Facing Direction","Object Rotation","Stacking Orientation","Tool Handedness"],
    "3D Geometry":["Spatial Containment","Shape Projection","Gravity Effects"],
    "Depth & Occlusion":["Complete Occlusion Inference","Layering Order","Partial Occlusion"],
    "Relative Positioning": ["Corner/Angle Positioning","Directional Relations","Proximity Gradients"],
}

# 🟣 Line 3: 边界框与物理尺寸
LINE3_CONFIG = {
    "Relative Positioning": ["Alignment Patterns"],
    "Size & Scale":["*"]
}

def get_task_route(category, sub_category):
    """基于元数据的绝对硬路由 (Oracle Routing)"""
    if category in LINE0_CONFIG and (sub_category in LINE0_CONFIG[category] or "*" in LINE0_CONFIG[category]):
        return "line0_pure_vlm"
    if category in LINE2_CONFIG and (sub_category in LINE2_CONFIG[category] or "*" in LINE2_CONFIG[category]):
        return "line2_orientation"
    if category in LINE3_CONFIG and (sub_category in LINE3_CONFIG[category] or "*" in LINE3_CONFIG[category]):
        return "line3_bbox"
    return "line1_base"

# ================= 4. 配置路径 =================
MODEL_PATH = "AIDC-AI/Ovis2.5-9B" 
JSON_PATH = "/root/autodl-fs/data.json"

OUTPUT_EVAL_JSON = f"1.json" 
OUTPUT_SG_JSON = f"1scene.json"

IMAGE_DIR = "/root/autodl-fs/img"
OUT_POINTS_DIR = "/root/autodl-fs/img_points_4Linenew"
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

    # 初始化知识注入器，受全局开关控制
    knowledge_module = GeometryKnowledgeInjector(
        kb_dir="/root/autodl-tmp/knowledge_bases", 
        use_common_dimensions=USE_COMMON_DIMENSIONS
    )

    if os.path.exists(OUTPUT_EVAL_JSON):
        print(f"发现已有进度文件 {OUTPUT_EVAL_JSON}，正在恢复...")
        with open(OUTPUT_EVAL_JSON, "r", encoding="utf-8") as f: 
            full_dataset = json.load(f)
    else:
        print(f"加载全新数据集 {JSON_PATH}...")
        with open(JSON_PATH, "r", encoding="utf-8") as f: 
            full_dataset = json.load(f)

    if os.path.exists(OUTPUT_SG_JSON):
        with open(OUTPUT_SG_JSON, "r", encoding="utf-8") as f: 
            scene_graphs_dict = json.load(f)
    else: 
        scene_graphs_dict = {}

    correct_count = 0
    total_run_count = 0

    # 🟢 升级：将配置统一转换为有效列表
    target_task_ids = [TARGET_TASK_ID] if isinstance(TARGET_TASK_ID, str) else TARGET_TASK_ID
    target_task_ids = [str(tid) for tid in target_task_ids if tid]
    
    target_cats = [TARGET_CATEGORY] if isinstance(TARGET_CATEGORY, str) else TARGET_CATEGORY
    target_cats = [c for c in target_cats if c] # 过滤掉空字符串
    
    target_subcats = [TARGET_SUB_CATEGORY] if isinstance(TARGET_SUB_CATEGORY, str) else TARGET_SUB_CATEGORY
    target_subcats = [sc for sc in target_subcats if sc]
    
    target_msg = []
    if TARGET_TASK_ID: target_msg.append(f"TaskID={TARGET_TASK_ID}")
    if TARGET_CATEGORY: target_msg.append(f"Category={TARGET_CATEGORY}")
    if TARGET_SUB_CATEGORY: target_msg.append(f"SubCategory={TARGET_SUB_CATEGORY}")
    target_str = " | ".join(target_msg) if target_msg else "全量测试"
    
    # 动态切片，进度条精准显示
    end_val = END_IDX if END_IDX is not None else len(full_dataset)
    test_subset = full_dataset[START_IDX:end_val]
    
    print(f"开始 3D-Grounded 评估。当前锁定范围: [{target_str}] (测试区间: {START_IDX} -> {end_val})")
    
    for item in tqdm(test_subset):
        task_id = str(item.get("id"))
        category = item.get("Category", "").strip()
        sub_category = item.get("Sub_Category", "").strip()
        
        # 🟢 升级：使用 in 关键字来进行多目标拦截
        if target_task_ids and task_id not in target_task_ids: continue  # <--- 把这行改掉
        if target_cats and category not in target_cats: continue
        if target_subcats and sub_category not in target_subcats: continue
            
        total_run_count += 1

        try:
            image_path = os.path.join(IMAGE_DIR, item.get("Image_Filename"))
            image = Image.open(image_path).convert("RGB")
            options_text = f"1. {item.get('Option_1', '')}\n2. {item.get('Option_2', '')}\n3. {item.get('Option_3', '')}\n4. {item.get('Option_4', '')}"
            
            # 根据字典执行硬路由分发
            task_route = get_task_route(category, sub_category)

            # ================= Line 0: 纯原生推理 (Pure VLM) 短路拦截 =================
            if task_route == "line0_pure_vlm":
                system_prompt = "Answer the question directly based on the image.\nOutput a <think> block for step-by-step reasoning, then your exact answer as a single number (1-4) inside <answer> tags."
                
                # 🟢 修复：为 Line 0 提前提取先验知识 (传入空列表 [] 代替 scene_graph)
                external_knowledge_text = knowledge_module.inject_knowledge(
                    [], 
                    category, 
                    sub_category, 
                    item['Question']
                )
                
                # 组装 Prompt
                prompt_text = system_prompt
                if external_knowledge_text:
                    prompt_text += f"\n\n--- EXTERNAL DOMAIN KNOWLEDGE ---\n{external_knowledge_text}"
                prompt_text += f"\n\nQuestion:\n{item['Question']}\nOptions:\n{options_text}\nAnswer (1-4):"

                messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt_text}]}]
                
                # 直接调用生成方法，开启 thinking 进行原生推理
                response = vision_module._generate_response(messages, max_tokens=2048, enable_thinking=True)
                prediction = extract_answer(response)
                
                think_content = ""
                if "<think>" in response and "</think>" in response:
                    think_content = response.split("<think>")[1].split("</think>")[0].strip()

                # item["Model_Think"] = think_content
                item["Model_Raw_Response"] = response
                item["Model_Prediction"] = prediction
                item["Is_Correct"] = (prediction == str(item["Answer"]))
                
                # 在 Scene Graph JSON 中做好记录，把注入的知识也记下来方便检查
                scene_graphs_dict[task_id] = {
                    "task_route_used": task_route, 
                    "objects": [], 
                    "spatial_relations_text": f"Bypassed via Pure VLM.\n{external_knowledge_text}" 
                }
                
                if item["Is_Correct"]: correct_count += 1
                
                # 执行完毕直接 continue，跳过后续所有 3D 逻辑
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

            # ================= 阶段 3.5：双路硬路由 =================
            spatial_relations_text = ""
            if task_route == "line2_orientation":
                spatial_relations_text = generate_spatial_relations(scene_graph, target_perspective)
            elif task_route == "line3_bbox":
                spatial_relations_text = generate_bbox_relations(scene_graph)
                
            if spatial_relations_text.strip() == "Not enough objects to determine relative positioning.":
                spatial_relations_text = ""

            # 🟢 ================= 阶段 3.8：外部先验知识注入 =================
            external_knowledge_text = knowledge_module.inject_knowledge(
                scene_graph, 
                category, 
                sub_category, 
                item['Question']
            )
            
            if external_knowledge_text:
                spatial_relations_text += "\n\n--- EXTERNAL DOMAIN KNOWLEDGE ---\n"
                spatial_relations_text += external_knowledge_text

            scene_graphs_dict[task_id] = {
                "task_route_used": task_route,
                "target_perspective": target_perspective,
                "objects": scene_graph,
                "spatial_relations_text": spatial_relations_text 
            }

            # ================= 阶段 4：纯净推理 =================
            response = vision_module.perform_final_reasoning(
                image, 
                item['Question'], 
                options_text, 
                scene_graph,
                spatial_relations_text,
                task_route=task_route
            )
            prediction = extract_answer(response)

            think_content = ""
            if "<think>" in response and "</think>" in response:
                think_content = response.split("<think>")[1].split("</think>")[0].strip()

            # item["Model_Think"] = think_content
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
            # 始终保存全量数据集 (full_dataset)，杜绝文件被切断的风险
            with open(OUTPUT_EVAL_JSON, "w", encoding="utf-8") as f: 
                json.dump(full_dataset, f, ensure_ascii=False, indent=4)
            with open(OUTPUT_SG_JSON, "w", encoding="utf-8") as f: 
                json.dump(scene_graphs_dict, f, ensure_ascii=False, indent=4)

    accuracy = (correct_count / total_run_count if total_run_count > 0 else 0) * 100
    print(f"\n=== 本次 [{TARGET_CATEGORY if TARGET_CATEGORY else '全部'}] 测试结束 ===")
    print(f"有效任务数: {total_run_count} | 正确数: {correct_count} | 准确率: {accuracy:.2f}%")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()