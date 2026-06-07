# coding=utf-8
# perception_ovis_3d_variant_A.py
import os
os.environ['MODELSCOPE_CACHE'] = '/root/autodl-tmp/modelscope_cache'

import json
import re
import torch
from PIL import Image
from modelscope import AutoModelForCausalLM

class Ovis3DPerceptionModuleVariantA:
    def __init__(self, model_path="AIDC-AI/Ovis2.5-9B"):
        print(f"[Variant A] Loading {model_path} 3D Module on vGPU (Full Precision bf16)...")
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2"
        ).cuda()
        self.model.eval()

        # ================= Prompts (保持原样，继续提取细粒度特征) =================
        self.prompt_line1_base = (
            "Analyze the image and the question. You MUST EXHAUSTIVELY extract ALL key objects that are explicitly mentioned or implicitly needed to answer the question. Missing any relevant reference object will cause downstream failures.\n"
            "OUTPUT FORMAT: Return a JSON object with strictly ONE key:\n"
            "1. 'objects': A JSON array containing EVERY relevant object. Do not omit any. Each object MUST have:\n"
            "   - 'name': Object name.\n"
            "   - 'point_2d': [x, y] normalized coords (0.0 to 1.0) strictly INSIDE the object.\n"
            "   - 'facing_direction': The object's 3D orientation (e.g., 'facing camera', 'facing left', 'facing red car', 'N/A').\n"
            "   - 'reason': Why it's extracted.\n"
            "Respond ONLY with valid JSON.\n\n"
        )

        self.prompt_line2_orientation = (
            "Analyze the image and the question. Determine the required perspective. You MUST EXHAUSTIVELY extract ALL key objects involved in the question. Include every reference object, target object, and landmark mentioned.\n"
            "OUTPUT FORMAT: Return a JSON object with strictly two keys:\n"
            "1. 'target_perspective': If asking from a specific object's viewpoint, output its EXACT 'name'. Otherwise, output 'camera'.\n"
            "2. 'objects': A JSON array containing EVERY involved object. Do not omit any. Each object MUST have:\n"
            "   - 'name': Object name.\n"
            "   - 'point_2d': [x, y] normalized coords (0.0 to 1.0). CRITICAL: This point MUST be located on a solid, unoccluded, and fully visible surface of the object. Do not place it on transparent parts, edges, or background.\n"
            "   - 'anchor_parts': What direction-defining parts are visible? (e.g., 'grille and left doors', 'only flat casing', 'both eyes').\n"
            "   - 'visible_profile': Determine its geometric pose. Strictly choose from: 'frontal', 'side_profile', 'back', 'diagonal_front', 'diagonal_back', or 'unknown'.\n"
            "   - '2d_pointing_direction': Where its front/opening points IN THE 2D IMAGE. Strictly choose from: 'left', 'right', 'camera', 'away', 'up', 'down', or 'unknown'.\n"
            "   - 'reason': Why it is extracted.\n"
            "Respond ONLY with valid JSON.\n\n"
        )

        self.prompt_line3_bbox = (
            "Analyze the image and the question. You MUST EXHAUSTIVELY identify and extract ALL objects required to answer the spatial or relative positioning question. Every reference point, subject, and target mentioned or logically implied MUST be included.\n"
            "OUTPUT FORMAT: Return a JSON object with strictly ONE key:\n"
            "1. 'objects': A JSON array containing EVERY relevant object. Do not omit any. Each object MUST have:\n"
            "   - 'name': Object name (with sequential number if multiple exist).\n"
            "   - 'bbox_2d': [x_min, y_min, x_max, y_max] normalized bounding box (0.0 to 1.0).\n"
            "   - 'point_2d': [x, y] normalized coords (0.0 to 1.0).\n"
            "   - 'reason': Why it is extracted.\n"
            "Respond ONLY with valid JSON.\n\n"
        )

    def _denormalize_point(self, point_norm, img_w, img_h):
        x = min(max(0, int(round(float(point_norm[0]) * img_w))), img_w - 1)
        y = min(max(0, int(round(float(point_norm[1]) * img_h))), img_h - 1)
        return [x, y]

    def _denormalize_bbox(self, bbox_norm, img_w, img_h):
        return [
            min(max(0, int(round(float(bbox_norm[0]) * img_w))), img_w - 1),
            min(max(0, int(round(float(bbox_norm[1]) * img_h))), img_h - 1),
            min(max(0, int(round(float(bbox_norm[2]) * img_w))), img_w - 1),
            min(max(0, int(round(float(bbox_norm[3]) * img_h))), img_h - 1)
        ]

    def _generate_response(self, messages, max_tokens):
        enable_thinking = True
        enable_thinking_budget = True
        thinking_budget = max(256, max_tokens - 128)

        input_ids, pixel_values, grid_thws = self.model.preprocess_inputs(
            messages=messages, add_generation_prompt=True, enable_thinking=enable_thinking
        )
        
        input_ids = input_ids.cuda()
        pixel_values = pixel_values.cuda() if pixel_values is not None else None
        grid_thws = grid_thws.cuda() if grid_thws is not None else None

        with torch.no_grad():
            outputs = self.model.generate(
                inputs=input_ids, 
                pixel_values=pixel_values, 
                grid_thws=grid_thws,
                enable_thinking=enable_thinking, 
                enable_thinking_budget=enable_thinking_budget,
                max_new_tokens=max_tokens, 
                thinking_budget=thinking_budget,
            )

        input_len = input_ids.shape[1]
        if len(outputs[0]) > input_len and torch.equal(outputs[0][:input_len], input_ids[0]):
            generated_tokens = outputs[0][input_len:]
        else:
            generated_tokens = outputs[0]
            
        output_text = self.model.text_tokenizer.decode(generated_tokens, skip_special_tokens=True)

        del input_ids, pixel_values, grid_thws, outputs
        torch.cuda.empty_cache()
        return output_text

    def extract_objects_3d(self, image: Image.Image, question_text: str, options_text: str, task_route: str = "line1_base"):
        if task_route == "line2_orientation": system_prompt = self.prompt_line2_orientation
        elif task_route == "line3_bbox": system_prompt = self.prompt_line3_bbox
        else: system_prompt = self.prompt_line1_base

        user_prompt = f"Question:\n{question_text}\nOptions:\n{options_text}\n\nOutput JSON:"
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": system_prompt + "\n\n" + user_prompt}]}]

        response_text = self._generate_response(messages, max_tokens=1024)

        try:
            clean_text = response_text.strip()
            if "</think>" in clean_text:
                clean_text = clean_text.split("</think>")[-1].strip()
            
            json_match = re.search(r'\{.*\}', clean_text, re.DOTALL)
            if json_match:
                clean_text = json_match.group(0)
                
            parsed_data = json.loads(clean_text)

            target_perspective = parsed_data.get("target_perspective", "camera") if task_route == "line2_orientation" else "camera"
            objects_list = parsed_data.get("objects", [])

            cleaned_data = []
            for item in objects_list:
                if not isinstance(item, dict): continue
                point = item.get("point_2d") or item.get("point")
                
                if task_route == "line3_bbox":
                    bbox = item.get("bbox_2d")
                    if bbox and len(bbox) == 4 and point and len(point) == 2:
                        cleaned_data.append({
                            "name": item.get("name", "Unknown"),
                            "bbox_2d": self._denormalize_bbox(bbox, image.width, image.height),
                            "point_2d": self._denormalize_point(point, image.width, image.height),
                            "reason": item.get("reason", "")
                        })
                elif task_route == "line2_orientation":
                    if point and len(point) == 2:
                        cleaned_data.append({
                            "name": item.get("name", "Unknown"),
                            "point_2d": self._denormalize_point(point, image.width, image.height),
                            "anchor_parts": item.get("anchor_parts", ""),
                            "visible_profile": item.get("visible_profile", "unknown"),
                            "2d_pointing_direction": item.get("2d_pointing_direction", "unknown"),
                            "reason": item.get("reason", "")
                        })
                else: 
                    if point and len(point) == 2:
                        cleaned_data.append({
                            "name": item.get("name", "Unknown"),
                            "point_2d": self._denormalize_point(point, image.width, image.height),
                            "facing_direction": item.get("facing_direction", "Unknown"),
                            "reason": item.get("reason", "")
                        })
            return target_perspective, cleaned_data
        except Exception as e:
            print(f"[ERROR] JSON 解析失败: {e}\nRaw Output: {response_text[:200]}")
            return "camera", []

    def perform_final_reasoning(self, image: Image.Image, question_text: str, options_text: str,
                                scene_graph_data: list, task_route: str = "line1_base"):
        # 【核心修改点】：强行打印裸数据，丢弃 spatial_relations_text
        context = "--- Auxiliary 3D Scene Graph (Naked Coordinates & Features) ---\n"
        context += "* Camera is at (0, 0, 0). Coordinates and dimensions are in meters.\n"
        context += "* +X is Right, +Y is Down, +Z is Forward (Depth).\n\n"

        for obj in scene_graph_data:
            pos = obj.get("global_3d_position", {})
            context += f"- Object: '{obj.get('name')}'\n"
            context += f"  Location: X={pos.get('X_meters', 'N/A')}, Y={pos.get('Y_meters', 'N/A')}, Z={pos.get('Z_meters', 'N/A')}\n"
            
            # 暴露变体A所需的 BBox 尺寸和 Pose 裸数据
            if task_route == "line3_bbox":
                context += f"  Physical Size: Height={obj.get('Physical_Height', 'N/A')}m, Width={obj.get('Physical_Width', 'N/A')}m\n"
                context += f"  Base Level (Bottom Y): {obj.get('Bottom_Y_meters', 'N/A')}m\n"
            elif task_route == "line2_orientation":
                context += f"  Pose Profile: {obj.get('visible_profile', 'unknown')}\n"
                context += f"  2D Pointing Direction: {obj.get('2d_pointing_direction', 'unknown')}\n"
            else:
                context += f"  Orientation: {obj.get('facing_direction', 'Unknown')}\n"
        context += "--------------------------------\n"

        # 【核心修改点】：修改 System Prompt，命令它自己计算
        reasoning_system_prompt = (
            "You are an expert spatial reasoning assistant. Use the image and the provided 'Auxiliary 3D Scene Graph' to answer the question.\n"
            "CRITICAL RULES:\n"
            "- You are provided with RAW numerical coordinates (X, Y, Z in meters), physical dimensions, and geometric profiles.\n"
            "- You MUST calculate and deduce the spatial relationships (e.g., occlusion, alignment, relative sizes, perpendicularity) YOURSELF based strictly on these numbers.\n"
            "- Output a <think> block for step-by-step mathematical and spatial reasoning, then your exact answer as a single number (1-4) inside <answer> tags."
        )

        user_prompt = f"{context}\nQuestion:\n{question_text}\nOptions:\n{options_text}\n\nAnswer (1-4):"
        messages = [
            {"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": reasoning_system_prompt + "\n\n" + user_prompt}]}
        ]

        return self._generate_response(messages, max_tokens=2048)