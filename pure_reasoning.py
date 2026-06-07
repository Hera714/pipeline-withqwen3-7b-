# coding=utf-8
import os

# ================= 核心修改：重定向模型下载路径到数据盘 =================
# 注意：这行代码必须在导入 modelscope 之前执行！
os.environ['MODELSCOPE_CACHE'] = '/root/autodl-tmp/modelscope_cache'

import json
import re
import torch
import gc
import traceback
from PIL import Image
from tqdm import tqdm
from modelscope import AutoModelForCausalLM

# ================= 配置路径与测试控制 =================
MODEL_PATH = "AIDC-AI/Ovis2.5-9B" 
JSON_PATH = "/root/autodl-fs/data.json"

OUTPUT_EVAL_JSON = "ovis_pure_eval_full.json" 
IMAGE_DIR = "/root/autodl-fs/img"

TARGET_CATEGORY = "" # 留空则跑全集，或者填入 "Relative Positioning" 等大类

def extract_answer(response):
    if not response: return "Unknown"
    # 优先匹配 <answer> 标签
    match = re.search(r'<answer>\s*([1-4])\s*</answer>', response)
    if match: return match.group(1)
    # 兜底匹配任意数字 1-4
    match = re.search(r'[1-4]', response)
    if match: return match.group(0)
    return "Unknown"

def generate_ovis_response(model, image, question_text, options_text):
    enable_thinking = True
    enable_thinking_budget = True
    max_tokens = 2048
    thinking_budget = 1024 # 给纯视觉模型留足思考空间

    system_prompt = (
        "You are an expert spatial reasoning assistant. Carefully analyze the image to answer the multiple-choice question.\n"
        "Output a <think> block for step-by-step reasoning, then your exact answer as a single number (1-4) inside <answer> tags."
    )
    
    user_prompt = f"Question:\n{question_text}\nOptions:\n{options_text}\n\nAnswer (1-4):"
    
    messages = [{
        "role": "user", 
        "content": [
            {"type": "image", "image": image}, 
            {"type": "text", "text": system_prompt + "\n\n" + user_prompt}
        ]
    }]

    input_ids, pixel_values, grid_thws = model.preprocess_inputs(
        messages=messages,
        add_generation_prompt=True,
        enable_thinking=enable_thinking
    )
    
    input_ids = input_ids.cuda()
    pixel_values = pixel_values.cuda() if pixel_values is not None else None
    grid_thws = grid_thws.cuda() if grid_thws is not None else None

    with torch.no_grad():
        outputs = model.generate(
            inputs=input_ids,
            pixel_values=pixel_values,
            grid_thws=grid_thws,
            enable_thinking=enable_thinking,
            enable_thinking_budget=enable_thinking_budget,
            max_new_tokens=max_tokens,
            thinking_budget=thinking_budget,
        )

    output_text = model.text_tokenizer.decode(outputs[0], skip_special_tokens=True)

    del input_ids, pixel_values, grid_thws, outputs
    torch.cuda.empty_cache()
    
    return output_text

def main():
    print(f"Loading {MODEL_PATH} for pure baseline evaluation...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).cuda()
    model.eval()

    if os.path.exists(OUTPUT_EVAL_JSON):
        print(f"发现已有进度文件 {OUTPUT_EVAL_JSON}，正在恢复...")
        with open(OUTPUT_EVAL_JSON, "r", encoding="utf-8") as f: 
            dataset = json.load(f)
    else:
        print(f"加载全新数据集 {JSON_PATH}...")
        with open(JSON_PATH, "r", encoding="utf-8") as f: 
            dataset = json.load(f)

    correct_count = 0
    total_run_count = 0
    
    print(f"开始 Pure Baseline 评估。当前锁定测试大类: [{TARGET_CATEGORY if TARGET_CATEGORY else '全部'}]")
    
    for item in tqdm(dataset):
        category = item.get("Category", "").strip()
        
        if TARGET_CATEGORY and category != TARGET_CATEGORY:
            continue
            
        total_run_count += 1
        
        # 检查是否已经跑过该数据
        if "Baseline_Prediction" in item and item["Baseline_Prediction"] not in ["", "Error"]: 
            if item.get("Baseline_Is_Correct"):
                correct_count += 1
            continue

        try:
            image_path = os.path.join(IMAGE_DIR, item.get("Image_Filename"))
            if not os.path.exists(image_path):
                item["Baseline_Prediction"], item["Baseline_Raw_Response"], item["Baseline_Is_Correct"] = "Missing", "Image File Missing", False
                continue

            image = Image.open(image_path).convert("RGB")
            options_text = f"1. {item.get('Option_1', '')}\n2. {item.get('Option_2', '')}\n3. {item.get('Option_3', '')}\n4. {item.get('Option_4', '')}"

            # 直接喂给模型进行推理
            response = generate_ovis_response(model, image, item['Question'], options_text)
            prediction = extract_answer(response)

            # 提取思维链
            think_content = ""
            if "<think>" in response and "</think>" in response:
                think_content = response.split("<think>")[1].split("</think>")[0].strip()

            # 为了不覆盖 3D pipeline 的结果，这里的字段加上了 Baseline_ 前缀
            item["Baseline_Think"] = think_content
            item["Baseline_Raw_Response"] = response
            item["Baseline_Prediction"] = prediction
            item["Baseline_Is_Correct"] = (prediction == str(item["Answer"]))
            
            if item["Baseline_Is_Correct"]: 
                correct_count += 1

        except Exception as e:
            error_msg = f"Runtime Error: {str(e)}"
            item["Baseline_Prediction"] = "Error"
            item["Baseline_Raw_Response"] = error_msg + "\n" + traceback.format_exc()
            item["Baseline_Is_Correct"] = False
            gc.collect(); torch.cuda.empty_cache()

        finally:
            # 增量保存进度
            with open(OUTPUT_EVAL_JSON, "w", encoding="utf-8") as f: 
                json.dump(dataset, f, ensure_ascii=False, indent=4)

    accuracy = (correct_count / total_run_count if total_run_count > 0 else 0) * 100
    print(f"\n=== 本次 Pure Baseline [{TARGET_CATEGORY if TARGET_CATEGORY else '全部'}] 测试结束 ===")
    print(f"有效任务数: {total_run_count} | 正确数: {correct_count} | 准确率: {accuracy:.2f}%")

if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()