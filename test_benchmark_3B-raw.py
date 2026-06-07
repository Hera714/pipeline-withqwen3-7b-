# coding=utf-8
import os
os.environ['MODELSCOPE_CACHE'] = '/root/autodl-tmp/modelscope_cache'

import json
import re
import torch
from tqdm import tqdm
from modelscope import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# ================= 配置路径 =================
MODEL_PATH = "Qwen/Qwen2.5-VL-3B-Instruct"
JSON_PATH = "/root/autodl-fs/data.json"
IMAGE_DIR = "/root/autodl-fs/img"
OUTPUT_JSON = "/root/autodl-fs/qwen2_5_vl_3b_raw_eval.json"  # 换了个新名字防止覆盖之前的测试

def extract_answer(response):
    """提取答案标签 1-4"""
    if not response: return "Unknown"
    match = re.search(r'<answer>\s*([1-4])\s*</answer>', response)
    if match: return match.group(1)
    match = re.search(r'[1-4]', response)
    if match: return match.group(0)
    return "Unknown"

def main():
    print("Loading Qwen2.5-VL-7B-Instruct...")
    # 使用 bfloat16 节省显存，并自动分配设备
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        MODEL_PATH, torch_dtype=torch.bfloat16, device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(MODEL_PATH)

    # 1. 加载数据集 (支持断点续跑)
    if os.path.exists(OUTPUT_JSON):
        print(f"发现已有进度文件 {OUTPUT_JSON}，正在恢复...")
        with open(OUTPUT_JSON, "r", encoding="utf-8") as f:
            dataset = json.load(f)
    else:
        print(f"加载全新数据集 {JSON_PATH}...")
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            dataset = json.load(f)

    correct_count = 0
    total_run_count = 0

    # 2. 开始评测
    for item in tqdm(dataset, desc="Evaluating"):
        # 跳过已经正确跑完的样本（断点续跑逻辑）
        if "Model_Prediction" in item and item["Model_Prediction"] != "Error":
            total_run_count += 1
            if item.get("Is_Correct", False):
                correct_count += 1
            continue

        try:
            # 组装图片路径与选项文本
            image_path = f"file://{os.path.abspath(os.path.join(IMAGE_DIR, item['Image_Filename']))}"
            options_text = f"1. {item.get('Option_1', '')}\n2. {item.get('Option_2', '')}\n3. {item.get('Option_3', '')}\n4. {item.get('Option_4', '')}"
            prompt_text = f"Question:\n{item['Question']}\nOptions:\n{options_text}\nAnswer the question directly based on the image. Output your exact answer as a single number (1-4) inside <answer> tags."

            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image_path},
                    {"type": "text", "text": prompt_text}
                ]
            }]

            # 3. 数据预处理
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
            ).to("cuda")

            # 4. 模型推理 (不开启思考模式)
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=128)

            # 截断 Prompt，仅获取新生成的 Tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            response = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0]

            # 5. 记录与统计
            prediction = extract_answer(response)
            item["Model_Raw_Response"] = response
            item["Model_Prediction"] = prediction
            item["Is_Correct"] = (prediction == str(item["Answer"]))

            total_run_count += 1
            if item["Is_Correct"]:
                correct_count += 1

        except Exception as e:
            print(f"\n[错误] Task ID {item.get('id')} 发生异常: {e}")
            item["Model_Prediction"] = "Error"
            item["Model_Raw_Response"] = str(e)
            item["Is_Correct"] = False

        finally:
            # 每跑完一条保存一次，防止意外中断
            with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
                json.dump(dataset, f, ensure_ascii=False, indent=4)

    # 6. 打印结果
    accuracy = (correct_count / total_run_count if total_run_count > 0 else 0) * 100
    print("\n=== 测试结束 ===")
    print(f"总计测试: {total_run_count} | 答对: {correct_count} | 准确率: {accuracy:.2f}%")
    print(f"结果已保存至: {OUTPUT_JSON}")

if __name__ == "__main__":
    main()