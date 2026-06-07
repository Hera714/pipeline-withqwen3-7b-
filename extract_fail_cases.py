# coding=utf-8
import json
import os

# ================= 配置区 =================
INPUT_JSON = "Size&Scale(Knowledge)/SizeScale.json"   # 替换为你实际跑完的完整结果文件路径
TARGET_CATEGORY = "Size & Scale"          # 必填：你想要提取的大类名称 (如 "Orientation", "3D Geometry" 等)
TARGET_SUB_CATEGORY = "Scale Consistency"                 # 选填：你想要提取的小类名称 (如 "Spatial Containment")。留空 "" 则提取整个大类
# ==========================================

# 自动生成友好的输出文件名 (把空格替换为下划线)
safe_category_name = TARGET_CATEGORY.replace(" ", "_").replace("&", "and")
if TARGET_SUB_CATEGORY:
    safe_sub_name = TARGET_SUB_CATEGORY.replace(" ", "_").replace("&", "and")
    OUTPUT_JSON = f"failed_cases_{safe_category_name}_{safe_sub_name}.json"
else:
    OUTPUT_JSON = f"failed_cases_{safe_category_name}.json"

def main():
    if not os.path.exists(INPUT_JSON):
        print(f"[错误] 找不到输入文件: {INPUT_JSON}")
        return

    print(f"正在加载结果文件: {INPUT_JSON}...")
    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    failed_samples = []
    category_total_tested = 0

    for item in dataset:
        # 跳过还没有被大模型测试过的样本
        if "Is_Correct" not in item:
            continue
            
        item_cat = item.get("Category", "").strip()
        item_sub_cat = item.get("Sub_Category", "").strip()

        # 匹配对应大类
        if item_cat == TARGET_CATEGORY:
            # 🟢 新增逻辑：如果配置了小类，且当前样本的小类不匹配，则跳过
            if TARGET_SUB_CATEGORY and item_sub_cat != TARGET_SUB_CATEGORY:
                continue
                
            category_total_tested += 1
            
            # 提取判断为 False 的样本 (即答错的题目)
            if item.get("Is_Correct") is False:
                failed_samples.append(item)

    # 保存至新文件
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(failed_samples, f, ensure_ascii=False, indent=4)

    # 打印统计摘要
    target_display = f"[{TARGET_CATEGORY}]"
    if TARGET_SUB_CATEGORY:
        target_display += f" -> [{TARGET_SUB_CATEGORY}]"

    print("\n" + "="*40)
    print(f"🎯 类别锁定: {target_display}")
    print(f"📊 该范围总测试数: {category_total_tested}")
    print(f"❌ 失败样本总数: {len(failed_samples)}")
    
    if category_total_tested > 0:
        error_rate = (len(failed_samples) / category_total_tested) * 100
        print(f"📉 该范围错误率: {error_rate:.2f}%")
        
    print(f"✅ 失败样本已成功导出至: {OUTPUT_JSON}")
    print("="*40 + "\n")

if __name__ == "__main__":
    main()