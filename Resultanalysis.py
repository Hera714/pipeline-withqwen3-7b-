#coding=utf-8
import json
from collections import defaultdict

# ================= 配置区域 =================
# 填入你跑完测试后生成的 json 文件路径
RESULT_JSON_PATH = "1.json"
# 输出结果文件名
OUTPUT_FILE = "1.txt"
# ============================================

def analyze_results():
    # 用于同时输出到控制台和保存文件
    output_lines = []

    def print_and_save(text):
        print(text)
        output_lines.append(text)

    print_and_save(f"正在读取文件: {RESULT_JSON_PATH} ...\n")

    try:
        with open(RESULT_JSON_PATH, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print_and_save(f"找不到文件: {RESULT_JSON_PATH}，请检查路径是否正确。")
        return
    except json.JSONDecodeError:
        print_and_save("JSON 文件格式解析错误，可能文件不完整。")
        return

    if not data:
        print_and_save("数据为空！")
        return

    # 1. 动态识别文件中测试了哪些模型
    models = set()
    for item in data:
        for key in item.keys():
            if key.endswith("Is_Correct"):
                model_name = key.replace("Is_Correct", "")
                models.add(model_name)

    if not models:
        print_and_save("未在数据中发现任何模型的测试结果（没有找到以 'Is_Correct' 结尾的字段）。")
        return

    models = sorted(list(models))
    print_and_save(f"检测到以下模型的测试结果: {', '.join(models)}\n")

    # 2. 初始化统计字典
    overall_stats = {model: {'correct': 0, 'total': 0} for model in models}
    cat_stats = {model: defaultdict(lambda: {'correct': 0, 'total': 0}) for model in models}
    subcat_stats = {model: defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0})) for model in models}

    # 3. 遍历数据并统计
    for item in data:
        category = item.get("Category", "Unknown")
        sub_category = item.get("Sub_Category", "Unknown")

        for model in models:
            is_correct_key = f"{model}Is_Correct"

            if is_correct_key in item:
                is_correct = item[is_correct_key]

                # 统计 Overall (总体)
                overall_stats[model]['total'] += 1
                if is_correct:
                    overall_stats[model]['correct'] += 1

                # 统计大分类
                cat_stats[model][category]['total'] += 1
                if is_correct:
                    cat_stats[model][category]['correct'] += 1
                
                # 统计子分类
                subcat_stats[model][category][sub_category]['total'] += 1
                if is_correct:
                    subcat_stats[model][category][sub_category]['correct'] += 1

    # 4. 获取所有存在的 Category 和 Sub_Category 以便展示
    all_categories = set()
    for model in models:
        for cat in cat_stats[model].keys():
            all_categories.add(cat)
    all_categories = sorted(list(all_categories))

    # 构建大类到子类的映射
    cat_to_subcats = defaultdict(set)
    for model in models:
        for cat, subcats in subcat_stats[model].items():
            for subcat in subcats.keys():
                cat_to_subcats[cat].add(subcat)
                
    for cat in cat_to_subcats:
        cat_to_subcats[cat] = sorted(list(cat_to_subcats[cat]))

    # 设置统一的列宽（为了能放下类似 "100.00 (1000/1000)" 这样的字符串）
    col_width = 20

    # ================= 打印大类总对比表格 =================
    print_and_save("\n" + "=" * 80)
    print_and_save("                             大类 (Category) 横向对比表格 (%)")
    print_and_save("=" * 80)

    # 构建表头
    header = f"{'Model':<18} | "
    for cat in all_categories:
        header += f"{cat[:col_width]:<{col_width}} | "
    header += f"{'Overall':<{col_width}}"
    
    line_len = len(header)
    print_and_save(header)
    print_and_save("-" * line_len)

    # 构建大类每一行
    for model in models:
        row_str = f"{model[:18]:<18} | "
        for cat in all_categories:
            c_data = cat_stats[model][cat]
            if c_data['total'] > 0:
                acc = (c_data['correct'] / c_data['total']) * 100
                cell_str = f"{acc:.2f} ({c_data['correct']}/{c_data['total']})"
                row_str += f"{cell_str:<{col_width}} | "
            else:
                row_str += f"{'-':<{col_width}} | "

        o_data = overall_stats[model]
        if o_data['total'] > 0:
            overall_acc = (o_data['correct'] / o_data['total']) * 100
            cell_str = f"{overall_acc:.2f} ({o_data['correct']}/{o_data['total']})"
            row_str += f"{cell_str:<{col_width}}"
        else:
            row_str += f"{'-':<{col_width}}"

        print_and_save(row_str)
    print_and_save("=" * line_len)


    # ================= 打印每个大类下的子类对比表格 =================
    for cat in all_categories:
        subcats = cat_to_subcats[cat]
        if not subcats:
            continue
            
        print_and_save("\n\n")
        # 构建子类表头
        header = f"{'Model':<18} | "
        for subcat in subcats:
            header += f"{subcat[:col_width]:<{col_width}} | "
        header += f"{'Total (' + cat[:8] + '...)':<{col_width}}" 
        
        sub_line_len = len(header)
        
        print_and_save("=" * sub_line_len)
        title = f" [{cat}] 子类 (Sub-Category) 准确率及个数对比 "
        print_and_save(title.center(sub_line_len))
        print_and_save("=" * sub_line_len)
        
        print_and_save(header)
        print_and_save("-" * sub_line_len)

        # 构建子类每一行
        for model in models:
            row_str = f"{model[:18]:<18} | "
            for subcat in subcats:
                sc_data = subcat_stats[model][cat][subcat]
                if sc_data['total'] > 0:
                    acc = (sc_data['correct'] / sc_data['total']) * 100
                    cell_str = f"{acc:.2f} ({sc_data['correct']}/{sc_data['total']})"
                    row_str += f"{cell_str:<{col_width}} | "
                else:
                    row_str += f"{'-':<{col_width}} | "
            
            # 当前大类在该模型上的总分
            c_data = cat_stats[model][cat]
            if c_data['total'] > 0:
                c_acc = (c_data['correct'] / c_data['total']) * 100
                cell_str = f"{c_acc:.2f} ({c_data['correct']}/{c_data['total']})"
                row_str += f"{cell_str:<{col_width}}"
            else:
                row_str += f"{'-':<{col_width}}"
                
            print_and_save(row_str)
        print_and_save("=" * sub_line_len)

    # ================= 新增：保存结果到文件 =================
    try:
        with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        print_and_save(f"\n✅ 分析结果已保存到文件：{OUTPUT_FILE}")
    except Exception as e:
        print_and_save(f"\n❌ 保存文件失败：{str(e)}")


if __name__ == "__main__":
    analyze_results()