# coding=utf-8
# knowledge_injector.py
import os
import json

class GeometryKnowledgeInjector:
    def __init__(self, kb_dir="/root/autodl-tmp/knowledge_bases", use_common_dimensions=False):
        """
        初始化知识注入器
        :param kb_dir: 知识库文件夹路径
        :param use_common_dimensions: 是否启用通用的实体尺寸先验字典
        """
        self.kb_dir = kb_dir
        self.use_common_dimensions = use_common_dimensions
        
        self.dimensions = {}
        if self.use_common_dimensions:
            dim_path = os.path.join(kb_dir, "common_object_dimensions.json")
            self.dimensions = self._load_json(dim_path)
            if self.dimensions:
                print("[Info] Common object dimensions KB loaded successfully.")
        
        self.category_rules_cache = {}

    def _load_json(self, path):
        if os.path.exists(path):
            try:
                with open(path, 'r', encoding='utf-8') as f:
                    # 过滤掉文件内部可能存在的 "_meta" 元数据信息 (如果你留着的话)
                    data = json.load(f)
                    if "_meta" in data:
                        del data["_meta"]
                    return data
            except Exception as e:
                print(f"[Error] Failed to parse JSON {path}: {str(e)}")
                return {}
        return {}

    def _get_category_rules(self, category_name):
        # 将带有空格或特殊字符的大类名转为合法的文件名
        safe_filename = category_name.replace(" ", "_").replace("&", "_").replace("__", "_") + ".json"
        
        if category_name not in self.category_rules_cache:
            file_path = os.path.join(self.kb_dir, safe_filename)
            self.category_rules_cache[category_name] = self._load_json(file_path)
            
        return self.category_rules_cache[category_name]

    def inject_knowledge(self, scene_graph, category, sub_category, question_text):
        """
        动态感知上下文的知识注入 (专为单层 List 结构优化)
        """
        injected_texts = []
        question_lower = question_text.lower() if question_text else ""

        # ================= 策略 1: 全局实体尺寸匹配 (受开关控制) =================
        if self.use_common_dimensions and scene_graph and self.dimensions:
            dim_knowledge = []
            for obj in scene_graph:
                obj_name = obj['name'].lower()
                if obj_name in question_lower or any(word in question_lower for word in obj_name.split()):
                    for kb_key, kb_data in self.dimensions.items():
                        if kb_key.lower() in obj_name:
                            dim_knowledge.append(f"[{obj_name} prior size]: {kb_data}")
                            break 
            
            if dim_knowledge:
                injected_texts.append("--- Object Size Priors ---")
                injected_texts.extend(dim_knowledge)

        # ================= 策略 2: 动态任务规则分发 (纯 List 解析) =================
        category_rules = self._get_category_rules(category)
        
        # 只要在字典里找到了这个 Sub_Category，并且它是个 List
        if sub_category in category_rules and isinstance(category_rules[sub_category], list):
            rules_list = category_rules[sub_category]
            
            # 确保 List 不是空的
            if len(rules_list) > 0:
                injected_texts.append(f"--- Domain Rules ({sub_category}) ---")
                
                # 直接遍历拼接
                for rule in rules_list:
                    injected_texts.append(f"- {rule}")

        # ================= 最终组装 =================
        # 如果注入了实际规则，返回合并后的文本；否则返回空字符串
        if len(injected_texts) > 1:
            return "\n".join(injected_texts)
        return ""