import json
import random

class PromptManager:
    def __init__(self, template_file):
        """初始化提示管理器"""
        with open(template_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            self.templates = data['templates']
    
    def apply_template(self, text, template_index=0):
        """应用模板到文本"""
        template = self.templates[template_index]
        return template.replace("[TEXT]", text)
    
    def get_random_template(self):
        """随机获取一个模板（用于数据增强）"""
        return random.choice(self.templates)
    
    def apply_random_template(self, text):
        """应用随机模板"""
        template = self.get_random_template()
        return template.replace("[TEXT]", text)