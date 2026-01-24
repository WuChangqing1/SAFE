import torch
import json
from models.lgpn import LGPNFramework
from data.processor import SciTechDataset
from utils.prompt_manager import PromptManager
from transformers import BertTokenizer

def inference():
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 加载配置
    with open('configs/label_words.json', 'r', encoding='utf-8') as f:
        label_words = json.load(f)
    
    # 初始化组件
    prompt_manager = PromptManager('configs/prompt_templates.json')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    # 加载模型
    model = LGPNFramework('bert-base-chinese').to(device)
    checkpoint = torch.load('lgpn_model.pth', map_location=device)
    model.encoder.load_state_dict(checkpoint['model_state_dict'])
    model.encoder.eval()
    
    # 示例推理
    test_texts = [
        "新型氢燃料电池效率提升技术",
        "基于人工智能的病理切片分析系统"
    ]
    
    # 假设的支持集（在实际应用中，这是您要分类的新类别的少量样本）
    support_texts = [
        "太阳能光伏发电新技术",
        "风能发电场优化设计", 
        "深度学习自然语言处理",
        "计算机视觉目标检测",
        "纳米材料制备工艺",
        "复合材料性能研究"
    ]
    support_labels = [
        "新能源与动力", "新能源与动力", 
        "数字智能技术", "数字智能技术",
        "新材料科技", "新材料科技"
    ]
    
    # 创建支持集数据集
    support_dataset = SciTechDataset(
        texts=support_texts,
        labels=support_labels,
        label_words=label_words,
        prompt_manager=prompt_manager,
        tokenizer=tokenizer
    )
    
    # 获取支持集表示
    support_input_ids = torch.stack([support_dataset[i]['input_ids'] for i in range(len(support_dataset))]).to(device)
    support_attention_mask = torch.stack([support_dataset[i]['attention_mask'] for i in range(len(support_dataset))]).to(device)
    support_label_indices = torch.stack([support_dataset[i]['label_idx'] for i in range(len(support_dataset))]).to(device)
    
    with torch.no_grad():
        support_representations = model.encoder(support_input_ids, support_attention_mask)
        label_representations = support_dataset.get_label_representations(model.encoder).to(device)
        
        # 计算原型
        prototypes, prototype_labels = model.compute_prototypes(
            support_representations, support_label_indices, label_representations, use_lds=True
        )
        
        # 对每个测试文本进行分类
        for text in test_texts:
            # 应用提示模板
            prompted_text = prompt_manager.apply_template(text)
            
            # Tokenization
            inputs = tokenizer(
                prompted_text,
                return_tensors='pt',
                truncation=True,
                padding='max_length',
                max_length=128
            ).to(device)
            
            # 获取查询样本表示
            query_representation = model.encoder(inputs['input_ids'], inputs['attention_mask'])
            
            # 预测
            predicted_label_idx, similarities = model.predict(query_representation, prototypes, prototype_labels)
            predicted_label = support_dataset.idx_to_label[predicted_label_idx.item()]
            
            print(f"文本: {text}")
            print(f"预测类别: {predicted_label}")
            print(f"相似度: {torch.max(similarities).item():.4f}")
            print("-" * 50)

if __name__ == "__main__":
    inference()