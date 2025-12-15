import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import json
import random
import numpy as np

from data.processor import SciTechDataset, EpisodeGenerator

def setup_seed(seed=42):
    """设置随机种子"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def train_lgpn():
    # 设置随机种子
    setup_seed(42)
    
    # 设备配置
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载配置
    with open('configs/label_words.json', 'r', encoding='utf-8') as f:
        label_words = json.load(f)
    
    # 初始化组件
    prompt_manager = PromptManager('configs/prompt_templates.json')
    
    # 这里应该是您的真实数据，这里用示例数据代替
    # texts = ["文本1", "文本2", ...]
    # labels = ["标签1", "标签2", ...]
    
    # 示例数据（实际使用时请替换）
    texts = [
        "新型太阳能电池效率突破25%",
        "深度学习算法在医疗影像诊断中的应用", 
        "高温超导材料研究取得新进展",
        "工业机器人自动化生产线",
        "区块链技术在金融领域的应用",
        "基因编辑技术的伦理问题研究"
    ]
    labels = [
        "新能源与动力",
        "数字智能技术", 
        "新材料科技",
        "智能制造与装备",
        "数字智能技术",
        "生命健康技术"
    ]
    
    # 创建数据集
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
    
    dataset = SciTechDataset(
        texts=texts,
        labels=labels,
        label_words=label_words,
        prompt_manager=prompt_manager,
        tokenizer=tokenizer
    )
    
    # 创建LGPN框架
    model = LGPNFramework('bert-base-chinese').to(device)
    
    # 损失函数
    lgpn_loss_fn = LGPNLoss(temperature=0.1)
    label_sep_loss_fn = LabelSeparationLoss(temperature=0.1)
    
    # 优化器
    optimizer = optim.AdamW(model.encoder.parameters(), lr=1e-5, weight_decay=0.01)
    
    # 训练循环
    model.encoder.train()
    num_episodes = 1000  # 训练任务数量
    
    for episode in range(num_episodes):
        # 生成少样本任务
        episode_generator = EpisodeGenerator(dataset, n_way=3, k_shot=1, n_query=2)
        support_set, query_set, selected_labels = episode_generator.generate_episode()
        
        if len(support_set) == 0:
            continue
        
        # 准备支持集数据
        support_input_ids = torch.stack([item['input_ids'] for item in support_set]).to(device)
        support_attention_mask = torch.stack([item['attention_mask'] for item in support_set]).to(device)
        support_labels = torch.stack([item['label_idx'] for item in support_set]).to(device)
        
        # 准备查询集数据
        query_input_ids = torch.stack([item['input_ids'] for item in query_set]).to(device)
        query_attention_mask = torch.stack([item['attention_mask'] for item in query_set]).to(device)
        query_true_labels = torch.stack([item['label_idx'] for item in query_set]).to(device)
        
        # 获取标签表示
        label_representations = dataset.get_label_representations(model.encoder)
        label_representations = label_representations.to(device)
        
        # 前向传播
        optimizer.zero_grad()
        
        # 获取支持集和查询集表示
        support_representations = model.encoder(support_input_ids, support_attention_mask)
        query_representations = model.encoder(query_input_ids, query_attention_mask)
        
        # 计算原型
        prototypes, prototype_labels = model.compute_prototypes(
            support_representations, support_labels, label_representations, use_lds=True
        )
        
        # 预测
        query_pred_labels, similarities = model.predict(query_representations, prototypes, prototype_labels)
        
        # 计算损失
        # 1. LGPN损失：让样本靠近正确标签
        lgpn_loss = lgpn_loss_fn(
            torch.cat([support_representations, query_representations]),
            label_representations,
            torch.cat([support_labels, query_true_labels])
        )
        
        # 2. 标签分离损失：让不同标签的表示相互远离
        label_sep_loss = label_sep_loss_fn(label_representations)
        
        # 总损失
        total_loss = lgpn_loss + 0.1 * label_sep_loss
        
        # 反向传播
        total_loss.backward()
        optimizer.step()
        
        if episode % 100 == 0:
            # 计算准确率
            accuracy = (query_pred_labels == query_true_labels).float().mean().item()
            print(f"Episode {episode}, Loss: {total_loss.item():.4f}, Accuracy: {accuracy:.4f}")
    
    # 保存模型
    torch.save({
        'model_state_dict': model.encoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, 'lgpn_model.pth')
    
    print("训练完成！")

if __name__ == "__main__":
    train_lgpn()