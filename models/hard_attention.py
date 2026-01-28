# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import os
from torch.amp import autocast

class Config(object):
    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'hard_attention'
        self.dataset = dataset
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt', 'r', encoding='utf-8').readlines()]
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'
        
        save_dir = os.path.dirname(self.save_path)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 1000
        self.num_classes = len(self.class_list)
        self.num_epochs = 10
        self.batch_size = 64
        self.pad_size = 64
        self.learning_rate = 2e-5
        self.bert_path = './pretrained/bert_pretrained'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.k_percent = 0.2  # 选取前20%的关键token (Hard Selection)
        self.dropout = 0.1

class Model(nn.Module):
    def __init__(self, config, static_emb_matrix=None):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        
        # 计算重要性分数的网络
        self.scorer = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.Tanh(),
            nn.Linear(config.hidden_size // 2, 1)
        )
        
        self.k_top = max(1, int(config.pad_size * config.k_percent))
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        
        # === 兼容性初始化 (适配 train_eval.py 的分析功能) ===
        self.attention_weights = None
        self.probabilities = None
        self.hidden_states = None

    def forward(self, x):
        context = x[0] # [batch, seq]
        mask = x[2]    # [batch, seq]
        
        with autocast('cuda'):
            bert_out = self.bert(context, attention_mask=mask)[0] # [batch, seq, 768]
            
            # 计算每个token的分数
            scores = self.scorer(bert_out).squeeze(-1) # [batch, seq]
            
            # Mask掉padding部分 (使用 -1e4 避免混合精度溢出)
            scores = scores.masked_fill(mask == 0, -1e4)
            
            # Hard Attention: 选取Top-K的索引
            # [batch, k]
            topk_scores, topk_indices = torch.topk(scores, self.k_top, dim=1)
            
            # 获取对应的向量
            # gather需要扩展索引维度: [batch, k, hidden]
            batch_size = bert_out.size(0)
            
            # 创建batch索引辅助
            batch_indices = torch.arange(batch_size, device=context.device).unsqueeze(1).expand(-1, self.k_top)
            
            # 提取Top-K特征
            # [batch, k, hidden]
            selected_features = bert_out[batch_indices, topk_indices, :]
            
            # 对选出的特征进行 Mean Pooling
            pooled_features = torch.mean(selected_features, dim=1) # [batch, hidden]
            
            out = self.dropout(pooled_features)
            logits = self.fc(out)
            
            # === 保存中间结果供 train_eval.py 分析 ===
            # 1. 保存概率分布
            self.probabilities = F.softmax(logits, dim=1).detach()
            
            # 2. 保存隐层状态 (池化后的特征)
            self.hidden_states = pooled_features.detach()
            
            # 3. 保存注意力权重
            # Hard Attention 本身是硬选择，但为了可视化分析，
            # 我们将 scorer 计算出的原始分数进行 softmax 归一化，
            # 这样可以在 CSV/可视化中看到模型认为哪些词最重要（即使它只选了前K个）
            self.attention_weights = F.softmax(scores, dim=1).detach()
            
        return logits