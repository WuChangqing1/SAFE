# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import os
from torch.amp import autocast

class Config(object):
    """配置参数"""
    def __init__(self, dataset):
        self.model_name = 'ernie'
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
        self.num_epochs = 17
        self.batch_size = 64
        self.pad_size = 64
        self.learning_rate = 2e-5
        
        # 确保该路径正确，或者改为 'nghuyong/ernie-1.0-base-zh' 等在线模型
        self.ernie_path = './pretrained/ernie' 
        self.tokenizer = AutoTokenizer.from_pretrained(self.ernie_path)
        self.hidden_size = 768
        self.dropout = 0.1

class Model(nn.Module):
    def __init__(self, config, static_emb_matrix=None):
        super(Model, self).__init__()
        # 加载预训练模型
        self.ernie = AutoModel.from_pretrained(config.ernie_path)
        self.dropout = nn.Dropout(config.dropout)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        
        # === 兼容性初始化 (适配 train_eval.py 的分析功能) ===
        self.attention_weights = None
        self.probabilities = None
        self.hidden_states = None

    def forward(self, x):
        context = x[0]  # [batch, seq_len]
        mask = x[2]     # [batch, seq_len]

        with autocast('cuda'):
            # 1. output_attentions=True: 让模型返回注意力权重
            # ERNIE/BERT 输出 tuple: (last_hidden_state, pooler_output, hidden_states, attentions)
            outputs = self.ernie(context, attention_mask=mask, output_attentions=True)
            
            # 获取池化后的输出 (通常用于分类)
            pooler_output = outputs.pooler_output
            
            # 兼容处理：某些精简版模型可能没有 pooler_output，使用 CLS token 替代
            if pooler_output is None:
                 pooler_output = outputs.last_hidden_state[:, 0, :]
            
            out = self.dropout(pooler_output)
            logits = self.fc(out)
            
            # === 保存中间结果供 train_eval.py 分析 ===
            
            # 1. 保存概率分布
            self.probabilities = F.softmax(logits, dim=1)
            
            # 2. 保存隐层状态 (使用 Pooler Output 作为句子表示)
            self.hidden_states = pooler_output
            
            # 3. 提取真实的注意力权重
            # outputs.attentions 是一个包含所有层attention的tuple
            # 形状: [batch, num_heads, seq_len, seq_len]
            # 我们取最后一层
            last_layer_attn = outputs.attentions[-1] 
            
            # 将多头注意力取平均: [batch, heads, seq, seq] -> [batch, seq, seq]
            avg_heads_attn = torch.mean(last_layer_attn, dim=1)
            
            # 为了适配可视化代码 (它需要 [batch, seq_len])
            # 我们取 [CLS] Token (索引0) 对其他所有 Token 的关注度
            # 这代表了模型在进行分类决策时"看"了句子的哪些部分
            self.attention_weights = avg_heads_attn[:, 0, :] 
            
        return logits