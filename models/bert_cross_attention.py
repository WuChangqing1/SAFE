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
        self.model_name = 'cross_attention_plus' # 改个名区分
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
        # self.num_epochs = 6 # 73.60%
        # self.num_epochs = 7 # 74.17%
        self.num_epochs = 6 # 
        self.batch_size = 64
        self.pad_size = 64
        self.learning_rate = 2e-5 
        self.bert_path = './pretrained/bert_pretrained'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.num_heads = 4
        self.dropout = 0.1 # 这里的dropout主要用于BERT输出层，后面还有Multi-Sample Dropout

class Model(nn.Module):
    def __init__(self, config, static_emb_matrix=None):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        
        # 可学习的 Query 向量
        self.query = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        
        # Cross Attention Layer
        self.multihead_attn = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=config.num_heads)
        
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        
        # --- 关键修改 1: 输入维度变为 hidden_size * 2 (因为拼接了 [CLS] 和 Attention 输出) ---
        self.fc = nn.Linear(config.hidden_size * 2, config.num_classes)
        
        # --- 关键修改 2: Multi-Sample Dropout ---
        # 定义多个 Dropout 层，虽然参数一样，但每次 forward 调用时随机掩码不同
        self.dropouts = nn.ModuleList([nn.Dropout(0.3) for _ in range(5)])
        
        # 初始化变量适配 train_eval.py
        self.attention_weights = None
        self.probabilities = None
        self.hidden_states = None

    def forward(self, x):
        context = x[0]
        mask = x[2]
        
        with autocast('cuda'):
            # BERT Output
            # outputs[0]: sequence_output [batch, seq, hidden]
            # outputs[1]: pooled_output [batch, hidden] -> 这是 [CLS] 经过线性层+Tanh后的向量
            outputs = self.bert(context, attention_mask=mask)
            sequence_output = outputs[0]
            pooled_output = outputs[1] 
            
            # --- Cross Attention 计算 ---
            # [seq, batch, hidden]
            key_value = sequence_output.permute(1, 0, 2) 
            # [1, batch, hidden]
            query = self.query.expand(-1, sequence_output.size(0), -1)
            
            key_padding_mask = (mask == 0)
            
            attn_output, attn_weights = self.multihead_attn(
                query, key_value, key_value, 
                key_padding_mask=key_padding_mask,
                need_weights=True
            )
            
            # 保存权重: [batch, seq_len]
            self.attention_weights = attn_weights.squeeze(1)
            
            # [batch, hidden]
            attn_output = attn_output.squeeze(0)
            attn_output = self.layer_norm(attn_output)
            
            # --- 关键修改 1: 特征融合 (Fusion) ---
            # 将 Attention 提取的特定特征 与 BERT 原生的大局特征 [CLS] 拼接
            # 形状变为 [batch, hidden * 2]
            combined_feature = torch.cat((pooled_output, attn_output), dim=1)
            
            # 保存 Hidden States (用于分析，保存融合后的向量)
            self.hidden_states = combined_feature

            # --- 关键修改 2: Multi-Sample Dropout ---
            # 通过 5 个不同的 Dropout 路径计算 logit，然后取平均
            # 这能显著提高模型的鲁棒性和准确率
            logits = None
            for i, dropout in enumerate(self.dropouts):
                if i == 0:
                    logits = self.fc(dropout(combined_feature))
                else:
                    logits += self.fc(dropout(combined_feature))
            
            logits = logits / len(self.dropouts)
            
            # 保存概率
            self.probabilities = F.softmax(logits, dim=1)
            
        return logits