import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import os
from utils import FocalLoss
from torch.amp import autocast


class Config(object):
    """配置参数"""
    def __init__(self, dataset):
        self.dataset = dataset
        self.model_name = 'bert'
        self.train_path = dataset + '/data/train.txt'
        self.dev_path = dataset + '/data/dev.txt'
        self.test_path = dataset + '/data/test.txt'
        self.class_list = [x.strip() for x in open(dataset + '/data/class.txt', 'r', encoding='utf-8').readlines()]
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'

         # 自动创建目录
        save_dir = os.path.dirname(self.save_path)  # 获取目录路径：CSTA-Corpus/saved_dict
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)
            print(f"Created directory: {save_dir}")

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.require_improvement = 1200
        self.num_classes = len(self.class_list)
        # self.num_epochs = 8
        self.num_epochs = 10
        self.batch_size = 64
        self.pad_size = 64
        self.learning_rate = 3e-5
        # self.learning_rate = 1e-6 # 结果是69.77%
        # self.learning_rate = 2e-5
        # D:\CodingData\Github\MALTEC\pretrained\Erlangshen-DeBERTa_pretrained
        # self.bert_path = './pretrained/Erlangshen-DeBERTa_pretrained'
        # self.bert_path = r'D:\CodingData\Github\MALTEC\pretrained\bert_pretrained'
        self.bert_path = './pretrained/bert_pretrained'
        # self.bert_path = './pretrained/bert_pretrained_new'

        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

        # 为了形状匹配，需要将hidden_size改为768
        # self.hidden_size = 1024
        self.hidden_size = 768
        self.hidden_dropout_prob = 0.5
        self.layer_norm_eps = 1e-7
        self.weight_decay = 0.1
        self.max_grad_norm = 1.0
        self.early_stop_patience = 3
        self.visualize_num = 20  # 每类可视化样本数量


# 修改 Model 类中的注意力模块
class MultiHeadAttentionPooling(nn.Module):
    def __init__(self, hidden_size, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.attentions = nn.ModuleList([
            nn.Linear(hidden_size, 1) for _ in range(num_heads)
        ])
        
    def forward(self, hidden_states, mask):
        all_weights = []
        for attn in self.attentions:
            scores = attn(hidden_states).squeeze(-1)
            scores = scores.masked_fill(mask == 0, -1e4)
            weights = F.softmax(scores, dim=-1)
            all_weights.append(weights.unsqueeze(1))
        
        weights = torch.cat(all_weights, dim=1)  # [batch, heads, seq_len]
        pooled = torch.einsum('bhs,bsd->bhd', weights, hidden_states)

        return pooled.mean(dim=1), weights.mean(dim=1)  # 多头平均

class Model(nn.Module):
    def __init__(self, config, static_emb_matrix=None):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, output_hidden_states=True)
        for param in self.bert.parameters():
            param.requires_grad = True
        
        # 静态词向量Embedding层（兼容混合精度）
        self.static_embed_dim = 300
        self.static_embedding = nn.Embedding(
            num_embeddings=config.tokenizer.vocab_size,
            embedding_dim=self.static_embed_dim,
            padding_idx=config.tokenizer.pad_token_id,
            dtype=torch.float32  # 显式指定float32，避免混合精度冲突
        )
        if static_emb_matrix is not None:
            self.static_embedding.weight.data.copy_(static_emb_matrix)
            self.static_embedding.weight.requires_grad = False  # 固定静态词向量，降低训练成本
        
        # 可学习的融合权重（兼容混合精度）
        self.emb_fusion_weight = nn.Parameter(torch.ones(2, dtype=torch.float32))
        
        self.attention_pool = MultiHeadAttentionPooling(config.hidden_size + self.static_embed_dim)
        # 调整归一化和分类头维度（BERT hidden_size*5 + 静态词向量300）
        self.layer_norm = nn.LayerNorm(config.hidden_size * 5 + self.static_embed_dim, dtype=torch.float32)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 5 + self.static_embed_dim, config.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(config.hidden_size * 2, config.num_classes)
        )
        self.attention_weights = None
        self.hidden_states = None
        self.probabilities = None

    def forward(self, x):
        context = x[0]  # [batch, seq_len]
        mask = x[2]     # [batch, seq_len]
        
        with autocast('cuda'):  # 适配混合精度
            # 1. BERT Embedding（float16计算）
            bert_outputs = self.bert(context, attention_mask=mask)
            bert_emb = bert_outputs.last_hidden_state  # [batch, seq_len, 768]
            
            # 2. 静态词向量Embedding（float32 -> float16）
            static_emb = self.static_embedding(context).to(bert_emb.dtype)  # 类型对齐
            
            # 3. 加权融合（可学习权重）
            fusion_weight = F.softmax(self.emb_fusion_weight, dim=0).to(bert_emb.dtype)
            fused_emb = torch.cat([
                bert_emb * fusion_weight[0],
                static_emb * fusion_weight[1]
            ], dim=-1)  # [batch, seq_len, 768+300=1068]
        
        # 后续池化和分类（兼容混合精度）
        hidden_states = bert_outputs.hidden_states
        last4 = torch.cat(hidden_states[-4:], dim=-1)  # [batch, seq_len, 768*4]
        cls_token = last4[:, 0, :]  # [batch, 768*4]
        
        # 注意力池化基于融合后的Embedding
        attn_pooled, attn_weights = self.attention_pool(fused_emb, mask)  # [batch, 1068]
        combined = torch.cat([cls_token, attn_pooled], dim=-1)  # [batch, 768*4 + 1068 = 4140]
        
        self.hidden_states = fused_emb
        self.attention_weights = attn_weights
        
        with autocast('cuda'):
            combined = self.layer_norm(combined)
            combined = self.dropout(combined)
            logits = self.classifier(combined)
            self.probabilities = F.softmax(logits, dim=1)
        
        return logits