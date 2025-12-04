import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
import os
from utils import FocalLoss


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
        self.num_epochs = 5
        self.batch_size = 64
        self.pad_size = 64
        self.learning_rate = 3e-5
        # self.learning_rate = 1e-6 # 结果是69.77%
        # self.learning_rate = 2e-5
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
            scores = scores.masked_fill(mask == 0, -1e9)
            weights = F.softmax(scores, dim=-1)
            all_weights.append(weights.unsqueeze(1))
        
        weights = torch.cat(all_weights, dim=1)  # [batch, heads, seq_len]
        pooled = torch.einsum('bhs,bsd->bhd', weights, hidden_states)

        return pooled.mean(dim=1), weights.mean(dim=1)  # 多头平均

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path, output_hidden_states=True)
        for param in self.bert.parameters():
            param.requires_grad = True

        self.attention_pool = MultiHeadAttentionPooling(config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size * 5)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # 两层 MLP
        self.classifier = nn.Sequential(
            nn.Linear(config.hidden_size * 5, config.hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(0.8),
            nn.Linear(config.hidden_size * 2, config.num_classes)
        )

        # 中间缓存
        self.attention_weights = None
        self.hidden_states = None
        self.probabilities = None

        

    def forward(self, x):
        context = x[0]  # [batch, seq_len]
        mask = x[2]     # [batch, seq_len]

        outputs = self.bert(context, attention_mask=mask)
        hidden_states = outputs.hidden_states  # list: 13 x [batch, seq_len, hidden]

        # Last4Layer concat pooling
        last4 = torch.cat(hidden_states[-4:], dim=-1)  # [batch, seq_len, hidden*4]
        cls_token = last4[:, 0, :]  # [batch, hidden*4]

        # Attention pooling on last_hidden_state
        attn_pooled, attn_weights = self.attention_pool(outputs.last_hidden_state, mask)  # [batch, hidden]

        # 拼接
        combined = torch.cat([cls_token, attn_pooled], dim=-1)  # [batch, hidden*5]

        self.hidden_states = outputs.last_hidden_state
        self.attention_weights = attn_weights

        combined = self.layer_norm(combined)
        combined = self.dropout(combined)

        logits = self.classifier(combined)
        self.probabilities = F.softmax(logits, dim=1)
        return logits
