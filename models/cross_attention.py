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
        self.model_name = 'cross_attention'
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
        self.num_epochs = 10 # 74.70%
        # self.num_epochs = 8  # 74.59%
        # self.num_epochs = 11 # 74.02%
        self.batch_size = 64
        self.pad_size = 64
        self.learning_rate = 2e-5
        self.bert_path = './pretrained/bert_pretrained'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)
        self.hidden_size = 768
        self.num_heads = 4
        self.dropout = 0.1

class Model(nn.Module):
    # 修改这里：删除了 static_emb_matrix=None
    def __init__(self, config):
        super(Model, self).__init__()
        self.bert = BertModel.from_pretrained(config.bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        
        self.query = nn.Parameter(torch.randn(1, 1, config.hidden_size))
        self.multihead_attn = nn.MultiheadAttention(embed_dim=config.hidden_size, num_heads=config.num_heads)
        self.layer_norm = nn.LayerNorm(config.hidden_size)
        self.fc = nn.Linear(config.hidden_size, config.num_classes)
        self.dropout = nn.Dropout(config.dropout)
        
        self.attention_weights = None
        self.probabilities = None
        self.hidden_states = None

    # ... (forward 函数保持不变) ...
    def forward(self, x):
        context = x[0]
        mask = x[2]
        
        with autocast('cuda'):
            bert_out = self.bert(context, attention_mask=mask)[0]
            key_value = bert_out.permute(1, 0, 2)
            query = self.query.expand(-1, bert_out.size(0), -1)
            key_padding_mask = (mask == 0)
            
            attn_output, attn_weights = self.multihead_attn(
                query, key_value, key_value, 
                key_padding_mask=key_padding_mask,
                need_weights=True
            )
            
            self.attention_weights = attn_weights.squeeze(1)
            out = attn_output.squeeze(0)
            out = self.layer_norm(out)
            out = self.dropout(out)
            self.hidden_states = out
            
            logits = self.fc(out)
            self.probabilities = F.softmax(logits, dim=1)
            
        return logits